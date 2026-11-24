"""M6 (ledger long-end-gate-stn-1121): the long-end correction of price-bias-long-end-1113 (candidate D) on gate_stn.

A power on the within-race log deviation of the horses longer than their race's mean log price (one for Flat/AW, one
for NH), the book renormalised; fitted on earlier folds by mean absolute log error, applied to the next. Writes the
corrected file beside the input for scripts/compare_oos_runs.py. Run locally:
    python research/queries/done/long_end_gate_stn_1121.py <oos_gate_stn.csv> <out.csv>
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, ".")
from model.race_shape import race_code  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
d = pd.read_csv(src, low_memory=False, dtype={"race_time": str})
keep = (d.bfsp > 1) & (d.predicted_bfsp > 1)
d["race"] = d.race_date.astype(str) + "|" + d.track.astype(str) + "|" + d.race_time.astype(str)
nh = race_code(d).isin(["hurdle", "chase", "nhflat"]).to_numpy().astype(int)
lp = np.log(d.predicted_bfsp.where(d.predicted_bfsp > 1)).to_numpy()
lb = np.log(d.bfsp.where(d.bfsp > 1)).to_numpy()
dev = lp - pd.Series(lp).groupby(d.race.to_numpy()).transform("mean").to_numpy()
rid = pd.factorize(d.race)[0]
fold = d.fold_idx.to_numpy()


def prices(idx, p):
    g = np.where(dev[idx] > 0, p[nh[idx]], 1.0)
    s = -(g * dev[idx])
    r = rid[idx]
    m = np.full(r.max() + 1, -np.inf)
    np.maximum.at(m, r, np.nan_to_num(s, nan=-np.inf))
    ex = np.exp(s - m[r])
    tot = np.zeros(r.max() + 1)
    np.add.at(tot, r, np.nan_to_num(ex))
    return -np.log(ex / tot[r])


out = lp.copy()
for fo in range(1, int(fold.max()) + 1):
    tr = np.where((fold < fo) & keep.to_numpy())[0]
    te = np.where(fold == fo)[0]
    p = minimize(lambda p: np.nanmean(np.abs(prices(tr, p) - lb[tr])), [1.0, 1.0], method="Nelder-Mead",
                 options={"xatol": 1e-4, "fatol": 1e-7}).x
    print(f"fold {fo}: long-end power Flat/AW {p[0]:.4f}, NH {p[1]:.4f}")
    out[te] = prices(te, p)
d["predicted_bfsp"] = np.exp(out)
d.drop(columns=["race"]).to_csv(dst, index=False)
print(f"wrote {dst}: {len(d):,} rows; fold 0 unchanged (no earlier fold)")
