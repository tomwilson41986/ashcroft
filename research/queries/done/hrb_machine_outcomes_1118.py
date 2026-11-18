"""Third run, on the most informative outcomes (the owner, 18 Nov 2026: "Hard rule to always use the most predictive
metric for comparison"): race-centred pounds beaten (model/form_windows.run_measures lbs_c, the primary) and the
normalised finishing position (nfp, race-centred), by least squares within the race, in place of the win alone. Second run: each component capped at its training 1st/99th percentiles (the first run's speed figures held
extreme values that swamped the fits). HorseRaceBase's Ratings Machine and System Builder components on our own results: how predictive each is, and the
weights each rating set should carry (the owner, 18 Nov 2026; ledger hrb-machine-1118, registered before the run;
read-only, nothing asked of HRB).

model/blocks/hrb_machine.py builds the components from race_results (2018 on, earlier days only). Then, on races with
a winner and every runner priced at the BSP, development window only (nothing from 1 Apr 2026, the locked holdout):
fit on 2021-2024, scored on 1 Jan 2025 - 31 Mar 2026.

1. Each component alone: a conditional logit (softmax within the race) on its z-score; the scored log-loss gain per
   race over an equal-chance field, and over the BSP when it is added to log p_BSP (the System Builder question:
   does it carry anything the market has not priced?), with race-clustered bootstrap intervals.
2. Each set (form, class, speed, connections, horse, weight, market) as one rating: the fitted weights (per standard
   deviation: the Ratings Machine's multipliers, put on one scale) and its gain alone and over the BSP.
3. All sets together over the BSP.
Missing values are read as the training mean (z = 0), as a points rating gives a missing component no score.
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.getcwd())
from model.blocks import hrb_machine as hm  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
TRAIN = ("2021-01-01", "2024-12-31")
TEST = ("2025-01-01", "2026-03-31")
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
have = {r[1] for r in conn.execute("PRAGMA table_info(race_results)")}
want = [c for c in hm.READS if c in have and c != "raceid"]
print("race_results columns read:", len(want), "; not in the table:", sorted(set(hm.READS) - have - {"raceid"}))
df = pd.read_sql_query(f"SELECT {', '.join(want)} FROM race_results WHERE race_date >= '2018-01-01' "
                       f"AND race_date <= '{TEST[1]}'", conn)
print(f"rows {len(df):,}, {df.race_date.min()} to {df.race_date.max()}")
df = hm.build(df)

d = df[(df.race_date >= TRAIN[0]) & (df.race_date <= TEST[1])].copy()
d["race"] = d.race_date.astype(str) + "|" + d.track.astype(str) + "|" + d.race_time.astype(str)
d["bsp"] = pd.to_numeric(d.bfsp, errors="coerce")
d["pos"] = pd.to_numeric(d.placing_numerical, errors="coerce")
g = d.groupby("race")
ok = (g.bsp.transform(lambda s: (s > 1).all())) & (g.pos.transform(lambda s: (s == 1).sum() == 1)) \
    & (g.race.transform("size") >= 3)
d = d[ok].copy()
d["won"] = (d.pos == 1).astype(float)
p = 1 / d.bsp
d["lp_bsp"] = np.log(p / p.groupby(d.race).transform("sum"))
d = d.sort_values(["race_date", "race"]).reset_index(drop=True)
tr = (d.race_date <= TRAIN[1]).to_numpy()
te = (d.race_date >= TEST[0]).to_numpy()
print(f"races: train {d.loc[tr, 'race'].nunique():,} ({tr.sum():,} runners), test {d.loc[te, 'race'].nunique():,} "
      f"({te.sum():,} runners)")


def z(cols):
    """z-scores on the training rows, each component first capped at its training 1st and 99th percentiles (the
    first run's speed figures held a few extreme values that swamped the fits), then clipped to +-5."""
    X = d[cols].astype(float).to_numpy()
    X = np.where(np.isfinite(X), X, np.nan)
    lo, hi = np.nanpercentile(X[tr], 1, axis=0), np.nanpercentile(X[tr], 99, axis=0)
    X = np.clip(X, lo, hi)
    mu = np.nanmean(X[tr], axis=0)
    sd = np.nanstd(X[tr], axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = np.clip((X - mu) / sd, -5, 5)
    return np.where(np.isfinite(Z), Z, 0.0)


def fit(X, idx, l2=1e-3):
    """Conditional logit (the softmax within each race) on rows idx; returns the weights."""
    Xs, y = X[idx], d.won.to_numpy()[idx]
    r = pd.factorize(d.race.to_numpy()[idx])[0]
    nr = r.max() + 1

    def f(w):
        s = Xs @ w
        m = np.zeros(nr)
        np.maximum.at(m, r, s)
        e = np.exp(s - m[r])
        den = np.bincount(r, e, nr)
        ll = (y * (s - m[r])).sum() - np.log(den).sum()
        pr = e / den[r]
        grad = Xs.T @ (y - pr)
        return -ll + l2 * w @ w, -grad + 2 * l2 * w

    w0 = np.zeros(X.shape[1])
    return minimize(f, w0, jac=True, method="L-BFGS-B").x


def race_ll(X, w, idx):
    """Each scored race's log-likelihood of its winner, and the race ids."""
    s = X[idx] @ w
    r = pd.factorize(d.race.to_numpy()[idx])[0]
    nr = r.max() + 1
    m = np.zeros(nr)
    np.maximum.at(m, r, s)
    e = np.exp(s - m[r])
    lp = s - m[r] - np.log(np.bincount(r, e, nr))[r]
    return np.bincount(r, lp * d.won.to_numpy()[idx], nr)


def gain(ll_with, ll_base, n_boot=1000, seed=0):
    diff = ll_with - ll_base
    rng = np.random.default_rng(seed)
    b = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
    return diff.mean(), np.quantile(b, 0.05), np.quantile(b, 0.95)



# ---- the outcomes (the hard rule): race-centred pounds beaten (primary; positive = beat the race's average) and the
# normalised finishing position (1 won, 0 last), race-centred; non-finishers have neither and are left out
from model.form_windows import run_measures  # noqa: E402
m = run_measures(d.assign(raceid=d.race))
d["y_lbs"] = m["lbsc"]
nfp = pd.Series(m["nfp"], index=d.index)
d["y_nfp"] = nfp - nfp.groupby(d.race).transform("mean")


def centred(v):
    v = pd.Series(np.asarray(v, dtype=float), index=d.index)
    return (v - v.groupby(d.race).transform("mean")).to_numpy()


BSPc = centred(d.lp_bsp)


def ls_gain(Xc, y, base_cols=None):
    """Least squares within the race (every column race-centred) fitted on the training rows; the scored mean squared
    error of the outcome on the test rows, and per test race the squared error, for the bootstrap."""
    ok = np.isfinite(y)
    A = Xc
    w, *_ = np.linalg.lstsq(A[tr & ok], y[tr & ok], rcond=None)
    e = (y - A @ w) ** 2
    t = te & ok
    per_race = pd.Series(e[t]).groupby(d.race.to_numpy()[t]).sum()
    return w, per_race


def gain_ci(base, wth, n_boot=1000, seed=0):
    b, w = base.align(wth, join="inner")
    diff = (b - w).to_numpy()
    rng = np.random.default_rng(seed)
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
    return diff.mean(), np.quantile(bs, 0.05), np.quantile(bs, 0.95)


ones = np.zeros((len(d), 0))
for yname in ("y_lbs", "y_nfp"):
    y = d[yname].to_numpy(dtype=float)
    print(f"\n==================== outcome {yname} (race-centred; {np.isfinite(y[te]).mean():.0%} of test runners have one)")
    zero = pd.Series(np.where(te & np.isfinite(y), y ** 2, 0.0)).groupby(d.race.to_numpy()).sum()
    zero = zero[zero.index.isin(d.race[te & np.isfinite(y)])]
    _, base = ls_gain(BSPc[:, None], y)
    print(f"the BSP explains {1 - base.sum() / zero.sum():.4f} of the outcome's within-race variance on the test races")
    rows = []
    for s, cols in hm.SETS.items():
        for c in cols:
            zc = centred(z([c])[:, 0])[:, None]
            _, alone = ls_gain(zc, y)
            w2, wth = ls_gain(np.column_stack([BSPc, zc]), y)
            g = gain_ci(base, wth)
            rows.append({"set": s, "feature": c, "r2_alone": 1 - alone.sum() / zero.sum(), "w_over_bsp": w2[1],
                         "gain_over_bsp": g[0], "lo90": g[1], "hi90": g[2]})
    R = pd.DataFrame(rows)
    print(R.round(5).to_string(index=False))
    print(f"   components whose gain over the BSP has its 90% interval above zero: {int((R.lo90 > 0).sum())} of {len(R)}"
          f" (by chance about {0.05 * len(R):.1f})")
    srows = []
    for s, cols in hm.SETS.items():
        Zs = np.column_stack([centred(col) for col in z(cols).T])
        w1, alone = ls_gain(Zs, y)
        w2, wth = ls_gain(np.column_stack([BSPc, Zs]), y)
        g = gain_ci(base, wth)
        srows.append({"set": s, "r2_alone": 1 - alone.sum() / zero.sum(), "gain_over_bsp": g[0], "lo90": g[1],
                      "hi90": g[2]})
        wt = pd.DataFrame({"feature": cols, "weight_alone": w1, "weight_over_bsp": w2[1:]})
        wt["share_of_rating"] = (wt.weight_alone.abs() / wt.weight_alone.abs().sum()).round(3)
        print(f"\n  {s} ({yname}): weights per sd")
        print(wt.round(4).to_string(index=False))
    print(f"\n  the sets ({yname}; gain = mean squared error a test race, BSP alone less BSP + the set):")
    print(pd.DataFrame(srows).round(5).to_string(index=False))
    allc = [c for v in hm.SETS.values() for c in v]
    Zall = np.column_stack([centred(col) for col in z(allc).T])
    _, alone = ls_gain(Zall, y)
    _, wth = ls_gain(np.column_stack([BSPc, Zall]), y)
    g = gain_ci(base, wth)
    print(f"  every component together: alone explains {1 - alone.sum() / zero.sum():.4f}; over the BSP {g[0]:.5f} "
          f"({g[1]:.5f} to {g[2]:.5f})")
