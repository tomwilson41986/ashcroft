"""Task 233: correct the closing model's expected CLV by time to the off and price, walk-forward by day on the live
fills (first back per horse). bias = log(BSP / expected BSP); corrected expected BSP = expected * exp(predicted bias);
corrected expected CLV = best back / corrected expected BSP - 1.

The horses (one row each, 30 Sep - 8 Oct, from research/queries/done/live_review_horses_1008.py; the Horses sheet of
reports/live_review_2026-10-08.xlsx) are in horses_0930_1008.csv.gz beside this file:
    python research/live/closing_correction/fit.py research/live/closing_correction/horses_0930_1008.csv.gz out.csv
Only horses the live rule backed are here, so the corrected rule can only drop horses, never add one it rejected."""
import sys
import numpy as np
import pandas as pd

h = pd.read_csv(sys.argv[1])
h = h[(h.bsp > 1) & h.exp_bsp.notna() & (h.first_price > 1) & (h.best_back > 1)].copy()
h["bias"] = np.log(h.bsp / h.exp_bsp)
h["live"] = h.model_feed.str.startswith("volume").astype(float)
h["lm"] = np.log(h.first_mins.clip(1, 600))
h["lp"] = np.log(h.best_back)
h["le"] = np.log1p(h.edge)
h["first_locked"] = h.first_stake * h.first_clv
days = sorted(h.day.unique())


def X(d, cols):
    return np.column_stack([np.ones(len(d))] + [d[c].values for c in cols])


def fit(tr, cols, lam=1.0):
    A = X(tr, cols); y = tr.bias.values
    reg = lam * np.eye(A.shape[1]); reg[0, 0] = 0
    return np.linalg.solve(A.T @ A + reg, A.T @ y)


SPECS = {"level": [], "time+price": ["lm", "lp"], "time+price+edge": ["lm", "lp", "le"]}
rows = []
preds = {k: pd.Series(np.nan, index=h.index) for k in SPECS}
for i, d in enumerate(days[1:], 1):
    te = h[h.day == d]
    feed = te.live.iloc[0] if te.live.nunique() == 1 else None
    prior = h[h.day < d]
    for k, cols in SPECS.items():
        for f in (0.0, 1.0):
            t = te[te.live == f]
            if t.empty:
                continue
            tr = prior[prior.live == f]
            if len(tr) < 60:      # too few fills of this feed: all prior fills, the feed as a shift
                b = fit(prior, cols + ["live"])
                preds[k].loc[t.index] = X(t, cols + ["live"]) @ b
            else:
                b = fit(tr, cols)
                preds[k].loc[t.index] = X(t, cols) @ b
            if k == "time+price+edge":
                rows.append((d, f, len(tr), np.round(b, 3).tolist()))
print("coefficients (time+price+edge; intercept, log mins, log price, log(1+edge)[, live]):")
for r in rows:
    print("  ", r)

out = []
for k in SPECS:
    p = preds[k]
    ok = p.notna()
    err = (h.bias - p)[ok]
    out.append((k, ok.sum(), round(float((h.bias[ok] ** 2).mean()), 4), round(float((err ** 2).mean()), 4),
                round(float(err.mean()), 4)))
print("\nsquared error of log(BSP/expected) walk-forward, 1 Oct - 8 Oct (uncorrected vs corrected):")
print(pd.DataFrame(out, columns=["spec", "n", "mse_uncorrected", "mse_corrected", "mean_error"]).to_string(index=False))

BAR = 0.03
K = "time+price"
for K in SPECS:
    h["cedge"] = h.best_back / (h.exp_bsp * np.exp(preds[K])) - 1
    t = h[h.cedge.notna()]
    keep = t[t.cedge >= BAR]
    g = t.groupby("day").agg(horses=("edge", "size"), staked=("staked", "sum"), locked=("locked", "sum"),
                             first_stake=("first_stake", "sum"), first_locked=("first_locked", "sum"))
    k2 = keep.groupby("day").agg(c_horses=("edge", "size"), c_staked=("staked", "sum"), c_locked=("locked", "sum"),
                                 c_first_stake=("first_stake", "sum"), c_first_locked=("first_locked", "sum"))
    g = g.join(k2).fillna(0)
    g.loc["all"] = g.sum()
    g["clv"] = g.locked / g.staked
    g["c_clv"] = g.c_locked / g.c_staked
    g["first_clv"] = g.first_locked / g.first_stake
    g["c_first_clv"] = g.c_first_locked / g.c_first_stake
    pd.set_option("display.width", 250)
    print(f"\n== corrected rule ({K}): back only where the corrected expected CLV >= 3%")
    print(g.round(3).to_string())
    h[f"cedge_{K}"] = h.cedge
h.to_csv(sys.argv[2], index=False)
