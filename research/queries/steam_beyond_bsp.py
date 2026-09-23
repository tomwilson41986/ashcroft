"""Does the move from the morning price to BSP predict the result beyond BSP itself?

The classic claim: the market under-reacts to late money, so horses that shorten
from the morning (steamers) win more often than even their BSP says, and
drifters less. If so, the move carries information beyond the closing price and
a model can use it at the off.

Development window only -- 2026-01-01 to 2026-03-31, the months of Betfair price
files that come before the locked holdout. The holdout is never read.

(1) Conditional logit on the winner, cross-fitted by calendar month (fit on two
    months, score the third): the market [ln pi, (ln pi)^2] against the market
    plus the morning->BSP move [m, m^2], plus the late move (pre-play WAP ->
    BSP), and plus both.
(2) A/E against BSP by quintile of the move.
(3) Rank-1 by the cross-fitted market+move probability, at BSP after 5%
    commission; and the BSP favourite split by whether it steamed or drifted.
"""
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import residual_screen as rs  # noqa: E402

DEV_FROM, LOCKBOX_FROM = "2026-01-01", "2026-04-01"
COMMISSION = 0.05

conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
q = f"""SELECT rr.race_date, rr.track, rr.race_time, rr.horse_name, rr.bfsp, rr.placing_numerical,
               bp.morningwap, bp.ppwap, bp.morning_vol, bp.pp_vol
        FROM race_results rr JOIN betfair_prices bp ON bp.race_results_id = rr.id
        WHERE LOWER(bp.market_type) = 'win' AND rr.race_date >= '{DEV_FROM}' AND rr.race_date < '{LOCKBOX_FROM}'"""
d = pd.read_sql_query(q, conn)
print(f"rows {len(d):,}, dates {d.race_date.min()} .. {d.race_date.max()} (holdout starts {LOCKBOX_FROM}, not read)")
assert d.race_date.max() < LOCKBOX_FROM
d["race"] = d.race_date + "|" + d.track + "|" + d.race_time.astype(str)
for c in ("bfsp", "morningwap", "ppwap", "morning_vol", "pp_vol"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
d["won"] = (pd.to_numeric(d.placing_numerical, errors="coerce") == 1).astype(float)
d = d.drop_duplicates(["race", "horse_name"])
g = d.groupby("race")
priced = (d.bfsp > 1) & (d.morningwap > 1) & (d.ppwap > 1)
ok = (g.won.transform("sum") == 1) & priced.groupby(d.race).transform("all") & (g.won.transform("size") >= 3)
book = (1 / d.bfsp).groupby(d.race).transform("sum")
d = d[ok & book.between(0.95, 1.15)].sort_values(["race_date", "race"]).reset_index(drop=True)
norm = lambda s: np.log(1 / s) - np.log((1 / s).groupby(d.race).transform("sum"))
d["lp"], d["lpm"], d["lpp"] = norm(d.bfsp), norm(d.morningwap), norm(d.ppwap)
d["move"] = d.lp - d.lpm            # > 0: shortened from the morning to the off
d["late"] = d.lp - d.lpp            # > 0: shortened in the final minutes, past the pre-play average
d["month"] = d.race_date.str[5:7].astype(int)
print(f"complete races {d.race.nunique():,}, runners {len(d):,}; move sd {d.move.std():.3f}, late sd {d.late.std():.3f}")

codes = pd.factorize(d.race)[0]
y = d.won.to_numpy()
lp = d.lp.to_numpy()
designs = {
    "market": np.column_stack([lp, lp ** 2]),
    "+move": np.column_stack([lp, lp ** 2, d.move, d.move ** 2]),
    "+late": np.column_stack([lp, lp ** 2, d.late, d.late ** 2]),
    "+both": np.column_stack([lp, lp ** 2, d.move, d.move ** 2, d.late, d.late ** 2]),
}
eta_oos = {k: np.full(len(d), np.nan) for k in designs}
ll = {k: [] for k in designs}
for m in sorted(d.month.unique()):
    te = (d.month == m).to_numpy(); tr = ~te
    s_tr, g_tr = rs.race_blocks(codes[tr]); s_te, g_te = rs.race_blocks(codes[te])
    for k, X in designs.items():
        b = rs.fit_clogit(X[tr], y[tr], s_tr, g_tr)
        eta_oos[k][te] = X[te] @ b
        ll[k].append(rs.race_loglik(X[te] @ b, y[te], s_te, g_te))
base = np.concatenate(ll["market"])
print("\n(1) Out of sample (each month scored by a fit on the other two), gain over the BSP market:")
for k in ("+move", "+late", "+both"):
    gg = np.concatenate(ll[k]) - base
    print(f"  {k:6s} {1000 * gg.mean():+6.2f} mnats/race  (t {gg.mean() / (gg.std(ddof=1) / np.sqrt(len(gg))):+.2f}, races {len(gg):,})")
starts, seg = rs.race_blocks(codes)
b_all = rs.fit_clogit(designs["+both"], y, starts, seg)
print("  full-sample coefficients [lp, lp^2, move, move^2, late, late^2]:", np.round(b_all, 3))

print("\n(2) Actual / BSP-expected wins by quintile of the morning->BSP move")
pi = np.exp(lp)
d["pi"] = pi
d["q"] = pd.qcut(d.move, 5, labels=["drift++", "drift", "flat", "steam", "steam++"])
for qn, x in d.groupby("q", observed=True):
    ae = x.won.sum() / x.pi.sum()
    se = np.sqrt(x.pi.sum()) / x.pi.sum()
    print(f"  {qn:8s} runners {len(x):6,}  move {x.move.mean():+.3f}  A/E {ae:.3f} (+-{1.64 * se:.3f})")

def roi(x, price):
    ret = np.where(x.won == 1, (x[price] - 1) * (1 - COMMISSION), -1.0)
    return ret.mean(), len(ret)

print("\n(3) Rank-1 at BSP after commission")
for k in ("market", "+move", "+both"):
    p = rs.softmax_blocks(eta_oos[k], starts, seg)
    d[f"p_{k}"] = p
    top = d.loc[d.groupby("race")[f"p_{k}"].idxmax()]
    r, n = roi(top, "bfsp")
    ev = top[f"p_{k}"] * (top.bfsp - 1) * (1 - COMMISSION) - (1 - top[f"p_{k}"])
    r2, n2 = roi(top[ev > 0], "bfsp") if (ev > 0).any() else (np.nan, 0)
    print(f"  {k:6s} rank-1 {r:+.2%} ({n:,} bets); EV>0 {r2:+.2%} ({n2:,} bets)")
fav = d.loc[d.groupby("race").lp.idxmax()]
for lab, x in (("steamed", fav[fav.move > 0]), ("drifted", fav[fav.move <= 0])):
    r, n = roi(x, "bfsp")
    print(f"  BSP favourite that {lab}: {r:+.2%} at BSP ({n:,})")
