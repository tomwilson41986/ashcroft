"""Is the feed's pre-race rating informative through its VALUE or through its ABSENCE?

(a) Races where every runner is rated -- absence cannot matter there. By year,
    the conditional-logit gain of adding the rating (z and z^2 within race) to
    the market's ln pi and (ln pi)^2, in millinats per race, fitted and scored
    on alternate months (so the gain is out of sample).
(b) Races where only some runners are rated: actual wins against BSP-expected
    wins (A/E) for rated and for unrated runners, by year. If the unrated lose
    far more often than their price says, the absence carries information the
    market did not have before the off -- the signature of ratings filled in
    after the race.
"""
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import residual_screen as rs  # noqa: E402

conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
q = """SELECT rr.race_date, rr.track, rr.race_time, rr.horse_name, rr.bfsp, rr.placing_numerical,
              b.pre_race_master_rating AS m
       FROM race_results rr LEFT JOIN blandford_results b ON b.race_results_id = rr.id
       WHERE rr.race_date >= '2021-01-01'"""
d = pd.read_sql_query(q, conn)
d["race"] = d.race_date + "|" + d.track + "|" + d.race_time.astype(str)
d["bfsp"] = pd.to_numeric(d.bfsp, errors="coerce")
d["won"] = (pd.to_numeric(d.placing_numerical, errors="coerce") == 1).astype(float)
d["m"] = pd.to_numeric(d.m, errors="coerce")
d.loc[(d.m <= 0) | (d.m >= 999), "m"] = np.nan
g = d.groupby("race")
ok = (g.won.transform("sum") == 1) & (d.bfsp > 1).groupby(d.race).transform("all") & (g.won.transform("size") >= 3)
d = d[ok].sort_values(["race_date", "race"]).reset_index(drop=True)
inv = 1 / d.bfsp
d["pi"] = inv / inv.groupby(d.race).transform("sum")
d["year"] = d.race_date.str[:4]
d["month"] = d.race_date.str[5:7].astype(int)

full = d.m.notna().groupby(d.race).transform("all")
print("(a) Every runner rated: value of the rating beyond the market, out of sample (alternate months)")
for yr, x in d[full].groupby("year"):
    x = x.reset_index(drop=True)
    z = (x.m - x.groupby("race").m.transform("mean")) / x.m.std()
    lp = np.log(x.pi)
    M = np.column_stack([lp, lp ** 2]); F = np.column_stack([M, z, z ** 2])
    codes = pd.factorize(x.race)[0]; y = x.won.to_numpy()
    gains = []
    for te_odd in (0, 1):
        te = (x.month % 2 == te_odd).to_numpy(); tr = ~te
        s_tr, g_tr = rs.race_blocks(codes[tr]); s_te, g_te = rs.race_blocks(codes[te])
        bm = rs.fit_clogit(M[tr], y[tr], s_tr, g_tr); bf = rs.fit_clogit(F[tr], y[tr], s_tr, g_tr)
        gains.append(rs.race_loglik(F[te] @ bf, y[te], s_te, g_te) - rs.race_loglik(M[te] @ bm, y[te], s_te, g_te))
    gg = np.concatenate(gains)
    print(f"  {yr}: races {len(gg):6,}  gain {1000 * gg.mean():+6.2f} mnats/race  (t {gg.mean() / (gg.std(ddof=1) / np.sqrt(len(gg))):+.2f})")

part = d.m.notna().groupby(d.race).transform("any") & d.m.isna().groupby(d.race).transform("any")
x = d[part]
print("\n(b) Partly rated races: actual / BSP-expected wins")
for yr, xx in x.groupby("year"):
    r, u = xx[xx.m.notna()], xx[xx.m.isna()]
    print(f"  {yr}: rated A/E {r.won.sum() / r.pi.sum():.3f} ({len(r):,})   unrated A/E {u.won.sum() / u.pi.sum():.3f} ({len(u):,})")
