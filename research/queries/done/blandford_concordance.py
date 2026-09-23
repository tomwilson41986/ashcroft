"""Can the feed's "pre-race" rating rank the winner better than the market?

A genuine public pre-race rating cannot out-rank the closing market; a rating
that has seen the race can. For each quarter: the within-race concordance of the
winner against every loser (share of winner-loser pairs the score orders
correctly, ties half) for (a) pre_race_master_rating, (b) 1/BSP, and (c) the
rating's own information beyond the price -- a conditional logit of the winner
on ln pi_bsp and the within-race z-score of the rating.

Also: does the rating's presence depend on the result? has_m for winners vs
losers inside races where some runners lack it."""
import sqlite3

import numpy as np
import pandas as pd

conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
q = """SELECT rr.race_date, rr.track, rr.race_time, rr.horse_name, rr.bfsp, rr.placing_numerical, rr.race_code,
              b.pre_race_master_rating AS m
       FROM race_results rr LEFT JOIN blandford_results b ON b.race_results_id = rr.id
       WHERE rr.race_date >= '2021-01-01'"""
d = pd.read_sql_query(q, conn)
d["race"] = d.race_date + "|" + d.track + "|" + d.race_time.astype(str)
d["bfsp"] = pd.to_numeric(d.bfsp, errors="coerce")
d["won"] = (pd.to_numeric(d.placing_numerical, errors="coerce") == 1).astype(int)
d["m"] = pd.to_numeric(d.m, errors="coerce")
d.loc[(d.m <= 0) | (d.m >= 999), "m"] = np.nan
g = d.groupby("race")
d = d[(g.won.transform("sum") == 1) & (g.bfsp.transform(lambda s: (s > 1).all()))]
d["quarter"] = pd.to_datetime(d.race_date).dt.to_period("Q").astype(str)


def concordance(frame, col):
    """Share of winner-vs-loser pairs in which the winner scores higher (ties 1/2)."""
    num = den = 0.0
    for _, r in frame.groupby("race"):
        s = r[col].to_numpy(float); w = r.won.to_numpy() == 1
        if w.sum() != 1 or np.isnan(s[w][0]):
            continue
        lo = s[~w]; lo = lo[~np.isnan(lo)]
        if len(lo) == 0:
            continue
        num += np.sum(s[w][0] > lo) + 0.5 * np.sum(s[w][0] == lo); den += len(lo)
    return num / den if den else np.nan


d["inv_bsp"] = 1.0 / d.bfsp
full = d.groupby("race").m.transform(lambda s: s.notna().all())
print("Per quarter, races where every runner has a rating (so both scores rank the same set):")
print(f"{'quarter':8s} {'races':>6s} {'conc_m':>7s} {'conc_bsp':>8s}  {'m beats bsp?':>12s}")
for qtr, x in d[full].groupby("quarter"):
    if x.race.nunique() < 300:
        continue
    sub = x[x.race.isin(x.race.drop_duplicates().sample(min(1500, x.race.nunique()), random_state=0))]
    cm, cb = concordance(sub, "m"), concordance(sub, "inv_bsp")
    print(f"{qtr:8s} {sub.race.nunique():6d} {cm:7.4f} {cb:8.4f}  {'YES' if cm > cb else 'no':>12s}")

part = d.groupby("race").m.transform(lambda s: s.notna().any() & s.isna().any())
x = d[part]
print(f"\nRaces where only some runners have a rating: {x.race.nunique():,}")
print(f"  has rating: winners {x[x.won == 1].m.notna().mean():.1%}   losers {x[x.won == 0].m.notna().mean():.1%}")
for yr, xx in x.groupby(x.race_date.str[:4]):
    print(f"    {yr}: winners {xx[xx.won == 1].m.notna().mean():.1%}  losers {xx[xx.won == 0].m.notna().mean():.1%}  (races {xx.race.nunique():,})")
