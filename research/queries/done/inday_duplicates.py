"""Can the in-day block see a race's own result through a duplicated race?

A horse cannot run twice on one day in Britain or Ireland, so a horse on two
rows of the same date is a data error -- and for the in-day features a harmful
one: if the copies carry different off times, the later copy's jockey and
trainer "earlier today" figures include the earlier copy, which is the same race
with the same result. Likewise a race recorded on two dates within three days
would feed the 72-hour figures.

(1) Same horse, same date, two or more rows: how many, by year, and how far
    apart their off times are (the in-day block only admits copies 10+ or 30+
    minutes apart).
(2) Same horse on two dates within 3 days with the same finishing position,
    field size and BSP (a race recorded twice under different dates).
(3) Whole races duplicated: same date and track, same set of horses, different
    off time.
"""
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from model.lagsafe import race_minutes  # noqa: E402

conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
d = pd.read_sql_query("""SELECT id, race_date, race_time, track, horse_name, jockey_name, trainer, bfsp,
                                placing_numerical, number_of_runners FROM race_results
                         WHERE race_date >= '2021-01-01' AND race_date < '2026-04-01'""", conn)
print(f"rows {len(d):,}  {d.race_date.min()} .. {d.race_date.max()} (holdout not read)")
d["t"] = race_minutes(d.race_time)
d["race"] = d.race_date + "|" + d.track + "|" + d.race_time.astype(str)
d["year"] = d.race_date.str[:4]

# (1)
g = d.groupby(["race_date", "horse_name"])
multi = d[g.id.transform("size") > 1].copy()
print(f"\n(1) horse on 2+ rows of one date: {multi.groupby(['race_date', 'horse_name']).ngroups:,} horse-days, {len(multi):,} rows")
if len(multi):
    span = multi.groupby(["race_date", "horse_name"]).agg(n=("id", "size"), races=("race", "nunique"),
                                                          tracks=("track", "nunique"), dt=("t", lambda s: s.max() - s.min()),
                                                          same_pos=("placing_numerical", "nunique"),
                                                          same_bsp=("bfsp", "nunique"))
    print("   by year:", multi.groupby("year").apply(lambda x: x.groupby(["race_date", "horse_name"]).ngroups).to_dict())
    print("   distinct races per horse-day:", span.races.value_counts().to_dict())
    print("   off-time gap between copies (minutes):",
          span.dt.describe(percentiles=[.1, .5, .9]).round(1).to_dict())
    print(f"   copies >= 10 min apart: {(span.dt >= 10).sum():,};  >= 30 min apart: {(span.dt >= 30).sum():,}")
    print(f"   copies with the SAME result (one placing, one BSP): {((span.same_pos == 1) & (span.same_bsp == 1)).sum():,}")
    print(span.sort_values("dt", ascending=False).head(12).to_string())

# (2)
d["date"] = pd.to_datetime(d.race_date)
s = d.sort_values(["horse_name", "date"]).copy()
prev = s.groupby("horse_name").shift(1)
lag = (s.date - prev.date).dt.days
same = (lag.between(1, 3)) & (s.placing_numerical == prev.placing_numerical) & (s.bfsp == prev.bfsp) & \
       (s.number_of_runners == prev.number_of_runners) & s.bfsp.notna()
print(f"\n(2) same horse, 1-3 days apart, identical placing, field size and BSP: {int(same.sum()):,} rows")
if same.any():
    print(s[same][["race_date", "track", "race_time", "horse_name", "placing_numerical", "bfsp"]].head(10).to_string())

# (3)
sig = d.groupby("race").agg(date=("race_date", "first"), track=("track", "first"), t=("t", "first"),
                            horses=("horse_name", lambda x: "|".join(sorted(x.astype(str)))))
dup = sig[sig.duplicated(["date", "track", "horses"], keep=False)]
print(f"\n(3) whole races duplicated under another off time: {dup.groupby(['date', 'track', 'horses']).ngroups:,} "
      f"({len(dup):,} race keys)")
if len(dup):
    print(dup.reset_index().sort_values(["date", "track", "t"]).head(12)[["race", "t"]].to_string())
