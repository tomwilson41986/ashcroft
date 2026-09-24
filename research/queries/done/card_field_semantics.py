"""How the results table defines the fields the live card lacks (QA C1), so a
card-side fill reproduces them rather than guessing. Development window only.

  career_runs      this run counted or not: compare with the horse's prior rows
  max_or_in_race   max of the race's ORs? over which runners (zeros? blanks?)
  median_or        median of the race's ORs? same question
  jockeys_claim    format and range
  surface_type, track_direction   constant per track? per track and going class?
  horse_sex, stallion, dam_stallion   constant per horse?
  dist_furlongs vs race_distance text   the text's format
  race_type vs race_name   how often the name's key words decide the type
"""
import re
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
d = pd.read_sql_query("""SELECT race_date, race_time, track, race_name, race_distance, dist_furlongs, race_type, race_code,
                                surface_type, track_direction, going_description, horse_name, horse_sex, stallion,
                                dam_stallion, career_runs, jockey_name, jockeys_claim, official_rating, max_or_in_race,
                                median_or, number_of_runners
                         FROM race_results WHERE race_date >= '2024-01-01' AND race_date < '2026-04-01'""", conn)
d["race"] = d.race_date + "|" + d.track + "|" + d.race_time.astype(str)
print(f"rows {len(d):,}, races {d.race.nunique():,}")

# career_runs against the horse's earlier rows in the whole table
full = pd.read_sql_query("SELECT race_date, race_time, horse_name FROM race_results", conn)
full = full.sort_values(["horse_name", "race_date", "race_time"])
full["prior"] = full.groupby("horse_name").cumcount()
d = d.merge(full.drop_duplicates(["race_date", "race_time", "horse_name"]), on=["race_date", "race_time", "horse_name"], how="left")
cr = pd.to_numeric(d.career_runs, errors="coerce")
diff = (cr - d.prior)
print("\ncareer_runs - (earlier rows of the horse in the table):", diff.value_counts().head(8).to_dict())
print("  (the table starts 2020; horses with runs before that show a positive offset)")
print("  career_runs on rows with no earlier row:", cr[d.prior == 0].value_counts().head(6).to_dict())

orr = pd.to_numeric(d.official_rating, errors="coerce")
for name, f in (("max", "max"), ("median", "median")):
    col = "max_or_in_race" if name == "max" else "median_or"
    v = pd.to_numeric(d[col], errors="coerce")
    all_or = orr.groupby(d.race).transform(f)
    pos_or = orr.where(orr > 0).groupby(d.race).transform(f)
    print(f"\n{col}: non-null {v.notna().mean():.3f}; equals {name} of all ORs {np.isclose(v, all_or).mean():.3f}, "
          f"of ORs > 0 {np.isclose(v, pos_or).mean():.3f}; sample {v.dropna().head(5).tolist()}")

print("\njockeys_claim values:", d.jockeys_claim.astype(str).value_counts().head(10).to_dict())
print("jockey names containing digits:", d.jockey_name.astype(str).str.contains(r"\d").mean().round(4))

for col in ("surface_type", "track_direction"):
    per = d.groupby("track")[col].nunique()
    print(f"\n{col}: tracks {len(per)}, with more than one value {int((per > 1).sum())}: "
          f"{per[per > 1].head(8).to_dict()}")
aw = d.going_description.astype(str).str.lower().str.contains("standard|slow")
print("surface_type by going class, multi-surface tracks:")
for t in d.groupby("track").surface_type.nunique().loc[lambda s: s > 1].index[:6]:
    x = d[d.track == t]
    print(f"  {t}: {x.groupby(aw.loc[x.index].map({True: 'AW going', False: 'turf going'})).surface_type.agg(lambda s: s.value_counts().head(2).to_dict()).to_dict()}")

for col in ("horse_sex", "stallion", "dam_stallion"):
    per = d.groupby("horse_name")[col].nunique()
    print(f"\n{col}: horses with more than one value {(per > 1).mean():.4f}; values {d[col].astype(str).value_counts().head(6).to_dict()}")

print("\nrace_distance text samples:", d.drop_duplicates("race").race_distance.astype(str).sample(12, random_state=0).tolist())
print("dist_furlongs samples:", d.drop_duplicates("race").dist_furlongs.sample(12, random_state=0).tolist())

print("\nrace_type values:", d.drop_duplicates("race").race_type.astype(str).value_counts().head(25).to_dict())
print("race_code values:", d.drop_duplicates("race").race_code.astype(str).value_counts().to_dict())
r = d.drop_duplicates("race")
print("race_name samples by type:")
for t, g in r.groupby("race_type"):
    if len(g) > 300:
        print(f"  {t}: {g.race_name.astype(str).sample(3, random_state=1).tolist()}")
