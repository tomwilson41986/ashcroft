"""Two gaps between the live card and the results table that the table alone cannot show (QA C1).

1. median_or is missing on about 18% of result rows, and not where half the field is
   unrated (the table keeps 0 there). What decides it? If it is a property of the race,
   the fill should reproduce it so the model sees what it was trained on. The column is
   an INTEGER and a median over an even field can end in .5, so that is tested first.
2. The model's form features find a runner's history by the horse's name, and its course
   features by the track's. card_enrich_check.py tested the fill on the table's own
   spellings. The 06:00 files keep the card's: are they the table's? (Names only; no
   outcome is read.)
"""
import os
import sqlite3
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from betfair_prices import normalise_horse, normalise_track, race_time_to_24h  # noqa: E402

conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)

# --- 1. what decides a missing median_or ----------------------------------------------
d = pd.read_sql_query("""SELECT race_date, race_time, track, race_type, race_class, official_rating, median_or,
                                max_or_in_race, created_at
                         FROM race_results WHERE race_date >= '2023-01-01' AND race_date < '2026-04-01'""", conn)
d["race"] = d.race_date + "|" + d.track + "|" + d.race_time.astype(str)
orr = pd.to_numeric(d.official_rating, errors="coerce").fillna(0)
med = orr.groupby(d.race).transform("median")
d["null"] = d.median_or.isna()
d["half"] = (med % 1) != 0
d["field"] = d.groupby("race")["race"].transform("size")
d["even"] = d.field % 2 == 0
print(f"median_or missing on {d.null.mean():.3f} of {len(d):,} rows")
print(f"\nmissing, by whether the median of every runner's OR (unrated 0) ends in .5:")
print(pd.crosstab(d.half, d.null, normalize="index").round(4).to_string())
print(pd.crosstab(d.half, d.null).to_string())
ok = ~d.null
print(f"\nwhere present: equals the median {np.isclose(pd.to_numeric(d.median_or[ok]), med[ok]).mean():.4f}; "
      f"equals it rounded down {np.isclose(pd.to_numeric(d.median_or[ok]), np.floor(med[ok])).mean():.4f}")
print(f"odd fields missing {d.null[~d.even].mean():.4f}; even fields missing {d.null[d.even].mean():.4f}")
per_race = d.groupby("race")["null"].mean()
print(f"races missing it on some runners but not all: {((per_race > 0) & (per_race < 1)).mean():.4f}")
left = d[d.null & ~d.half]
print(f"\nmissing although the median is whole: {len(left):,} rows")
if len(left):
    for by in ("race_type", "track"):
        print(f"  by {by}: " + "; ".join(f"{k} {v}" for k, v in left[by].value_counts().head(6).items()))
    print(f"  by load date: " + "; ".join(f"{k} {v}" for k, v in
                                          left.created_at.astype(str).str[:10].value_counts().head(6).items()))

# --- 2. the card's spellings against the table's ---------------------------------------
FROM, TO = date(2026, 7, 14), date(2026, 9, 22)
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
os.makedirs("live", exist_ok=True)
try:
    import boto3
    import botocore
    s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
    got = 0
    day = FROM
    while day <= TO:
        try:
            s3.download_file(bucket, f"predictions/{day.isoformat()}.csv", f"live/{day.isoformat()}.csv")
            got += 1
        except botocore.exceptions.ClientError:
            pass
        day += timedelta(days=1)
    print(f"\n06:00 files: {got} downloaded (bucket {bucket})")
except Exception as exc:                                     # noqa: BLE001
    print(f"\ncould not reach the predictions bucket: {type(exc).__name__}: {exc}")

frames = [pd.read_csv(os.path.join("live", f), dtype=str) for f in sorted(os.listdir("live")) if f.endswith(".csv")]
if not frames:
    raise SystemExit("no 06:00 files")
card = pd.concat(frames, ignore_index=True).rename(columns={"date": "race_date", "venue": "track",
                                                          "runner_name": "horse_name"})
card["race_date"] = card.race_date.str.slice(0, 10)
tab = pd.read_sql_query(f"""SELECT race_date, track, race_time, horse_name FROM race_results
                            WHERE race_date >= '{FROM}' AND race_date <= '{TO}'""", conn)
print(f"card rows {len(card):,} on {card.race_date.nunique()} days; table rows {len(tab):,}")

dates = set(tab.race_date)
card = card[card.race_date.isin(dates)]
tracks = tab.groupby("race_date")["track"].agg(set)
names = tab.groupby("race_date")["horse_name"].agg(set)
exact_track = np.array([t in tracks[d_] for d_, t in zip(card.race_date, card.track)])
exact_name = np.array([h in names[d_] for d_, h in zip(card.race_date, card.horse_name)])
tab["_h"] = tab.horse_name.map(normalise_horse)
loose = tab.groupby("race_date")["_h"].agg(set)
loose_name = np.array([normalise_horse(h) in loose[d_] for d_, h in zip(card.race_date, card.horse_name)])
print(f"\ncard track spelled exactly as the table's that day: {exact_track.mean():.4f}")
bad = card.loc[~exact_track, "track"].value_counts().head(10)
tt = tab.drop_duplicates("track").assign(_t=lambda x: x.track.map(normalise_track)).groupby("_t")["track"].first()
for t, n in bad.items():
    print(f"  card '{t}' ({n} rows) -> table '{tt.get(normalise_track(t), '?')}'")
print(f"card horse spelled exactly as in the table that day: {exact_name.mean():.4f}; "
      f"after normalising: {loose_name.mean():.4f} (the rest are non-runners)")
miss = card.loc[~exact_name & loose_name, "horse_name"].head(12).tolist()
print(f"  exact misses that normalise to a runner: {miss}")
suffix = card.horse_name.str.contains(r"\([A-Z]{2,3}\)\s*$", regex=True, na=False).mean()
suffix_tab = tab.horse_name.str.contains(r"\([A-Z]{2,3}\)\s*$", regex=True, na=False).mean()
print(f"names with a country suffix: card {suffix:.3f}, table {suffix_tab:.3f}")
print(f"race_time forms: card {card.race_time.head(3).tolist()}, table {tab.race_time.head(3).tolist()}")
k_card = card.race_date + "|" + card.track.map(normalise_track) + "|" + card.race_time.map(race_time_to_24h).astype(str)
k_tab = tab.race_date + "|" + tab.track.map(normalise_track) + "|" + tab.race_time.map(race_time_to_24h).astype(str)
print(f"card races found in the table by normalised key: {k_card.drop_duplicates().isin(set(k_tab)).mean():.4f}")
