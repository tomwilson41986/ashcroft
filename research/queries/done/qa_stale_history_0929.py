"""QA of 29 Sep (the owner: "we were getting the market wrong"): how current is the history the 06:00
features are built from, and does a missing recent run show in the model's disagreement with the market?

Read-only. horseracebase's results download has been paused, so the database stops short of the day
before. A horse whose last run is after the database's last day is priced on its earlier form: this
counts them on each forward-test day's 06:00 card (the card's own days-since-last-run against the
horse's last row in the database) and compares, for them and the rest, the model's gap to the card's
early price and how the card's price then moved.
"""

import os
import sqlite3
import sys
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

con = sqlite3.connect("horse_racing.db")
top = con.execute("SELECT MAX(race_date) FROM race_results").fetchone()[0]
print("race_results: last race_date", top)
recent = pd.read_sql("SELECT race_date, COUNT(*) n, COUNT(DISTINCT track || race_time) races, "
                     "SUM(CASE WHEN bfsp > 0 THEN 1 ELSE 0 END) with_bfsp FROM race_results "
                     "WHERE race_date >= '2026-09-10' GROUP BY race_date ORDER BY race_date", con)
print(recent.to_string(index=False))
try:
    bp = pd.read_sql("SELECT MAX(race_date) last_day, COUNT(*) n, SUM(CASE WHEN race_date >= '2026-09-01' THEN 1 ELSE 0 END) sep_rows FROM betfair_prices", con)
    print("betfair_prices:", bp.to_dict("records"))
except Exception as e:  # the table may be named differently or absent
    print("betfair_prices:", e)

last_run = pd.read_sql("SELECT horse_name, MAX(race_date) last_db FROM race_results GROUP BY horse_name", con)
last_run["last_db"] = pd.to_datetime(last_run["last_db"])
last = dict(zip(last_run["horse_name"].str.strip().str.casefold(), last_run["last_db"]))

bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
for day in ["2026-09-25", "2026-09-26", "2026-09-27", "2026-09-28", "2026-09-29"]:
    cards = sorted(o["Key"] for o in s3.list_objects_v2(Bucket=bucket, Prefix=f"racecards/{day}_").get("Contents", []))
    card = pd.read_csv(s3.get_object(Bucket=bucket, Key=cards[-1 if day != "2026-09-29" else 0])["Body"],
                       dtype={"race_time": str})
    d0 = pd.Timestamp(day)
    card["card_last"] = d0 - pd.to_timedelta(pd.to_numeric(card["days_since_lr"], errors="coerce"), unit="D")
    card["db_last"] = card["horse_name"].str.strip().str.casefold().map(last)
    card["missing"] = card["card_last"].notna() & (card["db_last"].isna() | (card["card_last"] > card["db_last"] + pd.Timedelta(days=1)))
    n_miss = int(card["missing"].sum())
    gap_days = (card.loc[card["missing"], "card_last"] - card.loc[card["missing"], "db_last"]).dt.days
    print(f"{day}: card {cards[-1 if day != '2026-09-29' else 0].split('/')[-1]}, {len(card)} runners; last run missing from the "
          f"database for {n_miss} ({n_miss / len(card):.1%}); of the runners that ran within 14 days, "
          f"{int(card.loc[pd.to_numeric(card['days_since_lr'], errors='coerce') <= 14, 'missing'].sum())} missing; "
          f"median days lost {gap_days.median() if n_miss else 0}")
    card[["race_time", "track", "horse_name", "days_since_lr", "card_last", "db_last", "missing", "odds"]].to_csv(
        f"out/stale_{day}.csv", index=False)
