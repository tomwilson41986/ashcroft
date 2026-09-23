"""The early-price trade on the live 06:00 record: is the edge there at bet time?

The backtest forecasts are built from result rows, so they know race-day facts a
morning bettor does not (the field after late non-runners, the going on the day,
the jockey who rode). The 06:00 job's files (s3://<predictions bucket>/predictions/
<date>.csv) were written from the morning card, before Betfair's morning window
closed -- the only forecasts that were really available at bet time.

The rule is #73's, unchanged: back where ln(morning WAP / forecast) >= 0.2 with
>= GBP100 matched in the morning; take the morning price, close at BSP; commission
on positive CLV only. Window: 2026-04-01 .. 2026-09-06, the out-of-sample window.

Read the result with the handicap it carries: from April until the ingestion fix
the live model ran on history frozen at 2026-03-22 (form months out of date) and
with part of its card features missing (QA C1). A positive result means the edge
is there at bet time even so; a null is ambiguous.
"""
import os
import sqlite3
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from betfair_prices import normalise_horse, normalise_track, race_time_to_24h  # noqa: E402

FROM, TO = date(2026, 4, 1), date(2026, 9, 6)
COMMISSION = 0.05
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
os.makedirs("live", exist_ok=True)
try:
    import boto3
    import botocore
    s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
    got = miss = 0
    d = FROM
    while d <= TO:
        try:
            s3.download_file(bucket, f"predictions/{d.isoformat()}.csv", f"live/{d.isoformat()}.csv")
            got += 1
        except botocore.exceptions.ClientError:
            miss += 1
        d += timedelta(days=1)
    print(f"06:00 files: {got} downloaded, {miss} days without one (bucket {bucket})")
except Exception as exc:                                     # noqa: BLE001
    print(f"could not reach the predictions bucket: {type(exc).__name__}: {exc}")

text = {c: str for c in ("date", "race_date", "venue", "track", "race_time", "runner_name", "horse_name")}
frames = []
for f in sorted(os.listdir("live")):
    try:
        frames.append(pd.read_csv(os.path.join("live", f), dtype=text))
    except Exception as exc:                                 # noqa: BLE001
        print(f"skip {f}: {exc}")
if not frames:
    raise SystemExit("no 06:00 files to test")
pred = pd.concat(frames, ignore_index=True)
for a, b in (("race_date", "date"), ("track", "venue"), ("horse_name", "runner_name")):
    if a not in pred.columns and b in pred.columns:
        pred[a] = pred[b]
pred["predicted_bfsp"] = pd.to_numeric(pred["predicted_bfsp"], errors="coerce")
print(f"06:00 rows {len(pred):,}; columns: {', '.join(pred.columns[:14])}")

conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
bf = pd.read_sql_query(f"""SELECT r.race_date, r.track, r.race_time, r.horse_name, r.placing_numerical,
                                  b.morningwap, b.morning_vol, b.bsp
                           FROM betfair_prices b JOIN race_results r ON r.id = b.race_results_id
                           WHERE LOWER(b.market_type) = 'win' AND r.race_date >= '{FROM}' AND r.race_date <= '{TO}'""", conn)


def key(x):
    return (x["race_date"].astype(str).str.slice(0, 10) + "|" + x["track"].map(normalise_track) + "|"
            + x["race_time"].map(race_time_to_24h).astype(str) + "|" + x["horse_name"].map(normalise_horse))


pred["_k"], bf["_k"] = key(pred), key(bf)
d = pred.drop_duplicates("_k", keep="last")[["_k", "predicted_bfsp"]].merge(bf.drop_duplicates("_k"), on="_k")
d = d[(d.morningwap > 1) & (d.bsp > 1) & (d.predicted_bfsp > 1)].copy()
print(f"joined to Betfair prices and results: {len(d):,} runners, {d.race_date.min()} .. {d.race_date.max()}")
d["race"] = d.race_date + "|" + d.track + "|" + d.race_time.astype(str)
d["won"] = pd.to_numeric(d.placing_numerical, errors="coerce") == 1
d["pred_move"] = np.log(d.morningwap / d.predicted_bfsp)
clv = d.morningwap / d.bsp - 1
d["net"] = np.where(clv > 0, clv * (1 - COMMISSION), clv)
d["hold_bsp"] = np.where(d.won, (d.bsp - 1) * (1 - COMMISSION), -1.0)
d["t24"] = d.race_time.map(race_time_to_24h).astype(str)
d["hour"] = d.t24.str.slice(0, 2)
d["day_order"] = d.groupby("race_date").t24.rank(method="dense")          # by off time, not by name


def interval(x, col="net", n=2000, seed=0):
    if len(x) < 20:
        return f"n={len(x)} (too few)"
    races = x.groupby("race")[col].agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(races), (n, len(races)))
    b = races["sum"].to_numpy()[idx].sum(1) / races["size"].to_numpy()[idx].sum(1)
    return f"{100 * x[col].mean():+6.2f}% ({100 * np.percentile(b, 5):+.2f} to {100 * np.percentile(b, 95):+.2f}), n={len(x):,}"


base = d[d.morning_vol >= 100]
rule = base[base.pred_move >= 0.2]
print("\nNet CLV per unit, 06:00 forecasts (90% race-bootstrap interval)")
print(f"  every runner              {interval(base)}")
for t in (0.1, 0.2, 0.3):
    print(f"  forecast >= {100 * (np.exp(t) - 1):3.0f}% shorter   {interval(base[base.pred_move >= t])}")
print(f"  forecast >=  22% LONGER    {interval(base[base.pred_move <= -0.2])}")
print("\nThe rule by month")
for m, g in rule.groupby(rule.race_date.str.slice(0, 7)):
    print(f"  {m}  {interval(g)}")
print("\nThe rule by off hour")
for h, g in rule.groupby("hour"):
    print(f"  {h}:00  {interval(g)}")
print("\nThe rule by order in the day")
for lab, m in (("first three of the day", rule.day_order <= 3), ("race ten onwards", rule.day_order >= 10)):
    print(f"  {lab:24s} {interval(rule[m])}")
print(f"\nHeld to settlement at BSP: {interval(rule, 'hold_bsp')}   every runner {100 * base.hold_bsp.mean():+.2f}%")
