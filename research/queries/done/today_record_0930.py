"""Copy today's 06:00 record out of S3 into the artifact, and check it against the card it priced.

Read-only. 30 Sep is the first 06:00 run on merge 56b1178 (PR #78): the card leaves out the horses
it lists that are not running (withdrawn, the jockey blanked; reserves not yet in;
daily_predictions.declared_runners). Daily Predictions run 202 logged in first time, scraped 367
horses in 38 races, left out 6 (1 withdrawn, 5 reserves) and wrote 361 prices at 06:42 UTC. The card
S3 keeps is the card after the fix. This copies the predictions file and every card fetched today,
with the time S3 stored each (the pre-registration counts a record written before 09:00 UK), and
checks the record against the card it was priced on:

- no horse on that card, or priced, that the card marks as withdrawn or a reserve;
- every horse priced is on the card, and every runner on it is priced;
- each race a book of 1, one rank 1 per race.
"""

import os
import sys
from pathlib import Path

import boto3
import pandas as pd

sys.path.insert(0, os.getcwd())
from daily_predictions import card_status  # noqa: E402

DAY = "2026-09-30"
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
out = Path("out")
out.mkdir(exist_ok=True)

pred_key = f"predictions/{DAY}.csv"
pred_path = out / f"predictions_{DAY}.csv"
s3.download_file(bucket, pred_key, str(pred_path))
written = s3.head_object(Bucket=bucket, Key=pred_key)["LastModified"]
pred = pd.read_csv(pred_path, dtype={"race_time": str})
print(f"{pred_key}: {len(pred)} rows, stored {written.isoformat()}")
print("  columns:", ", ".join(pred.columns))

# the card the file was priced on: the last one stored before the file was written
cards = sorted(o["Key"] for o in s3.list_objects_v2(Bucket=bucket, Prefix=f"racecards/{DAY}_").get("Contents", []))
card_key = None
for k in cards:
    stored = s3.head_object(Bucket=bucket, Key=k)["LastModified"]
    s3.download_file(bucket, k, str(out / k.replace("/", "_")))
    print(f"{k}: stored {stored.isoformat()}")
    if stored <= written:
        card_key = k
card_key = card_key or cards[-1]
card = pd.read_csv(out / card_key.replace("/", "_"), dtype={"race_time": str})
card["status"] = card["jockey_name"].map(card_status)
print(f"priced on {card_key}: {len(card)} on the card, {int((card.status == 'runner').sum())} runners, "
      f"{int((card.status == 'withdrawn').sum())} withdrawn (no jockey), "
      f"{int((card.status == 'reserve').sum())} reserves")

p = pred.rename(columns={"venue": "track", "runner_name": "horse_name"})
key = ["track", "race_time", "horse_name"]
m = p.merge(card[key + ["status"]], on=key, how="left", indicator=True)
priced_not_running = m[m["status"].isin(["withdrawn", "reserve"])]
priced_off_card = m[m["_merge"] == "left_only"]
runners = card[card["status"] == "runner"].merge(p[key], on=key, how="left", indicator=True)
unpriced = runners[runners["_merge"] == "left_only"]
race = p.groupby(["track", "race_time"])
books = race["predicted_win_prob"].sum()
rank1 = int((race["predicted_bfsp"].rank(method="first") == 1).sum())

print()
print(f"CHECK priced but not running on the card: {len(priced_not_running)} (must be 0)")
for r in priced_not_running.itertuples():
    print(f"    {r.race_time:>5} {r.track:16s} {r.horse_name:28s} {r.status}")
print(f"CHECK priced but not on the card: {len(priced_off_card)} (must be 0)")
for r in priced_off_card.itertuples():
    print(f"    {r.race_time:>5} {r.track:16s} {r.horse_name}")
print(f"CHECK runners on the card not priced: {len(unpriced)} (must be 0)")
for r in unpriced.itertuples():
    print(f"    {r.race_time:>5} {r.track:16s} {r.horse_name}")
print(f"CHECK races {len(books)}, books {books.min():.6f} to {books.max():.6f}, "
      f"unpriced {int(p['predicted_bfsp'].isna().sum())}, rank 1 per race {rank1}")
print(f"CHECK written {written:%H:%M} UTC, before 08:00 UTC (09:00 UK): {written.hour < 8}")

left = card[card["status"] != "runner"]
print(f"CHECK horses on the stored card marked withdrawn or reserve: {len(left)} (must be 0)")
for r in left.itertuples():
    print(f"    {r.race_time:>5} {r.track:16s} {r.horse_name:28s} {r.status}")
