"""The forward test's 06:00 records, 25-29 Sep: every horse priced that the card listed as not running.

Read-only. horseracebase can keep a withdrawn horse on its card with the jockey blanked, and lists
Irish reserves as RESERVE until they get in; the 06:00 job priced both as runners until the fix of
29 Sep (daily_predictions.card_status). For each day's record (the predictions file and the card
it priced, as in S3), each such horse, its price and the share of its race's book it took.
"""

import os
import sys
from pathlib import Path

import boto3
import pandas as pd

sys.path.insert(0, os.getcwd())
from daily_predictions import card_status  # noqa: E402

bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
out = Path("out")
out.mkdir(exist_ok=True)
rows = []
for day in ["2026-09-25", "2026-09-26", "2026-09-27", "2026-09-28", "2026-09-29"]:
    pred_key = f"predictions/{day}.csv"
    pred_path = out / f"predictions_{day}.csv"
    s3.download_file(bucket, pred_key, str(pred_path))
    written = s3.head_object(Bucket=bucket, Key=pred_key)["LastModified"]
    pred = pd.read_csv(pred_path, dtype={"race_time": str}).rename(
        columns={"venue": "track", "runner_name": "horse_name", "predicted_win_prob": "share"})
    cards = sorted(o["Key"] for o in s3.list_objects_v2(Bucket=bucket, Prefix=f"racecards/{day}_").get("Contents", []))
    # the card the file was priced on: the last one fetched before the file was written
    card_key = cards[-1]
    for k in cards:
        if s3.head_object(Bucket=bucket, Key=k)["LastModified"] <= written:
            card_key = k
    card = pd.read_csv(s3.get_object(Bucket=bucket, Key=card_key)["Body"], dtype={"race_time": str})
    card["status"] = card["jockey_name"].map(card_status)
    bad = card[card["status"] != "runner"].merge(
        pred[["track", "race_time", "horse_name", "predicted_bfsp", "share"]],
        on=["track", "race_time", "horse_name"], how="left")
    print(f"{day}: record {pred_key} written {written:%H:%M} UTC, priced on {card_key}; "
          f"{len(card)} on the card, {int((card.status == 'withdrawn').sum())} withdrawn (no jockey), "
          f"{int((card.status == 'reserve').sum())} reserves")
    for r in bad.itertuples():
        print(f"    {r.race_time:>5} {r.track:16s} {r.horse_name:28s} {r.status:9s} "
              f"{r.predicted_bfsp:9.2f}  {r.share:6.1%}")
        rows.append({"day": day, "race_time": r.race_time, "track": r.track, "horse_name": r.horse_name,
                     "status": r.status, "predicted_bfsp": r.predicted_bfsp, "share_of_book": r.share})
    races = bad.groupby(["track", "race_time"])["share"].sum()
    if len(races):
        print(f"    {len(races)} races; their book held {races.mean():.1%} of such horses on average, "
              f"at most {races.max():.1%}")
pd.DataFrame(rows).to_csv(out / "records_not_running_0925_0929.csv", index=False)
