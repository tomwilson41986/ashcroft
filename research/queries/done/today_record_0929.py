"""Copy today's 06:00 record out of S3 into the artifact: the predictions file and the card it priced.

Read-only. The files are the forward test's record for 29 Sep, the first day on the
gated three (runs 54 + 55 averaged, run 56 gated to their leaders). The scheduled
06:00 run could not log in to horseracebase and wrote nothing; Daily Predictions
run 201, dispatched at 06:42 UTC, wrote them (the card at 06:43, the predictions at
07:03 UTC). This copies them, with the time S3
stored each one (the pre-registration counts a record written before 09:00 UK),
so the day's workbook is built from exactly what the job wrote.
"""

import os
from pathlib import Path

import boto3
import pandas as pd

DAY = "2026-09-29"
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
out = Path("out")
out.mkdir(exist_ok=True)

keys = [f"predictions/{DAY}.csv"]
cards = s3.list_objects_v2(Bucket=bucket, Prefix=f"racecards/{DAY}_").get("Contents", [])
keys += sorted(o["Key"] for o in cards)
for k in keys:
    head = s3.head_object(Bucket=bucket, Key=k)
    dest = out / k.replace("/", "_")
    s3.download_file(bucket, k, str(dest))
    df = pd.read_csv(dest)
    print(f"{k}: {len(df)} rows, {df.shape[1]} columns, stored {head['LastModified'].isoformat()}")
    print("  columns:", ", ".join(df.columns))
    if "predicted_win_prob_norm" in df.columns:
        race = df.groupby(["track", "race_time"])
        books = race["predicted_win_prob_norm"].sum()
        print(f"  races {len(books)}, books {books.min():.6f} to {books.max():.6f}, "
              f"unpriced {int(df['predicted_bfsp'].isna().sum())}, "
              f"rank 1 per race {int((df['model_rank'] == 1).sum())}")
    print(df.head(3).to_string(max_cols=20, max_colwidth=24))
    print()
