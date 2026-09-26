"""Copy today's 06:00 record out of S3 into the artifact: the predictions file and the card it priced.

Read-only. The files are the forward test's record (the 615 model on 26 Sep,
written by Daily Predictions run 197); this copies them so the day's workbook can
be built from exactly what the 06:00 job wrote.
"""

import os
from pathlib import Path

import boto3
import pandas as pd

DAY = "2026-09-26"
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
out = Path("out")
out.mkdir(exist_ok=True)

keys = [f"predictions/{DAY}.csv"]
cards = s3.list_objects_v2(Bucket=bucket, Prefix=f"racecards/{DAY}_").get("Contents", [])
keys += sorted(o["Key"] for o in cards)
for k in keys:
    dest = out / k.replace("/", "_")
    s3.download_file(bucket, k, str(dest))
    df = pd.read_csv(dest)
    print(f"{k}: {len(df)} rows, {df.shape[1]} columns")
    print("  columns:", ", ".join(df.columns))
    print(df.head(3).to_string(max_cols=20, max_colwidth=24))
    print()
