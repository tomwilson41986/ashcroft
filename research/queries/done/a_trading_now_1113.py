"""13 Nov, read-only: is the trader betting since the prices landed at 07:48 UTC? Today's order ledger and the
predictions file as they stand in S3."""
import io
import os

import boto3
import pandas as pd

DAY = "2026-11-13"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
pd.set_option("display.width", 220)

for key in (f"predictions/{DAY}.csv",):
    h = s3.head_object(Bucket=bucket, Key=key)
    print(f"{key}: {h['ContentLength']:,} bytes, written {h['LastModified']}")

objs = s3.list_objects_v2(Bucket=bucket, Prefix=f"trading/live/{DAY}/").get("Contents", [])
if not objs:
    print(f"no data found: nothing under trading/live/{DAY}/ yet")
for o in objs:
    print(f"{o['Key']}: {o['Size']:,} bytes, written {o['LastModified']}")
    if o["Key"].endswith("ledger.csv"):
        df = pd.read_csv(io.BytesIO(s3.get_object(Bucket=bucket, Key=o["Key"])["Body"].read()), low_memory=False)
        print(f"   rows {len(df):,}; first {df['ts'].min()}, last {df['ts'].max()}")
        print(df.groupby("event").agg(rows=("ts", "size"), matched=("matched", "sum"), first=("ts", "min"), last=("ts", "max")).to_string())
        b = df[df.event == "back"]
        if len(b):
            print(f"   backs: {len(b)} fills on {b['selection_id'].nunique()} horses in {b['market_id'].nunique()} markets, "
                  f"GBP{b['matched'].sum():,.2f} matched")
