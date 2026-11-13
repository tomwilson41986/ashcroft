"""13 Nov 2026: the night-before card put in the morning's place (the owner, 13 Nov: "We need prices. We need to trade").

Both 04:08 UTC Daily Predictions runs died on a full runner disk and the fixed runs priced too slowly; by 07:30 UTC
there was no predictions/2026-11-13.csv and the trader (run 43371006084) was waiting for it. The night-before run
(predict-evening run 43338468655, 00:37 UTC, the same served model on the card as it stood then) wrote
predictions_evening/2026-11-13.csv. This copies that object to predictions/2026-11-13.csv, only if the morning's file
is not there, and prints what it copied. Not read-only: one S3 object written, once.
"""
import io
import os

import boto3
import pandas as pd

DAY = "2026-11-13"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
src, dst = f"predictions_evening/{DAY}.csv", f"predictions/{DAY}.csv"

try:
    head = s3.head_object(Bucket=bucket, Key=dst)
    print(f"{dst} is already there (written {head['LastModified']}, {head['ContentLength']:,} bytes): nothing copied")
except Exception as exc:
    if "404" not in str(exc) and "Not Found" not in str(exc):
        raise
    body = s3.get_object(Bucket=bucket, Key=src)["Body"].read()
    df = pd.read_csv(io.BytesIO(body), dtype={"race_time": str, "market_id": str})
    assert "predicted_bfsp" in df.columns and len(df) > 0, "the night-before card has no prices"
    assert (df["date"].astype(str) == DAY).all(), "the night-before card is not for today"
    print(f"{src}: {len(df):,} runners, {df['market_id'].nunique()} markets, {df['venue'].nunique()} venues; "
          f"predicted_bfsp nulls {int(df['predicted_bfsp'].isna().sum())}; market_id nulls {int(df['market_id'].isna().sum())}; "
          f"predicted_at {df['predicted_at'].min()} .. {df['predicted_at'].max()}")
    s3.put_object(Bucket=bucket, Key=dst, Body=body, ContentType="text/csv")
    head = s3.head_object(Bucket=bucket, Key=dst)
    print(f"copied to {dst}: {head['ContentLength']:,} bytes at {head['LastModified']}")
