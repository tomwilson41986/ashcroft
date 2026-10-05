"""Are tonight's evening record and 4 Oct's price files in S3? (read-only)

The trader's job recorded 6 Oct's GB/IE win markets 17:00-21:30 UK for the first time on 5 Oct, and the late look
after the last race added to the same day file (betfair_live/2026-10-06/books_evening.csv.gz). Betfair's price
files for 4 Oct's racing are named by the next day (dwbfprices*05102026.csv, betfair_prices_raw/).
"""

from __future__ import annotations

import gzip
import io
import os

import boto3
import pandas as pd

bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")

for prefix in ("betfair_live/2026-10-06/", "betfair_live/2026-10-05/"):
    objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])
    print(prefix, [(o["Key"].split("/")[-1], o["Size"], o["LastModified"].strftime("%H:%M")) for o in objs])
try:
    raw = s3.get_object(Bucket=bucket, Key="betfair_live/2026-10-06/books_evening.csv.gz")["Body"].read()
    ev = pd.read_csv(io.BytesIO(gzip.decompress(raw)), dtype={"market_id": str}, low_memory=False)
    t = pd.to_datetime(ev[[c for c in ev.columns if c in ("polled_utc", "ts", "t", "time")][0]], utc=True)
    print(f"books_evening 6 Oct: {len(ev):,} rows, {ev.market_id.nunique()} markets, "
          f"{t.dt.tz_convert('Europe/London').min():%H:%M} to {t.dt.tz_convert('Europe/London').max():%H:%M} UK, "
          f"{t.dt.floor('min').nunique()} poll minutes")
except Exception as e:                       # a missing file is the answer, not a failure
    print("books_evening 6 Oct:", type(e).__name__, e)
for name in ("dwbfpricesukwin05102026.csv", "dwbfpricesirewin05102026.csv", "dwbfpricesukplace05102026.csv",
             "dwbfpricesireplace05102026.csv"):
    try:
        h = s3.head_object(Bucket=bucket, Key=f"betfair_prices_raw/{name}")
        print(name, h["ContentLength"], h["LastModified"].strftime("%Y-%m-%d %H:%M"))
    except Exception as e:
        print(name, "not archived:", type(e).__name__)
