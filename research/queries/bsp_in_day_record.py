"""Where 6 Oct's BSPs are in the day record (read-only): the source of each book row in s3 betfair_live/2026-10-06/,
how many carry sp_actual, and a sample of the final pass's rows (the Tote comparison found none to join)."""
import gzip
import io
import os

import boto3
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
for name in ("books.csv.gz", "books_trader.csv.gz", "books_other_place.csv.gz"):
    try:
        obj = s3.get_object(Bucket=bucket, Key=f"betfair_live/2026-10-06/{name}")
    except Exception as exc:
        print(name, "not read", type(exc).__name__)
        continue
    b = pd.read_csv(io.BytesIO(gzip.decompress(obj["Body"].read())), dtype={"market_id": str}, on_bad_lines="skip")
    sp = pd.to_numeric(b.get("sp_actual"), errors="coerce")
    print(f"\n== {name}: {len(b):,} rows, last modified {obj['LastModified']:%H:%M} UTC; columns {list(b.columns)[-8:]}")
    print(b.assign(has_sp=sp.notna()).groupby("source").agg(rows=("market_id", "size"), with_sp=("has_sp", "sum"),
                                                           first=("polled_utc", "min"), last=("polled_utc", "max"))
          .to_string())
    fin = b[b["source"].astype(str).eq("final")]
    print(fin[["polled_utc", "market_id", "status", "selection_id", "runner_status", "sp_actual", "sp_near",
               "bsp_reconciled"]].head(8).to_string(index=False))
