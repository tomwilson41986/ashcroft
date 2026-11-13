"""The account's day against the horse trader's own result, 12 Nov (read-only).

The settle of 12 Nov (run 43320446205) read Betfair's funds at GBP4,959.75; 11 Nov closed at GBP4,970.83. That is
-GBP11.08 on the day, while the horse trader's settled result is +GBP25.31 (19 of its 20 markets settled): GBP36.39
is not in the horse trader's settled figure. Every trader that can place a real order uses the same account; on
11 Nov the greyhound early lay ran with real orders (paper-models.yml run 43031425801). Whether it ran live on
12 Nov is read here from its own files, not assumed.

Here each trader's own record of the day is read from S3 and printed: what files it left, its orders and what it says
it made. Nothing is computed that the files do not carry; a file that is not there is said to be not there.
"""
import io
import json
import os

import boto3
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)
DAY = os.environ.get("REPORT_DAY") or "2026-11-12"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
MONEY = ("pnl", "locked_pnl", "profit", "matched", "liability", "lay_matched", "size", "asked", "commission", "stake",
         "staked", "pnl_back", "pnl_lay")


def keys(prefix):
    out, token = [], None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        out += r.get("Contents", [])
        token = r.get("NextContinuationToken")
        if not token:
            return out


def show(name, prefix, by):
    print(f"\n== {name}: s3://{bucket}/{prefix}")
    ks = keys(prefix)
    if not ks:
        print("   no data found: nothing under this prefix")
        return
    for o in ks:
        print(f"   {o['Key'][len(prefix):]:40s} {o['Size']:>9,} bytes  written {o['LastModified']:%H:%M:%S} UTC")
    for o in ks:
        k = o["Key"]
        if o["Size"] > 5_000_000:
            continue
        raw = s3.get_object(Bucket=bucket, Key=k)["Body"].read()
        short = k[len(prefix):]
        if k.endswith(".json"):
            try:
                j = json.loads(raw)
            except Exception as e:  # noqa: BLE001
                print(f"   {short}: not read ({type(e).__name__})")
                continue
            if isinstance(j, dict):
                j = {a: (b if not isinstance(b, (list, dict)) or len(json.dumps(b)) < 400 else f"<{type(b).__name__} of {len(b)}>")
                     for a, b in j.items()}
            print(f"   {short}: {json.dumps(j)[:2500]}")
        elif k.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str}, on_bad_lines="skip")
            except Exception as e:  # noqa: BLE001
                print(f"   {short}: not read ({type(e).__name__})")
                continue
            print(f"   {short}: {len(df):,} rows; columns {list(df.columns)[:30]}")
            col = next((c for c in by if c in df.columns), None)
            num = [c for c in MONEY if c in df.columns]
            for c in num:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            if "mode" in df.columns:
                print("     mode:", df["mode"].astype(str).value_counts().to_dict())
            if "status" in df.columns:
                print("     status:", df["status"].astype(str).value_counts().head(8).to_dict())
            if col and num:
                print(f"     sums by {col}:")
                print(df.groupby(col)[num].sum(min_count=1).round(2).assign(rows=df.groupby(col).size()).to_string())
            if "market_id" in df.columns:
                print(f"     markets: {df['market_id'].nunique()}")
            tcol = next((c for c in ("time_utc", "ts") if c in df.columns), None)
            if tcol:
                print(f"     first row {df[tcol].astype(str).min()[:19]}, last {df[tcol].astype(str).max()[:19]}")


show("the horse trader", f"trading/live/{DAY}/", ("event", "action"))
show("the greyhound early lay, live", f"sources/greyhound/early_lay/live/{DAY}/", ("action", "event"))
show("the greyhound early lay, live (no sources/ root)", f"greyhound/early_lay/live/{DAY}/", ("action", "event"))
show("the morning models, live", f"morning/live/{DAY}/", ("event", "action", "model"))
show("the football trader, live", f"sources/football/live/live/{DAY}/", ("action", "event"))
show("the football trader, live (no sources/ root)", f"football/live/live/{DAY}/", ("action", "event"))
show("the greyhound trader, live", f"sources/greyhound/live/live/{DAY}/", ("action", "event"))
print("\nthe account (the settle logs): 11 Nov GBP4,970.83 -> 12 Nov GBP4,959.75 = -GBP11.08; the horse trader +GBP25.31 "
      "(19 of 20 markets settled); not in the horse trader's settled figure: -GBP36.39")
