"""The account's day against the horse trader's own result, 11 Nov (read-only).

The settle of 11 Nov (run 43112186658) read Betfair's funds at GBP4,970.83; 10 Nov closed at GBP5,047.97. That is
-GBP77.14 on the day, while the horse trader's settled result is +GBP57.41: GBP134.55 is not the horse trader's.
Every trader that can place a real order uses the same account. The greyhound early lay ran with real orders from
13:41 to 18:25 UTC on 11 Nov (paper-models.yml run 43031425801, GREYHOUND_EARLY_LAY live); the morning models and the
football trader bet only with their own switches.

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
DAY = os.environ.get("REPORT_DAY") or "2026-11-11"
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
print("\nthe account (the settle logs): 10 Nov GBP5,047.97 -> 11 Nov GBP4,970.83 = -GBP77.14; the horse trader +GBP57.41; "
      "not the horse trader's: -GBP134.55")
