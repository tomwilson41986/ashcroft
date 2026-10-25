"""25 Oct: GBP590 of the morning's stakes void by 08:43 UK, 7 of 29 races gone from the races to run. Which races,
and why (read-only: the ledger in S3 and the recorder's books)."""
import io
import os

import boto3
import pandas as pd

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
b = os.environ.get("PRED_BUCKET") or "ashcroft"
led = pd.read_csv(io.BytesIO(s3.get_object(Bucket=b, Key="trading/live/2026-10-25/ledger.csv")["Body"].read()),
                  dtype={"market_id": str, "selection_id": str})
led["matched"] = pd.to_numeric(led["matched"], errors="coerce")
print("events:", led.event.value_counts().to_dict(), "last ts:", led.ts.max())
bk = led[led.event.eq("back") & (led.matched > 0)]
print(bk.groupby("venue").agg(races=("market_id", "nunique"), horses=("selection_id", "nunique"), staked=("matched", "sum"))
      .round(2).to_string())
for ev in ("void", "settle", "skip"):
    x = led[led.event.eq(ev)]
    if len(x):
        print(f"\n== {ev}: {len(x)} rows")
        print(x.groupby(["venue", "off"]).agg(n=("selection_id", "size"), reasons=("reason", lambda s: "; ".join(sorted(set(map(str, s)))[:3])))
              .to_string()[:3000])
print("\nother events and reasons:")
print(led[~led.event.isin(["back", "trade_out", "skip"])][["ts", "event", "venue", "off", "runner", "reason", "status", "error"]]
      .tail(40).to_string())
