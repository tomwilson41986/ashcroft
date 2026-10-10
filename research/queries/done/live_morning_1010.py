"""The trader's ledger so far on 10 Oct, the first day of the owner's rule of 9 Oct (read-only): does every back read
the no-volume model, how much is staked by UK hour, first backs against top-ups, and how close the day is to GBP4,000."""
import io
import os

import boto3
import pandas as pd

bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
raw = s3.get_object(Bucket=bucket, Key="trading/live/2026-10-10/ledger.csv")["Body"].read()
d = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
for c in ["matched", "asked", "target", "edge", "minutes_to_off", "avg_price"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
d["ts"] = pd.to_datetime(d["ts"], utc=True)
print(f"== ledger rows {len(d)}, {d.ts.min()} to {d.ts.max()}")
print(d.event.value_counts().to_string())
print("\n== reasons (all rows with one)")
print(d.reason.fillna("").value_counts().head(15).to_string())
b = d[(d.matched > 0) & d.event.astype(str).str.contains("back")].copy()
if not len(b):
    b = d[d.matched > 0].copy()
print(f"\n== matched rows {len(b)}: GBP{b.matched.sum():,.2f} matched across {b.selection_id.nunique()} horses, "
      f"{b.market_id.nunique()} markets")
b = b.sort_values("ts")
b["first"] = ~b.duplicated(["market_id", "selection_id"])
b["uk_hour"] = b.ts.dt.tz_convert("Europe/London").dt.hour
print(b.groupby(["uk_hour", "first"]).matched.agg(["count", "sum"]).round(2).to_string())
print(f"\nfirst backs GBP{b[b['first']].matched.sum():,.2f}; top-ups GBP{b[~b['first']].matched.sum():,.2f}")
cum = b.matched.cumsum()
hit = b.ts[cum >= 4000]
print("day limit GBP4,000 reached at", hit.iloc[0] if len(hit) else "not yet", f"(GBP{cum.iloc[-1] if len(cum) else 0:,.2f} so far)")
print("\n== errors / refusals")
print(d.loc[d.error.notna(), ["ts", "event", "status", "error"]].tail(10).to_string())
