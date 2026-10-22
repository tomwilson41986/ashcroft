"""22 Oct (read-only): the markets the live ledger left unsettled (the trader's summary: 38 races settled of 40 markets;
the account moved GBP367.72 against the ledger's GBP294.68 before commission), our money in each, and what the
day's record says of them (market status, our horses' results and BSPs)."""
import io
import os

import boto3
import pandas as pd

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
pd.set_option("display.width", 250)
DAY = "2026-10-22"
led = pd.read_csv(io.BytesIO(s3.get_object(Bucket=bucket, Key=f"trading/live/{DAY}/ledger.csv")["Body"].read()),
                  low_memory=False, dtype={"market_id": str})
for c in ("matched", "avg_price", "pnl", "commission", "selection_id", "bsp"):
    led[c] = pd.to_numeric(led[c], errors="coerce")
backed = led[(led.event == "back") & (led.matched > 0)].groupby(["market_id", "venue", "off"]).matched.sum()
settled = set(led[led.event == "settle"].market_id)
print("markets backed:", len(backed), "settled in the ledger:", len(settled & set(backed.index.get_level_values(0))))
miss = backed[~backed.index.get_level_values(0).isin(settled)]
print("unsettled:\n", miss.to_string())
st = led[led.event == "settle"]
print(f"ledger: settled pnl GBP{st.pnl.sum():,.2f}, commission rows GBP{led[led.event == 'commission'].commission.sum():,.2f}")
print("event counts:", led.event.value_counts().to_dict())
for mid in miss.index.get_level_values(0):
    x = led[led.market_id == mid]
    print(f"\n== {mid}: events {x.event.value_counts().to_dict()}")
    print(x[["ts", "event", "runner", "matched", "avg_price", "status", "error", "result", "bsp", "pnl"]].tail(12).to_string(index=False))
