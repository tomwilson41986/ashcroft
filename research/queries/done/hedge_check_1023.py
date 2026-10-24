"""23 Oct: the trading session was cut at 07:53 UK (the UK runner's shutdown signal, run 39905555613). Is every back
closed by its lay at the SP? (read-only; the ledger in S3 and the last book times)"""
import io
import os

import boto3
import pandas as pd

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
b = os.environ.get("PRED_BUCKET") or "ashcroft"
for k in ("trading/live/2026-10-23/ledger.csv", "trading/live/2026-10-23/summary.json"):
    try:
        h = s3.head_object(Bucket=b, Key=k)
        print(k, h["LastModified"], h["ContentLength"])
    except Exception as e:  # noqa: BLE001
        print(k, e)
led = pd.read_csv(io.BytesIO(s3.get_object(Bucket=b, Key="trading/live/2026-10-23/ledger.csv")["Body"].read()))
print("events:", led.event.value_counts().to_dict(), "last ts:", led.ts.max())
for c in ("matched", "avg_price", "hedge_liability"):
    led[c] = pd.to_numeric(led[c], errors="coerce")
bk = led[led.event.eq("back") & (led.matched > 0)]
to = led[led.event.eq("trade_out")]
print(f"backs matched {len(bk)}, GBP{bk.matched.sum():.2f}; trade_out rows {len(to)}; statuses {to.status.value_counts().to_dict()}")
pos = bk.groupby(["market_id", "selection_id", "runner", "off"]).apply(
    lambda g: pd.Series({"staked": g.matched.sum(), "winnings": (g.matched * (g.avg_price - 1)).sum()})).reset_index()
lay = to[to.status.astype(str).str.upper().isin(["SUCCESS", "EXECUTABLE", "EXECUTION_COMPLETE"])].groupby(
    ["market_id", "selection_id"]).hedge_liability.sum().rename("laid")
pos = pos.merge(lay, on=["market_id", "selection_id"], how="left").fillna({"laid": 0})
pos["unhedged"] = (pos.winnings - pos.laid).round(2)
print(f"horses {len(pos)}; winnings GBP{pos.winnings.sum():.2f}, laid at SP GBP{pos.laid.sum():.2f}")
bad = pos[pos.unhedged > 1]
print(f"horses with over GBP1 of winnings not laid: {len(bad)}, GBP{bad.unhedged.sum():.2f}")
print(bad.sort_values("off").to_string())
