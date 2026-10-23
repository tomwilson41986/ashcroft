"""23 Oct, the session cut at 07:53 UK: the lays at the SP as the ledger records them (read-only; the second pass of
hedge_check_1023: the lays wait unmatched for the off, so their status is PENDING, not matched)."""
import io
import os

import boto3
import pandas as pd

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
led = pd.read_csv(io.BytesIO(s3.get_object(Bucket=os.environ.get("PRED_BUCKET") or "ashcroft",
                                           Key="trading/live/2026-10-23/ledger.csv")["Body"].read()),
                  dtype={"market_id": str, "selection_id": str, "bet_id": str})
for c in ("matched", "avg_price", "hedge_liability", "asked"):
    led[c] = pd.to_numeric(led[c], errors="coerce")
bk = led[led.event.eq("back") & (led.matched > 0)]
to = led[led.event.eq("trade_out")]
print("trade_out columns filled:", {c: int(to[c].notna().sum()) for c in ("bet_id", "status", "hedge_liability",
                                                                          "asked", "matched", "error", "reason")})
print(to[["ts", "runner", "status", "bet_id", "asked", "matched", "hedge_liability", "error", "reason"]].head(6).to_string())
pos = bk.assign(w=bk.matched * (bk.avg_price - 1)).groupby(["market_id", "selection_id"]).w.sum()
lay = to.groupby(["market_id", "selection_id"]).hedge_liability.sum()
j = pd.concat({"winnings": pos, "laid": lay}, axis=1).fillna(0)
j["gap"] = j.winnings - j.laid
print(f"horses {len(j)}: winnings GBP{j.winnings.sum():.2f}, lays at the SP sent GBP{j.laid.sum():.2f}; "
      f"horses with a lay under the winnings by over GBP1: {(j.gap > 1).sum()} (GBP{j.gap[j.gap > 1].sum():.2f}); "
      f"no bet id on {to.bet_id.isna().sum()} lays")
