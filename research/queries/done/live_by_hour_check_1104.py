"""Why the fills' results in live_by_hour_1104 ran GBP1,562 above the ledger's settled results: each horse's results
from its fills (laid at the SP for each fill's winnings) against the ledger's settle row (Betfair's settlement), by
the horse's result, whether it was fully laid, and the lay-off rows' status. Read-only, 25 Oct - 4 Nov."""
import io
import os
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
pd.set_option("display.width", 250)
K = ["market_id", "selection_id"]
out = []
for i in range(11):
    day = (date(2026, 10, 25) + timedelta(i)).isoformat()
    raw = s3.get_object(Bucket=bucket, Key=f"trading/live/{day}/ledger.csv")["Body"].read()
    led = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
    for c in ("matched", "avg_price", "selection_id", "pnl", "pnl_back", "pnl_lay", "bsp", "hedge_liability", "asked"):
        led[c] = pd.to_numeric(led[c], errors="coerce")
    st = led[led.event == "settle"].drop_duplicates(K, keep="last")
    b = led[(led.event == "back") & (led.matched > 0)]
    to = led[led.event == "trade_out"]
    if i == 0:
        print("trade_out statuses:", to.status.value_counts().to_dict())
        print("settle result values:", st.result.value_counts().to_dict())
        print("settle rows a horse:", led[led.event == "settle"].groupby(K).size().value_counts().to_dict())
        print(st[["selection_id", "matched", "avg_price", "hedged", "hedge_liability", "result", "bsp",
                  "pnl_back", "pnl_lay", "pnl", "error"]].head(12).to_string())
    g = b.groupby(K).apply(lambda x: pd.Series({"staked": x.matched.sum(), "won_at_fill": (x.matched * (x.avg_price - 1)).sum()})).reset_index()
    lay = to[to.status.isin(["SUCCESS", "PENDING"])].groupby(K).hedge_liability.sum().rename("laid").reset_index()
    g = g.merge(lay, on=K, how="left").merge(st[K + ["result", "bsp", "pnl", "pnl_back", "pnl_lay", "hedge_liability"]], on=K, how="left")
    g["day"] = day
    out.append(g)
g = pd.concat(out, ignore_index=True)
win = g.result.eq("WINNER")
g["mine"] = np.where(g.result.eq("REMOVED"), 0, np.where(win, g.won_at_fill - g.laid.fillna(0),
                     -g.staked + g.laid.fillna(0) / (g.bsp - 1)))
g["full"] = (g.laid.fillna(0) >= g.won_at_fill - 0.05)
g["diff"] = g.mine - g.pnl
print(f"\nhorses {len(g)}; mine GBP{g.mine.sum():+,.2f} against the ledger GBP{g.pnl.sum():+,.2f}")
print(g.groupby(["result", "full"]).agg(n=("diff", "size"), mine=("mine", "sum"), ledger=("pnl", "sum"),
                                        diff=("diff", "sum"), laid=("laid", "sum"), led_liab=("hedge_liability", "sum"),
                                        winnings=("won_at_fill", "sum")).round(2).to_string())
print(g.reindex(g["diff"].abs().sort_values(ascending=False).index).head(10)[
    ["day", "selection_id", "staked", "won_at_fill", "laid", "hedge_liability", "result", "bsp", "pnl_back", "pnl_lay",
     "pnl", "mine"]].round(2).to_string(index=False))
