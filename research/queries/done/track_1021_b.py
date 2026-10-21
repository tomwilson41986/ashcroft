"""The live tracker on 21 Oct, the afternoon (the owner: "how are we doing today so far?"): done/track_1021_a.py again,
with the races settled so far: the result before and after commission, the CLV at the BSP, by race (read-only)."""
import io
import os
import runpy
import sys

import boto3
import numpy as np
import pandas as pd

sys.argv = ["track_1021_a.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "track_1021_a.py"), run_name="__main__")

print("\n== the races settled so far")
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
raw = s3.get_object(Bucket=os.environ.get("PRED_BUCKET") or "ashcroft", Key="trading/live/2026-10-21/ledger.csv")["Body"].read()
led = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
for c in ("matched", "avg_price", "bsp", "pnl", "commission", "clv"):
    led[c] = pd.to_numeric(led[c], errors="coerce")
st = led[led.event == "settle"].drop_duplicates(["market_id", "selection_id"], keep="last")
cm = led[led.event == "commission"].groupby("market_id").commission.sum()
ok = st.result.isin(["WINNER", "LOSER"]) & (st.bsp > 1)
r = st.groupby(["off", "venue", "market_id"]).apply(lambda g: pd.Series({
    "horses": len(g), "staked": g.matched.sum(), "winners": int(g.result.eq("WINNER").sum()),
    "clv_at_bsp": (g[ok.loc[g.index]].clv * g[ok.loc[g.index]].matched).sum() / max(g[ok.loc[g.index]].matched.sum(), 1e-9),
    "result": g.pnl.sum()})).reset_index()
r["commission"] = r.market_id.map(cm).fillna(0.0)
r["after"] = r.result - r.commission
r["cumulative"] = r.after.cumsum()
print(r.drop(columns="market_id").round(3).to_string(index=False))
s = st[ok]
print(f"\nsettled: {len(st)} horses in {st.market_id.nunique()} races, GBP{st.matched.sum():,.2f}; CLV at the BSP "
      f"{(s.clv * s.matched).sum() / s.matched.sum():+.2%} (GBP{(s.clv * s.matched).sum():,.2f}); result "
      f"GBP{st.pnl.sum():,.2f} before commission, GBP{st.pnl.sum() - cm.sum():,.2f} after")
