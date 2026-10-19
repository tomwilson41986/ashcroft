"""The live tracker on 19 Oct, 09:45 UK: the first morning on the owner's staking (PR #155: to win GBP400 at the price
now, at least GBP15 and at most GBP200 a horse, top-ups to that stake), with a check that the trader stakes as built
(read-only)."""
import io
import os
import runpy
import sys

import boto3
import numpy as np
import pandas as pd

os.environ["TRACK_DAY"] = "2026-10-19"
sys.argv = ["track_day.py"]
try:
    runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "track_day.py"), run_name="__main__")
except SystemExit as e:
    print("tracker:", e)

print("\n== the staking as built? (the owner's of 17 Oct, live from 19 Oct)")
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
raw = s3.get_object(Bucket=os.environ.get("PRED_BUCKET") or "ashcroft", Key="trading/live/2026-10-19/ledger.csv")["Body"].read()
led = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
for c in ("matched", "avg_price", "target", "best_back", "selection_id"):
    led[c] = pd.to_numeric(led[c], errors="coerce")
K = ["market_id", "selection_id"]
rows = led[led.event.isin(["back", "skip"]) & led.target.notna() & (led.best_back > 1)].copy()
rows["expected"] = np.minimum(np.maximum(400.0 / (rows.best_back - 1.0), 15.0), 200.0)
off = (rows.target - rows.expected).abs() > 0.011
print(f"backs and skips with a target: {len(rows)}; target as the rule says, min(max(400/(price-1), 15), 200): "
      f"{int((~off).sum())}; otherwise: {int(off.sum())}")
if off.any():
    print(rows.loc[off, ["ts", "runner", "best_back", "target", "expected", "event"]].head(20).to_string(index=False))
b = led[(led.event == "back") & (led.matched > 0)].sort_values("ts").copy()
b["sxp"] = b.matched * b.avg_price
b["cum"] = b.groupby(K).matched.cumsum()
b["avg_before"] = (b.groupby(K).sxp.cumsum() - b.sxp) / (b.cum - b.matched)
b["k"] = b.groupby(K).cumcount()
pos = b.groupby(K).agg(runner=("runner", "first"), staked=("matched", "sum"), first_price=("avg_price", "first"),
                       backs=("matched", "size"), last_target=("target", "last")).reset_index()
print(f"horses {len(pos)}; largest stake on a horse GBP{pos.staked.max():.2f} (at most 200); "
      f"over GBP200.01: {int((pos.staked > 200.01).sum())}")
lng = pos[pos.first_price >= 25]
print(f"horses first backed at 25.0 or longer: {len(lng)}; their stakes: min GBP{lng.staked.min() if len(lng) else 0:.2f}, "
      f"median GBP{lng.staked.median() if len(lng) else 0:.2f}; under GBP14.99: {int((lng.staked < 14.99).sum())} "
      f"(a fill short of the floor can be the size offered)")
over = b[b.cum > b.target + 0.011]
print(f"backs taking a horse past its target at the time: {len(over)}")
tu = b[b.k > 0]
drift = tu[tu.avg_price > tu.avg_before * 1.0001]
print(f"top-ups {len(tu)} (GBP{tu.matched.sum():,.2f}); at a price above the horse's average matched (a drift): "
      f"{len(drift)} (GBP{drift.matched.sum():,.2f}), each within its target at the price then: "
      f"{bool((drift.cum <= drift.target + 0.011).all())} (a first back short of its stake can be filled up at a "
      f"longer price, never past the stake that wins GBP400 there)")
bands = pd.cut(pos.first_price, [1, 3, 5, 8, 13, 21, 1001], right=False,
               labels=["<3", "3-5", "5-8", "8-13", "13-21", "21+"])
print("stakes by first price:")
print(pos.groupby(bands, observed=True).staked.agg(["size", "sum", "median", "max"]).round(2).to_string())
