"""The owner (22 Oct): "never give me inflated expected value, i need real figures that i can expect and forecast"
(read-only). Every live day 10-21 Oct: the trader's expected value at the fill (sum of each back's expected CLV x its
stake, the ledger's edge), the CLV x stake Betfair settled (after the non-runners' cuts) and the result after
commission (Betfair's settled P&L less commission). The ratio of the settled to the expected is the calibration that
turns a day's expected value into a forecast; the spread of the days' results is the forecast's range. Then today's
book so far through it."""
import io
import os

import boto3
import numpy as np
import pandas as pd

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
pd.set_option("display.width", 250)
K = ["market_id", "selection_id"]
rows = []
for d in range(10, 23):
    day = f"2026-10-{d}"
    raw = s3.get_object(Bucket=bucket, Key=f"trading/live/{day}/ledger.csv")["Body"].read()
    led = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
    for c in ("matched", "avg_price", "edge", "bsp", "pnl", "commission", "clv"):
        led[c] = pd.to_numeric(led[c], errors="coerce")
    b = led[(led.event == "back") & (led.matched > 0)]
    st = led[led.event == "settle"].drop_duplicates(K, keep="last")
    ok = st.result.isin(["WINNER", "LOSER"]) & (st.bsp > 1)
    rows.append({"day": day, "staked": b.matched.sum(), "ev_at_fill": float((b.edge * b.matched).sum()),
                 "settled_clv_x_stake": float((st.clv * st.matched)[ok].sum()) if ok.any() else np.nan,
                 "void": float(st.matched[st.result.eq("REMOVED")].sum()),
                 "result": float(st.pnl.sum() - led[led.event == "commission"].commission.sum()) if len(st) else np.nan,
                 "races_settled": int(st.market_id.nunique())})
t = pd.DataFrame(rows).set_index("day")
print(t.round(2).to_string())
done = t[t.index < "2026-10-22"]
k_clv = done.settled_clv_x_stake.sum() / done.ev_at_fill.sum()
k_res = done.result.sum() / done.ev_at_fill.sum()
print(f"\n10-21 Oct: expected value at the fill GBP{done.ev_at_fill.sum():,.2f}; settled CLV x stake "
      f"GBP{done.settled_clv_x_stake.sum():,.2f} ({k_clv:.2f} of it); result after commission GBP{done.result.sum():,.2f} "
      f"({k_res:.2f} of it)")
for lab, x in (("10-21 Oct", done), ("15-21 Oct (the last 7 days)", done[done.index >= "2026-10-15"])):
    print(f"   {lab}: result a day mean GBP{x.result.mean():,.2f}, median GBP{x.result.median():,.2f}, sd "
          f"GBP{x.result.std():,.2f}, worst GBP{x.result.min():,.2f}, best GBP{x.result.max():,.2f}; days up "
          f"{int((x.result > 0).sum())} of {len(x)}; settled / expected {x.settled_clv_x_stake.sum() / x.ev_at_fill.sum():.2f}")
# the days' result against their own calibrated forecast: how far a day lands from it
f = done.ev_at_fill * k_res
err = done.result - f
print(f"   a day's result less its calibrated forecast: sd GBP{err.std():,.2f}; 10th-90th percentile GBP{err.quantile(0.1):,.2f} "
      f"to GBP{err.quantile(0.9):,.2f}")
today = t.loc["2026-10-22"]
print(f"\n22 Oct so far: staked GBP{today.staked:,.2f}, expected value at the fill GBP{today.ev_at_fill:,.2f}; forecast "
      f"(x {k_res:.2f}, the record's result over expected) GBP{today.ev_at_fill * k_res:,.2f}; "
      f"the last 7 days' ratio would give GBP{today.ev_at_fill * done[done.index >= '2026-10-15'].result.sum() / done[done.index >= '2026-10-15'].ev_at_fill.sum():,.2f}; "
      f"range (10th-90th of past days' misses) GBP{today.ev_at_fill * k_res + err.quantile(0.1):,.2f} to "
      f"GBP{today.ev_at_fill * k_res + err.quantile(0.9):,.2f}")
