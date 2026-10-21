"""The owner's question (21 Oct): "When one of our horses wins we earn nothing. Even if we get EV. Wouldn't it be better
to still get a proportion of profit in that scenario." (read-only)

Each back is laid at the SP for liability L. With back stake S at price P and the BSP B: a winner makes S(P-1) - L, a
loser -S + L/(B-1). The rule now: L = S(P-1) (a winner nets nothing). The expectation under the BSP's chances is
S(P/B - 1) whatever L is; L only moves the profit between the outcomes. Every settled position of 30 Sep - 20 Oct
replayed under:
  now          L = S(P-1);
  even         L = S P (E-1)/E with E the closing model's expected BSP at the first back (1 / p_pool): a winner
               makes S(P/E - 1), a loser about S(P/B - 1), the profit spread over both;
  half         L = S(P-1)/2 (half held to the result);
  none         no lay (held to the result).
Commission at 2% on each market's net winnings. The total, by day, the spread of the days, the worst day, and the
winners' and losers' shares of the profit.
"""
import io
import os
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
pd.set_option("display.width", 250)
K = ["day", "market_id", "selection_id"]
DAYS = [(date(2026, 9, 30) + timedelta(n)).isoformat() for n in range(21)]
leds = []
for day in DAYS:
    try:
        raw = s3.get_object(Bucket=bucket, Key=f"trading/live/{day}/ledger.csv")["Body"].read()
    except Exception:  # noqa: BLE001
        continue
    d = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
    d["day"] = day
    leds.append(d)
led = pd.concat(leds, ignore_index=True)
for c in ("matched", "avg_price", "bsp", "selection_id", "p_pool", "pnl"):
    led[c] = pd.to_numeric(led[c], errors="coerce")
led["t"] = pd.to_datetime(led.ts, utc=True, errors="coerce")
b = led[(led.event == "back") & (led.matched > 0)].sort_values("t").copy()
b["sxp"] = b.matched * b.avg_price
b["sxe"] = b.matched / b.p_pool                       # the expected BSP at each fill, stake-weighted
pos = b.groupby(K).agg(S=("matched", "sum"), sxp=("sxp", "sum"), sxe=("sxe", "sum")).reset_index()
pos["E"] = pos.sxe / pos.S
st = led[led.event == "settle"].sort_values("t").drop_duplicates(K, keep="last")[K + ["result", "bsp", "avg_price", "pnl"]] \
    .rename(columns={"avg_price": "P", "bsp": "B"})
v = pos.merge(st, on=K)
v = v[v.result.isin(["WINNER", "LOSER"]) & (v.B > 1) & (v.P > 1) & (v.E > 1)].copy()
win = v.result.eq("WINNER").to_numpy()
S, P, B, E = (v[c].to_numpy(float) for c in ("S", "P", "B", "E"))
L = {"now": S * (P - 1), "even": np.clip(S * P * (E - 1) / E, 0, S * (P - 1)), "half": S * (P - 1) / 2, "none": 0 * S}
print(f"positions settled with a BSP, 30 Sep - 20 Oct: {len(v):,} ({int(win.sum())} winners), GBP{S.sum():,.2f}; "
      f"expected at the BSP (CLV x stake) GBP{(S * (P / B - 1)).sum():,.2f}")
print(f"check: 'now' against the ledger's settled result: GBP{np.where(win, S * (P - 1) - L['now'], -S + L['now'] / (B - 1)).sum():,.2f} "
      f"against GBP{v.pnl.sum():,.2f}")
rows = []
for name, l in L.items():
    pl = np.where(win, S * (P - 1) - l, -S + l / (B - 1))
    v["pl"] = pl
    mkt = v.groupby(["day", "market_id"]).pl.sum()
    net = mkt - 0.02 * mkt.clip(lower=0)
    day = net.groupby(level=0).sum()
    rows.append(pd.Series({"total_before_commission": pl.sum(), "commission": (0.02 * mkt.clip(lower=0)).sum(),
                           "total_after": net.sum(), "winners_share": pl[win].sum(), "losers_share": pl[~win].sum(),
                           "winners_paid_median": float(np.median(pl[win])), "sd_day": day.std(),
                           "worst_day": day.min(), "losing_days": int((day < 0).sum()), "days": len(day),
                           "lay_liability": l.sum()}, name=name))
print(pd.DataFrame(rows).round(2).to_string())
