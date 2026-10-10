"""10 Oct, the first whole day of the owner's rule of 9 Oct (no-volume model, top-ups, GBP400 to win, GBP4,000 a day),
from the settled live ledger, with 9 Oct beside it (the evening check-in of 10 Oct; read-only): every back's reason,
when the day's stakes reached GBP4,000, and first backs against top-ups by UK hour, CLV at Betfair's settled price."""
import io
import os

import boto3
import numpy as np
import pandas as pd

bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
K = ["market_id", "selection_id"]


def ledger(day):
    raw = s3.get_object(Bucket=bucket, Key=f"trading/live/{day}/ledger.csv")["Body"].read()
    d = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
    for c in ["matched", "avg_price", "edge", "minutes_to_off", "bsp", "pnl", "commission", "selection_id"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["t"] = pd.to_datetime(d["ts"], utc=True, errors="coerce")
    d["uk_hour"] = d.t.dt.tz_convert("Europe/London").dt.hour
    return d


for day in ["2026-10-09", "2026-10-10"]:
    d = ledger(day)
    print(f"\n==================== {day}: {len(d)} ledger rows; events {d.event.value_counts().to_dict()}")
    b = d[(d.event == "back") & (d.matched > 0)].sort_values("t").copy()
    print("reasons on matched backs:", b.reason.fillna("").value_counts().to_dict())
    st = d[d.event == "settle"].drop_duplicates(K, keep="last")[K + ["result", "bsp", "pnl"]]
    b = b.drop(columns=["result", "bsp"]).merge(st.rename(columns={"pnl": "horse_pnl"}), on=K, how="left")
    b["bsp_used"] = b["bsp"]
    ok = b.result.isin(["WINNER", "LOSER"]) & (b.bsp_used > 1)
    b["first"] = ~b.duplicated(K)
    b["kind"] = np.where(b["first"], "first back", "top-up")
    b["clv"] = np.where(ok, b.avg_price / b.bsp_used - 1, np.nan)
    b["locked"] = b.matched * b.clv
    cum = b.matched.cumsum()
    hit = b.t[cum >= 4000]
    print(f"matched backs {len(b)} on {b.drop_duplicates(K).shape[0]} horses in {b.market_id.nunique()} markets; "
          f"GBP{b.matched.sum():,.2f}; first order {b.t.min()}, last {b.t.max()}; GBP4,000 reached "
          f"{hit.iloc[0].tz_convert('Europe/London') if len(hit) else 'never'}")
    s = b[ok]
    g = s.groupby(["uk_hour", "kind"]).agg(orders=("matched", "size"), staked=("matched", "sum"),
                                          locked=("locked", "sum"))
    g["clv"] = g.locked / g.staked
    print("by UK hour entered (each order) and kind, CLV at the settled BSP:")
    print(g.round(3).to_string())
    k = s.groupby("kind").agg(orders=("matched", "size"), staked=("matched", "sum"), locked=("locked", "sum"))
    k["clv"] = k.locked / k.staked
    print(k.round(3).to_string())
    print(f"day: staked GBP{s.matched.sum():,.2f}, CLV {s.locked.sum() / s.matched.sum():+.2%} on settled horses; "
          f"minutes to the off of first backs (median) {s[s['first']].minutes_to_off.median():.0f}")
    tot = d[d.event == "settle"]
    print(f"settled: {tot.drop_duplicates(K).shape[0]} horses, result {tot.pnl.sum():+.2f}, commission "
          f"{tot.commission.sum():.2f}" if "commission" in tot else "")
    err = d[d.event == "back"].error.fillna(d.status).astype(str)
    print("back outcomes:", err[d[d.event == "back"].matched.fillna(0) <= 0].str[:40].value_counts().head(6).to_dict())
