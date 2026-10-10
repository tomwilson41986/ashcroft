"""10 Oct (and 9 Oct) again: first backs against top-ups by UK hour, each order's price scaled to the price Betfair
settled the horse at (a non-runner's reduction lowers it), so the CLV is the trader's own (stake_weighted_clv)
(the evening check-in of 10 Oct; read-only). research/queries/done/live_day_1010.py read the matched price."""
import io
import os

import boto3
import numpy as np
import pandas as pd

bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
K = ["market_id", "selection_id"]
for day in ["2026-10-09", "2026-10-10"]:
    raw = s3.get_object(Bucket=bucket, Key=f"trading/live/{day}/ledger.csv")["Body"].read()
    d = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
    for c in ["matched", "avg_price", "bsp", "clv", "selection_id", "minutes_to_off"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["t"] = pd.to_datetime(d["ts"], utc=True, errors="coerce")
    d["uk_hour"] = d.t.dt.tz_convert("Europe/London").dt.hour
    b = d[(d.event == "back") & (d.matched > 0)].sort_values("t").copy()
    b["sxp"] = b.matched * b.avg_price
    hm = b.groupby(K).agg(m=("matched", "sum"), sxp=("sxp", "sum")).reset_index()
    hm["matched_price"] = hm.sxp / hm.m
    st = d[d.event == "settle"].drop_duplicates(K, keep="last")[K + ["result", "bsp", "avg_price", "clv"]] \
        .rename(columns={"avg_price": "settled_price", "clv": "horse_clv"})
    b = b.drop(columns=["result", "bsp", "clv"]).merge(hm[K + ["matched_price"]], on=K).merge(st, on=K, how="left")
    ok = b.result.isin(["WINNER", "LOSER"]) & (b.bsp > 1) & b.horse_clv.notna()
    b["price_s"] = b.avg_price * b.settled_price / b.matched_price
    b["first"] = ~b.duplicated(K)
    b["kind"] = np.where(b["first"], "first back", "top-up")
    b["clv"] = np.where(ok, b.price_s / b.bsp - 1, np.nan)
    b["locked"] = b.matched * b.clv
    s = b[ok]
    reduced = (st.settled_price.notna()) & (hm.set_index(K).reindex(pd.MultiIndex.from_frame(st[K])).matched_price
                                            .to_numpy() - st.settled_price.to_numpy() > 0.005)
    print(f"\n==================== {day}: settled horses {ok.groupby([b.market_id, b.selection_id]).any().sum()}, "
          f"price reduced for a non-runner on {int(reduced.sum())}")
    g = s.groupby(["uk_hour", "kind"]).agg(orders=("matched", "size"), staked=("matched", "sum"),
                                          locked=("locked", "sum"))
    g["clv"] = g.locked / g.staked
    print(g.round(3).to_string())
    k = s.groupby("kind").agg(orders=("matched", "size"), staked=("matched", "sum"), locked=("locked", "sum"))
    k["clv"] = k.locked / k.staked
    print(k.round(3).to_string())
    print(f"day: staked GBP{s.matched.sum():,.2f}, CLV {s.locked.sum() / s.matched.sum():+.2%} (the trader's horse-level "
          f"stake-weighted: {(st.horse_clv * hm.set_index(K).reindex(pd.MultiIndex.from_frame(st[K])).m.to_numpy()).sum() / hm.m.sum():+.2%})")
    cum = b.matched.cumsum()
    print("cumulative stake by UK time: GBP1,000 at", b.t[cum >= 1000].min(), "GBP2,000 at", b.t[cum >= 2000].min(),
          "GBP3,000 at", b.t[cum >= 3000].min(), "last at", b.t.max(), f"(GBP{cum.iloc[-1]:,.2f})")
