"""25 Oct (and 10-24 Oct), the sixteenth day (settled 22:18 UTC, behind the recorder's evening step: the clocks went back) of the owner's rule of 9 Oct: first backs against top-ups by UK hour, each order's price scaled to the price Betfair
settled the horse at (a non-runner's reduction lowers it), so the CLV is the trader's own (stake_weighted_clv)
(the evening check-in of 25 Oct; read-only). research/queries/done/live_day_1024.py with 25 Oct added; before that
research/queries/done/live_day_1023.py with 24 Oct added; before that
research/queries/done/live_day_1022.py with 23 Oct added; before that
research/queries/done/live_day_1020.py with 21 Oct added; before that
research/queries/done/live_day_1019.py with 20 Oct added; before that
research/queries/done/live_day_1018.py with 19 Oct added; before that
research/queries/done/live_day_1017.py with 18 Oct added; before that
research/queries/done/live_day_1016.py with 17 Oct added; before that
research/queries/done/live_day_1015.py with 16 Oct added, and each day's
shape (horses backed, the ten largest horses' share, the share staked under 5.0), and the early window (06:15-07:59 UK, the owner's start from 13 Oct) apart."""
import io
import os

import boto3
import numpy as np
import pandas as pd

bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
K = ["market_id", "selection_id"]
for day in ["2026-10-10", "2026-10-11", "2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16", "2026-10-17", "2026-10-18", "2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23", "2026-10-24", "2026-10-25"]:
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
    b["window"] = np.select([b.t.dt.tz_convert("Europe/London").dt.strftime("%H:%M") < "08:00",
                             b.uk_hour < 10], ["06:15-07:59", "08:00-09:59"], "10:00 on")
    s = b[ok]
    w = s.groupby(["window", "kind"]).agg(orders=("matched", "size"), staked=("matched", "sum"),
                                         locked=("locked", "sum"))
    w["clv"] = w.locked / w.staked
    print("by window entered (locked = the hedged profit before commission):")
    print(w.round(3).to_string())
    w2 = s.groupby("window").agg(orders=("matched", "size"), staked=("matched", "sum"), locked=("locked", "sum"),
                                 horses=("selection_id", "nunique"))
    w2["clv"] = w2.locked / w2.staked
    print(w2.round(3).to_string())
    hs = b.groupby(K).matched.sum()
    print(f"shape: horses backed {len(hs)}, first backs GBP{b[b['first']].matched.sum():,.2f}, top-ups "
          f"GBP{b[~b['first']].matched.sum():,.2f}; ten largest horses {hs.nlargest(10).sum() / hs.sum():.0%}; "
          f"staked under 5.0 {b[b.avg_price < 5].matched.sum() / b.matched.sum():.0%}")
    print("first back:", b.t.min(), "UK", b.t.min().tz_convert("Europe/London").strftime("%H:%M"))
    try:
        keys = [o["Key"] for o in s3.list_objects_v2(Bucket=bucket, Prefix=f"trading/live/{day}/").get("Contents", [])]
        print("objects:", keys)
        for k in keys:
            if k.endswith(".json"):
                j = s3.get_object(Bucket=bucket, Key=k)["Body"].read().decode()
                print(k, j[:1500])
    except Exception as e:  # noqa: BLE001
        print("summary read failed:", e)
for day in ["2026-10-24", "2026-10-25"]:
    h = s3.head_object(Bucket=bucket, Key=f"predictions/{day}.csv")
    print(f"predictions/{day}.csv written {h['LastModified']} ({h['ContentLength']:,} bytes)")
