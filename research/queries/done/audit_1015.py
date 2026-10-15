"""The owner's ask (15 Oct, 19:40 UTC): audit today's trading. Did something change, did we do something wrong, or is it
variance? Read-only.

Today is not settled yet (the settle runs at 21:45 UTC), so each horse's BSP is read from the day's recorded books
(the last sp_actual Betfair showed after the off), and the CLV at that BSP; 10-14 Oct are read both ways (the settle's
own rows, and the books' BSP) so the method is checked against the settled figure. Then the things that could have
changed: the rule (the config the session ran), the model (its file in S3 and the prices it gave), what was backed (price
band, time to off, first backs against top-ups, horses), the expected CLV at entry against the realised, the lays at the
SP (a back not laid off carries the race's result), non-runner reductions, and the hours entered."""
import io
import json
import os

import boto3
import numpy as np
import pandas as pd

bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
DAYS = ["2026-10-10", "2026-10-11", "2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15"]
K = ["market_id", "selection_id"]
pd.set_option("display.width", 250)


def get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as e:  # noqa: BLE001
        print("  missing", key, type(e).__name__)
        return None


def csv(key):
    raw = get(key)
    if raw is None:
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(raw), compression="gzip" if key.endswith(".gz") else None, low_memory=False,
                       dtype={"market_id": str})


def books_bsp(day):
    """Each runner's BSP as the recorded books last showed it after the off."""
    frames = [csv(f"betfair_live/{day}/{n}.csv.gz") for n in ("books_trader", "books")]
    b = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if b.empty or "sp_actual" not in b.columns:
        return pd.DataFrame(columns=K + ["bsp_books", "status_books"])
    b["sp_actual"] = pd.to_numeric(b.sp_actual, errors="coerce")
    b["selection_id"] = pd.to_numeric(b.selection_id, errors="coerce")
    b = b.sort_values("polled_utc")
    sp = b[b.sp_actual > 1].groupby(K).sp_actual.last().rename("bsp_books")
    st = b.groupby(K).runner_status.last().rename("status_books")
    return pd.concat([sp, st], axis=1).reset_index()


summary_rows = []
for day in DAYS:
    print(f"\n==================== {day}")
    led = csv(f"trading/live/{day}/ledger.csv")
    if led.empty:
        continue
    raw = get(f"trading/live/{day}/summary.json")
    summ = json.loads(raw) if raw else {}
    if day == DAYS[0]:
        print("ledger columns:", list(led.columns))
    print("events:", led.event.value_counts().to_dict())
    if summ:
        cfg = summ.get("config", {})
        print("summary:", {k: summ.get(k) for k in ("markets", "bets_matched", "turnover", "races_settled", "settled_pnl",
                                                     "stake_weighted_clv", "backs_refused", "funds_start")})
        print("config:", {k: cfg.get(k) for k in ("strategy", "target", "clv_bar", "closing_volume", "topups",
                                                   "trade_from", "trade_until", "stop_before_off", "trade_out",
                                                   "trade_out_at", "commission", "min_edge", "margin", "model_weight")},
              "limits:", cfg.get("limits"))
    for c in ["matched", "avg_price", "bsp", "clv", "selection_id", "minutes_to_off", "price", "size", "pnl"]:
        if c in led.columns:
            led[c] = pd.to_numeric(led[c], errors="coerce")
    led["t"] = pd.to_datetime(led["ts"], utc=True, errors="coerce")
    led["uk"] = led.t.dt.tz_convert("Europe/London")
    b = led[(led.event == "back") & (led.matched > 0)].sort_values("t").copy()
    b["first"] = ~b.duplicated(K)
    b["kind"] = np.where(b["first"], "first", "top-up")
    bb = books_bsp(day)
    st = led[led.event == "settle"].drop_duplicates(K, keep="last")
    b = b.drop(columns=[c for c in ("bsp", "clv", "result") if c in b.columns]).merge(bb, on=K, how="left")
    if len(st):
        b = b.merge(st[K + ["bsp", "result", "avg_price"]].rename(columns={"avg_price": "settled_price"}), on=K,
                    how="left")
    else:
        b["bsp"], b["result"], b["settled_price"] = np.nan, np.nan, np.nan
    hm = b.groupby(K).apply(lambda g: (g.matched * g.avg_price).sum() / g.matched.sum()).rename("mp").reset_index()
    b = b.merge(hm, on=K)
    # the price as settled (a non-runner's reduction lowers it); before the settle, the matched price
    b["price_s"] = np.where(b.settled_price.notna(), b.avg_price * b.settled_price / b.mp, b.avg_price)
    b["bsp_use"] = b.bsp.where(b.bsp > 1, b.bsp_books)
    b["clv"] = b.price_s / b.bsp_use - 1
    b["clv_books"] = b.avg_price / b.bsp_books - 1
    ok = b.bsp_use > 1
    s = b[ok]
    staked = s.matched.sum()
    clv = (s.clv * s.matched).sum() / staked if staked else np.nan
    print(f"backs {len(b)} on {b.selection_id.nunique()} horses in {b.market_id.nunique()} markets; with a BSP "
          f"{ok.sum()}; staked GBP{b.matched.sum():,.2f}; CLV at the BSP used {clv:+.2%}; at the books' BSP and the "
          f"matched price {(s.clv_books * s.matched).sum() / s.matched.sum():+.2%}")
    reduced = (b.settled_price.notna() & (b.mp - b.settled_price > 0.005)).groupby([b.market_id, b.selection_id]).any()
    print("horses with a price cut for a non-runner:", int(reduced.sum()), "| runners removed per the books:",
          int((bb.status_books == "REMOVED").sum()))
    # what was backed
    s = s.assign(band=pd.cut(s.avg_price, [1, 3, 5, 8, 13, 21, 1000], labels=["<3", "3-5", "5-8", "8-13", "13-21",
                                                                            "21+"]),
                 hr=s.uk.dt.hour,
                 mto=pd.cut(s.minutes_to_off, [-1, 60, 180, 360, 720, 2000], labels=["<1h", "1-3h", "3-6h", "6-12h",
                                                                                    "12h+"]) if "minutes_to_off" in s
                 else "na")

    def tab(by):
        g = s.groupby(by, observed=True).agg(n=("matched", "size"), staked=("matched", "sum"),
                                             locked=("clv", lambda x: (x * s.loc[x.index, "matched"]).sum()))
        g["clv"] = g.locked / g.staked
        g["share"] = g.staked / g.staked.sum()
        return g.round(3)
    print("by price band:\n", tab("band").to_string())
    print("by UK hour and kind:\n", tab(["hr", "kind"]).to_string())
    print("by time to off:\n", tab("mto").to_string())
    exp_cols = [c for c in led.columns if "exp" in c.lower() or "edge" in c.lower() or c.lower() in ("ev", "model_price",
                                                                                                     "predicted_bfsp")]
    if exp_cols:
        for c in exp_cols:
            b[c] = pd.to_numeric(b[c], errors="coerce")
        print("entry columns:", exp_cols, {c: round(float((b.loc[ok, c] * b.loc[ok, "matched"]).sum()
                                                         / b.loc[ok, "matched"].sum()), 4)
                                         for c in exp_cols if b[c].notna().any()})
    # the lays at the SP
    lays = led[led.event.astype(str).str.contains("lay|close|trade", case=False, na=False)]
    if len(lays):
        print("lay/close events:", lays.event.value_counts().to_dict())
        if "status" in lays.columns:
            print("  their status:", lays.status.value_counts().to_dict())
        if "error" in lays.columns:
            print("  errors:", lays.error.value_counts().head(8).to_dict())
    err = led[led.get("error").notna()] if "error" in led.columns else led.iloc[:0]
    if len(err):
        print("events with an error:", err.groupby(["event", "error"]).size().sort_values(ascending=False).head(10)
              .to_dict())
    if "pnl" in led.columns and len(st):
        stp = st.copy()
        stp["pnl"] = pd.to_numeric(stp.pnl, errors="coerce")
        print("settle pnl sum", round(float(stp.pnl.sum()), 2))
    # the top horses by money and their outcome
    h = s.groupby(K + ["result"], dropna=False).agg(staked=("matched", "sum"), price=("avg_price", "mean"),
                                                    bsp=("bsp_use", "first"), clv=("clv", "mean")).reset_index()
    print("the 8 horses with the most money:\n", h.sort_values("staked", ascending=False).head(8).round(3)
          .to_string(index=False))
    summary_rows.append(dict(day=day, staked=round(staked, 2), clv=round(clv, 4), horses=int(s.selection_id.nunique()),
                             first_clv=round(float((s[s["first"]].clv * s[s["first"]].matched).sum()
                                                   / max(s[s["first"]].matched.sum(), 1e-9)), 4),
                             topup_clv=round(float((s[~s["first"]].clv * s[~s["first"]].matched).sum()
                                                   / max(s[~s["first"]].matched.sum(), 1e-9)), 4),
                             median_price=round(float(s.avg_price.median()), 2),
                             share_under_5=round(float(s[s.avg_price < 5].matched.sum() / staked), 3) if staked else None,
                             last_back_uk=b.uk.max().strftime("%H:%M") if len(b) else None,
                             settled_pnl=summ.get("settled_pnl"), trader_clv=summ.get("stake_weighted_clv")))

print("\n== the days side by side (CLV at the BSP; today's BSP from the books, before the settle)")
print(pd.DataFrame(summary_rows).to_string(index=False))

# the model: the prices it gave, day by day, against the market the trader saw
print("\n== the 06:00 predictions by day")
for day in DAYS:
    raw = get(f"predictions/{day}.csv")
    if raw is None:
        continue
    p = pd.read_csv(io.BytesIO(raw))
    q = pd.to_numeric(p.get("predicted_bfsp"), errors="coerce")
    head = s3.head_object(Bucket=bucket, Key=f"predictions/{day}.csv")
    cols = [c for c in p.columns if "model" in c.lower() or "version" in c.lower()]
    print(day, f"written {head['LastModified']}", f"runners {len(p)}", f"races {p.groupby(['track', 'race_time']).ngroups}"
          if {"track", "race_time"} <= set(p.columns) else "", f"median predicted BFSP {q.median():.2f}",
          f"book {float((1 / q).groupby([p.track, p.race_time]).sum().median()):.3f}" if {"track", "race_time"} <=
          set(p.columns) else "", "model cols:", {c: p[c].iloc[0] for c in cols[:4]})
print("\n== the model files in S3 (last modified)")
for prefix in ("models/", "data/models/", "bfsp_model"):
    try:
        res = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=50).get("Contents", [])
        for o in res[:20]:
            print(" ", o["Key"], o["LastModified"], o["Size"])
    except Exception as e:  # noqa: BLE001
        print("  list", prefix, type(e).__name__)
