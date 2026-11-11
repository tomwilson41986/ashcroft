"""The owner, 11 Nov: no positions at Bangor-on-Dee today - why? Read-only.

For each of today's tracks: races on the card (S3 racecards), runners priced in the 06:00 file (predictions/<day>.csv),
WIN markets the recorder/trader saw (betfair_live/<day>/markets.csv.gz), books the trader read before 10:00 UK
(books_trader.csv.gz: reads, share with a back price, the money on offer), and the trader's ledger
(trading/live/<day>/ledger.csv): orders, fills, and skips with their reasons. Then Bangor race by race.
"""
import io
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import boto3
import pandas as pd

sys.path.insert(0, ".")
from betfair_prices import normalise_track  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
DAY = os.environ.get("REPORT_DAY") or "2026-11-11"
print("as of", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="minutes"), "UK; day", DAY)


def get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as e:  # noqa: BLE001
        print(f"  [{key}: {type(e).__name__}]")
        return None


def csv(key, **kw):
    raw = get(key)
    return None if raw is None else pd.read_csv(io.BytesIO(raw), low_memory=False, **kw)


def bangor(s):
    return s.astype(str).map(normalise_track).str.contains("bangor", case=False, na=False)


# --- the card and the 06:00 file
keys = sorted(o["Key"] for o in s3.list_objects_v2(Bucket=bucket, Prefix=f"racecards/{DAY}_").get("Contents", []))
print("\nracecards:", keys)
for k in keys[:1] + keys[-1:]:
    c = csv(k)
    if c is not None:
        tc = "track" if "track" in c.columns else "venue"
        print(f"  {k}: {len(c)} runners; by track:", c.groupby(tc).size().to_dict())
p = csv(f"predictions/{DAY}.csv")
if p is not None:
    tc = "track" if "track" in p.columns else "venue"
    tt = "race_time" if "race_time" in p.columns else None
    print(f"\npredictions/{DAY}.csv: {len(p)} runners; by track:",
          p.groupby(tc).agg(runners=(tc, "size"), races=(tt, "nunique")).to_dict("index") if tt else p.groupby(tc).size().to_dict())
    pb = p[bangor(p[tc])]
    print(f"  Bangor rows: {len(pb)}; predicted_bfsp missing {pb.predicted_bfsp.isna().sum() if len(pb) else 'n/a'}")
    if len(pb) and tt:
        print(pb.groupby(tt).agg(runners=(tc, "size"), min_price=("predicted_bfsp", "min")).to_string())

# --- the markets
mk = csv(f"betfair_live/{DAY}/markets.csv.gz", compression="gzip", dtype={"market_id": str})
bm = pd.DataFrame()
if mk is not None:
    mk["mt"] = mk.get("market_type", "WIN").fillna("WIN").str.upper()
    w = mk[mk.mt == "WIN"]
    print(f"\nmarkets.csv: {mk.market_id.nunique()} markets; WIN by venue:",
          w.groupby("venue").market_id.nunique().to_dict())
    bm = w[bangor(w.venue)]
    print("  Bangor WIN markets:")
    print(bm.drop_duplicates("market_id")[["market_id", "venue", "market_start_utc"] +
                                          [c for c in ("market_name", "country", "source") if c in bm.columns]].to_string(index=False))
ids = set(bm.market_id) if len(bm) else set()

# --- the trader's reads
bt = csv(f"betfair_live/{DAY}/books_trader.csv.gz", compression="gzip", dtype={"market_id": str})
if bt is not None:
    bt["pt"] = pd.to_datetime(bt.polled_utc, utc=True, errors="coerce")
    for c in ("back1", "back1_size", "lay1", "market_total_matched"):
        bt[c] = pd.to_numeric(bt.get(c), errors="coerce")
    venue = mk.drop_duplicates("market_id").set_index("market_id").venue if mk is not None else pd.Series(dtype=str)
    bt["venue"] = bt.market_id.map(venue)
    g = bt.groupby("venue", dropna=False).agg(markets=("market_id", "nunique"), reads=("pt", "nunique"),
                                              first=("pt", "min"), last=("pt", "max"),
                                              with_back=("back1", lambda x: x.notna().mean()),
                                              med_back_size=("back1_size", "median"),
                                              matched_max=("market_total_matched", "max"))
    print("\nthe trader's books by venue:")
    print(g.to_string())
    b = bt[bt.market_id.isin(ids)]
    if len(b):
        print("\nBangor, the trader's reads by market:")
        print(b.groupby("market_id").agg(reads=("pt", "nunique"), first=("pt", "min"), last=("pt", "max"),
                                         status=("status", lambda x: x.value_counts().to_dict()),
                                         with_back=("back1", lambda x: x.notna().mean()),
                                         spread=("lay1", lambda x: (x / b.loc[x.index, "back1"]).median()),
                                         matched=("market_total_matched", "max")).to_string())
    else:
        print("\nBangor: the trader read no book")

# --- the ledger
led = csv(f"trading/live/{DAY}/ledger.csv", dtype={"market_id": str})
if led is not None:
    vcol = next((c for c in ("venue", "track", "course") if c in led.columns), None)
    led["venue_"] = led[vcol] if vcol else led.market_id.map(
        mk.drop_duplicates("market_id").set_index("market_id").venue if mk is not None else {})
    print(f"\nledger: {len(led)} rows; events:", led.event.value_counts().to_dict())
    print(led.pivot_table(index="venue_", columns="event", values="market_id", aggfunc="count", fill_value=0).to_string())
    lb = led[bangor(led.venue_) | led.market_id.isin(ids)]
    print(f"\nBangor ledger rows: {len(lb)}")
    if len(lb):
        print("  by event and reason:")
        print(lb.groupby(["event", lb.error.fillna("")]).size().to_string())
        cols = [c for c in ("ts", "event", "market_id", "runner", "name", "price", "best_back", "p_model", "p_market",
                            "edge", "move", "error", "reason") if c in lb.columns]
        print(lb[cols].head(60).to_string(index=False))
    sk = led[led.event == "skip"]
    if len(sk):
        print("\nskips by venue and reason (all tracks):")
        print(sk.groupby(["venue_", sk.error.fillna("")]).size().to_string())
    for name in ("status.json", "summary.json", "session.json"):
        raw = get(f"trading/live/{DAY}/{name}")
        if raw:
            print(f"\n{name}:", raw.decode("utf-8", "replace")[:3000])
