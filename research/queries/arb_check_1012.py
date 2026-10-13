"""Worcester 16:37 on 12 Oct (the evening check-in of 13 Oct; read-only): the other-markets scan found 'lay every
runner' in its 4 TBP market and 'back PLACE + lay 4 TBP' locking GBP4-17 a poll, which only holds if that market paid
fewer places than its name. The record of each of the race's markets: name, type, the places it carried
(number_of_winners) through the day, the runners active, and how it settled."""
import io
import os

import boto3
import pandas as pd

bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
DAY = "2026-10-12"


def csv(key):
    try:
        raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as e:  # noqa: BLE001
        print("  missing", key, type(e).__name__)
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(raw), compression="gzip" if key.endswith(".gz") else None, low_memory=False,
                       dtype={"market_id": str})


cats = []
for n in ("markets", "markets_trader", "markets_other_place"):
    c = csv(f"betfair_live/{DAY}/{n}.csv.gz")
    if not c.empty:
        c["file"] = n
        cats.append(c)
cat = pd.concat(cats, ignore_index=True)
print("catalogue columns:", list(cat.columns))
w = cat[cat.venue.astype(str).str.contains("Worc", case=False, na=False)].copy()
w["start"] = pd.to_datetime(w.market_start_utc, utc=True)
w = w[w.start.dt.strftime("%H:%M").eq("15:37")]
cols = [c for c in ("file", "market_id", "market_type", "market_name", "number_of_winners", "each_way_divisor",
                    "selection_id", "runner_name") if c in w.columns]
print(w[cols].drop_duplicates(["market_id", "selection_id"]).to_string())
mids = set(w.market_id)
for n in ("books", "books_trader", "books_other_place"):
    b = csv(f"betfair_live/{DAY}/{n}.csv.gz")
    if b.empty:
        continue
    b = b[b.market_id.isin(mids)]
    if b.empty:
        continue
    print(f"\n== {n}: columns {list(b.columns)}")
    for mid, g in b.groupby("market_id"):
        nm = w.loc[w.market_id.eq(mid), "market_name"].iloc[0]
        g = g.sort_values("polled_utc")
        if "number_of_winners" in g.columns:
            nw = g.groupby("number_of_winners", dropna=False).polled_utc.agg(["min", "max", "nunique"])
            print(f"  {mid} {nm}: number_of_winners by poll\n{nw.to_string()}")
        act = g[g.runner_status.eq("ACTIVE")].groupby("polled_utc").selection_id.nunique()
        print(f"  {mid} {nm}: active runners per poll min {act.min()} max {act.max()}; polls {g.polled_utc.nunique()}")
        last = g[g.polled_utc.eq(g.polled_utc.max())]
        print(f"  last poll {last.polled_utc.iloc[0]}: statuses {last.runner_status.value_counts().to_dict()}")
        if "market_status" in g.columns:
            print("  market status:", g.drop_duplicates("market_status")[["polled_utc", "market_status"]].to_dict("records"))
