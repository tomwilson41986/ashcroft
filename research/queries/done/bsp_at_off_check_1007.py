"""The BSP at the off on its first day (7 Oct; read-only; PR #120).

From 7 Oct the recorder reads each market's BSP as Betfair reconciles it at the off: rows of source "bsp" in the day
files. A closed market's final book carries no BSP (ledger recorder-bsp-1006), so before 7 Oct the day record joined
none. For each of the day's files under s3 betfair_live/2026-10-07/ (books: the win and place record; books_other_place:
the 2/3/4 TBP and each-way markets, which have no SP of their own; books_greyhound): the markets in its catalogue by
type, how many have "bsp" rows with a BSP, the runners with one, the final-pass rows that carry one, and every WIN
market left without one, with its off (UK) against the record's polls around it. The trader was restarted at 12:00,
12:09, 12:14, 12:35, 12:41, 13:33 and 14:53 UTC, and its job (which runs these records) was down from 11:07 to 12:00
UTC, the market recorder recording from 11:07 to 11:57.
"""

from __future__ import annotations

import gzip
import io
import os

import boto3
import pandas as pd

DAY = "2026-10-07"
UK = "Europe/London"
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
pd.set_option("display.width", 220)


def _csv(key):
    try:
        raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(gzip.decompress(raw)), low_memory=False, dtype={"market_id": str})


listing = s3.list_objects_v2(Bucket=bucket, Prefix=f"betfair_live/{DAY}/").get("Contents", [])
print("S3", [(o["Key"].split("/")[-1], o["Size"], o["LastModified"].strftime("%H:%M UTC")) for o in listing])

for tag in ("", "other_place", "greyhound"):
    suffix = f"_{tag}" if tag else ""
    books = _csv(f"betfair_live/{DAY}/books{suffix}.csv.gz")
    cat = _csv(f"betfair_live/{DAY}/markets{suffix}.csv.gz")
    print(f"\n== books{suffix}: {len(books):,} rows; catalogue {cat.market_id.nunique() if not cat.empty else 0} markets")
    if books.empty or cat.empty:
        continue
    books["sp_actual"] = pd.to_numeric(books["sp_actual"], errors="coerce")
    books["t"] = pd.to_datetime(books["polled_utc"], utc=True, errors="coerce")
    print("   rows by source:", books["source"].astype(str).value_counts().to_dict())
    mk = cat.drop_duplicates("market_id").set_index("market_id")
    mk["off"] = pd.to_datetime(mk["market_start_utc"], utc=True, errors="coerce")
    at_off = books[books["source"].eq("bsp") & (books["sp_actual"] > 1)]
    final = books[books["source"].eq("final")]
    has = set(at_off["market_id"])
    rows = []
    for mtype, g in mk.groupby("market_type"):
        ids = set(g.index)
        rows.append(dict(type=mtype, markets=len(ids), with_bsp_at_off=len(ids & has),
                         runners_with_bsp=int(at_off[at_off.market_id.isin(ids)]
                                              .drop_duplicates(["market_id", "selection_id"]).shape[0]),
                         final_rows=int(final.market_id.isin(ids).sum()),
                         final_rows_with_bsp=int((final.market_id.isin(ids) & (final.sp_actual > 1)).sum())))
    print(pd.DataFrame(rows).to_string(index=False))
    win = mk[mk["market_type"].eq("WIN")]
    missing = win[~win.index.isin(has)].sort_values("off")
    if missing.empty:
        print("   every WIN market has its BSP read at the off")
        continue
    polls = books.loc[books["source"].isin(["recorder", "bsp"]), "t"].dropna().sort_values().unique()
    out = []
    for mid, r in missing.iterrows():
        off = r["off"]
        before = [p for p in polls if p <= off]
        after = [p for p in polls if p > off]
        own = books[books.market_id.eq(mid)]
        out.append(dict(market=mid, venue=r.get("venue"), off_uk=off.tz_convert(UK).strftime("%H:%M") if pd.notna(off) else "",
                        last_poll_before=pd.Timestamp(before[-1]).tz_convert(UK).strftime("%H:%M:%S") if before else "",
                        next_poll_after=pd.Timestamp(after[0]).tz_convert(UK).strftime("%H:%M:%S") if after else "",
                        own_rows=len(own), own_last=(own.t.max().tz_convert(UK).strftime("%H:%M:%S")
                                                     if len(own) else ""),
                        statuses=",".join(sorted(own["status"].astype(str).unique()))[:40] if len(own) else ""))
    print(f"   WIN markets without a BSP read at the off: {len(missing)} of {len(win)}")
    print(pd.DataFrame(out).to_string(index=False))
