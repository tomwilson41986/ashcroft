"""Why the BSP at the off was not recorded on its first day (7 Oct; read-only diagnostic).

bsp_at_off_check_1007.py found no row of source "bsp" in any of the day's files: 0 of 38 horse WIN markets, 0 of 151
greyhound. The recorder (betfair_recorder.record) marks a market done when its book shows it in play, CLOSED, or
SUSPENDED after the off, then asks for its BSP (listMarketBook with SP_TRADED) three times, 15 seconds apart, keeping
the books that carry an actualSP. What the books themselves show, for each WIN market of the day record (the recorder,
on the delayed key), the trader's books (the live key) and the greyhound record: the rows by source, status, in-play
and bspReconciled; any row at all with a BSP; and, market by market, how long after the off the record first saw it in
play, suspended or closed, and in what state.
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
pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 30)


def _csv(key):
    try:
        raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(gzip.decompress(raw)), low_memory=False, dtype={"market_id": str})


cats = {t: _csv(f"betfair_live/{DAY}/markets{t}.csv.gz") for t in ("", "_trader", "_greyhound")}
offs = {}
types = {}
for c in cats.values():
    if c.empty:
        continue
    for mid, start, mt in zip(c.market_id, c.market_start_utc, c.market_type):
        offs.setdefault(mid, pd.to_datetime(start, utc=True, errors="coerce"))
        types.setdefault(mid, mt)

for name in ("books", "books_trader", "books_greyhound"):
    b = _csv(f"betfair_live/{DAY}/{name}.csv.gz")
    print(f"\n== {name}: {len(b):,} rows")
    if b.empty:
        continue
    for c in ("sp_actual", "sp_near", "sp_far", "inplay", "bsp_reconciled", "complete", "is_delayed"):
        if c in b:
            b[c] = pd.to_numeric(b[c], errors="coerce")
    b["t"] = pd.to_datetime(b["polled_utc"], utc=True, errors="coerce")
    b["off"] = b["market_id"].map(offs)
    b["mtype"] = b["market_id"].map(types).fillna("?")
    b["after_off_s"] = (b["t"] - b["off"]).dt.total_seconds()
    print("   rows by source, status, in play, BSP reconciled:")
    print(b.groupby(["source", "status", "inplay", "bsp_reconciled"], dropna=False).size().to_string())
    print(f"   rows with a BSP (sp_actual > 1): {int((b.sp_actual > 1).sum())}; with a near/far SP projection: "
          f"{int(((b.sp_near > 1) | (b.sp_far > 1)).sum())}; is_delayed: {b.is_delayed.value_counts().to_dict()}")
    rows = []
    for mid, g in b[b.mtype.eq("WIN")].groupby("market_id"):
        g = g.sort_values("t")
        off = g["off"].iloc[0]
        late = g[(g.inplay == 1) | (g.status.isin(["SUSPENDED", "CLOSED"]) & (g.after_off_s >= -60))]
        first = late.iloc[0] if len(late) else None
        before = g[g.t <= off]
        rows.append(dict(
            market=mid, off_uk=off.tz_convert(UK).strftime("%H:%M") if pd.notna(off) else "",
            last_before_s=round(float(before.after_off_s.iloc[-1]), 0) if len(before) else None,
            first_late_s=round(float(first.after_off_s), 0) if first is not None else None,
            first_late=(f"{first.source}/{first.status}/ip{int(first.inplay) if pd.notna(first.inplay) else '-'}"
                        f"/rec{int(first.bsp_reconciled) if pd.notna(first.bsp_reconciled) else '-'}")
            if first is not None else "",
            inplay_rows=int((g.inplay == 1).sum()), reconciled_rows=int((g.bsp_reconciled == 1).sum()),
            first_closed_s=round(float(g[g.status.eq("CLOSED")].after_off_s.min()), 0) if g.status.eq("CLOSED").any()
            else None))
    if rows:
        r = pd.DataFrame(rows)
        print(f"   WIN markets: {len(r)}; seen in play {int((r.inplay_rows > 0).sum())}; seen reconciled "
              f"{int((r.reconciled_rows > 0).sum())}; seconds from the off to the first late row: "
              f"{r.first_late_s.describe().round(0).to_dict()}")
        print("   the first late row's state:", r.first_late.str.rsplit("/", n=2).str[0].value_counts().to_dict(),
              "| full:", r.first_late.value_counts().head(8).to_dict())
        print(r.head(40).to_string(index=False))
