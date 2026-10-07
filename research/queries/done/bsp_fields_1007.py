"""Does any Betfair read after the off carry Starting Price data, on either key? (7 Oct; read-only diagnostic)

bsp_at_off_diagnose_1007.py found no BSP in any book of 7 Oct: not in the recorder's (the delayed key; its reads at the
off ask SP_TRADED and keep only the books that carry an actualSP, and none did) and not in the trader's (the live key
since 6 Oct 15:25 UK; after the off it reads each race's book with SP_TRADED to settle). The trader's reads are kept
whole, so they say whether Betfair sends the StartingPrices object at all after the off: its near and far price, the
money taken at SP (backStakeTaken, layLiabilityTaken) and the actualSP. Each day 1-7 Oct, by writer, key (is_delayed)
and state: the rows, and how many carry each SP field.
"""

from __future__ import annotations

import gzip
import io
import os

import boto3
import pandas as pd

DAYS = [f"2026-10-0{d}" for d in range(1, 8)]
SP = ["sp_near", "sp_far", "sp_actual", "sp_back_taken", "sp_lay_taken"]
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


for day in DAYS:
    for name in ("books_trader", "books"):
        b = _csv(f"betfair_live/{day}/{name}.csv.gz")
        if b.empty:
            print(f"\n== {day} {name}: none")
            continue
        for c in SP + ["inplay", "bsp_reconciled", "is_delayed"]:
            b[c] = pd.to_numeric(b[c], errors="coerce") if c in b else float("nan")
        late = b[(b.inplay == 1) | (b.bsp_reconciled == 1) | b.status.eq("CLOSED")]
        print(f"\n== {day} {name}: {len(b):,} rows, {len(late):,} after the off (in play, reconciled or closed)")
        if late.empty:
            continue
        g = late.groupby(["source", "is_delayed", "status", "inplay", "bsp_reconciled"], dropna=False)
        out = g.size().rename("rows").to_frame()
        for c in SP:
            out[c] = g[c].apply(lambda s: int(s.notna().sum()))
        print(out.to_string())
        pre = b[(b.inplay != 1) & (b.bsp_reconciled != 1) & ~b.status.eq("CLOSED")]
        print(f"   before the off: {len(pre):,} rows; with any SP field: "
              f"{int(pre[SP].notna().any(axis=1).sum()):,}")
        anyrow = late[late[SP].notna().any(axis=1)]
        if len(anyrow):
            print("   the first rows after the off with an SP field:")
            print(anyrow[["polled_utc", "source", "market_id", "selection_id", "status", "inplay", "bsp_reconciled",
                          "is_delayed"] + SP].head(8).to_string(index=False))
