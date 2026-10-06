"""Is 6 Oct's restarted session working? (read-only; the owner's question at 13:00 UTC)

The trader's job failed at 09:47 UTC (a GitHub Actions internal error) and the owner dispatched it again at 12:55
UTC (run 37466865912, with the Tote recorder of PR #110 beside it). What S3 holds for the day: the trader's ledger
(copied whenever it grows) and summary, the Betfair record's day files, and the Tote record's day files.
"""

from __future__ import annotations

import gzip
import io
import json
import os
from collections import Counter

import boto3
import pandas as pd

bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
RESTART = pd.Timestamp("2026-10-06 12:55", tz="UTC")

for prefix in ("trading/live/2026-10-06/", "betfair_live/2026-10-06/", "tote/live/2026-10-06/"):
    objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])
    print(prefix, [(o["Key"].split("/")[-1], o["Size"], o["LastModified"].strftime("%H:%M")) for o in objs] or
          "nothing")

try:
    raw = s3.get_object(Bucket=bucket, Key="trading/live/2026-10-06/ledger.csv")["Body"].read()
    led = pd.read_csv(io.BytesIO(raw), low_memory=False)
    print(f"ledger: {len(led)} rows; columns {list(led.columns)[:30]}")
    tcol = next((c for c in ("placed_utc", "time_utc", "ts", "created_utc", "placed_at") if c in led.columns), None)
    if tcol:
        t = pd.to_datetime(led[tcol], utc=True, errors="coerce")
        print(f"  first {t.min()}, last {t.max()}; rows after the restart: {(t >= RESTART).sum()}")
        for k in ("side", "kind", "status", "order_status"):
            if k in led.columns:
                print(f"  by {k} (after the restart):", dict(Counter(led.loc[t >= RESTART, k].astype(str))))
    for k in ("matched", "size_matched", "stake", "matched_stake"):
        if k in led.columns:
            print(f"  {k}: total {pd.to_numeric(led[k], errors='coerce').sum():.2f}")
except Exception as e:
    print("ledger:", type(e).__name__, e)

for name in ("pools", "detail", "results"):
    try:
        raw = s3.get_object(Bucket=bucket, Key=f"tote/live/2026-10-06/{name}.jsonl.gz")["Body"].read()
        lines = [json.loads(x) for x in gzip.decompress(raw).decode().splitlines() if x.strip()]
        t = pd.to_datetime([x.get("polled_utc") for x in lines], utc=True, errors="coerce")
        print(f"tote {name}: {len(lines)} answers, {t.min()} to {t.max()}, statuses "
              f"{dict(Counter(str(x.get('status')) for x in lines))}")
        if name == "detail":
            print("  by pool and mark:", dict(Counter((x.get("pool"), x.get("mark")) for x in lines)))
        if name == "pools" and lines:
            pools = (lines[-1].get("answer") or {}).get("pools") or []
            print(f"  last poll: {len(pools)} pools, {sum(1 for p in pools if p.get('bettingOn'))} betting on")
    except Exception as e:
        print(f"tote {name}:", type(e).__name__, e)
