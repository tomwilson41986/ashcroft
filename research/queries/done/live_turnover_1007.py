"""Today's stakes so far against the owner's GBP4,000 day limit (7 Oct; read-only).

The owner chose GBP400 to win with the GBP4,000 day limit back (7 Oct, late morning). Today's session (run
37590881649, from 08:00 UTC) trades at GBP250 with no day limit; a restart takes up today's ledger, so everything
matched today counts against the GBP4,000. Reported from the ledger S3 holds (copied whenever it grows): backs sent,
matched and their stakes, by UK hour, the last back, and the money a restart today would still have under the limit.
"""

from __future__ import annotations

import io
import os
from collections import Counter

import boto3
import pandas as pd

bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
DAY, LIMIT = "2026-10-07", 4000.0
pd.set_option("display.width", 220)

objs = s3.list_objects_v2(Bucket=bucket, Prefix=f"trading/live/{DAY}/").get("Contents", [])
print("S3", [(o["Key"].split("/")[-1], o["Size"], o["LastModified"].strftime("%H:%M UTC")) for o in objs] or "nothing")
led = pd.read_csv(io.BytesIO(s3.get_object(Bucket=bucket, Key=f"trading/live/{DAY}/ledger.csv")["Body"].read()),
                  low_memory=False)
t = pd.to_datetime(led["ts"], utc=True, errors="coerce")
led["uk"] = t.dt.tz_convert("Europe/London")
print(f"ledger: {len(led)} rows to {led['uk'].max():%H:%M} UK; events {dict(Counter(led['event'].astype(str)))}")
backs = led[led["event"].astype(str).eq("back")].copy()
backs["m"] = pd.to_numeric(backs["matched"], errors="coerce").fillna(0)
got = backs[backs["m"] > 0]
print(f"backs sent {len(backs)}, matched {len(got)} on {got['selection_id'].nunique() if 'selection_id' in got else '?'} "
      f"horses for GBP{got['m'].sum():,.2f}; last matched back {got['uk'].max():%H:%M} UK")
print("matched by UK hour (GBP):", {f"{h:02d}": round(v, 2) for h, v in got.groupby(got["uk"].dt.hour)["m"].sum().items()})
cum = got.sort_values("uk")["m"].cumsum()
hit = got.sort_values("uk").loc[cum >= LIMIT, "uk"]
print(f"the GBP4,000 limit {'would have been reached at ' + format(hit.iloc[0], '%H:%M') + ' UK' if len(hit) else 'not reached'}")
print(f"left under GBP4,000 for a restart today: GBP{max(0.0, LIMIT - got['m'].sum()):,.2f}")
bad = led[led["status"].astype(str).eq("FAILURE")]
print("failures:", dict(Counter(bad["error"].astype(str).str[:70])) or "none")
