"""Is the trader backing again after the 12:00 UTC restart? (7 Oct; read-only)

The owner restarted the trader at 12:00 UTC (run 37617919958, on the rule merged in PR #131: GBP400 to win, GBP300 a
bet, no day limit, to 15 minutes before each off) after 53 minutes without trading while the market recorder held the
UK runner (ledger runner-give-way-1007). From the ledger S3 holds (copied whenever it grows): the backs sent and
matched by UK half hour, the stake to win before and after the restart, the refusals, and the last back.
"""

from __future__ import annotations

import io
import os
from collections import Counter

import boto3
import pandas as pd

bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
DAY = "2026-10-07"
RESTART_UK = pd.Timestamp("2026-10-07 13:00", tz="Europe/London")
pd.set_option("display.width", 220)

objs = s3.list_objects_v2(Bucket=bucket, Prefix=f"trading/live/{DAY}/").get("Contents", [])
print("S3", [(o["Key"].split("/")[-1], o["Size"], o["LastModified"].strftime("%H:%M UTC")) for o in objs] or "nothing")
led = pd.read_csv(io.BytesIO(s3.get_object(Bucket=bucket, Key=f"trading/live/{DAY}/ledger.csv")["Body"].read()),
                  low_memory=False)
led["uk"] = pd.to_datetime(led["ts"], utc=True, errors="coerce").dt.tz_convert("Europe/London")
print(f"ledger: {len(led)} rows to {led['uk'].max():%H:%M} UK; events {dict(Counter(led['event'].astype(str)))}")

backs = led[led["event"].astype(str).eq("back")].copy()
for c in ("matched", "avg_price", "target", "asked"):
    backs[c] = pd.to_numeric(backs[c], errors="coerce")
backs["m"] = backs["matched"].fillna(0.0)
backs["slot"] = backs["uk"].dt.floor("30min").dt.strftime("%H:%M")
got = backs[backs["m"] > 0]
print("\nbacks by UK half hour (sent, matched, GBP matched)")
print(backs.groupby("slot").agg(sent=("event", "size"), matched=("m", lambda s: int((s > 0).sum())),
                                gbp=("m", "sum")).round(2).to_string())

print()
for name, part in (("before 13:00 UK", got[got["uk"] < RESTART_UK]), ("from 13:00 UK", got[got["uk"] >= RESTART_UK])):
    if not len(part):
        print(f"{name}: none matched")
        continue
    to_win = part.assign(w=part["m"] * (part["avg_price"] - 1.0)).groupby(["market_id", "selection_id"])["w"].sum()
    targets = sorted({round(float(t)) for t in part["target"].dropna()})
    print(f"{name}: {len(part)} backs matched for GBP{part['m'].sum():,.2f} on {len(to_win)} horses; to win per horse "
          f"median GBP{to_win.median():,.0f}, largest GBP{to_win.max():,.0f}; targets {targets[:6]}; "
          f"largest back GBP{part['m'].max():,.2f}")
print(f"last back sent {backs['uk'].max():%H:%M:%S} UK; last matched {got['uk'].max():%H:%M:%S} UK")
bad = led[led["status"].astype(str).eq("FAILURE")]
for name, part in (("before 13:00 UK", bad[bad["uk"] < RESTART_UK]), ("from 13:00 UK", bad[bad["uk"] >= RESTART_UK])):
    print(f"failures {name}:", dict(Counter(part["error"].astype(str).str[:70])) or "none")
