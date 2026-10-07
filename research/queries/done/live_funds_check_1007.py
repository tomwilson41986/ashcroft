"""Today's backs against the account (the owner, 7 Oct: "GBP3,578 can't be staked today. I didn't have that much").

The GBP3,578.48 was the sum of the amounts Betfair reported matched on today's backs in the ledger to 11:22 UK
(research/queries/done/live_turnover_1007.py). Checked here from the ledger S3 holds: every back's Betfair bet id
(is any bet counted twice), the backs by session, the six "sync" rows (positions the 08:00 UTC restart corrected from
Betfair's own orders), every refusal and hold for funds with its time, and the money open (backs on races not yet
run, back until about 10 minutes after the off) at each refusal, at the last back and now. Read-only.
"""

from __future__ import annotations

import io
import os
import sys
from collections import Counter

import boto3
import pandas as pd

sys.path.insert(0, os.getcwd())
from trading.session import HELD  # noqa: E402

bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
DAY = "2026-10-07"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 90)

obj = s3.get_object(Bucket=bucket, Key=f"trading/live/{DAY}/ledger.csv")
print("ledger last written", obj["LastModified"].strftime("%H:%M UTC"))
led = pd.read_csv(io.BytesIO(obj["Body"].read()), low_memory=False, dtype={"market_id": str, "bet_id": str})
led["t"] = pd.to_datetime(led["ts"], utc=True, errors="coerce")
led["uk"] = led["t"].dt.tz_convert("Europe/London")
led["off_t"] = pd.to_datetime(led["off"], utc=True, errors="coerce")
led["m"] = pd.to_numeric(led["matched"], errors="coerce").fillna(0.0)
print(f"{len(led)} rows, {led['uk'].min():%H:%M}-{led['uk'].max():%H:%M} UK; events {dict(Counter(led['event']))}")

b = led[(led["event"] == "back") & (led["m"] > 0)].copy()
print(f"\nmatched backs: {len(b)}, GBP{b['m'].sum():,.2f}; bet ids {b['bet_id'].nunique()} distinct, "
      f"{int(b['bet_id'].duplicated().sum())} repeated (GBP{b.loc[b['bet_id'].duplicated(), 'm'].sum():,.2f})")
print(f"  once per bet id: GBP{b.drop_duplicates('bet_id')['m'].sum():,.2f} on "
      f"{b.groupby(['market_id', 'selection_id']).ngroups} horses in {b['market_id'].nunique()} races")
restart = pd.Timestamp(f"{DAY} 08:00", tz="UTC")
for name, part in (("07:06-08:00 UTC session", b[b["t"] < restart]), ("from the 08:00 UTC restart", b[b["t"] >= restart])):
    print(f"  {name}: {len(part)} backs, GBP{part['m'].sum():,.2f}, {part['uk'].min():%H:%M}-{part['uk'].max():%H:%M} UK")
per = b.groupby(["market_id", "selection_id"])["m"].agg(["size", "sum"])
print(f"  backs per horse: {dict(Counter(per['size']))}; the most on one horse GBP{per['sum'].max():,.2f}")

print("\nthe sync rows (positions the restart took from Betfair's own orders):")
sy = led[led["event"] == "sync"]
print(sy[["uk", "market_id", "selection_id", "runner", "matched", "avg_price", "hedge_liability", "error"]].to_string())

fail = led[led["status"].astype(str).eq("FAILURE")]
print("\nrefusals:", dict(Counter(fail["error"].astype(str).str[:60])))
print(fail[["uk", "event", "runner", "asked", "error"]].to_string())
held = led[(led["event"] == "skip") & (led["error"].astype(str) == HELD)]
print(f"\nholds for funds: {len(held)} rows, {held['uk'].min() if len(held) else '-'} to {held['uk'].max() if len(held) else '-'}")
print("skips by reason:", dict(Counter(led.loc[led["event"] == "skip", "error"].astype(str).str[:60]).most_common(10)))


def open_at(t: pd.Timestamp) -> float:
    """Backs matched by t on races not yet run (a race's stakes come back about 10 minutes after its off)."""
    x = b[(b["t"] <= t) & (b["off_t"] + pd.Timedelta(minutes=10) > t)]
    return round(float(x["m"].sum()), 2)


offs = b.drop_duplicates("market_id")["off_t"].sort_values()
print(f"\nthe races backed run from {offs.min().tz_convert('Europe/London'):%H:%M} to "
      f"{offs.max().tz_convert('Europe/London'):%H:%M} UK")
for label, t in [("first refusal for funds", fail.loc[fail["error"].astype(str).str.contains("INSUFFICIENT"), "t"].min()),
                 ("last matched back", b["t"].max()), ("the ledger's last row", led["t"].max())]:
    if pd.notna(t):
        print(f"  money open at the {label} ({t.tz_convert('Europe/London'):%H:%M} UK): GBP{open_at(t):,.2f}; "
              f"races already run {int((offs + pd.Timedelta(minutes=10) <= t).sum())}")
lays = led[led["event"] == "trade_out"]
print(f"\nthe lays at the SP: {len(lays)} rows, liability asked GBP{pd.to_numeric(lays['asked'], errors='coerce').sum():,.2f}; "
      f"statuses {dict(Counter(lays['status'].astype(str)))}")
