"""The lays agent by time of day (the owner, 4 Nov: "how do the lays do by time of the day?"). Read-only. Its fills
on 12 Oct - 2 Nov (paper/horses/shared/agents/lays/<day>/fills.csv.gz, scored at the SP after the cuts): by the UK
hour of the fill and by minutes to the off, the lays, the stake and liability, the value locked at the SP (the
expected result), the result after the cuts (before the 2%: the commission is a market's, not an hour's), the days
up; and the real live backs' value by hour beside it."""
import gzip
import io
import os
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
pd.set_option("display.width", 250)
DAYS = [date(2026, 10, 12) + timedelta(i) for i in range(22)]


def get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001
        return None


rows = []
for d in DAYS:
    raw = get(f"paper/horses/shared/agents/lays/{d}/fills.csv.gz")
    if raw is None:
        continue
    f = pd.read_csv(io.BytesIO(gzip.decompress(raw)), low_memory=False, dtype={"market_id": str})
    f["day"] = str(d)
    rows.append(f)
f = pd.concat(rows, ignore_index=True)
print("columns:", list(f.columns))
for c in ("price", "matched", "pnl", "pnl_red", "locked", "clv"):
    if c in f:
        f[c] = pd.to_numeric(f[c], errors="coerce")
f = f[f.side.eq("lay") & (f.matched > 0)].copy()
t = pd.to_datetime(f.ts, utc=True, errors="coerce").dt.tz_convert("Europe/London")
f["hour"] = t.dt.hour
off = pd.to_datetime(f.day + " " + f.off.astype(str), errors="coerce").dt.tz_localize(
    "Europe/London", ambiguous="NaT", nonexistent="NaT")
f["mins"] = (off - t).dt.total_seconds() / 60
f["liability"] = f.matched * (f.price - 1)
f["to_off"] = pd.cut(f.mins, [-1, 30, 60, 120, 240, 480, 2000],
                     labels=["<30m", "30-60m", "1-2h", "2-4h", "4-8h", "8h+"])


def table(by):
    g = f.groupby(by, observed=True)
    out = g.agg(lays=("matched", "size"), stake=("matched", "sum"), liability=("liability", "sum"),
                locked=("locked", "sum"), result=("pnl_red", "sum"))
    out["value_pct"] = out.locked / out.liability
    out["result_0.80"] = out.result * 0.80
    days = f.groupby(["day", by], observed=True).pnl_red.sum().unstack(by)
    out["days"] = days.notna().sum()
    out["days_up"] = (days > 0).sum()
    return out.round(3)


print(f"== the lays agent, {f.day.nunique()} days, {len(f)} lays: result after the cuts GBP{f.pnl_red.sum():+,.2f} "
      f"(before the 2%), locked at the SP GBP{f.locked.sum():+,.2f}")
print("\n-- by UK hour of the lay")
print(table("hour").to_string())
print("\n-- by time to the off")
print(table("to_off").to_string())
print("\n-- by price")
f["band"] = pd.cut(f.price, [1, 3, 5, 8, 12, 21], labels=["<3", "3-5", "5-8", "8-12", "12-20"], right=False)
print(table("band").to_string())
