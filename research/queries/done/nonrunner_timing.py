"""When are the day's non-runners taken out, and what would they cut from a back taken the evening before? (read-only)

For the owner's evening-before question (4 Oct). A back matched before a non-runner is taken out is cut by the
non-runner's reduction factor (when it is 2.5% or more; the cuts add up to at most 90%), and the lay at SP that
hedges it is not. From Betfair's final books of each recorded day (betfair_live/<day>/books.csv.gz, source final:
each REMOVED runner's removal time and reduction factor), for each GB/IE win market: the non-runners taken out by
UK hour, and the cut a back would carry if matched at 19:00 the evening before, at 08:00 on the day (the trader's
start) and at 11:00. The difference between the first two is what an evening back pays in non-runners over a
morning one, before any price advantage.
"""

from __future__ import annotations

import gzip
import io
import os
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
UK = "Europe/London"
FIRST = date.fromisoformat(os.environ.get("NR_FROM") or "2026-10-01")
LAST = date.fromisoformat(os.environ["NR_TO"]) if os.environ.get("NR_TO") else date.today() - timedelta(days=1)
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")


def _csv(key, **kw):
    try:
        raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(gzip.decompress(raw) if key.endswith(".gz") else raw), low_memory=False, **kw)


rows, races = [], []
day = FIRST
while day <= LAST:
    b = _csv(f"betfair_live/{day}/books.csv.gz", dtype={"market_id": str})
    m = _csv(f"betfair_live/{day}/markets.csv.gz", dtype={"market_id": str})
    if b.empty or m.empty:
        print(f"{day}: no final books")
        day += timedelta(days=1)
        continue
    win = set(m.loc[m.market_type.eq("WIN"), "market_id"]) if "market_type" in m else set(m.market_id)
    f = b[(b.source == "final") & b.market_id.isin(win)].drop_duplicates(["market_id", "selection_id"], keep="last")
    f = f.assign(rf=pd.to_numeric(f.adjustment_factor, errors="coerce").fillna(0.0),
                 removed=pd.to_datetime(f.removal_utc, utc=True, errors="coerce"))
    rem = f[f.runner_status == "REMOVED"]
    offs = m.drop_duplicates("market_id").set_index("market_id").market_start_utc.map(lambda s: pd.Timestamp(s))
    eve = pd.Timestamp(day - timedelta(days=1), tz=UK) + pd.Timedelta(hours=19)
    at = {"19:00 the evening before": eve, "08:00": pd.Timestamp(day, tz=UK) + pd.Timedelta(hours=8),
          "11:00": pd.Timestamp(day, tz=UK) + pd.Timedelta(hours=11)}
    for mid in sorted(set(f.market_id)):
        r = rem[rem.market_id == mid]
        cut = {k: min(float(r.rf[(r.removed > t) & (r.rf >= 2.5)].sum()), 90.0) for k, t in at.items()}
        races.append({"day": str(day), "market_id": mid, "off": offs.get(mid), "runners": int((f.market_id == mid).sum()),
                      "non_runners": len(r), **{f"cut_{k}": v for k, v in cut.items()}})
    rem = rem.assign(day=str(day), hour_uk=rem.removed.dt.tz_convert(UK).dt.floor("h"), after_eve=rem.removed > eve)
    rows.append(rem)
    print(f"{day}: {len(set(f.market_id))} win markets, {len(rem)} non-runners ({int(rem.removed.isna().sum())} "
          f"without a time), reduction factors median {rem.rf.median():.1f}%")
    day += timedelta(days=1)

if not races:
    raise SystemExit("nothing to read")
R = pd.DataFrame(races)
N = pd.concat(rows) if rows else pd.DataFrame()
print("\n== non-runners by the UK hour they were taken out (and their reduction factors)")
if len(N):
    h = N.groupby("hour_uk").agg(n=("rf", "size"), rf_median=("rf", "median"),
                                 rf_2_5_plus=("rf", lambda v: int((v >= 2.5).sum())))
    h.index = h.index.strftime("%a %d %b %H:00")
    print(h.to_string())
cols = [c for c in R.columns if c.startswith("cut_")]
print(f"\n== the cut on a back, by when it was matched ({len(R)} races; the mean over races, % of the price)")
print(R[cols].describe(percentiles=[0.5, 0.9]).round(2).to_string())
print("\nshare of races where a back would be cut at all:")
print((R[cols] > 0).mean().round(3).to_string())
extra = R["cut_19:00 the evening before"] - R["cut_08:00"]
print(f"\nan evening back's extra cut over an 08:00 back: mean {extra.mean():.2f}% of the price, in "
      f"{(extra > 0).mean():.0%} of races (when cut: mean {extra[extra > 0].mean() if (extra > 0).any() else 0:.1f}%)")
print("by day:")
print(R.assign(extra=extra).groupby("day").agg(races=("extra", "size"), extra_mean=("extra", "mean"),
                                               cut_0800_mean=("cut_08:00", "mean")).round(2).to_string())
