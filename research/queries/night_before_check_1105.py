"""The first night's overnight record (the owner's timing ask, 4 Nov; read-only). books_overnight for 5 Nov's racing
(betfair-prices.yml from 22:30 UTC on 4 Nov, every 10 minutes to 04:52 UTC): by UK hour, the polls, the markets, the
runners priced both sides, the spread (best lay / best back), the money at the best back, the market's matched; and
the same at 05:00-09:59 UK from the day's own books for comparison. The night-before predictions for 5 Nov do not
exist (predict-evening.yml at 21:50 UTC found the 48-hour declarations for 6 Nov; it runs at 00:10 UK from 6 Nov), so
the card comparison waits for 6 Nov."""
import io
import os

import boto3
import numpy as np
import pandas as pd

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
pd.set_option("display.width", 250)
DAY = "2026-11-05"
COLS = ("polled_utc", "market_id", "selection_id", "status", "inplay", "runner_status", "back1", "back1_size", "lay1",
        "lay1_size", "market_total_matched")


def get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001
        return None


for k in (f"predictions_evening/{DAY}.csv", f"predictions/{DAY}.csv"):
    print(f"{k}: {'present' if get(k) is not None else 'missing'}")


def table(name):
    raw = get(f"betfair_live/{DAY}/{name}.csv.gz")
    if raw is None:
        print(f"\n== {name}: not in S3")
        return None
    b = pd.read_csv(io.BytesIO(raw), compression="gzip", low_memory=False, dtype={"market_id": str},
                    usecols=lambda c: c in COLS)
    for c in ("back1", "back1_size", "lay1", "lay1_size", "market_total_matched"):
        b[c] = pd.to_numeric(b.get(c), errors="coerce")
    b["t"] = pd.to_datetime(b.polled_utc, utc=True, errors="coerce")
    b["uk_hour"] = b.t.dt.tz_convert("Europe/London").dt.strftime("%m-%d %H")
    act = b[b.runner_status.fillna("ACTIVE").eq("ACTIVE")]
    two = act[(act.back1 > 1) & (act.lay1 > 1)].copy()
    two["spread"] = two.lay1 / two.back1 - 1
    out = act.groupby("uk_hour").agg(polls=("polled_utc", "nunique"), markets=("market_id", "nunique"),
                                     runner_rows=("selection_id", "size"))
    out = out.join(two.groupby("uk_hour").agg(two_sided=("selection_id", "size"), spread_median=("spread", "median"),
                                              best_back_size_median=("back1_size", "median")))
    out["two_sided_share"] = out.two_sided / out.runner_rows
    mm = b.drop_duplicates(["uk_hour", "market_id"], keep="last").groupby("uk_hour").market_total_matched.median()
    out["market_matched_median"] = mm
    print(f"\n== {name}: {len(b):,} rows, {b.market_id.nunique()} markets, polls {b.polled_utc.nunique()}, "
          f"{b.t.min()} to {b.t.max()}")
    print(out.round(3).to_string())
    return out


table("books_overnight")
day = table("books")
