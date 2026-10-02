"""What the price archive holds, market by market, and a first market-only look at each (the owner, 2 Oct).

The owner asked what trading ideas the price files open beyond UK and Irish win markets: the place markets, AvB
(matchup) markets, other countries' racing. Betfair's files are archived from the UK server to
s3://$PRED_BUCKET/betfair_prices_raw/ (betfair_prices.py --archive), the last week of every market first, then the
backfill. This lists the archive by market (files, days, size), reads the most recent files of each market, and
for each prints what a trade there would rest on:

- the races and runners a day, how many carry a BSP, the menu hints (which countries and courses);
- what trades on a runner in the morning and before the off (the liquidity a stake would need);
- the morning price against the BSP by price band: the drift a backer at the morning price meets (mean of
  morning price / BSP - 1), and how often a runner shortens;
- the BSP against the result by band (the favourite-longshot shape a market-only rule would rest on), where the
  file carries the result (WIN_LOSE).

A few days of files show the shape, not an edge: no rule is chosen here. Read-only.
"""

from __future__ import annotations

import io
import os
import sys
from collections import Counter, defaultdict

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import betfair_prices as bp  # noqa: E402

bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
RECENT_FILES = 7                                   # the newest files of each market read in full
BANDS = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 51.0, 1001.0]

keys = []
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{bp.S3_PREFIX}/"):
    keys += [(o["Key"], o["Size"]) for o in page.get("Contents", [])]
by_market = defaultdict(list)
odd = []
for k, size in keys:
    got = bp.parse_listed_name(k.rsplit("/", 1)[-1])
    if got is None:
        odd.append(k)
        continue
    by_market[got[0]].append((got[1], k, size))
print(f"s3://{bucket}/{bp.S3_PREFIX}/: {len(keys):,} files in {len(by_market)} markets"
      + (f"; {len(odd)} names not read as a market and day: {odd[:5]}" if odd else ""))
inv = pd.DataFrame([{"market": m, "files": len(v), "first_day": min(d for d, _, _ in v),
                     "last_day": max(d for d, _, _ in v), "MB": round(sum(s for _, _, s in v) / 1e6, 1)}
                    for m, v in by_market.items()]).sort_values("files", ascending=False)
print(inv.to_string(index=False))


def read(key: str) -> pd.DataFrame | None:
    raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    try:
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except Exception as exc:
        print(f"    {key}: not read ({exc})")
        return None
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def num(s):
    return pd.to_numeric(s, errors="coerce")


for market in inv["market"]:
    files = sorted(by_market[market], reverse=True)[:RECENT_FILES]
    frames = []
    for day, key, _ in files:
        df = read(key)
        if df is not None and len(df):
            df["DAY"] = day.isoformat()
            frames.append(df)
    print(f"\n=== {market}: the newest {len(files)} files ({files[-1][0]} to {files[0][0]})")
    if not frames:
        print("    nothing readable")
        continue
    d = pd.concat(frames, ignore_index=True)
    cols = list(d.columns)
    print(f"    columns: {', '.join(c for c in cols if c != 'DAY')}")
    if "EVENT_ID" not in d or "BSP" not in d:
        print(f"    {len(d):,} rows; not Betfair's usual price-file layout: first row {d.iloc[0].to_dict()}")
        continue
    d["bsp"], d["mwap"], d["ppwap"] = num(d["BSP"]), num(d.get("MORNINGWAP")), num(d.get("PPWAP"))
    d["mvol"], d["ppvol"] = num(d.get("MORNINGTRADEDVOL")), num(d.get("PPTRADEDVOL"))
    d["won"] = num(d.get("WIN_LOSE"))
    races = d.groupby("DAY")["EVENT_ID"].nunique()
    hints = Counter(str(h).split("/", 1)[0].strip() for h in d.get("MENU_HINT", pd.Series(dtype=str)))
    print(f"    {len(d):,} runners in {d['EVENT_ID'].nunique():,} markets ({races.mean():.0f} a day); BSP on "
          f"{(d.bsp > 1).mean():.0%}; morning price on {(d.mwap > 1).mean():.0%}; countries/menus: "
          f"{dict(hints.most_common(6))}")
    names = d.get("EVENT_NAME", pd.Series(dtype=str)).astype(str)
    print(f"    market names, e.g.: {list(dict.fromkeys(names))[:6]}")
    print(f"    traded a runner: morning median GBP{d.mvol.median():,.0f} (90th GBP{d.mvol.quantile(0.9):,.0f}), "
          f"pre-play median GBP{d.ppvol.median():,.0f} (90th GBP{d.ppvol.quantile(0.9):,.0f})")
    ok = (d.bsp > 1) & (d.mwap > 1)
    if ok.sum() >= 50:
        x = d[ok].copy()
        x["band"] = pd.cut(x.bsp, BANDS)
        t = x.groupby("band", observed=True).agg(runners=("bsp", "size"), morning_over_bsp=("mwap", "median"),
                                                  bsp=("bsp", "median"))
        t["drift_mean"] = x.groupby("band", observed=True).apply(lambda g: (g.mwap / g.bsp - 1).mean(), include_groups=False)
        t["shortened"] = x.groupby("band", observed=True).apply(lambda g: (g.bsp < g.mwap).mean(), include_groups=False)
        t = t.drop(columns="morning_over_bsp")
        print("    the morning price against the BSP, by BSP band (drift_mean > 0: the morning price was the bigger):")
        print(t.round(3).to_string().replace("\n", "\n      "))
    res = (d.bsp > 1) & d.won.isin([0, 1])
    if res.sum() >= 50:
        x = d[res].copy()
        x["band"] = pd.cut(x.bsp, BANDS)
        g = x.groupby("band", observed=True).agg(runners=("won", "size"), won=("won", "mean"),
                                                 implied=("bsp", lambda b: (1 / b).mean()))
        g["back_at_bsp_roi_before_comm"] = x.groupby("band", observed=True).apply(
            lambda r: (np.where(r.won == 1, r.bsp - 1, -1.0)).mean(), include_groups=False)
        print("    the BSP against the result, by band (a few days: the shape only):")
        print(g.round(3).to_string().replace("\n", "\n      "))
