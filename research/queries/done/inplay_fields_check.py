"""What Betfair's in-running columns (IPMIN, IPMAX) hold, checked against the result (read-only).

Scan 1 (edge_inplay_offsets, run 37068154175) read IPMIN as the shortest price traded in running and IPMAX as the
longest. Its match rates were far below what the odds imply, and in the place markets near zero, though every
placed horse should trade near 1.01 in running. Before any in-play study, the columns are checked on the result:
for winners (placed horses, in the place markets) against losers, by market and year, IPMIN and IPMAX against the
SP, how often IPMIN is at the floor (1.01-1.05) and IPMAX at the ceiling (900+), and how often the in-running range
fails to straddle the SP. Development years only (to March 2026).
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import betfair_prices as bp  # noqa: E402

pd.set_option("display.width", 220)
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")

# two months a year is plenty for a field check
names = []
for name in bp.archive_index(s3, bucket):
    meta = bp.parse_file_name(name)
    if meta and date(2018, 1, 1) <= meta[2] <= date(2026, 3, 31) and meta[2].month in (3, 9):
        names.append(name)


def fetch(name):
    raw = s3.get_object(Bucket=bucket, Key=f"{bp.S3_PREFIX}/{name}")["Body"].read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    try:
        t = bp.parse_file_text(text, name)
    except Exception:
        return None
    return t[["country", "market_type", "race_date", "event_id", "event_name", "selection_name", "win_lose", "bsp",
              "ppwap", "ppmin", "ppmax", "ipmin", "ipmax", "ip_vol", "pp_vol"]]


with ThreadPoolExecutor(32) as ex:
    d = pd.concat([p for p in ex.map(fetch, names) if p is not None], ignore_index=True)
d = d[(d.bsp > 1)].copy()
d["year"] = d.race_date.str[:4]
d["won"] = d.win_lose == 1
print(f"{len(names):,} files (March and September), {len(d):,} runners with an SP")

print("\n== raw examples (UK win, 2025): a winner and two losers per race, first three races")
ex = d[(d.country == "uk") & (d.market_type == "win") & (d.year == "2025")]
for eid in ex.event_id.drop_duplicates().head(3):
    g = ex[ex.event_id == eid].sort_values("bsp")
    print(g[["event_name", "selection_name", "win_lose", "bsp", "ppwap", "ppmin", "ppmax", "ipmin", "ipmax", "ip_vol"]]
          .head(6).to_string(index=False))

print("\n== by market and outcome: medians and shares")


def describe(g):
    return pd.Series({
        "n": len(g), "ipmin_null": g.ipmin.isna().mean(), "ipmax_null": g.ipmax.isna().mean(),
        "ppmin<=bsp<=ppmax": ((g.ppmin <= g.bsp) & (g.bsp <= g.ppmax)).mean(),
        "ipmin_med": g.ipmin.median(), "ipmax_med": g.ipmax.median(),
        "ipmin/bsp_med": (g.ipmin / g.bsp).median(), "ipmax/bsp_med": (g.ipmax / g.bsp).median(),
        "ipmin<=1.05": (g.ipmin <= 1.05).mean(), "ipmax>=900": (g.ipmax >= 900).mean(),
        "ipmin>bsp": (g.ipmin > g.bsp).mean(), "ipmax<bsp": (g.ipmax < g.bsp).mean(),
        "ipmin==ipmax": (g.ipmin == g.ipmax).mean(), "no_ip_vol": (g.ip_vol.fillna(0) <= 0).mean(),
        "ip_vol_med": g.ip_vol.median()})


print(d.groupby(["country", "market_type", "won"]).apply(describe).round(3).T.to_string())
print("\n== by year (UK win and UK place): winners/placed at the floor and losers at the ceiling")
print(d[d.country == "uk"].groupby(["market_type", "year", "won"]).apply(
    lambda g: pd.Series({"n": len(g), "ipmin<=1.05": (g.ipmin <= 1.05).mean(), "ipmax>=900": (g.ipmax >= 900).mean(),
                         "ipmin/bsp_med": (g.ipmin / g.bsp).median(), "ipmax/bsp_med": (g.ipmax / g.bsp).median(),
                         "ip_vol_med": g.ip_vol.median()})).round(3).to_string())
