"""What the place files' pre-play columns hold (PPWAP, PPMIN, PPMAX, MORNINGWAP), against the place SP (read-only).

The in-running check (research/queries/done/inplay_fields_check.py, run 37070286949) found the place files' IPMIN
and IPMAX holding placeholder values on most rows (IPMIN 1001, IPMAX 1.0), and their pre-play range holding the place
SP on only 23-41% of runners, against 95% in the win files. The place pocket's pre-play reading (Jan-Mar 2026, report
other_markets_and_stakes_1002.md section 2) read the place PPWAP. Before any place study reads these columns again,
they are checked here, in the win and place files alike: how often each holds a placeholder (1001 or more, 1.0 or
less, empty), how the pre-play WAP and the morning WAP sit against the SP, how often the pre-play range holds the
pre-play WAP and the SP, by market and by year. March and September files, 2018 to March 2026 (development years only).
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
pd.set_option("display.max_rows", 200)
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")

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
    t["file"] = name
    return t[["file", "country", "market_type", "race_date", "event_id", "event_name", "selection_name", "win_lose",
              "bsp", "ppwap", "ppmin", "ppmax", "morningwap", "ipmin", "ipmax", "morning_vol", "pp_vol", "ip_vol"]]


with ThreadPoolExecutor(32) as ex:
    d = pd.concat([p for p in ex.map(fetch, names) if p is not None], ignore_index=True)
d = d[d.bsp > 1].copy()
d["year"] = d.race_date.str[:4]
print(f"{len(names):,} files (March and September), {len(d):,} runners with an SP")

print("\n== raw examples (UK place, March 2025): one race")
ex = d[(d.country == "uk") & (d.market_type == "place") & (d.race_date.str[:7] == "2025-03")]
for eid in ex.event_id.drop_duplicates().head(2):
    print(ex[ex.event_id == eid].sort_values("bsp")[["event_name", "selection_name", "win_lose", "bsp", "ppwap", "ppmin",
                                                     "ppmax", "morningwap", "ipmin", "ipmax", "pp_vol", "ip_vol"]]
          .to_string(index=False))


def describe(g):
    r = g.ppwap / g.bsp
    return pd.Series({
        "n": len(g),
        "ppwap_empty": (g.ppwap.isna() | (g.ppwap <= 1)).mean(),
        "ppmin>=1000": (g.ppmin >= 1000).mean(), "ppmax<=1": (g.ppmax <= 1).mean(),
        "ipmin>=1000": (g.ipmin >= 1000).mean(), "ipmax<=1": (g.ipmax <= 1).mean(),
        "ppwap/bsp_p05": r.quantile(0.05), "ppwap/bsp_p25": r.quantile(0.25), "ppwap/bsp_p50": r.median(),
        "ppwap/bsp_p75": r.quantile(0.75), "ppwap/bsp_p95": r.quantile(0.95),
        "ppwap_within_20pc": ((r >= 1 / 1.2) & (r <= 1.2)).mean(),
        "ppmin<=ppwap<=ppmax": ((g.ppmin <= g.ppwap) & (g.ppwap <= g.ppmax)).mean(),
        "ppmin<=bsp<=ppmax": ((g.ppmin <= g.bsp) & (g.bsp <= g.ppmax)).mean(),
        "morning/bsp_p50": (g.morningwap / g.bsp).median(),
        "pp_vol_p50": g.pp_vol.median(), "pp_vol_zero": (g.pp_vol.fillna(0) <= 0).mean()})


print("\n== by market (every runner with an SP)")
print(d.groupby(["country", "market_type"]).apply(describe).round(3).T.to_string())
print("\n== traded pre-play only (pp_vol > 0)")
print(d[d.pp_vol > 0].groupby(["country", "market_type"]).apply(describe).round(3).T.to_string())
print("\n== by year, UK place and UK win")
cols = ["n", "ppwap_empty", "ppmin>=1000", "ppmax<=1", "ipmin>=1000", "ipmax<=1", "ppwap/bsp_p50",
        "ppwap_within_20pc", "ppmin<=bsp<=ppmax", "pp_vol_p50"]
print(d[d.country == "uk"].groupby(["market_type", "year"]).apply(describe)[cols].round(3).to_string())
print("\n== by file, UK place March 2026 (the place pocket's pre-play months): placeholders and the WAP against the SP")
m = d[(d.country == "uk") & (d.market_type == "place") & (d.race_date >= "2026-03-01")]
print(m.groupby("file").apply(describe)[cols].round(3).to_string())
