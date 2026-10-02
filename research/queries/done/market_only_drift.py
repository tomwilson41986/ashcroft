"""The morning price against the Betfair SP, grouped by what is known in the morning (the owner, 2 Oct).

The first look at the archive (research/queries/done/price_archive_inventory.py) grouped the drift by the BSP,
which is known only at the off: a horse that ends short is a horse that shortened, so that table flatters the
favourites. A market-only rule (no model of our own: the only kind other countries' racing allows us) can read the
morning only, so here each runner is placed by its morning price (MORNINGWAP) and its morning volume, and scored on

- the CLV a back at the morning price meets: morning price / BSP - 1;
- the result of that back laid at the SP for its winnings (lost: -1 + (P - 1) / (BSP - 1) a pound; won: 0), the
  trade the live rule makes, before commission;
- the share that shortened.

UK and Irish win and place from 2023 (the archive holds them from 2017), Australia's and the greyhounds' last days
(their backfill runs from tonight). The morning price is an average of what traded in the morning, not a price one
could take at will, and the morning volume is small: this sizes the drift a rule would try to catch, it is not a
rule. Development window only for the UK/IE years scored (before 1 Apr 2026); the holdout is not read. Read-only.
"""

from __future__ import annotations

import io
import os
import sys
from collections import defaultdict

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import betfair_prices as bp  # noqa: E402

bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
pd.set_option("display.width", 220)
BANDS = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 51.0, 1001.0]
FROM, TO = "2023-01-01", "2026-03-31"            # UK/IE: the development window, not the holdout
RECENT = ("auswin", "ausplace", "greyhoundwin", "greyhoundplace")

files = defaultdict(list)
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{bp.S3_PREFIX}/"):
    for o in page.get("Contents", []):
        got = bp.parse_listed_name(o["Key"].rsplit("/", 1)[-1])
        if got:
            files[got[0]].append((got[1], o["Key"]))


def read(key: str) -> pd.DataFrame:
    raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    d = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    d.columns = [c.strip().upper() for c in d.columns]
    out = pd.DataFrame({"event": d.get("EVENT_ID"), "sid": d.get("SELECTION_ID")})
    for src, dst in (("BSP", "bsp"), ("MORNINGWAP", "mwap"), ("MORNINGTRADEDVOL", "mvol"), ("WIN_LOSE", "won"),
                     ("PPWAP", "ppwap")):
        out[dst] = pd.to_numeric(d.get(src), errors="coerce")
    return out


def score(d: pd.DataFrame, label: str) -> None:
    d = d[(d.bsp > 1) & (d.mwap > 1) & d.won.isin([0, 1])].copy()
    if len(d) < 200:
        print(f"\n=== {label}: {len(d)} runners with a morning price, a BSP and a result; too few")
        return
    d["clv"] = d.mwap / d.bsp - 1
    d["hedged"] = np.where(d.won == 1, 0.0, -1.0 + (d.mwap - 1) / (d.bsp - 1))
    d["rank"] = d.groupby("event").mwap.rank(method="first")
    print(f"\n=== {label}: {len(d):,} runners in {d.event.nunique():,} markets")
    for name, key in (("by morning price", pd.cut(d.mwap, BANDS)),
                      ("by place in the morning market", d["rank"].clip(upper=6).astype(int).astype(str)
                       .replace({"6": "6+"})),
                      ("by morning volume", pd.cut(d.mvol, [-1, 20, 100, 500, 2000, 1e12],
                                                   labels=["<20", "20-100", "100-500", "500-2k", "2k+"]))):
        g = d.groupby(key, observed=True).agg(runners=("clv", "size"), clv_mean=("clv", "mean"),
                                              clv_median=("clv", "median"), shortened=("bsp", lambda b: 0.0),
                                              hedged_per_pound=("hedged", "mean"))
        g["shortened"] = d.groupby(key, observed=True).apply(lambda x: (x.bsp < x.mwap).mean(), include_groups=False)
        print(f"  {name}:")
        print("    " + g.round(4).to_string().replace("\n", "\n    "))


for market in ("ukwin", "irewin", "ukplace", "ireplace"):
    keys = [k for day, k in sorted(files.get(market, [])) if FROM <= day.isoformat() <= TO]
    frames = []
    for i, k in enumerate(keys):
        try:
            frames.append(read(k))
        except Exception as exc:
            print(f"  {k}: not read ({exc})")
    print(f"\n{market}: {len(frames)} day files read ({FROM} to {TO})")
    if frames:
        score(pd.concat(frames, ignore_index=True), f"{market}, {FROM[:4]}-{TO[:7]}")
for market in RECENT:
    keys = [k for _, k in sorted(files.get(market, []))]
    frames = [read(k) for k in keys]
    if frames:
        score(pd.concat(frames, ignore_index=True), f"{market}, the {len(frames)} days archived so far")
