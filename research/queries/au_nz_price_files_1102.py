"""Australia and New Zealand in our own Betfair price-file archive (the owner's ask, 2 Nov 2026; read-only).

The free first step of reports/international_racing.md s.6 (step 1): every auswin/ausplace file (and any NZ market)
archived at s3://$PRED_BUCKET/betfair_prices_raw/ from the UK server, read without loading into horse_racing.db
(betfair_prices.COUNTRIES stays UK/IE; nothing in the live path changes). Prints:

  1. the inventory: files a market and year, first and last day;
  2. data checks: rows read, duplicates (EVENT_ID, SELECTION_ID), nulls in key fields, rows without a BSP, date range;
  3. the code split: thoroughbred against harness (EVENT_NAME 'Pace'/'Trot'), and the country split by MENU_HINT
     (whether NZ meetings sit in the AU files at all): races, race days, races a day, by year;
  4. money: pre-play money a runner and a market (median, p90), by year, code and country (currency as Betfair writes
     it: these files carry no unit);
  5. the pre-play average price (PPWAP) against the BSP by PPWAP band and year: median log(BSP / PPWAP), share
     shortened (thoroughbreds only);
  6. a level back at the BSP by BSP band and year, before and after 2% commission on winnings (thoroughbreds only).

MORNINGWAP in an AU file is the UK morning, which is the Australian evening after racing: its coverage is printed,
nothing is read from it.
"""
from __future__ import annotations

import io
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import betfair_prices as bp  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)

BUCKET = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
BANDS = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 51.0, 1001.0]
COMMISSION = 0.02
KEEP = ["EVENT_ID", "MENU_HINT", "EVENT_NAME", "EVENT_DT", "SELECTION_ID", "SELECTION_NAME", "WIN_LOSE", "BSP",
        "PPWAP", "MORNINGWAP", "PPMAX", "PPMIN", "MORNINGTRADEDVOL", "PPTRADEDVOL", "IPTRADEDVOL"]

# ---------------------------------------------------------------- 1. inventory
keys = []
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"{bp.S3_PREFIX}/"):
    keys += [o["Key"] for o in page.get("Contents", [])]
by_market = defaultdict(list)
for k in keys:
    got = bp.parse_listed_name(k.rsplit("/", 1)[-1])
    if got:
        by_market[got[0]].append((got[1], k))
print(f"== 1. s3://{BUCKET}/{bp.S3_PREFIX}/: {len(keys):,} objects, markets: {sorted(by_market)}")
wanted = sorted(m for m in by_market if m.startswith("aus") or "nz" in m)
for m in wanted:
    v = by_market[m]
    years = pd.Series([d.year for d, _ in v]).value_counts().sort_index()
    print(f"  {m}: {len(v):,} files, {min(d for d, _ in v)} to {max(d for d, _ in v)}; by year {years.to_dict()}")
if "auswin" not in by_market:
    print("no auswin files in the archive: nothing more to read")
    sys.exit(0)


def read(item):
    day, key = item
    raw = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    try:
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except Exception as exc:                                  # an empty or malformed file is counted, not fatal
        return day, key, None, str(exc)[:80]
    df.columns = [c.strip().upper() for c in df.columns]
    missing = [c for c in KEEP if c not in df.columns]
    if missing:
        return day, key, None, f"missing {missing}"
    df = df[KEEP].copy()
    df["FILE_DAY"] = day.isoformat()
    return day, key, df, None


files = sorted(by_market["auswin"])
with ThreadPoolExecutor(max_workers=24) as ex:
    got = list(ex.map(read, files))
bad = [(k, e) for _, k, df, e in got if df is None]
frames = [df for _, _, df, _ in got if df is not None and len(df)]
empty = sum(1 for _, _, df, _ in got if df is not None and not len(df))
d = pd.concat(frames, ignore_index=True)
print(f"\n== 2. data checks (auswin): {len(files):,} files read, {len(bad)} unreadable, {empty} empty")
for k, e in bad[:10]:
    print(f"  unreadable: {k}: {e}")
rows_in = len(d)
for c in ["BSP", "PPWAP", "MORNINGWAP", "PPMAX", "PPMIN", "MORNINGTRADEDVOL", "PPTRADEDVOL", "IPTRADEDVOL"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
d["WIN_LOSE"] = pd.to_numeric(d["WIN_LOSE"], errors="coerce")
d["EVENT_ID"] = pd.to_numeric(d["EVENT_ID"], errors="coerce")
d["SELECTION_ID"] = pd.to_numeric(d["SELECTION_ID"], errors="coerce")
dt = d["EVENT_DT"].map(bp.bf_event_time)
d["race_date"] = [x for x, _ in dt]
d["race_date"] = d["race_date"].fillna(d["FILE_DAY"])
d["year"] = d["race_date"].str[:4]
dups = d.duplicated(["EVENT_ID", "SELECTION_ID"]).sum()
print(f"  rows {rows_in:,}; duplicate (EVENT_ID, SELECTION_ID) {dups:,} (dropped, first kept)")
d = d.drop_duplicates(["EVENT_ID", "SELECTION_ID"])
for c in ["EVENT_ID", "SELECTION_ID", "MENU_HINT", "EVENT_NAME", "EVENT_DT"]:
    n = (d[c].isna() | (d[c].astype(str).str.strip() == "")).sum()
    print(f"  null/blank {c}: {n:,}")
print(f"  no BSP (or BSP <= 1): {(~(d.BSP > 1)).sum():,}; no result (WIN_LOSE not 0/1): {(~d.WIN_LOSE.isin([0, 1])).sum():,}")
print(f"  race dates {d.race_date.min()} to {d.race_date.max()}; morning price present on {(d.MORNINGWAP > 1).mean():.1%} "
      f"(the UK morning: not read)")

# ---------------------------------------------------------------- 3. country and code split
d["country"] = d["MENU_HINT"].astype(str).str.split("/", n=1).str[0].str.strip().str.upper()
name = d["EVENT_NAME"].astype(str)
d["code"] = np.where(name.str.contains(r"\b(?:Pace|Trot)\b", case=False, regex=True), "harness", "thoroughbred")
print("\n== 3. the split. MENU_HINT country prefixes (runners):")
print(d["country"].value_counts().head(15).to_string())
print("  event names, the first 3 of each code:")
for c, g in d.groupby("code"):
    print(f"    {c}: {list(dict.fromkeys(g.EVENT_NAME))[:3]}")
races = d.groupby(["year", "country", "code"]).agg(races=("EVENT_ID", "nunique"), days=("race_date", "nunique"),
                                                   runners=("SELECTION_ID", "size"))
races["races_a_day"] = (races.races / races.days).round(1)
print(races.to_string())

# ---------------------------------------------------------------- 4. money
t = d[d.code == "thoroughbred"]
mk = t.groupby(["year", "country", "EVENT_ID"]).PPTRADEDVOL.sum().reset_index()
money = t.groupby(["year", "country"]).agg(runner_pp_median=("PPTRADEDVOL", "median"),
                                           runner_pp_p90=("PPTRADEDVOL", lambda s: s.quantile(0.9)))
money["market_pp_median"] = mk.groupby(["year", "country"]).PPTRADEDVOL.median()
money["market_ip_median"] = t.groupby(["year", "country", "EVENT_ID"]).IPTRADEDVOL.sum().groupby(["year", "country"]).median()
print("\n== 4. thoroughbred pre-play money (the files' unit), by year and country:")
print(money.round(0).to_string())
h = d[d.code == "harness"]
if len(h):
    print("  harness, runner pre-play median by year:", h.groupby("year").PPTRADEDVOL.median().round(0).to_dict())

# ---------------------------------------------------------------- 5. PPWAP against the BSP
x = t[(t.BSP > 1) & (t.PPWAP > 1)].copy()
x["band"] = pd.cut(x.PPWAP, BANDS)
x["lr"] = np.log(x.BSP / x.PPWAP)
x["short"] = x.BSP < x.PPWAP
print("\n== 5. thoroughbreds: median log(BSP / PPWAP) by PPWAP band and year (+: the BSP bigger)")
print(x.pivot_table(index="band", columns="year", values="lr", aggfunc="median", observed=True).round(3).to_string())
print("  share that shortened (BSP < PPWAP):")
print(x.pivot_table(index="band", columns="year", values="short", aggfunc="mean", observed=True).round(2).to_string())
print("  runners:", x.groupby("year").size().to_dict())

# ---------------------------------------------------------------- 6. a level back at the BSP
r = t[(t.BSP > 1) & t.WIN_LOSE.isin([0, 1])].copy()
r["band"] = pd.cut(r.BSP, BANDS)
r["ret"] = np.where(r.WIN_LOSE == 1, r.BSP - 1, -1.0)
r["ret_net"] = np.where(r.WIN_LOSE == 1, (r.BSP - 1) * (1 - COMMISSION), -1.0)
print("\n== 6. thoroughbreds: a level back at the BSP, return a unit before commission, by BSP band and year")
print((100 * r.pivot_table(index="band", columns="year", values="ret", aggfunc="mean", observed=True)).round(1).to_string())
print("  after 2% commission on winnings:")
print((100 * r.pivot_table(index="band", columns="year", values="ret_net", aggfunc="mean", observed=True)).round(1).to_string())
print("  runners:", r.pivot_table(index="band", columns="year", values="ret", aggfunc="size", observed=True).to_string())
