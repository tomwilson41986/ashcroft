"""Ghost runs, the first step (reports/discovery_1111.md 4.1; read-only; a count).

A horse declared, traded on Betfair and then withdrawn leaves a price and no result row. Before any feature is built:
how many are there, and how many runners have one since their last run? The report's own stop: under 2% of 2024-26
runners touched and the idea ends here.

A ghost is read from Betfair's price files as loaded (betfair_prices, WIN): a row with money traded before the off
(pp_vol or morning_vol above zero, a pre-play WAP above 1), no BSP, and no result row for that horse on that day.
This is the table as held, not the minute files: it gives the traded price, not the horse's share of the book at the
moment it was taken out.
"""
import sqlite3

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
con = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
bp = pd.read_sql_query(
    "SELECT race_date, horse_norm, country, bsp, ppwap, morningwap, pp_vol, morning_vol, race_results_id "
    "FROM betfair_prices WHERE lower(market_type) = 'win' AND race_date >= '2023-01-01'", con)
for c in ("bsp", "ppwap", "morningwap", "pp_vol", "morning_vol"):
    bp[c] = pd.to_numeric(bp[c], errors="coerce")
bp["ran"] = bp.race_results_id.notna() | (bp.bsp > 1)
vol = bp.pp_vol.fillna(0) + bp.morning_vol.fillna(0)
bp["ghost"] = ~bp.ran & (vol > 0) & ((bp.ppwap > 1) | (bp.morningwap > 1))
print(f"price-file WIN rows from 2023: {len(bp):,}; ran {int(bp.ran.sum()):,}; did not run {int((~bp.ran).sum()):,}; "
      f"ghosts (traded, no BSP, no result) {int(bp.ghost.sum()):,}")
g = bp[bp.ghost].copy()
g["price"] = g.ppwap.where(g.ppwap > 1, g.morningwap)
g["vol"] = vol[bp.ghost]
print("ghosts by year and country:")
print(g.assign(year=g.race_date.str[:4]).groupby(["year", "country"]).size().unstack(fill_value=0).to_string())
print(f"their traded price: median {g.price.median():.1f}, 25th-75th pct {g.price.quantile(.25):.1f}-{g.price.quantile(.75):.1f}; "
      f"money traded: median GBP{g.vol.median():,.0f}, 25th pct GBP{g.vol.quantile(.25):,.0f}, 75th pct GBP{g.vol.quantile(.75):,.0f}")

rr = pd.read_sql_query("SELECT race_date, horse_name FROM race_results WHERE race_date >= '2023-01-01'", con)
import sys  # noqa: E402
sys.path.insert(0, ".")
from betfair_prices import normalise_horse  # noqa: E402
rr["horse_norm"] = rr.horse_name.map(normalise_horse)
rr["d"] = pd.to_datetime(rr.race_date.str[:10])
rr = rr.sort_values(["horse_norm", "d"])
rr["prev"] = rr.groupby("horse_norm").d.shift()
g["d"] = pd.to_datetime(g.race_date.str[:10])
gl = g.sort_values("d")[["horse_norm", "d", "price"]].rename(columns={"d": "gd"})
if g.empty:
    raise SystemExit("no data found: no ghost rows in the price table")
rr["horse_norm"], gl["horse_norm"] = rr.horse_norm.astype(object), gl.horse_norm.astype(object)
m = pd.merge_asof(rr.sort_values("d"), gl, left_on="d", right_on="gd", by="horse_norm", allow_exact_matches=False)
m["since_run"] = m.gd.notna() & (m.prev.isna() | (m.gd > m.prev))
m["in365"] = m.gd.notna() & ((m.d - m.gd).dt.days <= 365)
m["first_run"] = m.prev.isna()
s = m[m.d >= "2024-01-01"]
print(f"\nrunners 2024 on: {len(s):,}; with a ghost since their last run {int(s.since_run.sum()):,} ({s.since_run.mean():.2%}); "
      f"with a ghost in the last 365 days {int(s.in365.sum()):,} ({s.in365.mean():.2%})")
print(f"  of first runs in the table from 2023: {int(s.first_run.sum()):,}, with a ghost before {s[s.first_run].since_run.mean():.2%}")
print("  by year:", s.assign(y=s.d.dt.year).groupby("y").since_run.mean().round(4).to_dict())
days = (s[s.since_run].d - s[s.since_run].gd).dt.days
print(f"  days from the ghost to the run: median {days.median():.0f}, 25th-75th pct {days.quantile(.25):.0f}-{days.quantile(.75):.0f}")
print(f"the report's stop: under 2% of runners touched -> {'STOP' if s.since_run.mean() < 0.02 else 'go on to the block'}")
