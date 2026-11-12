"""Ghost runs, the first step (reports/discovery_1111.md 4.1; read-only; a count).

A horse declared, traded on Betfair and then withdrawn leaves a price and no result row. Before any feature is built:
how many are there? The report's own stop: under 2% of runners touched and the idea ends here.

The first run (43167188633) found that Betfair's price files as loaded (betfair_prices) carry runners only: 467,000
WIN rows from 2023, every one with a BSP, no withdrawn horse. So the only held record of a withdrawn horse's price is
the live record from 1 Oct 2026 (betfair_live_marks: each runner's book at clock marks and before the off, its status
at each). Here: of the horses on Betfair's WIN markets since 1 Oct, how many were removed, how many of those had a
price (a last trade or a back) at a mark while they stood, at what price, and how many horses that ran in the period
had been removed earlier in it. Six weeks cannot give the 2024-26 share the report asks for; it gives the rate.
"""
import sqlite3

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
con = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
mk = pd.read_sql_query("SELECT race_date, market_id, selection_id, mark, runner_status, last_traded, back1, lay1, "
                       "minutes_to_off FROM betfair_live_marks", con)
cat = pd.read_sql_query("SELECT market_id, selection_id, market_type, country, horse_norm FROM betfair_live_markets", con)
cat = cat[cat.market_type.fillna("WIN").str.upper().eq("WIN")].drop_duplicates(["market_id", "selection_id"])
mk = mk.merge(cat[["market_id", "selection_id", "country", "horse_norm"]], on=["market_id", "selection_id"])
for c in ("last_traded", "back1", "lay1"):
    mk[c] = pd.to_numeric(mk[c], errors="coerce")
fin = mk[mk.mark == "final"].set_index(["market_id", "selection_id"]).runner_status
r = mk.drop_duplicates(["market_id", "selection_id"])[["race_date", "market_id", "selection_id", "country", "horse_norm"]].copy()
r["final"] = pd.MultiIndex.from_frame(r[["market_id", "selection_id"]]).map(fin)
live = mk[(mk.runner_status == "ACTIVE") & (mk.mark != "final")].copy()
live["px"] = np.where((live.back1 > 1) & (live.lay1 > 1), np.sqrt(live.back1 * live.lay1), live.last_traded)
last = live.sort_values("minutes_to_off").groupby(["market_id", "selection_id"]).agg(
    px=("px", "first"), traded=("last_traded", lambda x: bool((x > 1).any())), marks=("mark", "size"))
r = r.join(last, on=["market_id", "selection_id"])
r["removed"] = r.final.eq("REMOVED")
r["ran"] = r.final.isin(["WINNER", "LOSER", "PLACED"])
r["ghost"] = r.removed & r.traded.fillna(False).astype(bool)
print(f"live record {r.race_date.min()} to {r.race_date.max()}: {len(r):,} horses on WIN markets with a final mark "
      f"{int(r.final.notna().sum()):,}; ran {int(r.ran.sum()):,}; removed {int(r.removed.sum()):,} "
      f"({r.removed.sum() / max(r.final.notna().sum(), 1):.1%})")
print(f"removed with a trade recorded while they stood (ghosts): {int(r.ghost.sum()):,} "
      f"({r.ghost.sum() / max(r.removed.sum(), 1):.0%} of the removed); removed with no mark while active "
      f"{int((r.removed & r.marks.isna()).sum()):,}")
g = r[r.ghost]
print("ghosts by country:", g.country.value_counts().to_dict())
print(f"their last price while they stood: median {g.px.median():.1f}, 25th-75th pct {g.px.quantile(.25):.1f}-"
      f"{g.px.quantile(.75):.1f}; under 10.0: {(g.px < 10).mean():.0%}")
ran = r[r.ran].merge(g[["horse_norm", "race_date"]].rename(columns={"race_date": "gd"}), on="horse_norm", how="left")
ran["after"] = ran.gd.notna() & (ran.gd < ran.race_date)
touched = ran.groupby(["market_id", "selection_id"]).after.any()
print(f"horses that ran in the period: {len(touched):,}; removed with a trade earlier in the same period: "
      f"{int(touched.sum()):,} ({touched.mean():.2%}) - a floor: the record is six weeks, the report's test is on "
      "every ghost since the horse's last run")
by = r.assign(week=pd.to_datetime(r.race_date).dt.strftime("%G-%V")).groupby("week").agg(horses=("removed", "size"),
                                                                                          removed=("removed", "sum"), ghosts=("ghost", "sum"))
print(by.to_string())
