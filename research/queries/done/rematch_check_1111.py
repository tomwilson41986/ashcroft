"""Would the re-match (betfair_recorder.rematch_unmatched) fix the live record's unmatched runners? Read-only: the
function's own matching run on a copy in memory of the unmatched rows, nothing written. By venue: unmatched before,
matched now, still unmatched, and whether race_results holds that track on those days at all."""
import sqlite3
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, ".")
import betfair_recorder as br  # noqa: E402
from betfair_prices import normalise_track  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
last = pd.read_sql_query("SELECT MAX(race_date) d FROM race_results", conn).d[0]
allm = pd.read_sql_query("SELECT * FROM betfair_live_markets WHERE race_date >= '2026-10-01' AND race_date <= ? "
                         "AND UPPER(COALESCE(market_type, 'WIN')) = 'WIN'", conn, params=[last])
todo = allm[allm.race_results_id.isna()]
print(f"WIN runners {len(allm):,} to {last}; unmatched {len(todo):,} ({len(todo) / len(allm):.1%})")
out = []
for d, part in todo.groupby("race_date"):
    m = br.match_markets(part.drop(columns=["race_results_id"]), conn, date.fromisoformat(d))
    rr = pd.read_sql_query("SELECT DISTINCT track FROM race_results WHERE race_date = ?", conn, params=[d])
    have = set(rr.track.map(normalise_track))
    m["now"] = m.race_results_id.notna()
    m["results_held"] = m.venue.map(normalise_track).isin(have)
    m["status"] = part.set_index(["market_id", "selection_id"]).reindex(
        list(zip(m.market_id, m.selection_id))).get("runner_status", pd.Series(dtype=str)).values if "runner_status" in part else ""
    out.append(m)
r = pd.concat(out)
print(f"matched by the re-match: {int(r.now.sum()):,}; still unmatched {int((~r.now).sum()):,} "
      f"(of which the track has no results that day: {int((~r.now & ~r.results_held).sum()):,})")
g = r.groupby("venue").agg(unmatched=("now", "size"), matched_now=("now", "sum"),
                           left=("now", lambda x: int((~x).sum())),
                           left_with_results=("results_held", lambda x: int((x & ~r.loc[x.index, "now"]).sum())))
print(g.sort_values("left", ascending=False).to_string())
left = r[~r.now & r.results_held]
print("\nstill unmatched though the track's results are held (non-runners are expected here): sample")
print(left[["race_date", "venue", "market_start_utc", "runner_name"]].head(25).to_string(index=False))
after = 1 - (len(todo) - r.now.sum()) / len(allm)
print(f"\nmatched share: before {1 - len(todo) / len(allm):.1%}, after {after:.1%}")
