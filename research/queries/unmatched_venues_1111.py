"""After Bangor (Betfair "Bangor-on-Dee", the card "Bangor": never matched, never traded): is any other course lost
the same way? Read-only. By Betfair venue since 1 Oct 2026: WIN runners in betfair_live_markets and the share matched
to a race_results row (match_markets: track, time, horse); and the trader's ledger by day: venues with markets but no
ledger row."""
import sqlite3

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
m = pd.read_sql_query("""SELECT venue, country, race_date, market_id, race_results_id FROM betfair_live_markets
                         WHERE race_date >= '2026-10-01' AND UPPER(COALESCE(market_type, 'WIN')) = 'WIN'""", conn)
g = m.groupby(["country", "venue"]).agg(days=("race_date", "nunique"), markets=("market_id", "nunique"),
                                        runners=("market_id", "size"),
                                        matched=("race_results_id", lambda x: x.notna().mean())).round(3)
print(f"{len(m):,} WIN runners, {m.race_date.min()} to {m.race_date.max()}; results loaded to "
      f"{pd.read_sql_query('SELECT MAX(race_date) d FROM race_results', conn).d[0]}")
print("\nvenues matched under 80% (results not yet loaded for the last day or two lower every venue a little):")
print(g[g.matched < 0.8].sort_values("matched").to_string())
print("\nall venues:")
print(g.sort_values("matched").to_string())
try:
    lo = pd.read_sql_query("""SELECT race_date, market_id FROM live_orders WHERE race_date >= '2026-10-01'""", conn)
    traded = set(lo.market_id.astype(str))
    m["traded"] = m.market_id.astype(str).isin(traded)
    t = m.drop_duplicates("market_id").groupby("venue").agg(markets=("market_id", "size"), traded=("traded", "sum"))
    print("\nmarkets with a live order, by venue (venues with none):")
    print(t[t.traded == 0].to_string())
except Exception as e:  # noqa: BLE001
    print("live_orders:", type(e).__name__, e)
