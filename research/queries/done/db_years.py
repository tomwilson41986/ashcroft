"""How far back the history goes: races, runners and prices by year, and Betfair's morning prices by month.

Training starts on 2021-01-01 (research_loop.START_DATE, train_bfsp). Whether earlier years exist,
and how complete they are (BSP, official ratings, comments, comptime, pedigree), decides whether a
longer history is worth a matrix build. The morning prices decide how long the early-price trade's
record can be. Read-only; the locked holdout is only counted, never scored.
"""

import sqlite3

import pandas as pd

conn = sqlite3.connect("horse_racing.db")
pd.set_option("display.width", 200)
cols = {r[1] for r in conn.execute("PRAGMA table_info(race_results)")}
fill = [c for c in ("bfsp", "official_rating", "comment", "comptime_numeric", "stallion", "dam_stallion", "stall",
                    "placing_numerical", "days_since_lr") if c in cols]
sel = ", ".join(f"AVG(CASE WHEN {c} IS NOT NULL AND TRIM(CAST({c} AS TEXT)) NOT IN ('', 'nan') THEN 1.0 ELSE 0 END) AS {c}"
                for c in fill)
q = f"""SELECT substr(race_date, 1, 4) AS year, COUNT(*) AS runners,
               COUNT(DISTINCT race_date || '|' || track || '|' || race_time) AS races,
               COUNT(DISTINCT race_date) AS days, {sel}
        FROM race_results GROUP BY year ORDER BY year"""
y = pd.read_sql_query(q, conn)
for c in fill:
    y[c] = (100 * y[c]).round(1)
print("race_results by year (share of runners with each field, %)")
print(y.to_string(index=False))

tabs = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
print("\ntables:", sorted(tabs))
if "betfair_prices" in tabs:
    b = pd.read_sql_query("""SELECT substr(r.race_date, 1, 7) AS month, COUNT(*) AS runners,
                                    SUM(CASE WHEN b.morningwap > 1 THEN 1 ELSE 0 END) AS with_morning,
                                    SUM(CASE WHEN b.morning_vol >= 100 THEN 1 ELSE 0 END) AS vol_100
                             FROM betfair_prices b JOIN race_results r ON r.id = b.race_results_id
                             WHERE b.market_type = 'win' GROUP BY month ORDER BY month""", conn)
    print("\nbetfair_prices (win) by month")
    print(b.to_string(index=False))
    u = pd.read_sql_query("SELECT COUNT(*) AS n, SUM(race_results_id IS NULL) AS unmatched, "
                          "MIN(event_date) AS first, MAX(event_date) AS last FROM betfair_prices", conn) \
        if "event_date" in {r[1] for r in conn.execute("PRAGMA table_info(betfair_prices)")} else None
    if u is not None:
        print(u.to_string(index=False))
conn.close()
