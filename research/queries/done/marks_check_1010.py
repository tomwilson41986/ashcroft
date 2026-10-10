"""The nightly load of 10 Oct: the final marks' BSP for 7-10 Oct and the price-file join by racing day 20 Sep - 10 Oct
(is 23 Sep - 1 Oct back after betfair_prices.match_days?) (the evening check-in of 10 Oct; read-only)."""
import sqlite3

conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
print("race_date, final marks, with a BSP")
for r in conn.execute("""SELECT race_date, COUNT(*), SUM(bsp > 1) FROM betfair_live_marks WHERE mark = 'final'
                         AND race_date >= '2026-10-07' GROUP BY race_date ORDER BY race_date"""):
    print("  ", r)
print("win-market price rows joined to a result, by racing day:")
for r in conn.execute(
        """SELECT r.race_date, COUNT(*) FROM betfair_prices b JOIN race_results r ON r.id = b.race_results_id
           WHERE r.race_date >= '2026-09-20' AND LOWER(b.market_type) = 'win' GROUP BY r.race_date
           ORDER BY r.race_date"""):
    print("  ", r)
print("results rows by day (HRB):", list(conn.execute(
    "SELECT race_date, COUNT(*) FROM race_results WHERE race_date >= '2026-10-05' GROUP BY race_date")))
