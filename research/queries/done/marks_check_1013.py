"""The nightly load of 13 Oct: the final marks' BSP for 10-13 Oct and the price-file join by racing day 1 Oct - 12 Oct
(the evening check-in of 13 Oct; read-only)."""
import sqlite3

conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
print("race_date, final marks, with a BSP")
for r in conn.execute("""SELECT race_date, COUNT(*), SUM(bsp > 1) FROM betfair_live_marks WHERE mark = 'final'
                         AND race_date >= '2026-10-10' GROUP BY race_date ORDER BY race_date"""):
    print("  ", r)
print("win-market price rows joined to a result, by racing day:")
for r in conn.execute(
        """SELECT r.race_date, COUNT(*) FROM betfair_prices b JOIN race_results r ON r.id = b.race_results_id
           WHERE r.race_date >= '2026-10-01' AND LOWER(b.market_type) = 'win' GROUP BY r.race_date
           ORDER BY r.race_date"""):
    print("  ", r)
print("results rows by day (HRB):", list(conn.execute(
    "SELECT race_date, COUNT(*) FROM race_results WHERE race_date >= '2026-10-08' GROUP BY race_date")))
