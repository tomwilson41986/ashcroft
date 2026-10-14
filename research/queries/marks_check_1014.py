"""The nightly load of 14 Oct: the final marks' BSP for 11-14 Oct and the price-file join by racing day 1 Oct - 13 Oct
(the evening check-in of 14 Oct; read-only)."""
import sqlite3

conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
print("race_date, final marks, with a BSP")
for r in conn.execute("""SELECT race_date, COUNT(*), SUM(bsp > 1) FROM betfair_live_marks WHERE mark = 'final'
                         AND race_date >= '2026-10-11' GROUP BY race_date ORDER BY race_date"""):
    print("  ", r)
print("win-market price rows joined to a result, by racing day:")
for r in conn.execute(
        """SELECT r.race_date, COUNT(*) FROM betfair_prices b JOIN race_results r ON r.id = b.race_results_id
           WHERE r.race_date >= '2026-10-01' AND LOWER(b.market_type) = 'win' GROUP BY r.race_date
           ORDER BY r.race_date"""):
    print("  ", r)
print("results rows by day (HRB):", list(conn.execute(
    "SELECT race_date, COUNT(*) FROM race_results WHERE race_date >= '2026-10-09' GROUP BY race_date")))
