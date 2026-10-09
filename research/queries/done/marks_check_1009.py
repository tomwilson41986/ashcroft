"""The nightly marks with PR #141's fix: the final mark's BSP for 5-9 Oct (the evening review of 9 Oct; read-only)."""
import sqlite3

conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
print("race_date, final marks, with a BSP")
for r in conn.execute("""SELECT race_date, COUNT(*), SUM(bsp > 1) FROM betfair_live_marks WHERE mark = 'final'
                         AND race_date >= '2026-10-05' GROUP BY race_date ORDER BY race_date"""):
    print("  ", r)
print("marks by day:", list(conn.execute("""SELECT race_date, COUNT(*) FROM betfair_live_marks
                                             WHERE race_date >= '2026-10-05' GROUP BY race_date ORDER BY race_date""")))
print("betfair_prices rows by day:", list(conn.execute(
    """SELECT r.race_date, COUNT(*) FROM betfair_prices b JOIN race_results r ON r.id = b.race_results_id
       WHERE r.race_date >= '2026-10-05' AND LOWER(b.market_type) = 'win' GROUP BY r.race_date ORDER BY r.race_date""")))
