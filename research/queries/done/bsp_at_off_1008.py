"""The BSP after the off on 8 Oct, the first day the trader's settling read and the recorder's read at the off ask
SP_AVAILABLE with SP_TRADED (ledger bsp-at-off-1007): the SP fields each read carried by key, the recorder's "bsp" rows,
and the nightly marks' final BSP (the evening check-in of 8 Oct; read-only)."""
import os
import sqlite3
from pathlib import Path

D = Path(os.getcwd()) / "research" / "queries" / "done"
print("== the SP fields after the off, 7 and 8 Oct")
code = (D / "bsp_fields_1007.py").read_text().replace("range(1, 8)", "range(7, 9)")
exec(compile(code, "bsp_fields_1007.py", "exec"), {"__name__": "__main__"})
print("\n== the recorder's reads at the off, 8 Oct")
code = (D / "bsp_at_off_check_1007.py").read_text().replace('DAY = "2026-10-07"', 'DAY = "2026-10-08"')
exec(compile(code, "bsp_at_off_check_1007.py", "exec"), {"__name__": "__main__"})
print("\n== the nightly marks: the final mark's BSP by day")
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
for r in conn.execute("""SELECT race_date, COUNT(*), SUM(bsp > 1) FROM betfair_live_marks WHERE mark = 'final'
                         AND race_date >= '2026-10-05' GROUP BY race_date ORDER BY race_date"""):
    print("  ", r)
