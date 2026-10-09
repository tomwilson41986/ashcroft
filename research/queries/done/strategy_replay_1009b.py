"""The strategy replay again with 8 and 9 Oct (the evening review of 9 Oct; read-only).
research/queries/done/strategy_replay_1009.py with 9 Oct added to its days (opening GBP4,951.66; 8 Oct, opening
GBP5,410.91, was in its list and now has its price file). A day whose price file is not archived waits."""
import os
import sys

path = os.path.join(os.getcwd(), "research", "queries", "done", "strategy_replay_1009.py")
src = open(path, encoding="utf-8").read()
old = 'date(2026, 10, 8): 5410.91}")'
assert old in src
src = src.replace(old, 'date(2026, 10, 8): 5410.91, date(2026, 10, 9): 4951.66}")')
sys.argv = ["strategy_replay_1009b.py"]
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
