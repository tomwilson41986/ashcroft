"""The strategy replay with 10 Oct, and higher day limits (the evening check-in of 11 Oct; read-only).
On 10 Oct the GBP4,000 was spent in 47 minutes (08:36-09:23 UK). research/queries/done/strategy_replay_1009.py with
9-11 Oct in its days (opening GBP4,951.66, GBP4,888.75, GBP5,472.86; a day whose price file is not archived waits),
GBP5,000 in the account as before, and the no-volume model's arms at GBP4,000, GBP5,000, GBP6,000 and no day limit."""
import os
import sys

path = os.path.join(os.getcwd(), "research", "queries", "done", "strategy_replay_1009.py")
src = open(path, encoding="utf-8").read()
old = 'date(2026, 10, 8): 5410.91}")'
assert old in src
src = src.replace(old, 'date(2026, 10, 8): 5410.91, date(2026, 10, 9): 4951.66, date(2026, 10, 10): 4888.75, '
                       'date(2026, 10, 11): 5472.86}")')
a, b = src.index("ARMS = ["), src.index("]\nBALANCE")
src = src[:a] + '''ARMS = [  # name, model, no top-ups, target, day limit, until
    ("no-volume, top-ups, 400, 4000/day (the rule)", "novol", False, 400.0, 4000.0, "21:30"),
    ("no-volume, top-ups, 400, 5000/day", "novol", False, 400.0, 5000.0, "21:30"),
    ("no-volume, top-ups, 400, 6000/day", "novol", False, 400.0, 6000.0, "21:30"),
    ("no-volume, top-ups, 400, no limit", "novol", False, 400.0, NOLIMIT, "21:30"),
    ("no-volume, top-ups, 400, no limit, to 12:00", "novol", False, 400.0, NOLIMIT, "12:00"),
    ("no-volume, one back, 400, 4000/day", "novol", True, 400.0, 4000.0, "21:30"),
''' + src[b:]
sys.argv = ["strategy_replay_1011.py"]
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
