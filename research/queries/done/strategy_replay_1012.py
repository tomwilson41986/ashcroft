"""The strategy replay with 11 Oct, and stops at 10:00, 11:00 and 12:00 UK under the GBP4,000 limit (the evening
check-in of 12 Oct; read-only). On 11 Oct the backs entered from 10:00 UK lost (10:00 -10.7%, 11:00 -13.1%; 08:00
+20.4%). research/queries/done/strategy_replay_1009.py with 9-12 Oct in its days (opening GBP4,951.66, GBP4,888.75,
GBP5,472.86, GBP5,000.00; a day whose price file is not archived waits), GBP5,000 in the account as before."""
import os
import sys

path = os.path.join(os.getcwd(), "research", "queries", "done", "strategy_replay_1009.py")
src = open(path, encoding="utf-8").read()
old = 'date(2026, 10, 8): 5410.91}")'
assert old in src
src = src.replace(old, 'date(2026, 10, 8): 5410.91, date(2026, 10, 9): 4951.66, date(2026, 10, 10): 4888.75, '
                       'date(2026, 10, 11): 5472.86, date(2026, 10, 12): 5000.0}")')
a, b = src.index("ARMS = ["), src.index("]\nBALANCE")
src = src[:a] + '''ARMS = [  # name, model, no top-ups, target, day limit, until
    ("no-volume, top-ups, 400, 4000/day (the rule)", "novol", False, 400.0, 4000.0, "21:30"),
    ("no-volume, top-ups, 400, 4000/day, to 12:00", "novol", False, 400.0, 4000.0, "12:00"),
    ("no-volume, top-ups, 400, 4000/day, to 11:00", "novol", False, 400.0, 4000.0, "11:00"),
    ("no-volume, top-ups, 400, 4000/day, to 10:00", "novol", False, 400.0, 4000.0, "10:00"),
    ("no-volume, top-ups, 400, 5000/day", "novol", False, 400.0, 5000.0, "21:30"),
    ("no-volume, one back, 400, 4000/day", "novol", True, 400.0, 4000.0, "21:30"),
''' + src[b:]
sys.argv = ["strategy_replay_1012.py"]
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
