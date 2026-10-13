"""The strategy replay with 12 Oct, and the owner's start of 13 Oct (06:15 UK) against 08:00 (the evening check-in of
13 Oct; read-only). research/queries/done/strategy_replay_1009.py with 9-13 Oct in its days (a day whose price file
is not archived waits: 13 Oct's comes tomorrow), GBP5,000 in the account as before. The replay's clock now starts at
05:00 UTC and each arm sets the window's opening (trading/config_live.json reads 06:15 from 13 Oct); before 13 Oct no
book was recorded before about 08:00 UK, so the two openings differ only from 13 Oct."""
import os
import sys

path = os.path.join(os.getcwd(), "research", "queries", "done", "strategy_replay_1009.py")
src = open(path, encoding="utf-8").read()
old = 'date(2026, 10, 8): 5410.91}")'
assert old in src
src = src.replace(old, 'date(2026, 10, 8): 5410.91, date(2026, 10, 9): 4951.66, date(2026, 10, 10): 4888.75, '
                       'date(2026, 10, 11): 5472.86, date(2026, 10, 12): 5000.0, date(2026, 10, 13): 5159.15}")')
old = '"    cfg.trade_until = UNTIL\\n")'
assert old in src
src = src.replace(old, '"    cfg.trade_until = UNTIL\\n    cfg.trade_from = FROM\\n")\n'
                       'head = head.replace("day.day, 6, 30, tzinfo", "day.day, 5, 0, tzinfo")\n'
                       'assert "day.day, 5, 0, tzinfo" in head')
old = '"POLICY": {}, "UNTIL": "21:30"}'
assert old in src
src = src.replace(old, '"POLICY": {}, "UNTIL": "21:30", "FROM": "06:15"}')
old = "    for name, model, one_back, target, max_day, until in ARMS:\n"
assert old in src
src = src.replace(old, "    for name, model, one_back, target, max_day, until, start in ARMS:\n"
                       "        g[\"FROM\"] = start\n")
a, b = src.index("ARMS = ["), src.index("]\nBALANCE")
src = src[:a] + '''ARMS = [  # name, model, no top-ups, target, day limit, until, from
    ("no-volume, top-ups, 400, 4000/day, from 06:15 (the rule)", "novol", False, 400.0, 4000.0, "21:30", "06:15"),
    ("no-volume, top-ups, 400, 4000/day, from 08:00", "novol", False, 400.0, 4000.0, "21:30", "08:00"),
    ("no-volume, top-ups, 400, 4000/day, from 06:15 to 10:00", "novol", False, 400.0, 4000.0, "10:00", "06:15"),
    ("no-volume, top-ups, 400, 5000/day, from 06:15", "novol", False, 400.0, 5000.0, "21:30", "06:15"),
    ("no-volume, one back, 400, 4000/day, from 06:15", "novol", True, 400.0, 4000.0, "21:30", "06:15"),
''' + src[b:]
sys.argv = ["strategy_replay_1013.py"]
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
