"""The strategy replay with 16 Oct (the evening check-in of 17 Oct; done/strategy_replay_1016.py with 17 Oct added;
before that: done/strategy_replay_1015.py with 16 Oct added;
before that: done/strategy_replay_1014.py with 15 Oct added, and the GBP200 and GBP150 caps a horse beside the rule, the owner having asked the optimal stake a horse; first written the evening of
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
                       'date(2026, 10, 11): 5472.86, date(2026, 10, 12): 5000.0, date(2026, 10, 13): 5159.15, date(2026, 10, 14): 5469.67, date(2026, 10, 15): 5505.0, date(2026, 10, 16): 5189.34, date(2026, 10, 17): 5155.53}")')
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
src = src.replace(old, "    for name, model, one_back, target, max_day, until, start, cap in ARMS:\n"
                       "        g[\"FROM\"] = start\n")
old = '        r = g["replay"](day, target, BALANCE, 300.0, max_day)\n'
assert old in src
src = src.replace(old, '        r = g["replay"](day, target, BALANCE, cap, max_day)\n')
a, b = src.index("ARMS = ["), src.index("]\nBALANCE")
src = src[:a] + '''ARMS = [  # name, model, no top-ups, target, day limit, until, from, cap a horse
    ("no-volume, top-ups, 400, 4000/day, from 06:15 (the rule)", "novol", False, 400.0, 4000.0, "21:30", "06:15", 300.0),
    ("no-volume, top-ups, 400, 4000/day, from 08:00", "novol", False, 400.0, 4000.0, "21:30", "08:00", 300.0),
    ("no-volume, top-ups, 400, 4000/day, from 06:15, cap 200", "novol", False, 400.0, 4000.0, "21:30", "06:15", 200.0),
    ("no-volume, top-ups, 400, 4000/day, from 06:15, cap 150", "novol", False, 400.0, 4000.0, "21:30", "06:15", 150.0),
    ("no-volume, top-ups, 400, 5000/day, from 06:15", "novol", False, 400.0, 5000.0, "21:30", "06:15", 300.0),
    ("no-volume, one back, 400, 4000/day, from 06:15", "novol", True, 400.0, 4000.0, "21:30", "06:15", 300.0),
''' + src[b:]
sys.argv = ["strategy_replay_1017.py"]
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
