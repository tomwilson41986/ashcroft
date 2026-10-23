"""The prediction model's 06:00 prices against Betfair to 23 Oct (the evening check-in of 23 Oct; read-only).
research/queries/done/model_quality_daily_1009.py stopped at 9 Oct (its TO), so the 1011 and 1012 runs left out
10 and 11 Oct; this runs it with TO moved to 23 Oct and nothing else changed."""
import os
import sys

path = os.path.join(os.getcwd(), "research", "queries", "done", "model_quality_daily_1009.py")
src = open(path).read()
old = "FROM, TO = date(2026, 9, 1), date(2026, 10, 9)"
assert old in src
src = src.replace(old, "FROM, TO = date(2026, 9, 1), date(2026, 10, 23)")
sys.argv = ["model_quality_daily_1009.py"]
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
