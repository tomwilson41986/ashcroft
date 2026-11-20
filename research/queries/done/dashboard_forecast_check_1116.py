"""Reconcile the dashboard's day-report forecast (dashboard/day_forecast.py) against the day report itself
(research/queries/done/day_report.py) for 16 Nov, read-only: the same S3 snapshot, uncached and cached."""
import os
import runpy
import sys
from datetime import date

sys.path.insert(0, os.getcwd())
os.environ["REPORT_DAY"] = "2026-11-16"
from dashboard.day_forecast import record_days, today_forecast  # noqa: E402
from dashboard.traders import TRADERS, S3Source  # noqa: E402

src = S3Source()
t = next(x for x in TRADERS if x.id == "horse-live")
day = date(2026, 11, 16)
past = record_days(src.days(t.kind, t.base + "/"), day, t.record_from, t.start_from)
for cache in (False, True):
    fc = today_forecast(src, t.id, t.kind, t.base, day, past, cache=cache)
    print(f"== dashboard (cache={cache}): books to {fc.get('books_to')}; settled {fc.get('settled'):+,.2f}; "
          f"{fc.get('races_left')} to run; mark mid {fc.get('mark_mid'):+,.2f}, Cash Out {fc.get('cash_out'):+,.2f}; "
          f"races to run {fc.get('forecast_left'):+,.2f} from the mid ({fc.get('forecast_left_cash_out'):+,.2f} from "
          f"Cash Out), range {fc.get('lo_left'):+,.2f} to {fc.get('hi_left'):+,.2f}; {fc.get('past_days')} days, "
          f"{fc.get('past_races')} races")
    for r in fc.get("races", []):
        print("  ", r)
sys.argv = ["day_report.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "day_report.py"), run_name="__main__")
