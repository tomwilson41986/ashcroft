"""The owner's day report for 4 Nov (the standard approach: research/queries/done/day_report.py; read-only), then the
greyhounds so far: the early lay's live orders and the paper trader's running table, from their ledgers in S3."""
import os
import runpy
import sys
from datetime import date

os.environ["REPORT_DAY"] = "2026-11-04"
sys.argv = ["day_report.py"]
try:
    runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "day_report.py"), run_name="__main__")
except SystemExit:
    pass
except Exception as exc:                                    # the greyhound tables still print
    print("day report failed:", type(exc).__name__, exc)

sys.path.insert(0, os.getcwd())
from sources.common import Store                            # noqa: E402
from greyhound import early_lay, live                       # noqa: E402

day, store = date(2026, 11, 4), Store()
print("\n# Greyhounds\n")
for name, fn in (("early lay live", lambda: early_lay.table(store, day, "live")),
                 ("paper trader", lambda: live.running_markdown(live.running(store, day, "paper")))):
    try:
        print(fn())
    except Exception as exc:
        print(f"{name}: failed: {type(exc).__name__} {exc}")
