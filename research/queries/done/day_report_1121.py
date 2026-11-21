"""The owner's day report for 21 Nov (the standard approach: research/queries/done/day_report.py; read-only)."""
import os
import runpy
import sys

os.environ["REPORT_DAY"] = "2026-11-21"
sys.argv = ["day_report.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "day_report.py"), run_name="__main__")
