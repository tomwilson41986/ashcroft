"""The owner's day report for 26 Oct (the standard approach: research/queries/done/day_report.py; read-only)."""
import os
import runpy
import sys

os.environ["REPORT_DAY"] = "2026-10-26"
sys.argv = ["day_report.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "day_report.py"), run_name="__main__")
