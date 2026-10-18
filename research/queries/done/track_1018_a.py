"""The live tracker on 18 Oct, 17:10 UK (the owner's QA of the day; read-only)."""
import os
import runpy
import sys

os.environ["TRACK_DAY"] = "2026-10-18"
sys.argv = ["track_day.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "track_day.py"), run_name="__main__")
