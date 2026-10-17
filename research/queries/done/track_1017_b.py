"""The live tracker on 17 Oct, 09:45 UK (has the GBP4,000 bound, and when; read-only)."""
import os
import runpy
import sys

os.environ["TRACK_DAY"] = "2026-10-17"
sys.argv = ["track_day.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "track_day.py"), run_name="__main__")
