"""The live tracker tried on 16 Oct (its first run, before 17 Oct; read-only)."""
import os
import runpy
import sys

os.environ["TRACK_DAY"] = "2026-10-16"
sys.argv = ["track_day.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "track_day.py"), run_name="__main__")
