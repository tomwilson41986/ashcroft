"""The live tracker on 17 Oct, 07:46 UK (the early pace; the owner's ask "Let's track tomorrow"; read-only)."""
import os
import runpy
import sys

os.environ["TRACK_DAY"] = "2026-10-17"
sys.argv = ["track_day.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "track_day.py"), run_name="__main__")
