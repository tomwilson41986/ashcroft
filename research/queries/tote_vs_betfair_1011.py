"""The Tote against Betfair on 11 Oct (the evening check-in of 11 Oct; read-only).
research/queries/done/tote_vs_betfair.py with TOTE_DAY=2026-10-11."""
import os
import runpy
import sys

os.environ["TOTE_DAY"] = "2026-10-11"
sys.argv = ["tote_vs_betfair.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "tote_vs_betfair.py"), run_name="__main__")
