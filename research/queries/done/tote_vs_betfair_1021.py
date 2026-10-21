"""The Tote against Betfair on 21 Oct (the evening check-in of 21 Oct; read-only).
research/queries/done/tote_vs_betfair.py with TOTE_DAY=2026-10-21."""
import os
import runpy
import sys

os.environ["TOTE_DAY"] = "2026-10-21"
sys.argv = ["tote_vs_betfair.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "tote_vs_betfair.py"), run_name="__main__")
