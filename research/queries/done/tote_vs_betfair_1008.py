"""The Tote against Betfair on 8 Oct (parts 1-2; its BSPs come with tomorrow's price file) (the evening check-in of
8 Oct; read-only). research/queries/done/tote_vs_betfair.py with TOTE_DAY=2026-10-08."""
import os
import runpy
import sys

os.environ["TOTE_DAY"] = "2026-10-08"
sys.argv = ["tote_vs_betfair.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "tote_vs_betfair.py"), run_name="__main__")
