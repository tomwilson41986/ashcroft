"""The Tote against Betfair on 9 Oct, parts 1-4 (its BSPs from the day record, read at the off) (the evening review of
9 Oct; read-only). research/queries/done/tote_vs_betfair.py with TOTE_DAY=2026-10-09."""
import os
import runpy
import sys

os.environ["TOTE_DAY"] = "2026-10-09"
sys.argv = ["tote_vs_betfair.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "tote_vs_betfair.py"), run_name="__main__")
