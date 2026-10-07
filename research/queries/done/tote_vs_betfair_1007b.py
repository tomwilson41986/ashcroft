"""The Tote against Betfair on 7 Oct again, each mark's last answer kept (a mark answered twice broke part 2: the Tote
record restarts with the trader's job) (the evening check-in of 7 Oct; read-only).

research/queries/done/tote_vs_betfair.py with TOTE_DAY=2026-10-07.
"""
import os
import runpy
import sys

os.environ["TOTE_DAY"] = "2026-10-07"
sys.argv = ["tote_vs_betfair.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "tote_vs_betfair.py"), run_name="__main__")
