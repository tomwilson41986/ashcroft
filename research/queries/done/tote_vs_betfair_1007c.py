"""The Tote against Betfair on 7 Oct again, its price file now in (parts 3, 3b and 4: against the BSP) (the evening
check-in of 8 Oct; read-only). research/queries/done/tote_vs_betfair.py with TOTE_DAY=2026-10-07."""
import os
import runpy
import sys

os.environ["TOTE_DAY"] = "2026-10-07"
sys.argv = ["tote_vs_betfair.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "tote_vs_betfair.py"), run_name="__main__")
