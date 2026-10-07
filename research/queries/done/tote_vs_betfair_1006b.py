"""The Tote against Betfair on 6 Oct again, with 6 Oct's price files loaded into betfair_prices tonight (its close
against the BSP, the winners' declared dividends, the exotics) (the evening check-in of 7 Oct; read-only).

research/queries/done/tote_vs_betfair.py with TOTE_DAY=2026-10-06.
"""
import os
import runpy
import sys

os.environ["TOTE_DAY"] = "2026-10-06"
sys.argv = ["tote_vs_betfair.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "tote_vs_betfair.py"), run_name="__main__")
