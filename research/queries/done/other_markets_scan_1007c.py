"""The other-markets scan on 7 Oct again, its price file now in: the close against the BSP (the evening check-in of
8 Oct; read-only). research/queries/done/other_markets_scan.py with SCAN_DAY=2026-10-07."""
import os
import runpy
import sys

os.environ["SCAN_DAY"] = "2026-10-07"
sys.argv = ["other_markets_scan.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "other_markets_scan.py"), run_name="__main__")
