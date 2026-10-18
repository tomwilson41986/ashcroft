"""The other-markets scan for 17 Oct, its price file now in (the evening check-in of 18 Oct; read-only).
research/queries/done/other_markets_scan.py with SCAN_DAY=2026-10-17."""
import os
import runpy
import sys

os.environ["SCAN_DAY"] = "2026-10-17"
sys.argv = ["other_markets_scan.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "other_markets_scan.py"), run_name="__main__")
