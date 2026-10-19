"""The other-markets scan for 18 Oct, its price file now in (the evening check-in of 19 Oct; read-only).
research/queries/done/other_markets_scan.py with SCAN_DAY=2026-10-18."""
import os
import runpy
import sys

os.environ["SCAN_DAY"] = "2026-10-18"
sys.argv = ["other_markets_scan.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "other_markets_scan.py"), run_name="__main__")
