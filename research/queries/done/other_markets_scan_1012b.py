"""The other-markets scan for 12 Oct again, with each market's places as its book showed them at the poll (the fix of 13 Oct: Worcester 16:37's 4 TBP read at its after-race 3 places) (the evening check-in of 13 Oct; read-only).
research/queries/done/other_markets_scan.py with SCAN_DAY=2026-10-12."""
import os
import runpy
import sys

os.environ["SCAN_DAY"] = "2026-10-12"
sys.argv = ["other_markets_scan.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "other_markets_scan.py"), run_name="__main__")
