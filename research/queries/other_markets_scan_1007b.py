"""The other-markets scan on 7 Oct again, with the arbitrage states fixed (a 5 TBP back alone looked riskless) and the
close read from the price files when they are in (the evening check-in of 7 Oct; read-only).

research/queries/done/other_markets_scan.py with SCAN_DAY=2026-10-07.
"""
import os
import runpy
import sys

os.environ["SCAN_DAY"] = "2026-10-07"
sys.argv = ["other_markets_scan.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "other_markets_scan.py"), run_name="__main__")
