"""The other-markets scan on 7 Oct's record, with the recorded terms and the BSP read at the off (read-only).

research/queries/done/other_markets_scan.py with SCAN_DAY=2026-10-07. The places each market pays and the each-way
divisors are recorded from 7 Oct, and the win BSP is read at each off (the scan now takes it from those rows: a closed
market's final book carries none). The records run inside the trader's job, which was down from 11:07 to 12:00 UTC and
restarted at 12:00, 12:09, 12:14, 12:35, 12:41, 13:33 and 14:53 UTC.
"""
import os
import runpy
import sys

os.environ["SCAN_DAY"] = "2026-10-07"
sys.argv = ["other_markets_scan.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "other_markets_scan.py"), run_name="__main__")
