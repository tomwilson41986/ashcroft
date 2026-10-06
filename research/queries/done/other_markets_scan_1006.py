"""The other-markets scan on 6 Oct's record (the evening check-in of 6 Oct; read-only).

research/queries/done/other_markets_scan.py with SCAN_DAY=2026-10-06. The 2/3/4 TBP and each-way books are recorded
inside the trader's job, which was down from 09:47 to 12:55 UTC: the books run to 10:47 UK, then from 13:55 to the
last race. The 10:00 mark stands; 12:00 does not; the marks before the offs stand from about 14:55 UK. The terms of
6 Oct's markets are the standard ones (the places paid and the each-way divisor are recorded from 7 Oct).
"""

import os
import runpy
import sys

os.environ["SCAN_DAY"] = "2026-10-06"
sys.argv = ["other_markets_scan.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "other_markets_scan.py"), run_name="__main__")
