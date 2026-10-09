"""The Tote on 9 Oct again, now that a pool with no winning ticket (no declared dividend) is skipped (the evening
review of 9 Oct; read-only). research/queries/done/tote_vs_betfair.py with TOTE_DAY=2026-10-09: the exotics' part."""
import os
import runpy
import sys

os.environ["TOTE_DAY"] = "2026-10-09"
sys.argv = ["tote_vs_betfair.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "tote_vs_betfair.py"), run_name="__main__")
