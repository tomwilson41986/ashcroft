"""The Tote against Betfair on the first whole day of the Tote record, with the BSP read at the off (7 Oct; read-only).

research/queries/done/tote_vs_betfair.py with TOTE_DAY=2026-10-07: the Tote's pools from 08:00 UK to the last race
(tote_recorder.py beside the trader), against the day's Betfair record, whose BSPs are read as each market reconciled
at the off (PR #120, from 7 Oct). 6 Oct is run again once its price files are in betfair_prices.
"""
import os
import runpy
import sys

os.environ["TOTE_DAY"] = "2026-10-07"
sys.argv = ["tote_vs_betfair.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "tote_vs_betfair.py"), run_name="__main__")
