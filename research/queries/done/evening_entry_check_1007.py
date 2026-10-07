"""Betting the evening before, 5-7 Oct (task 209; read-only).

research/queries/done/evening_entry_check.py from 5 Oct to 7 Oct: tomorrow's win markets recorded the evening before
(by the trader's job from 17:00 to 21:30 UK from 5 Oct, so 6 and 7 Oct's markets each have a whole evening), against
the morning as the trader took it and the rule's choices. 7 Oct's BSPs are the ones read at each off (PR #120; the
check now takes them from those rows); 5 and 6 Oct's come from Betfair's price files.
"""
import os
import runpy
import sys

os.environ["EVENING_FROM"] = "2026-10-05"
os.environ["EVENING_TO"] = "2026-10-07"
sys.argv = ["evening_entry_check.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "evening_entry_check.py"), run_name="__main__")
