"""The stop-time replay at GBP5,000 again with 6 Oct (the evening check-in of 7 Oct; read-only).

research/queries/done/time_cutoff_replay.py (run 37607762394, 1-5 Oct at GBP5,000) now that 6 Oct's price file is
archived: the owner's rule since 7 Oct (GBP400 to win, GBP300 a bet, no day limit, to 15 minutes before each off: the
"21:30" row), against each stop time and against the GBP4,000 day limit. The account closed 7 Oct at about GBP5,411
(the owner's deposits through the day), so GBP5,000 is now nearer the account than the morning's GBP3,578.
"""
import os
import runpy
import sys

sys.argv = ["time_cutoff_replay.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "time_cutoff_replay.py"), run_name="__main__")
