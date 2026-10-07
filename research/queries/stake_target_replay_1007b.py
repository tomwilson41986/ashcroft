"""The stake replay again with 6 Oct (the evening check-in of 7 Oct; read-only).

research/queries/done/stake_target_replay.py as run in the morning (ledger stake-target-replay-1007, 1-5 Oct: with
GBP5,000, GBP400 to win under the GBP4,000 day limit best, CLV x stake 1,314 against 899 at GBP250), now that 6 Oct's
price file (dwbfprices*07102026) is archived: each recorded day with its file, 1-6 Oct. 7 Oct's file comes tomorrow.
"""
import os
import runpy
import sys

sys.argv = ["stake_target_replay.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "stake_target_replay.py"), run_name="__main__")
