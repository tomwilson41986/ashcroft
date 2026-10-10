"""The strategy replay with 9 Oct's price file now loaded (the evening check-in of 10 Oct; read-only).
research/queries/done/strategy_replay_1009b.py unchanged: its list already holds 9 Oct (opening GBP4,951.66)."""
import os
import runpy
import sys

sys.argv = ["strategy_replay_1009b.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "strategy_replay_1009b.py"), run_name="__main__")
