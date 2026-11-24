"""The 100-strategy test, data set a_model (ledger strat100-1124, registered before this run; research only, no order).

Runs research/strat100/run.py for one data set; the results land in out/strat100_a_model/ (uploaded with the run).
"""
import os
import subprocess
import sys

os.environ.setdefault("STRAT100_WORK", "out/strat100_work")
sys.exit(subprocess.call([sys.executable, "-m", "research.strat100.run", "--dataset", "a_model",
                          "--out", "out/strat100_a_model", "--max-minutes", "300"]))
