"""The prediction model's 06:00 prices against Betfair with 9 Oct's price file and 23 Sep - 1 Oct re-matched (the
evening check-in of 10 Oct; read-only). research/queries/done/model_quality_daily_1009.py unchanged."""
import os
import runpy
import sys

sys.argv = ["model_quality_daily_1009.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "model_quality_daily_1009.py"),
               run_name="__main__")
