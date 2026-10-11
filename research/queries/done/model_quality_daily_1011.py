"""The prediction model's 06:00 prices against Betfair with 10 Oct's price file (the evening check-in of 11 Oct;
read-only). research/queries/done/model_quality_daily_1009.py unchanged."""
import os
import runpy
import sys

sys.argv = ["model_quality_daily_1009.py"]
runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "model_quality_daily_1009.py"),
               run_name="__main__")
