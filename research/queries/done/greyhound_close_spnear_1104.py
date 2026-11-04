"""The closing price with Betfair's projected SP (the owner, 4 Nov 2026: "fundamental is always just to get the most
accurate closing price, optimise towards this"; read-only). research/greyhound/bfsp_close.py now reads each book's
projected SP (near, far) and the SP money, and scores 'sp_near' (the projected SP itself) beside the mid and the
models: RMSE of log(BSP) at first traded, 5, 3 and 1 minute out, each day forecast from the days before it."""
import os
import sys

sys.path.insert(0, os.getcwd())
from research.greyhound import bfsp_close as bc        # noqa: E402

print("==== the recorder's books, re-timed (from 6 Oct) ====")
bc.main(["--source", "retimed", "--out", "out/greyhound_close_retimed"])
print("\n==== the live key's books (from 2 Nov) ====")
bc.main(["--source", "live", "--from", "2026-11-02", "--min-train-days", "1", "--out", "out/greyhound_close_live"])
