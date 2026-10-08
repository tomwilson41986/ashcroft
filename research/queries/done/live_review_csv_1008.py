"""The live record's horses and orders as CSV in the log (the workbook for the owner is built from it; the runner has
no Excel writer). research/queries/done/live_review_1008.py with its tables printed between markers."""
import os
import runpy
import sys
from pathlib import Path

import pandas as pd

src = Path(os.getcwd()) / "research" / "queries" / "done" / "live_review_1008.py"
code = src.read_text().split("os.makedirs(\"out\", exist_ok=True)")[0]
ns: dict = {"__name__": "live_review"}
import contextlib, io  # noqa: E401,E402
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(code, str(src), "exec"), ns)
h, o, b = ns["h"], ns["o"], ns["b"]
UK = "Europe/London"
hh = h.drop(columns=["sxp"]).copy()
hh["first_t"] = hh.first_t.dt.tz_convert(UK).dt.strftime("%Y-%m-%d %H:%M")
oo = o[["day", "t", "venue", "off", "minutes_to_off", "runner", "model_feed", "edge", "best_back", "back_size",
        "target", "asked", "matched", "avg_price", "outcome", "error"]].copy()
oo["t"] = oo.t.dt.tz_convert(UK).dt.strftime("%H:%M:%S")
for name, df in (("DAYS", b.reset_index()), ("HORSES", hh), ("ORDERS", oo)):
    print(f"@@@{name}@@@")
    sys.stdout.write(df.round(4).to_csv(index=False))
    print(f"@@@END{name}@@@")
