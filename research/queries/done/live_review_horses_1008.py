"""The live record's days and horses as CSV in the log, without the orders (the log of the run with all three was cut
at its head). research/queries/done/live_review_csv_1008.py, its first two tables."""
import os
from pathlib import Path

src = Path(os.getcwd()) / "research" / "queries" / "done" / "live_review_csv_1008.py"
code = src.read_text().replace('("ORDERS", oo)', '').replace('(("DAYS", b.reset_index()), ("HORSES", hh), )',
                                                              '(("DAYS", b.reset_index()), ("HORSES", hh))')
exec(compile(code, str(src), "exec"), {"__name__": "__main__"})
