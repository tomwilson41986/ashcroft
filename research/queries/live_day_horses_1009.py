"""The live record 30 Sep - 9 Oct, one row a horse, as CSV in the log (the evening review of 9 Oct; read-only).
research/queries/done/live_review_csv_1008.py's days and horses, with the closing model read from the ledger's reason:
"feed without volume" (the delayed key) and, from 9 Oct, "no-volume model" (the live key's feed read without volume,
the owner's rule from the 10:19 UK restart) are the no-volume model; anything else the volume model."""
import os
from pathlib import Path

src = Path(os.getcwd()) / "research" / "queries" / "done" / "live_review_csv_1008.py"
code = src.read_text().replace('("ORDERS", oo)', '').replace('(("DAYS", b.reset_index()), ("HORSES", hh), )',
                                                              '(("DAYS", b.reset_index()), ("HORSES", hh))')
review = Path(os.getcwd()) / "research" / "queries" / "done" / "live_review_1008.py"
old = 'o.reason.fillna("").str.contains("without volume")'
rtext = review.read_text()
assert old in rtext
patched = rtext.replace(old, 'o.reason.fillna("").str.contains("without volume|no-volume model")')
code = code.replace('src = Path(os.getcwd()) / "research" / "queries" / "done" / "live_review_1008.py"\n'
                    'code = src.read_text()', 'src = Path(os.getcwd()) / "research" / "queries" / "done" / '
                    '"live_review_1008.py"\ncode = PATCHED')
assert "code = PATCHED" in code
exec(compile(code, str(src), "exec"), {"__name__": "__main__", "PATCHED": patched})
