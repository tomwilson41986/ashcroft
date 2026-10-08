"""The stake target replay again with 7 Oct (the evening check-in of 8 Oct; read-only).

research/queries/done/stake_target_replay.py with 7 Oct added to its days (opening balance GBP2,577.13, the account at the close
of 6 Oct), now that 7 Oct's price file (dwbfprices*08102026) is due in the archive: each recorded day with its file,
1-7 Oct. Does the order of the rules change with the seventh day?
"""
import os
import sys
from datetime import date  # noqa: F401

path = os.path.join(os.getcwd(), "research", "queries", "done", "stake_target_replay.py")
src = open(path, encoding="utf-8").read()
old = "date(2026, 10, 6): 1956.62}"
assert old in src
src = src.replace(old, "date(2026, 10, 6): 1956.62, date(2026, 10, 7): 2577.13}")
sys.argv = ["stake_target_replay.py"]
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
