"""When to bet (the owner, 4 Nov 2026: "Evaluate when is the best time to bet; both on prices and market depth /
liquidity. Night before? At 4am? ... Model the optimal timing for profit"). research/arms/timing.py on the recorded
books, 12 Oct - 3 Nov: the live rule from the evening before, 05:15, 06:15, 07:00 and 08:00, with and without the
day limit, after the non-runners' cuts. Read-only."""
import os
import sys

sys.path.insert(0, os.getcwd())
from research.arms import timing  # noqa: E402

timing.main(["--days", "2026-10-12..2026-11-03", "--arms", "all", "--jobs", "4", "--out", "out/timing_1104",
             "--max-minutes", "290"])
