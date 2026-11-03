"""The strategy arms on every recorded day (the owner, 3 Nov 2026; read-only): 12 Oct - 2 Nov, every arm
(research/arms/), four days at a time; a day whose price file is not yet archived is skipped with a note."""
import os
import sys

sys.path.insert(0, os.getcwd())
from research.arms.run import main  # noqa: E402

sys.exit(main(["--days", "2026-10-12..2026-11-02", "--arms", "all", "--jobs", "4", "--out", "out/arms_full",
               "--max-minutes", "290"]))
