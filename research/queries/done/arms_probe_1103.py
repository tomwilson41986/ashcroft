"""The strategy arms' probe (the owner, 3 Nov 2026; read-only): every arm on two recorded days, 30 and 31 Oct, to
see the machinery run on the real books before the full 12 Oct - 2 Nov run (research/arms/)."""
import os
import sys

sys.path.insert(0, os.getcwd())
from research.arms.run import main  # noqa: E402

sys.exit(main(["--days", "2026-10-30,2026-10-31", "--arms", "all", "--jobs", "2", "--out", "out/arms_probe",
               "--max-minutes", "150"]))
