"""Limit-on-close rules against the morning price (research/greyhound/drift_loc.py; read-only): back at the SP only
when the dog drifts by a share of its morning WAP, lay only when it shortens; win and place markets; 2016-21 chooses,
2022-26 reads, 90% intervals by day."""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from sources.common import Store                    # noqa: E402
from research.greyhound import drift_loc as dl     # noqa: E402

rep = dl.run(Store())
print(dl.summary(rep))
os.makedirs("out", exist_ok=True)
open("out/drift_loc.json", "w").write(json.dumps(rep, indent=1, default=str))
print("PASSING:", json.dumps(rep["passing"], indent=1))
