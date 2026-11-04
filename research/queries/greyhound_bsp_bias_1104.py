"""The SP's bias by segment as limit-on-close rules (research/greyhound/bsp_bias.py; read-only): back at the SP at or
above a threshold, lay at or below one, by track and trap, track, trap, field and session; chosen on 2016-21, read on
2022-26 as one portfolio (bootstrap over days) and rule by rule."""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from sources.common import Store                  # noqa: E402
from research.greyhound import bsp_bias as bb    # noqa: E402

rep = bb.run(Store())
os.makedirs("out", exist_ok=True)
open("out/bsp_bias.json", "w").write(json.dumps(rep, indent=1, default=str))
for m in ("win", "place"):
    r = rep[m]
    print(f"\n==== {m}: rules tried {r['rules_tried']}, selected on 2016-21 {r['selected']}")
    print("holdout portfolio:", json.dumps(r["portfolio"]))
    for x in r["selected_rules"]:
        print(x)
