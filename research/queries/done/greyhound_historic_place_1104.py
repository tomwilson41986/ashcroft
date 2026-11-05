"""The place market against the win market on Betfair's historic data (research/greyhound/historic_place.py;
read-only): Jan-Sep 2026, GB and AU, at 1 and 15 minutes out, bet at the place SP; and the win SP forecast from both
markets' last traded prices against the win market's own at 1 minute out."""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from sources.common import Store                       # noqa: E402
from research.greyhound import historic_place as hp   # noqa: E402

rep = hp.run(Store())
os.makedirs("out", exist_ok=True)
open("out/historic_place.json", "w").write(json.dumps(rep, indent=1, default=str))
print({k: rep[k] for k in ("rows", "paired_runners", "races", "by_country", "months", "ltp_t_1_share", "rules_tried")})
for r in rep["rules"]:
    print(r["mark"], r["country"], r["side"], r["edge>"], "dev", r.get("dev"), "holdout", r.get("holdout"), r["passes"])
print("PASSING:", json.dumps(rep["passing"]))
print("CLOSING:", json.dumps(rep["closing"]))
