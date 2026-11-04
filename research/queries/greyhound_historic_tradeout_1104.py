"""Trading to the SP on a forecast of the close (research/greyhound/historic_tradeout.py; read-only): at 1 minute out,
back or lay a tick through the last traded price where it beats the forecast SP by a margin, closed at the SP;
Betfair's historic data Jan-Sep 2026, GB and AU, chosen on Jan-Apr, read on May-Sep."""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from sources.common import Store                          # noqa: E402
from research.greyhound import historic_tradeout as ht   # noqa: E402

rep = ht.run(Store())
os.makedirs("out", exist_ok=True)
open("out/historic_tradeout.json", "w").write(json.dumps(rep, indent=1, default=str))
print({k: rep[k] for k in rep if k.startswith("rmse") or k in ("runners", "rules_tried")})
for r in rep["rules"]:
    print(r["country"], r["side"], r["margin"], "dev", r.get("dev"), "holdout", r.get("holdout"), r["passes"])
print("PASSING:", json.dumps(rep["passing"]))
