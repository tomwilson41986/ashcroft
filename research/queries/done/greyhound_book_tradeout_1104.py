"""The trade-to-SP rule against real books (research/greyhound/book_tradeout.py; read-only): the forecast fitted on
Betfair's historic data Jan-Apr 2026, applied to the recorded GB books 60 s before the off (6 Oct on), entered at the
book's best price with at least GBP2 on offer and closed at the SP; beside it the tick-through-the-last-trade fill."""
import json
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.getcwd())
from sources.common import Store                       # noqa: E402
from research.greyhound import book_tradeout as bt    # noqa: E402

rep = bt.run(Store(), date(2026, 10, 6), date.today() - timedelta(days=1))
os.makedirs("out", exist_ok=True)
open("out/book_tradeout.json", "w").write(json.dumps(rep, indent=1, default=str))
print({k: v for k, v in rep.items() if k != "rules"})
for r in rep.get("rules", []):
    print(r)
