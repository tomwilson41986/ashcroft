"""Betfair's Historic Data for horse racing: the BASIC minute files of the UK/IE WIN markets (the owner, 12 Nov 2026:
"yes, fetch the minute files").

The price files (betfair_prices.py) hold the runners with a BSP only; a horse declared, traded and withdrawn is not
in them. The BASIC stream file of a market carries the last traded price each minute of every runner, the removed
ones too: the readings the ghost-run block needs (reports/discovery_1111.md 4.1; ledger ghost-runs-count-1112).

Read-only and research only: the fetch of betfair_historic.py with its own scope, raw to
``sources/betfair_historic/raw/horse_racing/<the service's path>``, newest month first, each file once. Nothing in
the live trader or its jobs reads this. The sport has to be "bought" (free) once at historicdata.betfair.com with the
trading account before the service lists it; the run says so and stops if it is not held.

    python -m research.horse_minute_files --max-minutes 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date

import betfair_historic as bh
from sources.common import Store

SPORT = "Horse Racing"
SCOPE = {"markets": ["WIN"], "countries": ["GB", "IE"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="date_from", default="2016-01-01")
    ap.add_argument("--to", dest="date_to", default=None)
    ap.add_argument("--max-minutes", type=float, default=100)
    ap.add_argument("--root", default=None, help="a local folder in place of S3")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    hd = bh.Historic(bh.login_token())
    held = {str(x.get("sport")) for x in hd.my_data() if str(x.get("plan", "")).lower() == bh.PLAN.lower()}
    if SPORT not in held:
        print(json.dumps({"held": sorted(held), "fetched": 0, "blocked":
                          f"the account does not hold '{SPORT}' on the {bh.PLAN}: get it (free) at "
                          "historicdata.betfair.com, signed in with the trading account"}, indent=1))
        return 2
    bh.PLAN_SCOPE = {SPORT: SCOPE}                 # this process only: the other sports' nightly fetch is untouched
    out = bh.fetch(Store(root=a.root), hd, [SPORT], date.fromisoformat(a.date_from),
                   date.fromisoformat(a.date_to) if a.date_to else None, a.max_minutes)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
