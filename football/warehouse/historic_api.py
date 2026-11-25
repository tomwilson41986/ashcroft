"""The warehouse kept up to date from Betfair's Historic Data API: every soccer file of the last days, into bronze.

racing² football data, 24 Nov 2026 (the owner: "automate the download"). Until now the warehouse grew only from the
bundles the owner downloaded by hand (``bundles.json``). This fetches the same files from the same service
(historicdata.betfair.com, the BASIC plan the account holds for Soccer) through its API, for the last ``days`` days,
and adds them to bronze. The paper tests that read the warehouse (the back-to-lay test, football-b2l-paper-1124)
then see each day's matches within a day or two.

- The scope is the bundles': MATCH_ODDS, OVER_UNDER_25 and ASIAN_HANDICAP, every country (``MARKETS``), so the new
  days are the same population as the history.
- A file the store already holds is not fetched again (its raw copy under ``betfair_historic/raw/Soccer/``, the
  same place ``betfair_historic.py --fetch`` keeps them). Bronze notes each file by its plan path and checksum, so
  adding a file twice is a no-op ("same").
- Betfair publishes a day's BASIC files some time after the day; a file that is not listed yet is fetched on a later
  run. Running every night with ``days`` of 10 or more covers the delay.
- Where it runs: Betfair refuses logins from GitHub's runners (403 at identitysso.betfair.com, run 45413693273), so
  the nightly fetch runs on the owner's paper-models runner, 05:15-05:38 UTC (football-warehouse.yml, the owner's
  choice, 24 Nov 2026), and the build after it on GitHub's runner.
- Read-only on Betfair: a login and the Historic Data API, no orders. The login is the trading account's
  (``BETFAIR_USERNAME``, ``BETFAIR_PASSWORD``, ``BETFAIR_APP_KEY``), never written anywhere.

    python -m football.warehouse historic --days 10 --max-minutes 20
    python -m football.warehouse historic --probe          # the login, what the account holds, yesterday's count
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

log = logging.getLogger(__name__)

SPORT = "Soccer"
MARKETS = ["MATCH_ODDS", "OVER_UNDER_25", "ASIAN_HANDICAP"]
SOURCE = "historic_api"


def months(d_from: date, d_to: date) -> list[tuple[date, date]]:
    """The calendar months the range touches, each clipped to the range (the API lists a range of days)."""
    out, d = [], d_from
    while d <= d_to:
        nxt = (d.replace(day=1) + timedelta(days=32)).replace(day=1)
        out.append((d, min(d_to, nxt - timedelta(days=1))))
        d = nxt
    return out


API_ROOT = "/xds_nfs/edp_processed/"


def held_raw(store, d_from: date, d_to: date) -> set[str]:
    """The raw keys the store holds for the months of the range (one listing a month, not the whole archive), under
    ``betfair_historic.raw_key``'s naming of the API's paths."""
    from betfair_historic import raw_key
    have: set[str] = set()
    for a, _ in months(d_from, d_to):
        have |= set(store.listing(raw_key(SPORT, f"{API_ROOT}BASIC/{a.year}/{a.strftime('%b')}/")))
    return have


def held_months(hd, sport: str = SPORT) -> set[str] | None:
    """The months ('YYYY-MM') the account holds of ``sport`` on the BASIC plan (``GetMyData``: one entry a month the
    account has "bought", at no charge, on historicdata.betfair.com). Listing a month the account does not hold is a
    400 Bad Request (the first nightly run, 25 Nov 2026). None when the service does not say."""
    from betfair_historic import PLAN
    try:
        mine = hd.my_data()
    except Exception as exc:                                   # reported; the listing will show what is refused
        log.warning("GetMyData not read (%s)", exc)
        return None
    out = {str(d.get("forDate", ""))[:7] for d in mine if d.get("sport") == sport and d.get("plan") == PLAN}
    return {m for m in out if len(m) == 7}


def update(store, hd, days: int = 10, today: date | None = None, max_minutes: float | None = None,
           workers: int = 4) -> dict:
    """Fetch the files of the last ``days`` days the store lacks and add them to bronze. ``hd`` is a
    ``betfair_historic.Historic`` (or anything with ``files(body)`` and ``download(path)``)."""
    from betfair_historic import Historic, raw_key
    from football.warehouse.bronze import Ingest
    from sources.common import Budget
    today = today or date.today()
    d_to, d_from = today - timedelta(days=1), today - timedelta(days=days)
    budget = Budget(max_minutes)
    have = held_raw(store, d_from, d_to)
    out = {"from": str(d_from), "to": str(d_to), "listed": 0, "held": 0, "fetched": 0, "failed": 0,
           "stopped": False, "by_day": {}}
    ing = Ingest(store, SOURCE)
    held = held_months(hd) if hasattr(hd, "my_data") else None
    out["months_held"] = sorted(held)[-3:] if held else held
    out["not_held"] = []
    for a, b in months(d_from, d_to):
        if held is not None and a.strftime("%Y-%m") not in held:
            out["not_held"].append(a.strftime("%Y-%m"))
            continue
        paths = hd.files(Historic.body(SPORT, a, b, MARKETS, []))
        out["listed"] += len(paths)
        todo = [p for p in paths if raw_key(SPORT, p) not in have]
        out["held"] += len(paths) - len(todo)

        def one(p):
            if not budget.left():
                return p, None
            try:
                return p, hd.download(p)
            except Exception as exc:                          # reported; the next run tries again
                log.warning("%s not fetched (%s)", p, exc)
                return p, b""
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for p, data in ex.map(one, todo):
                if data is None:
                    out["stopped"] = True
                    continue
                if not data:
                    out["failed"] += 1
                    continue
                store.put(raw_key(SPORT, p), data)
                ing.add(p, data)
                out["fetched"] += 1
                day = "/".join(p.split("/BASIC/")[-1].split("/")[:3])
                out["by_day"][day] = out["by_day"].get(day, 0) + 1
    out["bronze"] = ing.close()
    return out


def probe(hd, today: date | None = None) -> dict:
    """The login works, what the account holds, and how many soccer files the API lists for yesterday."""
    from betfair_historic import Historic
    today = today or date.today()
    y = today - timedelta(days=1)
    months_held = held_months(hd) or set()
    held = {"soccer_months_held": len(months_held), "latest": sorted(months_held)[-3:]}
    n = len(hd.files(Historic.body(SPORT, y, y, MARKETS, [])))
    return {"login": "ok", "held": held, "soccer_files_listed_for": str(y), "listed": n}
