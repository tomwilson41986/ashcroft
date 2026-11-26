"""A read-only Betfair client for the all-sports census and recorder, paced under Betfair's request limits.

Only the listing calls below exist on it: no order, no account call. It is its own module (not trading.exchange) so
nothing in the live horse system changes for the other sports (the owner's rule, 2 Nov 2026).

Pacing: at most ``PER_SECOND`` calls a second (the same ceiling the football feed keeps, football/exchange_feed.py
Throttle) and every catalogue call asks for at most ``MAX_RESULTS`` markets at a weight of 1 a market (the
MARKET_DESCRIPTION projection), so each call stays at Betfair's 200 weight points.
"""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)

READS = ("listEventTypes", "listMarketTypes", "listMarketCatalogue", "listCountries", "listCompetitions")
PER_SECOND = 5.0
MAX_RESULTS = 200                      # weight 1 a market (MARKET_DESCRIPTION; the rest weigh 0): 200 points a call
PROJECTION = ["EVENT", "EVENT_TYPE", "COMPETITION", "MARKET_START_TIME", "MARKET_DESCRIPTION"]
SESSION_HOURS = 4.0                    # log in again after this long, before Betfair's session can lapse


class Reader:
    """Listing calls only, paced, with one login retry on a lost session. Thread-safe: the recorder's shards share
    one reader and one session."""

    def __init__(self, client=None, per_second: float = PER_SECOND, clock=time.monotonic, sleep=time.sleep):
        if client is None:
            from betfair_client import BetfairClient
            client = BetfairClient()
        self.client = client
        self.gap = 1.0 / per_second
        self.clock, self.sleep = clock, sleep
        self.last = None
        self.logged_in_at = None
        self.calls = 0
        self.lock = threading.Lock()

    def login(self) -> None:
        with self.lock:
            self.client.login()
            self.logged_in_at = self.clock()

    def credentials(self) -> tuple[str, str]:
        """The app key and a current session token, for the stream's authentication message."""
        stale = self.logged_in_at is None or self.clock() - self.logged_in_at > SESSION_HOURS * 3600
        if stale or not self.client.session_token:
            self.login()
        return self.client.app_key, self.client.session_token

    def _pace(self) -> None:
        with self.lock:
            now = self.clock()
            if self.last is not None and now - self.last < self.gap:
                self.sleep(self.gap - (now - self.last))
                now = self.clock()
            self.last = now
            self.calls += 1

    def read(self, method: str, params: dict):
        if method not in READS:
            raise ValueError(f"{method}: this reader only lists markets")
        self._pace()
        try:
            return self.client._api_call(method, params)
        except Exception as exc:
            text = str(exc)
            if "INVALID_SESSION" in text or "NO_SESSION" in text or "Not logged in" in text:
                log.warning("Betfair session lost (%s); logging in again", text[:120])
                self.login()
                self._pace()
                return self.client._api_call(method, params)
            raise
