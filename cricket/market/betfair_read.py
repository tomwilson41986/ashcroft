"""Betfair, read-only, for the cricket trader: the market-reading methods and nothing else.

Every call goes through ``ReadOnly.read``, which refuses any method not in ``READ_METHODS`` before a request is
made: no order can be placed, changed or cancelled from cricket code (tests/test_cricket_guards.py). The client is
the repo's own ``betfair_client.BetfairClient`` on the delayed application key (``BETFAIR_APP_KEY``); the live key
is never read here. Betfair blocks requests from US hosts: run it from the UK or Ireland (build spec 3).
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

CRICKET = "4"
READ_METHODS = frozenset({"listEventTypes", "listCompetitions", "listEvents", "listMarketTypes",
                          "listMarketCatalogue", "listMarketBook"})


class ReadOnly:
    def __init__(self, client=None):
        if client is None:
            from betfair_client import BetfairClient   # the delayed key, from the environment
            client = BetfairClient()
        self.client = client
        self.calls = 0

    def login(self) -> None:
        self.client.login()

    def read(self, method: str, params: dict):
        if method not in READ_METHODS:
            raise PermissionError(f"{method}: the cricket trader only reads markets and prices")
        self.calls += 1
        try:
            return self.client._api_call(method, params)
        except Exception as exc:                       # an expired session: log in once and retry
            text = str(exc)
            if "INVALID_SESSION" in text or "NO_SESSION" in text or "Not logged in" in text:
                log.warning("Betfair session lost (%s); logging in again", text[:120])
                self.client.login()
                return self.client._api_call(method, params)
            raise
