"""A settlement stand-in for line markets Cricsheet does not cover: the line's last in-play trade before the market
first suspends at its settlement point (the end of over n), from Betfair BASIC files.

A trade a ball or two before the over ends can sit a few runs from the result, so a bet is only settled by the
stand-in when the last trade is at least ``margin`` runs from the bet's line; closer than that it is left unsettled.
"""

from __future__ import annotations

from collections import defaultdict

from cricket.market.inplay_panel import read_market

RAW = "betfair_historic/raw/cricket/"


def raw_index(store) -> dict[str, list[str]]:
    by_event = defaultdict(list)
    for k in store.listing(RAW):
        by_event[k.split("/")[-2]].append(k)
    return by_event


def last_inplay_trade(store, by_event: dict, event_id: str, market_id: str) -> tuple[float | None, float | None]:
    """(the last in-play traded line before the market's first in-play suspension, minutes between that trade and
    the suspension)."""
    m = read_market([store.get(k) for k in by_event.get(str(event_id), [])], market_id)
    if m["inplay_at"] is None:
        return None, None
    susp = [t for t, st, ip in m["status"] if ip and st in ("SUSPENDED", "CLOSED") and t > m["inplay_at"]]
    if not susp:
        return None, None
    end = min(susp)
    ups = sorted((t, p) for (t, _), p in m["ltp"].items() if m["inplay_at"] <= t <= end)
    if not ups:
        return None, None
    return ups[-1][1], (end - ups[-1][0]).total_seconds() / 60
