"""Queue-aware fills for a resting order (the owner, 3 Nov 2026: "count a lay as matched only when the money traded
at our price exceeds what was queued ahead of us"), from the live-key record: the books (``books_greyhound_live.csv``)
and the money traded at each price (``traded_greyhound_live.csv``, greyhound/book_record.py ``TradedLadder``).

A lay at price P rests among the offers to backers (Betfair's "available to back" side) at P; a back at P among the
offers to layers ("available to lay"). When placed:

- the part the other side already offers at P or better is matched at once (taken);
- the rest joins the queue at P behind the money already offered there: none when P improves on the best offer, the
  size shown at P when P is one of the three prices shown, unknown (deeper than the book shows) otherwise.

After that, it is matched as money trades at P beyond the queue ahead (Betfair's traded volume at P, less its value
when the order was placed), and in full once money trades strictly beyond P (every order at P, ours too, was matched
first). Cancellations ahead of us are not seen, so the queue is never shortened by them: a conservative count.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Fill:
    matched: float = 0.0
    taken: float = 0.0
    queue_ahead: float | None = None
    full_at: pd.Timestamp | None = None


def queue_ahead(side: str, price: float, offers: list[tuple[float, float]]) -> float | None:
    """The money ahead of a new order at ``price`` on its side of the book. ``offers``: that side's shown prices and
    sizes, best first (a lay rests among the backers' offers, best = highest; a back among the layers', best =
    lowest). None when the price is deeper than the book shows."""
    if not offers:
        return 0.0
    best = offers[0][0]
    better = price > best + 1e-9 if side == "lay" else price < best - 1e-9
    if better:
        return 0.0
    for p, s in offers:
        if abs(p - price) < 1e-9:
            return float(s)
    return None


def fill(side: str, price: float, size: float, placed_at, book: dict, traded: pd.DataFrame,
         until=None, through_min: float = 0.0) -> Fill:
    """What an order of ``size`` at ``price`` placed at ``placed_at`` had matched by ``until``.

    ``book``: the runner's book when placed: ``back`` and ``lay`` as [(price, size)], best first (``back``: the prices
    a backer can take, the layers' offers; ``lay``: the prices a layer can take). ``traded``: the runner's rows of the
    traded ladder (polled_utc, price, traded: cumulative), any span covering the order's life. ``through_min``: the
    money that must trade beyond our price, at one price, before it counts as having gone through us (0: any)."""
    placed_at = pd.Timestamp(placed_at)
    until = pd.Timestamp(until) if until is not None else None
    take_side = book.get("lay" if side == "lay" else "back") or []      # what this order can take at once
    rest_side = book.get("back" if side == "lay" else "lay") or []      # the offers it queues among
    taken = 0.0
    for p, s in take_side:
        ok = p <= price + 1e-9 if side == "lay" else p >= price - 1e-9
        if not ok or taken >= size:
            break
        taken += min(float(s), size - taken)
    out = Fill(taken=round(taken, 2), matched=round(taken, 2))
    rest = size - taken
    if rest <= 1e-9:
        out.full_at = placed_at
        return out
    q = queue_ahead(side, price, rest_side)
    out.queue_ahead = q
    if traded is None or not len(traded):
        return out
    # the recorder stamps a read's books to the whole second and its traded money with the fraction too: floored, the
    # money of the read an order was priced on counts as already traded when it was placed (unfloored, it read as
    # traded after: money matched before the order existed credited to it, 4 Nov 2026)
    t = traded.assign(polled=pd.to_datetime(traded.polled_utc, utc=True, format="ISO8601").dt.floor("s")).sort_values(
        "polled")
    if placed_at.tzinfo is None:
        placed_at = placed_at.tz_localize("UTC")
    before = t[t.polled <= placed_at]
    after = t[t.polled > placed_at]
    if until is not None:
        until = until if until.tzinfo else until.tz_localize("UTC")
        after = after[after.polled <= until]
    if not len(after):
        return out
    base = before.groupby("price").traded.last()
    # through: money traded strictly beyond our price after we joined: the queue at our price, ours too, cleared
    beyond = after[after.price < price - 1e-9] if side == "lay" else after[after.price > price + 1e-9]
    if len(beyond):
        grew = beyond.traded.to_numpy() > beyond.price.map(base).fillna(0.0).to_numpy() + max(through_min, 1e-9)
        if grew.any():
            out.matched = round(size, 2)
            out.full_at = beyond.polled[grew].iloc[0]
            return out
    at = after[(after.price - price).abs() < 1e-9]
    if len(at) and q is not None:
        vol = at.traded - float(base.get(price, 0.0))
        got = (vol - q).clip(lower=0.0, upper=rest)
        out.matched = round(taken + float(got.max()), 2)
        if (got >= rest - 1e-9).any():
            out.full_at = at.polled[(got >= rest - 1e-9).to_numpy()].iloc[0]
    return out
