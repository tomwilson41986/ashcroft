"""Accounting on Betfair's own record: orders to strategies by reference, positions, the worst case of a market,
settled P&L by strategy. Pure functions: no network, no S3 (the warden reads; this adds up).

The fields read are the ones the live traders already read from the same calls (trading/session.py sync and
settle_live; trading/exchange.py orders and cleared): marketId, selectionId, side, orderType, sizeMatched,
averagePriceMatched, bspLiability, customerOrderRef, betId (listCurrentOrders); profit, side, customerOrderRef
(listClearedOrders).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from desk.registry import Strategy, owner_of

UNATTRIBUTED = "(no reference)"     # an order with no customerOrderRef: placed by hand on the website
UNKNOWN = "(unknown reference)"     # a reference no registered strategy owns: never placed by a registered trader


@dataclass(frozen=True)
class Position:
    """One position's P&L on a win market: if its selection wins, and if it loses."""
    market_id: str
    selection_id: int
    if_wins: float
    if_loses: float


def worst_case(positions: list[Position]) -> float:
    """The worst P&L over the market's outcomes (each selection named here wins, or one not named does).
    One market's positions only; a negative number is a loss."""
    if not positions:
        return 0.0
    loses = sum(p.if_loses for p in positions)
    outcomes = [loses]                                       # a selection with no position wins
    for sid in {p.selection_id for p in positions}:
        outcomes.append(loses + sum(p.if_wins - p.if_loses for p in positions if p.selection_id == sid))
    return min(outcomes)


def _f(x) -> float:
    try:
        return float(x or 0.0)
    except (TypeError, ValueError):
        return 0.0


def order_position(o: dict) -> Position | None:
    """A current order's matched position. A lay at SP (MARKET_ON_CLOSE) counts its liability if the horse wins and
    nothing if it loses: its winnings depend on an SP not yet known, so the worst case stays a bound."""
    mid, sid, side = str(o.get("marketId")), int(_f(o.get("selectionId"))), o.get("side")
    if side == "LAY" and o.get("orderType") in ("MARKET_ON_CLOSE", "LIMIT_ON_CLOSE"):
        liab = _f(o.get("bspLiability"))
        return Position(mid, sid, -liab, 0.0) if liab > 0 else None
    size, price = _f(o.get("sizeMatched")), _f(o.get("averagePriceMatched"))
    if size <= 0 or price <= 1:
        return None
    if side == "BACK":
        return Position(mid, sid, size * (price - 1), -size)
    if side == "LAY":
        return Position(mid, sid, -size * (price - 1), size)
    return None


def who(o: dict, strategies: list[Strategy]) -> str:
    """The strategy id an order belongs to, or UNATTRIBUTED / UNKNOWN."""
    ref = str(o.get("customerOrderRef") or "")
    if not ref:
        return UNATTRIBUTED
    s = owner_of(ref, strategies)
    return s.id if s else UNKNOWN


def by_strategy(orders: list[dict], strategies: list[Strategy]) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for o in orders:
        out[who(o, strategies)].append(o)
    return dict(out)


def market_worst(orders: list[dict]) -> dict[str, float]:
    """Each market's worst case over every order on it (every strategy together, and orders placed by hand)."""
    by_market = defaultdict(list)
    for o in orders:
        p = order_position(o)
        if p is not None:
            by_market[p.market_id].append(p)
    return {m: worst_case(ps) for m, ps in by_market.items()}


def settled_by_strategy(cleared: list[dict], strategies: list[Strategy]) -> dict[str, dict]:
    """Betfair's settled profit (before commission, as listClearedOrders reports it) and bets, by strategy."""
    out = defaultdict(lambda: {"bets": 0, "profit": 0.0})
    for o in cleared:
        r = out[who(o, strategies)]
        r["bets"] += 1
        r["profit"] += _f(o.get("profit"))
    return {k: {"bets": v["bets"], "profit": round(v["profit"], 2)} for k, v in out.items()}
