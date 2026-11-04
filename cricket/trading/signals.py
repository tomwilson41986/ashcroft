"""Trade signals (build spec 7): the calibrated model against the price after commission.

A back at price O on a selection the model gives probability p is worth, per unit staked, after commission c on
winnings: ``p (O - 1)(1 - c) - (1 - p)``. A lay at O, per unit of the backer's stake: ``(1 - p)(1 - c) - p (O - 1)``.
A signal is the best side and price where that value clears ``threshold``, and only where:

- the market is open and the model has a calibrated quote;
- the quote is not inside the quiet window after a wicket or a boundary (build spec 8: the first 8 seconds of prices
  after each are discarded);
- the model's p is inside ``p_band`` (Norton, Gray and Faff found the ODI edge only in 40-60% states: the band is
  set from the backtest, open by default);
- the back book percentage is under ``max_book`` (their spread rule; read here as the sum of 1 / best back price,
  an interpretation of the paper's "spread under 100.75%" not yet checked against it).
"""

from __future__ import annotations

from dataclasses import dataclass

from cricket.market.book import Book

COMMISSION = 0.05


def back_value(p: float, price: float, c: float = COMMISSION) -> float:
    return p * (price - 1.0) * (1.0 - c) - (1.0 - p)


def lay_value(p: float, price: float, c: float = COMMISSION) -> float:
    return (1.0 - p) * (1.0 - c) - p * (price - 1.0)


@dataclass
class Signal:
    market_id: str
    selection_id: int
    side: str            # BACK or LAY
    price: float
    value: float         # expected value a unit, after commission
    p: float


@dataclass
class Rules:
    threshold: float = 0.03
    commission: float = COMMISSION
    quiet_seconds: float = 8.0
    p_band: tuple[float, float] = (0.0, 1.0)
    max_book: float | None = None
    min_price: float = 1.01
    max_price: float = 50.0


def signals(p_by_selection: dict[int, float], book: Book, rules: Rules, seconds_since_event: float | None
            ) -> list[Signal]:
    """At most one signal a selection: the side with the higher value, if it clears the threshold."""
    if not book.open or not p_by_selection:
        return []
    if seconds_since_event is not None and seconds_since_event < rules.quiet_seconds:
        return []
    if rules.max_book is not None:
        bb = book.back_book()
        if bb is None or bb > rules.max_book:
            return []
    out = []
    for sel, p in p_by_selection.items():
        r = book.runners.get(sel)
        if r is None or p is None or not (rules.p_band[0] <= p <= rules.p_band[1]):
            continue
        best = None
        if r.back and rules.min_price <= r.back[0][0] <= rules.max_price:
            v = back_value(p, r.back[0][0], rules.commission)
            best = Signal(book.market_id, sel, "BACK", r.back[0][0], v, p)
        if r.lay and rules.min_price <= r.lay[0][0] <= rules.max_price:
            v = lay_value(p, r.lay[0][0], rules.commission)
            if best is None or v > best.value:
                best = Signal(book.market_id, sel, "LAY", r.lay[0][0], v, p)
        if best is not None and best.value >= rules.threshold:
            out.append(best)
    return out
