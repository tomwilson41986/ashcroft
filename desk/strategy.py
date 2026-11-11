"""What a desk strategy is: a pure function from a market as it stands to the orders it would like (intents).

A strategy never touches the exchange, the account or another strategy. The desk session (desk/session.py) hands it
a ``MarketView`` and what the desk already holds on that market; it returns ``Intent`` objects; the risk kernel
(desk/risk.py) decides what, if anything, is sent. ``on_fill`` tells it what was matched, so a strategy that keeps its
own tally (a day's liability, top-ups a runner) stays in step with what really happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from desk.account import Position
from desk.risk import Intent


@dataclass
class MarketView:
    """One market at one moment: the market, its book, each active runner's view (trading.strategy.RunnerView), the
    clock and the closing model's view of the race on demand (computed once, only if a strategy asks)."""
    market: object                        # trading.exchange.Market
    book: object                          # trading.exchange.Book
    runners: list                         # [trading.strategy.RunnerView]
    now: datetime
    book_age_s: float = 0.0
    closing_fn: Callable | None = None
    _closing: object = field(default=None, repr=False)
    _closing_done: bool = field(default=False, repr=False)

    @property
    def market_id(self) -> str:
        return self.market.market_id

    @property
    def minutes_to_off(self) -> float:
        return (self.market.start - self.now).total_seconds() / 60.0

    def closing(self):
        """The closing model's view of the race (research/arms/sessions.py ArmSession.closing: market now, model
        price, q_hat, sigma, draws of 1/BSP), or None when a runner lacks a price."""
        if not self._closing_done:
            self._closing = self.closing_fn(self.runners) if self.closing_fn else None
            self._closing_done = True
        return self._closing


class Strategy:
    """Subclass and set ``id`` to the registry id (desk/strategies/<sport>/<id>.json)."""

    id: str = ""

    def intents(self, view: MarketView, held: list[Position]) -> list[Intent]:
        raise NotImplementedError

    def on_fill(self, intent: Intent, matched: float, avg_price: float) -> None:
        return None
