"""The limits every order passes before it is sent, and the day's running totals."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class DayState:
    turnover: float = 0.0                          # £ matched today
    bets: int = 0                                  # orders that matched
    settled_pnl: float = 0.0                       # £ after commission, races settled so far
    race_stake: dict = field(default_factory=dict) # market_id -> £ matched on the race
    stopped: str = ""                              # why no more bets today, once set

    def record(self, market_id: str, matched: float) -> None:
        if matched > 0:
            self.turnover += matched
            self.bets += 1
            self.race_stake[market_id] = self.race_stake.get(market_id, 0.0) + matched


def allowed_stake(stake: float, price: float, market_id: str, market_matched: float | None, back_size: float,
                  state: DayState, limits) -> tuple[float, str]:
    """The stake the limits allow (rounded down to the penny), or 0 and the reason.

    A market whose matched volume Betfair did not report (None) is not held to the volume floor;
    the share of the size on offer still caps the stake.
    """
    if state.stopped:
        return 0.0, state.stopped
    if state.settled_pnl <= -abs(limits.daily_stop_loss):
        state.stopped = f"daily stop-loss: settled {state.settled_pnl:+.2f}"
        return 0.0, state.stopped
    if state.bets >= limits.max_bets_per_day:
        return 0.0, "max bets per day"
    if not (limits.min_price <= price <= limits.max_price):
        return 0.0, f"price {price} outside {limits.min_price}-{limits.max_price}"
    if market_matched is not None and market_matched < limits.min_market_matched:
        return 0.0, f"market matched £{market_matched:,.0f} below £{limits.min_market_matched:,.0f}"
    s = min(stake, limits.max_stake,
            limits.max_race_stake - state.race_stake.get(market_id, 0.0),
            limits.max_daily_turnover - state.turnover,
            limits.liquidity_share * back_size)
    s = math.floor(max(s, 0.0) * 100) / 100
    if s < limits.min_stake:
        return 0.0, "below the minimum stake after limits"
    return s, ""
