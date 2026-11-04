"""Betfair's charges as the trader counts them (build spec 7). Figures from the research brief, to be confirmed
against the account before the live phase:

- commission: the account's Market Base Rate on each market's net winnings (nothing on a market lost);
- transaction charge: GBP 0.002 a transaction above 5,000 in a clock hour, failed ones included;
- Expert Fee: the rolling gross P&L over 52 active weeks against the GBP 25k (20%) and GBP 100k (40%) bands. The
  exact buffer formula is an open question (build spec 10), so the tracker reports the bands reached, not a fee.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter

TX_FREE_PER_HOUR = 5000
TX_CHARGE = 0.002
EXPERT_BANDS = ((25_000.0, 0.20), (100_000.0, 0.40))


def commission(net_market_profit: float, rate: float) -> float:
    return max(0.0, net_market_profit) * rate


class Transactions:
    """Every order placed or cancelled (a transaction to Betfair), counted by UTC clock hour."""

    def __init__(self):
        self.by_hour: Counter = Counter()

    def add(self, when: dt.datetime, n: int = 1) -> None:
        self.by_hour[when.strftime("%Y-%m-%dT%H")] += n

    def max_hour(self) -> int:
        return max(self.by_hour.values()) if self.by_hour else 0

    def charge(self) -> float:
        return sum(max(0, n - TX_FREE_PER_HOUR) * TX_CHARGE for n in self.by_hour.values())


def expert_fee_bands(weekly_gross: list[tuple[str, float]]) -> dict:
    """Rolling gross P&L over the last 52 active weeks (weeks with any bet), and the bands it has reached."""
    active = [g for _, g in sorted(weekly_gross) if g != 0][-52:]
    total = float(sum(active))
    return {"rolling_52_active_weeks": total, "weeks": len(active),
            "bands_reached": [f"GBP {int(t):,} ({int(r * 100)}%)" for t, r in EXPERT_BANDS if total >= t],
            "fee": "not computed: the buffer formula is unconfirmed (build spec 10)"}
