"""Stakes (build spec 3, 7): fractional Kelly at one third, a cap on what one match can lose, a drawdown stop.

Kelly for a back at O (b = (O - 1)(1 - c) net odds): f = (p b - (1 - p)) / b of the bank staked. For a lay at O,
the bet is on the selection losing at net odds b' = (1 - c) / (O - 1), with the liability at risk: f = ((1 - p) b' -
p) / b' of the bank as liability. One third of that is used.

The match cap holds the worst outcome of the whole position on the match (every bet on every market of it, from
``exposure``) to ``match_cap`` of the bank. Trading stops once the bank falls ``drawdown_stop`` below its peak.
"""

from __future__ import annotations

from dataclasses import dataclass

from cricket.trading.signals import Signal


def kelly_back(p: float, price: float, c: float) -> float:
    b = (price - 1.0) * (1.0 - c)
    return max(0.0, (p * b - (1.0 - p)) / b) if b > 0 else 0.0


def kelly_lay(p: float, price: float, c: float) -> float:
    b = (1.0 - c) / (price - 1.0) if price > 1.0 else 0.0
    return max(0.0, ((1.0 - p) * b - p) / b) if b > 0 else 0.0


def worst_case(position: dict[int, float]) -> float:
    """The worst outcome of a match position (selection -> profit if that selection wins), as a loss >= 0."""
    return max(0.0, -min(position.values())) if position else 0.0


def position_after(position: dict[int, float], selections: list[int], sig: Signal, stake: float,
                   c: float = 0.0) -> dict[int, float]:
    """Profit by winning selection after adding a bet (commission left out: it only lowers winnings)."""
    pos = {s: position.get(s, 0.0) for s in selections}
    for s in selections:
        win = s == sig.selection_id
        if sig.side == "BACK":
            pos[s] += stake * (sig.price - 1.0) if win else -stake
        else:
            pos[s] += -stake * (sig.price - 1.0) if win else stake
    return pos


@dataclass
class Sizer:
    bank: float
    fraction: float = 1.0 / 3.0
    match_cap: float = 0.05
    min_stake: float = 2.0
    max_stake: float | None = None
    drawdown_stop: float = 0.20
    commission: float = 0.05
    peak: float | None = None

    def __post_init__(self):
        self.peak = self.bank if self.peak is None else self.peak

    @property
    def stopped(self) -> bool:
        return self.bank < self.peak * (1.0 - self.drawdown_stop)

    def update_bank(self, bank: float) -> None:
        self.bank = bank
        self.peak = max(self.peak, bank)

    def stake(self, sig: Signal, position: dict[int, float], selections: list[int]) -> float:
        """The backer's stake for the signal (for a lay, the stake laid), 0 when it should not be placed."""
        if self.stopped:
            return 0.0
        if sig.side == "BACK":
            stake = self.fraction * kelly_back(sig.p, sig.price, self.commission) * self.bank
        else:
            liability = self.fraction * kelly_lay(sig.p, sig.price, self.commission) * self.bank
            stake = liability / (sig.price - 1.0)
        if self.max_stake is not None:
            stake = min(stake, self.max_stake)
        cap = self.match_cap * self.bank
        # shrink until the match's worst case is inside the cap (a bet that reduces the worst case always fits)
        for _ in range(30):
            if stake < self.min_stake:
                return 0.0
            after = worst_case(position_after(position, selections, sig, stake))
            if after <= cap or after <= worst_case(position):
                return round(stake, 2)
            stake *= 0.8
        return 0.0
