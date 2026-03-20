"""Streak and drawdown tracking for settled bets."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class DrawdownTracker:
    """Processes settlement records and computes streak / drawdown metrics.

    Each settlement dict must contain at least:
        - result: "WON", "LOST", or "VOID"
        - net_pnl: float (profit or loss after commission)
        - date: str (YYYY-MM-DD)
    """

    settlements: list[dict] = field(default_factory=list)

    # ── internal state ────────────────────────────────────────────
    _computed: bool = field(default=False, repr=False)

    # streaks
    _current_streak_type: str = field(default="", repr=False)
    _current_streak_len: int = field(default=0, repr=False)
    _longest_win_streak: int = field(default=0, repr=False)
    _longest_lose_streak: int = field(default=0, repr=False)

    # drawdown
    _peak_bank: float = field(default=0.0, repr=False)
    _max_drawdown_abs: float = field(default=0.0, repr=False)
    _max_drawdown_pct: float = field(default=0.0, repr=False)
    _current_drawdown_abs: float = field(default=0.0, repr=False)
    _current_drawdown_pct: float = field(default=0.0, repr=False)

    # cumulative
    _cumulative_pnl: float = field(default=0.0, repr=False)
    _total_won: int = field(default=0, repr=False)
    _total_lost: int = field(default=0, repr=False)

    # daily aggregates (date -> total net_pnl)
    _daily_pnl: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.settlements:
            self._compute()

    # ── core computation ──────────────────────────────────────────

    def _compute(self) -> None:
        """Walk through settlements and compute all metrics."""
        if not self.settlements:
            self._computed = True
            return

        daily: dict[str, float] = defaultdict(float)
        cumulative = 0.0
        peak = 0.0
        max_dd_abs = 0.0
        max_dd_pct = 0.0

        current_type = ""
        current_len = 0
        longest_win = 0
        longest_lose = 0
        total_won = 0
        total_lost = 0

        for s in self.settlements:
            result = s.get("result", "")
            net_pnl = float(s.get("net_pnl", 0.0))
            date = s.get("date", "")

            # Skip void bets for streak tracking
            if result == "VOID":
                continue

            # Daily aggregate
            daily[date] += net_pnl

            # Cumulative P&L
            cumulative += net_pnl

            # Track peak and drawdown
            if cumulative > peak:
                peak = cumulative
            dd_abs = peak - cumulative
            dd_pct = (dd_abs / peak * 100) if peak > 0 else 0.0
            if dd_abs > max_dd_abs:
                max_dd_abs = dd_abs
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

            # Streak tracking
            if result == "WON":
                total_won += 1
                if current_type == "WON":
                    current_len += 1
                else:
                    current_type = "WON"
                    current_len = 1
                longest_win = max(longest_win, current_len)
            elif result == "LOST":
                total_lost += 1
                if current_type == "LOST":
                    current_len += 1
                else:
                    current_type = "LOST"
                    current_len = 1
                longest_lose = max(longest_lose, current_len)

        # Store results
        self._daily_pnl = dict(daily)
        self._cumulative_pnl = cumulative
        self._peak_bank = peak
        self._max_drawdown_abs = max_dd_abs
        self._max_drawdown_pct = max_dd_pct
        self._current_drawdown_abs = peak - cumulative
        self._current_drawdown_pct = (
            (peak - cumulative) / peak * 100 if peak > 0 else 0.0
        )
        self._current_streak_type = current_type
        self._current_streak_len = current_len
        self._longest_win_streak = longest_win
        self._longest_lose_streak = longest_lose
        self._total_won = total_won
        self._total_lost = total_lost
        self._computed = True

    # ── public properties ─────────────────────────────────────────

    @property
    def current_streak(self) -> tuple[str, int]:
        """Return (type, length) of current streak, e.g. ('LOST', 5)."""
        if not self._computed:
            self._compute()
        return self._current_streak_type, self._current_streak_len

    @property
    def longest_win_streak(self) -> int:
        if not self._computed:
            self._compute()
        return self._longest_win_streak

    @property
    def longest_lose_streak(self) -> int:
        if not self._computed:
            self._compute()
        return self._longest_lose_streak

    @property
    def max_drawdown_abs(self) -> float:
        if not self._computed:
            self._compute()
        return round(self._max_drawdown_abs, 2)

    @property
    def max_drawdown_pct(self) -> float:
        if not self._computed:
            self._compute()
        return round(self._max_drawdown_pct, 2)

    @property
    def current_drawdown_abs(self) -> float:
        if not self._computed:
            self._compute()
        return round(self._current_drawdown_abs, 2)

    @property
    def current_drawdown_pct(self) -> float:
        if not self._computed:
            self._compute()
        return round(self._current_drawdown_pct, 2)

    @property
    def recovery_factor(self) -> float:
        """Net profit divided by max drawdown. Higher is better."""
        if not self._computed:
            self._compute()
        if self._max_drawdown_abs == 0:
            return 0.0
        return round(self._cumulative_pnl / self._max_drawdown_abs, 2)

    @property
    def sharpe_ratio(self) -> float:
        """Sharpe-like ratio: mean daily PnL / std daily PnL."""
        if not self._computed:
            self._compute()
        if not self._daily_pnl:
            return 0.0
        values = list(self._daily_pnl.values())
        n = len(values)
        if n < 2:
            return 0.0
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = variance ** 0.5
        if std == 0:
            return 0.0
        return round(mean / std, 2)

    # ── alerts ────────────────────────────────────────────────────

    def check_alerts(self) -> list[str]:
        """Return alert messages for concerning patterns."""
        if not self._computed:
            self._compute()

        alerts: list[str] = []

        # Losing streak >= 8
        streak_type, streak_len = self.current_streak
        if streak_type == "LOST" and streak_len >= 8:
            alerts.append(
                f"ALERT: Losing streak of {streak_len} bets "
                f"(threshold: 8)"
            )

        # Current drawdown > 30% of peak
        if self._current_drawdown_pct > 30:
            alerts.append(
                f"ALERT: Current drawdown {self._current_drawdown_pct:.1f}% "
                f"exceeds 30% of peak bank"
            )

        # Daily loss exceeds £100
        for date_str, daily_total in self._daily_pnl.items():
            if daily_total < -100:
                alerts.append(
                    f"ALERT: Daily loss on {date_str} was "
                    f"£{abs(daily_total):.2f} (threshold: £100)"
                )

        # Win rate below 25% over last 50 bets
        non_void = [
            s for s in self.settlements if s.get("result") in ("WON", "LOST")
        ]
        last_50 = non_void[-50:] if len(non_void) >= 50 else non_void
        if last_50:
            wins = sum(1 for s in last_50 if s["result"] == "WON")
            win_rate = wins / len(last_50) * 100
            if win_rate < 25 and len(non_void) >= 50:
                alerts.append(
                    f"ALERT: Win rate {win_rate:.1f}% over last "
                    f"{len(last_50)} bets is below 25%"
                )

        return alerts

    # ── summary ───────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a clean dict of all drawdown and streak metrics."""
        if not self._computed:
            self._compute()

        streak_type, streak_len = self.current_streak
        total_bets = self._total_won + self._total_lost
        win_rate = (
            round(self._total_won / total_bets * 100, 1) if total_bets else 0.0
        )

        return {
            "total_bets": total_bets,
            "total_won": self._total_won,
            "total_lost": self._total_lost,
            "win_rate_pct": win_rate,
            "net_pnl": round(self._cumulative_pnl, 2),
            "current_streak_type": streak_type,
            "current_streak_length": streak_len,
            "longest_win_streak": self._longest_win_streak,
            "longest_lose_streak": self._longest_lose_streak,
            "peak_bank": round(self._peak_bank, 2),
            "max_drawdown_abs": self.max_drawdown_abs,
            "max_drawdown_pct": self.max_drawdown_pct,
            "current_drawdown_abs": self.current_drawdown_abs,
            "current_drawdown_pct": self.current_drawdown_pct,
            "recovery_factor": self.recovery_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "alerts": self.check_alerts(),
        }


def analyze_drawdown(settlements: list[dict]) -> dict:
    """Convenience function: analyze settlements and return summary dict.

    Args:
        settlements: List of dicts, each containing at least
            ``result``, ``net_pnl``, and ``date``.

    Returns:
        Dict with all streak, drawdown, and alert metrics.
    """
    tracker = DrawdownTracker(settlements=settlements)
    return tracker.summary()
