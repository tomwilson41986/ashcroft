"""Tests for streak and drawdown tracking."""

import pytest

from ultra_betting.reporting.drawdown import DrawdownTracker, analyze_drawdown


def _settlement(result: str, net_pnl: float, date: str = "2026-03-01") -> dict:
    """Helper to build a minimal settlement dict."""
    return {"result": result, "net_pnl": net_pnl, "date": date}


# ── Winning streak ────────────────────────────────────────────────


class TestWinningStreak:
    def test_simple_winning_streak(self):
        settlements = [
            _settlement("WON", 10),
            _settlement("WON", 15),
            _settlement("WON", 20),
        ]
        tracker = DrawdownTracker(settlements=settlements)
        assert tracker.current_streak == ("WON", 3)
        assert tracker.longest_win_streak == 3

    def test_winning_streak_broken_by_loss(self):
        settlements = [
            _settlement("WON", 10),
            _settlement("WON", 15),
            _settlement("LOST", -5),
            _settlement("WON", 10),
        ]
        tracker = DrawdownTracker(settlements=settlements)
        assert tracker.current_streak == ("WON", 1)
        assert tracker.longest_win_streak == 2

    def test_void_does_not_break_streak(self):
        settlements = [
            _settlement("WON", 10),
            _settlement("VOID", 0),
            _settlement("WON", 15),
        ]
        tracker = DrawdownTracker(settlements=settlements)
        assert tracker.current_streak == ("WON", 2)


# ── Losing streak ────────────────────────────────────────────────


class TestLosingStreak:
    def test_simple_losing_streak(self):
        settlements = [
            _settlement("LOST", -10),
            _settlement("LOST", -10),
            _settlement("LOST", -10),
            _settlement("LOST", -10),
            _settlement("LOST", -10),
        ]
        tracker = DrawdownTracker(settlements=settlements)
        assert tracker.current_streak == ("LOST", 5)
        assert tracker.longest_lose_streak == 5

    def test_losing_streak_broken_by_win(self):
        settlements = [
            _settlement("LOST", -10),
            _settlement("LOST", -10),
            _settlement("LOST", -10),
            _settlement("WON", 30),
        ]
        tracker = DrawdownTracker(settlements=settlements)
        assert tracker.current_streak == ("WON", 1)
        assert tracker.longest_lose_streak == 3

    def test_multiple_losing_streaks_tracks_longest(self):
        settlements = [
            _settlement("LOST", -10),
            _settlement("LOST", -10),
            _settlement("WON", 30),
            _settlement("LOST", -10),
            _settlement("LOST", -10),
            _settlement("LOST", -10),
            _settlement("WON", 30),
        ]
        tracker = DrawdownTracker(settlements=settlements)
        assert tracker.longest_lose_streak == 3


# ── Max drawdown ─────────────────────────────────────────────────


class TestMaxDrawdown:
    def test_no_drawdown_on_pure_wins(self):
        settlements = [
            _settlement("WON", 10),
            _settlement("WON", 20),
            _settlement("WON", 15),
        ]
        tracker = DrawdownTracker(settlements=settlements)
        assert tracker.max_drawdown_abs == 0.0
        assert tracker.max_drawdown_pct == 0.0
        assert tracker.current_drawdown_abs == 0.0

    def test_drawdown_after_losses(self):
        settlements = [
            _settlement("WON", 100, "2026-03-01"),
            _settlement("LOST", -30, "2026-03-02"),
            _settlement("LOST", -20, "2026-03-02"),
        ]
        tracker = DrawdownTracker(settlements=settlements)
        # Peak was 100, dropped to 50, drawdown = 50
        assert tracker.max_drawdown_abs == 50.0
        assert tracker.max_drawdown_pct == 50.0  # 50/100 * 100
        assert tracker.current_drawdown_abs == 50.0

    def test_drawdown_recovery(self):
        settlements = [
            _settlement("WON", 100, "2026-03-01"),
            _settlement("LOST", -40, "2026-03-02"),
            _settlement("WON", 60, "2026-03-03"),
        ]
        tracker = DrawdownTracker(settlements=settlements)
        # Peak was 100, dropped to 60 (dd=40), then recovered to 120 (new peak)
        assert tracker.max_drawdown_abs == 40.0
        assert tracker.current_drawdown_abs == 0.0

    def test_recovery_factor(self):
        settlements = [
            _settlement("WON", 100, "2026-03-01"),
            _settlement("LOST", -40, "2026-03-02"),
            _settlement("WON", 80, "2026-03-03"),
        ]
        tracker = DrawdownTracker(settlements=settlements)
        # Net PnL = 140, max drawdown = 40
        assert tracker.recovery_factor == 3.5

    def test_recovery_factor_zero_drawdown(self):
        settlements = [_settlement("WON", 10)]
        tracker = DrawdownTracker(settlements=settlements)
        assert tracker.recovery_factor == 0.0


# ── Alerts ───────────────────────────────────────────────────────


class TestAlerts:
    def test_losing_streak_alert(self):
        settlements = [_settlement("LOST", -5) for _ in range(10)]
        tracker = DrawdownTracker(settlements=settlements)
        alerts = tracker.check_alerts()
        assert any("Losing streak" in a for a in alerts)

    def test_no_alert_for_short_losing_streak(self):
        settlements = [_settlement("LOST", -5) for _ in range(5)]
        tracker = DrawdownTracker(settlements=settlements)
        alerts = tracker.check_alerts()
        assert not any("Losing streak" in a for a in alerts)

    def test_drawdown_percentage_alert(self):
        # Build: big win then bigger losses to trigger >30% drawdown
        settlements = [
            _settlement("WON", 100, "2026-03-01"),
            _settlement("LOST", -50, "2026-03-02"),
        ]
        tracker = DrawdownTracker(settlements=settlements)
        alerts = tracker.check_alerts()
        assert any("drawdown" in a.lower() for a in alerts)

    def test_daily_loss_alert(self):
        settlements = [
            _settlement("LOST", -60, "2026-03-01"),
            _settlement("LOST", -60, "2026-03-01"),
        ]
        tracker = DrawdownTracker(settlements=settlements)
        alerts = tracker.check_alerts()
        assert any("Daily loss" in a for a in alerts)

    def test_no_daily_loss_alert_under_threshold(self):
        settlements = [_settlement("LOST", -50, "2026-03-01")]
        tracker = DrawdownTracker(settlements=settlements)
        alerts = tracker.check_alerts()
        assert not any("Daily loss" in a for a in alerts)

    def test_win_rate_alert(self):
        # 10 wins + 50 losses = 60 bets, win rate ~16.7%
        settlements = (
            [_settlement("WON", 10) for _ in range(10)]
            + [_settlement("LOST", -5) for _ in range(50)]
        )
        tracker = DrawdownTracker(settlements=settlements)
        alerts = tracker.check_alerts()
        assert any("Win rate" in a for a in alerts)

    def test_no_win_rate_alert_under_50_bets(self):
        # Even with low win rate, don't alert if < 50 bets
        settlements = [_settlement("LOST", -5) for _ in range(30)]
        tracker = DrawdownTracker(settlements=settlements)
        alerts = tracker.check_alerts()
        assert not any("Win rate" in a for a in alerts)


# ── Empty settlements ────────────────────────────────────────────


class TestEmptySettlements:
    def test_empty_list(self):
        tracker = DrawdownTracker(settlements=[])
        assert tracker.current_streak == ("", 0)
        assert tracker.longest_win_streak == 0
        assert tracker.longest_lose_streak == 0
        assert tracker.max_drawdown_abs == 0.0
        assert tracker.recovery_factor == 0.0
        assert tracker.sharpe_ratio == 0.0

    def test_empty_summary(self):
        result = analyze_drawdown([])
        assert result["total_bets"] == 0
        assert result["net_pnl"] == 0.0
        assert result["alerts"] == []

    def test_void_only(self):
        settlements = [_settlement("VOID", 0), _settlement("VOID", 0)]
        tracker = DrawdownTracker(settlements=settlements)
        assert tracker.current_streak == ("", 0)
        assert tracker.longest_win_streak == 0


# ── Mixed results ────────────────────────────────────────────────


class TestMixedResults:
    def test_realistic_sequence(self):
        settlements = [
            _settlement("LOST", -10, "2026-03-01"),
            _settlement("LOST", -10, "2026-03-01"),
            _settlement("WON", 25, "2026-03-01"),
            _settlement("LOST", -10, "2026-03-02"),
            _settlement("WON", 30, "2026-03-02"),
            _settlement("WON", 15, "2026-03-02"),
            _settlement("LOST", -10, "2026-03-03"),
        ]
        tracker = DrawdownTracker(settlements=settlements)
        summary = tracker.summary()

        assert summary["total_bets"] == 7
        assert summary["total_won"] == 3
        assert summary["total_lost"] == 4
        assert summary["net_pnl"] == 30.0  # -10-10+25-10+30+15-10 = 30
        assert summary["current_streak_type"] == "LOST"
        assert summary["current_streak_length"] == 1
        assert summary["longest_win_streak"] == 2
        assert summary["longest_lose_streak"] == 2

    def test_analyze_drawdown_function(self):
        settlements = [
            _settlement("WON", 50, "2026-03-01"),
            _settlement("LOST", -20, "2026-03-02"),
        ]
        result = analyze_drawdown(settlements)
        assert isinstance(result, dict)
        assert result["total_bets"] == 2
        assert result["net_pnl"] == 30.0
        assert result["max_drawdown_abs"] == 20.0
        assert "alerts" in result

    def test_sharpe_ratio_computed(self):
        settlements = [
            _settlement("WON", 20, "2026-03-01"),
            _settlement("WON", 30, "2026-03-02"),
            _settlement("LOST", -10, "2026-03-03"),
        ]
        tracker = DrawdownTracker(settlements=settlements)
        # Should be a finite number with 3 distinct days
        assert tracker.sharpe_ratio != 0.0

    def test_summary_contains_all_keys(self):
        settlements = [_settlement("WON", 10)]
        result = analyze_drawdown(settlements)
        expected_keys = {
            "total_bets", "total_won", "total_lost", "win_rate_pct",
            "net_pnl", "current_streak_type", "current_streak_length",
            "longest_win_streak", "longest_lose_streak", "peak_bank",
            "max_drawdown_abs", "max_drawdown_pct", "current_drawdown_abs",
            "current_drawdown_pct", "recovery_factor", "sharpe_ratio",
            "alerts",
        }
        assert set(result.keys()) == expected_keys
