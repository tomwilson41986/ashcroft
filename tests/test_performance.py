"""Tests for web.performance metrics computation."""

import pytest

from web.performance import compute_performance_metrics, compute_performance_summary


def _make_bet(status, net_pnl, stake=10.0, edge_pct=15.0, price=5.0,
              race_date="2026-03-10", track="Cheltenham"):
    return {
        "status": status,
        "net_pnl": net_pnl,
        "stake": stake,
        "edge_pct": edge_pct,
        "price": price,
        "race_date": race_date,
        "track": track,
    }


SAMPLE_BETS = [
    # Day 1 — Cheltenham
    _make_bet("WON", 40.0, stake=10, edge_pct=22, price=5.0, race_date="2026-03-01", track="Cheltenham"),
    _make_bet("LOST", -10.0, stake=10, edge_pct=8, price=2.5, race_date="2026-03-01", track="Cheltenham"),
    _make_bet("LOST", -10.0, stake=10, edge_pct=12, price=4.0, race_date="2026-03-01", track="Ascot"),
    # Day 2 — Kempton
    _make_bet("WON", 70.0, stake=10, edge_pct=18, price=8.0, race_date="2026-03-02", track="Kempton"),
    _make_bet("LOST", -10.0, stake=10, edge_pct=5, price=10.0, race_date="2026-03-02", track="Kempton"),
    # Day 3 — mixed
    _make_bet("LOST", -10.0, stake=10, edge_pct=7, price=3.5, race_date="2026-03-03", track="Ascot"),
    _make_bet("WON", 15.0, stake=10, edge_pct=14, price=2.5, race_date="2026-03-03", track="Cheltenham"),
    # PENDING — should be excluded
    _make_bet("PENDING", 0.0, stake=10, edge_pct=20, price=6.0, race_date="2026-03-04", track="York"),
]


class TestComputePerformanceMetrics:

    def test_empty_bets(self):
        result = compute_performance_metrics([])
        assert result["total_bets"] == 0
        assert result["roi_pct"] == 0.0
        assert result["roi_by_edge_band"] == []

    def test_pending_only(self):
        pending = [_make_bet("PENDING", 0)]
        result = compute_performance_metrics(pending)
        assert result["total_bets"] == 0

    def test_total_counts(self):
        result = compute_performance_metrics(SAMPLE_BETS)
        assert result["total_bets"] == 7
        assert result["wins"] == 3
        assert result["losses"] == 4

    def test_hit_rate(self):
        result = compute_performance_metrics(SAMPLE_BETS)
        # 3 / 7 = 42.86%
        assert result["hit_rate"] == pytest.approx(42.86, abs=0.01)

    def test_roi(self):
        result = compute_performance_metrics(SAMPLE_BETS)
        # total staked = 70, total pnl = 40-10-10+70-10-10+15 = 85
        assert result["total_staked"] == 70.0
        assert result["net_pnl"] == 85.0
        expected_roi = 85 / 70 * 100
        assert result["roi_pct"] == pytest.approx(expected_roi, abs=0.01)

    def test_roi_by_edge_band(self):
        result = compute_performance_metrics(SAMPLE_BETS)
        bands = {b["band"]: b for b in result["roi_by_edge_band"]}

        # 0-10%: edges 8, 5, 7 -> pnl -10, -10, -10 = -30, staked 30
        assert "0-10%" in bands
        assert bands["0-10%"]["bets"] == 3
        assert bands["0-10%"]["pnl"] == -30.0

        # 10-20%: edges 12, 18, 14 -> pnl -10, 70, 15 = 75, staked 30
        assert "10-20%" in bands
        assert bands["10-20%"]["bets"] == 3
        assert bands["10-20%"]["pnl"] == 75.0

        # 20%+: edge 22 -> pnl 40, staked 10
        assert "20%+" in bands
        assert bands["20%+"]["bets"] == 1
        assert bands["20%+"]["pnl"] == 40.0

    def test_roi_by_price_range(self):
        result = compute_performance_metrics(SAMPLE_BETS)
        ranges = {r["range"]: r for r in result["roi_by_price_range"]}

        # short (<3.0): prices 2.5, 2.5 -> pnl -10, 15
        assert "short (<3.0)" in ranges
        assert ranges["short (<3.0)"]["bets"] == 2
        assert ranges["short (<3.0)"]["pnl"] == 5.0

        # mid (3.0-8.0): prices 5.0, 4.0, 3.5 -> pnl 40, -10, -10
        assert "mid (3.0-8.0)" in ranges
        assert ranges["mid (3.0-8.0)"]["bets"] == 3
        assert ranges["mid (3.0-8.0)"]["pnl"] == 20.0

        # long (8.0+): prices 8.0, 10.0 -> pnl 70, -10
        assert "long (8.0+)" in ranges
        assert ranges["long (8.0+)"]["bets"] == 2
        assert ranges["long (8.0+)"]["pnl"] == 60.0

    def test_win_rate_by_venue(self):
        result = compute_performance_metrics(SAMPLE_BETS)
        venues = {v["venue"]: v for v in result["win_rate_by_venue"]}

        # Cheltenham: 3 bets, 2 won
        assert venues["Cheltenham"]["bets"] == 3
        assert venues["Cheltenham"]["won"] == 2
        assert venues["Cheltenham"]["win_rate"] == pytest.approx(66.67, abs=0.01)

        # Ascot: 2 bets, 0 won
        assert venues["Ascot"]["bets"] == 2
        assert venues["Ascot"]["won"] == 0

        # Kempton: 2 bets, 1 won
        assert venues["Kempton"]["bets"] == 2
        assert venues["Kempton"]["won"] == 1

    def test_avg_edge_winners_vs_losers(self):
        result = compute_performance_metrics(SAMPLE_BETS)
        # Winners: edges 22, 18, 14 -> avg 18.0
        assert result["avg_edge_winners"] == 18.0
        # Losers: edges 8, 12, 5, 7 -> avg 8.0
        assert result["avg_edge_losers"] == 8.0

    def test_profit_factor(self):
        result = compute_performance_metrics(SAMPLE_BETS)
        # gross wins = 40+70+15 = 125
        # gross losses = abs(-10-10-10-10) = 40
        assert result["profit_factor"] == pytest.approx(125 / 40, abs=0.01)

    def test_best_and_worst_day(self):
        result = compute_performance_metrics(SAMPLE_BETS)
        # Day 1: 40-10-10=20, Day 2: 70-10=60, Day 3: -10+15=5
        assert result["best_day"]["date"] == "2026-03-02"
        assert result["best_day"]["pnl"] == 60.0
        assert result["worst_day"]["date"] == "2026-03-03"
        assert result["worst_day"]["pnl"] == 5.0

    def test_rolling_7d_roi(self):
        result = compute_performance_metrics(SAMPLE_BETS)
        rolling = result["rolling_7d_roi"]
        # All 3 dates within 7 days of each other
        assert len(rolling) == 3
        dates = [r["date"] for r in rolling]
        assert dates == ["2026-03-01", "2026-03-02", "2026-03-03"]
        # Final entry should cover all bets: 85/70*100
        assert rolling[-1]["roi_pct"] == pytest.approx(85 / 70 * 100, abs=0.01)

    def test_rolling_30d_roi(self):
        result = compute_performance_metrics(SAMPLE_BETS)
        rolling = result["rolling_30d_roi"]
        assert len(rolling) == 3
        # 30-day window covers everything, same as 7d here
        assert rolling[-1]["roi_pct"] == pytest.approx(85 / 70 * 100, abs=0.01)


class TestComputePerformanceSummary:

    def test_empty_bets(self):
        result = compute_performance_summary([])
        assert result["total_bets"] == 0

    def test_summary_fields(self):
        result = compute_performance_summary(SAMPLE_BETS)
        assert result["total_bets"] == 7
        assert result["wins"] == 3
        assert result["hit_rate"] == pytest.approx(42.86, abs=0.01)
        assert result["roi_pct"] == pytest.approx(85 / 70 * 100, abs=0.01)
        assert result["net_pnl"] == 85.0
        assert result["profit_factor"] == pytest.approx(125 / 40, abs=0.01)

    def test_summary_has_no_breakdown_fields(self):
        """Summary should not include detailed breakdowns."""
        result = compute_performance_summary(SAMPLE_BETS)
        assert "roi_by_edge_band" not in result
        assert "roi_by_price_range" not in result
        assert "rolling_7d_roi" not in result

    def test_handles_none_values(self):
        """Bets with None values for numeric fields should not crash."""
        bets = [
            {"status": "WON", "net_pnl": None, "stake": None, "edge_pct": None, "price": None,
             "race_date": "2026-03-01", "track": "Test"},
            {"status": "LOST", "net_pnl": None, "stake": None, "edge_pct": None, "price": None,
             "race_date": "2026-03-01", "track": "Test"},
        ]
        result = compute_performance_summary(bets)
        assert result["total_bets"] == 2
        assert result["roi_pct"] == 0.0

    def test_all_winners(self):
        bets = [
            _make_bet("WON", 30.0, stake=10, edge_pct=15, price=4.0),
            _make_bet("WON", 20.0, stake=10, edge_pct=20, price=3.0),
        ]
        result = compute_performance_summary(bets)
        assert result["wins"] == 2
        assert result["hit_rate"] == 100.0
        # No losses -> profit_factor = 0 (division guard)
        assert result["profit_factor"] == 0.0

    def test_all_losers(self):
        bets = [
            _make_bet("LOST", -10.0, stake=10),
            _make_bet("LOST", -10.0, stake=10),
        ]
        result = compute_performance_summary(bets)
        assert result["wins"] == 0
        assert result["hit_rate"] == 0.0
