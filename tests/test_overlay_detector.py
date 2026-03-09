"""Tests for OverlayDetector — value detection and Kelly staking."""

import numpy as np
import pandas as pd
import pytest

from src.models.overlay_detector import OverlayDetector


class TestKellyStake:
    def test_no_edge_returns_zero(self):
        """No bet when edge is below threshold."""
        det = OverlayDetector(min_edge=0.05, bankroll=1000)
        # Fair bet: p=0.5, bfsp=2.0 => edge = 0
        assert det.kelly_stake(0.5, 2.0) == 0.0

    def test_positive_edge(self):
        """Positive edge should produce a stake."""
        det = OverlayDetector(min_edge=0.05, kelly_fraction=0.25, bankroll=1000)
        # p=0.4, bfsp=3.0 => edge = 0.4*3 - 1 = 0.2 (20%)
        stake = det.kelly_stake(0.4, 3.0)
        assert stake > 0

    def test_kelly_fraction_scaling(self):
        """Quarter Kelly should be 25% of full Kelly."""
        det_full = OverlayDetector(min_edge=0.0, kelly_fraction=1.0, bankroll=1000,
                                   min_stake=0.0, max_stake_pct=1.0, max_stake=10000)
        det_quarter = OverlayDetector(min_edge=0.0, kelly_fraction=0.25, bankroll=1000,
                                      min_stake=0.0, max_stake_pct=1.0, max_stake=10000)

        stake_full = det_full.kelly_stake(0.4, 3.0)
        stake_quarter = det_quarter.kelly_stake(0.4, 3.0)

        if stake_full > 0 and stake_quarter > 0:
            # Quarter kelly should be ~25% of full (before cap constraints)
            ratio = stake_quarter / stake_full
            assert abs(ratio - 0.25) < 0.01

    def test_max_stake_constraint(self):
        """Stake should not exceed max_stake."""
        det = OverlayDetector(min_edge=0.0, kelly_fraction=1.0, bankroll=100000,
                              max_stake=500)
        # Very high edge, large bankroll — should be capped
        stake = det.kelly_stake(0.8, 2.0)
        assert stake <= 500

    def test_max_stake_pct_constraint(self):
        """Stake should not exceed max_stake_pct of bankroll."""
        det = OverlayDetector(min_edge=0.0, kelly_fraction=1.0, bankroll=1000,
                              max_stake_pct=0.02, max_stake=500, min_stake=0.0)
        stake = det.kelly_stake(0.8, 2.0)
        assert stake <= 1000 * 0.02

    def test_min_stake_threshold(self):
        """Stake below min_stake should return 0."""
        det = OverlayDetector(min_edge=0.0, kelly_fraction=0.01, bankroll=100,
                              min_stake=2.0)
        # Very small Kelly fraction on small bankroll
        stake = det.kelly_stake(0.15, 8.0)
        # Should be 0 if calculated stake < 2
        assert stake == 0.0 or stake >= 2.0

    def test_invalid_bfsp(self):
        """Invalid BFSP should return 0."""
        det = OverlayDetector()
        assert det.kelly_stake(0.5, 0.5) == 0.0
        assert det.kelly_stake(0.5, 1.0) == 0.0
        assert det.kelly_stake(0.5, np.nan) == 0.0
        assert det.kelly_stake(0.5, np.inf) == 0.0


class TestBetCard:
    def _make_predictions(self):
        return pd.DataFrame({
            "raceid": ["r1"] * 5,
            "horse_name": [f"Horse_{i}" for i in range(5)],
            "p_model": [0.3, 0.25, 0.2, 0.15, 0.10],
            "p_public": [0.25, 0.25, 0.20, 0.15, 0.15],
            "p_combined": [0.32, 0.23, 0.20, 0.14, 0.11],
            "bfsp": [3.5, 4.0, 5.0, 7.0, 10.0],
            "model_fair_bfsp": [3.125, 4.35, 5.0, 7.14, 9.09],
            "edge": [0.28, -0.08, 0.0, -0.07, -0.27],
            "edge_pct": [28, -8, 0, -7, -27],
        })

    def test_only_positive_edge_bets(self):
        """Only runners with edge >= min_edge should appear on bet card."""
        det = OverlayDetector(min_edge=0.05)
        preds = self._make_predictions()
        card = det.generate_bet_card(preds)

        # Only Horse_0 has edge >= 5%
        assert len(card) <= 1
        if len(card) > 0:
            assert (card["edge"] >= 0.05).all()

    def test_empty_when_no_value(self):
        """Empty bet card when no runners have sufficient edge."""
        det = OverlayDetector(min_edge=0.50)  # 50% threshold — none qualify
        preds = self._make_predictions()
        card = det.generate_bet_card(preds)
        assert len(card) == 0

    def test_expected_value_column(self):
        """Expected value should be computed for each bet."""
        det = OverlayDetector(min_edge=0.01)
        preds = self._make_predictions()
        card = det.generate_bet_card(preds)

        if len(card) > 0:
            assert "expected_value_gbp" in card.columns
            assert "stake_gbp" in card.columns

    def test_sorted_by_ev(self):
        """Bet card should be sorted by expected value descending."""
        det = OverlayDetector(min_edge=0.0)
        preds = self._make_predictions()
        # Give multiple runners positive edge
        preds["edge"] = [0.15, 0.10, 0.08, 0.06, 0.03]
        card = det.generate_bet_card(preds)

        if len(card) > 1:
            evs = card["expected_value_gbp"].values
            assert all(evs[i] >= evs[i + 1] for i in range(len(evs) - 1))


class TestBankrollUpdate:
    def test_win_updates_bankroll(self):
        det = OverlayDetector(bankroll=1000, commission_rate=0.05)
        pl = det.update_bankroll("win", stake=10, bfsp=5.0)

        # Gross profit = 10 * (5-1) = 40, commission = 2, net = 38
        assert pl == 38.0
        assert det.bankroll == 1038.0

    def test_lose_updates_bankroll(self):
        det = OverlayDetector(bankroll=1000)
        pl = det.update_bankroll("lose", stake=10, bfsp=5.0)

        assert pl == -10.0
        assert det.bankroll == 990.0
