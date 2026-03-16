"""Tests for the rules engine."""

import pytest

from ultra_betting.rules.engine import RulesEngine
from ultra_betting.rules.edge import calculate_edge
from ultra_betting.rules.staking import calculate_stake, _kelly_criterion
from ultra_betting.data.schemas import Prediction, BetInstruction
from ultra_betting.betfair.pricing import MarketPrices


class TestEdgeCalculation:
    def test_back_positive_edge(self):
        """Live price > predicted = positive edge for backing."""
        edge = calculate_edge(predicted_bfsp=3.0, live_price=4.0, side="BACK")
        assert edge > 0
        assert abs(edge - 33.33) < 0.1

    def test_back_negative_edge(self):
        """Live price < predicted = negative edge for backing."""
        edge = calculate_edge(predicted_bfsp=4.0, live_price=3.0, side="BACK")
        assert edge < 0

    def test_lay_positive_edge(self):
        """Predicted > live = positive edge for laying."""
        edge = calculate_edge(predicted_bfsp=5.0, live_price=3.0, side="LAY")
        assert edge > 0

    def test_zero_price_returns_zero(self):
        assert calculate_edge(0, 3.0, "BACK") == 0.0
        assert calculate_edge(3.0, 0, "BACK") == 0.0


class TestStaking:
    def test_fixed_stake(self):
        stake = calculate_stake(0.3, 15.0, 4.0, {"method": "fixed", "fixed_stake": 10.0})
        assert stake == 10.0

    def test_percentage_bank(self):
        stake = calculate_stake(0.3, 15.0, 4.0, {
            "method": "percentage_bank",
            "starting_bank": 1000.0,
            "bank_percentage": 2.0,
        })
        assert stake == 20.0

    def test_kelly_criterion_basic(self):
        """50% chance at 3.0 = positive kelly."""
        k = _kelly_criterion(0.5, 3.0)
        assert k > 0
        # Kelly = (2*0.5 - 0.5) / 2 = 0.25
        assert abs(k - 0.25) < 0.01

    def test_kelly_no_edge(self):
        """Fair odds = zero kelly."""
        k = _kelly_criterion(0.5, 2.0)
        assert k == 0.0

    def test_kelly_negative_edge(self):
        """Negative edge = zero kelly."""
        k = _kelly_criterion(0.2, 2.0)
        assert k == 0.0

    def test_fractional_kelly(self):
        stake = calculate_stake(0.5, 20.0, 3.0, {
            "method": "fractional_kelly",
            "kelly_fraction": 0.25,
            "starting_bank": 1000.0,
        })
        assert stake > 0
        assert stake < 100  # Should be modest with quarter kelly


class TestRulesEngine:
    def _make_prediction(self, bfsp=3.0, prob=0.33, market_id="1.234", selection_id=12345):
        return Prediction(
            date="2026-03-16",
            venue="Cheltenham",
            race_time="14:30",
            market_id=market_id,
            selection_id=selection_id,
            runner_name="Test Horse",
            predicted_bfsp=bfsp,
            predicted_win_prob=prob,
        )

    def _make_prices(self, market_id="1.234", back=4.0, lay=4.2, total_matched=10000):
        return MarketPrices(
            market_id=market_id,
            runners=[{
                "selection_id": 12345,
                "best_back_price": back,
                "best_lay_price": lay,
                "last_traded_price": back,
            }],
            status="OPEN",
            total_matched=total_matched,
        )

    def test_generates_bet_with_edge(self):
        """Should generate a bet when there's sufficient edge."""
        engine = RulesEngine({
            "entry_conditions": {"min_edge_percent": 5.0},
            "side_logic": {"mode": "back"},
            "staking": {"method": "fixed", "fixed_stake": 10.0},
            "persistence": {"type": "LAPSE"},
        })
        pred = self._make_prediction(bfsp=3.0, prob=0.33)
        prices = {pred.market_id: self._make_prices(back=4.0)}

        instructions, skipped = engine.evaluate([pred], prices)
        assert len(instructions) == 1
        assert instructions[0].side == "BACK"
        assert instructions[0].stake == 10.0

    def test_skips_no_edge(self):
        """Should skip when edge is too low."""
        engine = RulesEngine({
            "entry_conditions": {"min_edge_percent": 50.0},
            "side_logic": {"mode": "back"},
            "staking": {"method": "fixed", "fixed_stake": 10.0},
            "persistence": {"type": "LAPSE"},
        })
        pred = self._make_prediction(bfsp=3.0, prob=0.33)
        prices = {pred.market_id: self._make_prices(back=3.5)}

        instructions, skipped = engine.evaluate([pred], prices)
        assert len(instructions) == 0
        assert len(skipped) == 1
        assert skipped[0].reason == "entry_condition_failed"

    def test_skips_closed_market(self):
        """Should skip when market is not OPEN."""
        engine = RulesEngine({
            "entry_conditions": {"min_edge_percent": 5.0},
            "side_logic": {"mode": "back"},
            "staking": {"method": "fixed", "fixed_stake": 10.0},
            "persistence": {"type": "LAPSE"},
        })
        pred = self._make_prediction()
        prices = {pred.market_id: MarketPrices(
            pred.market_id, [], status="CLOSED", total_matched=0
        )}

        instructions, skipped = engine.evaluate([pred], prices)
        assert len(instructions) == 0
        assert skipped[0].reason == "market_not_open"

    def test_skips_no_market(self):
        """Should skip when no prices available for market."""
        engine = RulesEngine({
            "entry_conditions": {},
            "side_logic": {"mode": "back"},
            "staking": {"method": "fixed", "fixed_stake": 10.0},
            "persistence": {"type": "LAPSE"},
        })
        pred = self._make_prediction()
        instructions, skipped = engine.evaluate([pred], {})
        assert len(instructions) == 0
        assert skipped[0].reason == "no_market_prices"
