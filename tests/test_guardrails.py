"""Tests for guardrails."""

import pytest

from ultra_betting.guardrails import Guardrails, GuardrailViolation
from ultra_betting.data.schemas import BetInstruction, Execution


class TestGuardrails:
    def _make_instruction(self, stake=10.0, side="BACK", price=3.0):
        return BetInstruction(
            prediction_id="test_1",
            market_id="1.234",
            selection_id=12345,
            runner_name="Test Horse",
            side=side,
            stake=stake,
            price=price,
        )

    def test_allows_valid_bet(self):
        g = Guardrails({"max_stake": 50, "max_daily_bets": 30, "max_daily_loss": -200})
        allowed, reason = g.check_instruction(self._make_instruction(10.0), [])
        assert allowed
        assert reason == "OK"

    def test_blocks_excessive_stake(self):
        g = Guardrails({"max_stake": 20, "max_daily_bets": 30, "max_daily_loss": -200})
        allowed, reason = g.check_instruction(self._make_instruction(25.0), [])
        assert not allowed
        assert "max" in reason.lower()

    def test_blocks_excessive_liability(self):
        g = Guardrails({
            "max_stake": 100, "max_liability": 50,
            "max_daily_bets": 30, "max_daily_loss": -200,
        })
        # LAY £20 @ 5.0 = liability of £80
        allowed, reason = g.check_instruction(
            self._make_instruction(20.0, side="LAY", price=5.0), []
        )
        assert not allowed
        assert "liability" in reason.lower()

    def test_blocks_daily_limit(self):
        g = Guardrails({"max_stake": 50, "max_daily_bets": 2, "max_daily_loss": -200})
        existing = [
            Execution(
                date="2026-03-16", venue="", race_time="", market_id="1.1",
                selection_id=1, runner_name="A", side="BACK", stake=10,
                price_requested=3.0, status="MATCHED", prediction_id="p1",
            ),
            Execution(
                date="2026-03-16", venue="", race_time="", market_id="1.2",
                selection_id=2, runner_name="B", side="BACK", stake=10,
                price_requested=3.0, status="MATCHED", prediction_id="p2",
            ),
        ]
        allowed, reason = g.check_instruction(self._make_instruction(), existing)
        assert not allowed
        assert "limit" in reason.lower()

    def test_apply_raises_on_violation(self):
        g = Guardrails({"max_stake": 5, "max_daily_bets": 30, "max_daily_loss": -200})
        with pytest.raises(GuardrailViolation):
            g.apply(self._make_instruction(10.0), [])
