"""Tests for execution logic (dry run mode)."""

from ultra_betting.data.schemas import BetInstruction, Execution
from ultra_betting.betfair.execution import place_bet


class TestDryRunExecution:
    def test_dry_run_returns_execution(self):
        """Dry run should return a valid execution record without hitting API."""
        instruction = BetInstruction(
            prediction_id="2026-03-16_1.234_12345",
            market_id="1.234",
            selection_id=12345,
            runner_name="Test Horse",
            side="BACK",
            stake=10.0,
            price=3.5,
            reasoning="Test edge",
        )

        execution = place_bet(instruction, dry_run=True)

        assert execution.status == "DRY_RUN"
        assert execution.dry_run is True
        assert execution.stake == 10.0
        assert execution.price_requested == 3.5
        assert execution.price_matched == 3.5
        assert execution.size_matched == 10.0
        assert execution.side == "BACK"
        assert execution.runner_name == "Test Horse"

    def test_dry_run_lay(self):
        instruction = BetInstruction(
            prediction_id="test",
            market_id="1.234",
            selection_id=12345,
            runner_name="Lay Horse",
            side="LAY",
            stake=5.0,
            price=6.0,
        )

        execution = place_bet(instruction, dry_run=True)
        assert execution.status == "DRY_RUN"
        assert execution.side == "LAY"
