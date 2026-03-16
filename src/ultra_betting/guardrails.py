"""Guardrails — hard limits on betting activity."""

import logging
from datetime import datetime

from ultra_betting.config import load_guardrails
from ultra_betting.data.schemas import BetInstruction, Execution

log = logging.getLogger(__name__)


class GuardrailViolation(Exception):
    """Raised when a guardrail is violated."""
    pass


class Guardrails:
    """Enforces hard limits on bets before they are placed."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_guardrails()

    @property
    def dry_run(self) -> bool:
        return self.config.get("dry_run", False)

    def check_instruction(
        self,
        instruction: BetInstruction,
        todays_executions: list[Execution],
    ) -> tuple[bool, str]:
        """Check a bet instruction against all guardrails.

        Returns:
            (allowed, reason) — allowed is True if the bet passes all checks.
        """
        # Max stake
        max_stake = self.config.get("max_stake", 50.0)
        if instruction.stake > max_stake:
            return False, f"Stake £{instruction.stake:.2f} exceeds max £{max_stake:.2f}"

        # Max liability for lay bets
        if instruction.side == "LAY":
            liability = instruction.stake * (instruction.price - 1)
            max_liability = self.config.get("max_liability", 100.0)
            if liability > max_liability:
                return False, f"Liability £{liability:.2f} exceeds max £{max_liability:.2f}"

        # Max daily bets
        max_daily = self.config.get("max_daily_bets", 30)
        placed_today = len([e for e in todays_executions if e.status != "FAILED"])
        if placed_today >= max_daily:
            return False, f"Daily bet limit reached ({max_daily})"

        # Max daily loss (stop-loss circuit breaker)
        max_loss = self.config.get("max_daily_loss", -200.0)
        daily_pnl = sum(e.stake * -1 for e in todays_executions if e.status == "MATCHED")
        # Conservative: assume all pending bets lose
        if daily_pnl <= max_loss:
            return False, f"Daily loss limit hit (£{daily_pnl:.2f} <= £{max_loss:.2f})"

        # Allowed countries (checked at market level, not here)
        # Min minutes to race (checked in pipeline, not here)
        # Min market liquidity (checked in pipeline, not here)

        return True, "OK"

    def apply(
        self,
        instruction: BetInstruction,
        todays_executions: list[Execution],
    ) -> BetInstruction:
        """Apply guardrails, capping stake if needed.

        Returns the (possibly modified) instruction.
        Raises GuardrailViolation if the bet must be rejected entirely.
        """
        allowed, reason = self.check_instruction(instruction, todays_executions)
        if not allowed:
            raise GuardrailViolation(reason)

        # Cap stake to max
        max_stake = self.config.get("max_stake", 50.0)
        if instruction.stake > max_stake:
            log.warning(f"Capping stake from £{instruction.stake:.2f} to £{max_stake:.2f}")
            instruction.stake = max_stake

        return instruction
