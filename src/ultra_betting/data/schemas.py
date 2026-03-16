"""Pydantic models for predictions, executions, and settlements."""

from datetime import datetime

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    """A single runner prediction from the Ashcroft model."""
    date: str
    venue: str
    race_time: str
    market_id: str = ""
    selection_id: int = 0
    runner_name: str
    predicted_bfsp: float
    predicted_win_prob: float
    predicted_at: datetime = Field(default_factory=datetime.utcnow)
    prediction_id: str = ""  # {date}_{market_id}_{selection_id}

    def model_post_init(self, __context):
        if not self.prediction_id:
            self.prediction_id = f"{self.date}_{self.market_id}_{self.selection_id}"


class BetInstruction(BaseModel):
    """Output of the rules engine: a bet to place."""
    prediction_id: str
    market_id: str
    selection_id: int
    runner_name: str
    side: str  # "BACK" or "LAY"
    stake: float
    price: float
    persistence: str = "LAPSE"
    reasoning: str = ""


class Execution(BaseModel):
    """A bet that has been placed (or attempted)."""
    date: str
    venue: str
    race_time: str
    market_id: str
    selection_id: int
    runner_name: str
    side: str
    stake: float
    price_requested: float
    price_matched: float = 0.0
    size_matched: float = 0.0
    bet_id: str = ""
    status: str = "PENDING"  # PENDING, MATCHED, FAILED, DRY_RUN
    dry_run: bool = False
    reasoning: str = ""
    executed_at: datetime = Field(default_factory=datetime.utcnow)
    prediction_id: str = ""


class Settlement(BaseModel):
    """A settled bet with P&L."""
    date: str
    venue: str
    race_time: str
    market_id: str
    selection_id: int
    runner_name: str
    side: str
    stake: float
    price_matched: float
    result: str = ""  # "WON", "LOST", "VOID"
    pnl: float = 0.0
    commission: float = 0.0
    net_pnl: float = 0.0
    bet_id: str = ""
    settled_at: datetime = Field(default_factory=datetime.utcnow)


class DailyPnL(BaseModel):
    """Daily profit & loss summary."""
    date: str
    bets_placed: int = 0
    bets_won: int = 0
    bets_lost: int = 0
    gross_pnl: float = 0.0
    commission: float = 0.0
    net_pnl: float = 0.0
    roi_percent: float = 0.0
    bank_start: float = 0.0
    bank_end: float = 0.0


class SkippedBet(BaseModel):
    """A prediction that didn't result in a bet, with reason."""
    prediction_id: str
    runner_name: str
    venue: str
    race_time: str
    reason: str  # e.g. "no_edge", "market_closed", "guardrail_hit"
    detail: str = ""
