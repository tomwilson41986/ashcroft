"""Betfair Exchange integration for automated betting.

Modules:
    client   — API-NG client (auth, markets, prices, orders)
    matcher  — HRB horse name → Betfair selection ID matching
    betting  — Bet placement engine with safety controls
    audit    — JSONL audit trail and daily exposure tracking
    config   — BettingConfig dataclass with env-var overrides
"""

from betfair.audit import BetAuditLog
from betfair.betting import BettingEngine
from betfair.client import BetfairClient
from betfair.config import BettingConfig
from betfair.matcher import HorseNameMatcher

__all__ = [
    "BetfairClient",
    "HorseNameMatcher",
    "BettingEngine",
    "BetAuditLog",
    "BettingConfig",
]
