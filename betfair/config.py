"""Betting configuration with safety controls.

All parameters can be overridden via environment variables (prefixed
with ``BETTING_``). For example, ``BETTING_BANKROLL=500`` sets the
bankroll to GBP 500.
"""

import os
from dataclasses import dataclass, field


@dataclass
class BettingConfig:
    """Central configuration for the automated betting pipeline.

    Attributes are grouped into staking parameters, safety limits, and
    runtime controls.  Call :meth:`from_env` to build an instance from
    environment variables with sensible defaults.
    """

    # --- Staking ---
    min_edge: float = 0.05
    kelly_fraction: float = 0.25
    bankroll: float = 1000.0
    min_stake: float = 2.0
    max_stake: float = 500.0
    max_stake_pct: float = 0.02
    commission_rate: float = 0.05

    # --- Safety limits ---
    max_daily_loss: float = 200.0
    max_daily_exposure: float = 500.0
    max_daily_bets: int = 20
    max_single_race_bets: int = 2

    # --- Price bounds ---
    min_back_price: float = 2.0
    max_back_price: float = 30.0

    # --- Controls ---
    dry_run: bool = False
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "BettingConfig":
        """Build a config from ``BETTING_*`` environment variables.

        Any unset variable falls back to the dataclass default.
        Boolean values accept ``true/false/1/0/yes/no``.
        """

        def _bool(val: str) -> bool:
            return val.strip().lower() in ("true", "1", "yes")

        def _get(name: str, default):
            raw = os.getenv(f"BETTING_{name.upper()}")
            if raw is None:
                return default
            if isinstance(default, bool):
                return _bool(raw)
            if isinstance(default, int):
                return int(raw)
            if isinstance(default, float):
                return float(raw)
            return raw

        defaults = cls()
        return cls(
            min_edge=_get("min_edge", defaults.min_edge),
            kelly_fraction=_get("kelly_fraction", defaults.kelly_fraction),
            bankroll=_get("bankroll", defaults.bankroll),
            min_stake=_get("min_stake", defaults.min_stake),
            max_stake=_get("max_stake", defaults.max_stake),
            max_stake_pct=_get("max_stake_pct", defaults.max_stake_pct),
            commission_rate=_get("commission_rate", defaults.commission_rate),
            max_daily_loss=_get("max_daily_loss", defaults.max_daily_loss),
            max_daily_exposure=_get("max_daily_exposure", defaults.max_daily_exposure),
            max_daily_bets=_get("max_daily_bets", defaults.max_daily_bets),
            max_single_race_bets=_get("max_single_race_bets", defaults.max_single_race_bets),
            min_back_price=_get("min_back_price", defaults.min_back_price),
            max_back_price=_get("max_back_price", defaults.max_back_price),
            dry_run=_get("dry_run", defaults.dry_run),
            enabled=_get("enabled", defaults.enabled),
        )
