"""The trading settings.

A setting comes from trading/config.json (the paper forward test) or trading/config_live.json (the owner's
live book), then from an environment variable TRADING_<NAME> (so a repository variable can change it without a
commit), then from the command line. The paper trader simulates orders against Betfair's live prices and places
none; the live trader (auto_trade.py --live, and only with TRADING_LIVE=yes) places them.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

STRATEGIES = ("rule", "value", "owner", "race_kelly", "closing_clv")
STAKINGS = ("level", "to_win", "kelly")


@dataclass
class Limits:
    bank: float = 1000.0                 # the bank the Kelly fractions and limits are sized on, £
    max_stake: float = 25.0              # one bet, £
    max_race_stake: float = 75.0         # all bets on one race, £
    max_daily_turnover: float = 1000.0   # everything staked today, £
    max_bets_per_day: int = 150
    daily_stop_loss: float = 150.0       # stop betting once today's settled loss reaches this, £
    min_stake: float = 2.0               # Betfair's smallest back stake, £
    min_price: float = 1.5
    max_price: float = 30.0
    min_market_matched: float = 500.0    # the market's matched volume before we trade, £
    min_runner_matched: float = 100.0    # closing_clv: the runner's matched money before we back it (when reported), £
    min_bsp_liability: float = 10.0      # the smallest lay at BSP sent; a smaller trade-out waits for more fills, £
    liquidity_share: float = 0.5         # of the size offered at the best back price


@dataclass
class TradingConfig:
    strategy: str = "rule"               # rule | value | owner | race_kelly | closing_clv
    staking: str = "level"               # level | to_win | kelly (race_kelly: Kelly; closing_clv: to win before commission)
    unit: float = 5.0                    # level stake, £
    target: float = 10.0                 # stake to win this much after commission, £
    kelly_fraction: float = 0.25
    model_weight: float = 0.5            # the model's weight in the pooled price (trading/pricing.py)
    margin: float = 0.2                  # rule: back where ln(best back / model price) >= margin
    min_edge: float = 0.02               # value, owner: expected profit per unit on the pooled price
    underlay: float = 0.10               # owner: back the favourite down to this far under fair
    clv_bar: float = 0.03                # closing_clv: back where the expected CLV at the best back is at least this
    closing_model: str = "data/models/closing_model.json"              # closing_clv: the closing model
    closing_model_novol: str = "data/models/closing_model_novol.json"  # ... for a race whose feed has no volume
    clv_draws: int = 20000               # closing_clv: draws of the BSP book per race
    commission: float = 0.05
    trade_from: str = "08:00"            # UK time: the tested edge is on the morning price
    trade_until: str = "11:00"
    stop_before_off: int = 5             # minutes: no new bets closer to the off than this
    trade_out: bool = True               # close each position with a lay at BSP: take the price's move, not the result
    trade_out_at: str = "fill"           # fill: lay at BSP as soon as the back matches; before_off: just before the off
    trade_out_before_off: int = 3        # minutes before the off, for before_off
    poll_seconds: int = 60
    countries: tuple = ("GB", "IE")
    limits: Limits = field(default_factory=Limits)

    def validate(self) -> "TradingConfig":
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy {self.strategy!r}: one of {STRATEGIES}")
        if self.staking not in STAKINGS:
            raise ValueError(f"staking {self.staking!r}: one of {STAKINGS}")
        if self.trade_out_at not in ("fill", "before_off"):
            raise ValueError("trade_out_at: fill or before_off")
        if not 0 < self.kelly_fraction <= 1:
            raise ValueError("kelly_fraction lies in (0, 1]")
        if not 0 <= self.model_weight <= 1:
            raise ValueError("model_weight lies in [0, 1]")
        if self.limits.max_stake > self.limits.max_race_stake or self.limits.min_stake > self.limits.max_stake:
            raise ValueError("limits: min_stake <= max_stake <= max_race_stake")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


def _coerce(value: str, like):
    if isinstance(like, bool):
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(like, int):
        return int(value)
    if isinstance(like, float):
        return float(value)
    if isinstance(like, tuple):
        return tuple(v.strip() for v in str(value).split(",") if v.strip())
    return str(value)


def load_config(path: str | None = None, env: dict | None = None, overrides: dict | None = None) -> TradingConfig:
    """The settings: defaults, then the JSON file, then TRADING_* variables, then overrides."""
    env = os.environ if env is None else env
    raw: dict = {}
    if path and Path(path).exists():
        raw = json.loads(Path(path).read_text())
    limits = Limits(**raw.pop("limits", {}))
    cfg = TradingConfig(**{k: (tuple(v) if k == "countries" else v) for k, v in raw.items()}, limits=limits)
    for f in fields(TradingConfig):
        if f.name == "limits":
            continue
        key = f"TRADING_{f.name.upper()}"
        if env.get(key, "") != "":
            setattr(cfg, f.name, _coerce(env[key], getattr(cfg, f.name)))
    for f in fields(Limits):
        key = f"TRADING_{f.name.upper()}"
        if env.get(key, "") != "":
            setattr(cfg.limits, f.name, _coerce(env[key], getattr(cfg.limits, f.name)))
    for k, v in (overrides or {}).items():
        if v is None:
            continue
        target = cfg.limits if hasattr(cfg.limits, k) and not hasattr(cfg, k) else cfg
        setattr(target, k, v)
    return cfg.validate()
