"""The owner's envelope: the limits inside which the desk may trade with real money.

Read from repository variables (the pattern of TRADING_LIVE and GREYHOUND_EARLY_LAY_DAILY_LOSS), which only the
owner can set and no agent's commit can change. Nothing here has a default: a missing, empty or malformed value
means no envelope, and no envelope means paper only. No figure is invented for the owner.

    DESK_BANKROLL               GBP the desk may have at risk across every sport
    DESK_DAILY_LOSS             GBP: the day stops for every desk strategy once the day's loss reaches it
    DESK_SPORT_SHARE            each sport's share of the day loss and the bankroll, e.g.
                                "horse_racing=0.6,greyhounds=0.1,football=0.1,tennis=0.1,cricket=0.1"
                                (a sport left out gets nothing; the shares add to at most 1)
    DESK_RESERVE                GBP of available funds kept back for the live horse rule (its own day limit)
    DESK_PILOT_BUDGET           GBP a day across every strategy at the pilot stage
    DESK_MAX_MARKET_EXPOSURE    GBP: the worst case of every desk strategy together on one market
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from desk import SPORTS

VARS = ("DESK_BANKROLL", "DESK_DAILY_LOSS", "DESK_SPORT_SHARE", "DESK_RESERVE", "DESK_PILOT_BUDGET",
        "DESK_MAX_MARKET_EXPOSURE")


@dataclass(frozen=True)
class Envelope:
    bankroll: float
    daily_loss: float
    sport_share: dict = field(default_factory=dict)
    reserve: float = 0.0
    pilot_budget: float = 0.0
    max_market_exposure: float = 0.0

    def sport_daily_loss(self, sport: str) -> float:
        return self.daily_loss * self.sport_share.get(sport, 0.0)

    def sport_bankroll(self, sport: str) -> float:
        return self.bankroll * self.sport_share.get(sport, 0.0)


def _money(raw: str | None) -> float | None:
    try:
        x = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x >= 0 else None


def parse_shares(raw: str | None) -> dict | None:
    """"sport=share,..." to a dict; None when any part is malformed, a sport unknown or the shares add to over 1."""
    if not raw or not str(raw).strip():
        return None
    out = {}
    for part in str(raw).split(","):
        if "=" not in part:
            return None
        k, v = (s.strip() for s in part.split("=", 1))
        x = _money(v)
        if k not in SPORTS or x is None or x > 1 or k in out:
            return None
        out[k] = x
    return out if sum(out.values()) <= 1 + 1e-9 else None


def from_env(env=None) -> tuple[Envelope | None, str]:
    """The envelope and "" when every variable is set and sound; else None and the reason (paper only)."""
    env = os.environ if env is None else env
    missing = [v for v in VARS if not str(env.get(v) or "").strip()]
    if missing:
        return None, "no envelope (unset: " + ", ".join(missing) + "): paper only"
    money = {v: _money(env.get(v)) for v in VARS if v != "DESK_SPORT_SHARE"}
    bad = [v for v, x in money.items() if x is None]
    if bad:
        return None, "envelope malformed (" + ", ".join(bad) + "): paper only"
    shares = parse_shares(env.get("DESK_SPORT_SHARE"))
    if shares is None:
        return None, "envelope malformed (DESK_SPORT_SHARE): paper only"
    if money["DESK_BANKROLL"] <= 0 or money["DESK_DAILY_LOSS"] <= 0:
        return None, "envelope has no bankroll or no day loss: paper only"
    if money["DESK_DAILY_LOSS"] > money["DESK_BANKROLL"]:
        return None, "envelope's day loss is over its bankroll: paper only"
    return Envelope(bankroll=money["DESK_BANKROLL"], daily_loss=money["DESK_DAILY_LOSS"], sport_share=shares,
                    reserve=money["DESK_RESERVE"], pilot_budget=money["DESK_PILOT_BUDGET"],
                    max_market_exposure=money["DESK_MAX_MARKET_EXPOSURE"]), ""
