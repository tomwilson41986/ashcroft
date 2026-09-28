"""Which runners of a race to back now, and how much in all by the end of the day.

`plan_race` reads one race as the exchange shows it (each runner's best back and lay) beside the
model's prices, and returns a target stake per runner: the most the day should have on it. The
session backs only the difference between that and what is already matched, so a price that
stays good is not backed twice and a target never shrinks into a lay.

Strategies (TradingConfig.strategy):
  rule        the tested early-price rule: back where the model is at least `margin` shorter in log
              than the best back price (0.2: the model at least 22% shorter)
  value       back where the pooled price gives an expected profit of at least `min_edge` per unit
  owner       value, and in a race with a value bet also the pooled favourite at fair odds or down
              to `underlay` under them, if the race as a whole still expects a profit
  race_kelly  race-level Kelly on the pooled price (trading/staking.py): overlays, and underlays
              only where they raise the bank's growth; always staked by Kelly

Stakes (TradingConfig.staking) for the first three: level (`unit` each), to_win (enough to win
`target` after commission), or kelly (single-runner Kelly on the pooled price, a fraction of it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trading import pricing, staking


@dataclass
class RunnerView:
    selection_id: int
    name: str
    model_price: float | None          # the model's predicted BSP; None if the runner was not priced
    back: float | None                 # best price offered to back now
    back_size: float = 0.0
    lay: float | None = None
    last_traded: float | None = None
    status: str = "ACTIVE"


@dataclass
class Decision:
    selection_id: int
    name: str
    target: float                      # £ the day should hold on this runner
    price: float | None                # the best back price now
    back_size: float
    p_model: float
    p_market: float
    p_pool: float
    edge: float                        # expected profit per unit at the best back price, pooled
    move: float                        # ln(best back / the model's price for the runners left)
    reason: str


def _market_price(r: RunnerView) -> float | None:
    """The market's price for a runner: the geometric mid of back and lay, else whichever exists."""
    if r.back and r.lay and r.back > 1 and r.lay > 1:
        return float(np.exp(0.5 * (np.log(r.back) + np.log(r.lay))))
    for p in (r.back, r.last_traded, r.lay):
        if p and p > 1:
            return float(p)
    return None


def plan_race(runners: list[RunnerView], cfg, bank: float, min_coverage: float = 0.9) -> list[Decision]:
    """Target stakes for one race now, and why."""
    live = [r for r in runners if r.status == "ACTIVE"]
    priced = [r for r in live if r.model_price and r.model_price > 1 and _market_price(r)]
    if not live or len(priced) < max(2, min_coverage * len(live)):
        return []                                   # the model or the market cannot price enough of the field
    race = np.zeros(len(priced), dtype=int)
    p_model = pricing.normalise(1.0 / np.array([r.model_price for r in priced], float), race)
    p_mkt = pricing.market_probabilities(np.array([_market_price(r) for r in priced], float), race)
    p_pool = pricing.pooled_probabilities(p_model, p_mkt, race, cfg.model_weight)
    back = np.array([r.back if (r.back and r.back > 1) else np.nan for r in priced], float)
    lim = cfg.limits
    tradable = np.isfinite(back) & (back >= lim.min_price) & (back <= lim.max_price)
    price = np.where(tradable, back, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        move = np.log(back * p_model)               # ln(back / (1 / p_model))
    edge = np.where(tradable, staking.edge(p_pool, np.nan_to_num(back, nan=1.0), cfg.commission), np.nan)

    if cfg.strategy == "race_kelly":
        target = bank * staking.race_kelly(p_pool, price, cfg.commission, cfg.kelly_fraction)
        chosen = target > 0
        reasons = np.where(chosen, np.where(edge >= 0, "race Kelly: overlay", "race Kelly: hedging underlay"), "")
    else:
        if cfg.strategy == "rule":
            chosen = tradable & (move >= cfg.margin)
        else:
            chosen = tradable & (edge >= cfg.min_edge)
        reasons = np.where(chosen, cfg.strategy, "")
        if cfg.strategy == "owner" and chosen.any():
            fav = int(np.nanargmax(p_pool))
            if tradable[fav] and not chosen[fav] and edge[fav] >= -cfg.underlay:
                trial = chosen.copy()
                trial[fav] = True
                stakes = _stakes(cfg, p_pool, back, trial, bank)
                if staking.race_expectation(p_pool, np.nan_to_num(back, nan=1.0), stakes, cfg.commission) > 0:
                    chosen = trial
                    reasons = reasons.astype(object)
                    reasons[fav] = "owner: favourite as a hedge"
        target = _stakes(cfg, p_pool, back, chosen, bank)
    out = []
    for i, r in enumerate(priced):
        if target[i] > 0:
            out.append(Decision(selection_id=r.selection_id, name=r.name, target=float(target[i]),
                                price=float(back[i]), back_size=float(r.back_size), p_model=float(p_model[i]),
                                p_market=float(p_mkt[i]), p_pool=float(p_pool[i]), edge=float(edge[i]),
                                move=float(move[i]), reason=str(reasons[i])))
    return out


def _stakes(cfg, p_pool, back, chosen, bank) -> np.ndarray:
    price = np.nan_to_num(back, nan=1.0)
    if cfg.staking == "level":
        s = np.full(len(back), float(cfg.unit))
    elif cfg.staking == "to_win":
        s = staking.fixed_win_stakes(price, cfg.target, cfg.commission)
    else:
        s = bank * staking.single_kelly(p_pool, price, cfg.commission, cfg.kelly_fraction)
    return np.where(chosen, s, 0.0)
