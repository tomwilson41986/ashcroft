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
  closing_clv the owner's rule: back where the expected CLV at the best back price, under the closing
              model (model/race_book.py), is at least `clv_bar`; always staked to win `target` before
              commission (target / (price - 1)), at most the per-bet limit

Stakes (TradingConfig.staking) for the first three: level (`unit` each), to_win (enough to win
`target` after commission), or kelly (single-runner Kelly on the pooled price, a fraction of it).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trading import pricing, staking

ROOT = Path(__file__).resolve().parent.parent


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
    traded: float = 0.0                # the runner's matched money (0 when the feed does not report it)


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
    if cfg.strategy == "closing_clv":
        return _plan_closing_clv(live, cfg)
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


_MODELS: dict = {}


def _closing_model(path: str):
    """The closing model at ``path`` (relative to the repository), read once."""
    if path not in _MODELS:
        from model import race_book as rb
        p = Path(path)
        _MODELS[path] = rb.load_closing_model(p if p.is_absolute() else ROOT / p)
    return _MODELS[path]


def _plan_closing_clv(live: list[RunnerView], cfg) -> list[Decision]:
    """The owner's rule on one race. Only a whole race is read: every runner still running priced by the model and
    by the market. A runner is backed where its expected CLV at the best back price is at least cfg.clv_bar and it
    has at least limits.min_runner_matched matched; a race whose feed reports no matched money at all (Betfair's
    delayed key) is read by the closing model fitted without volume and not held to that floor. The target is the
    day's whole stake on the runner: enough to win cfg.target before commission, at most limits.max_stake."""
    from model import race_book as rb
    if len(live) < 2 or any(not (r.model_price and r.model_price > 1) for r in live):
        return []
    fallback = [_market_price(r) for r in live]
    if any(p is None for p in fallback):
        return []
    back = np.array([r.back if (r.back and r.back > 1) else np.nan for r in live], float)
    lay = np.array([r.lay if (r.lay and r.lay > 1) else np.nan for r in live], float)
    traded = np.array([max(float(r.traded or 0.0), 0.0) for r in live], float)
    blind = not (traded > 0).any()
    model = _closing_model(cfg.closing_model_novol if blind else cfg.closing_model)
    with np.errstate(invalid="ignore", divide="ignore"):
        market = np.where(np.isfinite(back), rb.market_now(back, lay), np.array(fallback, float))
    model_price = np.array([r.model_price for r in live], float)
    rng = np.random.default_rng(sum(int(r.selection_id) for r in live) % (2 ** 32))   # the same race, the same draws
    ev, exp_bsp = rb.race_expected_clv(back, lay, model_price, traded, model, n_draws=cfg.clv_draws, rng=rng,
                                       market=market)
    lim = cfg.limits
    tradable = np.isfinite(back) & (back >= lim.min_price) & (back <= lim.max_price)
    enough = np.ones(len(live), bool) if blind else traded >= lim.min_runner_matched
    chosen = tradable & (np.nan_to_num(ev, nan=-1.0) >= cfg.clv_bar) & enough
    race = np.zeros(len(live), dtype=int)
    p_model = pricing.normalise(1.0 / model_price, race)
    p_mkt = pricing.market_probabilities(market, race)
    out = []
    for i, r in enumerate(live):
        if not chosen[i]:
            continue
        stake = min(cfg.target / (back[i] - 1.0), lim.max_stake)
        out.append(Decision(selection_id=r.selection_id, name=r.name, target=float(stake), price=float(back[i]),
                            back_size=float(r.back_size), p_model=float(p_model[i]), p_market=float(p_mkt[i]),
                            p_pool=float(1.0 / exp_bsp[i]), edge=float(ev[i]),
                            move=float(np.log(back[i] / r.model_price)),
                            reason="closing CLV" + (", feed without volume" if blind else "")))
    return out
