"""The fair probability of each runner, from the model, the market, or both.

The model on its own is not a better forecast of the result than the Betfair SP: its
Brier skill against the market is negative (reports/research_ledger.jsonl). What it does
forecast is where the price is going: beside the morning price it carries about half the
weight in the best forecast of the BSP (0.46-0.49 on Jan-Mar 2026). So the price we trade
on pools the two, log-linearly, with the model's weight `w`:

    p_i  proportional to  model_i ** w * market_i ** (1 - w),   renormalised in the race

w = 0 is the market, w = 1 the model. The market's probabilities are its prices with the
overround taken out in proportion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalise(p, race) -> np.ndarray:
    """Probabilities rescaled to sum to one within each race."""
    p = np.asarray(p, dtype=float)
    total = pd.Series(p).groupby(np.asarray(race)).transform("sum").to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return p / total


def market_probabilities(price, race) -> np.ndarray:
    """1 / price, with each race's overround removed in proportion."""
    with np.errstate(divide="ignore"):
        return normalise(1.0 / np.asarray(price, dtype=float), race)


def pooled_probabilities(p_model, p_market, race, w: float = 0.5) -> np.ndarray:
    """The log-linear pool of the model's and the market's probabilities, weight w on the model."""
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"the model's weight lies between 0 and 1, not {w}")
    pm = np.clip(np.asarray(p_model, dtype=float), 1e-9, 1.0)
    pk = np.clip(np.asarray(p_market, dtype=float), 1e-9, 1.0)
    return normalise(np.exp(w * np.log(pm) + (1.0 - w) * np.log(pk)), race)


def fair_price(p) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return 1.0 / np.asarray(p, dtype=float)
