"""Stake sizing — Kelly, fractional Kelly, fixed."""

import math


def calculate_stake(
    win_prob: float,
    edge: float,
    price: float,
    config: dict,
) -> float:
    """Calculate the stake size based on the configured method.

    Args:
        win_prob: Model's predicted win probability.
        edge: Edge percentage.
        price: The live price we'd bet at.
        config: Staking section from rules.yaml.

    Returns:
        Stake in GBP.
    """
    method = config.get("method", "fixed")

    if method == "fixed":
        return config.get("fixed_stake", 10.0)

    elif method == "fractional_kelly":
        fraction = config.get("kelly_fraction", 0.25)
        bank = config.get("starting_bank", 1000.0)
        kelly = _kelly_criterion(win_prob, price)
        return max(0, round(kelly * fraction * bank, 2))

    elif method == "percentage_bank":
        bank = config.get("starting_bank", 1000.0)
        pct = config.get("bank_percentage", 2.0) / 100
        return round(bank * pct, 2)

    else:
        return config.get("fixed_stake", 10.0)


def _kelly_criterion(win_prob: float, price: float) -> float:
    """Calculate Kelly criterion fraction.

    Kelly % = (bp - q) / b
    where b = price - 1 (net odds), p = win_prob, q = 1 - p
    """
    if price <= 1 or win_prob <= 0 or win_prob >= 1:
        return 0.0

    b = price - 1  # net decimal odds
    p = win_prob
    q = 1 - p

    kelly = (b * p - q) / b

    # Never return negative (means don't bet)
    return max(0, kelly)
