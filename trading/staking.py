"""How much to back each runner of a race.

Every function takes decimal prices and a commission rate and returns stakes in money,
or, for the Kelly functions, fractions of the bank. Betfair charges its commission on
net winnings in a market, so a winning back at price O returns (O - 1) * (1 - c) per
unit staked: `effective_odds` is 1 plus that.
"""

from __future__ import annotations

import numpy as np


def effective_odds(price, commission: float = 0.05) -> np.ndarray:
    """Decimal odds net of commission on the winnings: 1 + (O - 1)(1 - c)."""
    price = np.asarray(price, dtype=float)
    return 1.0 + (price - 1.0) * (1.0 - commission)


def edge(p, price, commission: float = 0.05) -> np.ndarray:
    """Expected profit per unit staked on a back bet held to the result."""
    return np.asarray(p, dtype=float) * effective_odds(price, commission) - 1.0


def level_stakes(price, unit: float = 1.0) -> np.ndarray:
    return np.full(np.shape(price), float(unit))


def fixed_win_stakes(price, target: float, commission: float = 0.05) -> np.ndarray:
    """Stake to win `target` (after commission) whatever the price: a long shot gets a small
    stake, a favourite a large one, so each winner is worth the same."""
    net = effective_odds(price, commission) - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(net > 0, target / net, 0.0)


def single_kelly(p, price, commission: float = 0.05, fraction: float = 1.0) -> np.ndarray:
    """Each runner on its own: f = (p * b - (1 - p)) / b with b the net odds, none where the
    edge is negative. Ignores that the runners of one race exclude each other."""
    p = np.asarray(p, dtype=float)
    b = effective_odds(price, commission) - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(b > 0, (p * b - (1.0 - p)) / b, 0.0)
    return np.clip(f, 0.0, None) * fraction


def race_kelly(p, price, commission: float = 0.05, fraction: float = 1.0) -> np.ndarray:
    """The Kelly bets on one race, whose runners exclude each other (Smoczynski and Tomkins,
    2010): the fractions of the bank that maximise the expected log of the bank after it.

    Runners are taken in order of expected return p * O; each joins while its return beats
    the reserve rate R = (1 - sum p) / (1 - sum 1/O) of those already in. Once runners with
    an edge are in, R falls below 1, so a runner priced a little short of fair can join: it
    pays when the others lose, and backing it raises the growth of the bank. That is the
    exact form of backing a favourite at fair odds or slightly under to steady a race. The
    stakes are f_i = p_i - R / O_i; `fraction` scales them (a fraction of Kelly)."""
    p = np.asarray(p, dtype=float)
    o = effective_odds(price, commission)
    n = len(p)
    f = np.zeros(n)
    if n == 0:
        return f
    ok = np.isfinite(p) & np.isfinite(o) & (o > 1.0) & (p > 0)
    er = np.where(ok, p * o, -np.inf)
    order = np.argsort(-er)
    chosen: list[int] = []
    reserve = 1.0
    for i in order:
        if not ok[i] or er[i] <= reserve:
            break
        trial = chosen + [i]
        p_in, b_in = p[trial].sum(), (1.0 / o[trial]).sum()
        if b_in >= 1.0:                       # backing these alone would be a sure loss or a book
            break
        chosen = trial
        reserve = (1.0 - p_in) / (1.0 - b_in)
    if chosen:
        idx = np.array(chosen)
        f[idx] = np.clip(p[idx] - reserve / o[idx], 0.0, None)
    return f * fraction


def race_expectation(p, price, stakes, commission: float = 0.05) -> float:
    """The expected profit of a race's back bets held to the result."""
    return float(np.sum(np.asarray(stakes, dtype=float) * edge(p, price, commission)))


def race_log_growth(p, price, fractions, commission: float = 0.05) -> float:
    """E[log(bank after / bank before)] for back bets of the given fractions on one race."""
    p = np.asarray(p, dtype=float)
    f = np.asarray(fractions, dtype=float)
    o = effective_odds(price, commission)
    total = f.sum()
    outcomes = 1.0 - total + f * o                 # the bank if runner i wins
    lose_all = 1.0 - total                         # if a runner we did not back wins
    p_other = max(0.0, 1.0 - p.sum()) + p[f == 0].sum()
    backed = f > 0
    with np.errstate(divide="ignore"):
        g = np.sum(p[backed] * np.log(outcomes[backed]))
        if p_other > 0:
            g += p_other * np.log(lose_all) if lose_all > 0 else -np.inf
    return float(g)
