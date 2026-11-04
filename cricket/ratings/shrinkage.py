"""Empirical-Bayes shrinkage with recency decay (build spec 6.2): the recurring recipes, as-of each match.

- ``decayed_asof``: for each (entity, match) row in date order, the decayed sum of the entity's earlier rows only,
  ``S_k = r * S_{k-1} + x_{k-1}`` (the latest earlier match weighs 1, the one before r, ...). The current match never
  counts towards its own rating: this is what keeps the ratings free of leakage.
- ``dirichlet``: counts shrunk towards a prior distribution with strength ``c`` balls (SFU: c = 60, r = 0.88).
- ``n_over_n_plus_k``: a rate shrunk towards its population, weight n / (n + k) (Ganesh: k = 50).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def decayed_asof(df: pd.DataFrame, entity: str, order: list[str], value_cols: list[str], r: float) -> np.ndarray:
    """Decayed sums of ``value_cols`` over each entity's earlier rows; rows returned in ``df``'s own order."""
    s = df.sort_values([entity] + order, kind="stable")
    x = s[value_cols].to_numpy(dtype=float)
    ent = s[entity].to_numpy()
    out = np.zeros_like(x)
    acc = np.zeros(x.shape[1])
    prev = None
    for i in range(len(x)):
        if ent[i] != prev:
            acc = np.zeros(x.shape[1])
            prev = ent[i]
        out[i] = acc
        acc = r * acc + x[i]
    res = np.empty_like(out)
    res[df.index.get_indexer(s.index)] = out
    return res


def dirichlet(counts: np.ndarray, prior: np.ndarray, c: float) -> np.ndarray:
    """Posterior mean of a Dirichlet-multinomial: (counts + c * prior) / (n + c), row-wise."""
    n = counts.sum(axis=1, keepdims=True)
    return (counts + c * prior[None, :]) / (n + c)


def n_over_n_plus_k(rate: np.ndarray, n: np.ndarray, pop: float | np.ndarray, k: float) -> np.ndarray:
    w = n / (n + k)
    return w * np.nan_to_num(rate) + (1 - w) * pop
