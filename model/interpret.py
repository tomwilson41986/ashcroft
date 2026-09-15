"""
Correlation-robust model interpretation.

SHAP and plain permutation importance split credit arbitrarily among
correlated features (ratings, speed figures, class and pace features are
all inter-correlated in racing). Two robust alternatives:

    accumulated_local_effects          ALE (Apley & Zhu 2020): local
                                       differences, so correlated features
                                       are not extrapolated off-manifold
    conditional_permutation_importance Strobl et al. (2008): permute a
                                       feature *within strata* of the
                                       features it is correlated with
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def accumulated_local_effects(predict_fn, X: pd.DataFrame, feature: str, n_bins: int = 20) -> pd.DataFrame:
    """First-order ALE curve for one numeric feature.

    predict_fn(X_df) -> 1-D array. Returns the centred ALE curve evaluated
    at each bin edge ``x`` (n = points in the bin ending at that edge).
    """
    x = pd.to_numeric(X[feature], errors="coerce")
    ok = x.notna()
    Xo, xo = X.loc[ok], x.loc[ok]
    edges = np.unique(np.quantile(xo, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return pd.DataFrame(columns=["x", "ale", "n"])
    idx = np.clip(np.searchsorted(edges, xo.values, side="right") - 1, 0, len(edges) - 2)
    effects, counts = [], []
    for k in range(len(edges) - 1):
        m = idx == k
        if not m.any():
            effects.append(0.0); counts.append(0); continue
        lo = Xo.loc[m].copy(); hi = Xo.loc[m].copy()
        lo[feature] = edges[k]; hi[feature] = edges[k + 1]
        effects.append(float(np.mean(predict_fn(hi) - predict_fn(lo))))
        counts.append(int(m.sum()))
    ale = np.concatenate([[0.0], np.cumsum(effects)])  # value at each bin edge
    counts = np.asarray(counts)
    # centre so the count-weighted mean of the curve over the data is zero
    mid = 0.5 * (ale[1:] + ale[:-1])
    ale = ale - np.sum(mid * counts) / max(counts.sum(), 1)
    return pd.DataFrame({"x": edges, "ale": ale, "n": np.concatenate([[0], counts])})


def conditional_permutation_importance(predict_fn, X: pd.DataFrame, y, score_fn, features=None,
                                       n_repeats: int = 5, corr_threshold: float = 0.5,
                                       n_strata_bins: int = 4, random_state: int = 0) -> pd.DataFrame:
    """Permutation importance where each feature is shuffled *within*
    strata defined by its most correlated companions.

    score_fn(y, pred) -> float (higher = better, e.g. negative log-loss).
    Reports the drop in score when the feature is conditionally permuted.
    """
    rng = np.random.default_rng(random_state)
    feats = list(features or X.columns)
    num = X[feats].apply(pd.to_numeric, errors="coerce")
    corr = num.corr().abs()
    base = float(score_fn(y, predict_fn(X)))
    rows = []
    for f in feats:
        partners = corr[f].drop(f).sort_values(ascending=False)
        partners = [p for p, v in partners.items() if v >= corr_threshold][:2]
        if partners:
            strata = pd.Series(0, index=X.index)
            for i, p in enumerate(partners):
                b = pd.qcut(num[p].rank(method="first"), n_strata_bins, labels=False, duplicates="drop")
                strata = strata * n_strata_bins + b.fillna(-1).astype(int)
        else:
            strata = pd.Series(0, index=X.index)
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            col = Xp[f].values.copy()
            for s in np.unique(strata.values):
                m = np.where(strata.values == s)[0]
                col[m] = col[rng.permutation(m)]
            Xp[f] = col
            drops.append(base - float(score_fn(y, predict_fn(Xp))))
        rows.append({"feature": f, "importance": float(np.mean(drops)), "sd": float(np.std(drops)),
                     "conditioned_on": ",".join(partners)})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
