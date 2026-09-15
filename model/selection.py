"""
Feature selection with error-rate control.

Why: the BFSP model has ~420 candidate features, many of them
correlated transforms of the same signal. Selecting by importance alone
invites false discoveries. These tools give honest selection:

    benjamini_hochberg       BH FDR control over a vector of p-values
    univariate_pvalues       Race-demeaned correlation tests per feature
                             (a cheap conditional-logit screen)
    knockoff_filter          Model-X Gaussian knockoffs (Candes, Fan,
                             Janson & Lv 2018) with the lasso
                             coefficient-difference statistic and the
                             knockoff+ threshold
    mrmr_select              Max-relevance / min-redundancy (Peng, Long &
                             Ding 2005) via mutual information

Recommended order: BH on univariate screens first (fast), knockoffs on
the survivors (honest joint selection), then feed the survivors to
LightGBM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression


# ---------------------------------------------------------------------------
# Benjamini-Hochberg
# ---------------------------------------------------------------------------

def benjamini_hochberg(pvalues, q: float = 0.1) -> dict:
    """BH step-up procedure. Returns rejections mask, threshold, adjusted p."""
    p = np.asarray(pvalues, dtype=float)
    ok = ~np.isnan(p)
    m = ok.sum()
    reject = np.zeros_like(p, dtype=bool)
    adj = np.full_like(p, np.nan)
    if m == 0:
        return {"reject": reject, "threshold": np.nan, "p_adjusted": adj, "n_rejected": 0}
    order = np.argsort(p[ok])
    ps = p[ok][order]
    ranks = np.arange(1, m + 1)
    crit = ranks / m * q
    below = np.where(ps <= crit)[0]
    thr = ps[below.max()] if len(below) else np.nan
    # adjusted p-values (monotone)
    adj_sorted = np.minimum.accumulate((ps * m / ranks)[::-1])[::-1]
    adj_ok = np.empty(m); adj_ok[order] = np.clip(adj_sorted, 0, 1)
    adj[ok] = adj_ok
    if not np.isnan(thr):
        rej_ok = p[ok] <= thr
        reject[ok] = rej_ok
    return {"reject": reject, "threshold": float(thr) if not np.isnan(thr) else np.nan,
            "p_adjusted": adj, "n_rejected": int(reject.sum())}


def univariate_pvalues(X: pd.DataFrame, y, race_id=None, min_n: int = 100) -> pd.DataFrame:
    """Per-feature correlation test of (race-demeaned) feature vs outcome.

    Demeaning within race removes the race-level component, so the test
    asks "does this feature separate runners *within* a race?" — the
    question the conditional logit answers.
    """
    yv = pd.Series(np.asarray(y, float), index=X.index)
    rows = []
    for c in X.columns:
        x = pd.to_numeric(X[c], errors="coerce")
        d = pd.DataFrame({"x": x, "y": yv})
        if race_id is not None:
            r = pd.Series(np.asarray(race_id), index=X.index)
            d["x"] = d["x"] - d["x"].groupby(r).transform("mean")
            d["y"] = d["y"] - d["y"].groupby(r).transform("mean")
        d = d.dropna()
        if len(d) < min_n or d["x"].std() == 0:
            rows.append({"feature": c, "n": len(d), "corr": np.nan, "p_value": np.nan})
            continue
        res = stats.pearsonr(d["x"], d["y"])
        rows.append({"feature": c, "n": len(d), "corr": float(res.statistic), "p_value": float(res.pvalue)})
    return pd.DataFrame(rows).set_index("feature")


# ---------------------------------------------------------------------------
# Model-X knockoffs
# ---------------------------------------------------------------------------

def gaussian_knockoffs(X: np.ndarray, random_state: int = 0, method: str = "equi") -> np.ndarray:
    """Sample second-order Gaussian model-X knockoffs for X (n x p).

    Equi-correlated construction: s_j = min(2 * lambda_min(Sigma), 1) on the
    correlation scale. X_tilde | X ~ N(mu + (X - mu)(I - Sigma^-1 S),
    2S - S Sigma^-1 S).
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, float)
    n, p = X.shape
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    Sigma = LedoitWolf().fit(Z).covariance_
    Sigma = (Sigma + Sigma.T) / 2
    lam_min = float(np.linalg.eigvalsh(Sigma).min())
    s_val = min(2 * max(lam_min, 1e-6), 1.0) * (1 - 1e-4)
    S = np.eye(p) * s_val
    Sigma_inv = np.linalg.pinv(Sigma)
    mean = Z @ (np.eye(p) - Sigma_inv @ S)
    V = 2 * S - S @ Sigma_inv @ S
    V = (V + V.T) / 2 + 1e-8 * np.eye(p)
    try:
        L = np.linalg.cholesky(V)
    except np.linalg.LinAlgError:
        w, U = np.linalg.eigh(V)
        L = U @ np.diag(np.sqrt(np.clip(w, 1e-10, None)))
    Zt = mean + rng.standard_normal((n, p)) @ L.T
    return Zt * sd + mu


def knockoff_threshold(W: np.ndarray, fdr: float = 0.1, offset: int = 1) -> float:
    """Knockoff (offset=0) / knockoff+ (offset=1) data-dependent threshold."""
    Wabs = np.sort(np.unique(np.abs(W[W != 0])))
    for t in Wabs:
        ratio = (offset + np.sum(W <= -t)) / max(1, np.sum(W >= t))
        if ratio <= fdr:
            return float(t)
    return np.inf


def knockoff_filter(X, y, fdr: float = 0.1, binary: bool | None = None, random_state: int = 0,
                    lasso_alpha: float | None = None, offset: int = 1) -> dict:
    """Model-X knockoff filter with the lasso coefficient-difference statistic.

    Returns the boolean selection mask (in column order), the W statistics
    and the threshold. Set ``binary`` to force the classification path.
    """
    Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    names = list(Xdf.columns)
    Xa = Xdf.astype(float).values
    yv = np.asarray(y, float)
    keep = ~np.isnan(Xa).any(1) & ~np.isnan(yv)
    Xa, yv = Xa[keep], yv[keep]
    if binary is None:
        binary = set(np.unique(yv)).issubset({0.0, 1.0})
    Xt = gaussian_knockoffs(Xa, random_state=random_state)
    XX = np.hstack([Xa, Xt])
    XX = (XX - XX.mean(0)) / np.where(XX.std(0) == 0, 1, XX.std(0))
    p = Xa.shape[1]
    if binary:
        C = 1.0 / (lasso_alpha * len(yv)) if lasso_alpha else 1.0
        m = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=5000, tol=1e-4)
        m.fit(XX, yv.astype(int))
        coef = m.coef_.ravel()
    else:
        m = Lasso(alpha=lasso_alpha, max_iter=20000) if lasso_alpha else LassoCV(cv=5, random_state=random_state, max_iter=20000)
        m.fit(XX, yv)
        coef = m.coef_
    W = np.abs(coef[:p]) - np.abs(coef[p:])
    thr = knockoff_threshold(W, fdr=fdr, offset=offset)
    selected = W >= thr if np.isfinite(thr) else np.zeros(p, dtype=bool)
    return {"selected": pd.Series(selected, index=names), "W": pd.Series(W, index=names),
            "threshold": thr, "n_selected": int(selected.sum()), "n_used": int(keep.sum())}


# ---------------------------------------------------------------------------
# mRMR
# ---------------------------------------------------------------------------

def mrmr_select(X: pd.DataFrame, y, k: int = 20, discrete_target: bool = True,
                random_state: int = 0, n_neighbors: int = 5) -> pd.DataFrame:
    """Greedy max-relevance / min-redundancy (MID criterion) selection.

    Relevance = MI(feature, target); redundancy = mean MI(feature, chosen).
    Returns the chosen features in order with their scores.
    """
    Xd = X.astype(float).copy()
    Xd = Xd.fillna(Xd.median())
    yv = np.asarray(y)
    mi_fn = mutual_info_classif if discrete_target else mutual_info_regression
    rel = pd.Series(mi_fn(Xd.values, yv, random_state=random_state, n_neighbors=n_neighbors), index=Xd.columns)
    chosen, scores = [], []
    remaining = list(Xd.columns)
    red_cache: dict[str, pd.Series] = {}
    for _ in range(min(k, len(remaining))):
        best, best_score = None, -np.inf
        for c in remaining:
            if chosen:
                red = np.mean([red_cache[ch][c] for ch in chosen])
            else:
                red = 0.0
            score = rel[c] - red
            if score > best_score:
                best, best_score = c, score
        chosen.append(best); scores.append(best_score); remaining.remove(best)
        red_cache[best] = pd.Series(
            mutual_info_regression(Xd[remaining].values, Xd[best].values, random_state=random_state,
                                   n_neighbors=n_neighbors) if remaining else [], index=remaining)
    return pd.DataFrame({"feature": chosen, "mrmr_score": scores, "relevance": rel[chosen].values})
