"""
Causal-inference designs for racing interventions.

Why: "first-time blinkers", "gelded since last run", "wind surgery",
trainer switches and class drops are *chosen* by connections in response
to form, so a naive before/after form delta is confounded. The tools
here estimate intervention effects with the standard observational
designs from epidemiology:

    fit_propensity          P(treated | X), logistic with standardisation
    ipw_effect              Inverse-probability-weighted ATE / ATT
    aipw_effect             Doubly-robust (AIPW) with 2-fold cross-fitting
                            and influence-function standard errors
    did_2x2 / did_panel     Difference-in-differences (2x2 and two-way FE
                            with unit-clustered SEs)
    rd_estimate             Regression discontinuity at a threshold
                            (handicap-mark / class boundaries)
    e_value                 VanderWeele & Ding (2017) sensitivity to
                            unmeasured confounding
    first_time_flag         Lag-safe "first time X" indicator
    intervention_study      One-call wrapper: propensity + AIPW + E-value

Nothing here is a betting feature by itself. The estimates tell you
whether an intervention has a real effect, its size and how fragile it
is; the *feature* is then the treatment flag (already in the pipeline)
with a prior on its coefficient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _default_propensity_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def _default_outcome_model(binary: bool):
    if binary:
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def _is_binary(y) -> bool:
    u = np.unique(np.asarray(y)[~np.isnan(np.asarray(y, float))])
    return len(u) <= 2 and set(u).issubset({0.0, 1.0})


def _predict(model, X, binary):
    return model.predict_proba(X)[:, 1] if binary else model.predict(X)


# ---------------------------------------------------------------------------
# Propensity / weighting
# ---------------------------------------------------------------------------

def fit_propensity(X, t, model=None, clip=(0.02, 0.98)) -> np.ndarray:
    """Estimated P(T=1 | X), clipped away from 0/1 for stable weights."""
    m = clone(model) if model is not None else _default_propensity_model()
    m.fit(np.asarray(X, float), np.asarray(t, int))
    return np.clip(m.predict_proba(np.asarray(X, float))[:, 1], *clip)


def ipw_effect(y, t, ps, estimand: str = "ATE", stabilised: bool = True) -> dict:
    """Hajek (ratio) IPW estimate of the ATE or ATT with a plug-in SE."""
    y, t, e = (np.asarray(v, float) for v in (y, t, ps))
    if estimand.upper() == "ATE":
        w1, w0 = t / e, (1 - t) / (1 - e)
    elif estimand.upper() == "ATT":
        w1, w0 = t, (1 - t) * e / (1 - e)
    else:
        raise ValueError("estimand must be ATE or ATT")
    if stabilised:
        w1, w0 = w1 / w1.sum(), w0 / w0.sum()
        mu1, mu0 = np.sum(w1 * y), np.sum(w0 * y)
    else:
        mu1, mu0 = np.mean(w1 * y), np.mean(w0 * y)
    # Linearised variance of the Hajek difference
    n = len(y)
    a1 = w1 * (y - mu1) * (n if stabilised else 1)
    a0 = w0 * (y - mu0) * (n if stabilised else 1)
    se = float(np.sqrt(np.var(a1 - a0, ddof=1) / n))
    est = float(mu1 - mu0)
    return {"estimand": estimand.upper(), "estimate": est, "se": se,
            "ci95": (est - 1.96 * se, est + 1.96 * se), "mu1": float(mu1), "mu0": float(mu0),
            "n_treated": int(t.sum()), "n_control": int(n - t.sum()),
            "ess_treated": float(np.sum(w1) ** 2 / np.sum(w1**2)),
            "ess_control": float(np.sum(w0) ** 2 / np.sum(w0**2))}


def aipw_effect(y, t, X, propensity_model=None, outcome_model=None, n_folds: int = 2,
                clip=(0.02, 0.98), random_state: int = 0) -> dict:
    """Augmented IPW (doubly robust) ATE with cross-fitting.

    Consistent if EITHER the propensity model OR the outcome model is
    right. Influence-function SE. Also returns the ATT.
    """
    y, t, X = np.asarray(y, float), np.asarray(t, int), np.asarray(X, float)
    n = len(y)
    binary = _is_binary(y)
    e = np.zeros(n); mu1 = np.zeros(n); mu0 = np.zeros(n)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for tr, te in kf.split(X):
        pm = clone(propensity_model) if propensity_model is not None else _default_propensity_model()
        pm.fit(X[tr], t[tr])
        e[te] = np.clip(pm.predict_proba(X[te])[:, 1], *clip)
        for arm, store in ((1, mu1), (0, mu0)):
            idx = tr[t[tr] == arm]
            om = clone(outcome_model) if outcome_model is not None else _default_outcome_model(binary)
            if binary and len(np.unique(y[idx])) < 2:
                store[te] = y[idx].mean()
                continue
            om.fit(X[idx], y[idx])
            store[te] = _predict(om, X[te], binary)
    psi = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
    ate = float(psi.mean()); se = float(psi.std(ddof=1) / np.sqrt(n))
    # ATT: E[Y1 - Y0 | T=1]
    p1 = t.mean()
    psi_att = (t * (y - mu0) - (1 - t) * e / (1 - e) * (y - mu0)) / p1
    att = float(psi_att.mean()); se_att = float(psi_att.std(ddof=1) / np.sqrt(n))
    return {"estimate": ate, "se": se, "ci95": (ate - 1.96 * se, ate + 1.96 * se),
            "att": att, "att_se": se_att, "att_ci95": (att - 1.96 * se_att, att + 1.96 * se_att),
            "mu1_mean": float(mu1.mean()), "mu0_mean": float(mu0.mean()),
            "propensity_range": (float(e.min()), float(e.max())),
            "n_treated": int(t.sum()), "n_control": int(n - t.sum()), "binary_outcome": binary}


# ---------------------------------------------------------------------------
# Difference-in-differences
# ---------------------------------------------------------------------------

def did_2x2(y, treated, post) -> dict:
    """Classic 2x2 DiD: (T,post - T,pre) - (C,post - C,pre)."""
    y, g, p = (np.asarray(v, float) for v in (y, treated, post))
    cells = {}
    for gi in (0, 1):
        for pi in (0, 1):
            m = (g == gi) & (p == pi)
            cells[(gi, pi)] = (y[m].mean() if m.any() else np.nan, y[m].var(ddof=1) / max(m.sum(), 1) if m.sum() > 1 else np.nan, int(m.sum()))
    est = (cells[(1, 1)][0] - cells[(1, 0)][0]) - (cells[(0, 1)][0] - cells[(0, 0)][0])
    se = float(np.sqrt(np.nansum([c[1] for c in cells.values()])))
    return {"estimate": float(est), "se": se, "ci95": (est - 1.96 * se, est + 1.96 * se),
            "cells": {f"treated={k[0]},post={k[1]}": {"mean": v[0], "n": v[2]} for k, v in cells.items()}}


def did_panel(df: pd.DataFrame, unit_col: str, time_col: str, treat_post_col: str, outcome_col: str,
              covariate_cols=None) -> dict:
    """Two-way fixed-effects DiD via within (double-demeaning) transformation.

    y_it = a_i + g_t + tau * D_it + X_it b + e_it, SE clustered by unit.
    ``treat_post_col`` is the interaction (1 when unit i is treated at t).
    """
    d = df[[unit_col, time_col, treat_post_col, outcome_col] + list(covariate_cols or [])].dropna().copy()
    cols = [treat_post_col] + list(covariate_cols or [])
    Z = d[cols + [outcome_col]].astype(float)
    # double demeaning (approximate for unbalanced panels; iterate twice)
    for _ in range(2):
        Z = Z - Z.groupby(d[unit_col]).transform("mean")
        Z = Z - Z.groupby(d[time_col]).transform("mean") + Z.mean()
    X = Z[cols].values; y = Z[outcome_col].values
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    # cluster-robust sandwich by unit
    meat = np.zeros((X.shape[1], X.shape[1]))
    groups = d[unit_col].values
    for gkey in np.unique(groups):
        m = groups == gkey
        s = X[m].T @ u[m]
        meat += np.outer(s, s)
    G = len(np.unique(groups))
    V = XtX_inv @ meat @ XtX_inv * (G / max(G - 1, 1))
    se = float(np.sqrt(V[0, 0]))
    return {"estimate": float(beta[0]), "se": se, "ci95": (beta[0] - 1.96 * se, beta[0] + 1.96 * se),
            "n_obs": int(len(y)), "n_units": int(G), "coef": dict(zip(cols, beta.tolist()))}


# ---------------------------------------------------------------------------
# Regression discontinuity
# ---------------------------------------------------------------------------

def rd_estimate(x, y, cutoff: float, bandwidth: float, kernel: str = "triangular") -> dict:
    """Sharp RD: local-linear fits either side of ``cutoff`` within ``bandwidth``.

    Use e.g. x = official rating, cutoff = class/handicap-band boundary,
    y = subsequent performance, to see whether horses just above/below a
    mark behave differently (handicapper targeting, class relief).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    z = x - cutoff
    m = np.abs(z) <= bandwidth
    z, y = z[m], y[m]
    w = 1 - np.abs(z) / bandwidth if kernel == "triangular" else np.ones_like(z)

    def side(mask):
        Xs = np.column_stack([np.ones(mask.sum()), z[mask]])
        W = np.diag(w[mask])
        XtWX_inv = np.linalg.pinv(Xs.T @ W @ Xs)
        b = XtWX_inv @ Xs.T @ W @ y[mask]
        r = y[mask] - Xs @ b
        meat = Xs.T @ (W * (r**2)[None, :] * W) @ Xs
        V = XtWX_inv @ meat @ XtWX_inv
        return b[0], np.sqrt(max(V[0, 0], 0)), int(mask.sum())

    aL, seL, nL = side(z < 0)
    aR, seR, nR = side(z >= 0)
    tau = float(aR - aL); se = float(np.sqrt(seL**2 + seR**2))
    return {"estimate": tau, "se": se, "ci95": (tau - 1.96 * se, tau + 1.96 * se),
            "n_left": nL, "n_right": nR, "left_intercept": float(aL), "right_intercept": float(aR)}


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------

def e_value(rr: float, ci_limit: float | None = None) -> dict:
    """E-value: the minimum strength of association (on the RR scale) an
    unmeasured confounder would need with BOTH treatment and outcome to
    explain away the estimate (VanderWeele & Ding 2017)."""
    def _e(r):
        r = float(r)
        if r < 1:
            r = 1 / r
        return r + np.sqrt(r * (r - 1))
    out = {"e_value": float(_e(rr))}
    if ci_limit is not None:
        # E-value for the CI limit closest to the null
        lim = float(ci_limit)
        if (rr >= 1 and lim <= 1) or (rr < 1 and lim >= 1):
            out["e_value_ci"] = 1.0
        else:
            out["e_value_ci"] = float(_e(lim))
    return out


def risk_ratio_from_means(mu1: float, mu0: float) -> float:
    return float(mu1 / mu0) if mu0 > 0 else np.nan


# ---------------------------------------------------------------------------
# Racing helpers
# ---------------------------------------------------------------------------

def first_time_flag(df: pd.DataFrame, col: str, group_col: str = "horse_name",
                    date_cols=("race_date", "race_time"), truthy=None) -> pd.Series:
    """Lag-safe 'first time this horse shows a truthy value in ``col``'.

    ``truthy`` is a callable on the raw column (default: non-empty string).
    """
    d = df[[group_col, col] + list(date_cols)].copy()
    d["_ord"] = np.arange(len(d))
    d = d.sort_values([group_col] + list(date_cols))
    if truthy is None:
        v = d[col].fillna("").astype(str).str.strip() != ""
    else:
        v = d[col].map(truthy).astype(bool)
    prior = v.groupby(d[group_col]).cumsum() - v.astype(int)
    flag = (v & (prior == 0)).astype(int)
    return pd.Series(flag.values, index=d["_ord"].values).sort_index().set_axis(df.index)


def intervention_study(df: pd.DataFrame, treatment_col: str, outcome_col: str, covariate_cols,
                       method: str = "aipw", **kw) -> dict:
    """Propensity + (A)IPW estimate of an intervention effect with E-value.

    Outcome can be binary (won) or continuous (performance figure in lbs).
    Covariates should be *pre-treatment* form features (lag-safe columns).
    """
    d = df[[treatment_col, outcome_col] + list(covariate_cols)].dropna()
    y = d[outcome_col].astype(float).values
    t = d[treatment_col].astype(int).values
    X = d[list(covariate_cols)].astype(float).values
    if t.sum() < 20 or (len(t) - t.sum()) < 20:
        raise ValueError("need at least 20 treated and 20 control rows")
    if method == "aipw":
        res = aipw_effect(y, t, X, **kw)
    else:
        ps = fit_propensity(X, t)
        res = ipw_effect(y, t, ps, **kw)
    mu1 = res.get("mu1_mean", res.get("mu1")); mu0 = res.get("mu0_mean", res.get("mu0"))
    res["treatment"] = treatment_col; res["outcome"] = outcome_col; res["method"] = method
    if mu0 is not None and mu0 > 0 and mu1 is not None and _is_binary(y):
        rr = risk_ratio_from_means(mu1, mu0)
        lo, hi = res["ci95"]
        rr_lim = risk_ratio_from_means(mu0 + (lo if rr >= 1 else hi), mu0)
        res["risk_ratio"] = rr
        res.update(e_value(rr, rr_lim))
    return res
