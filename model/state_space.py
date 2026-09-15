"""
State-space (Kalman) horse ratings — Section IIA.1 of the racing² master
framework, "the highest-value import".

Ability is a latent state that drifts between runs; each run is a noisy
observation of it:

    θ_t = θ_{t−1} + w_t,   w_t ~ N(0, q · Δyears)     ability drifts (more over a long gap)
    y_t = θ_t + v_t,       v_t ~ N(0, r · r_scale_t)  the run is a noisy read of it

The filter returns, for every run, the PRIOR mean and variance — the
lag-safe rating before the run — plus the posterior after it. Two things
fall out that fixed-λ recency weighting cannot give: a per-horse
uncertainty (feeds Baker–McHale stake shrinkage), and gap-awareness (a
horse off 200 days automatically has a wider posterior).

Extensions from the framework: q as a function of career stage
(``q_career``: multiplier for the first few runs), r as a function of race
conditions (``r_scale_col``: e.g. larger for tactical races so they are
discounted). q, r and the initial variance are fitted by maximum
likelihood through the prediction-error decomposition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _sorted_positions(df: pd.DataFrame, horse_col: str, date_col: str, time_col: str | None):
    keys = [horse_col, date_col] + ([time_col] if time_col and time_col in df.columns else [])
    tmp = df[keys].reset_index(drop=True)
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    return tmp.sort_values(keys, kind="stable").index.values, tmp


def kalman_filter_runs(values: np.ndarray, days: np.ndarray, horses: np.ndarray, q: float, r: float, init_var: float,
                       init_mean: float | np.ndarray, q_career: np.ndarray | None = None, r_scale: np.ndarray | None = None):
    """Local-level filter over rows sorted by horse then time.

    Returns prior_mean, prior_var, post_mean, post_var, n_prior_obs, loglik contribution per row.
    NaN observations are skipped (state propagates, no update).
    """
    n = len(values)
    pm = np.full(n, np.nan); pv = np.full(n, np.nan); qm = np.full(n, np.nan); qv = np.full(n, np.nan)
    nobs = np.zeros(n); ll = np.zeros(n)
    init_mean = np.broadcast_to(np.asarray(init_mean, float), (n,))
    cur = None; m = 0.0; P = 0.0; last_day = 0; k = 0
    for i in range(n):
        if horses[i] != cur:
            cur = horses[i]; m = init_mean[i]; P = init_var; last_day = days[i]; k = 0
        else:
            gap_years = max(days[i] - last_day, 0) / 365.0
            qi = q * (q_career[i] if q_career is not None else 1.0)
            P = P + qi * max(gap_years, 1.0 / 365.0)
        pm[i] = m; pv[i] = P; nobs[i] = k
        y = values[i]
        if not np.isnan(y):
            ri = r * (r_scale[i] if r_scale is not None else 1.0)
            S = P + ri
            e = y - m
            ll[i] = -0.5 * (np.log(2 * np.pi * S) + e * e / S)
            K = P / S
            m = m + K * e; P = (1 - K) * P
            k += 1; last_day = days[i]
        qm[i] = m; qv[i] = P
    return pm, pv, qm, qv, nobs, ll


def add_kalman_features(df: pd.DataFrame, value_col: str, prefix: str = "kf", horse_col: str = "horse_name",
                        date_col: str = "race_date", time_col: str | None = "race_time", race_col: str = "raceid",
                        q: float = 25.0, r: float = 50.0, init_var: float = 100.0, init_mean: float | None = None,
                        q_career_mult: float = 2.0, q_career_runs: int = 4, r_scale_col: str | None = None) -> pd.DataFrame:
    """Attach lag-safe Kalman rating features for ``value_col`` (e.g. perf_lbs,
    performance_rating, nmfp): {prefix}_rating (prior mean), {prefix}_sd,
    {prefix}_n, {prefix}_post (posterior after the run — NOT lag-safe),
    {prefix}_innov (surprise of the run), and within-race z / rank of the rating."""
    order, tmp = _sorted_positions(df, horse_col, date_col, time_col)
    vals = pd.to_numeric(df[value_col], errors="coerce").values.astype(float)[order]
    days = tmp.loc[order, date_col].values.astype("datetime64[D]").astype(np.int64)
    horses = tmp.loc[order, horse_col].astype(str).values
    if init_mean is None:
        init_mean = float(np.nanmean(vals))
    # career-stage drift multiplier: young/lightly raced horses move faster
    run_idx = pd.Series(np.arange(len(order))).groupby(horses).cumcount().values
    q_career = np.where(run_idx < q_career_runs, q_career_mult, 1.0)
    r_scale = pd.to_numeric(df[r_scale_col], errors="coerce").fillna(1.0).values[order] if r_scale_col else None
    pm, pv, qm, qv, nobs, _ = kalman_filter_runs(vals, days, horses, q, r, init_var, init_mean, q_career, r_scale)
    out = df.copy()
    for name, arr in ((f"{prefix}_rating", pm), (f"{prefix}_sd", np.sqrt(pv)), (f"{prefix}_n", nobs),
                      (f"{prefix}_post", qm), (f"{prefix}_innov", vals - pm)):
        full = np.full(len(df), np.nan); full[order] = arr; out[name] = full
    if race_col in out.columns:
        g = out.groupby(race_col)[f"{prefix}_rating"]
        out[f"{prefix}_z"] = (out[f"{prefix}_rating"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        out[f"{prefix}_rank"] = g.rank(ascending=False, method="min")
        out[f"{prefix}_vs_max"] = out[f"{prefix}_rating"] - g.transform("max")
    return out


def fit_kalman_params(df: pd.DataFrame, value_col: str, horse_col: str = "horse_name", date_col: str = "race_date",
                      time_col: str | None = "race_time", q_career_mult: float = 2.0, q_career_runs: int = 4,
                      r_scale_col: str | None = None, x0=(25.0, 50.0, 100.0)) -> dict:
    """Maximum-likelihood q, r, init_var via the prediction-error decomposition."""
    order, tmp = _sorted_positions(df, horse_col, date_col, time_col)
    vals = pd.to_numeric(df[value_col], errors="coerce").values.astype(float)[order]
    days = tmp.loc[order, date_col].values.astype("datetime64[D]").astype(np.int64)
    horses = tmp.loc[order, horse_col].astype(str).values
    init_mean = float(np.nanmean(vals))
    run_idx = pd.Series(np.arange(len(order))).groupby(horses).cumcount().values
    q_career = np.where(run_idx < q_career_runs, q_career_mult, 1.0)
    r_scale = pd.to_numeric(df[r_scale_col], errors="coerce").fillna(1.0).values[order] if r_scale_col else None

    def nll(theta):
        q, r, v0 = np.exp(theta)
        return -np.nansum(kalman_filter_runs(vals, days, horses, q, r, v0, init_mean, q_career, r_scale)[5])

    res = minimize(nll, np.log(np.asarray(x0, float)), method="Nelder-Mead", options={"maxiter": 300, "xatol": 1e-3, "fatol": 1e-2})
    q, r, v0 = np.exp(res.x)
    return {"q": float(q), "r": float(r), "init_var": float(v0), "nll": float(res.fun), "n_obs": int(np.sum(~np.isnan(vals))),
            "steady_state_gain": float(((q + np.sqrt(q * q + 4 * q * r)) / 2) / (((q + np.sqrt(q * q + 4 * q * r)) / 2) + r))}


def ewm_baseline(df: pd.DataFrame, value_col: str, horse_col: str = "horse_name", date_col: str = "race_date",
                 time_col: str | None = "race_time", half_life_days: float = 270.0) -> pd.Series:
    """Fixed-λ recency-weighted mean of prior values (the Part 3.1 baseline the
    Kalman rating must beat)."""
    order, tmp = _sorted_positions(df, horse_col, date_col, time_col)
    vals = pd.to_numeric(df[value_col], errors="coerce").values.astype(float)[order]
    days = tmp.loc[order, date_col].values.astype("datetime64[D]").astype(np.int64)
    horses = tmp.loc[order, horse_col].astype(str).values
    out = np.full(len(order), np.nan); cur = None; sw = sx = 0.0; last = 0
    for i in range(len(order)):
        if horses[i] != cur:
            cur = horses[i]; sw = sx = 0.0; last = days[i]
        if sw > 0:
            k = 0.5 ** ((days[i] - last) / half_life_days); out[i] = sx * k / (sw * k)
        if not np.isnan(vals[i]):
            k = 0.5 ** ((days[i] - last) / half_life_days) if sw > 0 else 1.0
            sw = sw * k + 1; sx = sx * k + vals[i]; last = days[i]
    full = np.full(len(df), np.nan); full[order] = out
    return pd.Series(full, index=df.index)
