"""
Uncertainty quantification for staking inputs.

Why: Kelly consumes a probability, but the model gives a point estimate
with unknown error. Betting the point estimate over-stakes exactly where
the model is least sure. Tools:

    SplitConformalRegressor     Distribution-free prediction intervals on
                                the model's regression target (log BFSP)
                                with finite-sample coverage (Vovk et al.)
    prob_interval_from_log_bfsp Turn a log-BFSP interval into a win-prob
                                interval
    conformal_kelly_stake       Fractional Kelly sized from the
                                *conservative* end of the interval
    shrunk_probability          Baker-McHale-style shrinkage of the point
                                probability by interval width
    block_bootstrap             Resample whole races / meetings for
                                parameter and P&L uncertainty
    stationary_bootstrap_indices  Politis-Romano for daily P&L series
    bootstrap_ci                Percentile CI helper
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Conformal prediction
# ---------------------------------------------------------------------------

class SplitConformalRegressor:
    """Split (inductive) conformal intervals on residuals.

    Fit on a *calibration* set that the base model has not trained on.
    Optionally normalise residuals by a scale estimate (e.g. a second
    model predicting |residual|) so intervals adapt to local difficulty.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.q_: float = np.nan
        self.n_cal_: int = 0

    def fit(self, y_cal, y_pred_cal, scale_cal=None):
        y, yp = np.asarray(y_cal, float), np.asarray(y_pred_cal, float)
        s = np.ones_like(y) if scale_cal is None else np.asarray(scale_cal, float)
        r = np.abs(y - yp) / s
        r = r[~np.isnan(r)]
        n = len(r)
        k = int(np.ceil((n + 1) * (1 - self.alpha)))
        k = min(max(k, 1), n)
        self.q_ = float(np.sort(r)[k - 1])
        self.n_cal_ = n
        return self

    def interval(self, y_pred, scale=None):
        yp = np.asarray(y_pred, float)
        s = np.ones_like(yp) if scale is None else np.asarray(scale, float)
        return yp - self.q_ * s, yp + self.q_ * s

    def coverage(self, y, y_pred, scale=None) -> float:
        lo, hi = self.interval(y_pred, scale)
        y = np.asarray(y, float)
        return float(np.mean((y >= lo) & (y <= hi)))


def prob_interval_from_log_bfsp(pred_log_bfsp, q: float, race_id=None):
    """Win-probability interval implied by a symmetric log-BFSP interval.

    p = 1/BFSP, so p_hi = exp(-(pred - q)), p_lo = exp(-(pred + q)).
    If ``race_id`` is given, bounds are rescaled by the same factor that
    normalises the point probabilities within each race.
    """
    yp = np.asarray(pred_log_bfsp, float)
    p_point = np.exp(-yp); p_lo = np.exp(-(yp + q)); p_hi = np.exp(-(yp - q))
    if race_id is not None:
        s = pd.Series(p_point).groupby(np.asarray(race_id)).transform("sum").values
        p_point, p_lo, p_hi = p_point / s, p_lo / s, p_hi / s
    return p_lo, np.clip(p_point, 0, 1), np.clip(p_hi, 0, 1)


def kelly_fraction(p, odds, commission: float = 0.05) -> np.ndarray:
    """Full-Kelly fraction for a back bet at decimal ``odds`` net of commission."""
    p, o = np.asarray(p, float), np.asarray(odds, float)
    b = (o - 1) * (1 - commission)
    f = (p * b - (1 - p)) / np.where(b > 0, b, np.nan)
    return np.clip(np.nan_to_num(f, nan=0.0), 0, 1)


def conformal_kelly_stake(p_lo, p_point, odds, fraction: float = 0.25, commission: float = 0.05,
                          mode: str = "lower") -> np.ndarray:
    """Fractional Kelly with the interval as the uncertainty input.

    mode="lower": stake on the lower probability bound (bet only if even
                  the pessimistic probability shows an edge).
    mode="shrink": stake on the point estimate, scaled by 1 - width/point
                   (wide interval relative to the estimate -> smaller stake).
    """
    p_lo, p_pt, o = (np.asarray(v, float) for v in (p_lo, p_point, odds))
    if mode == "lower":
        return fraction * kelly_fraction(p_lo, o, commission)
    if mode == "shrink":
        width = np.clip(p_pt - p_lo, 0, None)
        scale = np.clip(1 - width / np.where(p_pt > 0, p_pt, np.nan), 0, 1)
        return fraction * kelly_fraction(p_pt, o, commission) * np.nan_to_num(scale)
    raise ValueError("mode must be 'lower' or 'shrink'")


def shrunk_probability(p_point, p_lo, p_hi, k: float = 1.0) -> np.ndarray:
    """Shrink the point probability toward the market-neutral side by
    k * half-width (Baker & McHale flavour of ability shrinkage)."""
    p, lo, hi = (np.asarray(v, float) for v in (p_point, p_lo, p_hi))
    return np.clip(p - k * (hi - lo) / 2, 1e-6, 1 - 1e-6)


# ---------------------------------------------------------------------------
# Bootstrap for clustered data
# ---------------------------------------------------------------------------

def block_bootstrap(df: pd.DataFrame, group_col: str, stat_fn, n_boot: int = 500, random_state: int = 0) -> np.ndarray:
    """Cluster bootstrap: resample whole groups (races or meetings) with
    replacement and recompute ``stat_fn(df_resampled)``."""
    rng = np.random.default_rng(random_state)
    groups = df[group_col].values
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        out.append(stat_fn(df.iloc[idx]))
    return np.asarray(out, float)


def stationary_bootstrap_indices(n: int, mean_block: float, n_boot: int, random_state: int = 0) -> np.ndarray:
    """Politis & Romano (1994) stationary bootstrap index sets (n_boot x n)."""
    rng = np.random.default_rng(random_state)
    p = 1.0 / mean_block
    out = np.empty((n_boot, n), dtype=int)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(n)
        restart = rng.random(n) < p
        for i in range(1, n):
            idx[i] = rng.integers(n) if restart[i] else (idx[i - 1] + 1) % n
        out[b] = idx
    return out


def bootstrap_ci(samples, level: float = 0.95) -> tuple[float, float]:
    s = np.asarray(samples, float)
    a = (1 - level) / 2
    return float(np.nanquantile(s, a)), float(np.nanquantile(s, 1 - a))
