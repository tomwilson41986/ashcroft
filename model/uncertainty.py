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

And, because a backtest Sharpe is a selected maximum rather than a draw
(Section VI.2.6 / IX: "DSR and PBO are mandatory, not optional"):

    ConfigurationLedger         The honest count of configurations tried,
                                persisted so the count survives the session
    probabilistic_sharpe_ratio  P(true SR > benchmark) given skew/kurtosis
    expected_max_sharpe         E[max SR] under the null over N trials
    deflated_sharpe_ratio       PSR against that expected maximum
    probability_of_backtest_overfitting   CSCV logit of the selected
                                configuration's out-of-sample rank
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


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


# ---------------------------------------------------------------------------
# Backtest overfitting: DSR and PBO (Bailey & Lopez de Prado)
# ---------------------------------------------------------------------------
#
# A backtest Sharpe is not a draw from the strategy's return distribution, it
# is the *maximum* over every configuration that was tried. With enough tries
# a pure-noise strategy clears any fixed threshold, so the honest question is
# whether the observed Sharpe beats what the best of N coin flips would have
# produced. That needs the count of configurations tried, which is why
# ``ConfigurationLedger`` exists: the number is worthless if it is recalled
# after the fact.

EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns, ddof: int = 1) -> float:
    """Per-observation Sharpe of a return series. Deliberately *not*
    annualised: the DSR/PSR formulas want the SR and the observation count in
    the same frequency, and annualising first double-counts the sample length."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=ddof)
    return float(r.mean() / sd) if sd > 0 else float("nan")


def _return_moments(returns) -> tuple[float, int, float, float]:
    """(SR, n, skew, kurtosis) with the plain standardised moments the
    Bailey & Lopez de Prado formulas are written in (kurtosis = 3 under
    normality, i.e. not the excess)."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return (float("nan"), len(r), 0.0, 3.0)
    return (sharpe_ratio(r), len(r), float(stats.skew(r)), float(stats.kurtosis(r, fisher=False)))


def probabilistic_sharpe_ratio(returns=None, sr_benchmark: float = 0.0, *, sr: float | None = None,
                               n: int | None = None, skew: float = 0.0, kurtosis: float = 3.0) -> float:
    """P(true SR > ``sr_benchmark``) given the observed SR and the higher
    moments (Bailey & Lopez de Prado 2012):

        PSR = Phi( (SR - SR*) sqrt(n - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2) )

    Pass ``returns`` to estimate SR, n and the moments from the series, or
    supply ``sr``/``n``/``skew``/``kurtosis`` directly. Negative skew and fat
    tails inflate the denominator, which is the point: the same Sharpe earned
    by selling tails is worth less."""
    if returns is not None:
        sr, n, skew, kurtosis = _return_moments(returns)
    if sr is None or n is None or n < 2 or not np.isfinite(sr):
        return float("nan")
    var = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr ** 2
    if not np.isfinite(var) or var <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(var)))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """E[max SR] over ``n_trials`` independent trials of a strategy with no
    skill, whose Sharpes vary with ``sr_variance`` (Bailey & Lopez de Prado
    2014, the Gumbel approximation to the maximum of N normals):

        SR* = sd(SR) [ (1 - g) Z^-1(1 - 1/N) + g Z^-1(1 - 1/(N e)) ]
    """
    if n_trials is None or n_trials < 2 or not np.isfinite(sr_variance) or sr_variance <= 0:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(returns=None, n_trials: int | None = None, trial_sharpes=None,
                          sr_variance: float | None = None, *, sr: float | None = None,
                          n: int | None = None, skew: float = 0.0, kurtosis: float = 3.0) -> dict:
    """Deflated Sharpe ratio: PSR measured against the Sharpe the *best of
    ``n_trials``* would have reached by luck alone.

    ``trial_sharpes`` (the Sharpe of every configuration tried, e.g.
    ``ConfigurationLedger.sharpes()``) supplies both the trial count and the
    cross-trial variance; otherwise give ``n_trials`` and ``sr_variance``.
    DSR < 0.95 means the backtest has not cleared its own selection bias."""
    if returns is not None:
        sr, n, skew, kurtosis = _return_moments(returns)
    if trial_sharpes is not None:
        t = np.asarray(trial_sharpes, float)
        t = t[np.isfinite(t)]
        if n_trials is None:
            n_trials = int(len(t))
        if sr_variance is None:
            sr_variance = float(t.var(ddof=1)) if len(t) > 1 else 0.0
    sr_star = expected_max_sharpe(int(n_trials or 1), float(sr_variance or 0.0))
    return {"sharpe": sr, "n_obs": n, "skew": skew, "kurtosis": kurtosis,
            "n_trials": int(n_trials or 1), "sr_variance": float(sr_variance or 0.0), "sr_star": sr_star,
            "psr_vs_zero": probabilistic_sharpe_ratio(sr=sr, n=n, skew=skew, kurtosis=kurtosis, sr_benchmark=0.0),
            "dsr": probabilistic_sharpe_ratio(sr=sr, n=n, skew=skew, kurtosis=kurtosis, sr_benchmark=sr_star)}


def _block_sharpes(s1: np.ndarray, s2: np.ndarray, cnt: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Sharpe of every configuration on every block subset, from block sums.

    sel is (C, S) in {0,1}; s1/s2 are (S, N) sums of x and x^2; returns (C, N).
    Sharpe is not additive but its ingredients are, so the whole CSCV grid is
    two matrix products rather than C x N passes over the return matrix."""
    n = sel @ cnt
    m = (sel @ s1) / n[:, None]
    var = ((sel @ s2) - n[:, None] * m ** 2) / np.maximum(n - 1, 1)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(var > 0, m / np.sqrt(np.where(var > 0, var, np.nan)), np.nan)


def probability_of_backtest_overfitting(perf_matrix, n_splits: int = 16, max_combinations: int = 20000,
                                        random_state: int = 0) -> dict:
    """PBO by combinatorially-symmetric cross-validation (Bailey, Borwein,
    Lopez de Prado & Zhu 2015).

    ``perf_matrix`` is (T observations x N configurations) of per-period
    returns — one column per configuration that was tried, same rows for all.
    The T rows are cut into ``n_splits`` contiguous blocks; for every way of
    splitting the blocks in half, the configuration with the best in-sample
    Sharpe is selected and its *rank* among the N on the held-out half is
    recorded as the relative rank w in (0, 1) and its logit ln(w / (1 - w)).

        PBO = P(logit <= 0) = P(the selected configuration lands in the
        bottom half out of sample)

    PBO near 0.5 says selection is a coin flip: the backtest ranking carries
    no out-of-sample information. Also returns the performance-degradation
    line (OOS Sharpe regressed on IS Sharpe of the selected configuration);
    a negative slope is the signature of overfitting."""
    M = perf_matrix.values if isinstance(perf_matrix, pd.DataFrame) else np.asarray(perf_matrix, float)
    M = np.asarray(M, float)
    T, N = M.shape
    if N < 2:
        raise ValueError("PBO needs at least 2 configurations to rank")
    if n_splits % 2 or n_splits < 2:
        raise ValueError("n_splits must be even and >= 2")
    if T < n_splits * 2:
        raise ValueError(f"need at least {n_splits * 2} observations for {n_splits} splits")

    block = T // n_splits
    M = M[: block * n_splits]                                  # CSCV wants equal blocks
    blocks = M.reshape(n_splits, block, N)
    s1 = np.nansum(blocks, axis=1)
    s2 = np.nansum(blocks ** 2, axis=1)
    cnt = np.sum(np.isfinite(blocks[:, :, 0]), axis=1).astype(float)

    combos = list(combinations(range(n_splits), n_splits // 2))
    if len(combos) > max_combinations:
        rng = np.random.default_rng(random_state)
        combos = [combos[i] for i in rng.choice(len(combos), max_combinations, replace=False)]
    sel = np.zeros((len(combos), n_splits))
    for i, c in enumerate(combos):
        sel[i, list(c)] = 1.0

    sr_is = _block_sharpes(s1, s2, cnt, sel)
    sr_oos = _block_sharpes(s1, s2, cnt, 1.0 - sel)
    rows = np.arange(len(combos))
    best = np.nanargmax(np.where(np.isfinite(sr_is), sr_is, -np.inf), axis=1)
    ranks = stats.rankdata(np.where(np.isfinite(sr_oos), sr_oos, -np.inf), axis=1)   # 1 = worst
    w = ranks[rows, best] / (N + 1.0)
    logits = np.log(w / (1.0 - w))
    is_best, oos_best = sr_is[rows, best], sr_oos[rows, best]
    ok = np.isfinite(is_best) & np.isfinite(oos_best)
    slope = float(np.polyfit(is_best[ok], oos_best[ok], 1)[0]) if ok.sum() > 2 else float("nan")
    return {"pbo": float(np.mean(logits <= 0)), "logits": logits, "n_combinations": int(len(combos)),
            "n_configs": int(N), "n_splits": int(n_splits), "median_logit": float(np.median(logits)),
            "degradation_slope": slope, "prob_oos_loss": float(np.mean(oos_best[ok] < 0)) if ok.any() else float("nan"),
            "sr_is_selected": is_best, "sr_oos_selected": oos_best}


class ConfigurationLedger:
    """The honest count of configurations tried (Section VI.2.6).

    DSR is only as honest as ``n_trials``, and a count reconstructed from
    memory at the end of a research programme is always too small — the
    variants that were abandoned after one look are exactly the ones that
    inflate the maximum. So record every fit as it happens, to a JSON file
    that outlives the session, and read ``n_trials`` back off the ledger.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None
        self.trials: list[dict] = []
        if self.path is not None and self.path.exists():
            self.trials = json.loads(self.path.read_text()).get("trials", [])

    def record(self, config: dict | None = None, returns=None, sharpe: float | None = None, **metrics) -> dict:
        """Log one configuration. ``returns`` (a per-bet or per-day P&L
        series) is summarised to its Sharpe and moments; nothing is stored raw."""
        entry = {"config": dict(config or {}), **metrics}
        if returns is not None:
            sr, n, sk, ku = _return_moments(returns)
            entry.update({"sharpe": sr, "n_obs": n, "skew": sk, "kurtosis": ku})
        if sharpe is not None:
            entry["sharpe"] = float(sharpe)
        self.trials.append(entry)
        self.save()
        return entry

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    def sharpes(self) -> np.ndarray:
        return np.array([t.get("sharpe", np.nan) for t in self.trials], float)

    def best(self) -> dict | None:
        s = self.sharpes()
        return self.trials[int(np.nanargmax(s))] if np.isfinite(s).any() else None

    def deflated_sharpe(self, returns=None, **kw) -> dict:
        """DSR of ``returns`` (default: the best trial on the ledger) deflated
        by every trial the ledger has seen."""
        if returns is None:
            b = self.best() or {}
            kw = {"sr": b.get("sharpe"), "n": b.get("n_obs"), "skew": b.get("skew", 0.0),
                  "kurtosis": b.get("kurtosis", 3.0), **kw}
        return deflated_sharpe_ratio(returns, trial_sharpes=self.sharpes(), n_trials=self.n_trials, **kw)

    def to_frame(self) -> pd.DataFrame:
        return pd.json_normalize(self.trials) if self.trials else pd.DataFrame()

    def save(self) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"n_trials": self.n_trials, "trials": self.trials}, indent=2, default=float))
