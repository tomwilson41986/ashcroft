"""
Forecast-evaluation diagnostics imported from meteorology and sports
forecasting.

Why: the evaluator already reports log-loss / Brier / ROI, but a headline
score cannot say *why* a forecast scores as it does. These tools split the
score into calibration (reliability) and discrimination (resolution),
express it as skill relative to the Betfair market, and add rank-based
and drift diagnostics.

Contents
    murphy_decomposition        Brier = REL - RES + UNC (+ within-bin terms),
                                with the Ferro & Fricker (2012) bias correction
    brier_skill_score /         Skill relative to a reference forecast
    log_loss_skill_score        (use Betfair SP as the reference)
    ranked_probability_score    Ordinal outcomes (win / placed / unplaced)
    crps_ensemble / crps_gaussian   Continuous targets (ABM margins, times)
    reliability_table, expected_calibration_error
    concordance_index, kendall_tau_by_race   Rank quality within races
    kl_divergence, js_divergence, divergence_by_period   Drift monitor
    market_implied_probs, race_normalise
    scoring_report              One-call race-level report

All functions are pure numpy/pandas; nothing here touches the database.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm

EPS = 1e-12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arr(y, p):
    y = np.asarray(y, dtype=float).ravel()
    p = np.clip(np.asarray(p, dtype=float).ravel(), EPS, 1 - EPS)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y {y.shape} vs p {p.shape}")
    return y, p


def race_normalise(df: pd.DataFrame, p_col: str, race_col: str = "raceid") -> pd.Series:
    """Rescale probabilities so they sum to one within each race."""
    tot = df.groupby(race_col)[p_col].transform("sum")
    return df[p_col] / tot.replace(0, np.nan)


def market_implied_probs(df: pd.DataFrame, price_col: str = "bfsp", race_col: str = "raceid") -> pd.Series:
    """Betfair SP -> normalised implied win probability (overround removed)."""
    inv = 1.0 / pd.to_numeric(df[price_col], errors="coerce")
    tot = inv.groupby(df[race_col]).transform("sum")
    return inv / tot


def log_loss(y, p) -> float:
    y, p = _arr(y, p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    y, p = _arr(y, p)
    return float(np.mean((p - y) ** 2))


# ---------------------------------------------------------------------------
# Murphy decomposition
# ---------------------------------------------------------------------------

def murphy_decomposition(
    y, p, n_bins: int = 10, strategy: str = "quantile", bias_corrected: bool = True
) -> dict:
    """Murphy (1973) vector partition of the Brier score.

        Brier = REL - RES + UNC + WBV - WBC

    REL  reliability (calibration error; lower is better)
    RES  resolution (how far bin outcome rates sit from the base rate; higher is better)
    UNC  uncertainty (base-rate Brier; property of the data)
    WBV / WBC  within-bin variance / covariance (Stephenson, Coelho &
               Jolliffe 2008) so the identity holds exactly for un-binned
               forecasts.

    With ``bias_corrected`` the Ferro & Fricker (2012) small-sample
    correction is also returned (REL_c, RES_c, UNC_c); the raw
    decomposition over-states reliability and under-states uncertainty.
    """
    y, p = _arr(y, p)
    n = len(y)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    if len(edges) < 2:
        edges = np.array([0.0, 1.0])
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)

    ybar = y.mean()
    unc = ybar * (1 - ybar)
    rel = res = wbv = wbc = 0.0
    corr = 0.0  # sum over bins of n_k * v_k, v_k = o_k(1-o_k)/(n_k-1)
    table = []
    for k in np.unique(idx):
        m = idx == k
        nk = int(m.sum())
        pk, ok = p[m].mean(), y[m].mean()
        rel += nk * (pk - ok) ** 2
        res += nk * (ok - ybar) ** 2
        wbv += np.sum((p[m] - pk) ** 2)
        wbc += 2.0 * np.sum((p[m] - pk) * (y[m] - ok))
        if nk > 1:
            corr += nk * ok * (1 - ok) / (nk - 1)
        table.append({"bin": int(k), "n": nk, "p_mean": pk, "obs_rate": ok})

    rel, res, wbv, wbc, corr = rel / n, res / n, wbv / n, wbc / n, corr / n
    out = {
        "brier": float(np.mean((p - y) ** 2)),
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": float(unc),
        "within_bin_variance": float(wbv),
        "within_bin_covariance": float(wbc),
        "n": n,
        "n_bins": len(table),
        "bins": pd.DataFrame(table),
    }
    out["identity_gap"] = float(out["brier"] - (rel - res + unc + wbv - wbc))
    if bias_corrected and n > 1:
        out["reliability_c"] = float(rel - corr)
        out["resolution_c"] = float(res - corr + unc / (n - 1))
        out["uncertainty_c"] = float(unc * n / (n - 1))
    return out


# ---------------------------------------------------------------------------
# Skill scores
# ---------------------------------------------------------------------------

def brier_skill_score(y, p, p_ref) -> float:
    """1 - Brier(p)/Brier(p_ref). Positive = better than the reference."""
    b, b_ref = brier(y, p), brier(y, p_ref)
    return float(1 - b / b_ref) if b_ref > 0 else np.nan


def log_loss_skill_score(y, p, p_ref) -> float:
    """1 - LL(p)/LL(p_ref) ("Ignorance skill"). Positive = better than reference."""
    l, l_ref = log_loss(y, p), log_loss(y, p_ref)
    return float(1 - l / l_ref) if l_ref > 0 else np.nan


# ---------------------------------------------------------------------------
# Ordinal / continuous proper scores
# ---------------------------------------------------------------------------

def ranked_probability_score(probs: np.ndarray, outcome_idx) -> float:
    """Epstein (1969) RPS for ordered categories.

    probs: (n, K) forecast probabilities over K *ordered* categories
           (e.g. [win, placed-not-won, unplaced]).
    outcome_idx: (n,) index of the realised category.
    """
    P = np.asarray(probs, dtype=float)
    P = P / P.sum(axis=1, keepdims=True)
    n, K = P.shape
    o = np.asarray(outcome_idx, dtype=int)
    Y = np.zeros_like(P)
    Y[np.arange(n), o] = 1.0
    cp, cy = np.cumsum(P, axis=1), np.cumsum(Y, axis=1)
    return float(np.mean(np.sum((cp - cy) ** 2, axis=1) / (K - 1)))


def crps_ensemble(samples: np.ndarray, obs) -> float:
    """CRPS of an ensemble forecast (e.g. ABM Monte-Carlo draws).

    samples: (n, m) draws per observation; obs: (n,).
    Uses CRPS = E|X - y| - 0.5 E|X - X'| with the sorted-sample identity.
    """
    X = np.asarray(samples, dtype=float)
    y = np.asarray(obs, dtype=float).reshape(-1, 1)
    m = X.shape[1]
    term1 = np.mean(np.abs(X - y), axis=1)
    Xs = np.sort(X, axis=1)
    i = np.arange(1, m + 1)
    term2 = np.sum((2 * i - m - 1) * Xs, axis=1) / (m * m)
    return float(np.mean(term1 - term2))


def crps_gaussian(mu, sigma, obs) -> float:
    mu, sigma, y = (np.asarray(v, dtype=float) for v in (mu, sigma, obs))
    z = (y - mu) / sigma
    return float(np.mean(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


# ---------------------------------------------------------------------------
# Calibration tables
# ---------------------------------------------------------------------------

def reliability_table(y, p, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    y, p = _arr(y, p)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for k in np.unique(idx):
        m = idx == k
        rows.append({
            "bin": int(k), "n": int(m.sum()), "p_lo": float(edges[k]), "p_hi": float(edges[k + 1]),
            "p_mean": float(p[m].mean()), "obs_rate": float(y[m].mean()),
            "gap": float(p[m].mean() - y[m].mean()),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    t = reliability_table(y, p, n_bins)
    return float(np.sum(t["n"] * np.abs(t["gap"])) / t["n"].sum())


# ---------------------------------------------------------------------------
# Rank-based diagnostics (within race)
# ---------------------------------------------------------------------------

def concordance_index(score, position, race_id) -> float:
    """Fraction of within-race pairs where the higher score finished better.

    Equivalent to the Cox C-statistic with race as stratum. Ties in
    position (dead-heats) are skipped; ties in score count as half.
    """
    df = pd.DataFrame({"s": np.asarray(score, float), "pos": np.asarray(position, float), "r": race_id})
    df = df.dropna()
    conc = pairs = 0.0
    for _, g in df.groupby("r"):
        s, pos = g["s"].values, g["pos"].values
        n = len(s)
        if n < 2:
            continue
        ds = s[:, None] - s[None, :]
        dp = pos[:, None] - pos[None, :]
        valid = dp != 0
        conc += np.sum(valid & (np.sign(ds) == -np.sign(dp))) + 0.5 * np.sum(valid & (ds == 0))
        pairs += np.sum(valid)
    return float(conc / pairs) if pairs else np.nan


def kendall_tau_by_race(score, position, race_id) -> pd.Series:
    df = pd.DataFrame({"s": np.asarray(score, float), "pos": np.asarray(position, float), "r": race_id}).dropna()
    out = {}
    for r, g in df.groupby("r"):
        if len(g) >= 3:
            # higher score should mean lower (better) position
            out[r] = kendalltau(-g["s"].values, g["pos"].values).statistic
    return pd.Series(out, name="kendall_tau")


# ---------------------------------------------------------------------------
# Divergence / drift
# ---------------------------------------------------------------------------

def kl_divergence(p, q) -> float:
    p = np.clip(np.asarray(p, float), EPS, None); q = np.clip(np.asarray(q, float), EPS, None)
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p, q) -> float:
    p = np.clip(np.asarray(p, float), EPS, None); q = np.clip(np.asarray(q, float), EPS, None)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m))


def divergence_by_period(
    df: pd.DataFrame, model_col: str, market_col: str, race_col: str = "raceid",
    period_col: str = "race_date", freq: str = "MS",
) -> pd.DataFrame:
    """Mean per-race JS divergence between model and market, by period.

    A rising series signals concept drift (the model and the market are
    disagreeing more) — investigate before it shows up in P&L.
    """
    rows = []
    for r, g in df.groupby(race_col):
        rows.append({"race": r, "period": pd.to_datetime(g[period_col].iloc[0]),
                     "js": js_divergence(g[model_col].values, g[market_col].values)})
    d = pd.DataFrame(rows).dropna(subset=["js"])
    if d.empty:
        return d
    return d.set_index("period")["js"].resample(freq).agg(["mean", "median", "count"]).dropna()


# ---------------------------------------------------------------------------
# One-call report
# ---------------------------------------------------------------------------

def scoring_report(
    df: pd.DataFrame,
    p_col: str = "predicted_win_prob_norm",
    y_col: str = "won",
    race_col: str = "raceid",
    price_col: str = "bfsp",
    position_col: str = "placing_numerical",
    n_bins: int = 10,
) -> dict:
    """Race-level scoring report: model vs market, decomposed.

    Expects one row per runner with a normalised model probability, the
    Betfair SP and the binary outcome. Returns a flat dict plus the
    reliability table under ``reliability``.
    """
    d = df.copy()
    if race_col not in d.columns:
        from model.perf_figures import ensure_raceid
        d = ensure_raceid(d)
    d = d.dropna(subset=[p_col, price_col, y_col])
    d = d[pd.to_numeric(d[price_col], errors="coerce") > 1.0]
    d["_p_model"] = race_normalise(d, p_col, race_col)
    d["_p_market"] = market_implied_probs(d, price_col, race_col)
    y = d[y_col].astype(float).values
    pm, pk = d["_p_model"].values, d["_p_market"].values

    dm, dk = murphy_decomposition(y, pm, n_bins), murphy_decomposition(y, pk, n_bins)
    out = {
        "n_runners": int(len(d)),
        "n_races": int(d[race_col].nunique()),
        "base_rate": float(y.mean()),
        "model_log_loss": log_loss(y, pm),
        "market_log_loss": log_loss(y, pk),
        "model_brier": dm["brier"],
        "market_brier": dk["brier"],
        "brier_skill_vs_market": brier_skill_score(y, pm, pk),
        "log_loss_skill_vs_market": log_loss_skill_score(y, pm, pk),
        "model_reliability": dm["reliability"],
        "model_resolution": dm["resolution"],
        "market_reliability": dk["reliability"],
        "market_resolution": dk["resolution"],
        "uncertainty": dm["uncertainty"],
        "model_reliability_c": dm.get("reliability_c", np.nan),
        "model_resolution_c": dm.get("resolution_c", np.nan),
        "market_reliability_c": dk.get("reliability_c", np.nan),
        "market_resolution_c": dk.get("resolution_c", np.nan),
        "model_ece": expected_calibration_error(y, pm, n_bins),
        "market_ece": expected_calibration_error(y, pk, n_bins),
        "mean_js_model_vs_market": float(np.mean([
            js_divergence(g["_p_model"].values, g["_p_market"].values) for _, g in d.groupby(race_col)
        ])),
        "reliability": reliability_table(y, pm, n_bins),
    }
    if position_col in d.columns:
        out["model_concordance"] = concordance_index(pm, d[position_col].values, d[race_col].values)
        out["market_concordance"] = concordance_index(pk, d[position_col].values, d[race_col].values)
    return out
