"""Probability scores shared by every cricket model report: log loss, Brier, expected calibration error."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def log_loss(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error: ten equal-width bins of p, |mean p - mean y| weighted by the bin's share."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def reliability(y, p, bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    df = pd.DataFrame({"bin": b, "p": p, "y": y})
    out = df.groupby("bin").agg(n=("y", "size"), mean_p=("p", "mean"), win_rate=("y", "mean")).reset_index()
    out["lo"] = out["bin"] / bins
    return out[["lo", "n", "mean_p", "win_rate"]]


def score(y, p) -> dict:
    return {"n": int(len(y)), "log_loss": log_loss(y, p), "brier": brier(y, p), "ece": ece(y, p)}


def multiclass_log_loss(y_idx, P) -> float:
    P = np.clip(np.asarray(P, dtype=float), EPS, 1.0)
    return float(-np.mean(np.log(P[np.arange(len(y_idx)), np.asarray(y_idx)])))


def ll_rows(y, p) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_bootstrap(y, p_model, p_base, groups, n: int = 2000, seed: int = 0) -> dict:
    """Log loss of ``p_model`` minus ``p_base`` on the same rows, with a 95% interval from resampling whole groups
    (matches: balls within a match are not independent). Negative: the model is better."""
    d = ll_rows(y, p_model) - ll_rows(y, p_base)
    g = pd.Series(d).groupby(np.asarray(groups))
    sums, counts = g.sum().to_numpy(), g.size().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n, len(sums)))
    boot = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "groups": int(len(sums)), "rows": int(len(d))}
