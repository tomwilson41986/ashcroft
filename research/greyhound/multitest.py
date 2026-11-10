"""Many tests at once (the review of 10 Nov 2026, reports/greyhound_model_review.md): the greyhound labs read dozens of
rules on two targets every night, and about one in twenty clears a one-sided 90% bound by chance. These helpers put a
number on that for a whole family of tests: each test's one-sided p-value from its z-score, and its q-value (Benjamini
and Hochberg, 1995): the share of false discoveries expected among every test at least as strong. A rule whose q is
under 0.10 is one the family as a whole supports; a rule with a low p and a high q is what chance looks like.
"""

from __future__ import annotations

import math

import numpy as np


def p_one_sided(z) -> np.ndarray:
    """P(Z >= z) under the null of no edge, for each z (NaN stays NaN)."""
    z = np.asarray(z, float)
    return np.array([0.5 * math.erfc(v / math.sqrt(2)) if np.isfinite(v) else np.nan for v in z.ravel()]).reshape(z.shape)


def bh(p) -> np.ndarray:
    """Benjamini-Hochberg q-values over the finite p-values given (NaN stays NaN)."""
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return q
    order = ok[np.argsort(p[ok])]
    ranked = p[order] * len(ok) / np.arange(1, len(ok) + 1)
    q[order] = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    return q


MIN_RACES = 100      # under this a z-score of betting returns means little: a lay wins small and often, loses big and rarely


def z_by_race(by_race, min_races: int = MIN_RACES) -> float | None:
    """The mean of the race-level returns over its standard error (None under ``min_races`` races or with no spread:
    18 lays that all won have no spread, and no evidence either)."""
    v = np.asarray(by_race, float)
    v = v[np.isfinite(v)]
    if len(v) < min_races or v.std(ddof=1) <= 1e-9 * max(1.0, abs(v.mean())):
        return None
    return float(v.mean() / (v.std(ddof=1) / math.sqrt(len(v))))
