"""
Ordering probabilities: Harville and the Benter / Lo–Bacon-Shone correction
(Section IV.3 of the racing² master framework).

Harville assumes the horse that finishes second is drawn from the rest with
its raw win probability; empirically low-probability horses finish second
and third more often than that predicts. Benter's fix flattens the win
vector for the later places:

    σ_i ∝ π_i^γ  (second),  τ_i ∝ π_i^δ  (third),  γ, δ < 1
    P(i, j, k) = π_i · σ_j/(1−σ_i) · τ_k/(1−τ_i−τ_j)

γ and δ are fitted by maximum likelihood on observed 1-2-3 finishes; they
are not universal constants (HK ≈ 0.81 / 0.65) — fit per jurisdiction and
field-size band. Place probabilities and exotics follow from the same
three-way distribution.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _flatten(p: np.ndarray, g: float) -> np.ndarray:
    q = np.power(np.clip(p, 1e-12, 1), g)
    return q / q.sum()


def top3_probabilities(p: np.ndarray, gamma: float = 1.0, delta: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P(win), P(2nd), P(3rd) per runner for one race. γ = δ = 1 is Harville."""
    p = np.asarray(p, float); p = p / p.sum(); n = len(p)
    s = _flatten(p, gamma); t = _flatten(p, delta)
    p2 = np.zeros(n); p3 = np.zeros(n)
    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            pij = p[i] * s[j] / (1 - s[i])
            p2[j] += pij
            denom = 1 - t[i] - t[j]
            if denom > 1e-12:
                for k in range(n):
                    if k != i and k != j:
                        p3[k] += pij * t[k] / denom
    return p, p2, p3


def place_probabilities(p: np.ndarray, n_places: int, gamma: float = 1.0, delta: float = 1.0) -> np.ndarray:
    p1, p2, p3 = top3_probabilities(p, gamma, delta)
    if n_places <= 1:
        return p1
    if n_places == 2:
        return p1 + p2
    return p1 + p2 + p3  # 4th place omitted (rare terms) — extend with a fourth exponent if needed


def fit_ordering_params(races, x0=(0.8, 0.65)) -> dict:
    """races: iterable of (p_vector, (idx_1st, idx_2nd, idx_3rd)) with finishing
    indices into p (idx_3rd may be None). Maximises Π π_1 · σ_2/(1−σ_1) · τ_3/(1−τ_1−τ_2)."""
    data = [(np.asarray(p, float) / np.sum(p), o) for p, o in races]

    def nll(theta):
        g, d = theta
        tot = 0.0
        for p, (a, b, c) in data:
            s = _flatten(p, g); t = _flatten(p, d)
            tot -= np.log(max(s[b] / (1 - s[a]), 1e-12))
            if c is not None:
                tot -= np.log(max(t[c] / max(1 - t[a] - t[b], 1e-12), 1e-12))
        return tot

    res = minimize(nll, np.asarray(x0, float), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-4})
    base = nll(np.array([1.0, 1.0]))
    return {"gamma": float(res.x[0]), "delta": float(res.x[1]), "nll": float(res.fun), "nll_harville": float(base),
            "n_races": len(data)}
