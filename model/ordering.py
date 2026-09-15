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
are not universal constants (HK ≈ 0.81 / 0.65) — `fit_ordering_params_by_band`
fits them per field-size band, since the second- and third-place bias is not
the same shape in a 6-runner maiden as in a 20-runner handicap.

Beyond the first three places the closed form stops being worth writing out:
each extra place adds a nested sum over the field. `simulate_finishing_orders`
samples whole finishing orders instead — sequential sampling without
replacement, which is Plackett-Luce when one exponent is used throughout and
the Benter model when the exponent changes with the position being filled.
Everything downstream (place probabilities beyond third, placepots, any
n-runner permutation bet) is then a counting exercise on the simulated orders.

Why the exotics matter commercially: probability differences compound
multiplicatively, so the more specific the bet, the larger the advantage.
Benter's example has two horses with expected returns of 0.955 and 0.996 —
both losing win bets — combining into a quinella whose expected return is
1.16. Exacta, forecast and placepot markets are therefore worth pricing even
when the win market is unbeatable; place and show run the other way, being
less specific than a win bet, so they water the edge down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

#: Field-size bands for the per-band γ/δ fit. Small fields have few ways to
#: fill the places and little flattening to find; big handicaps have the most.
DEFAULT_FIELD_BANDS = ((2, 7), (8, 11), (12, 15), (16, 99))


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


def place_probabilities(p: np.ndarray, n_places: int, gamma: float = 1.0, delta: float = 1.0,
                        epsilon: float | None = None, n_sims: int = 20000, rng=None) -> np.ndarray:
    """P(finish in the first ``n_places``) per runner.

    Up to three places this is the exact closed form. Four places (paid in
    big-field handicaps, and the unit of a placepot leg) has no cheap closed
    form, so it is simulated with ``epsilon`` as the fourth-place exponent —
    the flattening keeps going, so it defaults to ``delta`` rather than to 1.
    """
    if n_places <= 3:
        p1, p2, p3 = top3_probabilities(p, gamma, delta)
        if n_places <= 1:
            return p1
        if n_places == 2:
            return p1 + p2
        return p1 + p2 + p3
    q = np.asarray(p, float); q = q / q.sum()
    if n_places >= len(q):
        return np.ones(len(q))
    orders = simulate_finishing_orders(q, n_sims=n_sims, gamma=gamma, delta=delta, epsilon=epsilon, rng=rng)
    return position_probabilities(orders, len(q), n_positions=n_places).sum(axis=1)


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


def fit_ordering_params_by_band(races, bands=DEFAULT_FIELD_BANDS, min_races: int = 200, x0=(0.8, 0.65)) -> dict:
    """γ and δ per field-size band, with the global fit as the fallback.

    §IV.3 is explicit that Benter's 0.81 / 0.65 are not universal constants and
    that other authors derive materially different values elsewhere: fit per
    jurisdiction and per field-size band. A band with fewer than ``min_races``
    usable races keeps the global pair rather than a noisy one of its own, and
    says so in ``fallback``.

    ``races`` is the same (p_vector, (idx_1st, idx_2nd, idx_3rd)) sequence
    `fit_ordering_params` takes; the band is read off ``len(p)``.
    """
    races = list(races)
    out = {"global": fit_ordering_params(races, x0=x0), "bands": {}, "field_bands": tuple(tuple(b) for b in bands)}
    for lo, hi in bands:
        sub = [r for r in races if lo <= len(r[0]) <= hi]
        label = f"{lo}-{hi}"
        if len(sub) >= min_races:
            fit = fit_ordering_params(sub, x0=x0)
            fit["fallback"] = False
        else:
            fit = dict(out["global"], n_races=len(sub), fallback=True)
        fit["lo"], fit["hi"] = int(lo), int(hi)
        out["bands"][label] = fit
    return out


def ordering_params_for(fit: dict, n_runners: int) -> tuple[float, float]:
    """(γ, δ) for a field of ``n_runners`` from a `fit_ordering_params_by_band` result."""
    for b in fit.get("bands", {}).values():
        if b["lo"] <= n_runners <= b["hi"]:
            return float(b["gamma"]), float(b["delta"])
    g = fit.get("global", fit)
    return float(g["gamma"]), float(g["delta"])


# ---------------------------------------------------------------------------
# Monte-Carlo finishing orders (§IV.3: simulate rather than enumerate)
# ---------------------------------------------------------------------------

def _rank_exponents(n: int, gamma: float, delta: float, epsilon: float | None) -> np.ndarray:
    e = np.full(n, float(delta if epsilon is None else epsilon))
    e[0] = 1.0
    if n > 1:
        e[1] = float(gamma)
    if n > 2:
        e[2] = float(delta)
    return e


def simulate_finishing_orders(p, n_sims: int = 10000, gamma: float = 1.0, delta: float = 1.0,
                              epsilon: float | None = None, rng=None) -> np.ndarray:
    """Sample whole finishing orders. Returns an (n_sims, n) array of runner
    indices, column k holding the runner that finished k-th.

    Sampling is sequential without replacement, and the weight used to fill
    position k is π^e_k: e = (1, γ, δ, ε, ε, …). With γ = δ = 1 this is exactly
    Plackett-Luce on the win vector — so the simulated win frequencies converge
    on π, which is the sanity check the test suite runs. With γ, δ < 1 it is
    Benter's correction, extended past third place by ``epsilon`` (default δ).

    Enumerating permutations costs n!/(n−k)!; this costs one pass per position
    regardless of how specific the bet is, which is why §IV.3 says to simulate
    for the large exotics.
    """
    q = np.asarray(p, float)
    q = np.clip(q, 1e-12, None)
    q = q / q.sum()
    n = len(q)
    rng = np.random.default_rng() if rng is None else rng
    w_by_rank = np.power(q[None, :], _rank_exponents(n, gamma, delta, epsilon)[:, None])   # (position, runner)
    avail = np.ones((n_sims, n), dtype=bool)
    orders = np.empty((n_sims, n), dtype=np.int64)
    rows = np.arange(n_sims)
    for k in range(n):
        w = np.where(avail, w_by_rank[k][None, :], 0.0)
        c = np.cumsum(w, axis=1)
        u = rng.random((n_sims, 1)) * c[:, -1:]
        pick = np.minimum((c < u).sum(axis=1), n - 1)
        orders[:, k] = pick
        avail[rows, pick] = False
    return orders


def position_probabilities(orders: np.ndarray, n_runners: int, n_positions: int | None = None) -> np.ndarray:
    """(n_runners, n_positions) matrix of P(runner finishes in position k) from
    simulated orders."""
    n_sims, n = orders.shape
    n_positions = n if n_positions is None else min(n_positions, n)
    out = np.zeros((n_runners, n_positions))
    for k in range(n_positions):
        out[:, k] = np.bincount(orders[:, k], minlength=n_runners)[:n_runners] / n_sims
    return out


# ---------------------------------------------------------------------------
# Exotic pricing (§IV.3)
# ---------------------------------------------------------------------------

def exacta_matrix(p, gamma: float = 1.0) -> np.ndarray:
    """P(i first, j second) for every ordered pair — the exacta / straight
    forecast surface. Exact, no simulation needed: P(i,j) = π_i·σ_j/(1−σ_i).

    Rows are the winner, columns the runner-up; the diagonal is zero and the
    whole matrix sums to one.
    """
    q = np.asarray(p, float); q = q / q.sum()
    s = _flatten(q, gamma)
    m = np.outer(q / (1.0 - s), s)
    np.fill_diagonal(m, 0.0)
    return m


def exacta_probability(p, first: int, second: int, gamma: float = 1.0) -> float:
    """P(``first`` wins and ``second`` is runner-up). The UK straight forecast."""
    return float(exacta_matrix(p, gamma)[first, second])


#: the straight forecast is the exacta under another name
forecast_probability = exacta_probability


def quinella_probability(p, a: int, b: int, gamma: float = 1.0) -> float:
    """P(``a`` and ``b`` fill the first two places in either order)."""
    m = exacta_matrix(p, gamma)
    return float(m[a, b] + m[b, a])


def trifecta_probability(p, first: int, second: int, third: int, gamma: float = 1.0, delta: float = 1.0) -> float:
    """P(exact 1-2-3) from the Benter three-way form."""
    q = np.asarray(p, float); q = q / q.sum()
    s = _flatten(q, gamma); t = _flatten(q, delta)
    denom = (1.0 - s[first]) * (1.0 - t[first] - t[second])
    if denom <= 1e-12:
        return 0.0
    return float(q[first] * s[second] * t[third] / denom)


def fair_price(prob: float, commission: float = 0.0) -> float:
    """Break-even decimal price for a probability, net of exchange commission
    on winnings: 1 + (1/p − 1)/(1 − c). Anything above this is value."""
    if not np.isfinite(prob) or prob <= 0:
        return float("inf")
    return float(1.0 + (1.0 / prob - 1.0) / (1.0 - commission))


def expected_return(prob: float, price: float, commission: float = 0.0) -> float:
    """Expected return per unit staked, Benter's units — 1.0 is break-even and
    his worked quinella scores 1.16 off two losing win bets."""
    return float(prob * (1.0 + (price - 1.0) * (1.0 - commission)))


def exacta_table(p, gamma: float = 1.0, runners=None, top_k: int = 20, commission: float = 0.0,
                 market_prices: dict | None = None) -> pd.DataFrame:
    """The top ``top_k`` exacta combinations with probability and fair price.

    ``market_prices`` maps (first, second) index pairs to the offered decimal
    price; where one is supplied the table carries the expected return, so the
    +EV combinations can be read straight off it.
    """
    m = exacta_matrix(p, gamma)
    n = m.shape[0]
    names = list(runners) if runners is not None else list(range(n))
    idx = np.dstack(np.unravel_index(np.argsort(m, axis=None)[::-1], m.shape))[0][:top_k]
    rows = []
    for i, j in idx:
        prob = float(m[i, j])
        if prob <= 0:
            continue
        row = {"first": names[i], "second": names[j], "prob": prob, "fair_price": fair_price(prob, commission)}
        if market_prices is not None and (int(i), int(j)) in market_prices:
            row["market_price"] = float(market_prices[(int(i), int(j))])
            row["expected_return"] = expected_return(prob, row["market_price"], commission)
        rows.append(row)
    return pd.DataFrame(rows)


def placepot_probability(legs) -> float:
    """P(every leg places), ``legs`` being one array of place probabilities per
    leg for the runners you have taken in it.

    Legs are treated as independent, which is the honest first approximation:
    the races are separate events, and what correlation there is comes through
    going and pace bias rather than through the result of one race driving the
    next. The per-leg sum is capped at 1 because two of your selections cannot
    both take the same single place.
    """
    total = 1.0
    for leg in legs:
        total *= float(min(np.nansum(np.asarray(leg, float)), 1.0))
    return total
