"""
Portfolio staking, execution and bet-performance attribution: what changes
once stakes interact.

``model.staking`` sizes one bet at a time. That is wrong in two directions and
the framework (§V.3, §IIA.5) is explicit about both:

* **Within a race the runners are mutually exclusive.** Backing only the
  single highest-advantage runner ignores that the other overlays hedge it.
  The joint solution stakes *more* on the top pick than solo Kelly does,
  because the second bet pays when the first loses, and it trades a little
  expected return for a materially higher chance of a return at all.
* **Across a card the bets are positively correlated** through shared model
  error -- a wrong going read is wrong for every race on that card.
  Independence understates the exposure, and Kelly inverts the covariance
  matrix, so covariance error is punished hard (§IIA.5).

What is here:

    net_odds                 decimal odds with commission on net winnings
    back_return_moments      mean and sd of a back bet's return per unit staked
    race_return_covariance   exact multinomial covariance of back-bet returns
    block_covariance         three-level target: exact within race, shared
                             latent error across meeting and across day
    ledoit_wolf_shrinkage    shrink a sample covariance toward that target
    estimate_block_correlations  measure the two cross-race correlations from
                             realised returns rather than assuming them
    race_kelly_exact         closed-form joint Kelly for one race
    race_kelly_numeric       the same by concave optimisation, with caps
    portfolio_kelly          whole betting cycle, caps, fractional phi
    expected_log_growth      Monte-Carlo check of any stake vector
    ladder_fill              depth-walked fill and the slippage it costs
    ladder_price_fn          exchange depth as an impact curve
    pool_price_fn            pari-mutuel dilution as an impact curve
    ev_stake_curve           EV against stake size; it is not linear
    max_ev_stake             the max-EV point and the two-thirds default
    split_order              slices across the pre-off window and into BSP
    apply_depth_fills        depth-limited stakes and the returns they earn
    liquidity_stress         growth with and without the depth constraint
    realised_ic              Grinold information coefficient, per race
    effective_breadth        independent bets after the correlation is paid for
    grinold_report           realised IR against IC * sqrt(breadth)
    required_bets            sample size to detect an edge of a given size
    factor_attribution       per-bet P&L regressed on race factors
    alpha_decay              edge as a function of days since the last refit

**This is machinery, not a claim.** ``STAKING_REPORT.md`` measured the current
production probabilities over 253,532 real runners and found Kelly destroys
the bank at every fraction from full to 1/20, because the probabilities are
less resolved than Betfair SP. Portfolio Kelly on probabilities that are not
worth betting loses the bank more efficiently, not less. These functions exist
so that the sizing is correct when the probabilities finally are worth
betting, and so that the diagnostics below can say whether they are.

Conventions, shared with ``model.staking``: returns are per unit staked with
commission taken on net winnings; bankroll paths are carried in logs; a filter
on the settled BSP is look-ahead, so selection filters use the model's own
forecast price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from model.staking import bankroll_path, growth_stats

__all__ = [
    "net_odds", "back_return_moments", "race_return_covariance", "block_covariance",
    "ledoit_wolf_shrinkage", "estimate_block_correlations", "nearest_psd",
    "race_kelly_exact", "race_kelly_numeric", "portfolio_kelly", "expected_log_growth",
    "ladder_fill", "ladder_price_fn", "pool_price_fn", "ev_stake_curve", "max_ev_stake",
    "split_order", "apply_depth_fills", "liquidity_stress",
    "realised_ic", "effective_breadth", "grinold_report", "required_bets",
    "factor_attribution", "alpha_decay",
]

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Return moments and the block-structured covariance (§IIA.5, B18 / N7)
# ---------------------------------------------------------------------------

def net_odds(price, commission: float = 0.05) -> np.ndarray:
    """Gross decimal odds with commission taken on the net winnings.

    ``o_net = 1 + (o - 1)(1 - c)``, so a unit stake returns ``o_net`` when the
    horse wins and 0 when it does not. Every formula below uses ``o_net``, which
    keeps the house convention: commission on winnings, never on turnover."""
    o = np.asarray(price, float)
    return 1.0 + (o - 1.0) * (1.0 - float(commission))


def back_return_moments(p, price, commission: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of a back bet's return per unit staked.

    The return is ``R = o_net * 1{win} - 1``, so ``E[R] = p * o_net - 1`` (the
    edge) and ``sd(R) = o_net * sqrt(p(1-p))``."""
    p = np.clip(np.asarray(p, float), _EPS, 1.0 - _EPS)
    o = net_odds(price, commission)
    return p * o - 1.0, o * np.sqrt(p * (1.0 - p))


def race_return_covariance(p, price, commission: float = 0.05) -> np.ndarray:
    """Covariance of back-bet returns for the runners of one race.

    This block is **exact and known from the probability vector**, not
    estimated (§IIA.5). The win indicators are multinomial with one trial, so
    ``Cov(1_i, 1_j) = p_i * 1{i=j} - p_i * p_j``, and the returns are affine in
    the indicators:

        Cov(R_i, R_j) = o_i * o_j * (p_i * 1{i=j} - p_i * p_j)

    Off the diagonal that is negative for every pair -- only one of them can
    win -- which is the whole reason a race has to be solved jointly."""
    p = np.clip(np.asarray(p, float), _EPS, 1.0 - _EPS)
    o = net_odds(price, commission)
    C = -np.outer(o * p, o * p)
    np.fill_diagonal(C, o ** 2 * p * (1.0 - p))
    return C


def nearest_psd(S: np.ndarray, ridge: float = 1e-10, tol: float = 1e-10) -> tuple[np.ndarray, float]:
    """Clip negative eigenvalues. Returns the repaired matrix and the size of
    the repair as a fraction of the largest eigenvalue (0.0 = nothing to do).

    Eigenvalues within ``tol`` of zero are float noise on a singular matrix --
    a within-race block whose probabilities sum to one is exactly singular --
    and are not counted as a repair."""
    S = (np.asarray(S, float) + np.asarray(S, float).T) / 2.0
    w, V = np.linalg.eigh(S)
    top = float(max(w.max(), _EPS))
    if w.min() >= -tol * top:
        return S + ridge * np.eye(len(S)), 0.0
    return (V @ np.diag(np.clip(w, 0.0, None)) @ V.T + ridge * np.eye(len(S)),
            float(-w.min() / top))


def _group_codes(d: pd.DataFrame, col: str | None, fallback: np.ndarray) -> np.ndarray:
    if col is None or col not in d.columns:
        return fallback
    return pd.factorize(d[col].astype(str), use_na_sentinel=False)[0]


def block_covariance(bets: pd.DataFrame, p_col: str = "p_model", price_col: str = "bsp",
                     race_col: str = "raceid", meeting_col: str | None = None,
                     day_col: str | None = "race_date", commission: float = 0.05,
                     rho_meeting: float = 0.20, rho_day: float = 0.05,
                     returns_panel: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """The three-level covariance §IIA.5 asks for, as a variance-components model.

    Level 1, **within a race**: the exact multinomial block above. Known, not
    estimated.
    Level 2, **across a meeting**: a shared latent shock -- the going read, the
    track bias, the trainer's day -- giving every pair of bets on that card a
    correlation ``rho_meeting``.
    Level 3, **across a day**: a weaker shock through weather and model drift,
    correlation ``rho_day``. Different days are independent.

    Written as variance components,

        Sigma = (1 - rho_m) * Sigma_exact
              + (rho_m - rho_d) * sum_meetings u_m u_m'
              + rho_d * sum_days u_d u_d'

    with ``u`` carrying ``sd_i`` on its own group and zero elsewhere. Every
    added term is rank one and positive, so the result is PSD **by
    construction** and needs no repair -- which matters because Kelly inverts
    it. The construction also leaves the total variance of each bet exactly
    ``sd_i**2``: the shared error redistributes correlation, it does not invent
    variance. It requires ``1 >= rho_meeting >= rho_day >= 0``.

    Note that laying the naive equicorrelation on top of the exact blocks --
    the obvious thing to write -- is *never* PSD for ``rho > 0``: each block is
    singular along the direction ``v_i = 1/o_i`` and the cross terms make that
    direction negative. Hence the factor form.

    ``rho_meeting`` and ``rho_day`` default to plausible values and should be
    replaced by ``estimate_block_correlations`` on real settled bets.

    If ``returns_panel`` (T x n, columns aligned to ``bets``) is supplied the
    structure becomes the *target* and the sample covariance is shrunk toward
    it by ``ledoit_wolf_shrinkage``. With no panel there is nothing to shrink
    -- a live betting cycle happens once -- and the structure is used as is.
    """
    if not (1.0 >= rho_meeting >= rho_day >= 0.0):
        raise ValueError("need 1 >= rho_meeting >= rho_day >= 0")
    d = bets
    p = np.clip(np.nan_to_num(pd.to_numeric(d[p_col], errors="coerce").to_numpy(float), nan=_EPS),
                _EPS, 1 - _EPS)
    o = net_odds(np.nan_to_num(pd.to_numeric(d[price_col], errors="coerce").to_numpy(float), nan=1.0),
                 commission)
    n = len(p)
    rid = pd.factorize(d[race_col].astype(str), use_na_sentinel=False)[0]
    did = _group_codes(d, day_col, np.zeros(n, dtype=int))
    if meeting_col is not None and meeting_col in d.columns:
        mid = _group_codes(d, meeting_col, rid)
    elif "track" in d.columns:                       # a meeting is a track on a day
        mid = pd.factorize(d["track"].astype(str) + "|" + pd.Series(did, index=d.index).astype(str),
                           use_na_sentinel=False)[0]
    else:
        mid = did
    sd = o * np.sqrt(p * (1.0 - p))

    same_r = rid[:, None] == rid[None, :]
    exact = np.where(same_r, -np.outer(o * p, o * p), 0.0)
    np.fill_diagonal(exact, o ** 2 * p * (1.0 - p))
    outer_sd = np.outer(sd, sd)
    F = ((1.0 - rho_meeting) * exact
         + (rho_meeting - rho_day) * np.where(mid[:, None] == mid[None, :], outer_sd, 0.0)
         + rho_day * np.where(did[:, None] == did[None, :], outer_sd, 0.0))
    np.fill_diagonal(F, sd ** 2)
    info = {"n": n, "races": int(pd.unique(rid).size), "meetings": int(pd.unique(mid).size),
            "days": int(pd.unique(did).size), "rho_meeting": float(rho_meeting),
            "rho_day": float(rho_day), "shrinkage": 1.0, "psd_repair": 0.0}
    if returns_panel is None:
        F, info["psd_repair"] = nearest_psd(F)
        return F, info
    S, delta = ledoit_wolf_shrinkage(np.asarray(returns_panel, float), F)
    info["shrinkage"] = float(delta)
    S, info["psd_repair"] = nearest_psd(S)
    return S, info


def ledoit_wolf_shrinkage(panel: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage of a sample covariance toward a *known* target.

    ``Sigma = delta * F + (1 - delta) * S`` with the optimal intensity

        delta* = pi_hat / (T * gamma_hat),   pi_hat = sum_ij AsyVar(sqrt(T) s_ij),
                                             gamma_hat = ||F - S||_F^2

    The usual third term of the Ledoit-Wolf intensity is the covariance
    between the estimation error of the sample and that of the target; here the
    target is computed from the fitted probability vector rather than from the
    returns, so it carries no estimation error and that term is zero.

    ``sklearn.covariance.LedoitWolf`` implements the same estimator against a
    scaled-identity target; it is used by ``estimate_block_correlations`` where
    the target genuinely is uninformative, but a scaled identity is exactly the
    wrong target here -- it would shrink away the negative within-race
    correlation, which is the one part of the matrix that is known exactly."""
    X = np.asarray(panel, float)
    if X.ndim != 2 or X.shape[0] < 2:
        return np.asarray(target, float), 1.0
    T, n = X.shape
    if target.shape != (n, n):
        raise ValueError(f"target is {target.shape}, panel implies ({n}, {n})")
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / T
    A = (Xc ** 2).T @ (Xc ** 2) / T
    pi_hat = float(np.sum(A - S ** 2))
    gamma_hat = float(np.sum((np.asarray(target, float) - S) ** 2))
    delta = 1.0 if gamma_hat <= _EPS else float(np.clip(pi_hat / (T * gamma_hat), 0.0, 1.0))
    return delta * np.asarray(target, float) + (1.0 - delta) * S, delta


def _cross_pair_sums(z: np.ndarray, outer: np.ndarray, inner: np.ndarray) -> tuple[float, float]:
    """Sum of z_i*z_j and the pair count over pairs in the same ``outer`` group
    but different ``inner`` groups."""
    df = pd.DataFrame({"z": z, "outer": outer, "inner": inner})
    o = df.groupby("outer")["z"].agg(["sum", "size"])
    i = df.groupby(["outer", "inner"])["z"].agg(["sum", "size"])
    tot_sq = float((o["sum"] ** 2).sum()) - float((df["z"] ** 2).sum())
    tot_n = float((o["size"] ** 2).sum()) - float(o["size"].sum())
    in_sq = float((i["sum"] ** 2).sum()) - float((df["z"] ** 2).sum())
    in_n = float((i["size"] ** 2).sum()) - float(i["size"].sum())
    return tot_sq - in_sq, tot_n - in_n


def estimate_block_correlations(bets: pd.DataFrame, ret_col: str = "ret_back",
                                p_col: str = "p_model", price_col: str = "bsp",
                                race_col: str = "raceid", meeting_col: str | None = None,
                                day_col: str = "race_date", commission: float = 0.05
                                ) -> dict:
    """Measure ``rho_meeting`` and ``rho_day`` from settled bets.

    Standardise each realised return by its own model-implied mean and sd, so
    that what is left is the shared error rather than the price. The
    correlation at each level is then a method-of-moments estimate: the mean
    product over ordered pairs at that level,

        rho = sum over cross pairs of z_i z_j / (number of cross pairs)

    computed from group sums, so it is O(n) rather than O(n^2). Estimates are
    clipped to ``[0, 0.95]`` and forced monotone (a meeting is at least as
    correlated as a day), which is the precondition ``block_covariance``
    requires. Cross-race *within* a meeting is measured against the day level;
    within-race pairs are excluded because that block is known exactly."""
    d = bets.dropna(subset=[ret_col, p_col, price_col]).copy()
    if d.empty:
        return {"rho_meeting": np.nan, "rho_day": np.nan, "n": 0, "meeting_pairs": 0, "day_pairs": 0}
    mu, sd = back_return_moments(d[p_col], d[price_col], commission)
    z = (pd.to_numeric(d[ret_col], errors="coerce").to_numpy(float) - mu) / np.where(sd > 0, sd, np.nan)
    ok = np.isfinite(z)
    d, z = d.loc[ok], z[ok]
    z = z - z.mean()                                  # a constant edge bias is not correlation
    rid = d[race_col].astype(str).to_numpy()
    did = d[day_col].astype(str).to_numpy() if day_col in d.columns else np.zeros(len(d), dtype=str)
    if meeting_col is not None and meeting_col in d.columns:
        mid = d[meeting_col].astype(str).to_numpy()
    elif "track" in d.columns:
        mid = (d["track"].astype(str) + "|" + pd.Series(did, index=d.index)).to_numpy()
    else:
        mid = did
    m_sq, m_n = _cross_pair_sums(z, mid, rid)         # same meeting, different race
    d_sq, d_n = _cross_pair_sums(z, did, mid)         # same day, different meeting
    rho_m = float(np.clip(m_sq / m_n, 0.0, 0.95)) if m_n > 0 else np.nan
    rho_d = float(np.clip(d_sq / d_n, 0.0, 0.95)) if d_n > 0 else np.nan
    if np.isfinite(rho_m) and np.isfinite(rho_d):
        rho_d = min(rho_d, rho_m)
    return {"rho_meeting": rho_m, "rho_day": rho_d, "n": int(len(d)),
            "meeting_pairs": int(m_n), "day_pairs": int(d_n)}


# ---------------------------------------------------------------------------
# Portfolio Kelly (§V.3, N5 / N6)
# ---------------------------------------------------------------------------

def race_kelly_exact(p, price, commission: float = 0.05) -> np.ndarray:
    """Joint Kelly stakes for one race, in closed form.

    One runner wins, so the states are enumerable and expected log wealth is
    exact rather than approximated:

        E[log W] = sum_i p_i log(1 - F + f_i o_i) + p_0 log(1 - F),  F = sum f

    with ``p_0`` the probability that none of the backed runners wins. The
    first-order conditions give, for the optimal backed set S,

        f_k = p_k - (1 - p_S) / ((1 - r_S) * o_k),
        p_S = sum_{k in S} p_k,   r_S = sum_{k in S} 1 / o_k

    and S is a prefix of the runners ordered by advantage ``p_k * o_k``, grown
    until a stake turns non-positive. Because the log of an affine function is
    concave, this is the global optimum.

    The point §V.3 makes falls straight out: with p = (0.5, 0.3) at 2.5 and 4.0
    the solo Kelly stake on the favourite is 0.167, the joint solution stakes
    0.271 on it *and* 0.157 on the second runner, and expected log growth goes
    from 0.0204 to 0.0543. Backing only the top-advantage runner throws away
    more than half the growth."""
    p = np.clip(np.asarray(p, float), _EPS, 1.0 - _EPS)
    o = net_odds(price, commission)
    f = np.zeros(len(p))
    order = np.argsort(-(p * o))                     # by advantage, as the FOCs require
    for k in range(1, len(p) + 1):
        S = order[:k]
        p_S, r_S = float(p[S].sum()), float((1.0 / o[S]).sum())
        if r_S >= 1.0:                               # an arbitrage book; caps handle it
            break
        cand = p[S] - (1.0 - p_S) / ((1.0 - r_S) * o[S])
        if cand.min() <= 0.0:
            break
        f[:] = 0.0
        f[S] = cand
    return f


def _race_log_objective(p: np.ndarray, o: np.ndarray):
    """Exact expected log wealth for one race, and its gradient."""
    p_null = max(1.0 - float(p.sum()), 0.0)

    def neg(f: np.ndarray) -> float:
        F = float(f.sum())
        w0 = max(1.0 - F, _EPS)
        w = np.maximum(1.0 - F + f * o, _EPS)
        return -(float(np.sum(p * np.log(w))) + p_null * np.log(w0))

    def grad(f: np.ndarray) -> np.ndarray:
        F = float(f.sum())
        w0 = max(1.0 - F, _EPS)
        w = np.maximum(1.0 - F + f * o, _EPS)
        return -(p * o / w - float(np.sum(p / w)) - p_null / w0)

    return neg, grad


def race_kelly_numeric(p, price, commission: float = 0.05, cap_race: float = 1.0,
                       cap_selection: float = 1.0, bounds_hi=None) -> np.ndarray:
    """The same problem by concave optimisation, so exposure caps can bind.

    Maximises the exact expected-log-wealth objective over the whole stake
    vector under ``sum f <= cap_race`` and ``0 <= f_i <= cap_selection``
    (``bounds_hi`` overrides the upper bound per runner, which is where a
    per-selection cap on available exchange volume enters). With the caps
    slack this reproduces ``race_kelly_exact`` to solver tolerance."""
    p = np.clip(np.asarray(p, float), _EPS, 1.0 - _EPS)
    o = net_odds(price, commission)
    n = len(p)
    hi = np.full(n, float(cap_selection)) if bounds_hi is None else np.asarray(bounds_hi, float)
    hi = np.clip(hi, 0.0, 1.0 - 1e-6)
    total = float(min(cap_race, 1.0 - 1e-6))
    start = np.minimum(race_kelly_exact(p, price, commission), hi)
    if start.sum() > total:
        start = start * total / max(start.sum(), _EPS)
    if total <= 0 or np.all(hi <= 0):
        return np.zeros(n)
    neg, grad = _race_log_objective(p, o)
    res = minimize(neg, start, jac=grad, method="SLSQP",
                   bounds=list(zip(np.zeros(n), hi)),
                   constraints=[{"type": "ineq", "fun": lambda f: total - f.sum(),
                                 "jac": lambda f: -np.ones(n)}],
                   options={"maxiter": 500, "ftol": 1e-12})
    f = np.clip(res.x, 0.0, hi)
    return f if neg(f) <= neg(start) else start      # never return worse than the start


def _quadratic_kelly(mu: np.ndarray, M: np.ndarray, hi: np.ndarray, groups: np.ndarray,
                     cap_group: float, cap_total: float, phi: float) -> np.ndarray:
    """Maximise ``f'mu - f'Mf / (2 phi)`` under the caps.

    This is the second-order expansion of ``E[log(1 + f'R)]`` -- ``log(1+x) =
    x - x^2/2 + ...`` -- with ``M = E[RR'] = Sigma + mu mu'``. It is the form
    §IIA.5 presupposes when it says Kelly inverts the covariance matrix, and it
    is what makes a whole cycle tractable: the exact objective would need the
    product of every race's outcome space. ``phi`` enters as risk aversion, so
    the unconstrained solution is exactly ``phi`` times the full-Kelly vector
    and the caps then bind on the stakes actually struck."""
    n = len(mu)
    hi = np.clip(np.asarray(hi, float), 0.0, None)
    if n == 0 or cap_total <= 0 or np.all(hi <= 0):
        return np.zeros(n)
    Mp = M / max(phi, _EPS)

    def neg(f):
        return -(float(f @ mu) - 0.5 * float(f @ Mp @ f))

    def grad(f):
        return -(mu - Mp @ f)

    uniq = np.unique(groups)
    cons = [{"type": "ineq", "fun": lambda f: cap_total - f.sum(), "jac": lambda f: -np.ones(n)}]
    for g in uniq:
        sel = (groups == g).astype(float)
        cons.append({"type": "ineq", "fun": (lambda f, s=sel: cap_group - float(s @ f)),
                     "jac": (lambda f, s=sel: -s)})

    def enforce(f: np.ndarray) -> np.ndarray:        # SLSQP can leave a cap a hair violated
        f = np.clip(f, 0.0, hi)
        for g in uniq:
            m = groups == g
            s = f[m].sum()
            if s > cap_group > 0:
                f[m] *= cap_group / s
        if f.sum() > cap_total > 0:
            f *= cap_total / f.sum()
        return f

    start = enforce(np.nan_to_num(np.clip(phi * mu / np.maximum(np.diag(M), _EPS), 0.0, None)))
    res = minimize(neg, start, jac=grad, method="SLSQP", bounds=list(zip(np.zeros(n), hi)),
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
    f = enforce(res.x)
    return f if neg(f) <= neg(start) else start


def portfolio_kelly(bets: pd.DataFrame, p_col: str = "p_model", price_col: str = "bsp",
                    race_col: str = "raceid", meeting_col: str | None = None,
                    day_col: str | None = "race_date", commission: float = 0.05,
                    phi: float = 0.25, cap_total: float = 0.25, cap_race: float = 0.10,
                    cap_selection: float = 0.05, volume_col: str | None = None,
                    volume_fraction: float = 0.05, bank: float = 1.0,
                    method: str = "quadratic", rho_meeting: float = 0.20,
                    rho_day: float = 0.05, returns_panel: np.ndarray | None = None
                    ) -> pd.DataFrame:
    """Solve one betting cycle's whole stake vector at once.

    Two methods, and the difference is exactly where the approximation sits.

    ``method="quadratic"`` (default) maximises the second-order expansion of
    expected log wealth over every bet in the cycle jointly, using the shrunk
    block covariance from ``block_covariance``. It is the only one of the two
    that sees the positive correlation across a card.

    ``method="exact"`` solves each race with the exact enumerated objective and
    then scales the cycle down proportionally if the total cap binds. Exact
    within a race, blind to correlation across races.

    Caps, all as a fraction of bank and all binding on the stakes actually
    struck (§V.3): ``cap_total`` over the cycle, ``cap_race`` within a race,
    ``cap_selection`` per bet, and -- when ``volume_col`` names available
    exchange volume in the same units as ``bank`` -- a per-selection cap of
    ``volume_fraction`` of that volume. ``phi`` is the fractional-Kelly
    multiplier of §V.1: 0.25 to start, never 1.

    ``price_col`` is the price the stake is struck at. In a backtest that is
    BSP, which is the one place the market is allowed in (see
    ``model.staking``); in live use it must be the price actually available on
    the ladder, and no *selection* rule may condition on the settled price.

    Returns the frame with ``mu``, ``sd``, ``kelly_solo`` (single-bet Kelly, for
    comparison), ``kelly_joint`` (the full-Kelly joint vector), ``stake_fraction``
    (after ``phi`` and the caps) and ``stake``. Diagnostics land in
    ``result.attrs["portfolio"]``."""
    d = bets.copy()
    if d.empty:
        for c in ("mu", "sd", "kelly_solo", "kelly_joint", "stake_fraction", "stake"):
            d[c] = np.array([], dtype=float)
        d.attrs["portfolio"] = {"bets": 0, "exposure": 0.0}
        return d
    p_raw = pd.to_numeric(d[p_col], errors="coerce").to_numpy(float)
    price_raw = pd.to_numeric(d[price_col], errors="coerce").to_numpy(float)
    usable = np.isfinite(p_raw) & np.isfinite(price_raw) & (price_raw > 1.0)
    p = np.clip(np.where(usable, p_raw, _EPS), _EPS, 1 - _EPS)
    price = np.where(usable, price_raw, 1.0 + 1e-9)
    mu, sd = back_return_moments(p, price, commission)
    o = net_odds(price, commission)
    d["mu"], d["sd"] = mu, sd
    d["kelly_solo"] = np.clip(np.where(o > 1, (p * o - 1.0) / np.maximum(o - 1.0, _EPS), 0.0), 0.0, 1.0)

    hi = np.full(len(d), float(cap_selection))
    if volume_col is not None and volume_col in d.columns and bank > 0:
        vol = pd.to_numeric(d[volume_col], errors="coerce").fillna(0.0).to_numpy(float)
        hi = np.minimum(hi, volume_fraction * vol / float(bank))
    hi = np.where(usable, np.clip(hi, 0.0, 1.0 - 1e-6), 0.0)   # a missing price is never backed

    rid = pd.factorize(d[race_col].astype(str), use_na_sentinel=False)[0]
    if method == "exact":
        # caps live in the pre-phi frame so that phi * f obeys the caps as given
        f = np.zeros(len(d))
        for r in np.unique(rid):
            m = rid == r
            f[m] = race_kelly_numeric(p[m], price[m], commission,
                                      cap_race=min(cap_race / max(phi, _EPS), 1.0),
                                      cap_selection=1.0, bounds_hi=hi[m] / max(phi, _EPS))
        joint = f.copy()
        f = phi * f
        if f.sum() > cap_total > 0:
            f *= cap_total / f.sum()
        cov_info = {"method": "exact", "shrinkage": np.nan, "rho_meeting": np.nan, "rho_day": np.nan}
    elif method == "quadratic":
        Sigma, cov_info = block_covariance(d, p_col=p_col, price_col=price_col, race_col=race_col,
                                           meeting_col=meeting_col, day_col=day_col,
                                           commission=commission, rho_meeting=rho_meeting,
                                           rho_day=rho_day, returns_panel=returns_panel)
        M = Sigma + np.outer(mu, mu)                 # E[RR']
        f = _quadratic_kelly(mu, M, hi, rid, cap_race, cap_total, phi)
        joint = _quadratic_kelly(mu, M, hi, rid, cap_race, cap_total, 1.0)
        cov_info["method"] = "quadratic"
    else:
        raise ValueError("method must be 'quadratic' or 'exact'")

    d["kelly_joint"], d["stake_fraction"] = joint, f
    d["stake"] = f * float(bank)
    per_race = pd.Series(f).groupby(rid).sum()
    d.attrs["portfolio"] = {
        "bets": int((f > 1e-9).sum()), "runners": int(len(d)), "races": int(pd.unique(rid).size),
        "phi": float(phi), "exposure": float(f.sum()), "max_race_exposure": float(per_race.max()),
        "max_selection": float(f.max()), "cap_total": float(cap_total), "cap_race": float(cap_race),
        "cap_selection": float(cap_selection),
        "solo_exposure": float(np.minimum(phi * d["kelly_solo"].to_numpy(float), hi).sum()),
        "expected_edge": float(np.sum(f * mu)), **cov_info}
    return d


def expected_log_growth(stakes, p, price, race_ids, commission: float = 0.05,
                        n_sims: int = 20000, random_state: int = 0) -> float:
    """Monte-Carlo ``E[log(1 + f'R)]`` for a stake vector across several races.

    Exact enumeration needs the product of every race's outcome space, so the
    joint objective is checked by simulation instead: draw a winner per race
    from that race's probability vector (with the residual mass going to "none
    of these"), settle every stake against it, and average the log multiple.
    Used to verify that the quadratic solution is close to the exact one and
    that the joint solution beats the single-bet one -- not in the hot path."""
    f = np.asarray(stakes, float)
    p = np.clip(np.asarray(p, float), _EPS, 1.0)
    o = net_odds(price, commission)
    rid = pd.factorize(np.asarray(race_ids).astype(str), use_na_sentinel=False)[0]
    rng = np.random.default_rng(random_state)
    payoff = np.zeros(n_sims)
    for r in np.unique(rid):
        m = np.flatnonzero(rid == r)
        pr = p[m]
        probs = np.append(pr, max(1.0 - pr.sum(), 0.0))
        probs = probs / probs.sum()
        draw = rng.choice(len(probs), size=n_sims, p=probs)
        gross = np.append(f[m] * o[m], 0.0)
        payoff += gross[draw]
    w = 1.0 - f.sum() + payoff
    return float(np.mean(np.log(np.maximum(w, 1e-300))))


# ---------------------------------------------------------------------------
# Execution: depth, impact, slippage and order splitting (§V.4, N8 / N9 / N11)
# ---------------------------------------------------------------------------

def _ladder_arrays(prices, sizes) -> tuple[np.ndarray, np.ndarray]:
    pr = np.asarray(prices, float).ravel()
    sz = np.asarray(sizes, float).ravel()
    if pr.shape != sz.shape:
        raise ValueError("prices and sizes must be the same length")
    ok = np.isfinite(pr) & np.isfinite(sz) & (pr > 1.0) & (sz > 0)
    pr, sz = pr[ok], sz[ok]
    order = np.argsort(-pr)                          # a backer takes the longest price first
    return pr[order], sz[order]


def ladder_fill(prices, sizes, stake: float) -> dict:
    """Walk the back ladder and report what actually gets matched.

    ``sizes`` is the money available to back at each price, as
    ``betfair_client.get_market_odds`` returns it in ``ex.availableToBack``. A
    backer consumes the longest price first, so the average price is a
    step-weighted mean that can only fall as the order grows: the EV of a stake
    is not linear in its size (§V.4), and a backtest that assumes every stake
    fills at the top of the book is reporting a price nobody could have had.

    ``slippage_pct`` is the shortfall of the achieved price against the best
    available one, ``1 - avg_price / best_price``. It is zero for an order that
    fits at the top of the book and rises monotonically with size."""
    pr, sz = _ladder_arrays(prices, sizes)
    stake = float(max(stake, 0.0))
    if pr.size == 0 or stake <= 0:
        return {"stake": stake, "matched": 0.0, "unmatched": stake, "avg_price": np.nan,
                "best_price": np.nan if pr.size == 0 else float(pr[0]), "slippage_pct": 0.0,
                "levels": 0, "depth": float(sz.sum()) if sz.size else 0.0}
    room = np.maximum(stake - np.concatenate(([0.0], np.cumsum(sz)[:-1])), 0.0)
    take = np.minimum(sz, room)
    matched = float(take.sum())
    avg = float((take * pr).sum() / matched) if matched > 0 else np.nan
    return {"stake": stake, "matched": matched, "unmatched": stake - matched, "avg_price": avg,
            "best_price": float(pr[0]), "slippage_pct": float(1.0 - avg / pr[0]),
            "levels": int((take > 0).sum()), "depth": float(sz.sum())}


def ladder_price_fn(prices, sizes):
    """Impact function for ``ev_stake_curve``: exchange depth."""
    pr, sz = _ladder_arrays(prices, sizes)

    def fn(stake: float) -> tuple[float, float]:
        f = ladder_fill(pr, sz, stake)
        return f["matched"], f["avg_price"]

    fn.capacity = float(sz.sum()) if sz.size else 0.0
    fn.best = float(pr[0]) if pr.size else np.nan
    return fn


def pool_price_fn(pool: float, on_horse: float, takeout: float = 0.17):
    """Impact function for ``ev_stake_curve``: pari-mutuel dilution.

    The dividend is ``(1 - takeout) * (pool + s) / (on_horse + s)``, so money
    bet dilutes its own return. This is the setting of Benter's illustration in
    §V.4 -- the exchange mechanism differs, you consume depth rather than
    dilute a pool, but the curve has the same concave shape and the two-thirds
    rule comes from the shape, not the mechanism."""
    def fn(stake: float) -> tuple[float, float]:
        return float(stake), float((1.0 - takeout) * (pool + stake) / (on_horse + stake))

    fn.capacity = float("inf")
    fn.best = float((1.0 - takeout) * pool / on_horse)
    return fn


def ev_stake_curve(p: float, price_fn, commission: float = 0.05, n_grid: int = 200,
                   max_stake: float | None = None) -> pd.DataFrame:
    """Expected profit against stake size, over the impact curve.

    ``EV(s) = matched(s) * (p * o_net(s) - 1)`` where ``o_net`` is the
    depth-weighted average price net of commission. The first factor grows and
    the second shrinks, so EV rises, peaks and falls: there is a stake beyond
    which expected profit *falls*, which is the whole of §V.4."""
    cap = max_stake if max_stake is not None else getattr(price_fn, "capacity", np.nan)
    if not np.isfinite(cap):
        cap = 10.0 * float(getattr(price_fn, "best", 1.0))
    stakes = np.linspace(0.0, float(cap), int(n_grid) + 1)[1:]
    matched, avg = np.zeros(len(stakes)), np.zeros(len(stakes))
    for i, s in enumerate(stakes):
        matched[i], avg[i] = price_fn(float(s))
    o_net = net_odds(avg, commission)
    return pd.DataFrame({"stake": stakes, "matched": matched, "avg_price": avg,
                         "edge_per_unit": p * o_net - 1.0, "ev": matched * (p * o_net - 1.0)})


def max_ev_stake(p: float, price_fn, commission: float = 0.05, n_grid: int = 400,
                 max_stake: float | None = None, default: str = "two_thirds") -> dict:
    """The max-EV stake, and the two-thirds point the framework defaults to.

    §V.4: "two-thirds of that stake captures 90% of the profit for a third less
    risk". That is a property of the shape, not of the example -- EV is smooth
    and zero at the origin, so near its peak it is approximately quadratic and
    ``EV(2s*/3) = (8/9) EV(s*)``, which is the 90% in the framework's numbers.
    ``recommended_stake`` is the two-thirds point unless ``default="max_ev"``.

    ``at_capacity`` is True when the edge is still positive at the bottom of
    the book handed in, so the maximum is the whole of the visible depth rather
    than an interior optimum. The two-thirds point then captures less than 8/9
    -- it is buying risk reduction, not protecting EV -- and the honest reading
    is that the book shown is too shallow to price the impact."""
    curve = ev_stake_curve(p, price_fn, commission, n_grid, max_stake)
    if curve.empty or curve["ev"].max() <= 0:
        return {"max_ev_stake": 0.0, "max_ev": 0.0, "two_thirds_stake": 0.0, "two_thirds_ev": 0.0,
                "ev_captured_pct": np.nan, "recommended_stake": 0.0, "curve": curve}
    i = int(curve["ev"].to_numpy().argmax())
    s_star, ev_star = float(curve["stake"].iloc[i]), float(curve["ev"].iloc[i])
    s_two = 2.0 * s_star / 3.0
    m, a = price_fn(s_two)
    ev_two = float(m * (p * net_odds(a, commission) - 1.0))
    return {"max_ev_stake": s_star, "max_ev": ev_star, "two_thirds_stake": s_two,
            "two_thirds_ev": ev_two, "ev_captured_pct": 100.0 * ev_two / ev_star,
            "avg_price_at_max": float(curve["avg_price"].iloc[i]), "avg_price_at_two_thirds": float(a),
            "at_capacity": bool(i == len(curve) - 1),
            "recommended_stake": s_two if default == "two_thirds" else s_star, "curve": curve}


def split_order(total_stake: float, prices, sizes, n_slices: int = 4, replenish: float = 0.5,
                bsp_share: float = 0.0, bsp_price: float | None = None,
                commission: float = 0.05) -> pd.DataFrame:
    """Split an order across the pre-off window and into BSP (§V.4).

    A share ``bsp_share`` is left to the closing auction, where it matches at
    BSP with no ladder impact but at a price not known when the order is
    placed. The rest is cut into ``n_slices`` equal slices worked across the
    window; between slices the consumed depth comes back by a factor
    ``replenish`` (1.0 = the book refills completely, 0.0 = it never does).

    The reason to split is arithmetic: each slice starts again at the top of a
    partly refilled book, so the blended price beats one order of the same size
    whenever ``replenish > 0``. ``replenish`` is a modelling assumption and
    should be measured against real fills before it is trusted; with
    ``replenish = 0`` splitting achieves exactly nothing, which is the honest
    null."""
    pr, sz = _ladder_arrays(prices, sizes)
    total = float(max(total_stake, 0.0))
    bsp_amt = total * float(np.clip(bsp_share, 0.0, 1.0))
    work = total - bsp_amt
    rows, avail = [], sz.astype(float).copy()
    per = work / max(int(n_slices), 1)
    for k in range(max(int(n_slices), 1)):
        f = ladder_fill(pr, avail, per)
        rows.append({"slice": k + 1, "venue": "exchange", "stake": per, "matched": f["matched"],
                     "unmatched": f["unmatched"], "avg_price": f["avg_price"],
                     "slippage_pct": f["slippage_pct"]})
        room = np.maximum(per - np.concatenate(([0.0], np.cumsum(avail)[:-1])), 0.0)
        consumed = np.minimum(avail, room)
        avail = avail - consumed + float(np.clip(replenish, 0.0, 1.0)) * consumed
    if bsp_amt > 0:
        rows.append({"slice": len(rows) + 1, "venue": "bsp", "stake": bsp_amt, "matched": bsp_amt,
                     "unmatched": 0.0, "avg_price": bsp_price if bsp_price else np.nan,
                     "slippage_pct": np.nan})
    out = pd.DataFrame(rows)
    m = out["matched"].to_numpy(float)
    pxl = out["avg_price"].to_numpy(float)
    ok = np.isfinite(pxl) & (m > 0)
    blended = float((m[ok] * pxl[ok]).sum() / m[ok].sum()) if ok.any() else np.nan
    one_shot = ladder_fill(pr, sz, work) if work > 0 else {"avg_price": np.nan, "matched": 0.0}
    out.attrs["split"] = {"total": total, "matched": float(m.sum()), "blended_price": blended,
                          "blended_net_price": float(net_odds(blended, commission)) if np.isfinite(blended) else np.nan,
                          "one_shot_price": float(one_shot["avg_price"]),
                          "one_shot_matched": float(one_shot["matched"]),
                          "bsp_share": float(np.clip(bsp_share, 0.0, 1.0)), "replenish": float(replenish)}
    return out


def apply_depth_fills(bets: pd.DataFrame, stake_col: str = "stake", depth_col: str = "depth",
                      price_col: str = "bsp", impact: float = 0.5, commission: float = 0.05,
                      max_depth_share: float = 1.0) -> pd.DataFrame:
    """Depth-limited stakes and the returns they actually earn.

    Used where a per-runner ladder is not stored but a single depth figure is
    (``betfair_client`` persists ``best_back_size``; the historic files carry
    matched volume only). The stake is capped at ``max_depth_share`` of the
    available depth and the price is degraded linearly in the share of depth
    consumed, ``avg = price * (1 - impact * matched / depth)`` -- a crude stand
    -in for the ladder walk, deliberately conservative, and replaced by
    ``ladder_fill`` wherever the real ladder was captured at decision time.

    Adds ``stake_matched``, ``avg_price``, ``slippage_pct`` and
    ``ret_back_filled``: the per-unit return at the achieved price, commission
    on net winnings as everywhere else."""
    d = bets.copy()
    stake = pd.to_numeric(d[stake_col], errors="coerce").fillna(0.0).to_numpy(float)
    price = pd.to_numeric(d[price_col], errors="coerce").to_numpy(float)
    if depth_col in d.columns:
        depth = pd.to_numeric(d[depth_col], errors="coerce").fillna(0.0).to_numpy(float)
    else:
        depth = np.full(len(d), np.inf)
    cap = np.where(np.isfinite(depth), max_depth_share * depth, np.inf)
    matched = np.minimum(stake, cap)
    share = np.where(np.isfinite(depth) & (depth > 0), matched / depth, 0.0)
    avg = price * (1.0 - float(impact) * np.clip(share, 0.0, 1.0))
    avg = np.maximum(avg, 1.0 + 1e-9)
    won = d["won"].to_numpy() if "won" in d.columns else np.zeros(len(d), dtype=bool)
    d["stake_matched"] = matched
    d["unmatched"] = stake - matched
    d["avg_price"] = avg
    d["slippage_pct"] = np.where(price > 0, 1.0 - avg / price, 0.0)
    d["ret_back_filled"] = np.where(np.asarray(won, bool), (avg - 1.0) * (1.0 - commission), -1.0)
    return d


def liquidity_stress(bets: pd.DataFrame, stake_col: str = "kelly_full", depth_col: str = "depth",
                     price_col: str = "bsp", race_col: str = "raceid", fraction: float = 0.25,
                     impact: float = 0.5, commission: float = 0.05,
                     max_depth_share: float = 0.2, bank: float = 1.0) -> pd.DataFrame:
    """§VI.3(d): does the plan survive the liquidity it actually had?

    Runs the bankroll path twice -- once assuming every stake fills at the
    quoted price, once with stakes capped at ``max_depth_share`` of available
    depth and the price degraded by ``apply_depth_fills`` -- and reports both
    side by side, plus the share of intended turnover that never got matched.
    A strategy whose growth depends on filling the whole book is not a
    strategy."""
    d = bets.copy()
    d["_stake"] = fraction * pd.to_numeric(d[stake_col], errors="coerce").fillna(0.0) * float(bank)
    ideal = bankroll_path(d.assign(**{stake_col: pd.to_numeric(d[stake_col], errors="coerce").fillna(0.0)}),
                          fraction=fraction, stake_col=stake_col, race_col=race_col, commission=commission)
    filled = apply_depth_fills(d, stake_col="_stake", depth_col=depth_col, price_col=price_col,
                               impact=impact, commission=commission, max_depth_share=max_depth_share)
    filled["_f"] = np.where(bank > 0, filled["stake_matched"] / float(bank), 0.0)
    stressed = bankroll_path(filled.assign(ret_back=filled["ret_back_filled"]),
                             fraction=1.0, stake_col="_f", race_col=race_col, commission=commission)
    intended = float(d["_stake"].sum())
    rows = []
    for label, path in (("quoted price, unlimited depth", ideal), ("depth-capped with slippage", stressed)):
        s = growth_stats(path)
        s["scenario"] = label
        rows.append(s)
    out = pd.DataFrame(rows)
    out.attrs["liquidity"] = {
        "intended_turnover": intended, "matched_turnover": float(filled["stake_matched"].sum()),
        "unmatched_pct": float(100.0 * filled["unmatched"].sum() / intended) if intended > 0 else 0.0,
        "mean_slippage_pct": float(100.0 * filled.loc[filled["stake_matched"] > 0, "slippage_pct"].mean())
        if (filled["stake_matched"] > 0).any() else 0.0,
        "max_depth_share": max_depth_share, "impact": impact}
    cols = ["scenario"] + [c for c in out.columns if c != "scenario"]
    return out[cols]


# ---------------------------------------------------------------------------
# Grinold: IC, breadth and the IR they imply (§IIA.2, B11 / B12 / O21)
# ---------------------------------------------------------------------------

def realised_ic(bets: pd.DataFrame, p_col: str = "p_model", race_col: str = "raceid",
                y_col: str = "won", method: str = "pearson") -> pd.DataFrame:
    """Information coefficient per race: skill on one forecast.

    The cross-sectional correlation, within a race, between what the model said
    and what happened. With one winner per race the outcome vector is a single
    1 among zeros, so a race's IC is bounded and noisy; the useful object is the
    mean over many races and its t-statistic, which is what ``grinold_report``
    uses. Races where every runner carries the same probability, or where no
    winner is recorded, have no IC and are dropped.

    ``method="spearman"`` correlates ranks instead, which is the equity
    convention and is insensitive to how compressed the probabilities are --
    worth comparing against the Pearson version, because a model whose ordering
    is sound but whose confidence is not shows a much better rank IC (which is
    exactly what ``STAKING_REPORT.md`` found for this one)."""
    d = bets[[race_col, p_col, y_col]].dropna().copy()
    if d.empty:
        return pd.DataFrame(columns=[race_col, "n", "ic"])
    x = pd.to_numeric(d[p_col], errors="coerce")
    y = pd.to_numeric(d[y_col].astype(float), errors="coerce")
    if method == "spearman":
        x = x.groupby(d[race_col]).rank()
    elif method != "pearson":
        raise ValueError("method must be 'pearson' or 'spearman'")
    g = d[race_col]
    xc = x - x.groupby(g).transform("mean")
    yc = y - y.groupby(g).transform("mean")
    num = (xc * yc).groupby(g).sum()
    den = np.sqrt((xc ** 2).groupby(g).sum() * (yc ** 2).groupby(g).sum())
    ic = (num / den.replace(0.0, np.nan)).rename("ic")
    out = pd.concat([ic, g.groupby(g).size().rename("n")], axis=1).reset_index()
    return out.dropna(subset=["ic"])


def effective_breadth(bets: pd.DataFrame, race_col: str = "raceid",
                      meeting_col: str | None = None, day_col: str = "race_date",
                      rho_meeting: float = 0.20, rho_day: float = 0.05) -> dict:
    """How many independent bets there really are (§IIA.2).

    Breadth counts independent *forecasts*, and a race is one forecast however
    many runners are backed in it. Races on the same card share a going read
    and a track bias, so they are correlated at ``rho_meeting``; races on the
    same day at ``rho_day``. For a nested equicorrelation structure the
    effective count is

        N_eff = R^2 / sum_{r,s} rho_rs

    and the double sum is available from group sizes alone, so nothing of order
    R^2 is ever built. ``N_eff = R`` only when the two correlations are zero;
    a full card of eight races at rho = 0.2 is worth about 3.4 independent
    bets, not 8. That shortfall is the diagnostic §IIA.2 asks for."""
    d = bets
    if d.empty:
        return {"races": 0, "breadth_effective": 0.0, "meetings": 0, "days": 0}
    day = d[day_col].astype(str) if day_col in d.columns else pd.Series("all", index=d.index)
    if meeting_col is not None and meeting_col in d.columns:
        meet = d[meeting_col].astype(str)
    elif "track" in d.columns:
        meet = d["track"].astype(str) + "|" + day
    else:
        meet = day
    key = pd.DataFrame({"race": d[race_col].astype(str), "meet": meet, "day": day}).drop_duplicates("race")
    R = len(key)
    n_m = key.groupby("meet").size().to_numpy(float)
    n_d = key.groupby("day").size().to_numpy(float)
    m_per_d = key.drop_duplicates("meet").groupby("day").size()
    within_meet = float((n_m ** 2 - n_m).sum())
    within_day = float((n_d ** 2 - n_d).sum())
    total = R + rho_meeting * within_meet + rho_day * (within_day - within_meet)
    return {"races": int(R), "meetings": int(key["meet"].nunique()), "days": int(key["day"].nunique()),
            "races_per_meeting": float(n_m.mean()), "meetings_per_day": float(m_per_d.mean()),
            "breadth_effective": float(R ** 2 / total) if total > 0 else 0.0,
            "breadth_ratio": float(R / total) if total > 0 else 0.0,
            "rho_meeting": float(rho_meeting), "rho_day": float(rho_day)}


def grinold_report(bets: pd.DataFrame, p_col: str = "p_model", ret_col: str = "ret_back",
                   race_col: str = "raceid", day_col: str = "race_date",
                   meeting_col: str | None = None, y_col: str = "won",
                   rho_meeting: float | None = None, rho_day: float | None = None,
                   price_col: str = "bsp", commission: float = 0.05,
                   field: pd.DataFrame | None = None) -> dict:
    """Realised information ratio against what independence would give (§IIA.2, O21).

    ``IR = IC * sqrt(breadth)`` is the framework's strategy frame: skill per
    bet times the square root of the number of *independent* bets. Turned into
    a diagnostic, the usable question is not whether the realised IR equals
    ``IC * sqrt(N)`` on some common scale -- the two are measured in different
    units and forcing them together invites a circular answer -- but whether
    the bet stream behaves like ``N`` independent bets at all:

        ir_per_bet        = mean(return) / sd(return), one bet
        ir_if_independent = ir_per_bet * sqrt(mean bets per day)
        ir_realised       = mean(daily return) / sd(daily return)

    Under independence the daily standard deviation falls as the square root of
    the count and the last two agree. ``ir_shortfall_ratio`` below 1 means the
    daily series is noisier than the count says, which is positive correlation
    inside the day, and ``breadth_implied`` -- the count backed out of the
    realised series, ``(ir_realised / ir_per_bet)^2`` -- is the number of
    genuinely independent bets a day actually delivered. §IIA.2 names the usual
    culprit: shared model error across a card, going estimation and track bias,
    not the selection logic.

    Alongside it, the structural view: the mean per-race IC, and
    ``breadth_effective`` from ``effective_breadth`` on correlations measured by
    ``estimate_block_correlations`` on these same bets. ``ir_grinold`` applies
    the framework's formula to those two; it is on the IC scale rather than the
    P&L scale, so read it against itself over time rather than against
    ``ir_realised``.

    The IC is a *cross-sectional* correlation, so it needs the whole race, not
    the one runner that was backed: pass every runner as ``field`` when
    ``bets`` holds selections. Without it a one-bet-per-race stream has no IC
    at all and the function says so with a NaN rather than inventing one.

    The law also prices the alternative: doubling breadth is worth the same as
    a 41% improvement in IC, which is the arithmetic behind "many bets, small
    margins" and against waiting for the big-edge race."""
    d = bets.dropna(subset=[ret_col]).copy()
    if d.empty:
        return {"days": 0, "races": 0, "bets": 0, "ic_mean": np.nan, "ir_realised": np.nan,
                "ir_if_independent": np.nan, "ir_shortfall_ratio": np.nan}
    if rho_meeting is None or rho_day is None:
        est = estimate_block_correlations(d, ret_col=ret_col, p_col=p_col, price_col=price_col,
                                          race_col=race_col, meeting_col=meeting_col,
                                          day_col=day_col, commission=commission)
        if rho_meeting is None:
            rho_meeting = float(np.nan_to_num(est["rho_meeting"], nan=0.0))
        if rho_day is None:
            rho_day = float(np.nan_to_num(est["rho_day"], nan=0.0))
        rho_day = min(rho_day, rho_meeting)
    f = d if field is None else field
    ic = realised_ic(f, p_col=p_col, race_col=race_col, y_col=y_col)
    ic_rank = realised_ic(f, p_col=p_col, race_col=race_col, y_col=y_col, method="spearman")
    day = d[day_col].astype(str) if day_col in d.columns else pd.Series("all", index=d.index)
    r = pd.to_numeric(d[ret_col], errors="coerce")
    daily = r.groupby(day).mean()
    per_day_bets = float(r.groupby(day).size().mean())
    br = effective_breadth(d, race_col=race_col, meeting_col=meeting_col, day_col=day_col,
                           rho_meeting=rho_meeting, rho_day=rho_day)
    n_days = max(br["days"], 1)
    ic_mean = float(ic["ic"].mean()) if len(ic) else np.nan
    ic_se = float(ic["ic"].std(ddof=1) / np.sqrt(len(ic))) if len(ic) > 1 else np.nan
    sd_bet = float(r.std(ddof=1)) if len(r) > 1 else np.nan
    sd_daily = float(daily.std(ddof=1)) if len(daily) > 1 else np.nan
    ir_bet = float(r.mean() / sd_bet) if sd_bet and np.isfinite(sd_bet) and sd_bet > 0 else np.nan
    ir_ind = float(ir_bet * np.sqrt(per_day_bets)) if np.isfinite(ir_bet) else np.nan
    ir_real = float(daily.mean() / sd_daily) if sd_daily and np.isfinite(sd_daily) and sd_daily > 0 else np.nan
    breadth_per_day = br["breadth_effective"] / n_days
    return {
        "days": int(br["days"]), "races": int(br["races"]), "bets": int(len(d)),
        "bets_per_day": per_day_bets,
        "ic_mean": ic_mean, "ic_se": ic_se,
        "ic_t": float(ic_mean / ic_se) if ic_se and np.isfinite(ic_se) and ic_se > 0 else np.nan,
        "ic_rank_mean": float(ic_rank["ic"].mean()) if len(ic_rank) else np.nan,
        "breadth_nominal": float(br["races"]), "breadth_effective": float(br["breadth_effective"]),
        "breadth_per_day": float(breadth_per_day),
        "breadth_lost_pct": float(100.0 * (1.0 - br["breadth_effective"] / br["races"])) if br["races"] else np.nan,
        "rho_meeting": float(rho_meeting), "rho_day": float(rho_day),
        "mean_bet_return": float(r.mean()), "sd_bet_return": sd_bet, "ir_per_bet": ir_bet,
        "mean_daily_return": float(daily.mean()), "sd_daily_return": sd_daily,
        "ir_if_independent": ir_ind, "ir_realised": ir_real,
        "ir_shortfall_ratio": float(ir_real / ir_ind) if ir_ind and np.isfinite(ir_ind) and ir_ind != 0 else np.nan,
        "breadth_implied": float((ir_real / ir_bet) ** 2) if ir_bet and np.isfinite(ir_bet) and ir_bet != 0 else np.nan,
        "ir_grinold": float(ic_mean * np.sqrt(max(breadth_per_day, 0.0))) if np.isfinite(ic_mean) else np.nan,
        "t_stat_daily": float(ir_real * np.sqrt(len(daily))) if np.isfinite(ir_real) else np.nan,
        "ic_gain_equivalent_to_doubling_breadth_pct": 41.4,
    }


def required_bets(edge: float, sd: float | None = None, bets: pd.DataFrame | None = None,
                  ret_col: str = "ret_back", alpha: float = 0.05, power: float = 0.80,
                  one_sided: bool = True) -> dict:
    """Sample size to detect an edge of a given size (§VI.2.7).

    ``n = ((z_alpha + z_beta) * sd / edge)^2`` on the per-bet return
    distribution. Supply ``sd`` or a frame of settled bets to take it from.

    The framework's "of the order of 1,000+ bets" holds only where the return
    standard deviation is small. A flat back bet averaging 5.0 has ``sd`` near
    2.0, and detecting a 2% edge at 5% and 80% power then needs tens of
    thousands of bets, not thousands. The function returns the honest number for
    the distribution handed to it, which is the point of pre-registering it."""
    from scipy.stats import norm
    if sd is None:
        if bets is None or ret_col not in bets.columns:
            raise ValueError("supply sd or a frame of settled bets")
        sd = float(pd.to_numeric(bets[ret_col], errors="coerce").std(ddof=1))
    z_a = float(norm.ppf(1 - alpha)) if one_sided else float(norm.ppf(1 - alpha / 2))
    z_b = float(norm.ppf(power))
    n = float(((z_a + z_b) * sd / abs(edge)) ** 2) if edge else np.inf
    return {"edge": float(edge), "sd": float(sd), "alpha": alpha, "power": power,
            "one_sided": bool(one_sided), "required_bets": n,
            "detectable_edge_at_1000": float((z_a + z_b) * sd / np.sqrt(1000.0))}


# ---------------------------------------------------------------------------
# Attribution and decay (§IIA.7, §VI.4, B20 / B21 / O20)
# ---------------------------------------------------------------------------

def _ols_cluster(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS with cluster-robust standard errors. Returns (beta, se, r2)."""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    df = pd.DataFrame(X * u[:, None])
    Sg = df.groupby(groups).sum().to_numpy(float)
    G = Sg.shape[0]
    meat = Sg.T @ Sg
    adj = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1)) if G > 1 else 1.0
    V = adj * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = float(1.0 - (u ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan
    return beta, se, r2


def _default_factors(d: pd.DataFrame, price_col: str) -> dict[str, pd.Series]:
    """Factor columns for attribution, from whatever the frame carries."""
    out: dict[str, pd.Series] = {}
    if price_col in d.columns:
        out["price_band"] = pd.cut(pd.to_numeric(d[price_col], errors="coerce"),
                                   [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, np.inf],
                                   labels=["1-2", "2-3", "3-5", "5-8", "8-12", "12-20", "20+"])
    if "field" in d.columns:
        out["field_band"] = pd.cut(pd.to_numeric(d["field"], errors="coerce"),
                                   [0, 5, 8, 11, 15, 100], labels=["<=5", "6-8", "9-11", "12-15", "16+"])
    for col, name in (("race_class", "class"), ("class", "class"), ("going", "going"),
                      ("track", "course"), ("course", "course"), ("race_type", "race_type")):
        if col in d.columns and name not in out:
            out[name] = d[col].astype(str)
    for col in ("distance_yards", "distance_f", "distance"):
        if col in d.columns:
            v = pd.to_numeric(d[col], errors="coerce")
            if v.notna().any():
                out["distance_band"] = pd.qcut(v, 4, duplicates="drop")
                break
    return out


def factor_attribution(bets: pd.DataFrame, ret_col: str = "ret_back", factors=None,
                       price_col: str = "model_price", cluster_col: str = "raceid",
                       min_count: int = 30) -> pd.DataFrame:
    """Decompose realised P&L onto race factors (§IIA.7).

    Per-bet profit and loss is regressed on dummies for favourite/longshot
    band, field size, class, going, distance and course, with one reference
    level per factor and standard errors clustered on the race, since bets in a
    race share an outcome. The question it answers is the one segment ROI
    tables cannot: **is the edge one real signal, or an accidental concentrated
    bet on a single factor?** If four fifths of the P&L traces to short prices
    in small fields, that is a risk to know about before it reverses, not after.

    ``price_col`` defaults to ``model_price`` -- the model's own forecast -- so
    the bands are the ones known when the bet was struck. Banding on the
    settled BSP is legitimate in an ex-post decomposition but tells you about
    the result rather than about the rule.

    Levels with fewer than ``min_count`` bets are pooled into the factor's
    reference level rather than fitted on noise. Returns one row per level with
    its share of total P&L, its coefficient against the reference and a
    cluster-robust t; the summary sits in ``result.attrs["attribution"]``."""
    d = bets.dropna(subset=[ret_col]).copy()
    if d.empty:
        return pd.DataFrame(columns=["factor", "level", "n", "mean_return_pct", "pnl_share_pct",
                                     "coef_pct", "se_pct", "t"])
    y = pd.to_numeric(d[ret_col], errors="coerce").to_numpy(float)
    cols = factors if factors is not None else _default_factors(d, price_col)
    if not isinstance(cols, dict):
        cols = {c: d[c].astype(str) for c in cols if c in d.columns}
    design, rows = [np.ones(len(d))], []
    names = ["intercept"]
    for fname, series in cols.items():
        s = pd.Series(series, index=d.index).astype(str).fillna("na")
        counts = s.value_counts()
        keep = [lv for lv in counts.index if counts[lv] >= min_count]
        if len(keep) < 2:
            continue
        ref = counts[keep].idxmax()                 # the biggest level is the reference
        for lv in keep:
            if lv == ref:
                continue
            design.append((s == lv).to_numpy(float))
            names.append(f"{fname}={lv}")
            rows.append({"factor": fname, "level": lv, "reference": ref})
    X = np.column_stack(design)
    groups = d[cluster_col].astype(str).to_numpy() if cluster_col in d.columns else np.arange(len(d))
    beta, se, r2 = _ols_cluster(X, y, groups)
    total_pnl = float(y.sum())
    out = []
    for i, meta in enumerate(rows, start=1):
        mask = X[:, i] > 0
        out.append({**meta, "n": int(mask.sum()),
                    "mean_return_pct": float(100 * y[mask].mean()),
                    "pnl_share_pct": float(100 * y[mask].sum() / total_pnl) if total_pnl else np.nan,
                    "coef_pct": float(100 * beta[i]), "se_pct": float(100 * se[i]),
                    "t": float(beta[i] / se[i]) if se[i] > 0 else np.nan})
    res = pd.DataFrame(out)
    shares = []
    for fname, series in cols.items():
        s = pd.Series(series, index=d.index).astype(str).fillna("na")
        g = pd.Series(y, index=d.index).groupby(s).sum()
        if total_pnl:
            shares.append((fname, str(g.abs().idxmax()), float(100 * g.loc[g.abs().idxmax()] / total_pnl)))
    top = max(shares, key=lambda t: abs(t[2])) if shares else ("", "", np.nan)
    res.attrs["attribution"] = {
        "bets": int(len(d)), "total_pnl": total_pnl, "r2": r2,
        "intercept_pct": float(100 * beta[0]), "intercept_se_pct": float(100 * se[0]),
        "n_terms": int(X.shape[1] - 1), "clusters": int(pd.unique(groups).size),
        "largest_single_level": f"{top[0]}={top[1]}", "largest_level_pnl_share_pct": top[2],
        "concentration_note": "a single level carrying most of the P&L is a factor bet, not an edge"}
    return res.sort_values("pnl_share_pct", key=lambda s: s.abs(), ascending=False) if len(res) else res


def _race_losses(d: pd.DataFrame, p_col: str, race_col: str, y_col: str,
                 market_col: str | None) -> pd.DataFrame:
    """Per-race negative log likelihood of the winner, model / reference / uniform."""
    t = d.dropna(subset=[p_col]).copy()
    t["_n"] = t.groupby(race_col)[race_col].transform("size")
    t["_p"] = pd.to_numeric(t[p_col], errors="coerce").clip(1e-12, 1)
    t["_p"] = t["_p"] / t.groupby(race_col)["_p"].transform("sum")
    w = t[pd.to_numeric(t[y_col].astype(float), errors="coerce") > 0]
    out = pd.DataFrame({"ll_model": -np.log(w["_p"].to_numpy(float)),
                        "ll_unif": np.log(w["_n"].to_numpy(float))},
                       index=w[race_col].astype(str))
    if market_col and market_col in t.columns:
        t["_m"] = pd.to_numeric(t[market_col], errors="coerce").clip(1e-12, 1)
        t["_m"] = t["_m"] / t.groupby(race_col)["_m"].transform("sum")
        out["ll_ref"] = -np.log(t.loc[w.index, "_m"].to_numpy(float))
    else:
        out["ll_ref"] = out["ll_unif"]
    return out


DECAY_BUCKETS = (0, 7, 14, 30, 60, 90, 180, 10_000)


def alpha_decay(preds: pd.DataFrame, date_col: str = "race_date", p_col: str = "p_model",
                race_col: str = "raceid", y_col: str = "won", market_col: str | None = "p_market",
                refit_col: str | None = None, refit_dates=None, ret_col: str | None = "ret_back",
                buckets=DECAY_BUCKETS) -> pd.DataFrame:
    """How fast the edge decays after a refit (§IIA.7, B21).

    ``dR2 = R2_model - R2_reference`` with ``R2 = 1 - L(.)/L(1/n)``, computed in
    buckets of days since the model was last refitted. The reference is the
    market when ``market_col`` is present and the uniform 1/n otherwise, in
    which case ``dR2`` is just the model's own McFadden R2 -- still decaying,
    but against a weaker bar; the reference used is recorded in the attrs.

    Because R2 is a ratio of summed losses rather than a mean, the bucket
    figures are computed from the summed per-race log likelihoods, which is
    exact. The slope is fitted on the per-race quantity
    ``(ll_ref - ll_model) / ll_unif`` against days since refit, with a
    cluster-robust t by day, and ``days_to_zero`` extrapolates where the edge
    reaches zero. That number, not convention, is what should set the refit
    cadence -- and if the slope is not distinguishable from zero the honest
    reading is that the data cannot yet say.

    Refit dates come from ``refit_col`` (a per-row date), from ``refit_dates``
    (each race is attributed to the most recent refit at or before it), or
    failing both from the earliest date in the frame."""
    d = preds.dropna(subset=[date_col, p_col]).copy()
    if d.empty:
        return pd.DataFrame(columns=["days_since_refit", "races", "delta_r2"])
    when = pd.to_datetime(d[date_col], errors="coerce")
    if refit_col and refit_col in d.columns:
        refit = pd.to_datetime(d[refit_col], errors="coerce")
    elif refit_dates is not None:
        rd = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(list(refit_dates)), errors="coerce").dropna()))
        pos = np.clip(rd.searchsorted(when.to_numpy(), side="right") - 1, 0, len(rd) - 1)
        refit = pd.Series(rd[pos], index=d.index)
    else:
        refit = pd.Series(when.min(), index=d.index)
    d["_days"] = (when - refit).dt.days.clip(lower=0)
    d["_day"] = when.dt.strftime("%Y-%m-%d")
    loss = _race_losses(d, p_col, race_col, y_col, market_col)
    key = d.drop_duplicates(race_col).set_index(d.drop_duplicates(race_col)[race_col].astype(str))
    loss = loss.join(key[["_days", "_day"]], how="inner")
    if ret_col and ret_col in d.columns:
        roi = pd.to_numeric(d[ret_col], errors="coerce").groupby(d[race_col].astype(str)).mean()
        loss = loss.join(roi.rename("ret"), how="left")
    loss["bucket"] = pd.cut(loss["_days"], list(buckets), right=False)
    rows = []
    for b, g in loss.groupby("bucket", observed=True):
        if g.empty:
            continue
        unif = float(g["ll_unif"].sum())
        rows.append({"days_since_refit": str(b), "days_mid": float(g["_days"].mean()),
                     "races": int(len(g)),
                     "r2_model": float(1 - g["ll_model"].sum() / unif) if unif else np.nan,
                     "r2_reference": float(1 - g["ll_ref"].sum() / unif) if unif else np.nan,
                     "delta_r2": float((g["ll_ref"].sum() - g["ll_model"].sum()) / unif) if unif else np.nan,
                     "roi_pct": float(100 * g["ret"].mean()) if "ret" in g.columns else np.nan})
    out = pd.DataFrame(rows)
    per_race = ((loss["ll_ref"] - loss["ll_model"]) / loss["ll_unif"].replace(0, np.nan)).to_numpy(float)
    ok = np.isfinite(per_race)
    X = np.column_stack([np.ones(ok.sum()), loss.loc[ok, "_days"].to_numpy(float)])
    beta, se, _ = _ols_cluster(X, per_race[ok], loss.loc[ok, "_day"].to_numpy())
    slope30 = float(beta[1] * 30.0)
    out.attrs["alpha_decay"] = {
        "reference": "market" if (market_col and market_col in d.columns) else "uniform 1/n",
        "races": int(ok.sum()), "delta_r2_at_refit": float(beta[0]),
        "slope_per_30d": slope30, "slope_se_per_30d": float(se[1] * 30.0),
        "slope_t": float(beta[1] / se[1]) if se[1] > 0 else np.nan,
        "days_to_zero": (0.0 if beta[0] <= 0 else
                         float(-beta[0] / beta[1]) if beta[1] < 0 else np.inf),
        "refit_cadence_note": "days_to_zero is an extrapolation; refit well inside it"}
    return out
