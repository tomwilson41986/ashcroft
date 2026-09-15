"""
Stage F / Stage C of the racing² master framework (v3), Sections II, IV, VI.

    Stage F  fundamental model: race-grouped softmax over market-free features
             (conditional logit; or LightGBM with the race-softmax objective)
    Stage C  combination: c ∝ exp(α ln f + γ ln π [+ β ln π_early]) fitted on
             OUT-OF-FOLD fundamental probabilities
    Metric   McFadden pseudo-R² vs the uniform race model and
             ΔR² = R²_combined − R²_market with a bootstrap CI by race — the
             only number the framework treats as predictive of profit.

Why this exists: train_bfsp.py regresses on log(BFSP), so the market is its
*target*; such a model can only reproduce the market (lossily) and its
"overlays" are its own reconstruction error. A true Stage F must be trained
on the winner with market-free features so that ΔR² over the market is
measurable. This module provides that estimator and the measurement.

Everything here works on plain numpy arrays: X (n, p), y (n,) in {0,1} with
exactly one winner per race, groups (n,) race identifiers.

Two measurement-integrity pieces live here because they belong to the same
boundary as the estimator:

    apply_unratable_rule    Benter's confidence gate (Section IV.1): a runner
                            the fundamental model cannot rate gets the
                            market's probability, the rest are renormalised,
                            and a race with nothing ratable is skipped
    purge/embargo           in ``walk_forward_stage_f``, the gap either side
                            of a fold boundary that stops trailing-window
                            features carrying training labels into the test
                            period (Section VI.2.3)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp


# ---------------------------------------------------------------------------
# Race-grouped softmax utilities
# ---------------------------------------------------------------------------

def _encode_groups(groups):
    codes, uniq = pd.factorize(pd.Series(np.asarray(groups)), sort=False)
    return codes, len(uniq)


def _softmax_sorted(sc: np.ndarray, starts: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Softmax over contiguous groups (rows sorted by group; starts = first index of each group)."""
    m = np.maximum.reduceat(sc, starts)
    e = np.exp(sc - m[seg])
    return e / np.add.reduceat(e, starts)[seg]


def _logsumexp_sorted(sc: np.ndarray, starts: np.ndarray, seg: np.ndarray) -> np.ndarray:
    m = np.maximum.reduceat(sc, starts)
    return m + np.log(np.add.reduceat(np.exp(sc - m[seg]), starts))


def race_softmax(scores: np.ndarray, groups) -> np.ndarray:
    """Softmax of scores within each race (any row order)."""
    codes, G = _encode_groups(groups)
    s = np.asarray(scores, float)
    order = np.argsort(codes, kind="stable")
    sc, cs = s[order], codes[order]
    starts = np.r_[0, np.flatnonzero(np.diff(cs)) + 1]
    seg = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(sc)]))
    p = np.empty(len(s)); p[order] = _softmax_sorted(sc, starts, seg)
    return p


def race_log_loss(p: np.ndarray, y: np.ndarray) -> float:
    """Mean over races of −ln p(winner)."""
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 1e-12, 1)
    return float(-np.sum(y * np.log(p)) / y.sum())


def mcfadden_r2(p: np.ndarray, y: np.ndarray, groups) -> float:
    """1 − L(model)/L(uniform), L = Σ_races ln p(winner)."""
    codes, _ = _encode_groups(groups)
    n_r = np.bincount(codes)[codes]
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 1e-12, 1)
    L = np.sum(y * np.log(p)); L0 = np.sum(y * np.log(1.0 / n_r))
    return float(1 - L / L0)


def delta_r2(p_model: np.ndarray, p_market: np.ndarray, y: np.ndarray, groups, n_boot: int = 500,
             level: float = 0.9, random_state: int = 0) -> dict:
    """ΔR² = R²(model) − R²(market) with a cluster (race) bootstrap CI."""
    codes, G = _encode_groups(groups)
    y = np.asarray(y, float)
    lm = y * np.log(np.clip(p_model, 1e-12, 1)); lk = y * np.log(np.clip(p_market, 1e-12, 1))
    n_r = np.bincount(codes)[codes]; l0 = y * np.log(1.0 / n_r)
    Lm = np.bincount(codes, weights=lm); Lk = np.bincount(codes, weights=lk); L0 = np.bincount(codes, weights=l0)
    r2m = 1 - Lm.sum() / L0.sum(); r2k = 1 - Lk.sum() / L0.sum()
    rng = np.random.default_rng(random_state)
    w = rng.multinomial(G, np.full(G, 1.0 / G), size=n_boot).astype(float)
    d = (1 - (w @ Lm) / (w @ L0)) - (1 - (w @ Lk) / (w @ L0))
    a = (1 - level) / 2
    return {"r2_model": float(r2m), "r2_market": float(r2k), "delta_r2": float(r2m - r2k),
            "ci": (float(np.quantile(d, a)), float(np.quantile(d, 1 - a))), "n_races": int(G),
            "p_delta_le_0": float(np.mean(d <= 0))}


# ---------------------------------------------------------------------------
# Purge and embargo (Section VI.2.3)
# ---------------------------------------------------------------------------
#
# A walk-forward split on the date alone is not clean, because the features are
# trailing statistics. A test race on 1 July carries trainer_form_30d, which is
# a function of the *labels* of every runner that trainer sent out in June — and
# those June rows are in the training set. The model can memorise them and read
# them back off the test feature. Purging drops the training rows inside that
# window; the embargo drops the test rows on the other side of the boundary,
# the ones whose trailing windows are most heavily made of training labels.
#
# (Lopez de Prado embargoes *training* rows after the test block. In a strict
# walk-forward, training is always earlier than test, so that direction has
# nothing to drop and the same defence has to be spent on the test side.)
#
# The defaults: PURGE_DAYS matches the longest bounded trailing window in the
# Stage F block (``{entity}_form_30d`` in model/connections.py), so no training
# row inside any rolling window survives. Unbounded statistics — expanding
# career means, the Kalman filter, the 270-day EWM — cannot be purged away
# without deleting the training set, so their (1/n-attenuated) channel stays
# open and is a known, documented limit rather than an accident. EMBARGO_DAYS
# is a week, which covers a horse's realistic turnaround and the fact that a
# meeting's runners reappear within days.

PURGE_DAYS = 30
EMBARGO_DAYS = 7


def purge_embargo_masks(dates, test_dates, gap_days: int = PURGE_DAYS, embargo_days: int = EMBARGO_DAYS):
    """Train / test row masks for one fold with a purge and an embargo.

    ``dates`` is the frame's date column, ``test_dates`` the fold's dates.
    Training keeps rows strictly before (fold start − gap_days); the test fold
    keeps rows from (fold start + embargo_days) on. Returns (train, test)
    boolean numpy arrays."""
    dt = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    start = pd.Timestamp(np.min(np.asarray(test_dates, dtype="datetime64[ns]")))
    tr = (dt < start - pd.Timedelta(days=int(gap_days))).values
    te = (dt.isin(pd.to_datetime(test_dates)) & (dt >= start + pd.Timedelta(days=int(embargo_days)))).values
    return tr, te


# ---------------------------------------------------------------------------
# Unratable runners (Section IV.1)
# ---------------------------------------------------------------------------

def apply_unratable_rule(p_model, p_market, groups, runs_count=None, kf_n=None, min_runs: int = 3,
                         min_kf_n: int = 1, min_ratable: int = 2):
    """Benter's rule: a runner the fundamental model cannot rate is given the
    market's probability, and the ratable runners are renormalised into what is
    left. Do not guess.

    A softmax has no way to say "no opinion": a first-time starter with no form
    still gets a score, the race normalises to one, and that invented
    probability is then compared with the market as if it meant something. The
    gate makes the abstention explicit.

        ratable_i   runs_count_i >= min_runs  and  kf_n_i >= min_kf_n
        p_i         = pi_i                             for unratable i
        p_i         = p_i (1 − Σ_unratable pi) / Σ_ratable p    otherwise

    ``runs_count`` and ``kf_n`` are both optional; whichever is supplied is
    applied (kf_n, the Kalman observation count, is the better gate because it
    counts runs the filter could actually read).

    Race skip: a race with fewer than ``min_ratable`` ratable runners is
    dropped entirely (NaN). The default of 2 is one stricter than Benter's
    "only unratable runners" — with a single ratable runner the
    renormalisation pins its probability to 1 − Σ pi, so the race is a copy of
    the market and contributes nothing but noise to ΔR². A race is also
    dropped if an unratable runner has no market price to stand in for it.

    Returns (p, ratable, race_ok): probabilities with NaN on skipped races, the
    per-runner gate, and the per-runner race-level keep flag."""
    p = np.asarray(p_model, float).copy(); pi = np.asarray(p_market, float)
    codes, G = _encode_groups(groups)
    ratable = np.isfinite(p)
    if runs_count is not None:
        ratable &= np.nan_to_num(np.asarray(runs_count, float), nan=-1.0) >= min_runs
    if kf_n is not None:
        ratable &= np.nan_to_num(np.asarray(kf_n, float), nan=-1.0) >= min_kf_n

    def per_race(v):
        return np.bincount(codes, weights=np.asarray(v, float), minlength=G)

    n_ratable = per_race(ratable)
    pi_unratable = per_race(np.where(~ratable, np.nan_to_num(pi), 0.0))
    p_ratable = per_race(np.where(ratable, np.nan_to_num(p), 0.0))
    no_price = per_race(~ratable & ~np.isfinite(pi)) > 0
    keep = (n_ratable >= min_ratable) & ~no_price & (pi_unratable < 1.0) & (p_ratable > 0)

    scale = np.where(keep, (1.0 - pi_unratable) / np.where(p_ratable > 0, p_ratable, 1.0), np.nan)
    out = np.where(ratable, np.nan_to_num(p) * scale[codes], pi)
    return np.where(keep[codes], out, np.nan), ratable, keep[codes]


# ---------------------------------------------------------------------------
# Conditional logit (Stage F, linear V)
# ---------------------------------------------------------------------------

class ConditionalLogit:
    """Race-grouped softmax with L2 penalty, fitted by L-BFGS on the concave
    log-likelihood. Features are standardised internally; race-constant
    features cancel in the softmax and should be interacted or dropped."""

    def __init__(self, l2: float = 1.0, max_iter: int = 500, tol: float = 1e-8):
        self.l2 = l2; self.max_iter = max_iter; self.tol = tol
        self.coef_ = None; self.mean_ = None; self.scale_ = None; self.n_iter_ = 0

    def _prep(self, X):
        X = np.asarray(X, float)
        X = np.where(np.isnan(X), 0.0, X)  # missing → 0 after standardisation (i.e. the mean)
        return (X - self.mean_) / self.scale_

    def fit(self, X, y, groups, sample_weight=None):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.mean_ = np.nanmean(X, axis=0); self.scale_ = np.nanstd(X, axis=0); self.scale_[self.scale_ == 0] = 1.0
        Z = self._prep(X)
        codes, G = _encode_groups(groups)
        order = np.argsort(codes, kind="stable")
        Zs, ys, cs = Z[order], y[order], codes[order]
        w = np.ones(G) if sample_weight is None else np.asarray(sample_weight, float)[order][np.r_[0, np.flatnonzero(np.diff(cs)) + 1]]
        starts = np.r_[0, np.flatnonzero(np.diff(cs)) + 1]; ends = np.r_[starts[1:], len(cs)]
        seg = np.repeat(np.arange(G), ends - starts)

        def f_and_g(beta):
            s = Zs @ beta
            lse = _logsumexp_sorted(s, starts, seg)
            p = np.exp(s - lse[seg])
            ll = np.sum(w[seg] * ys * (s - lse[seg]))
            nll = -ll / G + 0.5 * self.l2 * beta @ beta / G
            grad = -(Zs.T @ (w[seg] * (ys - p))) / G + self.l2 * beta / G
            return nll, grad

        res = minimize(f_and_g, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B",
                       options={"maxiter": self.max_iter, "gtol": self.tol})
        self.coef_ = res.x; self.n_iter_ = int(res.nit); self.converged_ = bool(res.success)
        return self

    def decision_function(self, X):
        return self._prep(X) @ self.coef_

    def predict_proba(self, X, groups):
        return race_softmax(self.decision_function(X), groups)

    def coef_table(self, names) -> pd.DataFrame:
        return pd.DataFrame({"feature": names, "coef_std": self.coef_}).sort_values("coef_std", key=np.abs, ascending=False)


# ---------------------------------------------------------------------------
# LightGBM race-softmax objective (Stage F, non-linear V)
# ---------------------------------------------------------------------------

def lgb_race_softmax(group_sizes: np.ndarray):
    """Custom objective + metric for lgb.train on a dataset SORTED BY RACE.

    grad = p − y, hess = p(1 − p) (diagonal), with p the within-race softmax of
    the raw scores — ListNet top-1 / Benter's conditional logit as a boosting
    objective (Section II.2 of the framework).

        fobj, feval = lgb_race_softmax(sizes)
        booster = lgb.train({**params, "objective": fobj}, dtrain, feval=feval, ...)   # LightGBM >= 4
    """
    sizes = np.asarray(group_sizes, int)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]; ends = np.cumsum(sizes)
    seg = np.repeat(np.arange(len(sizes)), sizes)

    def _p(scores):
        return _softmax_sorted(np.asarray(scores, float), starts, seg)

    def fobj(preds, dataset):
        y = dataset.get_label(); p = _p(np.asarray(preds, float))
        return p - y, np.clip(p * (1 - p), 1e-6, None)

    def feval(preds, dataset):
        y = dataset.get_label(); p = _p(np.asarray(preds, float))
        return "race_logloss", race_log_loss(p, y), False

    return fobj, feval


# ---------------------------------------------------------------------------
# Stage C and calibration
# ---------------------------------------------------------------------------

class StageC:
    """Second logit: c ∝ exp(α ln f + γ ln π + Σ β_k ln x_k). Fit on OOF f only."""

    def __init__(self, l2: float = 1e-6):
        self.model = ConditionalLogit(l2=l2)
        self.names_ = None

    @staticmethod
    def _design(f, pi, extra=None):
        cols = [np.log(np.clip(np.asarray(f, float), 1e-9, 1)), np.log(np.clip(np.asarray(pi, float), 1e-9, 1))]
        names = ["ln_f", "ln_pi"]
        if extra is not None:
            for k, v in extra.items():
                cols.append(np.log(np.clip(np.asarray(v, float), 1e-9, 1))); names.append(f"ln_{k}")
        return np.column_stack(cols), names

    def fit(self, f, pi, y, groups, extra=None):
        X, self.names_ = self._design(f, pi, extra)
        self.model.fit(X, y, groups)
        # coefficients on the ORIGINAL log scale (undo the internal standardisation)
        self.coef_raw_ = self.model.coef_ / self.model.scale_
        self.alpha_, self.gamma_ = float(self.coef_raw_[0]), float(self.coef_raw_[1])
        return self

    def predict_proba(self, f, pi, groups, extra=None):
        X, _ = self._design(f, pi, extra)
        return self.model.predict_proba(X, groups)


class TemperatureScaler:
    """p_T ∝ p^(1/T) within race; T fitted by log-loss on a held-out block.
    Preserves within-race ordering and the sum-to-one constraint."""

    def __init__(self):
        self.T_ = 1.0

    def fit(self, p, y, groups):
        lp = np.log(np.clip(np.asarray(p, float), 1e-12, 1))
        res = minimize_scalar(lambda T: race_log_loss(race_softmax(lp / T, groups), y), bounds=(0.2, 5.0), method="bounded")
        self.T_ = float(res.x)
        return self

    def transform(self, p, groups):
        return race_softmax(np.log(np.clip(np.asarray(p, float), 1e-12, 1)) / self.T_, groups)


# ---------------------------------------------------------------------------
# Walk-forward harness
# ---------------------------------------------------------------------------

def walk_forward_stage_f(df: pd.DataFrame, feature_cols, y_col: str, race_col: str, date_col: str,
                         market_col: str | None = None, l2: float = 1.0, n_folds: int = 6, min_train_races: int = 500,
                         gap_days: int = PURGE_DAYS, embargo_days: int = EMBARGO_DAYS) -> pd.DataFrame:
    """Chronological folds: fit Stage F on earlier dates, emit OOF probabilities
    (``f_oof``); if ``market_col`` is given, also fit Stage C on the OOF f of the
    *previous* folds and emit ``c_oof``. Returns the frame with those columns.

    ``gap_days`` purges and ``embargo_days`` embargoes either side of each fold
    boundary — see ``purge_embargo_masks`` for what that buys."""
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    dates = np.sort(d[date_col].unique()); edges = np.linspace(0, len(dates), n_folds + 1).astype(int)
    d["f_oof"] = np.nan; d["c_oof"] = np.nan; d["fold"] = -1
    X = d[feature_cols].astype(float).values; y = d[y_col].astype(float).values; g = d[race_col].values
    for k in range(1, n_folds):
        te_dates = dates[edges[k]: edges[k + 1]]
        if len(te_dates) == 0:
            continue
        tr, te = purge_embargo_masks(d[date_col], te_dates, gap_days=gap_days, embargo_days=embargo_days)
        if d.loc[tr, race_col].nunique() < min_train_races or te.sum() == 0:
            continue
        m = ConditionalLogit(l2=l2).fit(X[tr], y[tr], g[tr])
        d.loc[te, "f_oof"] = m.predict_proba(X[te], g[te]); d.loc[te, "fold"] = k
        if market_col is not None:
            prev = d["f_oof"].notna().values & ~te & (d["fold"].values < k) & (d["fold"].values > 0)
            if d.loc[prev, race_col].nunique() >= min_train_races:
                c = StageC().fit(d.loc[prev, "f_oof"].values, d.loc[prev, market_col].values, y[prev], g[prev])
                d.loc[te, "c_oof"] = c.predict_proba(d.loc[te, "f_oof"].values, d.loc[te, market_col].values, g[te])
    return d


def conditional_calibration(f, pi, y, groups, bands=(0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 1.01)) -> pd.DataFrame:
    """Benter Tables 3/4-style: where the fundamental model rates a horse above
    (and below) the market, expected vs actual win rate by model-probability band."""
    d = pd.DataFrame({"f": f, "pi": pi, "y": np.asarray(y, float)})
    d["side"] = np.where(d["f"] > d["pi"], "model > market", "model <= market")
    d["band"] = pd.cut(d["f"], bands, right=False)
    out = d.groupby(["side", "band"], observed=True).agg(n=("y", "size"), expected_model=("f", "mean"),
                                                        expected_market=("pi", "mean"), actual=("y", "mean")).reset_index()
    out["z_vs_model"] = (out["actual"] - out["expected_model"]) / np.sqrt(out["expected_model"] * (1 - out["expected_model"]) / out["n"])
    out["band"] = out["band"].astype(str)
    return out
