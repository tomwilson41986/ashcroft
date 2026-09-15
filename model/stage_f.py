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


def race_softmax(scores: np.ndarray, groups) -> np.ndarray:
    """Softmax of scores within each race (any row order)."""
    codes, G = _encode_groups(groups)
    s = np.asarray(scores, float)
    order = np.argsort(codes, kind="stable")
    sc, cs = s[order], codes[order]
    starts = np.r_[0, np.flatnonzero(np.diff(cs)) + 1]
    lse = np.empty(len(sc))
    for a, b in zip(starts, np.r_[starts[1:], len(sc)]):
        lse[a:b] = logsumexp(sc[a:b])
    p = np.empty(len(s)); p[order] = np.exp(sc - lse)
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
            lse = np.array([logsumexp(s[a:b]) for a, b in zip(starts, ends)])
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
        booster = lgb.train(params, dtrain, fobj=fobj, feval=feval, ...)
    """
    sizes = np.asarray(group_sizes, int)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]; ends = np.cumsum(sizes)
    seg = np.repeat(np.arange(len(sizes)), sizes)

    def _p(scores):
        lse = np.array([logsumexp(scores[a:b]) for a, b in zip(starts, ends)])
        return np.exp(scores - lse[seg])

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
                         market_col: str | None = None, l2: float = 1.0, n_folds: int = 6, min_train_races: int = 500) -> pd.DataFrame:
    """Chronological folds: fit Stage F on earlier dates, emit OOF probabilities
    (``f_oof``); if ``market_col`` is given, also fit Stage C on the OOF f of the
    *previous* folds and emit ``c_oof``. Returns the frame with those columns."""
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    dates = np.sort(d[date_col].unique()); edges = np.linspace(0, len(dates), n_folds + 1).astype(int)
    d["f_oof"] = np.nan; d["c_oof"] = np.nan; d["fold"] = -1
    X = d[feature_cols].astype(float).values; y = d[y_col].astype(float).values; g = d[race_col].values
    for k in range(1, n_folds):
        tr = d[date_col].isin(dates[: edges[k]]).values; te = d[date_col].isin(dates[edges[k]: edges[k + 1]]).values
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
