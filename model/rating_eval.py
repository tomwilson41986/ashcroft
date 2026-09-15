"""
Does an external rating add information beyond the model and the market?

``evaluate_rating_sets`` takes runner-level predictions (model probability,
BSP, outcome) joined to one or more rating columns and reports, per rating:

    coverage             share of runners with the rating
    concordance          within-race C-index of the rating vs finishing position
    corr_demeaned        race-demeaned correlation with the win outcome
    softmax_logloss      log-loss of softmax(rating / T) with T fitted on the
                         first half of the dates, scored on the second half,
                         next to the model's and the market's log-loss on the
                         same rows
    stack_gain_*         walk-forward stacked logistic regression on
                         [ln p_model, ln p_market] with and without the rating's
                         within-race z-score: the change in log-loss and in
                         Brier skill vs the market is the incremental value

``knockoff_screen`` runs model-X knockoffs over all rating z-scores at once
(plus ln p_model and ln p_market as always-in covariates) so correlated
sets do not all claim the same credit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

from model.diagnostics import brier, concordance_index, log_loss
from model.selection import knockoff_filter, univariate_pvalues


def _race_softmax(x: pd.Series, race: pd.Series, T: float) -> np.ndarray:
    z = x / max(T, 1e-6)
    z = z - z.groupby(race).transform("max")
    e = np.exp(z)
    return (e / e.groupby(race).transform("sum")).values


def fit_temperature(x: pd.Series, race: pd.Series, y: np.ndarray) -> float:
    res = minimize_scalar(lambda T: log_loss(y, _race_softmax(x, race, T)), bounds=(0.05, 200), method="bounded")
    return float(res.x)


def _stack_walk_forward(d: pd.DataFrame, cols: list[str], n_folds: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Time-ordered folds: fit logistic on earlier dates, predict later. Returns (pred, mask)."""
    dates = np.sort(d["race_date"].unique())
    edges = np.linspace(0, len(dates), n_folds + 1).astype(int)
    pred = np.full(len(d), np.nan)
    for k in range(1, n_folds):
        train_dates = dates[: edges[k]]; test_dates = dates[edges[k]: edges[k + 1]]
        tr = d["race_date"].isin(train_dates).values; te = d["race_date"].isin(test_dates).values
        if tr.sum() < 500 or te.sum() == 0:
            continue
        m = LogisticRegression(C=10.0, max_iter=2000).fit(d.loc[tr, cols].values, d.loc[tr, "won"].values)
        p = m.predict_proba(d.loc[te, cols].values)[:, 1]
        # renormalise within race
        s = pd.Series(p, index=d.index[te]).groupby(d.loc[te, "raceid"]).transform("sum")
        pred[te] = p / s.values
    return pred, ~np.isnan(pred)


def evaluate_rating_sets(pred: pd.DataFrame, rating_cols: list[str], p_col: str = "p_model", market_col: str = "p_market",
                         race_col: str = "raceid", y_col: str = "won", pos_col: str = "placing_numerical") -> pd.DataFrame:
    """Per-rating evaluation table (see module docstring)."""
    d = pred.copy()
    d["won"] = d[y_col].astype(float)
    d["ln_model"] = np.log(d[p_col].clip(1e-6)); d["ln_market"] = np.log(d[market_col].clip(1e-6))
    rows = []
    base_pred, base_mask = _stack_walk_forward(d, ["ln_model", "ln_market"])
    for c in rating_cols:
        r = d[c].astype(float)
        ok = r.notna()
        races_ok = d.loc[ok].groupby(race_col)["won"].transform("size") >= 2
        sub = d.loc[ok].loc[races_ok.values].copy()
        if len(sub) < 200:
            rows.append({"rating": c, "coverage": ok.mean(), "n": int(len(sub))}); continue
        sub["z"] = (sub[c] - sub.groupby(race_col)[c].transform("mean")) / sub.groupby(race_col)[c].transform("std").replace(0, np.nan)
        sub["z"] = sub["z"].fillna(0.0)
        y = sub["won"].values
        # standalone
        conc = concordance_index(sub[c].values, sub[pos_col].values, sub[race_col].values) if pos_col in sub.columns else np.nan
        pv = univariate_pvalues(sub[[c]], y, race_id=sub[race_col].values)
        dates = np.sort(sub["race_date"].unique()); half = dates[len(dates) // 2]
        first, second = sub["race_date"] < half, sub["race_date"] >= half
        T = fit_temperature(sub.loc[first, c], sub.loc[first, race_col], y[first.values])
        p_soft = _race_softmax(sub.loc[second, c], sub.loc[second, race_col], T)
        ll_soft = log_loss(y[second.values], p_soft); ll_model = log_loss(y[second.values], sub.loc[second, p_col]); ll_mkt = log_loss(y[second.values], sub.loc[second, market_col])
        # incremental (stack)
        sub_idx = sub.index
        with_pred, with_mask = _stack_walk_forward(sub.assign(**{"z_": sub["z"]}), ["ln_model", "ln_market", "z_"])
        bp = pd.Series(base_pred, index=d.index).reindex(sub_idx).values
        both = with_mask & ~np.isnan(bp)
        yy = y[both]
        ll_base, ll_with = log_loss(yy, bp[both]), log_loss(yy, with_pred[both])
        mk = sub.loc[both, market_col].values
        bss_base = 1 - brier(yy, bp[both]) / brier(yy, mk); bss_with = 1 - brier(yy, with_pred[both]) / brier(yy, mk)
        rows.append({"rating": c, "coverage": float(ok.mean()), "n": int(len(sub)), "races": int(sub[race_col].nunique()),
                     "concordance": conc, "corr_demeaned": float(pv["corr"].iloc[0]), "p_value": float(pv["p_value"].iloc[0]),
                     "softmax_T": T, "softmax_logloss": ll_soft, "model_logloss_same_rows": ll_model, "market_logloss_same_rows": ll_mkt,
                     "stack_logloss_base": ll_base, "stack_logloss_with": ll_with, "stack_gain_logloss": ll_base - ll_with,
                     "stack_bss_vs_market_base": bss_base, "stack_bss_vs_market_with": bss_with, "stack_gain_bss": bss_with - bss_base,
                     "n_stack_rows": int(both.sum())})
    return pd.DataFrame(rows)


def knockoff_screen(pred: pd.DataFrame, rating_cols: list[str], p_col: str = "p_model", market_col: str = "p_market",
                    race_col: str = "raceid", y_col: str = "won", fdr: float = 0.1) -> pd.DataFrame:
    """Model-X knockoffs over race z-scores of all ratings (rows with every rating present)."""
    d = pred.dropna(subset=rating_cols).copy()
    X = pd.DataFrame(index=d.index)
    for c in rating_cols:
        g = d.groupby(race_col)[c]
        X[c] = ((d[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)
    X["ln_model_z"] = np.log(d[p_col].clip(1e-6)); X["ln_model_z"] -= X["ln_model_z"].groupby(d[race_col]).transform("mean")
    X["ln_market_z"] = np.log(d[market_col].clip(1e-6)); X["ln_market_z"] -= X["ln_market_z"].groupby(d[race_col]).transform("mean")
    y = d[y_col].astype(float).values
    res = knockoff_filter(X, y, fdr=fdr, binary=True, lasso_alpha=1e-3)
    return pd.DataFrame({"feature": res["W"].index, "W": res["W"].values, "selected": res["selected"].values}).sort_values("W", ascending=False)
