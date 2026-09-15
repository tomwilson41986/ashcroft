"""
Cross-classified ridge effects ("RAPM" for racing).

Why: a jockey who rides almost exclusively for one yard, or a horse that
never leaves one trainer, cannot be credited separately by one-at-a-time
form statistics — the entities are collinear. Regularised adjusted
plus-minus (Sill 2010; Macdonald 2012) solves exactly this: fit ALL
entity effects jointly with a ridge (zero-mean Gaussian) prior so that
correlated predictors are shrunk *toward each other* rather than one of
them absorbing all the credit.

    perf_run = mu + horse_effect + jockey_effect + trainer_effect
               + sire_effect + numeric covariates + noise

with a separate shrinkage strength per entity block. The block
strengths can be set by hand or estimated empirically
(lambda_b = sigma_e^2 / sigma_b^2, a few fixed-point iterations — the
Henderson mixed-model view of ridge).

This is also the single implementation of the "horse-heterogeneity"
route (Cox frailty = panel random effect = ridge cross-classified
effect); build one, not three.

Lag safety: fit on strictly earlier dates than the rows you score.
``walk_forward_effects`` does this for you.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import cg, spsolve

log = logging.getLogger(__name__)


class CrossClassifiedRidge:
    """Joint ridge estimation of entity effects with per-block shrinkage.

    Parameters
    ----------
    entity_cols : list[str]
        Categorical columns, one effect per level (e.g. horse_name,
        jockey_name, trainer, stallion).
    lambdas : float | dict[str, float]
        Ridge strength per entity block (higher = more shrinkage). A
        float applies to every block.
    numeric_cols : list[str] | None
        Standardised numeric covariates (unpenalised-ish: lambda 1e-6).
    prior_col : str | None
        Column holding an external prior mean of the target (e.g. an
        official rating). The model then explains target - prior, so
        effects are shrunk toward the external rating rather than zero.
    min_count : int
        Levels with fewer training rows than this are pooled into an
        "__other__" level (still estimated, heavily shrunk by data volume).
    """

    def __init__(self, entity_cols, lambdas=10.0, numeric_cols=None, prior_col=None, min_count=1):
        self.entity_cols = list(entity_cols)
        self.numeric_cols = list(numeric_cols or [])
        self.prior_col = prior_col
        self.min_count = min_count
        if isinstance(lambdas, dict):
            self.lambdas = {c: float(lambdas.get(c, 10.0)) for c in self.entity_cols}
        else:
            self.lambdas = {c: float(lambdas) for c in self.entity_cols}
        self.levels_: dict[str, pd.Index] = {}
        self.effects_: dict[str, pd.Series] = {}
        self.coef_numeric_: pd.Series | None = None
        self.intercept_: float = 0.0
        self.num_mean_: pd.Series | None = None
        self.num_std_: pd.Series | None = None
        self.sigma2_e_: float = np.nan
        self.sigma2_b_: dict[str, float] = {}

    # ------------------------------------------------------------------
    def _design(self, df: pd.DataFrame, fit: bool):
        blocks, sizes = [], []
        n = len(df)
        for c in self.entity_cols:
            vals = df[c].fillna("__missing__").astype(str)
            if fit:
                counts = vals.value_counts()
                keep = counts[counts >= self.min_count].index
                vals = vals.where(vals.isin(keep), "__other__")
                levels = pd.Index(sorted(vals.unique()))
                self.levels_[c] = levels
            else:
                levels = self.levels_[c]
                vals = vals.where(vals.isin(levels), "__other__")
            codes = levels.get_indexer(vals)
            ok = codes >= 0
            rows = np.arange(n)[ok]
            blocks.append(sp.csr_matrix((np.ones(ok.sum()), (rows, codes[ok])), shape=(n, len(levels))))
            sizes.append(len(levels))
        if self.numeric_cols:
            X = df[self.numeric_cols].astype(float)
            if fit:
                self.num_mean_, self.num_std_ = X.mean(), X.std().replace(0, 1.0)
            Xn = ((X - self.num_mean_) / self.num_std_).fillna(0.0).values
            blocks.append(sp.csr_matrix(Xn))
            sizes.append(len(self.numeric_cols))
        return sp.hstack(blocks).tocsr(), sizes

    def _penalty(self, sizes):
        diag = []
        for c, s in zip(self.entity_cols, sizes[: len(self.entity_cols)]):
            diag.append(np.full(s, self.lambdas[c]))
        if self.numeric_cols:
            diag.append(np.full(len(self.numeric_cols), 1e-6))
        return np.concatenate(diag)

    def _target(self, df, target):
        y = pd.to_numeric(df[target], errors="coerce").astype(float)
        if self.prior_col:
            y = y - pd.to_numeric(df[self.prior_col], errors="coerce").astype(float)
        return y

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, target: str, sample_weight=None, eb_iterations: int = 0):
        """Fit by solving the (sparse) ridge normal equations.

        eb_iterations > 0 re-estimates each block's lambda empirically
        (sigma_e^2 / sigma_b^2) that many times before the final solve.
        """
        y = self._target(df, target)
        mask = y.notna().values
        d = df.loc[mask]
        y = y.loc[mask].values
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)[mask]
        self.intercept_ = float(np.average(y, weights=w))
        yc = y - self.intercept_

        X, sizes = self._design(d, fit=True)
        Wd = sp.diags(w)
        XtWX = (X.T @ Wd @ X).tocsc()
        XtWy = X.T @ (w * yc)

        beta = None
        for it in range(eb_iterations + 1):
            pen = self._penalty(sizes)
            A = (XtWX + sp.diags(pen)).tocsc()
            beta = self._solve(A, XtWy)
            resid = yc - X @ beta
            self.sigma2_e_ = float(np.sum(w * resid**2) / max(len(y) - 1, 1))
            if it < eb_iterations:
                # Update lambda_b = sigma_e^2 / sigma_b^2 using a diagonal
                # approximation to the posterior variance of each effect.
                offset = 0
                diagA = np.asarray(XtWX.diagonal()).ravel()
                for c, s in zip(self.entity_cols, sizes):
                    b = beta[offset: offset + s]
                    post_var = self.sigma2_e_ / (diagA[offset: offset + s] + self.lambdas[c])
                    sigma2_b = float(np.mean(b**2 + post_var))
                    self.sigma2_b_[c] = sigma2_b
                    self.lambdas[c] = float(np.clip(self.sigma2_e_ / max(sigma2_b, 1e-9), 1e-3, 1e6))
                    offset += s

        offset = 0
        for c, s in zip(self.entity_cols, sizes):
            self.effects_[c] = pd.Series(beta[offset: offset + s], index=self.levels_[c], name=f"effect_{c}")
            offset += s
        if self.numeric_cols:
            self.coef_numeric_ = pd.Series(beta[offset: offset + len(self.numeric_cols)], index=self.numeric_cols)
        return self

    @staticmethod
    def _solve(A, b):
        try:
            x, info = cg(A, b, rtol=1e-8, maxiter=5000)
            if info != 0:
                x = spsolve(A, b)
        except TypeError:  # older scipy: tol instead of rtol
            x, info = cg(A, b, tol=1e-8, maxiter=5000)
            if info != 0:
                x = spsolve(A, b)
        return np.asarray(x).ravel()

    # ------------------------------------------------------------------
    def effect_for(self, entity: str, keys) -> pd.Series:
        """Look up effects for a vector of entity keys (NaN if unseen)."""
        eff = self.effects_[entity]
        k = pd.Series(keys).fillna("__missing__").astype(str)
        return pd.Series(eff.reindex(k.values).values, index=k.index)

    def predict(self, df: pd.DataFrame, include_prior: bool = True) -> np.ndarray:
        X, _ = self._design(df, fit=False)
        beta = np.concatenate([self.effects_[c].values for c in self.entity_cols]
                              + ([self.coef_numeric_.values] if self.numeric_cols else []))
        pred = self.intercept_ + X @ beta
        if include_prior and self.prior_col:
            pred = pred + pd.to_numeric(df[self.prior_col], errors="coerce").fillna(0).values
        return np.asarray(pred).ravel()

    def summary(self, entity: str, top: int = 15, min_effect=None) -> pd.DataFrame:
        eff = self.effects_[entity].sort_values(ascending=False)
        eff = eff[~eff.index.isin(["__other__", "__missing__"])]
        return pd.concat([eff.head(top), eff.tail(top)]).to_frame()


# ---------------------------------------------------------------------------
# Lag-safe scoring
# ---------------------------------------------------------------------------

def walk_forward_effects(
    df: pd.DataFrame,
    target: str,
    entity_cols,
    date_col: str = "race_date",
    freq: str = "MS",
    min_train_days: int = 365,
    lambdas=10.0,
    numeric_cols=None,
    prior_col=None,
    eb_iterations: int = 2,
    prefix: str = "rapm",
) -> pd.DataFrame:
    """Add lag-safe cross-classified effect columns to ``df``.

    For each period (default: calendar month) the model is fitted on all
    rows strictly before the period start and used to score the rows in
    the period. Adds ``{prefix}_{entity}`` per entity and ``{prefix}_pred``.
    Rows before the first fit window are left NaN.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    for c in entity_cols:
        out[f"{prefix}_{c}"] = np.nan
    out[f"{prefix}_pred"] = np.nan

    start = out[date_col].min() + pd.Timedelta(days=min_train_days)
    periods = pd.date_range(start.to_period(freq[0]).to_timestamp(), out[date_col].max(), freq=freq)
    for i, p0 in enumerate(periods):
        p1 = periods[i + 1] if i + 1 < len(periods) else out[date_col].max() + pd.Timedelta(days=1)
        train = out[out[date_col] < p0]
        test_mask = (out[date_col] >= p0) & (out[date_col] < p1)
        if train[target].notna().sum() < 200 or test_mask.sum() == 0:
            continue
        m = CrossClassifiedRidge(entity_cols, lambdas=lambdas, numeric_cols=numeric_cols, prior_col=prior_col)
        m.fit(train, target, eb_iterations=eb_iterations)
        test = out.loc[test_mask]
        for c in entity_cols:
            out.loc[test_mask, f"{prefix}_{c}"] = m.effect_for(c, test[c]).values
        out.loc[test_mask, f"{prefix}_pred"] = m.predict(test)
        log.info("walk_forward_effects: %s -> %s  train=%d test=%d lambdas=%s",
                 p0.date(), p1.date(), len(train), int(test_mask.sum()),
                 {k: round(v, 2) for k, v in m.lambdas.items()})
    return out
