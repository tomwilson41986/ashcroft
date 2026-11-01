"""The drift models: what the close will be, read from the opening book, fitted walk-forward by month.

Four forecasts of ``y = log(close / open)`` per selection, each a class with ``fit(train)`` and ``predict(test)``:

(a) ``NoChange``: close = open, the baseline every other model is scored against (the efficient-market null).
(b) ``BandShrinkage``: ordinary least squares of y on log open within each price band, the favourite-longshot
    unwind (the market's early prices sit too long on the outsiders and too short on the favourites, and the close
    corrects part of it).
(c) ``GradientBoosting``: LightGBM on the opening-book features (features.FEATURES, plus the model's forecast in
    the "with model" arm), fixed hyper-parameters, no early stopping on the scored month.
(d) ``ConditionalLogit``: a coherent book. The closing shares ``q_i = (1/close_i) / sum_j 1/close_j`` are soft labels
    and ``q_hat = softmax(X beta)`` within the market (model/race_book.ClosingModel's form, fitted by cross-entropy
    with the same helpers), X from the open's log share and the model's; the close's overround is a log-linear
    function of the open's fitted on the training markets, so the predicted closes form a book that sums to what
    closes do. ``renormalise_book`` applies the same coherence to any model's per-selection forecast (the
    ``GradientBoosting + book`` arm).

``walk_forward`` fits every model on the months before each scored month and scores that month only (never a
training row); ``refit_every`` lets a long panel refit less often, the scored month always after the fit's last.
``shuffle_test`` refits with the training target permuted: a model that still scores above the baseline is
reading the answer from somewhere (a leak).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from model.race_book import group_log_softmax, normalise_within

from .features import BAND_LABELS, MODEL_FEATURES, feature_columns, market_codes, price_band

_EPS = 1e-12


# ---------------------------------------------------------------------------
# the close's overround from the open's, and the book coherence
# ---------------------------------------------------------------------------

@dataclass
class BookModel:
    """log(close overround) = a + b log(open overround), fitted by least squares on the training markets."""

    a: float = 0.0
    b: float = 1.0

    @classmethod
    def fit(cls, open_over: np.ndarray, close_over: np.ndarray) -> "BookModel":
        x, y = np.log(np.asarray(open_over, float)), np.log(np.asarray(close_over, float))
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 10 or np.std(x[ok]) < 1e-9:
            return cls(a=float(np.mean(y[ok])) if ok.any() else 0.0, b=0.0)
        b, a = np.polyfit(x[ok], y[ok], 1)
        return cls(a=float(a), b=float(b))

    def predict(self, open_over) -> np.ndarray:
        return np.exp(self.a + self.b * np.log(np.asarray(open_over, float)))


def market_overrounds(d: pd.DataFrame, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each market's open and close overround, one value a market (in code order)."""
    o = np.bincount(g, weights=1.0 / d["open_price"].to_numpy(float))
    c = np.bincount(g, weights=1.0 / d["close_price"].to_numpy(float))
    return o, c


def renormalise_book(yhat: np.ndarray, open_price: np.ndarray, g: np.ndarray, book: BookModel) -> np.ndarray:
    """A per-selection forecast of log(close/open) made coherent: the implied closing probabilities are scaled so
    each market's sum is the overround the book model predicts for it. Returns the coherent log(close/open)."""
    open_price = np.asarray(open_price, float)
    p = np.exp(-np.asarray(yhat, float)) / open_price
    share = normalise_within(p, g)
    open_over = np.bincount(g, weights=1.0 / open_price)
    close_hat = 1.0 / (share * book.predict(open_over)[g])
    return np.log(close_hat / open_price)


# ---------------------------------------------------------------------------
# (a) no change
# ---------------------------------------------------------------------------

class NoChange:
    name = "no_change"

    def fit(self, train: pd.DataFrame) -> "NoChange":
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(test))


# ---------------------------------------------------------------------------
# (b) shrinkage by band
# ---------------------------------------------------------------------------

@dataclass
class BandShrinkage:
    """y = a_b + c_b (log open - m_b) within each price band of the open; the pooled line where a band is thin."""

    name: str = "band_ols"
    min_rows: int = 200
    coef: dict = field(default_factory=dict)
    pooled: tuple = (0.0, 0.0, 0.0)

    @staticmethod
    def _line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
        m = float(x.mean())
        xc = x - m
        var = float(xc @ xc)
        c = float(xc @ (y - y.mean())) / var if var > 1e-12 else 0.0
        return float(y.mean()), c, m

    def fit(self, train: pd.DataFrame) -> "BandShrinkage":
        x, y = train["log_open"].to_numpy(float), train["y"].to_numpy(float)
        self.pooled = self._line(x, y)
        band = price_band(train["open_price"].to_numpy()).astype(str).to_numpy()
        self.coef = {}
        for b in BAND_LABELS:
            idx = band == b
            if idx.sum() >= self.min_rows:
                self.coef[b] = self._line(x[idx], y[idx])
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        x = test["log_open"].to_numpy(float)
        band = price_band(test["open_price"].to_numpy()).astype(str).to_numpy()
        out = np.empty(len(test))
        for b in np.unique(band):
            a, c, m = self.coef.get(b, self.pooled)
            idx = band == b
            out[idx] = a + c * (x[idx] - m)
        return out


# ---------------------------------------------------------------------------
# (c) gradient boosting
# ---------------------------------------------------------------------------

@dataclass
class GradientBoosting:
    """LightGBM regression of y on the opening-book features; fixed settings chosen in advance (no tuning on the
    scored months). ``with_model`` adds the model's forecast features where the panel holds them."""

    with_model: bool = False
    name: str = "gbm"
    n_estimators: int = 400
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_child_samples: int = 100
    n_jobs: int = 4
    seed: int = 0
    columns: list = field(default_factory=list)
    categories: dict = field(default_factory=dict)
    model: object = None

    def __post_init__(self):
        if self.with_model and self.name == "gbm":
            self.name = "gbm_model"

    def _frame(self, d: pd.DataFrame) -> pd.DataFrame:
        X = d[self.columns].copy()
        for c in self.columns:
            if c in self.categories:                      # the categories the fit saw, in the same order
                X[c] = pd.Categorical(X[c].astype(str), categories=self.categories[c])
            else:
                X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
        return X

    def fit(self, train: pd.DataFrame) -> "GradientBoosting":
        import lightgbm as lgb
        self.columns = feature_columns(train, self.with_model)
        self.categories = {c: sorted(train[c].astype(str).unique()) for c in self.columns
                           if str(train[c].dtype) == "category"}
        X = self._frame(train)
        self.model = lgb.LGBMRegressor(objective="regression", n_estimators=self.n_estimators,
                                       learning_rate=self.learning_rate, num_leaves=self.num_leaves,
                                       min_child_samples=self.min_child_samples, subsample=0.8, subsample_freq=1,
                                       colsample_bytree=0.8, reg_lambda=1.0, random_state=self.seed,
                                       n_jobs=self.n_jobs, verbose=-1)
        self.model.fit(X, train["y"].to_numpy(float))
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._frame(test))

    def importance(self) -> dict:
        if self.model is None:
            return {}
        return dict(zip(self.columns, (float(v) for v in self.model.booster_.feature_importance("gain"))))


# ---------------------------------------------------------------------------
# (d) the coherent book: a conditional logit on the closing shares
# ---------------------------------------------------------------------------

@dataclass
class ConditionalLogit:
    """q_hat = softmax(X beta) within the market, X = [log open share, its square, log open share x log field,
    (log model share, log model share x log open matched)], fitted to the closing shares by cross-entropy with an
    L2 penalty; the close's overround from BookModel. A book of 1 a market, as race_book.ClosingModel."""

    with_model: bool = False
    name: str = "clogit"
    l2: float = 1e-4
    beta: np.ndarray = field(default_factory=lambda: np.zeros(0))
    book: BookModel = field(default_factory=BookModel)

    def __post_init__(self):
        if self.with_model and self.name == "clogit":
            self.name = "clogit_model"

    def _design(self, d: pd.DataFrame, g: np.ndarray) -> np.ndarray:
        lq = np.log(normalise_within(1.0 / d["open_price"].to_numpy(float), g))
        n = np.log(np.bincount(g)[g].astype(float))
        cols = [lq, lq * lq, lq * n]
        if self.with_model and "log_model_vs_open" in d.columns:
            move = np.nan_to_num(d["log_model_vs_open"].to_numpy(float), nan=0.0)   # log(open / model): 0 = none
            lm = lq - move                                                            # the model's log share
            vol = np.nan_to_num(d["log_open_matched"].to_numpy(float), nan=0.0)
            cols += [lm, lm * vol]
        return np.column_stack(cols)

    def fit(self, train: pd.DataFrame) -> "ConditionalLogit":
        g = market_codes(train)
        q = normalise_within(1.0 / train["close_price"].to_numpy(float), g)
        X = self._design(train, g)
        n_markets = g.max() + 1

        def loss(beta):
            lqh = group_log_softmax(X @ beta, g)
            val = -float(np.sum(q * lqh)) / n_markets + 0.5 * self.l2 * float(beta @ beta)
            grad = -((q - np.exp(lqh))[:, None] * X).sum(axis=0) / n_markets + self.l2 * beta
            return val, grad

        start = np.zeros(X.shape[1])
        start[0] = 1.0                                                   # the open's share, unchanged
        res = minimize(loss, start, jac=True, method="L-BFGS-B", options={"maxiter": 200})
        self.beta = res.x
        o, c = market_overrounds(train, g)
        self.book = BookModel.fit(o, c)
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        g = market_codes(test)
        q_hat = np.exp(group_log_softmax(self._design(test, g) @ self.beta, g))
        open_price = test["open_price"].to_numpy(float)
        close_hat = 1.0 / (q_hat * self.book.predict(np.bincount(g, weights=1.0 / open_price))[g])
        return np.log(close_hat / open_price)


class BookCoherent:
    """Any per-selection model made coherent by market (``renormalise_book``)."""

    def __init__(self, inner, name: str | None = None):
        self.inner = inner
        self.name = name or f"{inner.name}_book"
        self.book = BookModel()

    def fit(self, train: pd.DataFrame):
        self.inner.fit(train)
        o, c = market_overrounds(train, market_codes(train))
        self.book = BookModel.fit(o, c)
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return renormalise_book(self.inner.predict(test), test["open_price"].to_numpy(float), market_codes(test),
                                self.book)


def default_models(with_model: bool = False, n_jobs: int = 4) -> list:
    """The arms a run scores: (a) to (d), and the boosted model made coherent."""
    models = [NoChange(), BandShrinkage(), GradientBoosting(n_jobs=n_jobs), ConditionalLogit()]
    models.append(BookCoherent(GradientBoosting(n_jobs=n_jobs)))
    if with_model:
        models += [GradientBoosting(with_model=True, n_jobs=n_jobs), ConditionalLogit(with_model=True)]
    return models


# ---------------------------------------------------------------------------
# walk-forward
# ---------------------------------------------------------------------------

def months_of(d: pd.DataFrame) -> list[str]:
    return sorted(d["month"].dropna().unique())


def walk_forward(d: pd.DataFrame, models: list, min_train_months: int = 12, refit_every: int = 1,
                 score_from: str | None = None, score_to: str | None = None, shuffle_seed: int | None = None,
                 log=None) -> pd.DataFrame:
    """Every model fitted on the months before each scored month and scored on that month only.

    ``refit_every``: refit after this many scored months (1: every month); between refits the last fit scores the
    next months, still fitted only on months before them. ``score_from``/``score_to`` bound the scored months
    (YYYY-MM, inclusive). ``shuffle_seed``: the training target permuted before every fit (the leak test).
    Returns the scored rows with a ``pred_<name>`` column per model and ``fit_month`` (the last month the fit saw).
    """
    months = months_of(d)
    out, fitted, last_fit_at = [], None, None
    for i, m in enumerate(months):
        if i < min_train_months or (score_from and m < score_from) or (score_to and m > score_to):
            continue
        if fitted is None or (i - last_fit_at) >= refit_every:
            train = d[d["month"] < m]
            if shuffle_seed is not None:
                rng = np.random.default_rng(shuffle_seed + i)
                train = train.assign(y=rng.permutation(train["y"].to_numpy()),
                                     y_share=rng.permutation(train["y_share"].to_numpy()))
            fitted = [mod.fit(train) for mod in models]
            last_fit_at, fit_month = i, months[i - 1]
            if log:
                log(f"  fitted on {len(train):,} rows to {fit_month}; scoring from {m}")
        test = d[d["month"] == m].copy()
        test["fit_month"] = fit_month
        for mod in fitted:
            test[f"pred_{mod.name}"] = mod.predict(test)
        out.append(test)
    return pd.concat(out, ignore_index=True) if out else d.iloc[:0].copy()


def shuffle_test(d: pd.DataFrame, models: list, seed: int = 0, **kw) -> pd.DataFrame:
    """walk_forward with the training targets permuted: no model should beat the baseline on these."""
    return walk_forward(d, models, shuffle_seed=seed, **kw)
