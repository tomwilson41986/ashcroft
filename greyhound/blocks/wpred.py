"""A predicted weigh-in (the owner, 4 Nov 2026: "define a 'predicted weight' for each dog today based on historic
patterns ... then predict it for today's races when we don't have it yet"; the inputs: "weight last run, historic weight
patterns, recency, form"). No pre-race weigh-in source reaches us (greyhounds.today has returned no cards since 7 Oct;
GBGB fills the weight in with the result), so the day's weight is forecast from the dog's earlier days only.

With the raw weight in the cache (``weight_kg``):

- **weight last run**: the dog's weight at its last three runs;
- **historic weight patterns**: its running mean, a recency-weighted mean, spread, heaviest and lightest, the trend over
  its last three runs and its last change;
- **recency**: the days since its last run and the gap before that, its runs so far, the month;
- **form**: its last finishing position, its mean position over the last three, whether it won last time;

a LightGBM model of the day's change from the last run's weight, refitted each year on the years before only
(walk-forward), gives the predicted weight and from it the four weigh-in features the model reads: the change from the
last run, the weight against the dog's mean, against its heaviest and against its lightest.

Without it (an older cache): a linear model of the day's ``weight_change`` on the last run's ``weight_change`` and
``weight_vs_mean``, the gap and the month, refitted each month on the months before.

Earlier days only: a dog's own same-day rows are never its "previous run". Each year's out-of-sample R2 is logged (how
much of the day's weight change history foresees): the weigh-in helps the model through the part history cannot
foresee, so a forecast can add the predictable part only.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

PREFIX = "wp_"
log = logging.getLogger(__name__)
NAMES_FULL = ("pred_weight", "pred_change", "pred_vs_mean", "pred_vs_max", "pred_vs_min", "prev_weight", "gap_days")
NAMES_REL = ("pred_change", "pred_vs_mean", "prev_vs_mean", "gap_days")
LGB = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 200,
       "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1,
       "seed": 0, "num_threads": 4}
ROUNDS = 300
MIN_FIT = 50_000                                  # a year's fit needs this many earlier runs


def _days(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """A dog's rows folded to one a day (its first race of the day), sorted by dog and time; and each original row's
    day-row label (so every same-day row reads the same earlier-days features)."""
    t = pd.to_datetime(df.t)
    d = pd.DataFrame({"dog": df.dog_id.to_numpy(), "t": t.to_numpy(), "day": t.dt.normalize().to_numpy(),
                      **{c: pd.to_numeric(df[c], errors="coerce").to_numpy(float) for c in cols}}, index=df.index)
    d = d.sort_values(["dog", "t"], kind="stable")
    first = ~d.duplicated(["dog", "day"], keep="first")
    days = d[first].copy()
    key = pd.MultiIndex.from_frame(d[["dog", "day"]])
    label = pd.Series(pd.MultiIndex.from_frame(days[["dog", "day"]]).get_indexer(key), index=d.index)
    return days.reset_index(drop=True), label


def _r2(y, p) -> float:
    s = np.isfinite(y) & np.isfinite(p)
    return float(1 - np.sum((y[s] - p[s]) ** 2) / np.sum((y[s] - y[s].mean()) ** 2)) if s.sum() > 100 else np.nan


def _full(df: pd.DataFrame) -> list[str]:
    """The day's weight from the last runs' weights, the dog's weight history, recency and form (LightGBM, yearly)."""
    import lightgbm as lgb
    cols = ["weight_kg"] + [c for c in ("position", "won") if c in df]
    days, label = _days(df, cols)
    g = days.groupby("dog", sort=False)
    w = days.weight_kg
    prev = g.weight_kg.shift(1)
    dog = days.dog
    cnt = prev.notna().astype(float).groupby(dog).cumsum()
    csum = prev.fillna(0.0).groupby(dog).cumsum()
    csq = (prev ** 2).fillna(0.0).groupby(dog).cumsum()
    mean = csum / cnt.where(cnt > 0)
    sd = np.sqrt(((csq - csum ** 2 / cnt.where(cnt > 0)) / (cnt - 1).where(cnt > 1)).clip(lower=0))
    X = pd.DataFrame({
        "w1": prev, "w2": g.weight_kg.shift(2), "w3": g.weight_kg.shift(3), "mean": mean,
        "ewm": prev.groupby(dog).ewm(alpha=0.3, ignore_na=True).mean().reset_index(level=0, drop=True),
        "sd": sd,
        "max": prev.groupby(dog).cummax().groupby(dog).ffill(),
        "min": prev.groupby(dog).cummin().groupby(dog).ffill(),
        "runs": g.cumcount().astype(float),
        "gap": (days.t - g.t.shift(1)).dt.days.astype(float),
        "gap2": (g.t.shift(1) - g.t.shift(2)).dt.days.astype(float),
        "month": pd.to_datetime(days.t).dt.month.astype(float)})
    X["trend"] = (X.w1 - X.w3) / 2.0
    X["last_change"] = X.w1 - X.w2
    X["w1_vs_mean"] = X.w1 - X["mean"]
    X["w1_vs_ewm"] = X.w1 - X["ewm"]
    if "position" in days:
        pos = g.position.shift(1)
        X["pos1"] = pos
        X["pos3"] = pos.groupby(dog).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    if "won" in days:
        X["won1"] = g.won.shift(1)
    y = (w - prev).to_numpy()                                    # the day's change from the last run's weight
    year = pd.to_datetime(days.t).dt.year.to_numpy()
    pred = np.full(len(days), np.nan)
    fit_ok = np.isfinite(y) & X.w1.notna().to_numpy()
    for yr in sorted(set(year)):
        tr = fit_ok & (year < yr)
        te = (year == yr) & X.w1.notna().to_numpy()
        if tr.sum() < MIN_FIT or not te.any():
            continue
        m = lgb.train(LGB, lgb.Dataset(X[tr], y[tr]), ROUNDS)
        pred[te] = m.predict(X[te])
        log.info("wpred %s: out-of-sample R2 of the weight change %.3f, of the weight %.4f (%d runs; fitted on %d)",
                 yr, _r2(y[te], pred[te]), _r2(w.to_numpy()[te], X.w1.to_numpy()[te] + pred[te]), int(te.sum()),
                 int(tr.sum()))
    pw = X.w1.to_numpy() + pred
    out = {"pred_weight": pw, "pred_change": pred, "pred_vs_mean": pw - X["mean"].to_numpy(),
           "pred_vs_max": pw - X["max"].to_numpy(), "pred_vs_min": pw - X["min"].to_numpy(),
           "prev_weight": X.w1.to_numpy(), "gap_days": X.gap.to_numpy()}
    names = []
    for n in NAMES_FULL:
        col = f"{PREFIX}{n}"
        v = np.asarray(out[n], float)[label.to_numpy()]
        df[col] = pd.Series(v, index=label.index).reindex(df.index).astype(np.float32)
        names.append(col)
    return names


def _design(prev_wvm, prev_wc, gap, runs, month) -> np.ndarray:
    lg = np.log1p(np.clip(gap, 0, 365))
    return np.column_stack([np.ones(len(gap)), prev_wvm, prev_wc, lg, lg * prev_wvm, np.minimum(runs, 20) / 20.0,
                            np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)])


def _relative(df: pd.DataFrame) -> list[str]:
    """The fallback without raw weights: the last run's relative weigh-in features, a linear model, monthly."""
    names = [f"{PREFIX}{n}" for n in NAMES_REL]
    days, label = _days(df, ["weight_change", "weight_vs_mean"])
    g = days.groupby("dog", sort=False)
    prev_wvm, prev_wc = g.weight_vs_mean.shift(1).to_numpy(), g.weight_change.shift(1).to_numpy()
    gap = (days.t - g.t.shift(1)).dt.days.to_numpy(float)
    runs = g.cumcount().to_numpy(float)
    month = pd.to_datetime(days.t).dt.month.to_numpy(float)
    X = _design(prev_wvm, prev_wc, gap, runs, month)
    y = days.weight_change.to_numpy()
    ok_x = np.isfinite(X).all(1)
    ok = ok_x & np.isfinite(y)
    period = pd.to_datetime(days.t).dt.to_period("M").to_numpy()
    pred = np.full(len(days), np.nan)
    k = X.shape[1]
    xtx, xty = np.zeros((k, k)), np.zeros(k)
    for p in sorted(set(period)):                       # walk-forward: each month from the months before it only
        m = period == p
        if xtx[0, 0] >= 500:
            beta = np.linalg.solve(xtx + 1e-6 * np.eye(k), xty)
            pm = m & ok_x
            pred[pm] = X[pm] @ beta
        f = m & ok
        xtx += X[f].T @ X[f]
        xty += X[f].T @ y[f]
    pred_vs_mean = pred + prev_wvm * np.where(runs > 0, runs / (runs + 1.0), 0.0)
    years = pd.to_datetime(days.t).dt.year.to_numpy()
    for yr in sorted(set(years)):
        s = years == yr
        if np.isfinite(pred[s]).sum() > 100:
            log.info("wpred %s (relative): out-of-sample R2 of the weight change %.3f (%d runs)", yr,
                     _r2(y[s], pred[s]), int(s.sum()))
    out = {"pred_change": pred, "pred_vs_mean": pred_vs_mean, "prev_vs_mean": prev_wvm, "gap_days": gap}
    for n in NAMES_REL:
        v = np.asarray(out[n], float)[label.to_numpy()]
        df[f"{PREFIX}{n}"] = pd.Series(v, index=label.index).reindex(df.index).astype(np.float32)
    return names


def build(df: pd.DataFrame) -> list[str]:
    if "weight_kg" in df and pd.to_numeric(df.weight_kg, errors="coerce").notna().any():
        return _full(df)
    if "weight_change" in df and "weight_vs_mean" in df:
        return _relative(df)
    names = [f"{PREFIX}{n}" for n in NAMES_REL]
    for n in names:
        df[n] = np.float32(np.nan)
    return names
