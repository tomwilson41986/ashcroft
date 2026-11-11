"""Next-move models, from the simplest to the richest, each scored against the "no information" baseline.

Every model is fitted on earlier days and predicts a later day (``walk_forward``). Each returns, for every test row,
a probability for each class of the target (-1 down, 0 none, +1 up; ``y_next`` has no 0 class).

- ``baseline``  the training days' class shares: the bar every model has to clear
- ``qi_logit``  a multinomial logistic on queue imbalance at the touch (the published benchmark for limit-order books)
- ``markov``    a Markov chain on discrete book states (queue-imbalance bucket x spread in ticks x seconds to the off):
                the training days' class shares out of each state, with add-one smoothing toward the baseline
- ``kalman``    a local-level Kalman filter on the log-odds of the microprice, run along each runner's rows; the gap
                between the filtered "fair" log-odds and the mid's, in spreads, fed to a logistic (the filter's noise
                ratio chosen on the training days)
- ``gbm``       LightGBM on every feature in features.FEATURES

Scoring: the log loss in millinats per row, the gain over the baseline on the same rows (positive is better), and
the accuracy on the rows that moved. A day is the unit of the interval (rows within a day are not independent).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.micro.features import FEATURES

MODELS = ("baseline", "qi_logit", "markov", "kalman", "gbm")
EPS = 1e-6


def _classes(y: pd.Series) -> list:
    return sorted(int(c) for c in pd.unique(y.dropna()))


def _logloss(p: np.ndarray, y: np.ndarray, classes: list) -> np.ndarray:
    idx = np.searchsorted(classes, y.astype(int))
    return -np.log(np.clip(p[np.arange(len(y)), idx], EPS, 1.0))


def baseline(tr: pd.DataFrame, te: pd.DataFrame, target: str, classes: list) -> np.ndarray:
    share = np.array([(tr[target] == c).mean() for c in classes])
    share = np.clip(share, EPS, None)
    return np.tile(share / share.sum(), (len(te), 1))


def _logit(tr_x, tr_y, te_x, classes):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(max_iter=500, C=1.0)
    m.fit(tr_x, tr_y.astype(int))
    p = np.zeros((len(te_x), len(classes)))
    for j, c in enumerate(m.classes_):
        p[:, classes.index(int(c))] = m.predict_proba(te_x)[:, j]
    return p


def qi_logit(tr, te, target, classes):
    return _logit(tr[["qi"]].to_numpy(), tr[target], te[["qi"]].to_numpy(), classes)


def _state(df: pd.DataFrame) -> pd.Series:
    qb = np.digitize(df.qi.to_numpy(float), [-0.6, -0.2, 0.2, 0.6])
    sb = np.minimum(df.spread_ticks.to_numpy(float).round().astype(int), 3)
    tb = np.digitize(df.to_off.to_numpy(float), [30, 60, 120])
    return pd.Series(qb * 100 + sb * 10 + tb, index=df.index)


def markov(tr, te, target, classes):
    base = baseline(tr, te.iloc[:1], target, classes)[0]
    s_tr, s_te = _state(tr), _state(te)
    counts = pd.crosstab(s_tr, tr[target].astype(int)).reindex(columns=classes, fill_value=0)
    probs = (counts.to_numpy(float) + len(classes) * base) / (counts.sum(axis=1).to_numpy(float)[:, None] + len(classes))
    table = pd.DataFrame(probs, index=counts.index, columns=classes)
    out = np.array(table.reindex(s_te.to_numpy()).to_numpy(), dtype=float)
    miss = np.isnan(out).any(axis=1)
    out[miss] = base
    return out


def kalman_gap(df: pd.DataFrame, q_over_r: float) -> np.ndarray:
    """Along each runner's rows: a local-level filter on logit(microprice), observation noise 1, state noise
    ``q_over_r``; the filtered level after this row's microprice (known at the row's time) minus the logit of the mid,
    over the spread in logit units. Rows must be ordered by runner, then time."""
    micro = (df.pm + df.micro_dev * (1.0 / df.bb - 1.0 / df.bl)).clip(1e-4, 1 - 1e-4).to_numpy(float)
    z = np.log(micro / (1 - micro))
    pm = df.pm.clip(1e-4, 1 - 1e-4).to_numpy(float)
    lm = np.log(pm / (1 - pm))
    half = (np.log((1 / df.bb) / (1 - 1 / df.bb)) - np.log((1 / df.bl) / (1 - 1 / df.bl))).abs().to_numpy(float)
    half = np.where(half > 0, half, 1.0)
    gap = np.zeros(len(df))
    keys = (df.market.astype(str) + ":" + df.sid.astype(str)).to_numpy()
    x = p = None
    prev = None
    for i in range(len(df)):
        if keys[i] != prev:
            x, p, prev = z[i], 1.0, keys[i]
        p = p + q_over_r                      # predict
        k = p / (p + 1.0)
        x = x + k * (z[i] - x)                # update with this row's microprice
        p = (1 - k) * p
        gap[i] = (x - lm[i]) / half[i]
    return gap


def kalman(tr, te, target, classes, ratios=(0.01, 0.1, 1.0)):
    best, best_ll = ratios[0], np.inf
    last = tr.day.max()                       # the ratio is chosen on the last training day, fitted on the others
    fit = (tr.day < last).to_numpy() if tr.day.nunique() > 1 else np.arange(len(tr)) < int(len(tr) * 0.8)
    for q in ratios:
        g = kalman_gap(tr, q)
        if tr[target][fit].nunique() < 2:
            break
        p = _logit(g[fit][:, None], tr[target][fit], g[~fit][:, None], classes)
        ll = _logloss(p, tr[target][~fit].to_numpy(), classes).mean()
        if ll < best_ll:
            best, best_ll = q, ll
    g_tr, g_te = kalman_gap(tr, best), kalman_gap(te, best)
    return _logit(g_tr[:, None], tr[target], g_te[:, None], classes)


def gbm(tr, te, target, classes):
    import lightgbm as lgb
    cols = [c for c in FEATURES if c in tr]
    y = np.searchsorted(classes, tr[target].astype(int).to_numpy())
    m = lgb.LGBMClassifier(objective="multiclass" if len(classes) > 2 else "binary", n_estimators=300,
                           learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, random_state=0, deterministic=True,
                           force_row_wise=True, verbose=-1)
    m.fit(tr[cols], y)
    return m.predict_proba(te[cols])


FITTERS = {"baseline": baseline, "qi_logit": qi_logit, "markov": markov, "kalman": kalman, "gbm": gbm}


def walk_forward(df: pd.DataFrame, target: str, models=MODELS, min_train_days: int = 2) -> pd.DataFrame:
    """Each test day predicted by models fitted on every earlier day: one row per (day, model) with the log loss,
    the gain over the baseline (millinats a row) and the accuracy on rows that moved."""
    d = df.dropna(subset=[target]).copy()
    d = d.sort_values(["market", "sid", "t"], kind="mergesort")
    days = sorted(d.day.unique())
    classes = _classes(d[target])
    out = []
    for i, day in enumerate(days):
        if i < min_train_days:
            continue
        tr, te = d[d.day < day], d[d.day == day]
        if te.empty or tr[target].nunique() < 2:
            continue
        y = te[target].to_numpy()
        ll_base = None
        for name in models:
            p = FITTERS[name](tr, te, target, classes)
            ll = _logloss(p, y, classes)
            if name == "baseline":
                ll_base = ll
            moved = y != 0
            pred = np.asarray(classes)[p.argmax(axis=1)]
            out.append({"day": day, "target": target, "model": name, "rows": int(len(te)),
                        "logloss_mnats": round(1000 * float(ll.mean()), 3),
                        "gain_mnats": round(1000 * float((ll_base - ll).mean()), 3) if ll_base is not None else None,
                        "acc_moved": round(float((pred[moved] == y[moved]).mean()), 4) if moved.any() else None})
    return pd.DataFrame(out)


def day_interval(per_day: pd.Series, n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """The mean of a per-day series and its 90% day-bootstrap interval."""
    x = per_day.dropna().to_numpy(float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boot = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    lo, hi = np.percentile(boot, [5, 95])
    return float(x.mean()), float(lo), float(hi)
