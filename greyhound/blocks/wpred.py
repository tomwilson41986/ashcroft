"""A predicted weigh-in (the owner, 4 Nov 2026: "define a 'predicted weight' for each dog today based on historic
patterns ... then predict it for today's races when we don't have it yet"). No pre-race weigh-in source reaches us
(greyhounds.today has returned no cards since 7 Oct; GBGB fills the weight in with the result), so the day's weight is
forecast from the dog's earlier runs only:

- the dog's previous run: how far its weight was from its running mean then (``weight_vs_mean``) and how much it had
  changed (``weight_change``), the days since, and how many runs it has had;
- a linear model of the day's ``weight_change`` on those, refitted each month on the earlier months only (walk-forward:
  no month's prediction uses its own weights or later ones);
- the forecast of the two weigh-in features the model reads: the change from the last run, and the weight against the
  dog's running mean (the change plus where the last run stood, the mean having moved by a run).

Earlier days only: a dog's own same-day rows are never its "previous run". The block logs each year's out-of-sample R2
(how much of the day's weight change history foresees): the weigh-in helps the model through the part history cannot
foresee, so a forecast can add the predictable part only.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

PREFIX = "wp_"
log = logging.getLogger(__name__)


def _design(prev_wvm, prev_wc, gap, runs, month) -> np.ndarray:
    lg = np.log1p(np.clip(gap, 0, 365))
    return np.column_stack([np.ones(len(gap)), prev_wvm, prev_wc, lg, lg * prev_wvm, np.minimum(runs, 20) / 20.0,
                            np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)])


def build(df: pd.DataFrame) -> list[str]:
    names = [f"{PREFIX}{n}" for n in ("pred_change", "pred_vs_mean", "prev_vs_mean", "gap_days")]
    if "weight_change" not in df or "weight_vs_mean" not in df:
        for n in names:
            df[n] = np.float32(np.nan)
        return names
    t = pd.to_datetime(df.t)
    d = pd.DataFrame({"dog": df.dog_id.to_numpy(), "t": t.to_numpy(), "day": t.dt.normalize().to_numpy(),
                      "wc": pd.to_numeric(df.weight_change, errors="coerce").to_numpy(float),
                      "wvm": pd.to_numeric(df.weight_vs_mean, errors="coerce").to_numpy(float)}, index=df.index)
    d = d.sort_values(["dog", "t"], kind="stable")
    # the dog's last row on an earlier day: a day's own rows are folded to their first so a heat is never "previous"
    first = ~d.duplicated(["dog", "day"], keep="first")
    days = d[first]
    g = days.groupby("dog", sort=False)
    lag = pd.DataFrame({"prev_wvm": g.wvm.shift(1), "prev_wc": g.wc.shift(1),
                        "gap": (days.t - g.t.shift(1)).dt.days, "runs": g.cumcount().astype(float)})
    lag = lag.reindex(d.index)
    lag[~first.to_numpy()] = np.nan
    lag = lag.groupby([d.dog, d.day]).transform("first")          # every same-day row reads the day's first's lags
    month = pd.to_datetime(d.t).dt.month.to_numpy(float)
    X = _design(lag.prev_wvm.to_numpy(), lag.prev_wc.to_numpy(), lag.gap.to_numpy(), lag.runs.to_numpy(), month)
    y = d.wc.to_numpy()
    ok_x = np.isfinite(X).all(1)
    ok = ok_x & np.isfinite(y)
    period = pd.to_datetime(d.t).dt.to_period("M").to_numpy()
    pred = np.full(len(d), np.nan)
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
    runs = lag.runs.to_numpy()
    pred_vs_mean = pred + lag.prev_wvm.to_numpy() * np.where(runs > 0, runs / (runs + 1.0), 0.0)
    out = pd.DataFrame({names[0]: pred, names[1]: pred_vs_mean, names[2]: lag.prev_wvm.to_numpy(),
                        names[3]: lag.gap.to_numpy()}, index=d.index).reindex(df.index)
    for n in names:
        df[n] = out[n].astype(np.float32)
    years = pd.to_datetime(d.t).dt.year.to_numpy()
    for yr in sorted(set(years)):
        s = (years == yr) & np.isfinite(pred) & np.isfinite(y)
        if s.sum() > 100:
            r2 = 1 - np.sum((y[s] - pred[s]) ** 2) / np.sum((y[s] - y[s].mean()) ** 2)
            r2m = np.corrcoef(pred_vs_mean[s & np.isfinite(d.wvm.to_numpy())],
                              d.wvm.to_numpy()[s & np.isfinite(d.wvm.to_numpy())])[0, 1] ** 2
            log.info("wpred %s: out-of-sample R2 of the weight change %.3f, of the weight against its mean %.3f "
                     "(%d runs)", yr, r2, r2m, int(s.sum()))
    return names
