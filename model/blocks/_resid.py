"""A run's result against its price, without the price's own tilt (shared by the blocks that rate on it).

The plain residual "normalised finishing position less the position the Betfair SP rank implied" leans with the
price: a favourite's rank implies it wins, so it can only fall short, and an outsider can only do better (the
scoring query of 12 Nov 2026, run 43200063928: fifths of that residual run from -0.12 for the shortest prices to
+0.12 for the longest). A rating built on it mostly says who runs short-priced horses.

Here the expectation is the market's own record: the mean normalised finishing position (a non-finisher 0) of the
runners on EARLIER days whose price stood where this one's does, by the log of (runners x the runner's share of the
Betfair SP book) in BINS steps. The residual is the run's position less that, so it averages zero at every price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.freshness_features import _col
from model.race_shape import asof_decayed_mean

BINS = np.linspace(-3.0, 2.5, 23)


def price_residual(df: pd.DataFrame, day: np.ndarray, race: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(the residual, the run's normalised finishing position); NaN where the race is not priced."""
    pos = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(float)
    n = pd.to_numeric(_col(df, "number_of_runners"), errors="coerce").to_numpy(float)
    bsp = pd.to_numeric(_col(df, "bfsp"), errors="coerce").to_numpy(float)
    priced = np.isfinite(bsp) & (bsp > 1.0)
    inv = pd.Series(np.where(priced, 1.0 / np.where(priced, bsp, 1.0), 0.0), index=race.index)
    tot = inv.groupby(race).transform("sum").to_numpy()
    rows = pd.Series(1.0, index=race.index).groupby(race).transform("sum").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        nfp = np.where(np.isfinite(pos) & (pos >= 1) & (n > 1), np.clip((n - pos) / (n - 1), 0.0, 1.0),
                       np.where(n > 1, 0.0, np.nan))
        x = np.where(priced & (tot > 0), np.log(rows * inv.to_numpy() / np.where(tot > 0, tot, 1.0)), np.nan)
    b = np.where(np.isfinite(x), np.digitize(np.clip(x, BINS[0], BINS[-1]), BINS), -1).astype(np.int64)
    exp, cnt = asof_decayed_mean(b, day, nfp, b, day)
    return np.where((b >= 0) & (cnt >= 20), nfp - exp, np.nan), nfp
