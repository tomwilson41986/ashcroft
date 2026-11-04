"""Post-hoc calibration of the simulator's win probability (build spec 6.5): isotonic, fitted on earlier seasons only.

The calibrator for season Y is fitted on the out-of-sample simulator output of every evaluated season before Y,
so no season calibrates itself. Scores by phase and required-run-rate band use cricket/model/metrics.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from cricket.model.metrics import score

RRR_BANDS = [0, 6, 8, 10, 12, 15, 36.01]


def fit_isotonic(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip").fit(p, y)


def walk_forward_calibrate(df: pd.DataFrame, col: str = "p_sim", out: str = "p_cal") -> pd.DataFrame:
    """Adds ``out``: for each season, isotonic on all earlier seasons' ``col``; NaN for the first season."""
    df = df.copy()
    df[out] = np.nan
    years = sorted(df.year.unique())
    for y in years[1:]:
        tr = df[df.year < y]
        iso = fit_isotonic(tr[col].to_numpy(), tr.bat_won.to_numpy())
        m = df.year == y
        df.loc[m, out] = iso.predict(df.loc[m, col].to_numpy())
    return df


def by_band(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Scores of each probability column by phase and by required-run-rate band."""
    rows = []
    bands = pd.cut(df.rrr, RRR_BANDS, right=False)
    for kind, key in (("phase", df.phase), ("rrr band", bands.astype(str))):
        for k, g in df.groupby(key, observed=True):
            r = {"split": kind, "group": str(k), "n": len(g)}
            for c in cols:
                s = score(g.bat_won, g[c])
                r[f"{c}_log_loss"] = s["log_loss"]
                r[f"{c}_ece"] = s["ece"]
            rows.append(r)
    return pd.DataFrame(rows)
