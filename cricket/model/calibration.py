"""Post-hoc calibration of the simulator's win probability (build spec 6.5), fitted on earlier seasons only.

Three layers, each fitted for season Y on the out-of-sample output of every evaluated season before Y (no season
calibrates itself; the first is calibration-only):

- ``isotonic``: isotonic regression on the simulator's p;
- ``platt``: logistic regression on logit(p) (Platt scaling: shrinks an over-confident simulator towards 0.5);
- ``stack``: logistic regression on logit(p) and the logit of the same-rows baseline (a logistic on the state
  alone): the simulator's weight says what it adds beyond the state.

The layer used in production is chosen by one rule fixed before the results were seen (4 Nov 2026): the lowest log
loss on 2024 (calibrated on 2023), confirmed on 2025-2026. A layer is stored as plain numbers (``to_json``) so the
live engine applies it without scikit-learn (``apply``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from cricket.model.metrics import score

RRR_BANDS = [0, 6, 8, 10, 12, 15, 36.01]
METHODS = ("isotonic", "platt", "stack")
SELECT_ON = 2024


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_isotonic(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip").fit(p, y)


def fit(method: str, p: np.ndarray, y: np.ndarray, p_base: np.ndarray | None = None) -> dict:
    """A calibrator as plain numbers."""
    if method == "isotonic":
        iso = fit_isotonic(p, y)
        return {"method": "isotonic", "x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
    X = logit(p)[:, None] if method == "platt" else np.c_[logit(p), logit(p_base)]
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)
    return {"method": method, "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}


def apply(cal: dict, p, p_base=None) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if cal["method"] == "isotonic":
        return np.interp(p, cal["x"], cal["y"])
    z = cal["intercept"] + cal["coef"][0] * logit(p)
    if cal["method"] == "stack":
        z = z + cal["coef"][1] * logit(p_base)
    return 1.0 / (1.0 + np.exp(-z))


def walk_forward_calibrate(df: pd.DataFrame, col: str = "p_sim", out: str = "p_cal",
                           method: str = "isotonic") -> pd.DataFrame:
    """Adds ``out``: for each season, ``method`` fitted on all earlier seasons; NaN for the first season."""
    df = df.copy()
    df[out] = np.nan
    years = sorted(df.year.unique())
    for y in years[1:]:
        tr = df[df.year < y]
        cal = fit(method, tr[col].to_numpy(), tr.bat_won.to_numpy(),
                  tr.p_base.to_numpy() if "p_base" in tr else None)
        m = df.year == y
        df.loc[m, out] = apply(cal, df.loc[m, col].to_numpy(), df.loc[m, "p_base"].to_numpy()
                               if "p_base" in df else None)
    return df


def all_methods(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Every layer walk-forward (columns p_isotonic, p_platt, p_stack) and the one the fixed rule picks, copied to
    ``p_cal``."""
    for m in METHODS:
        df = walk_forward_calibrate(df, out=f"p_{m}", method=m)
    sel = df[df.year == SELECT_ON]
    if len(sel):
        chosen = min(METHODS, key=lambda m: score(sel.bat_won, sel[f"p_{m}"])["log_loss"])
    else:
        chosen = "isotonic"
    df["p_cal"] = df[f"p_{chosen}"]
    return df, chosen


def by_band(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Scores of each probability column by phase and by required-run-rate band."""
    rows = []
    bands = pd.cut(df.rrr, RRR_BANDS, right=False)
    phase = pd.Categorical(df.phase, ["powerplay", "middle", "death"], ordered=True)
    for kind, key in (("phase", phase), ("rrr band", bands)):
        for k, g in df.groupby(key, observed=True):
            r = {"split": kind, "group": str(k), "n": len(g)}
            for c in cols:
                s = score(g.bat_won, g[c])
                r[f"{c}_log_loss"] = s["log_loss"]
                r[f"{c}_ece"] = s["ece"]
            rows.append(r)
    return pd.DataFrame(rows)
