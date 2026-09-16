"""
Calibrating the BFSP forecast against realised BFSP.

The model regresses ln(BFSP) on features, one runner at a time, under squared
error. Two things follow from that and neither is harmless:

* **Compression.** A squared-error fit shrinks toward the conditional mean, so
  the spread of forecast prices inside a race is narrower than the spread of
  the prices that actually trade. The model's top pick is quoted too long and
  its outsiders too short. Measured on 253,532 real runners, the top pick's
  forecast was 8.3% above the BFSP it settled at, decaying to roughly zero by
  the fifth choice.
* **An incoherent book.** exp of a fitted log price is a conditional *median*,
  and reciprocals of medians do not add up. The raw forecasts implied a book of
  0.870 against a Betfair SP book of 1.0016 -- every price about 13% too long.

Neither is fixed by calibrating probabilities against win/lose outcomes. That
answers "how often does this horse win", and BFSP is not a win probability: it
is a price, carrying the market's own book and its favourite-longshot slope.
Calibrate the price against the price.

The correction is a quantile regression in log-price space, fitted on realised
BFSP and evaluated walk-forward:

    ln BFSP_i ≈ a + b·mean_r(ln p̂) + c·ln N_r + d·(ln p̂_i - mean_r(ln p̂))
                  + e·u_i + f·u_i² + g·u_i·(ln p̂_i - mean_r(ln p̂))

with u_i the runner's rank within the race scaled to [0, 1]. The race-mean and
ln N terms fix the level, the deviation term fixes the compression, and the
rank terms fix what is left -- the part that depends on being *selected* as the
race's shortest forecast, which no transform of a single runner's price can
reach. Every input is known before the off.

Fitting at the median (q=0.5) gives a median-unbiased price. Fitting at other
quantiles gives the distribution the early-price rule actually needs: back now
if the price on offer beats the q-th percentile of where BFSP will land.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

FEATURES = ["const", "race_mean_log", "log_n", "dev", "u", "u2", "u_dev"]
DEFAULT_QUANTILES = (0.25, 0.5, 0.75)


def build_design(df: pd.DataFrame, price_col: str = "predicted_bfsp",
                 race_col: str = "raceid") -> tuple[np.ndarray, pd.DataFrame]:
    """Design matrix from the raw forecast prices. Uses no post-race information."""
    d = df.copy()
    lp = np.log(pd.to_numeric(d[price_col], errors="coerce").clip(lower=1.01))
    d["_lp"] = lp
    g = d.groupby(race_col)["_lp"]
    d["_m"] = g.transform("mean")
    d["_n"] = g.transform("size")
    d["_dev"] = d["_lp"] - d["_m"]
    d["_rank"] = d.groupby(race_col)["_lp"].rank(method="first")
    d["_u"] = (d["_rank"] - 1.0) / np.maximum(d["_n"] - 1.0, 1.0)
    X = np.column_stack([np.ones(len(d)), d["_m"], np.log(d["_n"]), d["_dev"],
                         d["_u"], d["_u"] ** 2, d["_u"] * d["_dev"]])
    return X, d


def quantile_fit(X: np.ndarray, y: np.ndarray, q: float = 0.5, iters: int = 30,
                 eps: float = 1e-3) -> np.ndarray:
    """Quantile regression by iteratively reweighted least squares.

    Weights 1/|r| recover the L1 (median) fit; the asymmetric q/(1-q) split
    moves it to the q-th quantile. Thirty passes is ample for seven parameters
    and avoids a linear-programming dependency."""
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X[ok], y[ok]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    for _ in range(iters):
        r = y - X @ beta
        w = np.where(r >= 0, q, 1.0 - q) / np.maximum(np.abs(r), eps)
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return beta


class BSPPriceCalibrator:
    """Maps a raw forecast price to calibrated BFSP quantiles."""

    def __init__(self, quantiles=DEFAULT_QUANTILES):
        self.quantiles = tuple(float(q) for q in quantiles)
        self.coef_: dict[float, np.ndarray] = {}
        self.meta_: dict = {}

    def fit(self, df: pd.DataFrame, price_col: str = "predicted_bfsp",
            target_col: str = "bfsp", race_col: str = "raceid") -> "BSPPriceCalibrator":
        X, d = build_design(df, price_col, race_col)
        y = np.log(pd.to_numeric(d[target_col], errors="coerce").clip(lower=1.01)).values
        ok = np.isfinite(y) & (pd.to_numeric(d[target_col], errors="coerce") > 1.0).values
        for q in self.quantiles:
            self.coef_[q] = quantile_fit(X[ok], y[ok], q)
        self.meta_ = {"rows": int(ok.sum()), "races": int(d.loc[ok, race_col].nunique()),
                      "price_col": price_col, "target_col": target_col}
        if "race_date" in d.columns:
            self.meta_["trained_through"] = str(pd.to_datetime(d.loc[ok, "race_date"]).max().date())
        return self

    def predict(self, df: pd.DataFrame, q: float = 0.5, price_col: str = "predicted_bfsp",
                race_col: str = "raceid") -> np.ndarray:
        if not self.coef_:
            raise ValueError("Calibrator not fitted.")
        q = min(self.coef_, key=lambda k: abs(k - q))
        X, _ = build_design(df, price_col, race_col)
        return np.exp(np.clip(X @ self.coef_[q], np.log(1.01), np.log(1000.0)))

    def transform(self, df: pd.DataFrame, price_col: str = "predicted_bfsp",
                  race_col: str = "raceid", prefix: str = "bsp_forecast") -> pd.DataFrame:
        """Add `bsp_forecast` (median) and `bsp_forecast_qNN` columns."""
        out = df.copy()
        for q in sorted(self.coef_):
            vals = self.predict(out, q, price_col, race_col)
            out[prefix if abs(q - 0.5) < 1e-9 else f"{prefix}_q{int(round(q * 100)):02d}"] = vals
        return out

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"features": FEATURES, "meta": self.meta_,
                       "coef": {str(q): self.coef_[q].tolist() for q in self.coef_}}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BSPPriceCalibrator":
        with open(path) as f:
            data = json.load(f)
        obj = cls(quantiles=[float(q) for q in data["coef"]])
        obj.coef_ = {float(q): np.asarray(v, dtype=float) for q, v in data["coef"].items()}
        obj.meta_ = data.get("meta", {})
        return obj


# ---------------------------------------------------------------------------
# Honest evaluation
# ---------------------------------------------------------------------------

def walk_forward_calibrate(df: pd.DataFrame, price_col: str = "predicted_bfsp",
                           target_col: str = "bfsp", race_col: str = "raceid",
                           date_col: str = "race_date", n_folds: int = 6,
                           quantiles=DEFAULT_QUANTILES, out_prefix: str = "bsp_forecast") -> pd.DataFrame:
    """Fit on the past, apply to the future, fold by fold.

    A calibration scored on the rows it was fitted on will always look good;
    only the walk-forward version says whether it will hold next month."""
    d = df.copy()
    d["_date"] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.sort_values(["_date", race_col])
    dates = np.sort(d["_date"].dropna().unique())
    blocks = np.array_split(dates, n_folds + 1)
    for q in quantiles:
        col = out_prefix if abs(q - 0.5) < 1e-9 else f"{out_prefix}_q{int(round(q * 100)):02d}"
        d[col] = np.nan
    for i in range(1, n_folds + 1):
        train = d[d["_date"] < blocks[i][0]]
        mask = d["_date"].isin(blocks[i])
        if len(train) < 1000 or not mask.any():
            continue
        cal = BSPPriceCalibrator(quantiles).fit(train, price_col, target_col, race_col)
        block = cal.transform(d[mask], price_col, race_col, out_prefix)
        for c in block.columns:
            if c.startswith(out_prefix):
                d.loc[mask, c] = block[c].values
    return d.drop(columns=["_date"])


def _book(d: pd.DataFrame, col: str, race_col: str) -> float:
    return float(d.groupby(race_col)[col].apply(lambda s: (1.0 / s).sum()).mean())


def within_race_slope(d: pd.DataFrame, col: str, target_col: str = "bfsp",
                      race_col: str = "raceid") -> float:
    """Regression of log realised price on log forecast, both race-demeaned.

    This is the number that says whether a calibrator has any work to do. A
    slope above 1 means the forecast is compressed -- it does not spread the
    field as widely as the market does -- and a shrinkage correction would
    help. A slope of 1 means the within-race spread is already right, and
    since the race level is fixed by normalising the book, there is nothing
    left for a price calibrator to correct.

    Race-demeaning matters: without it the race level dominates and the slope
    measures how well the model knows a handicap from a Group 1, which is not
    the question."""
    e = d[(d[col] > 1.0) & (d[target_col] > 1.0)].copy()
    if len(e) < 2:
        return float("nan")
    lp = np.log(e[col]); lt = np.log(e[target_col])
    x = (lp - lp.groupby(e[race_col]).transform("mean")).to_numpy()
    y = (lt - lt.groupby(e[race_col]).transform("mean")).to_numpy()
    denom = float((x ** 2).sum())
    return float((x * y).sum() / denom) if denom > 0 else float("nan")


def bias_by_rank(d: pd.DataFrame, cols: dict[str, str], target_col: str = "bfsp",
                 race_col: str = "raceid", max_rank: int = 8) -> pd.DataFrame:
    """Median forecast/actual ratio and mean absolute log error, by model rank."""
    t = d.copy()
    base = list(cols.values())[0]
    t["_rank"] = t.groupby(race_col)[base].rank(method="first")
    t["_rk"] = np.minimum(t["_rank"], max_rank).astype(int)
    rows = []
    for rk, g in t.groupby("_rk"):
        row = {"rank": f"{rk}+" if rk == max_rank else str(rk), "n": len(g)}
        for label, col in cols.items():
            ratio = g[col] / g[target_col]
            row[f"ratio_{label}"] = float(ratio.median())
            row[f"mae_{label}"] = float(np.abs(np.log(ratio.clip(1e-6))).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_report(d: pd.DataFrame, raw_col: str = "predicted_bfsp",
                       cal_col: str = "bsp_forecast", target_col: str = "bfsp",
                       race_col: str = "raceid") -> dict:
    e = d[d[cal_col].notna() & (d[target_col] > 1.0)].copy()
    cols = {"raw": raw_col, "cal": cal_col}
    out = {"n": len(e), "races": int(e[race_col].nunique()),
           "by_rank": bias_by_rank(e, cols, target_col, race_col)}
    summary = []
    for label, col in cols.items():
        lr = np.log(e[col] / e[target_col])
        summary.append({"forecast": label, "median_ratio": float((e[col] / e[target_col]).median()),
                        "mean_abs_log_err": float(np.abs(lr).mean()),
                        "rmse_log": float(np.sqrt((lr ** 2).mean())),
                        "implied_book": _book(e, col, race_col),
                        "within_race_slope": within_race_slope(e, col, target_col, race_col)})
    summary.append({"forecast": "actual BSP", "median_ratio": 1.0, "mean_abs_log_err": 0.0,
                    "rmse_log": 0.0, "implied_book": _book(e, target_col, race_col),
                    "within_race_slope": 1.0})
    out["summary"] = pd.DataFrame(summary)
    qcols = sorted(c for c in e.columns if c.startswith(cal_col + "_q"))
    if qcols:
        out["coverage"] = pd.DataFrame([
            {"quantile": c, "share_bsp_below_pct": float(100 * (e[target_col] <= e[c]).mean())}
            for c in qcols + [cal_col]])
    return out
