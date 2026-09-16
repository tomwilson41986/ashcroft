"""One definition of what the BFSP model predicts, shared by train, evaluate and live.

Three scripts turned a booster's output into a price, and they disagreed.

`train_bfsp.py`, `evaluate_oos.py` and `predict_bfsp_today.py` each normalised the
implied probabilities so a race's book summed to one, and each then overwrote the
result with a quantile calibrator fitted on the *un-normalised* price. The
calibrator has no constraint that the book stay coherent, so on the last
evaluation the per-race book ran from 0.16 to 1.85 — fold 0, which has no prior
folds to fit a calibrator on, was the only one that came out at exactly 1.000.
Where the two disagreed, `1 / predicted_bfsp` and `predicted_win_prob_norm`
differed by a median of 13%, and downstream code picks whichever it happens to
read: the Kelly simulation uses the price, the bet analysis uses the probability.

Live made a third choice. `predict_bfsp_today.py` looks for the calibrator in
`--model-dir`, the file sits in `models/`, so it silently served the normalised
price and warned about a bias that the normalisation had already removed.

`predict_prices` below is the one definition. The price and the probability are
the same number, the book is one by construction, and the calibrator is out of
the serving path — it stays in `model/price_calibration.py` as a research
diagnostic, where `research_lab.py price-cal` can fit it against the raw column
this function now exports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RACE_KEY_PARTS = ("race_date", "track", "race_time")


def ensure_race_id(df: pd.DataFrame, race_col: str = "raceid") -> pd.Series:
    """The frame's race identifier, rebuilt from date/track/time when absent.

    Kept identical to the three copies it replaces so a cached matrix keyed on
    the old construction still groups the same way."""
    if race_col in df.columns:
        return df[race_col]
    date = pd.to_datetime(df["race_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return date + "_" + df["track"].astype(str) + "_" + df["race_time"].astype(str)


def normalise_prices(
    raw_price, race_ids, target_book: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Scale each race's implied probabilities to `target_book`.

    Returns `(probability, price)` where `price == 1 / probability` exactly, so
    no caller can pick up one without the other.

    `target_book` is one by default rather than the market's ~1.0015 over-round.
    A coherent book is the point; matching the market's margin would make the
    forecast a price to bet *against* rather than an estimate of where BFSP
    lands, and on the last evaluation rescaling to the market's book moved mean
    absolute log error by about a percent and moved Brier skill, concordance and
    ROI by nothing at all."""
    p = 1.0 / pd.to_numeric(pd.Series(raw_price), errors="coerce").to_numpy(dtype=float)
    book = pd.Series(p).groupby(pd.Series(race_ids).to_numpy()).transform("sum").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        prob = np.where(book > 0, p * target_book / book, np.nan)
        price = np.where(prob > 0, 1.0 / prob, np.nan)
    return prob, price


def predict_prices(
    booster,
    df: pd.DataFrame,
    feature_cols: list[str],
    race_col: str = "raceid",
    target_book: float = 1.0,
) -> pd.DataFrame:
    """Booster output → one coherent price and probability per runner.

    Adds, and these are the only price columns any caller should read:

    - `predicted_log_bfsp_raw` / `predicted_bfsp_raw` — the booster's own output,
      exported so the calibrator can be fitted honestly against it offline.
    - `predicted_win_prob_norm` — implied probability, race book = `target_book`.
    - `predicted_bfsp` — `1 / predicted_win_prob_norm`.
    - `predicted_win_prob` — an alias of `predicted_win_prob_norm`, because a
      column named `predicted_win_prob` that disagreed with `predicted_bfsp` is
      exactly the bug this module exists to remove.
    - `predicted_log_bfsp` — `log(predicted_bfsp)`.
    """
    out = df.copy()
    if race_col not in out.columns:
        out[race_col] = ensure_race_id(out, race_col)

    X = out[feature_cols].astype(float)
    log_raw = np.asarray(booster.predict(X), dtype=float)
    out["predicted_log_bfsp_raw"] = log_raw
    out["predicted_bfsp_raw"] = np.exp(log_raw)

    prob, price = normalise_prices(
        out["predicted_bfsp_raw"], out[race_col], target_book=target_book
    )
    out["predicted_win_prob_norm"] = prob
    out["predicted_win_prob"] = prob
    out["predicted_bfsp"] = price
    with np.errstate(invalid="ignore", divide="ignore"):
        out["predicted_log_bfsp"] = np.log(price)
    return out


class MissingFeatureColumns(RuntimeError):
    """A feature the model expects was not produced by the feature build."""


def resolve_feature_columns(
    df: pd.DataFrame,
    wanted,
    drop_prefixes: tuple = (),
    drop_names: tuple = (),
    strict: bool = True,
    log=None,
) -> list[str]:
    """The feature columns to train on, in order, with the absent ones named.

    The evaluation used to fill any missing column with NaN and train on it
    anyway. A feature block that silently failed to build therefore became a
    constant-NaN column rather than an error, and the run reported the full
    feature count while the model saw nothing. Nothing in the output said so.

    `strict` raises instead. `strict=False` drops them and returns the list, so
    a caller can go on deliberately after saying which ones it lost."""
    wanted = list(dict.fromkeys(wanted))
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        head = ", ".join(missing[:20]) + (f" ... and {len(missing) - 20} more" if len(missing) > 20 else "")
        if strict:
            raise MissingFeatureColumns(
                f"{len(missing)} of {len(wanted)} feature columns were not built: {head}. "
                "Training on them as NaN would report a feature count the model never saw. "
                "Pass --allow-missing-features to proceed without them."
            )
        if log is not None:
            log.warning("DROPPING %d feature columns the build did not produce: %s",
                        len(missing), head)

    cols = [c for c in wanted if c in df.columns]
    if drop_prefixes:
        cols = [c for c in cols if not c.startswith(tuple(drop_prefixes))]
    if drop_names:
        cols = [c for c in cols if c not in set(drop_names)]
    return cols


def all_nan_columns(df: pd.DataFrame, cols) -> list[str]:
    """Columns that are present but entirely missing -- built, and built empty."""
    present = [c for c in cols if c in df.columns]
    if not present or df.empty:
        return []
    return [c for c in present if not df[c].notna().any()]
