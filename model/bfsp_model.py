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


def _invert(pred, df, target, race_col):
    """Model output -> raw price. Defined late; see invert_target below."""
    return invert_target(np.asarray(pred, dtype=float), df, target, race_col)


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
    target: str = "log_bfsp",
    num_iteration: int | None = None,
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
    raw_out = np.asarray(booster.predict(X, num_iteration=num_iteration), dtype=float)
    # The booster's scale depends on the target it was fitted on; the price is
    # what every caller wants. `demeaned_log` and `logit_norm_prob` both drop a
    # race-level term that the normalisation below would cancel anyway.
    raw_price = _invert(raw_out, out, target, race_col)
    out["predicted_bfsp_raw"] = raw_price
    with np.errstate(invalid="ignore", divide="ignore"):
        out["predicted_log_bfsp_raw"] = np.log(raw_price)

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


# ---------------------------------------------------------------------------
# One training recipe, used by train_bfsp.py, evaluate_oos.py and any variant.
#
# These were two different models. `evaluate_oos.py` trained plain L2 on
# log(BFSP) over every runner with no sample weights; `train_bfsp.py` defaulted
# to a profit-weighted objective that weights a runner by 1/sqrt(BFSP) and
# penalises predicting too short 1.5x, plus exponential recency decay that gives
# a three-year-old race 5% of today's weight. So the honest walk-forward numbers
# described one estimator and the published artefact was another, and nothing in
# either output said which.
#
# Both now build a TrainConfig and call fit_bfsp. The config is recorded in the
# model metadata, so a future reader can tell what was trained without reading
# the argparse defaults of whichever script produced it.
# ---------------------------------------------------------------------------

import gc
import logging
from dataclasses import dataclass, field, asdict

import lightgbm as lgb

log = logging.getLogger(__name__)

#: LightGBM settings shared by every objective. Unchanged from the trainer's
#: historical defaults except for the seed, which was absent: two runs of the
#: same configuration could differ, which makes a paired head-to-head between
#: objectives unreadable.
DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "seed": 42,
}


def profit_weighted_objective(preds, train_data):
    """Asymmetric, price-weighted loss in log space.

    Weights a runner by sqrt(1/BFSP), so favourites dominate the gradient, and
    penalises under-prediction (quoting shorter than the horse settles) 1.5x
    because that is what manufactures a false overlay.

    It is not the default. The model is asked to forecast the price of every
    runner, and this objective explicitly tells it that most of the field does
    not matter. It stays here as a measured variant.

    The row weight has to be applied here by hand. LightGBM applies a Dataset's
    weight inside each built-in objective's own gradient computation; a custom
    objective's gradients are passed straight through to the booster and the
    weight is never touched. So the recency decay the old trainer set alongside
    this objective -- computed, passed as `weight=`, and logged on every run --
    changed nothing at all, and `--decay-rate` was inert in production for as
    long as the custom objective was the default. Measured directly: with a
    100x weight skew, a built-in objective's predictions move and a custom
    one's are identical to five decimal places."""
    labels = train_data.get_label()
    residuals = preds - labels
    price_weight = np.sqrt(np.exp(-labels))
    asymmetry = np.where(residuals < 0, 1.5, 1.0)
    grad = residuals * price_weight * asymmetry
    hess = np.ones_like(grad) * price_weight * asymmetry
    row_weight = train_data.get_weight()
    if row_weight is not None:
        grad = grad * row_weight
        hess = hess * row_weight
    return grad, hess


def profit_weighted_metric(preds, train_data):
    """Price-weighted MAE, the eval metric that matches the objective above.

    Applies the row weight for the same reason the objective does: a built-in
    metric weights by the Dataset's weight, so early stopping would otherwise
    be judged on a differently-weighted quantity from the one being fitted.

    Dividing by the total weight is part of that, not a separate change. An
    unnormalised weighted sum scales with the weights, so the same model would
    score differently under `--decay-rate 0` and `--decay-rate 1` for no reason
    but the weighting, and the two variants' numbers could not be read side by
    side. It does not affect early stopping either way, which only compares the
    metric with itself."""
    labels = train_data.get_label()
    price_weight = np.sqrt(np.exp(-labels))
    row_weight = train_data.get_weight()
    if row_weight is not None:
        price_weight = price_weight * row_weight
    err = np.abs(preds - labels) * price_weight
    denom = float(np.sum(price_weight))
    return "profit_wmae", (float(np.sum(err) / denom) if denom > 0 else float("nan")), False


OBJECTIVES = ("l2", "profit_weighted")
TARGETS = ("log_bfsp", "logit_norm_prob", "demeaned_log")


@dataclass
class TrainConfig:
    """Everything that decides what gets fitted, in one recordable object."""

    #: "l2" -- plain squared error on log(BFSP), every runner weighted equally.
    #: This is what the evaluation has always measured and what the review
    #: settled on: the model is a price forecaster for the whole field.
    objective: str = "l2"

    #: Exponential recency weight, exp(-decay * days_ago / 365). 0 disables it.
    #: The trainer defaulted to 1.0, which hands a three-year-old race 5% of the
    #: weight of a current one -- a silent decision to discard most of the
    #: history. Off by default, measured as a variant.
    decay_rate: float = 0.0

    target: str = "log_bfsp"

    #: Days at the end of the training window held out for early stopping.
    #: Both scripts used to early-stop on the very fold they then scored, which
    #: chooses the iteration count with knowledge of the test set. Costs a
    #: little measured accuracy; buys a number that means what it says.
    holdout_days: int = 60

    #: Drop training rows within this many days of the fold. Trailing-window
    #: features (30-day trainer form, rolling strike rates) straddle the
    #: boundary otherwise. Matches model/stage_f.PURGE_DAYS.
    purge_days: int = 30

    #: Drop the fold's first N days. Defaults off: the label here is a public
    #: closing price a live model would legitimately have seen, which is not
    #: the leak an embargo defends against. Measured as a sensitivity.
    embargo_days: int = 0

    num_boost_round: int = 3000
    early_stopping_rounds: int = 100

    #: Refit on the whole training window at the chosen iteration count once
    #: early stopping has picked it. True for the published model, False per
    #: evaluation fold (one fit, and the fold is what is being measured).
    refit_on_full: bool = False

    seed: int = 42
    native_categoricals: bool = False
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))

    def __post_init__(self):
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {self.objective!r}")
        if self.target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, got {self.target!r}")
        self.params = dict(self.params)
        self.params["seed"] = self.seed

    def describe(self) -> dict:
        """The config as it goes into the model metadata."""
        d = asdict(self)
        d["lightgbm_objective"] = (
            "custom:profit_weighted" if self.objective == "profit_weighted" else self.params.get("objective")
        )
        d["sample_weighting"] = (
            {"type": "none", "decay_rate": 0.0} if self.decay_rate <= 0
            else {"type": "exponential_decay", "decay_rate": self.decay_rate,
                  "half_life_days": round(365.0 * np.log(2) / self.decay_rate, 1)}
        )
        return d


@dataclass
class FitResult:
    booster: object
    best_iteration: int
    holdout_start: object
    holdout_metrics: dict
    n_train: int
    n_holdout: int
    refit: bool
    #: False when best_iteration hit num_boost_round -- the holdout was still
    #: improving at the cap, so the fit is fixed-length, not early-stopped. The
    #: smoke run stopped at exactly 3000 of 3000, which reads like a converged
    #: model unless the distinction is recorded.
    early_stopped: bool = True


def sample_weights(dates, decay_rate: float, reference=None):
    """exp(-decay * days_before_reference / 365), or None when decay is off."""
    if decay_rate <= 0:
        return None
    d = pd.to_datetime(pd.Series(dates))
    ref = pd.Timestamp(reference) if reference is not None else d.max()
    days = (ref - d).dt.days.to_numpy(dtype=float)
    return np.exp(-decay_rate * days / 365.0)


def build_target(df: pd.DataFrame, target: str, race_col: str = "raceid") -> np.ndarray:
    """The label to regress on.

    All three describe the same thing differently, and because the served price
    is race-normalised, a race-level shift cancels in the output -- so
    `demeaned_log` needs no race-level model to go with it.

    - `log_bfsp`         log of the settled price, the historical target.
    - `demeaned_log`     log price minus its race mean: only the within-race
                         differences, which is all the served price keeps.
    - `logit_norm_prob`  logit of the race-normalised implied probability, which
                         is the target that lives on the same scale as the
                         thing the model is finally judged on.
    """
    lp = np.log(pd.to_numeric(df["bfsp"], errors="coerce").to_numpy(dtype=float))
    if target == "log_bfsp":
        return lp
    races = df[race_col] if race_col in df.columns else ensure_race_id(df)
    if target == "demeaned_log":
        return lp - pd.Series(lp).groupby(races.to_numpy()).transform("mean").to_numpy()
    if target == "logit_norm_prob":
        ip = 1.0 / np.exp(lp)
        book = pd.Series(ip).groupby(races.to_numpy()).transform("sum").to_numpy()
        q = np.clip(ip / book, 1e-6, 1 - 1e-6)
        return np.log(q / (1.0 - q))
    raise ValueError(f"unknown target {target!r}")


def invert_target(pred: np.ndarray, df: pd.DataFrame, target: str,
                  race_col: str = "raceid") -> np.ndarray:
    """Model output back to a raw price, before the book is normalised."""
    if target in ("log_bfsp", "demeaned_log"):
        return np.exp(pred)
    if target == "logit_norm_prob":
        q = 1.0 / (1.0 + np.exp(-np.clip(pred, -30, 30)))
        return 1.0 / np.clip(q, 1e-9, None)
    raise ValueError(f"unknown target {target!r}")


def fold_masks(dates, val_start, val_end, cfg: TrainConfig):
    """Train/validation row masks for one fold, purged and embargoed.

    Training keeps rows strictly before (val_start - purge_days); the fold keeps
    rows from (val_start + embargo_days) to val_end."""
    d = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    start, end = pd.Timestamp(val_start), pd.Timestamp(val_end)
    train = (d < start - pd.Timedelta(days=int(cfg.purge_days))).to_numpy()
    val = ((d >= start + pd.Timedelta(days=int(cfg.embargo_days))) & (d < end)).to_numpy()
    return train, val


def fit_bfsp(train_df: pd.DataFrame, feature_cols: list[str], cfg: TrainConfig,
             race_col: str = "raceid") -> FitResult:
    """Fit one booster on `train_df`, early-stopping on its own tail.

    The holdout is the last `cfg.holdout_days` of the training window, never the
    rows the caller intends to score. That is the whole point: choosing the
    iteration count on the scored fold is model selection on the test set."""
    d = train_df.sort_values("race_date")
    dates = pd.to_datetime(d["race_date"])
    holdout_start = dates.max() - pd.Timedelta(days=int(cfg.holdout_days))
    is_holdout = (dates >= holdout_start).to_numpy()
    if is_holdout.all() or not is_holdout.any():
        # Too little history to carve a holdout: fall back to a fraction.
        cut = max(1, int(len(d) * 0.9))
        is_holdout = np.zeros(len(d), dtype=bool)
        is_holdout[cut:] = True
        holdout_start = dates.iloc[cut] if cut < len(d) else dates.max()

    y = build_target(d, cfg.target, race_col)
    w = sample_weights(d["race_date"], cfg.decay_rate)

    cat = [c for c in feature_cols if c.endswith("_cat")] if cfg.native_categoricals else None

    def _ds(mask, ref=None):
        """A Dataset over the masked rows, materialising one slice at a time.

        The whole history is 632k rows by 501 float64 features -- 2.5 GB for
        one copy -- and the train job has no swap. Building a single X and
        slicing it held the full matrix plus each slice at once, and
        `free_raw_data=False` then told LightGBM to keep the raw copy after
        binning it too. The published retrain died 22 seconds into the final
        fit with a runner shutdown. Slice straight out of the frame and let
        LightGBM free the raw data once it has binned it."""
        return lgb.Dataset(
            d.loc[mask, feature_cols].astype(float),
            label=y[mask],
            weight=None if w is None else w[mask],
            reference=ref, categorical_feature=cat or "auto",
        )

    fit_set = _ds(~is_holdout)
    hold_set = _ds(is_holdout, ref=fit_set)

    params = dict(cfg.params)
    feval = None
    if cfg.objective == "profit_weighted":
        params["objective"] = profit_weighted_objective
        params.pop("metric", None)
        feval = profit_weighted_metric
        # A custom fobj turns off boost_from_average, so the fit starts at 0 in
        # log space and spends its first rounds travelling to the mean.
        params["boost_from_average"] = False

    booster = lgb.train(
        params, fit_set,
        num_boost_round=cfg.num_boost_round,
        valid_sets=[hold_set], valid_names=["holdout"],
        feval=feval,
        callbacks=[lgb.log_evaluation(period=0),
                   lgb.early_stopping(stopping_rounds=cfg.early_stopping_rounds, verbose=False)],
    )
    best = int(booster.best_iteration or cfg.num_boost_round)

    hp = booster.predict(d.loc[is_holdout, feature_cols].astype(float), num_iteration=best)
    hy = y[is_holdout]
    holdout_metrics = {
        "n": int(is_holdout.sum()),
        "mae": float(np.mean(np.abs(hp - hy))),
        "rmse": float(np.sqrt(np.mean((hp - hy) ** 2))),
    }

    if cfg.refit_on_full:
        # Early stopping picked the iteration count; the published model should
        # then see the most recent weeks too, which the holdout withheld.
        #
        # Release the split datasets first: on the full history each is a
        # gigabyte of binned features, and the refit needs its own.
        del fit_set, hold_set
        gc.collect()
        full = _ds(np.ones(len(d), dtype=bool))
        booster = lgb.train(params, full, num_boost_round=best, feval=feval,
                            callbacks=[lgb.log_evaluation(period=0)])
        del full
        gc.collect()

    early_stopped = best < cfg.num_boost_round
    if not early_stopped:
        log.warning(
            "Early stopping did not fire: best_iteration %d == num_boost_round. "
            "The holdout was still improving at the cap, so this is a "
            "fixed-length fit, not a converged one.", best,
        )

    return FitResult(
        booster=booster, best_iteration=best, holdout_start=holdout_start,
        holdout_metrics=holdout_metrics, n_train=int((~is_holdout).sum()),
        n_holdout=int(is_holdout.sum()), refit=bool(cfg.refit_on_full),
        early_stopped=early_stopped,
    )


def model_meta(cfg: TrainConfig, feature_cols: list[str], fit: FitResult | None = None,
               vocab: dict | None = None, **extra) -> dict:
    """The metadata a served model must carry, so a reader can tell what it is.

    The artefact this replaces recorded a params dict and nothing about the
    recipe, and it was not even written by this trainer -- its schema came from
    `build_datasets.py`'s single-split path. Its top three features by gain were
    `NFP_residual`, `FSALB` and `win_surprise`: the race's own result, columns
    the leak guard now forbids. Nothing in the file said so, and the live job
    loaded it."""
    from model import feature_cache

    meta = {
        "model_type": "bfsp_regression",
        "feature_cols": list(feature_cols),
        "n_features": len(feature_cols),
        "categorical_vocab": vocab or {},
        "params": dict(cfg.params),
        "post_race_guard": "passed",
    }
    meta.update(cfg.describe())
    if fit is not None:
        meta.update({
            "best_iteration": fit.best_iteration,
            "holdout_metrics": fit.holdout_metrics,
            "holdout_start": str(pd.Timestamp(fit.holdout_start).date()),
            "train_rows": fit.n_train,
            "holdout_rows": fit.n_holdout,
            "refit_on_full": fit.refit,
            "early_stopped": fit.early_stopped,
            "num_boost_round": cfg.num_boost_round,
        })
    try:
        meta["feature_code_hash"] = feature_cache.feature_code_hash()
    except Exception:                                    # pragma: no cover
        meta["feature_code_hash"] = None
    meta.update(extra)
    return meta


def assert_meta_is_servable(meta: dict) -> None:
    """Refuse to serve a model whose metadata says it should not be served.

    Two things have actually gone wrong here and neither was visible at load
    time: an artefact trained on post-race columns, and an artefact whose
    provenance was a different script from the one that was supposed to have
    produced it."""
    from model.bfsp_features import assert_no_post_race_features

    cols = meta.get("feature_cols")
    if not cols:
        raise ValueError(
            "model metadata carries no feature_cols, so what it was trained on "
            "cannot be checked. Retrain with train_bfsp.py."
        )
    assert_no_post_race_features(cols)

    if not meta.get("objective"):
        raise ValueError(
            "model metadata records no objective. This artefact predates the "
            "shared training recipe and may be the profit-weighted model while "
            "the published numbers describe the plain one. Retrain with "
            "train_bfsp.py."
        )
