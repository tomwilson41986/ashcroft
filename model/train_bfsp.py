"""
BFSP Prediction Model — End-to-End Training Pipeline.

Builds features from the Betfair price data in horse_racing.db and trains
a LightGBM model to predict Betfair Starting Price (BFSP).

The model uses historical horse/track performance, market signals, and
race context to predict log(BFSP). Walk-forward validation ensures no
lookahead bias.

Usage:
    python -m model.train_bfsp
    python -m model.train_bfsp --db horse_racing.db --min-train-days 180
"""

import argparse
import json
import logging
import os
import sqlite3
from datetime import timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_DB = os.path.join(PROJECT_DIR, "horse_racing.db")
MODEL_DIR = os.path.join(PROJECT_DIR, "data", "models")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────

def load_data(db_path: str) -> pd.DataFrame:
    """Load race results from SQLite."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM race_results ORDER BY race_date, race_time", conn
    )
    conn.close()

    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df = df.dropna(subset=["race_date"])

    # Create raceid from event_id + race_number
    df["raceid"] = df["event_id"].astype(str) + "_" + df["race_number"].astype(str)

    # Clean numeric columns
    for col in ["bfsp", "bfsp_place", "ppwap", "morningwap", "ppmax", "ppmin",
                "ipmax", "ipmin", "number_of_runners", "placing_numerical",
                "win_result"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter to rows with valid BFSP and cap extreme values
    df = df[df["bfsp"].notna() & (df["bfsp"] > 1.0)].copy()
    df["bfsp"] = df["bfsp"].clip(upper=1000.0)

    # Target
    df["log_bfsp"] = np.log(df["bfsp"])
    df["won"] = (df["win_result"] == 1).astype(float)

    log.info(f"Loaded {len(df):,} rows, {df['raceid'].nunique():,} races, "
             f"{df['horse_name'].nunique():,} horses")
    return df


# ──────────────────────────────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────────────────────────────

RECENCY_WEIGHTS = {1: 1.0, 2: 0.5, 3: 1/3, 4: 0.25, 5: 0.2}


def _lagged_expanding(series: pd.Series, func: str = "mean") -> pd.Series:
    """Compute shift(1).expanding().func() preserving original index."""
    shifted = series.shift(1)
    exp = shifted.expanding()
    return getattr(exp, func)()


def _lagged_rolling(series: pd.Series, window: int, func: str = "mean") -> pd.Series:
    """Compute shift(1).rolling(w).func() preserving original index."""
    shifted = series.shift(1)
    return getattr(shifted.rolling(window, min_periods=1), func)()


def _grouped_lagged_stat(df: pd.DataFrame, group_cols, value_col: str,
                          stat: str = "expanding_mean", window: int = 3) -> pd.Series:
    """Apply lagged stat per group, returning a flat Series aligned with df.index."""
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    results = pd.Series(np.nan, index=df.index, dtype=float)

    for _, idx in df.groupby(group_cols).groups.items():
        sub = df.loc[idx, value_col]
        if stat == "expanding_mean":
            results.loc[idx] = _lagged_expanding(sub, "mean")
        elif stat == "expanding_std":
            results.loc[idx] = _lagged_expanding(sub, "std")
        elif stat == "expanding_sum":
            results.loc[idx] = sub.shift(1).expanding().sum()
        elif stat.startswith("rolling_mean"):
            results.loc[idx] = _lagged_rolling(sub, window, "mean")
        elif stat == "shift":
            results.loc[idx] = sub.shift(1)
        elif stat == "cumsum":
            results.loc[idx] = sub.shift(1).cumsum()

    return results


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix from betfair race data.

    All features use shift(1) or expanding windows on prior rows to prevent
    data leakage. Features are grouped into:
      - Horse historical performance (win rate, avg BSP, form)
      - Track-specific horse performance
      - Market signal features (morning WAP, pre-play prices)
      - Race context (field size, track encoding)
      - Within-race rankings
    """
    df = df.copy()
    df = df.sort_values(["race_date", "raceid"]).reset_index(drop=True)

    log.info("Building horse features...")
    df = _horse_features(df)

    log.info("Building track-horse features...")
    df = _horse_track_features(df)

    log.info("Building market signal features...")
    df = _market_features(df)

    log.info("Building race context features...")
    df = _race_context_features(df)

    log.info("Building NFP and RB metrics...")
    df = _nfp_rb_features(df)

    log.info("Building within-race rankings...")
    df = _within_race_ranks(df)

    log.info("Building recency-weighted features...")
    df = _recency_weighted_features(df)

    log.info(f"Feature matrix: {df.shape}")
    return df


def _horse_features(df: pd.DataFrame) -> pd.DataFrame:
    """Historical horse performance features."""
    df = df.sort_values(["horse_name", "race_date"]).reset_index(drop=True)
    grp = df.groupby("horse_name")

    # Career counts (0-indexed, so this is runs before this one)
    df["h_runs"] = grp.cumcount()

    # Win/place rates (lagged)
    df["h_win_rate"] = _grouped_lagged_stat(df, "horse_name", "won", "expanding_mean")
    df["_placed"] = (df["placing_numerical"] <= 3).astype(float)
    df["h_place_rate"] = _grouped_lagged_stat(df, "horse_name", "_placed", "expanding_mean")

    # BSP history
    df["h_avg_log_bfsp"] = _grouped_lagged_stat(df, "horse_name", "log_bfsp", "expanding_mean")
    df["h_std_log_bfsp"] = _grouped_lagged_stat(df, "horse_name", "log_bfsp", "expanding_std")
    df["h_last_log_bfsp"] = grp["log_bfsp"].shift(1)

    # Recent form: last 3 and last 5 races
    for w in [3, 5]:
        df[f"h_recent{w}_win_rate"] = _grouped_lagged_stat(
            df, "horse_name", "won", "rolling_mean", window=w
        )
        df[f"h_recent{w}_avg_log_bfsp"] = _grouped_lagged_stat(
            df, "horse_name", "log_bfsp", "rolling_mean", window=w
        )

    # Days since last run
    df["_date_num"] = df["race_date"].astype(np.int64) // 10**9 // 86400
    df["h_days_since_lr"] = df["_date_num"] - grp["_date_num"].shift(1)

    # BSP trend (last vs avg)
    df["h_bfsp_trend"] = df["h_last_log_bfsp"] - df["h_avg_log_bfsp"]

    # Cumulative win index value (WIV)
    df["_xwinrand"] = 1.0 / df["number_of_runners"].replace(0, np.nan)
    cum_wins = _grouped_lagged_stat(df, "horse_name", "won", "cumsum")
    cum_xwin = _grouped_lagged_stat(df, "horse_name", "_xwinrand", "cumsum")
    df["h_wiv"] = cum_wins / cum_xwin.replace(0, np.nan)

    # Wins above expected (WAX)
    df["_wax_raw"] = df["won"] - df["_xwinrand"]
    df["h_wax"] = _grouped_lagged_stat(df, "horse_name", "_wax_raw", "expanding_mean")

    # Last finishing position
    df["h_last_placing"] = grp["placing_numerical"].shift(1)

    # Avg placing
    df["h_avg_placing"] = _grouped_lagged_stat(
        df, "horse_name", "placing_numerical", "expanding_mean"
    )

    df.drop(columns=["_date_num"], inplace=True)

    return df


def _horse_track_features(df: pd.DataFrame) -> pd.DataFrame:
    """Horse performance at specific tracks."""
    df = df.sort_values(["horse_name", "track", "race_date"]).reset_index(drop=True)
    ht_grp = df.groupby(["horse_name", "track"])

    df["ht_runs"] = ht_grp.cumcount()
    df["ht_win_rate"] = _grouped_lagged_stat(
        df, ["horse_name", "track"], "won", "expanding_mean"
    )
    df["ht_avg_log_bfsp"] = _grouped_lagged_stat(
        df, ["horse_name", "track"], "log_bfsp", "expanding_mean"
    )

    return df


def _market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features derived from pre-play and in-play market prices."""
    # Morning WAP as a feature (log-transformed)
    df["log_morningwap"] = np.log(df["morningwap"].clip(lower=1.01))
    df["log_ppwap"] = np.log(df["ppwap"].clip(lower=1.01))

    # Price movement: morning WAP vs pre-play WAP
    df["price_movement"] = df["log_ppwap"] - df["log_morningwap"]

    # Market spread (max - min in pre-play)
    df["pp_spread"] = np.log(df["ppmax"].clip(lower=1.01)) - np.log(df["ppmin"].clip(lower=1.01))

    # Market implied probability from morning WAP
    df["morning_impl_prob"] = 1.0 / df["morningwap"].replace(0, np.nan)

    return df


def _race_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Race-level context features."""
    df["n_runners"] = df["number_of_runners"]
    df["log_n_runners"] = np.log(df["n_runners"].clip(lower=1))

    # Track encoding (frequency-based)
    track_freq = df["track"].value_counts(normalize=True).to_dict()
    df["track_freq"] = df["track"].map(track_freq)

    # Track category codes
    df["track_cat"] = df["track"].astype("category").cat.codes

    # Country encoding
    df["is_uk"] = (df["country"] == "uk").astype(int)

    return df


def _nfp_rb_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalised Finishing Position and Race Beaten metrics."""
    nr = df["number_of_runners"].replace(0, np.nan)
    denom = (nr - 1).replace(0, np.nan)

    # NFP: 1.0 for winner, 0.0 for last
    df["NFP"] = (nr - df["placing_numerical"]) / denom

    # RB: 1.0 for winner, 0.0 for last
    df["RB"] = 1 - (df["placing_numerical"] - 1) / denom

    # Field-size adjusted RB
    median_fs = nr.median()
    if pd.isna(median_fs) or median_fs == 0:
        median_fs = 10.0
    df["FSARB"] = df["RB"] * (nr / median_fs)

    # Career metrics (lagged)
    df = df.sort_values(["horse_name", "race_date"]).reset_index(drop=True)

    df["h_career_nfp"] = _grouped_lagged_stat(df, "horse_name", "NFP", "expanding_mean")
    df["h_career_rb"] = _grouped_lagged_stat(df, "horse_name", "RB", "expanding_mean")
    df["h_career_fsarb"] = _grouped_lagged_stat(df, "horse_name", "FSARB", "expanding_mean")

    # Recent NFP averages
    for w in [3, 5]:
        df[f"h_nfp_lr{w}"] = _grouped_lagged_stat(
            df, "horse_name", "NFP", "rolling_mean", window=w
        )

    # ORR2: market probability / random probability
    bf_prob = 1.0 / df["bfsp"].replace(0, np.nan)
    fs_prob = 1.0 / nr
    df["ORR2"] = bf_prob / fs_prob.replace(0, np.nan)

    df["h_career_orr2"] = _grouped_lagged_stat(df, "horse_name", "ORR2", "expanding_mean")
    grp = df.groupby("horse_name")
    df["h_last_orr2"] = grp["ORR2"].shift(1)

    return df


def _within_race_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Rank runners within each race on key metrics."""
    df = df.sort_values(["race_date", "raceid"]).reset_index(drop=True)

    rank_cols = {
        "rank_career_nfp": "h_career_nfp",
        "rank_career_rb": "h_career_rb",
        "rank_career_wiv": "h_wiv",
        "rank_h_runs": "h_runs",
        "rank_win_rate": "h_win_rate",
        "rank_last_bfsp": "h_last_log_bfsp",
        "rank_morningwap": "log_morningwap",
    }

    for rank_name, source_col in rank_cols.items():
        if source_col in df.columns:
            df[rank_name] = df.groupby("raceid")[source_col].rank(
                ascending=False, method="min", na_option="bottom"
            )

    return df


def _recency_weighted_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recency-weighted ORR2 and BSP features."""
    df = df.sort_values(["horse_name", "race_date"]).reset_index(drop=True)
    grp = df.groupby("horse_name")

    for metric, prefix in [("ORR2", "rwo"), ("log_bfsp", "rwbfsp")]:
        for w in [3, 5]:
            weights = [RECENCY_WEIGHTS[i] for i in range(1, w + 1)]
            lagged = pd.DataFrame(
                {f"lag{i}": grp[metric].shift(i) for i in range(1, w + 1)}
            )
            weight_arr = np.array(weights)
            weighted_sum = (lagged * weight_arr).sum(axis=1)
            wsum = (lagged.notna().astype(float) * weight_arr).sum(
                axis=1
            ).replace(0, np.nan)
            df[f"{prefix}_lr{w}"] = weighted_sum / wsum

    return df


# ──────────────────────────────────────────────────────────────────────
# Feature Columns
# ──────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    # Horse history
    "h_runs", "h_win_rate", "h_place_rate",
    "h_avg_log_bfsp", "h_std_log_bfsp", "h_last_log_bfsp",
    "h_recent3_win_rate", "h_recent5_win_rate",
    "h_recent3_avg_log_bfsp", "h_recent5_avg_log_bfsp",
    "h_days_since_lr", "h_bfsp_trend",
    "h_wiv", "h_wax",
    "h_last_placing", "h_avg_placing",
    # NFP/RB
    "h_career_nfp", "h_career_rb", "h_career_fsarb",
    "h_nfp_lr3", "h_nfp_lr5",
    "h_career_orr2", "h_last_orr2",
    # Track-specific
    "ht_runs", "ht_win_rate", "ht_avg_log_bfsp",
    # Market signals
    "log_morningwap", "log_ppwap", "price_movement",
    "pp_spread", "morning_impl_prob",
    # Race context
    "n_runners", "log_n_runners", "track_freq", "track_cat", "is_uk",
    # Within-race ranks
    "rank_career_nfp", "rank_career_rb", "rank_career_wiv",
    "rank_h_runs", "rank_win_rate", "rank_last_bfsp", "rank_morningwap",
    # Recency-weighted
    "rwo_lr3", "rwo_lr5", "rwbfsp_lr3", "rwbfsp_lr5",
]

# Columns for win probability model (classification)
WIN_FEATURE_COLS = FEATURE_COLS.copy()

TARGET_COL = "log_bfsp"
WIN_TARGET_COL = "won"


# ──────────────────────────────────────────────────────────────────────
# Model Training
# ──────────────────────────────────────────────────────────────────────

BFSP_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
}

WIN_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "is_unbalance": True,
}


def create_walk_forward_folds(
    df: pd.DataFrame,
    min_train_days: int = 365,
    val_window_days: int = 30,
    step_days: int = 30,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Create temporal walk-forward folds."""
    df = df.sort_values("race_date")
    min_date = df["race_date"].min()
    max_date = df["race_date"].max()

    folds = []
    train_end = min_date + timedelta(days=min_train_days)

    while train_end + timedelta(days=val_window_days) <= max_date:
        val_start = train_end
        val_end = val_start + timedelta(days=val_window_days)
        folds.append((train_end, val_start, val_end))
        train_end += timedelta(days=step_days)

    return folds


def _clean_features(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Remove rows with inf/nan in target and replace inf in features."""
    mask = np.isfinite(y)
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, y


def train_bfsp_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict | None = None,
) -> tuple[lgb.Booster, dict]:
    """Train LightGBM regression model for log(BFSP)."""
    params = params or BFSP_PARAMS.copy()
    available = [c for c in feature_cols if c in train_df.columns]

    X_train = train_df[available].astype(float)
    y_train = train_df[TARGET_COL].astype(float)
    X_val = val_df[available].astype(float)
    y_val = val_df[TARGET_COL].astype(float)

    X_train, y_train = _clean_features(X_train, y_train)
    X_val, y_val = _clean_features(X_val, y_val)

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=200),
        ],
    )

    # Evaluate
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)

    # Convert to BFSP space for interpretability
    actual_bfsp = np.exp(y_val)
    pred_bfsp = np.exp(preds)
    bfsp_mae = mean_absolute_error(actual_bfsp, pred_bfsp)
    bfsp_mape = np.mean(np.abs(actual_bfsp - pred_bfsp) / actual_bfsp) * 100

    metrics = {
        "log_rmse": round(rmse, 6),
        "log_mae": round(mae, 6),
        "r2": round(r2, 6),
        "bfsp_mae": round(bfsp_mae, 4),
        "bfsp_mape_pct": round(bfsp_mape, 2),
        "best_iteration": model.best_iteration,
        "n_features": len(available),
        "train_size": len(train_df),
        "val_size": len(val_df),
    }

    return model, metrics


def train_win_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict | None = None,
) -> tuple[lgb.Booster, dict]:
    """Train LightGBM classification model for P(win)."""
    params = params or WIN_PARAMS.copy()
    available = [c for c in feature_cols if c in train_df.columns]

    X_train = train_df[available].astype(float)
    y_train = train_df[WIN_TARGET_COL].astype(float)
    X_val = val_df[available].astype(float)
    y_val = val_df[WIN_TARGET_COL].astype(float)

    X_train, y_train = _clean_features(X_train, y_train)
    X_val, y_val = _clean_features(X_val, y_val)

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=200),
        ],
    )

    # Evaluate
    preds = model.predict(X_val)
    ll = log_loss(y_val, np.clip(preds, 1e-7, 1 - 1e-7))

    try:
        auc = roc_auc_score(y_val, preds)
    except ValueError:
        auc = None

    # Normalise probabilities per race and evaluate
    val_copy = val_df.loc[y_val.index, ["raceid", WIN_TARGET_COL]].copy()
    val_copy["p_raw"] = preds
    val_copy["p_norm"] = val_copy.groupby("raceid")["p_raw"].transform(
        lambda x: x / x.sum()
    )
    norm_ll = log_loss(val_copy[WIN_TARGET_COL], val_copy["p_norm"].clip(1e-7, 1 - 1e-7))

    metrics = {
        "log_loss_raw": round(ll, 6),
        "log_loss_normalised": round(norm_ll, 6),
        "auc_roc": round(auc, 6) if auc is not None else None,
        "best_iteration": model.best_iteration,
        "n_features": len(available),
        "train_size": len(train_df),
        "val_size": len(val_df),
    }

    return model, metrics


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_predictions(
    df: pd.DataFrame,
    bfsp_model: lgb.Booster,
    win_model: lgb.Booster,
    feature_cols: list[str],
) -> dict:
    """Comprehensive evaluation of both models."""
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].astype(float).replace([np.inf, -np.inf], np.nan)

    # Filter to rows with valid target
    valid_mask = np.isfinite(df[TARGET_COL].values) & np.isfinite(df[WIN_TARGET_COL].values)
    df = df.loc[valid_mask].copy()
    X = X.loc[valid_mask].copy()

    results = {}

    # BFSP predictions
    bfsp_preds = bfsp_model.predict(X)
    actual_log_bfsp = df[TARGET_COL].values
    results["bfsp_rmse"] = round(np.sqrt(mean_squared_error(actual_log_bfsp, bfsp_preds)), 6)
    results["bfsp_mae"] = round(mean_absolute_error(actual_log_bfsp, bfsp_preds), 6)
    results["bfsp_r2"] = round(r2_score(actual_log_bfsp, bfsp_preds), 6)

    actual_bfsp = np.exp(actual_log_bfsp)
    pred_bfsp = np.exp(bfsp_preds)
    results["bfsp_mae_price"] = round(mean_absolute_error(actual_bfsp, pred_bfsp), 4)
    results["bfsp_mape_pct"] = round(
        np.mean(np.abs(actual_bfsp - pred_bfsp) / actual_bfsp) * 100, 2
    )

    # Win probability predictions
    win_preds = win_model.predict(X)
    y_true = df[WIN_TARGET_COL].values
    results["win_logloss"] = round(
        log_loss(y_true, np.clip(win_preds, 1e-7, 1 - 1e-7)), 6
    )
    try:
        results["win_auc"] = round(roc_auc_score(y_true, win_preds), 6)
    except ValueError:
        results["win_auc"] = None

    # Normalised win predictions
    temp = df[["raceid", WIN_TARGET_COL]].copy()
    temp["p_raw"] = win_preds
    temp["p_norm"] = temp.groupby("raceid")["p_raw"].transform(
        lambda x: x / x.sum()
    )

    # Top-N accuracy
    for n in [1, 2, 3]:
        correct = 0
        total = 0
        for _, race in temp.groupby("raceid"):
            if race[WIN_TARGET_COL].sum() == 0:
                continue
            total += 1
            top_n = race.nlargest(n, "p_norm")
            if top_n[WIN_TARGET_COL].sum() > 0:
                correct += 1
        if total > 0:
            results[f"top_{n}_accuracy"] = round(correct / total, 4)

    # Calibration: predicted vs actual win rate by decile
    temp["pred_decile"] = pd.qcut(temp["p_norm"], 10, labels=False, duplicates="drop")
    calibration = []
    for decile, group in temp.groupby("pred_decile"):
        calibration.append({
            "decile": int(decile),
            "avg_predicted": round(group["p_norm"].mean(), 4),
            "actual_win_rate": round(group[WIN_TARGET_COL].mean(), 4),
            "count": len(group),
        })
    results["calibration"] = calibration

    # Betting simulation using Benter blending
    results.update(_simulate_betting(df, win_preds))

    results["n_predictions"] = len(df)
    results["n_races"] = df["raceid"].nunique()
    results["n_winners"] = int(y_true.sum())

    return results


def _simulate_betting(
    df: pd.DataFrame,
    win_preds: np.ndarray,
    lambda_: float = 0.80,
    min_edge: float = 0.05,
    kelly_fraction: float = 0.25,
    bankroll: float = 1000.0,
    commission: float = 0.05,
) -> dict:
    """Simulate betting using Benter blending."""
    temp = df[["raceid", "bfsp", "won"]].copy()
    temp["p_model"] = win_preds

    # Market probability
    temp["p_public"] = 1.0 / temp["bfsp"].replace(0, np.nan)

    # Benter blend
    p_m = temp["p_model"].clip(lower=1e-10)
    p_p = temp["p_public"].clip(lower=1e-10)
    raw = np.power(p_m, 1 - lambda_) * np.power(p_p, lambda_)
    temp["_raw"] = raw
    temp["p_combined"] = temp.groupby("raceid")["_raw"].transform(
        lambda x: x / x.sum()
    )

    # Edge
    temp["edge"] = (temp["p_combined"] / temp["p_public"]) - 1

    # Filter to overlays
    bets = temp[temp["edge"] >= min_edge].copy()
    if len(bets) == 0:
        return {"betting_n_bets": 0, "betting_roi_pct": 0}

    # Kelly stakes
    p = bets["p_combined"].clip(1e-7, 1 - 1e-7)
    b = bets["bfsp"] - 1
    kelly_f = ((p * b) - (1 - p)) / b
    kelly_f = kelly_f.clip(lower=0)
    stake = kelly_f * kelly_fraction * bankroll
    stake = stake.clip(lower=2.0, upper=min(bankroll * 0.02, 500.0))
    bets["stake"] = stake

    # Settle
    bets["returns"] = np.where(
        bets["won"] == 1,
        bets["stake"] * bets["bfsp"] * (1 - commission),
        0,
    )
    bets["pnl"] = bets["returns"] - bets["stake"]

    total_staked = bets["stake"].sum()
    total_pl = bets["pnl"].sum()
    n_bets = len(bets)
    n_winners = int(bets["won"].sum())

    return {
        "betting_n_bets": n_bets,
        "betting_n_winners": n_winners,
        "betting_strike_rate": round(n_winners / n_bets, 4) if n_bets > 0 else 0,
        "betting_total_staked": round(total_staked, 2),
        "betting_total_pl": round(total_pl, 2),
        "betting_roi_pct": round(total_pl / total_staked * 100, 2) if total_staked > 0 else 0,
    }


# ──────────────────────────────────────────────────────────────────────
# Feature Importance
# ──────────────────────────────────────────────────────────────────────

def get_feature_importance(model: lgb.Booster, importance_type: str = "gain") -> pd.DataFrame:
    """Get feature importance rankings."""
    importance = model.feature_importance(importance_type=importance_type)
    return pd.DataFrame({
        "feature": model.feature_name(),
        "importance": importance,
    }).sort_values("importance", ascending=False)


# ──────────────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────────────

def train_pipeline(
    db_path: str,
    output_dir: str,
    min_train_days: int = 365,
    val_window_days: int = 30,
    step_days: int = 30,
) -> dict:
    """Full training pipeline with walk-forward validation.

    Steps:
    1. Load data from SQLite
    2. Build features
    3. Walk-forward validation
    4. Train final models
    5. Evaluate and save
    """
    log.info("=" * 60)
    log.info("BFSP PREDICTION MODEL — TRAINING PIPELINE")
    log.info("=" * 60)

    # Step 1: Load data
    log.info("\nStep 1: Loading data...")
    df = load_data(db_path)

    # Step 2: Build features
    log.info("\nStep 2: Building features...")
    df = build_features(df)

    # Filter available features
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    log.info(f"Using {len(available_features)} features")

    # Drop rows with no valid target
    df = df[df[TARGET_COL].notna()].copy()

    # Step 3: Walk-forward validation
    log.info("\nStep 3: Walk-forward validation...")
    folds = create_walk_forward_folds(df, min_train_days, val_window_days, step_days)
    log.info(f"Created {len(folds)} folds")

    bfsp_fold_metrics = []
    win_fold_metrics = []
    all_val_preds = []

    if not folds:
        log.warning("Not enough data for walk-forward. Using 80/20 temporal split.")
        dates = df["race_date"].sort_values().unique()
        cutoff = dates[int(len(dates) * 0.8)]
        folds = [(cutoff, cutoff, df["race_date"].max())]

    for i, (train_end, val_start, val_end) in enumerate(folds):
        train_df = df[df["race_date"] < train_end].copy()
        val_df = df[
            (df["race_date"] >= val_start) & (df["race_date"] < val_end)
        ].copy()

        if len(train_df) < 100 or len(val_df) < 10:
            continue

        log.info(
            f"  Fold {i+1}/{len(folds)}: train={len(train_df):,} "
            f"({train_end.date()}), val={len(val_df):,}"
        )

        # Train BFSP regression model
        bfsp_model, bfsp_metrics = train_bfsp_model(
            train_df, val_df, available_features
        )
        bfsp_fold_metrics.append(bfsp_metrics)
        log.info(f"    BFSP: RMSE={bfsp_metrics['log_rmse']:.4f}, "
                 f"R²={bfsp_metrics['r2']:.4f}, "
                 f"MAPE={bfsp_metrics['bfsp_mape_pct']:.1f}%")

        # Train win probability model
        win_model, win_metrics = train_win_model(
            train_df, val_df, available_features
        )
        win_fold_metrics.append(win_metrics)
        log.info(f"    Win:  LogLoss={win_metrics['log_loss_normalised']:.4f}, "
                 f"AUC={win_metrics['auc_roc']}")

        # Store validation predictions
        X_val = val_df[available_features].astype(float)
        val_df = val_df.copy()
        val_df["pred_log_bfsp"] = bfsp_model.predict(X_val)
        val_df["pred_bfsp"] = np.exp(val_df["pred_log_bfsp"])
        val_df["p_model"] = win_model.predict(X_val)
        all_val_preds.append(val_df)

    # Step 4: Train final models on all data except last 10%
    log.info("\nStep 4: Training final models...")
    dates = df["race_date"].sort_values().unique()
    cutoff = dates[int(len(dates) * 0.9)]
    final_train = df[df["race_date"] < cutoff].copy()
    final_val = df[df["race_date"] >= cutoff].copy()

    log.info(f"  Final train: {len(final_train):,} rows")
    log.info(f"  Final val: {len(final_val):,} rows")

    final_bfsp_model, final_bfsp_metrics = train_bfsp_model(
        final_train, final_val, available_features
    )
    final_win_model, final_win_metrics = train_win_model(
        final_train, final_val, available_features
    )

    log.info(f"  Final BFSP: RMSE={final_bfsp_metrics['log_rmse']:.4f}, "
             f"R²={final_bfsp_metrics['r2']:.4f}")
    log.info(f"  Final Win:  LogLoss={final_win_metrics['log_loss_normalised']:.4f}")

    # Step 5: Evaluate on walk-forward predictions
    log.info("\nStep 5: Evaluation...")
    eval_results = {}
    if all_val_preds:
        combined_preds = pd.concat(all_val_preds, ignore_index=True)
        eval_results = evaluate_predictions(
            combined_preds, final_bfsp_model, final_win_model, available_features
        )
        log.info(f"  BFSP RMSE: {eval_results.get('bfsp_rmse', 'N/A')}")
        log.info(f"  BFSP R²: {eval_results.get('bfsp_r2', 'N/A')}")
        log.info(f"  Win AUC: {eval_results.get('win_auc', 'N/A')}")
        log.info(f"  Top-1 Accuracy: {eval_results.get('top_1_accuracy', 'N/A')}")
        log.info(f"  Betting ROI: {eval_results.get('betting_roi_pct', 'N/A')}%")

    # Step 6: Save artifacts
    log.info(f"\nStep 6: Saving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    final_bfsp_model.save_model(os.path.join(output_dir, "bfsp_model.lgb"))
    final_win_model.save_model(os.path.join(output_dir, "win_model.lgb"))

    # Feature importance
    bfsp_fi = get_feature_importance(final_bfsp_model)
    win_fi = get_feature_importance(final_win_model)
    bfsp_fi.to_csv(os.path.join(output_dir, "bfsp_feature_importance.csv"), index=False)
    win_fi.to_csv(os.path.join(output_dir, "win_feature_importance.csv"), index=False)

    # Training summary
    summary = {
        "data_range": {
            "min_date": str(df["race_date"].min().date()),
            "max_date": str(df["race_date"].max().date()),
            "total_rows": len(df),
            "total_races": int(df["raceid"].nunique()),
            "total_horses": int(df["horse_name"].nunique()),
        },
        "feature_cols": available_features,
        "n_folds": len(bfsp_fold_metrics),
        "bfsp_model": {
            "final_metrics": final_bfsp_metrics,
            "fold_metrics": bfsp_fold_metrics,
        },
        "win_model": {
            "final_metrics": final_win_metrics,
            "fold_metrics": win_fold_metrics,
        },
        "evaluation": eval_results,
        "top_bfsp_features": bfsp_fi.head(15).to_dict(orient="records"),
        "top_win_features": win_fi.head(15).to_dict(orient="records"),
    }

    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save model metadata
    meta = {
        "feature_cols": available_features,
        "bfsp_params": BFSP_PARAMS,
        "win_params": WIN_PARAMS,
        "bfsp_metrics": final_bfsp_metrics,
        "win_metrics": final_win_metrics,
    }
    with open(os.path.join(output_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    log.info("\n" + "=" * 60)
    log.info("TRAINING COMPLETE")
    log.info("=" * 60)
    log.info(f"  BFSP Model: RMSE={final_bfsp_metrics['log_rmse']:.4f}, "
             f"R²={final_bfsp_metrics['r2']:.4f}, "
             f"MAPE={final_bfsp_metrics['bfsp_mape_pct']:.1f}%")
    log.info(f"  Win Model:  LogLoss={final_win_metrics['log_loss_normalised']:.4f}, "
             f"AUC={final_win_metrics.get('auc_roc', 'N/A')}")
    if eval_results:
        log.info(f"  Betting ROI: {eval_results.get('betting_roi_pct', 'N/A')}%")
    log.info(f"  Models saved to: {output_dir}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Train BFSP prediction model"
    )
    parser.add_argument(
        "--db", type=str, default=DEFAULT_DB,
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--output-dir", type=str, default=MODEL_DIR,
        help="Directory to save model artifacts",
    )
    parser.add_argument(
        "--min-train-days", type=int, default=365,
        help="Minimum training window in days",
    )
    parser.add_argument(
        "--val-window", type=int, default=30,
        help="Validation window in days",
    )
    parser.add_argument(
        "--step-days", type=int, default=30,
        help="Step between folds in days",
    )
    args = parser.parse_args()

    train_pipeline(
        db_path=args.db,
        output_dir=args.output_dir,
        min_train_days=args.min_train_days,
        val_window_days=args.val_window,
        step_days=args.step_days,
    )


if __name__ == "__main__":
    main()
