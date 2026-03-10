"""
Train BFSP and Win Probability prediction models using Betfair price data.

Uses data from 2024 onwards (betfair_prices_*.csv) to build two models:
  1. BFSP Model: LightGBM regression on log(win_bsp)
  2. Win Probability Model: LightGBM binary classification on win_result

All features are lag-safe (shift(1) pattern) to prevent lookahead bias.
Temporal train/test split ensures no future data leaks into training.

Usage:
    python -m model.train_models                    # Train both models
    python -m model.train_models --cutoff 2025-09-01  # Custom split date
    python -m model.train_models --importance       # Show feature importance
"""

import argparse
import json
import logging
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data", "betfair")
MODEL_DIR = os.path.join(PROJECT_DIR, "data", "models")


# ── Data Loading ─────────────────────────────────────────────────────

def load_betfair_data() -> pd.DataFrame:
    """Load and combine all Betfair price CSVs from 2024 onwards."""
    frames = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.startswith("betfair_prices_") or not fname.endswith(".csv"):
            continue
        year = fname.replace("betfair_prices_", "").replace(".csv", "")
        try:
            if int(year) < 2024:
                continue
        except ValueError:
            continue
        path = os.path.join(DATA_DIR, fname)
        df = pd.read_csv(path)
        log.info(f"  Loaded {fname}: {len(df):,} rows")
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No betfair_prices_*.csv files in {DATA_DIR}")

    df = pd.concat(frames, ignore_index=True)
    log.info(f"  Total raw rows: {len(df):,}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare the raw Betfair data."""
    df = df.copy()

    # Parse dates
    df["event_date"] = pd.to_datetime(df["event_date"])

    # Remove non-runners (win_result == -1) and rows with no result
    df = df[df["win_result"].isin([0.0, 1.0])].copy()

    # Remove rows with missing or invalid BSP
    df = df[df["win_bsp"].notna() & (df["win_bsp"] > 1.0)].copy()

    # Normalise horse name for consistency
    df["horse"] = df["selection_name"].str.strip().str.upper()

    # Normalise track name
    df["track"] = df["track"].str.strip().str.title()

    # Extract race class from event_name
    df["race_class"] = df["event_name"].str.extract(
        r"\(Class (\w+)\)", expand=False
    )
    class_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8}
    df["race_class_num"] = df["race_class"].map(class_map)

    # Extract race type hints from event_name
    name_lower = df["event_name"].str.lower()
    df["is_handicap"] = name_lower.str.contains("handicap", na=False).astype(int)
    df["is_chase"] = name_lower.str.contains("chase", na=False).astype(int)
    df["is_hurdle"] = name_lower.str.contains("hurdle", na=False).astype(int)
    df["is_flat"] = (
        ~name_lower.str.contains("chase|hurdle|bumper|nhf", na=False)
    ).astype(int)
    df["is_listed_plus"] = name_lower.str.contains(
        "listed|group 1|group 2|group 3|grade 1|grade 2|grade 3", na=False
    ).astype(int)

    # Country encoding
    df["is_uk"] = (df["country"] == "uk").astype(int)

    # Create race ID
    df["raceid"] = (
        df["event_date"].dt.strftime("%Y-%m-%d")
        + "_"
        + df["track"].astype(str)
        + "_"
        + df["event_id"].astype(str)
    )

    # Number of runners per race
    df["n_runners"] = df.groupby("raceid")["horse"].transform("count")

    # Log BSP (target for regression)
    df["log_bsp"] = np.log(df["win_bsp"])

    # Remove any inf values in log_bsp
    df = df[np.isfinite(df["log_bsp"])].copy()

    # Sort by date for temporal consistency
    df = df.sort_values(["event_date", "track", "event_id"]).reset_index(drop=True)

    log.info(f"  After cleaning: {len(df):,} rows, {df['raceid'].nunique():,} races")
    return df


# ── Feature Engineering ──────────────────────────────────────────────

def _lagged_expanding_mean(df: pd.DataFrame, group_col: str, val_col: str) -> pd.Series:
    """Compute expanding mean of val_col within groups, lagged by 1 (no lookahead)."""
    grp = df.groupby(group_col)
    cum_sum = grp[val_col].cumsum() - df[val_col]  # sum of all previous in group
    cum_count = grp.cumcount()  # 0-indexed = count before this row
    cum_count = cum_count.replace(0, np.nan)
    return cum_sum / cum_count


def _lagged_expanding_mean_multi(df: pd.DataFrame, group_cols: list[str], val_col: str) -> pd.Series:
    """Expanding mean with multi-column grouping, lagged by 1."""
    grp = df.groupby(group_cols)
    cum_sum = grp[val_col].cumsum() - df[val_col]  # sum of all previous rows in group
    cum_count = grp.cumcount()  # 0-indexed = count of rows before this one
    cum_count = cum_count.replace(0, np.nan)
    return cum_sum / cum_count


def _lagged_expanding_std(df: pd.DataFrame, group_col: str, val_col: str) -> pd.Series:
    """Compute expanding std of val_col within groups, lagged by 1."""
    shifted = df.groupby(group_col)[val_col].shift(1)
    return df.groupby(group_col)[val_col].transform(
        lambda x: x.shift(1).expanding().std()
    )


def _lagged_rolling_mean(df: pd.DataFrame, group_col: str, val_col: str, window: int) -> pd.Series:
    """Rolling mean of last `window` values, lagged by 1."""
    shifted = df.groupby(group_col)[val_col].shift(1)
    return shifted.groupby(df[group_col]).transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build lag-safe features from Betfair price data.

    All horse/track historical features use shift(1) to prevent lookahead.
    Uses cumsum/cumcount pattern for pandas 3.x compatibility.
    """
    df = df.copy()

    # ── Horse historical features ──
    df = df.sort_values(["horse", "event_date"]).reset_index(drop=True)

    # Career stats (lagged via cumsum/cumcount)
    df["h_runs"] = df.groupby("horse").cumcount()  # 0-indexed = runs before this one
    df["h_win_rate"] = _lagged_expanding_mean(df, "horse", "win_result")
    df["h_place_rate"] = _lagged_expanding_mean(df, "horse", "place_result")
    df["h_avg_log_bsp"] = _lagged_expanding_mean(df, "horse", "log_bsp")

    # Std via transform (works with pandas 3.x)
    df["h_std_log_bsp"] = df.groupby("horse")["log_bsp"].transform(
        lambda x: x.shift(1).expanding().std()
    )

    df["h_last_log_bsp"] = df.groupby("horse")["log_bsp"].shift(1)
    df["h_last_win"] = df.groupby("horse")["win_result"].shift(1)
    df["h_last_place"] = df.groupby("horse")["place_result"].shift(1)

    # Recent form: last 3 and 5 races
    for w in [3, 5]:
        df[f"h_lr{w}_win_rate"] = _lagged_rolling_mean(df, "horse", "win_result", w)
        df[f"h_lr{w}_avg_bsp"] = _lagged_rolling_mean(df, "horse", "log_bsp", w)

    # BSP trend: last BSP vs career average
    df["h_bsp_trend"] = df["h_last_log_bsp"] - df["h_avg_log_bsp"]

    # Implied win probability from horse's historical BSP
    df["h_avg_implied_prob"] = np.exp(-df["h_avg_log_bsp"])

    # Days since last run
    df["_date_num"] = df["event_date"].astype(np.int64) // 10**9 // 86400
    df["h_days_since_lr"] = df["_date_num"] - df.groupby("horse")["_date_num"].shift(1)
    df.drop(columns=["_date_num"], inplace=True)

    # ── Track-specific horse features ──
    df = df.sort_values(["horse", "track", "event_date"]).reset_index(drop=True)
    df["h_track_runs"] = df.groupby(["horse", "track"]).cumcount()
    df["h_track_win_rate"] = _lagged_expanding_mean_multi(df, ["horse", "track"], "win_result")

    # ── Race-class-specific horse features ──
    df = df.sort_values(["horse", "race_class_num", "event_date"]).reset_index(drop=True)
    df["h_class_runs"] = df.groupby(["horse", "race_class_num"]).cumcount()
    df["h_class_win_rate"] = _lagged_expanding_mean_multi(df, ["horse", "race_class_num"], "win_result")

    # ── Race-level features ──
    df = df.sort_values(["event_date", "track", "event_id"]).reset_index(drop=True)

    # Field strength: average historical BSP of field
    df["race_avg_h_bsp"] = df.groupby("raceid")["h_avg_log_bsp"].transform("mean")
    df["race_std_h_bsp"] = df.groupby("raceid")["h_avg_log_bsp"].transform("std")

    # Horse's BSP relative to field average
    df["h_bsp_vs_field"] = df["h_avg_log_bsp"] - df["race_avg_h_bsp"]

    # Field win experience: average runs in field
    df["race_avg_runs"] = df.groupby("raceid")["h_runs"].transform("mean")

    # This horse's experience relative to field
    df["h_exp_vs_field"] = df["h_runs"] - df["race_avg_runs"]

    # Field quality: average win rate of runners
    df["race_avg_win_rate"] = df.groupby("raceid")["h_win_rate"].transform("mean")
    df["h_wr_vs_field"] = df["h_win_rate"] - df["race_avg_win_rate"]

    # Market rank within race (BSP rank — available pre-race on Betfair)
    # Note: win_bsp IS known pre-race as it settles at race start
    # But for a true pre-race model we'd use ppwap/morningwap if available.
    # Since ppwap is mostly NaN in hub data, we skip market price as a feature
    # and rely on fundamentals only.

    # ── Track features (track-level stats) ──
    # Track win rate by favourite: what % of lowest-BSP runners win at this track
    # This is a track characteristic, not horse-specific
    df = df.sort_values(["track", "event_date"]).reset_index(drop=True)
    t_grp = df.groupby("track")
    df["track_n_races"] = t_grp["raceid"].transform("nunique")

    # ── Encode categoricals ──
    df["track_cat"] = df["track"].astype("category").cat.codes
    df["country_cat"] = df["country"].astype("category").cat.codes

    # ── Race event_time features ──
    if df["event_time"].notna().any():
        # Parse time as minutes since midnight
        def parse_time_minutes(t):
            if pd.isna(t):
                return np.nan
            parts = str(t).split(":")
            if len(parts) >= 2:
                try:
                    return int(parts[0]) * 60 + int(parts[1])
                except ValueError:
                    return np.nan
            return np.nan

        df["race_time_mins"] = df["event_time"].apply(parse_time_minutes)
    else:
        df["race_time_mins"] = np.nan

    # Restore sort order
    df = df.sort_values(["event_date", "track", "event_id"]).reset_index(drop=True)

    return df


# ── Feature Column Definitions ───────────────────────────────────────

FEATURE_COLS = [
    # Horse history
    "h_runs",
    "h_win_rate",
    "h_place_rate",
    "h_avg_log_bsp",
    "h_std_log_bsp",
    "h_last_log_bsp",
    "h_last_win",
    "h_last_place",
    "h_lr3_win_rate",
    "h_lr5_win_rate",
    "h_lr3_avg_bsp",
    "h_lr5_avg_bsp",
    "h_bsp_trend",
    "h_avg_implied_prob",
    "h_days_since_lr",
    # Horse at track/class
    "h_track_runs",
    "h_track_win_rate",
    "h_class_runs",
    "h_class_win_rate",
    # Race context
    "n_runners",
    "race_class_num",
    "is_handicap",
    "is_chase",
    "is_hurdle",
    "is_flat",
    "is_listed_plus",
    "is_uk",
    "track_cat",
    # Relative to field
    "race_avg_h_bsp",
    "race_std_h_bsp",
    "h_bsp_vs_field",
    "race_avg_runs",
    "h_exp_vs_field",
    "race_avg_win_rate",
    "h_wr_vs_field",
]


# ── Model Training ───────────────────────────────────────────────────

def train_bfsp_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[lgb.Booster, dict]:
    """Train LightGBM regression model to predict log(BFSP)."""
    X_train = train_df[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    y_train = train_df["log_bsp"].astype(float)
    X_val = val_df[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    y_val = val_df["log_bsp"].astype(float)

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
    }

    callbacks = [
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=50),
    ]

    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    # Evaluate
    y_pred = model.predict(X_val)
    y_pred = np.nan_to_num(y_pred, nan=np.nanmedian(y_train), posinf=10, neginf=-10)

    mae_log = mean_absolute_error(y_val, y_pred)
    rmse_log = np.sqrt(mean_squared_error(y_val, y_pred))
    r2_log = r2_score(y_val, y_pred)

    bfsp_true = np.exp(y_val)
    bfsp_pred = np.exp(y_pred)
    mae_bfsp = mean_absolute_error(bfsp_true, bfsp_pred)
    pct_errors = np.abs(bfsp_true - bfsp_pred) / bfsp_true
    median_ape = float(np.median(pct_errors) * 100)

    metrics = {
        "model_type": "bfsp_regression",
        "best_iteration": model.best_iteration,
        "n_features": len(feature_cols),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "log_scale": {
            "mae": round(mae_log, 4),
            "rmse": round(rmse_log, 4),
            "r2": round(r2_log, 4),
        },
        "bfsp_scale": {
            "mae": round(mae_bfsp, 2),
            "median_ape_pct": round(median_ape, 2),
        },
    }

    return model, metrics


def train_win_probability_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[lgb.Booster, dict]:
    """Train LightGBM binary classification model to predict P(win)."""
    X_train = train_df[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    y_train = train_df["win_result"].astype(float)
    X_val = val_df[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    y_val = val_df["win_result"].astype(float)

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    # Compute class weight manually for stable training
    pos_count = float(y_train.sum())
    neg_count = float(len(y_train) - pos_count)
    scale_pos = neg_count / pos_count if pos_count > 0 else 1.0

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.02,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 30,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "scale_pos_weight": scale_pos,
    }

    callbacks = [
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=100),
    ]

    model = lgb.train(
        params,
        train_set,
        num_boost_round=3000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    # Evaluate
    y_pred_raw = model.predict(X_val)
    y_pred = np.clip(y_pred_raw, 1e-7, 1 - 1e-7)

    ll = log_loss(y_val, y_pred)
    brier = brier_score_loss(y_val, y_pred)

    try:
        auc = roc_auc_score(y_val, y_pred)
    except ValueError:
        auc = None

    # Normalise probabilities per race and compute normalised log-loss
    val_copy = val_df[["raceid", "win_result"]].copy()
    val_copy["p_raw"] = y_pred_raw
    val_copy["p_norm"] = val_copy.groupby("raceid")["p_raw"].transform(
        lambda x: x / x.sum()
    )
    norm_ll = log_loss(y_val, val_copy["p_norm"].clip(1e-7, 1 - 1e-7))

    # Top-N accuracy
    top1 = _top_n_accuracy(val_df, y_pred_raw, n=1)
    top3 = _top_n_accuracy(val_df, y_pred_raw, n=3)

    # F2 statistic
    uniform_prob = 1.0 / val_df["n_runners"].replace(0, np.nan)
    numer = -(y_val * np.log(np.clip(y_pred, 1e-10, None))).sum()
    denom = -(y_val * np.log(np.clip(uniform_prob, 1e-10, None))).sum()
    f2 = 1 - (numer / denom) if denom > 0 else 0

    # Calibration table
    cal_table = _calibration_table(y_val, y_pred)

    metrics = {
        "model_type": "win_probability",
        "best_iteration": model.best_iteration,
        "n_features": len(feature_cols),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "log_loss_raw": round(ll, 6),
        "log_loss_normalised": round(norm_ll, 6),
        "brier_score": round(brier, 6),
        "auc_roc": round(auc, 6) if auc is not None else None,
        "f2_statistic": round(f2, 6),
        "top_1_accuracy": top1,
        "top_3_accuracy": top3,
        "base_rate": round(float(y_val.mean()), 6),
        "calibration_table": cal_table,
    }

    return model, metrics


def _top_n_accuracy(df: pd.DataFrame, probs: np.ndarray, n: int) -> float:
    """What % of actual winners are in model's top N picks per race."""
    temp = df[["raceid", "win_result"]].copy()
    temp["prob"] = probs
    total = 0
    correct = 0
    for _, race in temp.groupby("raceid"):
        if race["win_result"].sum() == 0:
            continue
        total += 1
        top_n = race.nlargest(n, "prob")
        if top_n["win_result"].sum() > 0:
            correct += 1
    return round(correct / total, 4) if total > 0 else 0


def _calibration_table(
    y_true: pd.Series, y_pred: np.ndarray, n_bins: int = 10
) -> list[dict]:
    """Bin predictions and compare predicted vs actual win rate."""
    temp = pd.DataFrame({"true": y_true.values, "pred": y_pred})
    temp["bin"] = pd.qcut(temp["pred"], n_bins, duplicates="drop")
    table = []
    for bin_label, group in temp.groupby("bin", observed=True):
        table.append({
            "bin": str(bin_label),
            "count": len(group),
            "avg_predicted": round(group["pred"].mean(), 4),
            "actual_win_rate": round(group["true"].mean(), 4),
        })
    return table


def show_feature_importance(model: lgb.Booster, title: str, top_n: int = 20):
    """Print top feature importances."""
    importance = model.feature_importance(importance_type="gain")
    feature_names = model.feature_name()
    pairs = sorted(zip(feature_names, importance), key=lambda x: -x[1])

    print(f"\n  {title}: Top {top_n} Features by Gain")
    print("  " + "=" * 55)
    max_imp = pairs[0][1] if pairs else 1
    for name, imp in pairs[:top_n]:
        bar_len = int(30 * imp / max_imp)
        bar = "█" * bar_len
        print(f"  {name:<25s} {bar} {imp:.0f}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train BFSP and win probability models from Betfair data"
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default=None,
        help="Train/test cutoff date (YYYY-MM-DD). Default: last 20%% of dates",
    )
    parser.add_argument(
        "--importance",
        action="store_true",
        help="Show feature importance after training",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=MODEL_DIR,
        help="Directory to save model artifacts",
    )
    args = parser.parse_args()

    # ── Load data ──
    log.info("Loading Betfair price data...")
    raw_df = load_betfair_data()

    log.info("Cleaning data...")
    df = clean_data(raw_df)

    log.info("Building features...")
    df = build_features(df)

    # Filter to available features
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    log.info(f"  {len(available_features)} features available")

    # ── Temporal split ──
    if args.cutoff:
        cutoff = pd.Timestamp(args.cutoff)
    else:
        dates = df["event_date"].sort_values().unique()
        cutoff_idx = int(len(dates) * 0.8)
        cutoff = pd.Timestamp(dates[cutoff_idx])

    train_df = df[df["event_date"] < cutoff].copy()
    test_df = df[df["event_date"] >= cutoff].copy()

    # Drop rows with NaN target
    train_bfsp = train_df.dropna(subset=["log_bsp"])
    test_bfsp = test_df.dropna(subset=["log_bsp"])
    train_win = train_df.dropna(subset=["win_result"])
    test_win = test_df.dropna(subset=["win_result"])

    log.info(f"  Train/test cutoff: {cutoff.date()}")
    log.info(f"  Train: {len(train_df):,} rows ({train_df['raceid'].nunique():,} races)")
    log.info(f"  Test:  {len(test_df):,} rows ({test_df['raceid'].nunique():,} races)")

    if len(train_df) < 100 or len(test_df) < 10:
        log.error("Insufficient data for training.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Train BFSP Model ──
    print("\n" + "=" * 60)
    print("MODEL 1: BFSP PREDICTION (Regression)")
    print("=" * 60)
    log.info("Training BFSP regression model...")

    bfsp_model, bfsp_metrics = train_bfsp_model(
        train_bfsp, test_bfsp, available_features
    )

    print(f"\n  Results ({bfsp_metrics['val_size']:,} test samples)")
    print("  " + "-" * 40)
    print(f"  Log-scale MAE:      {bfsp_metrics['log_scale']['mae']:.4f}")
    print(f"  Log-scale RMSE:     {bfsp_metrics['log_scale']['rmse']:.4f}")
    print(f"  Log-scale R²:       {bfsp_metrics['log_scale']['r2']:.4f}")
    print(f"  BFSP MAE:           {bfsp_metrics['bfsp_scale']['mae']:.2f}")
    print(f"  Median APE:         {bfsp_metrics['bfsp_scale']['median_ape_pct']:.1f}%")

    # Save BFSP model
    bfsp_model_path = os.path.join(args.output_dir, "bfsp_model.lgb")
    bfsp_model.save_model(bfsp_model_path)
    log.info(f"  BFSP model saved to: {bfsp_model_path}")

    with open(os.path.join(args.output_dir, "bfsp_model_metrics.json"), "w") as f:
        json.dump(bfsp_metrics, f, indent=2, default=str)

    if args.importance:
        show_feature_importance(bfsp_model, "BFSP Model")

    # ── Train Win Probability Model ──
    print("\n" + "=" * 60)
    print("MODEL 2: WIN PROBABILITY (Classification)")
    print("=" * 60)
    log.info("Training win probability model...")

    win_model, win_metrics = train_win_probability_model(
        train_win, test_win, available_features
    )

    print(f"\n  Results ({win_metrics['val_size']:,} test samples)")
    print("  " + "-" * 40)
    print(f"  Log-loss (raw):     {win_metrics['log_loss_raw']:.6f}")
    print(f"  Log-loss (normed):  {win_metrics['log_loss_normalised']:.6f}")
    print(f"  Brier score:        {win_metrics['brier_score']:.6f}")
    print(f"  AUC-ROC:            {win_metrics['auc_roc']}")
    print(f"  F2 statistic:       {win_metrics['f2_statistic']:.6f}")
    print(f"  Top-1 accuracy:     {win_metrics['top_1_accuracy']:.4f}")
    print(f"  Top-3 accuracy:     {win_metrics['top_3_accuracy']:.4f}")
    print(f"  Base rate:          {win_metrics['base_rate']:.4f}")

    # Calibration table
    print("\n  Calibration:")
    print(f"  {'Bin':<30s} {'Count':>6s} {'Predicted':>10s} {'Actual':>10s}")
    print("  " + "-" * 58)
    for row in win_metrics["calibration_table"]:
        print(
            f"  {row['bin']:<30s} {row['count']:>6d} "
            f"{row['avg_predicted']:>10.4f} {row['actual_win_rate']:>10.4f}"
        )

    # Save win probability model
    win_model_path = os.path.join(args.output_dir, "win_probability_model.lgb")
    win_model.save_model(win_model_path)
    log.info(f"  Win probability model saved to: {win_model_path}")

    with open(os.path.join(args.output_dir, "win_probability_metrics.json"), "w") as f:
        json.dump(win_metrics, f, indent=2, default=str)

    if args.importance:
        show_feature_importance(win_model, "Win Probability Model")

    # ── Save training summary ──
    summary = {
        "data_source": "betfair_prices (2024+)",
        "cutoff_date": str(cutoff.date()),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_date_range": {
            "min": str(train_df["event_date"].min().date()),
            "max": str(train_df["event_date"].max().date()),
        },
        "test_date_range": {
            "min": str(test_df["event_date"].min().date()),
            "max": str(test_df["event_date"].max().date()),
        },
        "feature_cols": available_features,
        "bfsp_model_metrics": bfsp_metrics,
        "win_probability_metrics": win_metrics,
    }
    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Models saved to: {args.output_dir}")
    print(f"  BFSP R²:           {bfsp_metrics['log_scale']['r2']:.4f}")
    print(f"  Win prob AUC:      {win_metrics['auc_roc']}")

    return bfsp_model, win_model, summary


if __name__ == "__main__":
    main()
