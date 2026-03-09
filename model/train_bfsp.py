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
import re
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

from model.custom_metrics import CustomMetricsEngine

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
                "win_result", "official_rating", "median_or", "max_or_in_race",
                "horse_age", "stall", "pounds", "odds", "dist_furlongs",
                "days_since_lr", "career_runs"]:
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


def _parse_race_class(event_name: str) -> int:
    """Extract race class from event_name. Returns 1-7 (1=Class A/Group 1, 7=unknown)."""
    if not isinstance(event_name, str):
        return 7
    name = event_name.lower()
    if "(class a)" in name or "group 1" in name:
        return 1
    if "(class b)" in name or "group 2" in name:
        return 2
    if "(class c)" in name or "group 3" in name or "listed" in name:
        return 3
    if "(class d)" in name:
        return 4
    if "(class e)" in name:
        return 5
    if "(class f)" in name:
        return 6
    if "(class g)" in name or "(class h)" in name:
        return 7
    return 7


def _parse_race_type(event_name: str) -> int:
    """Extract race type from event_name. Returns category code."""
    if not isinstance(event_name, str):
        return 0
    name = event_name.lower()
    if "chase" in name:
        return 1
    if "hurdle" in name:
        return 2
    if "national hunt flat" in name or "nhf" in name or "bumper" in name:
        return 3
    if "handicap" in name:
        return 4
    # Default: flat race
    return 0


def _has_data(df: pd.DataFrame, col: str, min_frac: float = 0.01) -> bool:
    """Check if a column has enough non-null data to be useful."""
    if col not in df.columns:
        return False
    return df[col].notna().mean() >= min_frac


def _going_features(df: pd.DataFrame) -> pd.DataFrame:
    """Going description features: encode going and horse performance by going."""
    if not _has_data(df, "going_description"):
        return df

    # Standardise going descriptions
    going_map = {
        "firm": 1, "good to firm": 2, "good": 3, "good to soft": 4,
        "soft": 5, "heavy": 6, "yielding": 4, "yielding to soft": 5,
        "standard": 3, "standard to slow": 4, "slow": 5, "fast": 2,
    }
    df["going_num"] = df["going_description"].str.strip().str.lower().map(going_map)

    # Binary: soft/heavy ground
    df["is_soft_ground"] = (df["going_num"] >= 5).astype(float)
    df["is_firm_ground"] = (df["going_num"] <= 2).astype(float)

    # Horse performance on this going type (lagged)
    if df["going_num"].notna().sum() > 100:
        df = df.sort_values(["horse_name", "going_num", "race_date"]).reset_index(drop=True)
        df["hg_runs"] = _grouped_lagged_stat(
            df, ["horse_name", "going_num"], "won", "cumsum"
        )
        hg_grp = df.groupby(["horse_name", "going_num"])
        df["hg_count"] = hg_grp.cumcount()
        df["hg_win_rate"] = _grouped_lagged_stat(
            df, ["horse_name", "going_num"], "won", "expanding_mean"
        )
        df["hg_avg_bfsp"] = _grouped_lagged_stat(
            df, ["horse_name", "going_num"], "log_bfsp", "expanding_mean"
        )

    return df


def _distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Distance features: raw distance and horse performance by distance."""
    if not _has_data(df, "dist_furlongs"):
        return df

    df["dist_f"] = df["dist_furlongs"].clip(lower=4, upper=36)
    df["log_dist"] = np.log(df["dist_f"].clip(lower=1))

    # Sprint vs staying: sprint < 7f, middle 7-10f, staying > 10f
    df["is_sprint"] = (df["dist_f"] < 7).astype(float)
    df["is_staying"] = (df["dist_f"] > 12).astype(float)

    # Horse performance at similar distances (within 1 furlong, lagged)
    df["dist_bucket"] = (df["dist_f"] * 2).round() / 2  # Round to nearest 0.5f
    if df["dist_bucket"].notna().sum() > 100:
        df = df.sort_values(["horse_name", "dist_bucket", "race_date"]).reset_index(drop=True)
        df["hd_count"] = df.groupby(["horse_name", "dist_bucket"]).cumcount()
        df["hd_win_rate"] = _grouped_lagged_stat(
            df, ["horse_name", "dist_bucket"], "won", "expanding_mean"
        )
        df["hd_avg_bfsp"] = _grouped_lagged_stat(
            df, ["horse_name", "dist_bucket"], "log_bfsp", "expanding_mean"
        )

    return df


def _official_rating_features(df: pd.DataFrame) -> pd.DataFrame:
    """Official rating features: raw OR, race-relative OR, OR trends."""
    if not _has_data(df, "official_rating"):
        return df

    df["or_raw"] = df["official_rating"]

    # OR relative to race average and max
    race_or_mean = df.groupby("raceid")["official_rating"].transform("mean")
    race_or_max = df.groupby("raceid")["official_rating"].transform("max")
    df["or_vs_race_avg"] = df["official_rating"] - race_or_mean
    df["or_vs_race_max"] = df["official_rating"] - race_or_max

    # OR rank within race (1 = highest OR)
    df["or_rank"] = df.groupby("raceid")["official_rating"].rank(
        ascending=False, method="min"
    )

    # OR trend: current vs horse's previous OR (lagged)
    df = df.sort_values(["horse_name", "race_date"]).reset_index(drop=True)
    grp = df.groupby("horse_name")
    df["or_prev"] = grp["official_rating"].shift(1)
    df["or_change"] = df["official_rating"] - df["or_prev"]
    df["or_avg"] = _grouped_lagged_stat(df, "horse_name", "official_rating", "expanding_mean")
    df["or_vs_own_avg"] = df["official_rating"] - df["or_avg"]

    # Median OR and max OR in race (from DB)
    if _has_data(df, "median_or"):
        df["or_vs_median"] = df["official_rating"] - df["median_or"]
    if _has_data(df, "max_or_in_race"):
        df["or_vs_max_in_race"] = df["official_rating"] - df["max_or_in_race"]

    return df


def _trainer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trainer features: career stats, track performance, recent form."""
    if not _has_data(df, "trainer"):
        return df

    df = df.sort_values(["trainer", "race_date"]).reset_index(drop=True)

    # Trainer career win rate and runs (lagged)
    df["tr_runs"] = df.groupby("trainer").cumcount()
    df["tr_win_rate"] = _grouped_lagged_stat(df, "trainer", "won", "expanding_mean")
    df["tr_wins"] = _grouped_lagged_stat(df, "trainer", "won", "cumsum")

    # Trainer recent form (last 10 runners)
    df["tr_recent_wr"] = _grouped_lagged_stat(
        df, "trainer", "won", "rolling_mean", window=10
    )

    # Trainer-track combo
    if df["track"].notna().all():
        df = df.sort_values(["trainer", "track", "race_date"]).reset_index(drop=True)
        df["trt_runs"] = df.groupby(["trainer", "track"]).cumcount()
        df["trt_win_rate"] = _grouped_lagged_stat(
            df, ["trainer", "track"], "won", "expanding_mean"
        )

    # Trainer strike rate at this class
    if "race_class_num" in df.columns:
        df = df.sort_values(["trainer", "race_class_num", "race_date"]).reset_index(drop=True)
        df["trc_win_rate"] = _grouped_lagged_stat(
            df, ["trainer", "race_class_num"], "won", "expanding_mean"
        )

    return df


def _jockey_features(df: pd.DataFrame) -> pd.DataFrame:
    """Jockey features: career stats, track performance, recent form."""
    if not _has_data(df, "jockey_name"):
        return df

    df = df.sort_values(["jockey_name", "race_date"]).reset_index(drop=True)

    # Jockey career win rate and runs (lagged)
    df["jk_runs"] = df.groupby("jockey_name").cumcount()
    df["jk_win_rate"] = _grouped_lagged_stat(df, "jockey_name", "won", "expanding_mean")
    df["jk_wins"] = _grouped_lagged_stat(df, "jockey_name", "won", "cumsum")

    # Jockey recent form (last 10 rides)
    df["jk_recent_wr"] = _grouped_lagged_stat(
        df, "jockey_name", "won", "rolling_mean", window=10
    )

    # Jockey-track combo
    if df["track"].notna().all():
        df = df.sort_values(["jockey_name", "track", "race_date"]).reset_index(drop=True)
        df["jkt_runs"] = df.groupby(["jockey_name", "track"]).cumcount()
        df["jkt_win_rate"] = _grouped_lagged_stat(
            df, ["jockey_name", "track"], "won", "expanding_mean"
        )

    # Jockey-trainer combo
    if _has_data(df, "trainer"):
        df = df.sort_values(["jockey_name", "trainer", "race_date"]).reset_index(drop=True)
        df["jktr_runs"] = df.groupby(["jockey_name", "trainer"]).cumcount()
        df["jktr_win_rate"] = _grouped_lagged_stat(
            df, ["jockey_name", "trainer"], "won", "expanding_mean"
        )

    return df


def _horse_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    """Horse profile features: age, weight, stall, sex, headgear."""
    # Age
    if _has_data(df, "horse_age"):
        df["age"] = df["horse_age"].clip(lower=2, upper=15)
        df["age_sq"] = df["age"] ** 2  # Quadratic for peak-age effect

    # Weight carried (pounds)
    if _has_data(df, "pounds"):
        df["weight_lbs"] = df["pounds"]
        # Weight relative to race average
        race_avg_wt = df.groupby("raceid")["pounds"].transform("mean")
        df["weight_vs_avg"] = df["pounds"] - race_avg_wt

    # Stall/draw position
    if _has_data(df, "stall"):
        df["draw"] = df["stall"]
        n_runners = df.groupby("raceid")["stall"].transform("count")
        df["draw_pct"] = df["stall"] / n_runners.clip(lower=1)  # 0=inside, 1=outside

    # Horse sex encoding
    if _has_data(df, "horse_sex"):
        sex_map = {"c": 0, "f": 1, "g": 2, "m": 3, "h": 4, "r": 5}
        df["sex_code"] = df["horse_sex"].str.strip().str.lower().map(sex_map)

    # Headgear: binary flags
    if _has_data(df, "headgear"):
        df["has_headgear"] = df["headgear"].notna().astype(float)
        df["first_headgear"] = 0.0
        # First time headgear = potentially significant
        df = df.sort_values(["horse_name", "race_date"]).reset_index(drop=True)
        grp = df.groupby("horse_name")
        prev_hg = grp["has_headgear"].shift(1)
        df["first_headgear"] = ((df["has_headgear"] == 1) & (prev_hg == 0)).astype(float)

    # Days since last run
    if _has_data(df, "days_since_lr"):
        df["dslr_raw"] = df["days_since_lr"].clip(lower=0, upper=365)
        df["log_dslr"] = np.log1p(df["dslr_raw"])

    # Career runs
    if _has_data(df, "career_runs"):
        df["career_runs_feat"] = df["career_runs"]

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix using CustomMetricsEngine + additional features.

    Integrates all metrics from custom_metrics.py plus place market signals,
    race context, and rich metadata (trainer, jockey, OR, going, distance).
    """
    df = df.copy()
    df = df.sort_values(["race_date", "raceid"]).reset_index(drop=True)

    # --- Phase 1: CustomMetricsEngine (horse-level metrics) ---
    log.info("Running CustomMetricsEngine (NFP, RB, WIV, WAX, WOA, CWO, ORR2, "
             "FSS, PFD, OFS, DSLR, recency, rankings)...")
    engine = CustomMetricsEngine(windows=[3, 5, 10])
    df = engine.calculate_all(df)

    # --- Phase 2: Original features that complement custom metrics ---
    log.info("Building horse BSP history features...")
    df = _horse_features(df)

    log.info("Building track-horse features...")
    df = _horse_track_features(df)

    log.info("Building race context features...")
    df = _race_context_features(df)

    log.info("Building place market features...")
    df = _place_market_features(df)

    log.info("Building recency-weighted BSP features...")
    df = _recency_weighted_features(df)

    # --- Phase 3: Rich metadata features (conditional on data availability) ---
    log.info("Building going features...")
    df = _going_features(df)

    log.info("Building distance features...")
    df = _distance_features(df)

    log.info("Building official rating features...")
    df = _official_rating_features(df)

    log.info("Building trainer features...")
    df = _trainer_features(df)

    log.info("Building jockey features...")
    df = _jockey_features(df)

    log.info("Building horse profile features...")
    df = _horse_profile_features(df)

    log.info(f"Feature matrix: {df.shape}")
    return df


def _horse_features(df: pd.DataFrame) -> pd.DataFrame:
    """Historical horse BSP performance features."""
    df = df.sort_values(["horse_name", "race_date"]).reset_index(drop=True)
    grp = df.groupby("horse_name")

    # BSP history
    df["h_avg_log_bfsp"] = _grouped_lagged_stat(df, "horse_name", "log_bfsp", "expanding_mean")
    df["h_std_log_bfsp"] = _grouped_lagged_stat(df, "horse_name", "log_bfsp", "expanding_std")
    df["h_last_log_bfsp"] = grp["log_bfsp"].shift(1)

    # Recent form: last 3 and last 5 races
    for w in [3, 5]:
        df[f"h_recent{w}_avg_log_bfsp"] = _grouped_lagged_stat(
            df, "horse_name", "log_bfsp", "rolling_mean", window=w
        )

    # BSP trend (last vs avg)
    df["h_bfsp_trend"] = df["h_last_log_bfsp"] - df["h_avg_log_bfsp"]

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


def _race_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Race-level context features including parsed class and type."""
    df["n_runners"] = df["number_of_runners"]
    df["log_n_runners"] = np.log(df["n_runners"].clip(lower=1))

    # Track encoding (frequency-based)
    track_freq = df["track"].value_counts(normalize=True).to_dict()
    df["track_freq"] = df["track"].map(track_freq)

    # Track category codes
    df["track_cat"] = df["track"].astype("category").cat.codes

    # Country encoding
    df["is_uk"] = (df["country"] == "uk").astype(int)

    # Parse race class and type from event_name
    event_col = "event_name" if "event_name" in df.columns else "race_name"
    if event_col in df.columns:
        df["race_class_num"] = df[event_col].apply(_parse_race_class)
        df["race_type_num"] = df[event_col].apply(_parse_race_type)
        df["is_handicap"] = df[event_col].apply(
            lambda x: 1 if isinstance(x, str) and "handicap" in x.lower() else 0
        )
        df["is_group_listed"] = df[event_col].apply(
            lambda x: 1 if isinstance(x, str) and (
                "group" in x.lower() or "listed" in x.lower()
            ) else 0
        )

    # Horse performance in this class (lagged)
    if "race_class_num" in df.columns:
        df = df.sort_values(["horse_name", "race_class_num", "race_date"]).reset_index(drop=True)
        hc_grp = df.groupby(["horse_name", "race_class_num"])
        df["hc_runs"] = hc_grp.cumcount()
        df["hc_win_rate"] = _grouped_lagged_stat(
            df, ["horse_name", "race_class_num"], "won", "expanding_mean"
        )

    return df


def _place_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features from the place (each-way) market."""
    if "bfsp_place" not in df.columns or df["bfsp_place"].notna().sum() == 0:
        return df

    place = df["bfsp_place"].clip(lower=1.01)
    df["log_bfsp_place"] = np.log(place)

    # Place implied probability
    df["place_impl_prob"] = 1.0 / place.replace(0, np.nan)

    # Win-place spread: difference between win and place log prices
    df["win_place_spread"] = df["log_bfsp"] - df["log_bfsp_place"]

    # Win-place probability ratio
    win_prob = 1.0 / df["bfsp"].replace(0, np.nan)
    df["win_place_ratio"] = win_prob / df["place_impl_prob"].replace(0, np.nan)

    # Lagged place BSP per horse
    df = df.sort_values(["horse_name", "race_date"]).reset_index(drop=True)
    grp = df.groupby("horse_name")
    df["h_last_log_bfsp_place"] = grp["log_bfsp_place"].shift(1)
    df["h_avg_log_bfsp_place"] = _grouped_lagged_stat(
        df, "horse_name", "log_bfsp_place", "expanding_mean"
    )

    # Recency-weighted place BSP
    for w in [3, 5]:
        weights = [RECENCY_WEIGHTS[i] for i in range(1, w + 1)]
        lagged = pd.DataFrame(
            {f"lag{i}": grp["log_bfsp_place"].shift(i) for i in range(1, w + 1)}
        )
        weight_arr = np.array(weights)
        weighted_sum = (lagged * weight_arr).sum(axis=1)
        wsum = (lagged.notna().astype(float) * weight_arr).sum(
            axis=1
        ).replace(0, np.nan)
        df[f"rwplace_lr{w}"] = weighted_sum / wsum

    return df


def _recency_weighted_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recency-weighted BSP features (ORR2 already done by CustomMetricsEngine)."""
    df = df.sort_values(["horse_name", "race_date"]).reset_index(drop=True)
    grp = df.groupby("horse_name")

    for w in [3, 5]:
        weights = [RECENCY_WEIGHTS[i] for i in range(1, w + 1)]
        lagged = pd.DataFrame(
            {f"lag{i}": grp["log_bfsp"].shift(i) for i in range(1, w + 1)}
        )
        weight_arr = np.array(weights)
        weighted_sum = (lagged * weight_arr).sum(axis=1)
        wsum = (lagged.notna().astype(float) * weight_arr).sum(
            axis=1
        ).replace(0, np.nan)
        df[f"rwbfsp_lr{w}"] = weighted_sum / wsum

    return df


# ──────────────────────────────────────────────────────────────────────
# Feature Columns
# ──────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    # ── Horse BSP history ──
    "h_avg_log_bfsp", "h_std_log_bfsp", "h_last_log_bfsp",
    "h_recent3_avg_log_bfsp", "h_recent5_avg_log_bfsp",
    "h_bfsp_trend",
    # ── Track-specific ──
    "ht_runs", "ht_win_rate", "ht_avg_log_bfsp",
    # ── Race context ──
    "n_runners", "log_n_runners", "track_freq", "track_cat", "is_uk",
    "race_class_num", "race_type_num", "is_handicap", "is_group_listed",
    # ── Horse class performance ──
    "hc_runs", "hc_win_rate",
    # ── Place market signals ──
    "log_bfsp_place", "place_impl_prob", "win_place_spread",
    "win_place_ratio", "h_last_log_bfsp_place", "h_avg_log_bfsp_place",
    "rwplace_lr3", "rwplace_lr5",
    # ── Recency-weighted BSP ──
    "rwbfsp_lr3", "rwbfsp_lr5",
    # ── CustomMetricsEngine: Horse career ──
    "preracehorsecareerNFP", "preracehorsecareerRB",
    "preracehorsecareerFSARB", "preracehorsecareerFSARB2",
    "preracehorsecareerWIV", "preracehorsecareerWAX",
    "preracehorsecareerWOA", "preracehorsecareerCWO",
    "preracehorsecareerWins", "preracehorsecareerRuns",
    "preracehorsecareerPlaces", "preracehorsecareerORR2",
    # ── CustomMetricsEngine: NFP windows ──
    "LRNFP", "LR3NFPtotal", "LR5NFPtotal", "LR10NFPtotal",
    # ── CustomMetricsEngine: ORR2 and RWO ──
    "LR_ORR2", "LR3_RWO", "LR5_RWO", "LR10_RWO",
    # ── CustomMetricsEngine: PFD ──
    "PFD3", "PFD5", "PFD10",
    # ── CustomMetricsEngine: OFS ──
    "OFS1", "OFS3", "OFS5", "OFS10",
    # ── CustomMetricsEngine: DSLR ──
    "DSLR1", "DSLR2", "WgtDSLR", "FinalDSLR",
    # ── CustomMetricsEngine: FSS and FCS ──
    "FSS", "FCS", "Race_avgOR",
    # ── CustomMetricsEngine: Prize money ──
    "WPMRF3", "WPMRF5", "WPMRF10",
    "PMW3", "PMW5", "PMW10",
    "RACE_WPMRF",
    # ── CustomMetricsEngine: Recency/confidence ──
    "LR3COUNT", "LR5COUNT", "LR10COUNT",
    "LR3wsum", "LR5wsum", "LR10wsum",
    "CIL3", "CIL5", "CIL10",
    # ── CustomMetricsEngine: LRP jockey momentum ──
    "LRPTotalScore", "totalLRPjockeyindex",
    # ── CustomMetricsEngine: Race strength ──
    "RACE_RB", "RACE_WIV", "RACE_NFP", "RACE_Wins", "RACE_WOA",
    "LR_RACE_RB", "LR_RACE_WIV", "LR_RACE_NFP", "LR_RACE_Wins", "LR_RACE_WOA",
    # ── CustomMetricsEngine: Within-race rankings ──
    "rNFP", "rNFPLR3", "rNFPLR5", "rNFPLR10",
    "horseRBrank", "horseFSARBrank", "horseFSARB2rank",
    "horseNFPrank", "horseWIVrank", "horseWAXrank",
    "horseWOArank", "horseCWOrank",
    "horseRunsrank", "horseWinsrank", "horsePlacesrank",
    "rORR2LR", "rRWOLR3", "rRWOLR5", "rRWOLR10",
    "rDSLR", "rFSS", "rFCS",
    "rPFD3", "rPFD5", "rPFD10",
    "rWPMRF3", "rWPMRF5", "rWPMRF10",
    "rPMW3", "rPMW5", "rPMW10",
    "rOFS3", "rOFS5", "rOFS10",
    "rTJWIV", "rTJNFP",
    # ── Going ──
    "going_num", "is_soft_ground", "is_firm_ground",
    "hg_runs", "hg_count", "hg_win_rate", "hg_avg_bfsp",
    # ── Distance ──
    "dist_f", "log_dist", "is_sprint", "is_staying",
    "hd_count", "hd_win_rate", "hd_avg_bfsp",
    # ── Official Rating ──
    "or_raw", "or_vs_race_avg", "or_vs_race_max", "or_rank",
    "or_change", "or_vs_own_avg", "or_vs_median", "or_vs_max_in_race",
    # ── Trainer ──
    "tr_runs", "tr_win_rate", "tr_wins", "tr_recent_wr",
    "trt_runs", "trt_win_rate", "trc_win_rate",
    # ── Jockey ──
    "jk_runs", "jk_win_rate", "jk_wins", "jk_recent_wr",
    "jkt_runs", "jkt_win_rate", "jktr_runs", "jktr_win_rate",
    # ── Horse profile ──
    "age", "age_sq", "weight_lbs", "weight_vs_avg",
    "draw", "draw_pct", "sex_code",
    "has_headgear", "first_headgear",
    "dslr_raw", "log_dslr", "career_runs_feat",
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
