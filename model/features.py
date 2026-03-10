"""
Feature engineering for BFSP prediction.

For each horse in a race, compiles historical features using only data
available BEFORE that race (no data leakage).

Features are grouped into:
  - Horse historical performance (win/place rates, avg BFSP, form, OR)
  - Jockey statistics (win rate, overall and at track/distance)
  - Trainer statistics (win rate, overall and at track/distance)
  - Race context (class, distance, runners, going, surface)
  - Cross-features (horse at track, horse on going, jockey-trainer combo)
"""

import sqlite3
import numpy as np
import pandas as pd


def load_data(db_path: str) -> pd.DataFrame:
    """Load race results from SQLite, sorted by date."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM race_results ORDER BY race_date, race_time", conn
    )
    conn.close()

    df["race_date"] = pd.to_datetime(df["race_date"])
    return df


def _rolling_stats(group: pd.Series, min_periods: int = 1) -> dict:
    """Compute expanding mean, std, count for a numeric series."""
    return {
        "mean": group.expanding(min_periods=min_periods).mean(),
        "std": group.expanding(min_periods=min_periods).std(),
        "count": group.expanding(min_periods=min_periods).count(),
    }


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from race results.

    Each row represents one horse in one race. Features use only historical
    data (all computations use shift(1) or expanding windows on prior rows).

    Returns DataFrame with features and target (log_bfsp).
    """
    # Filter to rows with valid BFSP (our target)
    df = df.copy()
    df = df[df["bfsp"].notna() & (df["bfsp"] > 1.0)].copy()
    df = df.sort_values(["race_date", "race_time"]).reset_index(drop=True)

    # Target: log of BFSP (odds are log-normally distributed)
    df["log_bfsp"] = np.log(df["bfsp"])

    # Binary outcome columns
    df["won"] = (df["placing_numerical"] == 1).astype(float)
    df["placed"] = (df["placing_numerical"] <= 3).astype(float)

    # --- Horse historical features ---
    # Sort by horse + date so expanding windows work correctly
    df = df.sort_values(["horse_name", "race_date", "race_time"]).reset_index(
        drop=True
    )

    horse_grp = df.groupby("horse_name", group_keys=False)

    # Shift all historical stats by 1 to prevent leakage (use only pre-race data)
    df["h_runs"] = horse_grp.cumcount()  # 0-indexed, so this is runs before this one
    df["h_win_rate"] = horse_grp["won"].apply(
        lambda x: x.shift(1).expanding().mean()
    )
    df["h_place_rate"] = horse_grp["placed"].apply(
        lambda x: x.shift(1).expanding().mean()
    )
    df["h_avg_bfsp"] = horse_grp["log_bfsp"].apply(
        lambda x: x.shift(1).expanding().mean()
    )
    df["h_std_bfsp"] = horse_grp["log_bfsp"].apply(
        lambda x: x.shift(1).expanding().std()
    )
    df["h_avg_or"] = horse_grp["official_rating"].apply(
        lambda x: x.shift(1).expanding().mean()
    )
    df["h_last_or"] = horse_grp["official_rating"].shift(1)
    df["h_avg_placing"] = horse_grp["placing_numerical"].apply(
        lambda x: x.shift(1).expanding().mean()
    )

    # Recent form: last 3 races
    df["h_recent_win_rate"] = horse_grp["won"].apply(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df["h_recent_place_rate"] = horse_grp["placed"].apply(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df["h_recent_avg_bfsp"] = horse_grp["log_bfsp"].apply(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # Last BFSP (most recent market view of the horse)
    df["h_last_bfsp"] = horse_grp["log_bfsp"].shift(1)

    # --- Jockey historical features ---
    df = df.sort_values(["jockey_name", "race_date", "race_time"]).reset_index(
        drop=True
    )
    jockey_grp = df.groupby("jockey_name", group_keys=False)

    df["j_runs"] = jockey_grp.cumcount()
    df["j_win_rate"] = jockey_grp["won"].apply(
        lambda x: x.shift(1).expanding().mean()
    )
    df["j_place_rate"] = jockey_grp["placed"].apply(
        lambda x: x.shift(1).expanding().mean()
    )
    df["j_avg_bfsp"] = jockey_grp["log_bfsp"].apply(
        lambda x: x.shift(1).expanding().mean()
    )

    # --- Trainer historical features ---
    df = df.sort_values(["trainer", "race_date", "race_time"]).reset_index(
        drop=True
    )
    trainer_grp = df.groupby("trainer", group_keys=False)

    df["t_runs"] = trainer_grp.cumcount()
    df["t_win_rate"] = trainer_grp["won"].apply(
        lambda x: x.shift(1).expanding().mean()
    )
    df["t_place_rate"] = trainer_grp["placed"].apply(
        lambda x: x.shift(1).expanding().mean()
    )
    df["t_avg_bfsp"] = trainer_grp["log_bfsp"].apply(
        lambda x: x.shift(1).expanding().mean()
    )

    # --- Race context features (known pre-race) ---
    df["race_class_num"] = pd.to_numeric(df["race_class"], errors="coerce")
    df["dist_furlongs"] = pd.to_numeric(df["dist_furlongs"], errors="coerce")
    df["n_runners"] = df["number_of_runners"]

    # Encode categoricals
    for col in ["going_description", "surface_type", "race_type", "track",
                "horse_sex", "headgear"]:
        df[f"{col}_cat"] = df[col].astype("category").cat.codes

    # --- Cross-features: horse at this track ---
    df = df.sort_values(["horse_name", "track", "race_date"]).reset_index(
        drop=True
    )
    ht_grp = df.groupby(["horse_name", "track"], group_keys=False)
    df["h_track_runs"] = ht_grp.cumcount()
    df["h_track_win_rate"] = ht_grp["won"].apply(
        lambda x: x.shift(1).expanding().mean()
    )

    # --- Cross-features: horse on this going ---
    df = df.sort_values(
        ["horse_name", "going_description", "race_date"]
    ).reset_index(drop=True)
    hg_grp = df.groupby(["horse_name", "going_description"], group_keys=False)
    df["h_going_runs"] = hg_grp.cumcount()
    df["h_going_win_rate"] = hg_grp["won"].apply(
        lambda x: x.shift(1).expanding().mean()
    )

    # --- Cross-features: horse at this distance (within 1 furlong) ---
    # Use dist_furlongs rounded to nearest integer as a proxy
    df["dist_bucket"] = df["dist_furlongs"].round(0)
    df = df.sort_values(
        ["horse_name", "dist_bucket", "race_date"]
    ).reset_index(drop=True)
    hd_grp = df.groupby(["horse_name", "dist_bucket"], group_keys=False)
    df["h_dist_runs"] = hd_grp.cumcount()
    df["h_dist_win_rate"] = hd_grp["won"].apply(
        lambda x: x.shift(1).expanding().mean()
    )

    # --- Pre-race known numeric features ---
    df["horse_age_num"] = pd.to_numeric(df["horse_age"], errors="coerce")
    df["pounds_num"] = pd.to_numeric(df["pounds"], errors="coerce")
    df["stall_num"] = pd.to_numeric(df["stall"], errors="coerce")
    df["days_since_lr_num"] = pd.to_numeric(df["days_since_lr"], errors="coerce")
    df["career_runs_num"] = pd.to_numeric(df["career_runs"], errors="coerce")
    df["or_num"] = pd.to_numeric(df["official_rating"], errors="coerce")
    df["max_or_race"] = pd.to_numeric(df["max_or_in_race"], errors="coerce")
    df["median_or_num"] = pd.to_numeric(df["median_or"], errors="coerce")
    df["jockeys_claim_num"] = pd.to_numeric(df["jockeys_claim"], errors="coerce")

    # OR relative to field
    df["or_vs_max"] = df["or_num"] - df["max_or_race"]
    df["or_vs_median"] = df["or_num"] - df["median_or_num"]

    # Restore date sort
    df = df.sort_values(["race_date", "race_time", "track"]).reset_index(
        drop=True
    )

    return df


# Feature columns used by the model
FEATURE_COLS = [
    # Horse history
    "h_runs", "h_win_rate", "h_place_rate", "h_avg_bfsp", "h_std_bfsp",
    "h_avg_or", "h_last_or", "h_avg_placing",
    "h_recent_win_rate", "h_recent_place_rate", "h_recent_avg_bfsp",
    "h_last_bfsp",
    # Jockey
    "j_runs", "j_win_rate", "j_place_rate", "j_avg_bfsp",
    # Trainer
    "t_runs", "t_win_rate", "t_place_rate", "t_avg_bfsp",
    # Race context
    "race_class_num", "dist_furlongs", "n_runners",
    "going_description_cat", "surface_type_cat", "race_type_cat",
    "track_cat", "horse_sex_cat", "headgear_cat",
    # Cross-features
    "h_track_runs", "h_track_win_rate",
    "h_going_runs", "h_going_win_rate",
    "h_dist_runs", "h_dist_win_rate",
    # Pre-race numerics
    "horse_age_num", "pounds_num", "stall_num", "days_since_lr_num",
    "career_runs_num", "or_num", "max_or_race", "median_or_num",
    "jockeys_claim_num", "or_vs_max", "or_vs_median",
]

TARGET_COL = "log_bfsp"
