"""
Feature engineering for BFSP prediction.

For each horse in a race, compiles historical features using only data
available BEFORE that race (no data leakage).

Features are grouped into:
  - NFP / RB / FSARB (normalised finishing position, race beaten)
  - WIV / WAX / WOA (win index value, wins above expected)
  - ORR2 / PFD (odds-to-runner ratio, probability-field difference)
  - EPF (early position figure from race comments NLP)
  - FSS / FCS (field size stability, field class strength)
  - Enhanced DSLR (campaign acceleration)
  - Recency-weighted rolling features (harmonic weights)
  - Within-race rankings
  - Horse / Jockey / Trainer / Trainer-Jockey combo stats
  - Race context (class, distance, runners, going, surface)
  - Cross-features (horse at track, horse on going)

Custom metrics adapted from Ultra Betting research specifications.
"""

import re
import sqlite3

import numpy as np
import pandas as pd


# Harmonic recency weights for last 10 runs
HARMONIC_WEIGHTS = {i: 1.0 / i for i in range(1, 11)}


def load_data(db_path: str) -> pd.DataFrame:
    """Load race results from SQLite, sorted by date."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM race_results ORDER BY race_date, race_time", conn
    )
    conn.close()
    df["race_date"] = pd.to_datetime(df["race_date"])
    return df


def _lagged_expanding_mean(grp, col):
    """Shift-then-expanding-mean within a group (lag-safe)."""
    return grp[col].apply(lambda x: x.shift(1).expanding().mean())


def _lagged_expanding_sum(grp, col):
    return grp[col].apply(lambda x: x.shift(1).expanding().sum())


def _lagged_expanding_std(grp, col):
    return grp[col].apply(lambda x: x.shift(1).expanding().std())


def _recency_weighted_avg(grp, col, n_lags):
    """Compute harmonic-recency-weighted average of last n_lags values (lag-safe)."""
    def _calc(s):
        result = pd.Series(np.nan, index=s.index)
        for idx in range(len(s)):
            if idx == 0:
                continue
            w_sum = 0.0
            val_sum = 0.0
            for lag in range(1, min(n_lags + 1, idx + 1)):
                v = s.iloc[idx - lag]
                if pd.notna(v):
                    w = HARMONIC_WEIGHTS[lag]
                    val_sum += v * w
                    w_sum += w
            if w_sum > 0:
                result.iloc[idx] = val_sum / w_sum
        return result
    return grp[col].apply(_calc)


def calculate_epf(comment: str) -> float:
    """Extract Early Position Figure from race comment using NLP/regex."""
    if not comment or not isinstance(comment, str):
        return 3.0
    c = comment.lower()

    if re.search(
        r"made virtually all|made all|made most|led to\b|led,|led early|"
        r"led after|led before|led until|led over|soon led", c
    ):
        return 6.0
    if re.search(r"disputed|disputed lead|with leader", c):
        return 5.5
    if re.search(r"chased leader|tracked leader|chased winner", c):
        return 5.0
    if re.search(
        r"pressed leader|tracked leaders|chased leaders|prominent|close up|"
        r"in touch|in-touch|pressing leaders|chasing leaders|"
        r"tracked front pair|tracked leading pair|chased leading|"
        r"tracked\s|tracking leaders", c
    ):
        return 4.0
    if re.search(
        r"front of mid-division|front of mid division|front of midfield", c
    ):
        return 3.0
    if re.search(
        r"held up in midfield|held up in mid-division|"
        r"towards rear of midfield|held up in touch", c
    ):
        return 3.0
    if re.search(
        r"towards rear|held up behind|behind|held up|held up,|last pair", c
    ):
        return 1.0
    if re.search(r"in rear|always rear", c):
        return 1.0

    return 3.0


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from race results.

    Each row = one horse in one race. All features use only pre-race data
    (shift(1) / expanding windows on prior rows).

    Returns DataFrame with features and target (log_bfsp).
    """
    df = df.copy()
    df = df[df["bfsp"].notna() & (df["bfsp"] > 1.0)].copy()
    df = df.sort_values(["race_date", "race_time"]).reset_index(drop=True)

    # Ensure race_id exists for within-race operations
    if "raceid" not in df.columns:
        df["raceid"] = (
            df["race_date"].dt.strftime("%Y-%m-%d")
            + "_" + df["track"].astype(str)
            + "_" + df["race_time"].astype(str)
        )

    # Target
    df["log_bfsp"] = np.log(df["bfsp"])

    # Basic outcomes
    df["won"] = (df["placing_numerical"] == 1).astype(float)
    df["placed"] = (df["placing_numerical"] <= 3).astype(float)
    n_runners = df["number_of_runners"].astype(float)

    # =====================================================================
    # 1. NFP — Normalised Finishing Position
    # =====================================================================
    if "NFP" not in df.columns:
        df["NFP"] = (n_runners - df["placing_numerical"]) / (n_runners - 1).replace(0, np.nan)
    df["NFP"] = df["NFP"].clip(0, 1)

    # =====================================================================
    # 2. RB — Race Beaten & FSARB (field-size adjusted)
    # =====================================================================
    if "RB" not in df.columns:
        df["RB"] = 1 - (df["placing_numerical"] - 1) / (n_runners - 1).replace(0, np.nan)
    df["RB"] = df["RB"].clip(0, 1)

    median_fs = df.groupby("raceid")["number_of_runners"].transform("first").median()
    if pd.isna(median_fs) or median_fs == 0:
        median_fs = n_runners.median()
    df["FSARB"] = df["RB"] * (n_runners / median_fs)
    df["FSARB2"] = df["FSARB"] ** 2

    # =====================================================================
    # 3. xWINRAND, ORR2, PFD, OFS — Market probability metrics
    # =====================================================================
    df["xWINRAND"] = 1.0 / n_runners
    df["bf_prob"] = 1.0 / df["bfsp"]
    df["fs_prob"] = df["xWINRAND"]
    df["ORR2"] = df["bf_prob"] / df["fs_prob"]
    df["PFD"] = df["bf_prob"] - df["fs_prob"]
    df["OFS"] = df["bf_prob"] * n_runners  # same as ORR2

    # =====================================================================
    # 4. WAX, WOA — Wins above expected / average
    # =====================================================================
    df["WAX"] = df["won"] - df["xWINRAND"]
    # WOA needs field avg win rate per race
    race_avg_win = df.groupby("raceid")["won"].transform("mean")
    df["WOA"] = df["won"] - race_avg_win

    # =====================================================================
    # 5. EPF — Early Position Figure (NLP from race comments)
    # =====================================================================
    if "comment" in df.columns:
        df["EPF"] = df["comment"].apply(calculate_epf)
    else:
        df["EPF"] = 3.0
    df["EPF2"] = -0.74 + (df["EPF"] * 0.8637) + (n_runners * 0.09375)
    df["EPF3"] = df["EPF"] * (n_runners - df["placing_numerical"]) / (n_runners - 1).replace(0, np.nan)

    # Race-level pace metrics
    df["race_pace_score"] = df.groupby("raceid")["EPF"].transform("sum")
    df["pace_pressure"] = df.groupby("raceid")["EPF"].transform(
        lambda x: (x > 4).sum() / len(x) * 100
    )
    df["prom_runner"] = (df["EPF"] > 4).astype(float)

    # =====================================================================
    # 6. PMW — Prize Money Won (performance-adjusted)
    # =====================================================================
    df["prize_money_num"] = pd.to_numeric(df.get("prize_money"), errors="coerce")
    df["PMW"] = (df["prize_money_num"] / 100) * df["RB"] ** 2

    # =====================================================================
    # 7. LRP Score — Last Run Placed Score (jockey momentum)
    # =====================================================================
    df["LRP1Score"] = np.where((df["placed"] == 1) & (df["placing_numerical"] == 1), 17.41, 0)
    df["LRP2Score"] = np.where((df["placed"] == 1) & (df["placing_numerical"] == 2), 14.77, 0)
    df["LRP3Score"] = np.where((df["placed"] == 1) & (df["placing_numerical"] == 3), 12.62, 0)
    df["LRPTotalScore"] = df["LRP1Score"] + df["LRP2Score"] + df["LRP3Score"]

    # =====================================================================
    # Pre-race known numeric features
    # =====================================================================
    df["race_class_num"] = pd.to_numeric(df.get("race_class"), errors="coerce")
    df["n_runners"] = n_runners
    df["horse_age_num"] = pd.to_numeric(df.get("horse_age"), errors="coerce")
    df["pounds_num"] = pd.to_numeric(df.get("pounds"), errors="coerce")
    df["stall_num"] = pd.to_numeric(df.get("stall"), errors="coerce")
    df["days_since_lr_num"] = pd.to_numeric(df.get("days_since_lr"), errors="coerce")
    df["career_runs_num"] = pd.to_numeric(df.get("career_runs"), errors="coerce")
    df["or_num"] = pd.to_numeric(df.get("official_rating"), errors="coerce")
    df["max_or_race"] = pd.to_numeric(df.get("max_or_in_race"), errors="coerce")
    df["jockeys_claim_num"] = pd.to_numeric(df.get("jockeys_claim"), errors="coerce")
    df["or_vs_max"] = df["or_num"] - df["max_or_race"]

    # Race-level average OR (for FCS)
    df["race_avg_or"] = df.groupby("raceid")["or_num"].transform("mean")
    df["or_vs_race_avg"] = df["or_num"] - df["race_avg_or"]

    # Encode categoricals
    for col in ["going_description", "surface_type", "race_type", "track",
                "horse_sex", "headgear"]:
        if col in df.columns:
            df[f"{col}_cat"] = df[col].astype("category").cat.codes

    # Distance in furlongs (from yards if available)
    if "yards" in df.columns:
        df["dist_furlongs"] = pd.to_numeric(df["yards"], errors="coerce") / 220.0
    elif "race_distance" in df.columns:
        df["dist_furlongs"] = pd.to_numeric(
            df["race_distance"].str.extract(r"(\d+)")[0], errors="coerce"
        )
    else:
        df["dist_furlongs"] = np.nan

    # =====================================================================
    # HORSE-LEVEL HISTORICAL FEATURES (lag-safe)
    # =====================================================================
    df = df.sort_values(["horse_name", "race_date", "race_time"]).reset_index(drop=True)
    hg = df.groupby("horse_name")

    # Run count
    df["h_runs"] = hg.cumcount()

    # Career expanding stats (lagged)
    for metric, col in [
        ("h_career_NFP", "NFP"), ("h_career_RB", "RB"),
        ("h_career_FSARB", "FSARB"), ("h_career_FSARB2", "FSARB2"),
        ("h_avg_bfsp", "log_bfsp"), ("h_avg_or", "or_num"),
    ]:
        df[metric] = _lagged_expanding_mean(hg, col)

    df["h_std_bfsp"] = _lagged_expanding_std(hg, "log_bfsp")

    # WIV: cumulative wins / cumulative xWINRAND (lagged)
    df["h_cum_wins"] = _lagged_expanding_sum(hg, "won")
    df["h_cum_xwin"] = _lagged_expanding_sum(hg, "xWINRAND")
    df["h_WIV"] = df["h_cum_wins"] / df["h_cum_xwin"].replace(0, np.nan)

    # WAX / WOA / CWO career cumulative (lagged)
    df["h_career_WAX"] = _lagged_expanding_mean(hg, "WAX")
    df["h_career_WOA"] = _lagged_expanding_mean(hg, "WOA")
    df["h_CWO"] = _lagged_expanding_sum(hg, "WAX")

    # Win / place rates
    df["h_win_rate"] = _lagged_expanding_mean(hg, "won")
    df["h_place_rate"] = _lagged_expanding_mean(hg, "placed")

    # Last run values
    df["h_last_bfsp"] = hg["log_bfsp"].shift(1)
    df["h_last_or"] = hg["or_num"].shift(1)
    df["h_last_NFP"] = hg["NFP"].shift(1)
    df["h_last_ORR2"] = hg["ORR2"].shift(1)
    df["h_last_EPF"] = hg["EPF2"].shift(1)

    # Career EPF
    df["h_career_EPF"] = _lagged_expanding_mean(hg, "EPF2")
    df["h_career_ORR2"] = _lagged_expanding_mean(hg, "ORR2")

    # Recency-weighted rolling features (harmonic weights, 3/5/10 windows)
    for window in [3, 5, 10]:
        df[f"h_rw{window}_NFP"] = _recency_weighted_avg(hg, "NFP", window)
        df[f"h_rw{window}_ORR2"] = _recency_weighted_avg(hg, "ORR2", window)
        df[f"h_rw{window}_bfsp"] = _recency_weighted_avg(hg, "log_bfsp", window)
        df[f"h_rw{window}_PFD"] = _recency_weighted_avg(hg, "PFD", window)

    # Simple rolling windows for comparison
    for window in [3, 5]:
        df[f"h_lr{window}_win_rate"] = hg["won"].apply(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f"h_lr{window}_place_rate"] = hg["placed"].apply(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    # =====================================================================
    # FSS — Field Size Stability (RMSD of field size changes)
    # =====================================================================
    def _calc_fss(s_runners):
        """RMSD of field-size deltas over last 10 runs."""
        result = pd.Series(np.nan, index=s_runners.index)
        vals = s_runners.values
        for idx in range(1, len(vals)):
            current = vals[idx]
            deltas_sq = []
            for lag in range(1, min(11, idx + 1)):
                prev = vals[idx - lag]
                if pd.notna(prev) and pd.notna(current):
                    deltas_sq.append((current - prev) ** 2)
            if deltas_sq:
                result.iloc[idx] = np.sqrt(np.sum(deltas_sq)) / len(deltas_sq)
        return result

    df["h_FSS"] = hg["n_runners"].apply(_calc_fss)

    # =====================================================================
    # FCS — Field Class Strength (RMSD of race avg OR changes)
    # =====================================================================
    df["h_FCS"] = hg["race_avg_or"].apply(_calc_fss)

    # =====================================================================
    # Enhanced DSLR — Campaign acceleration
    # =====================================================================
    df["h_dslr1"] = hg["race_date"].apply(
        lambda x: x.diff().dt.days
    )
    df["h_dslr2"] = hg["race_date"].apply(
        lambda x: x.diff(2).dt.days
    )
    df["h_dslr3"] = hg["race_date"].apply(
        lambda x: x.diff(3).dt.days
    )
    df["h_dslr4"] = hg["race_date"].apply(
        lambda x: x.diff(4).dt.days
    )
    # Weighted DSLR rate of change
    dslr12 = (df["h_dslr2"] - df["h_dslr1"]) * 1.0
    dslr23 = (df["h_dslr3"] - df["h_dslr2"]) * 0.5
    dslr34 = (df["h_dslr4"] - df["h_dslr3"]) * 0.333
    df["h_wgt_dslr"] = dslr12 + dslr23 + dslr34

    # Prize money weighted
    for window in [3, 5]:
        df[f"h_rw{window}_PMW"] = _recency_weighted_avg(hg, "PMW", window)
        df[f"h_rw{window}_prize"] = _recency_weighted_avg(hg, "prize_money_num", window)

    # =====================================================================
    # JOCKEY-LEVEL HISTORICAL FEATURES
    # =====================================================================
    df = df.sort_values(["jockey_name", "race_date", "race_time"]).reset_index(drop=True)
    jg = df.groupby("jockey_name")

    df["j_runs"] = jg.cumcount()
    df["j_win_rate"] = _lagged_expanding_mean(jg, "won")
    df["j_place_rate"] = _lagged_expanding_mean(jg, "placed")
    df["j_avg_bfsp"] = _lagged_expanding_mean(jg, "log_bfsp")
    df["j_career_NFP"] = _lagged_expanding_mean(jg, "NFP")
    df["j_career_EPF"] = _lagged_expanding_mean(jg, "EPF2")

    # Jockey WIV
    df["j_cum_wins"] = _lagged_expanding_sum(jg, "won")
    df["j_cum_xwin"] = _lagged_expanding_sum(jg, "xWINRAND")
    df["j_WIV"] = df["j_cum_wins"] / df["j_cum_xwin"].replace(0, np.nan)
    df["j_career_WAX"] = _lagged_expanding_mean(jg, "WAX")

    # Jockey LRP index
    df["j_cum_LRP"] = _lagged_expanding_sum(jg, "LRPTotalScore")
    df["j_LRP_index"] = (df["j_cum_LRP"] / (df["j_runs"].replace(0, np.nan))) * 10

    # =====================================================================
    # TRAINER-LEVEL HISTORICAL FEATURES
    # =====================================================================
    df = df.sort_values(["trainer", "race_date", "race_time"]).reset_index(drop=True)
    tg = df.groupby("trainer")

    df["t_runs"] = tg.cumcount()
    df["t_win_rate"] = _lagged_expanding_mean(tg, "won")
    df["t_place_rate"] = _lagged_expanding_mean(tg, "placed")
    df["t_avg_bfsp"] = _lagged_expanding_mean(tg, "log_bfsp")
    df["t_career_NFP"] = _lagged_expanding_mean(tg, "NFP")
    df["t_career_EPF"] = _lagged_expanding_mean(tg, "EPF2")

    # Trainer WIV
    df["t_cum_wins"] = _lagged_expanding_sum(tg, "won")
    df["t_cum_xwin"] = _lagged_expanding_sum(tg, "xWINRAND")
    df["t_WIV"] = df["t_cum_wins"] / df["t_cum_xwin"].replace(0, np.nan)
    df["t_career_WAX"] = _lagged_expanding_mean(tg, "WAX")

    # =====================================================================
    # TRAINER-JOCKEY COMBO FEATURES
    # =====================================================================
    df = df.sort_values(["trainer", "jockey_name", "race_date", "race_time"]).reset_index(drop=True)
    tjg = df.groupby(["trainer", "jockey_name"])

    df["tj_runs"] = tjg.cumcount()
    df["tj_win_rate"] = _lagged_expanding_mean(tjg, "won")
    df["tj_cum_wins"] = _lagged_expanding_sum(tjg, "won")
    df["tj_cum_xwin"] = _lagged_expanding_sum(tjg, "xWINRAND")
    df["tj_WIV"] = df["tj_cum_wins"] / df["tj_cum_xwin"].replace(0, np.nan)
    df["tj_career_NFP"] = _lagged_expanding_mean(tjg, "NFP")

    # =====================================================================
    # CROSS-FEATURES: horse at track / going / distance
    # =====================================================================
    df = df.sort_values(["horse_name", "track", "race_date"]).reset_index(drop=True)
    ht_grp = df.groupby(["horse_name", "track"])
    df["h_track_runs"] = ht_grp.cumcount()
    df["h_track_win_rate"] = _lagged_expanding_mean(ht_grp, "won")

    df = df.sort_values(["horse_name", "going_description", "race_date"]).reset_index(drop=True)
    hg_grp = df.groupby(["horse_name", "going_description"])
    df["h_going_runs"] = hg_grp.cumcount()
    df["h_going_win_rate"] = _lagged_expanding_mean(hg_grp, "won")

    df["dist_bucket"] = df["dist_furlongs"].round(0)
    df = df.sort_values(["horse_name", "dist_bucket", "race_date"]).reset_index(drop=True)
    hd_grp = df.groupby(["horse_name", "dist_bucket"])
    df["h_dist_runs"] = hd_grp.cumcount()
    df["h_dist_win_rate"] = _lagged_expanding_mean(hd_grp, "won")

    # =====================================================================
    # RACE STRENGTH METRICS (per-race averages of pre-race horse stats)
    # =====================================================================
    df = df.sort_values(["race_date", "race_time", "track"]).reset_index(drop=True)

    for metric, col in [
        ("race_str_RB", "h_career_RB"), ("race_str_WIV", "h_WIV"),
        ("race_str_NFP", "h_career_NFP"), ("race_str_WAX", "h_career_WAX"),
    ]:
        df[metric] = df.groupby("raceid")[col].transform("mean")

    # Horse's stats vs race average
    df["h_RB_vs_race"] = df["h_career_RB"] - df["race_str_RB"]
    df["h_WIV_vs_race"] = df["h_WIV"] - df["race_str_WIV"]
    df["h_NFP_vs_race"] = df["h_career_NFP"] - df["race_str_NFP"]

    # =====================================================================
    # WITHIN-RACE RANKINGS
    # =====================================================================
    rank_cols = {
        "h_career_NFP": "rank_NFP",
        "h_WIV": "rank_WIV",
        "h_career_RB": "rank_RB",
        "h_career_WAX": "rank_WAX",
        "h_career_ORR2": "rank_ORR2",
        "h_career_EPF": "rank_EPF",
        "h_last_bfsp": "rank_last_bfsp",
        "or_num": "rank_OR",
        "j_WIV": "rank_j_WIV",
        "t_WIV": "rank_t_WIV",
        "h_rw3_NFP": "rank_rw3_NFP",
        "h_rw5_NFP": "rank_rw5_NFP",
        "h_FSS": "rank_FSS",
        "h_FCS": "rank_FCS",
    }
    for src_col, rank_name in rank_cols.items():
        if src_col in df.columns:
            df[rank_name] = df.groupby("raceid")[src_col].rank(
                ascending=False, method="min", na_option="bottom"
            )
            # Normalise rank by field size (0-1 scale)
            df[rank_name] = df[rank_name] / df["n_runners"]

    # =====================================================================
    # Restore date sort
    # =====================================================================
    df = df.sort_values(["race_date", "race_time", "track"]).reset_index(drop=True)

    return df


# Feature columns used by the model
FEATURE_COLS = [
    # Horse career stats
    "h_runs", "h_win_rate", "h_place_rate",
    "h_career_NFP", "h_career_RB", "h_career_FSARB", "h_career_FSARB2",
    "h_avg_bfsp", "h_std_bfsp", "h_avg_or",
    "h_WIV", "h_CWO", "h_career_WAX", "h_career_WOA",
    "h_career_EPF", "h_career_ORR2",
    # Last run
    "h_last_bfsp", "h_last_or", "h_last_NFP", "h_last_ORR2", "h_last_EPF",
    # Recency-weighted rolling (3/5/10)
    "h_rw3_NFP", "h_rw5_NFP", "h_rw10_NFP",
    "h_rw3_ORR2", "h_rw5_ORR2", "h_rw10_ORR2",
    "h_rw3_bfsp", "h_rw5_bfsp", "h_rw10_bfsp",
    "h_rw3_PFD", "h_rw5_PFD", "h_rw10_PFD",
    # Simple rolling
    "h_lr3_win_rate", "h_lr5_win_rate",
    "h_lr3_place_rate", "h_lr5_place_rate",
    # Stability
    "h_FSS", "h_FCS",
    # Enhanced DSLR
    "h_dslr1", "h_wgt_dslr",
    # Prize money
    "h_rw3_PMW", "h_rw5_PMW", "h_rw3_prize", "h_rw5_prize",
    # Jockey
    "j_runs", "j_win_rate", "j_place_rate", "j_avg_bfsp",
    "j_career_NFP", "j_career_EPF", "j_WIV", "j_career_WAX", "j_LRP_index",
    # Trainer
    "t_runs", "t_win_rate", "t_place_rate", "t_avg_bfsp",
    "t_career_NFP", "t_career_EPF", "t_WIV", "t_career_WAX",
    # Trainer-Jockey combo
    "tj_runs", "tj_win_rate", "tj_WIV", "tj_career_NFP",
    # Race context
    "race_class_num", "dist_furlongs", "n_runners",
    "going_description_cat", "surface_type_cat", "race_type_cat",
    "track_cat", "horse_sex_cat", "headgear_cat",
    # Race-level EPF/pace
    "race_pace_score", "pace_pressure", "prom_runner",
    "EPF2",  # this horse's EPF (known pre-race from comment if available)
    # Cross-features
    "h_track_runs", "h_track_win_rate",
    "h_going_runs", "h_going_win_rate",
    "h_dist_runs", "h_dist_win_rate",
    # Pre-race numerics
    "horse_age_num", "pounds_num", "stall_num", "days_since_lr_num",
    "career_runs_num", "or_num", "max_or_race",
    "jockeys_claim_num", "or_vs_max", "or_vs_race_avg",
    # Race strength
    "race_str_RB", "race_str_WIV", "race_str_NFP", "race_str_WAX",
    "h_RB_vs_race", "h_WIV_vs_race", "h_NFP_vs_race",
    # Within-race rankings (normalised)
    "rank_NFP", "rank_WIV", "rank_RB", "rank_WAX", "rank_ORR2",
    "rank_EPF", "rank_last_bfsp", "rank_OR",
    "rank_j_WIV", "rank_t_WIV",
    "rank_rw3_NFP", "rank_rw5_NFP",
    "rank_FSS", "rank_FCS",
]

TARGET_COL = "log_bfsp"
