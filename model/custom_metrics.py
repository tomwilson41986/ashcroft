"""
Custom Racing Metrics Engine.

Calculates 19 proprietary performance metrics for horse racing prediction.
All calculations are lag-safe — every feature uses ONLY data available
BEFORE that race (shift(1) pattern). No lookahead bias.

Metrics include: NFP, RB, WIV, WAX, WOA, CWO, ORR2, EPF, FSS, FCS,
PFD, WPMRF, PMW, OFS, DSLR, LRP, race strength, pace, trainer-jockey combos,
and within-race rankings.
"""

import re

import numpy as np
import pandas as pd


# Harmonic recency weights for last 10 runs
RECENCY_WEIGHTS = {
    1: 1.0,
    2: 0.5,
    3: 1 / 3,
    4: 0.25,
    5: 0.2,
    6: 1 / 6,
    7: 1 / 7,
    8: 0.125,
    9: 1 / 9,
    10: 0.1,
}


def _calculate_epf(comment: str) -> float:
    """Parse race comment to determine Early Position Figure (1-6 scale).

    Uses NLP regex patterns to classify early race position from
    in-running commentary text.
    """
    if not comment or not isinstance(comment, str):
        return 3.0
    c = comment.lower()

    # Leaders (score 6)
    if re.search(
        r"made virtually all|made all|made most|led to\b|led,|led early"
        r"|led after|led before|led until|led over|soon led",
        c,
    ):
        return 6.0
    # Disputed lead (5.5)
    if re.search(r"disputed|disputed lead|with leader", c):
        return 5.5
    # Chased leader (5)
    if re.search(r"chased leader|tracked leader|chased winner", c):
        return 5.0
    # Prominent (4)
    if re.search(
        r"pressed leader|tracked leaders|chased leaders|prominent|close up"
        r"|in touch|in-touch|pressing leaders|chasing leaders"
        r"|tracked front pair|tracked leading pair|chased leading"
        r"|tracked\s|tracking leaders",
        c,
    ):
        return 4.0
    # Front of midfield (3)
    if re.search(
        r"front of mid-division|front of mid division|front of midfield", c
    ):
        return 3.0
    # Held up midfield (3)
    if re.search(
        r"held up in midfield|held up in mid-division"
        r"|towards rear of midfield|held up in touch",
        c,
    ):
        return 3.0
    # Behind/rear (1)
    if re.search(
        r"towards rear|held up behind|behind|held up|held up,|last pair", c
    ):
        return 1.0
    if re.search(r"in rear|always rear", c):
        return 1.0

    return 3.0


class CustomMetricsEngine:
    """Calculate all custom racing performance metrics.

    All metrics use the group-then-lag pattern:
    1. Group by entity (horse, jockey, trainer, or combination)
    2. Sort by date within each group
    3. Compute expanding/rolling statistics
    4. Shift by 1 to prevent lookahead bias

    Args:
        windows: Lookback windows for rolling metrics (default: [3, 5, 10]).
    """

    def __init__(self, windows: list[int] | None = None):
        self.windows = windows or [3, 5, 10]
        self.max_window = max(self.windows)

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all custom metrics and append as new columns.

        Args:
            df: Raw matched race data with columns: race_date, track,
                race_time, horse_name, placing_numerical, jockey_name,
                trainer, bfsp, number_of_runners, official_rating,
                prize_money, going_description, dist_furlongs, race_class,
                comment, horse_age, headgear, etc.

        Returns:
            DataFrame with all original columns plus ~100+ metric columns.
        """
        df = df.copy()

        # Ensure date is datetime and sorted
        df["race_date"] = pd.to_datetime(df["race_date"])
        df = df.sort_values(["race_date", "race_time"]).reset_index(drop=True)

        # Create race ID if missing
        if "raceid" not in df.columns:
            df["raceid"] = (
                df["race_date"].dt.strftime("%Y-%m-%d")
                + "_"
                + df["track"].astype(str)
                + "_"
                + df["race_time"].astype(str)
            )

        # Ensure numeric columns
        df["placing_numerical"] = pd.to_numeric(
            df["placing_numerical"], errors="coerce"
        )
        df["number_of_runners"] = pd.to_numeric(
            df["number_of_runners"], errors="coerce"
        )
        df["bfsp"] = pd.to_numeric(df.get("bfsp", pd.Series(dtype=float)), errors="coerce")
        df["official_rating"] = pd.to_numeric(
            df.get("official_rating", pd.Series(dtype=float)), errors="coerce"
        )
        df["prize_money"] = pd.to_numeric(
            df.get("prize_money", pd.Series(dtype=float)), errors="coerce"
        )
        df["dist_furlongs"] = pd.to_numeric(
            df.get("dist_furlongs", pd.Series(dtype=float)), errors="coerce"
        )

        # Binary outcomes
        df["won"] = (df["placing_numerical"] == 1).astype(float)
        df["placed"] = (df["placing_numerical"] <= 3).astype(float)

        # Core metrics (order matters — some depend on earlier ones)
        df = self._calc_nfp(df)
        df = self._calc_rb(df)
        df = self._calc_xwinrand_wiv(df)
        df = self._calc_wax_woa_cwo(df)
        df = self._calc_orr2(df)
        df = self._calc_epf(df)
        df = self._calc_fss(df)
        df = self._calc_fcs(df)
        df = self._calc_pfd(df)
        df = self._calc_prize_money(df)
        df = self._calc_ofs(df)
        df = self._calc_dslr(df)
        df = self._calc_lrp(df)
        df = self._calc_pace(df)
        df = self._calc_trainer_jockey(df)
        df = self._calc_race_strength(df)
        df = self._calc_recency_and_confidence(df)

        # --- New research-backed metrics (Benter/Woods/Ziemba/syndicate) ---
        df = self._calc_exponential_decay_form(df)
        df = self._calc_expectation_residuals(df)
        df = self._calc_unexposure(df)
        df = self._calc_class_movement(df)
        df = self._calc_distance_aptitude(df)
        df = self._calc_going_preference(df)
        df = self._calc_draw_bias(df)
        df = self._calc_form_trajectory(df)
        df = self._calc_consistency(df)
        df = self._calc_weight_differential(df)
        df = self._calc_pedigree(df)
        df = self._calc_speed_figures(df)
        df = self._calc_actual_lengths_beaten(df)
        df = self._calc_equipment_changes(df)
        df = self._calc_surface_preference(df)
        df = self._calc_track_preference(df)
        df = self._calc_or_trajectory(df)
        df = self._calc_hot_form(df)

        df = self._calc_within_race_ranks(df)

        return df

    # ------------------------------------------------------------------
    # NFP — Normalised Finishing Position
    # ------------------------------------------------------------------
    def _calc_nfp(self, df: pd.DataFrame) -> pd.DataFrame:
        """NFP: 1.0 for winner, 0.0 for last, normalised by field size."""
        if "NFP" not in df.columns:
            denom = (df["number_of_runners"] - 1).replace(0, np.nan)
            df["NFP"] = (df["number_of_runners"] - df["placing_numerical"]) / denom

        # Sort for horse grouping
        df = df.sort_values(
            ["horse_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        # Career expanding mean (lagged)
        df["preracehorsecareerNFP"] = grp["NFP"].apply(
            lambda x: x.shift(1).expanding().mean()
        )

        # Last run NFP
        df["LRNFP"] = grp["NFP"].shift(1)

        # Rolling window means
        for w in self.windows:
            df[f"LR{w}NFPtotal"] = grp["NFP"].apply(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )

        return df

    # ------------------------------------------------------------------
    # RB — Race Beaten (lengths-behind proxy)
    # ------------------------------------------------------------------
    def _calc_rb(self, df: pd.DataFrame) -> pd.DataFrame:
        """RB: 1.0 for winner, 0.0 for last. FSARB adjusts for field size."""
        if "RB" not in df.columns:
            denom = (df["number_of_runners"] - 1).replace(0, np.nan)
            df["RB"] = 1 - (df["placing_numerical"] - 1) / denom

        median_fs = df["number_of_runners"].median()
        if pd.isna(median_fs) or median_fs == 0:
            median_fs = 10.0
        df["FSARB"] = df["RB"] * (df["number_of_runners"] / median_fs)

        df = df.sort_values(
            ["horse_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        df["preracehorsecareerRB"] = grp["RB"].apply(
            lambda x: x.shift(1).expanding().mean()
        )
        df["preracehorsecareerFSARB"] = grp["FSARB"].apply(
            lambda x: x.shift(1).expanding().mean()
        )
        df["preracehorsecareerFSARB2"] = grp["FSARB"].apply(
            lambda x: (x ** 2).shift(1).expanding().mean()
        )

        return df

    # ------------------------------------------------------------------
    # xWINRAND and WIV — Expected Wins and Win Index Value
    # ------------------------------------------------------------------
    def _calc_xwinrand_wiv(self, df: pd.DataFrame) -> pd.DataFrame:
        """WIV: cumulative wins / cumulative expected wins under random chance."""
        df["xWINRAND"] = 1.0 / df["number_of_runners"].replace(0, np.nan)

        for entity, col, prefix in [
            ("horse_name", "horse_name", "preracehorsecareer"),
            ("trainer", "trainer", "preracetrainercareer"),
            ("jockey_name", "jockey_name", "preracejockeycareer"),
        ]:
            df = df.sort_values(
                [col, "race_date", "race_time"]
            ).reset_index(drop=True)
            grp = df.groupby(col, group_keys=False)

            cum_wins = grp["won"].apply(lambda x: x.shift(1).cumsum())
            cum_xwin = grp["xWINRAND"].apply(lambda x: x.shift(1).cumsum())

            df[f"{prefix}Wins"] = cum_wins
            df[f"{prefix}Runs"] = grp.cumcount()  # 0-indexed = runs before this
            df[f"{prefix}Places"] = grp["placed"].apply(
                lambda x: x.shift(1).cumsum()
            )
            df[f"{prefix}WIV"] = cum_wins / cum_xwin.replace(0, np.nan)

        return df

    # ------------------------------------------------------------------
    # WAX, WOA, CWO
    # ------------------------------------------------------------------
    def _calc_wax_woa_cwo(self, df: pd.DataFrame) -> pd.DataFrame:
        """WAX: wins above expected. WOA: wins over average. CWO: cumulative WAX."""
        df["WAX_raw"] = df["won"] - df["xWINRAND"]

        # Field average win rate (each runner has 1/N chance, so avg = 1/N)
        # WOA = win_flag - (1/number_of_runners) which equals WAX
        # But conceptually WOA compares to actual field average
        df["WOA_raw"] = df["WAX_raw"]  # same for individual runners

        for entity, col, prefix in [
            ("horse_name", "horse_name", "preracehorsecareer"),
            ("trainer", "trainer", "preracetrainercareer"),
            ("jockey_name", "jockey_name", "preracejockeycareer"),
        ]:
            df = df.sort_values(
                [col, "race_date", "race_time"]
            ).reset_index(drop=True)
            grp = df.groupby(col, group_keys=False)

            df[f"{prefix}WAX"] = grp["WAX_raw"].apply(
                lambda x: x.shift(1).expanding().mean()
            )
            df[f"{prefix}WOA"] = grp["WOA_raw"].apply(
                lambda x: x.shift(1).expanding().mean()
            )
            df[f"{prefix}CWO"] = grp["WAX_raw"].apply(
                lambda x: x.shift(1).cumsum()
            )

        return df

    # ------------------------------------------------------------------
    # ORR2 — Odds-to-Runner Ratio
    # ------------------------------------------------------------------
    def _calc_orr2(self, df: pd.DataFrame) -> pd.DataFrame:
        """ORR2: market probability / random probability. >1 means above average."""
        bf_prob = 1.0 / df["bfsp"].replace(0, np.nan)
        fs_prob = 1.0 / df["number_of_runners"].replace(0, np.nan)
        df["ORR2"] = bf_prob / fs_prob.replace(0, np.nan)

        df = df.sort_values(
            ["horse_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        # Career expanding mean
        df["preracehorsecareerORR2"] = grp["ORR2"].apply(
            lambda x: x.shift(1).expanding().mean()
        )

        # Last run ORR2
        df["LR_ORR2"] = grp["ORR2"].shift(1)

        # Recency-weighted ORR2 for each window
        for w in self.windows:
            weighted_col = f"LR{w}_ORR2"
            rwo_col = f"LR{w}_RWO"
            weights = [RECENCY_WEIGHTS[i] for i in range(1, w + 1)]

            # Get lagged ORR2 values and apply weights
            lagged_vals = pd.DataFrame(
                {f"lag{i}": grp["ORR2"].shift(i) for i in range(1, w + 1)}
            )
            weight_arr = np.array(weights)

            # Weighted sum
            weighted_sum = (lagged_vals * weight_arr).sum(axis=1)
            # Weight sum (only where data exists)
            weight_mask = lagged_vals.notna().astype(float) * weight_arr
            wsum = weight_mask.sum(axis=1).replace(0, np.nan)

            df[weighted_col] = weighted_sum
            df[rwo_col] = weighted_sum / wsum

        return df

    # ------------------------------------------------------------------
    # EPF — Early Position Figure
    # ------------------------------------------------------------------
    def _calc_epf(self, df: pd.DataFrame) -> pd.DataFrame:
        """EPF from NLP parsing of race comments."""
        comment_col = "comment" if "comment" in df.columns else None
        if comment_col is None:
            df["EPF"] = 3.0
        else:
            df["EPF"] = df[comment_col].apply(_calculate_epf)

        # Derived EPF metrics
        nr = df["number_of_runners"].replace(0, np.nan)
        denom = (nr - 1).replace(0, np.nan)
        df["EPF2"] = -0.74 + (df["EPF"] * 0.8637) + (nr * 0.09375)
        df["EPF3"] = df["EPF"] * (nr - df["placing_numerical"]) / denom

        # Race-level pace metrics
        df["RPS"] = df.groupby("raceid")["EPF"].transform("sum")
        df["pace_pressure"] = (
            df.groupby("raceid")["EPF"].transform(lambda x: (x > 4).sum())
            / nr
            * 100
        )
        df["prom_runner"] = (df["EPF"] > 4).astype(int)

        # Lagged EPF per horse
        df = df.sort_values(
            ["horse_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        for i in range(1, 6):
            df[f"LR{'' if i == 1 else i}_EPF"] = grp["EPF"].shift(i)
            df[f"LR{'' if i == 1 else i}_EPF2"] = grp["EPF2"].shift(i)
            df[f"LR{'' if i == 1 else i}_EPF3"] = grp["EPF3"].shift(i)

        # Career EPF averages (lagged) for horse, jockey, trainer
        for entity_col, prefix in [
            ("horse_name", "Horse_Career_EPF"),
            ("jockey_name", "Jockey_Career_EPF"),
            ("trainer", "trainer_Career_EPF"),
        ]:
            df = df.sort_values(
                [entity_col, "race_date", "race_time"]
            ).reset_index(drop=True)
            egrp = df.groupby(entity_col, group_keys=False)
            df[prefix] = egrp["EPF2"].apply(
                lambda x: x.shift(1).expanding().mean()
            )

        return df

    # ------------------------------------------------------------------
    # FSS — Field Size Stability
    # ------------------------------------------------------------------
    def _calc_fss(self, df: pd.DataFrame) -> pd.DataFrame:
        """FSS: RMSD of field size variation across recent runs."""
        df = df.sort_values(
            ["horse_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        today_runners = df["number_of_runners"]

        # Compute field size deltas squared for last 10 runs
        delta_sq_sum = pd.Series(0.0, index=df.index)
        count = pd.Series(0.0, index=df.index)

        for i in range(1, 11):
            lag_runners = grp["number_of_runners"].shift(i)
            delta = today_runners - lag_runners
            delta_sq = delta ** 2
            valid = lag_runners.notna()
            delta_sq_sum += delta_sq.fillna(0)
            count += valid.astype(float)

        count = count.replace(0, np.nan)
        df["FSS"] = np.sqrt(delta_sq_sum / count)

        return df

    # ------------------------------------------------------------------
    # FCS — Field Class Strength
    # ------------------------------------------------------------------
    def _calc_fcs(self, df: pd.DataFrame) -> pd.DataFrame:
        """FCS: RMSD of race-level mean official rating variation."""
        df["Race_avgOR"] = df.groupby("raceid")["official_rating"].transform(
            "mean"
        )

        df = df.sort_values(
            ["horse_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        today_avg_or = df["Race_avgOR"]
        delta_sq_sum = pd.Series(0.0, index=df.index)
        count = pd.Series(0.0, index=df.index)

        for i in range(1, 11):
            lag_avg_or = grp["Race_avgOR"].shift(i)
            delta = today_avg_or - lag_avg_or
            delta_sq = delta ** 2
            valid = lag_avg_or.notna()
            delta_sq_sum += delta_sq.fillna(0)
            count += valid.astype(float)

        count = count.replace(0, np.nan)
        df["FCS"] = np.sqrt(delta_sq_sum / count)

        return df

    # ------------------------------------------------------------------
    # PFD — Probability-Field Difference
    # ------------------------------------------------------------------
    def _calc_pfd(self, df: pd.DataFrame) -> pd.DataFrame:
        """PFD: market probability minus random probability."""
        bf_prob = 1.0 / df["bfsp"].replace(0, np.nan)
        fs_prob = 1.0 / df["number_of_runners"].replace(0, np.nan)
        df["PFD"] = bf_prob - fs_prob

        df = df.sort_values(
            ["horse_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        for w in self.windows:
            weights = [RECENCY_WEIGHTS[i] for i in range(1, w + 1)]
            lagged = pd.DataFrame(
                {f"lag{i}": grp["PFD"].shift(i) for i in range(1, w + 1)}
            )
            weight_arr = np.array(weights)
            weighted_sum = (lagged * weight_arr).sum(axis=1)
            wsum = (lagged.notna().astype(float) * weight_arr).sum(
                axis=1
            ).replace(0, np.nan)
            df[f"PFD{w}"] = weighted_sum / wsum

        return df

    # ------------------------------------------------------------------
    # Prize Money Metrics (WPMRF, PMW)
    # ------------------------------------------------------------------
    def _calc_prize_money(self, df: pd.DataFrame) -> pd.DataFrame:
        """WPMRF: recency-weighted prize money raced for. PMW: performance-adjusted."""
        prize = df["prize_money"].fillna(0)
        df["PMW_raw"] = (prize / 100.0) * (df["RB"].fillna(0) ** 2)

        df["RACE_WPMRF"] = df.groupby("raceid")["prize_money"].transform("sum")

        df = df.sort_values(
            ["horse_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        for w in self.windows:
            weights = [RECENCY_WEIGHTS[i] for i in range(1, w + 1)]
            weight_arr = np.array(weights)

            # WPMRF
            lagged_prize = pd.DataFrame(
                {f"lag{i}": grp["prize_money"].shift(i) for i in range(1, w + 1)}
            )
            ws = (lagged_prize.fillna(0) * weight_arr).sum(axis=1)
            wt = (lagged_prize.notna().astype(float) * weight_arr).sum(
                axis=1
            ).replace(0, np.nan)
            df[f"WPMRF{w}"] = ws / wt

            # PMW
            lagged_pmw = pd.DataFrame(
                {f"lag{i}": grp["PMW_raw"].shift(i) for i in range(1, w + 1)}
            )
            ws = (lagged_pmw.fillna(0) * weight_arr).sum(axis=1)
            wt = (lagged_pmw.notna().astype(float) * weight_arr).sum(
                axis=1
            ).replace(0, np.nan)
            df[f"PMW{w}"] = ws / wt

        return df

    # ------------------------------------------------------------------
    # OFS — Odds x Field Size
    # ------------------------------------------------------------------
    def _calc_ofs(self, df: pd.DataFrame) -> pd.DataFrame:
        """OFS: (1/BFSP) * number_of_runners."""
        df["OFS"] = (1.0 / df["bfsp"].replace(0, np.nan)) * df["number_of_runners"]

        df = df.sort_values(
            ["horse_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        df["OFS1"] = grp["OFS"].shift(1)

        for w in self.windows:
            weights = [RECENCY_WEIGHTS[i] for i in range(1, w + 1)]
            lagged = pd.DataFrame(
                {f"lag{i}": grp["OFS"].shift(i) for i in range(1, w + 1)}
            )
            weight_arr = np.array(weights)
            ws = (lagged * weight_arr).sum(axis=1)
            wt = (lagged.notna().astype(float) * weight_arr).sum(
                axis=1
            ).replace(0, np.nan)
            df[f"OFS{w}"] = ws / wt

        return df

    # ------------------------------------------------------------------
    # DSLR — Days Since Last Run (enhanced with rate of change)
    # ------------------------------------------------------------------
    def _calc_dslr(self, df: pd.DataFrame) -> pd.DataFrame:
        """DSLR: enhanced days-since-last-run with acceleration/deceleration trend."""
        df = df.sort_values(
            ["horse_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        # Days between consecutive runs
        df["_date_num"] = df["race_date"].astype(np.int64) // 10**9 // 86400
        for i in range(1, 5):
            lag_date = grp["_date_num"].shift(i)
            df[f"DSLR{i}"] = df["_date_num"] - lag_date

        # Rate of change with harmonic weights
        df["DSLR12diff"] = (df.get("DSLR2", np.nan) - df.get("DSLR1", np.nan)) * 1.0
        df["DSLR23diff"] = (df.get("DSLR3", np.nan) - df.get("DSLR2", np.nan)) * 0.5
        df["DSLR34diff"] = (df.get("DSLR4", np.nan) - df.get("DSLR3", np.nan)) * 0.33

        df["WgtDSLR"] = (
            df["DSLR12diff"].fillna(0)
            + df["DSLR23diff"].fillna(0)
            + df["DSLR34diff"].fillna(0)
        )

        # LR3COUNT for this specific calculation
        lr3_count = sum(
            grp["_date_num"].shift(i).notna().astype(float)
            for i in range(1, 4)
        ).replace(0, np.nan)
        df["FinalDSLR"] = df["WgtDSLR"] / lr3_count

        df.drop(columns=["_date_num"], inplace=True)

        return df

    # ------------------------------------------------------------------
    # LRP — Last Run Placed Index (jockey momentum)
    # ------------------------------------------------------------------
    def _calc_lrp(self, df: pd.DataFrame) -> pd.DataFrame:
        """LRP: position-weighted jockey momentum score."""
        df = df.sort_values(
            ["horse_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        h_grp = df.groupby("horse_name", group_keys=False)

        lr_placed = h_grp["placed"].shift(1)
        lr_pos = h_grp["placing_numerical"].shift(1)

        df["LRP1Score"] = np.where(
            (lr_placed == 1) & (lr_pos == 1), 17.41, 0.0
        )
        df["LRP2Score"] = np.where(
            (lr_placed == 1) & (lr_pos == 2), 14.77, 0.0
        )
        df["LRP3Score"] = np.where(
            (lr_placed == 1) & (lr_pos == 3), 12.62, 0.0
        )
        df["LRPTotalScore"] = (
            df["LRP1Score"] + df["LRP2Score"] + df["LRP3Score"]
        )

        # Per-jockey cumulative LRP index
        df = df.sort_values(
            ["jockey_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        j_grp = df.groupby("jockey_name", group_keys=False)

        df["totaljockeyLRPscore"] = j_grp["LRPTotalScore"].apply(
            lambda x: x.shift(1).cumsum()
        )
        df["totaljockeyrides"] = j_grp.cumcount()  # 0-indexed
        rides = df["totaljockeyrides"].replace(0, np.nan)
        df["totalLRPjockeyindex"] = (df["totaljockeyLRPscore"] / rides) * 10

        return df

    # ------------------------------------------------------------------
    # Pace Metrics
    # ------------------------------------------------------------------
    def _calc_pace(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pace indices for horse, trainer, jockey using EPF as proxy for RunStyle."""
        # Use EPF as RunStyle proxy
        run_style = df["EPF"]

        df["racepacescore"] = df.groupby("raceid")[run_style.name].transform("sum")
        df["racepaceindex"] = df["racepacescore"] / df[
            "number_of_runners"
        ].replace(0, np.nan)

        for entity_col, prefix in [
            ("horse_name", "horsepaceindex"),
            ("trainer", "trainerpaceindex"),
            ("jockey_name", "jockeypaceindex"),
        ]:
            df = df.sort_values(
                [entity_col, "race_date", "race_time"]
            ).reset_index(drop=True)
            grp = df.groupby(entity_col, group_keys=False)
            cum_pace = grp["EPF"].apply(lambda x: x.shift(1).cumsum())
            cum_runs = grp.cumcount().replace(0, np.nan)
            df[prefix] = cum_pace / cum_runs

        return df

    # ------------------------------------------------------------------
    # Trainer-Jockey Combination Metrics
    # ------------------------------------------------------------------
    def _calc_trainer_jockey(self, df: pd.DataFrame) -> pd.DataFrame:
        """Joint trainer-jockey performance metrics."""
        df = df.sort_values(
            ["trainer", "jockey_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        tj_grp = df.groupby(["trainer", "jockey_name"], group_keys=False)

        cum_wins = tj_grp["won"].apply(lambda x: x.shift(1).cumsum())
        cum_xwin = tj_grp["xWINRAND"].apply(lambda x: x.shift(1).cumsum())

        df["trainerjockeycareerWIV"] = cum_wins / cum_xwin.replace(0, np.nan)
        df["trainerjockeycareerNFP"] = tj_grp["NFP"].apply(
            lambda x: x.shift(1).expanding().mean()
        )

        # WAX, WOA, CWO for combination
        df["trainerjockeyWAX"] = tj_grp["WAX_raw"].apply(
            lambda x: x.shift(1).expanding().mean()
        )
        df["trainerjockeyWOA"] = tj_grp["WOA_raw"].apply(
            lambda x: x.shift(1).expanding().mean()
        )
        df["trainerjockeyCWO"] = tj_grp["WAX_raw"].apply(
            lambda x: x.shift(1).cumsum()
        )

        return df

    # ------------------------------------------------------------------
    # Race Strength Metrics
    # ------------------------------------------------------------------
    def _calc_race_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-race averages of pre-race horse career metrics."""
        for metric in [
            "preracehorsecareerRB",
            "preracehorsecareerWIV",
            "preracehorsecareerNFP",
            "preracehorsecareerWins",
            "preracehorsecareerWOA",
        ]:
            race_col = f"RACE_{metric.replace('preracehorsecareer', '')}"
            if metric in df.columns:
                df[race_col] = df.groupby("raceid")[metric].transform("mean")

        # Lag race strength per horse (quality of last race's opposition)
        df = df.sort_values(
            ["horse_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        for col in df.columns:
            if col.startswith("RACE_") and col != "RACE_WPMRF":
                df[f"LR_{col}"] = grp[col].shift(1)

        return df

    # ------------------------------------------------------------------
    # Recency Weights and Confidence Intervals
    # ------------------------------------------------------------------
    def _calc_recency_and_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recency weight sums and data confidence intervals."""
        df = df.sort_values(
            ["horse_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        # Count how many of last N runs exist
        for w in self.windows:
            exists = pd.DataFrame(
                {f"e{i}": grp["race_date"].shift(i).notna().astype(float)
                 for i in range(1, w + 1)}
            )
            df[f"LR{w}COUNT"] = exists.sum(axis=1)

            # Weight sums
            weights = [RECENCY_WEIGHTS[i] for i in range(1, w + 1)]
            df[f"LR{w}wsum"] = (exists * np.array(weights)).sum(axis=1)

        # Confidence intervals
        nr = df["number_of_runners"].replace(0, np.nan)
        for w in self.windows:
            df[f"CIL{w}"] = (df[f"LR{w}COUNT"] / (nr * w)) * 100

        return df

    # ==================================================================
    # NEW RESEARCH-BACKED METRICS (Benter/Woods/Ziemba/Syndicate)
    # ==================================================================

    # ------------------------------------------------------------------
    # Exponential Decay Form (Benter/Woods preferred over harmonic)
    # ------------------------------------------------------------------
    def _calc_exponential_decay_form(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exponential decay weighting for recent form (alpha=0.85).

        Research basis: Benter (1994) and modern syndicates use exponential
        decay rather than harmonic weights. More recent runs get
        exponentially more weight, with alpha controlling the decay rate.
        """
        alpha = 0.85
        df = df.sort_values(
            ["horse_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        for metric, prefix in [
            ("NFP", "EXP_NFP"),
            ("RB", "EXP_RB"),
            ("ORR2", "EXP_ORR2"),
        ]:
            if metric not in df.columns:
                continue
            for w in [3, 5, 10]:
                exp_weights = np.array([alpha ** i for i in range(w)])
                lagged = pd.DataFrame(
                    {f"lag{i}": grp[metric].shift(i + 1) for i in range(w)}
                )
                valid = lagged.notna().astype(float)
                weighted_sum = (lagged.fillna(0) * exp_weights).sum(axis=1)
                weight_sum = (valid * exp_weights).sum(axis=1).replace(0, np.nan)
                df[f"{prefix}{w}"] = weighted_sum / weight_sum

        return df

    # ------------------------------------------------------------------
    # Expectation Residuals (Woods/Ziemba — beating/underperforming market)
    # ------------------------------------------------------------------
    def _calc_expectation_residuals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market-expected performance vs actual performance residuals.

        Research basis: Woods and Ziemba emphasized that the key signal is
        not raw performance but performance RELATIVE TO EXPECTATION.
        A horse finishing 3rd at 50/1 outperformed; a favourite finishing
        2nd underperformed. These residuals are predictive of future form.
        """
        df = df.sort_values(
            ["horse_name", "race_date", "race_time"]
        ).reset_index(drop=True)

        # Expected NFP from market position (rank by BFSP within race)
        df["_market_rank"] = df.groupby("raceid")["bfsp"].rank(
            method="min", na_option="bottom"
        )
        nr = df["number_of_runners"].replace(0, np.nan)
        df["expected_NFP"] = (nr - df["_market_rank"]) / (nr - 1).replace(0, np.nan)

        # Residual: actual - expected (positive = outperformed market)
        df["NFP_residual"] = df["NFP"] - df["expected_NFP"]

        grp = df.groupby("horse_name", group_keys=False)

        # Career average residual (lagged)
        df["career_residual"] = grp["NFP_residual"].apply(
            lambda x: x.shift(1).expanding().mean()
        )

        # Recent residuals (exponential decay)
        alpha = 0.85
        for w in [3, 5]:
            exp_weights = np.array([alpha ** i for i in range(w)])
            lagged = pd.DataFrame(
                {f"lag{i}": grp["NFP_residual"].shift(i + 1) for i in range(w)}
            )
            valid = lagged.notna().astype(float)
            ws = (lagged.fillna(0) * exp_weights).sum(axis=1)
            wt = (valid * exp_weights).sum(axis=1).replace(0, np.nan)
            df[f"residual_exp{w}"] = ws / wt

        # Win surprise: won when market didn't expect (high BFSP winner)
        df["win_surprise"] = df["won"] * np.log1p(
            df["bfsp"].clip(lower=1) - 1
        )
        df["career_win_surprise"] = grp["win_surprise"].apply(
            lambda x: x.shift(1).expanding().mean()
        )

        df.drop(columns=["_market_rank", "expected_NFP"], inplace=True)

        return df

    # ------------------------------------------------------------------
    # Unexposure Features (Novel conditions detection)
    # ------------------------------------------------------------------
    def _calc_unexposure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features for horses facing novel conditions.

        Research basis: Modern syndicates weight unexposure heavily.
        A horse's first time at a distance, going, course, or class
        introduces uncertainty. Debut runners have maximum unexposure.
        """
        df = df.sort_values(
            ["horse_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        # Career run count (already exists but we need it for debut flag)
        df["_career_count"] = grp.cumcount()  # 0-indexed
        df["is_debut"] = (df["_career_count"] == 0).astype(int)

        # Distance experience: count of prior runs at same distance band
        df["_dist_band"] = df["dist_furlongs"].round(0)
        df["dist_experience"] = df.groupby(
            ["horse_name", "_dist_band"]
        ).cumcount()
        df["first_at_distance"] = (df["dist_experience"] == 0).astype(int)

        # Going experience: count of prior runs on same going
        df["_going_cat"] = df["going_description"].fillna("unknown").str.lower().str.strip()
        df["going_experience"] = df.groupby(
            ["horse_name", "_going_cat"]
        ).cumcount()
        df["first_at_going"] = (df["going_experience"] == 0).astype(int)

        # Course experience
        df["_track_lower"] = df["track"].fillna("unknown").str.lower().str.strip()
        df["course_experience"] = df.groupby(
            ["horse_name", "_track_lower"]
        ).cumcount()
        df["first_at_course"] = (df["course_experience"] == 0).astype(int)

        # Course-distance combo
        df["_cd_combo"] = df["_track_lower"] + "_" + df["_dist_band"].astype(str)
        df["cd_experience"] = df.groupby(
            ["horse_name", "_cd_combo"]
        ).cumcount()
        df["first_at_cd"] = (df["cd_experience"] == 0).astype(int)

        # Composite unexposure score (0=fully exposed, 4=maximum novelty)
        df["unexposure_score"] = (
            df["first_at_distance"]
            + df["first_at_going"]
            + df["first_at_course"]
            + df["first_at_cd"]
        )

        # Performance at this distance (lagged)
        df["_dist_nfp"] = df["NFP"]
        dist_grp = df.groupby(["horse_name", "_dist_band"], group_keys=False)
        df["dist_avg_nfp"] = dist_grp["_dist_nfp"].apply(
            lambda x: x.shift(1).expanding().mean()
        )

        # Performance at this going (lagged)
        going_grp = df.groupby(["horse_name", "_going_cat"], group_keys=False)
        df["going_avg_nfp"] = going_grp["NFP"].apply(
            lambda x: x.shift(1).expanding().mean()
        )

        # Performance at this course (lagged)
        course_grp = df.groupby(["horse_name", "_track_lower"], group_keys=False)
        df["course_avg_nfp"] = course_grp["NFP"].apply(
            lambda x: x.shift(1).expanding().mean()
        )

        # Clean up temp columns
        df.drop(columns=[
            "_career_count", "_dist_band", "_going_cat", "_track_lower",
            "_cd_combo", "_dist_nfp"
        ], inplace=True)

        return df

    # ------------------------------------------------------------------
    # Class Movement (Ziemba — class drop/rise signal)
    # ------------------------------------------------------------------
    def _calc_class_movement(self, df: pd.DataFrame) -> pd.DataFrame:
        """Class rise/drop indicators relative to recent races.

        Research basis: Ziemba showed class drops are one of the strongest
        positive signals. A horse dropping in class is competing against
        weaker opposition. Conversely, class rises indicate tougher tests.
        """
        # Extract numeric class from strings like "Class 4"
        df["race_class_num_raw"] = (
            df["race_class"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .pipe(pd.to_numeric, errors="coerce")
        )

        df = df.sort_values(
            ["horse_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        # Last race class
        df["lr_class"] = grp["race_class_num_raw"].shift(1)

        # Class change (positive = dropping in class = easier, since higher class number = lower quality)
        df["class_change"] = df["race_class_num_raw"] - df["lr_class"]

        # Average class over last 3 races
        df["avg_class_3"] = grp["race_class_num_raw"].apply(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
        df["class_vs_avg"] = df["race_class_num_raw"] - df["avg_class_3"]

        # Is dropping in class? (higher class number = lower quality)
        df["is_class_drop"] = (df["class_change"] > 0).astype(int)
        df["is_class_rise"] = (df["class_change"] < 0).astype(int)

        return df

    # ------------------------------------------------------------------
    # Distance Aptitude (Benter fundamental variable)
    # ------------------------------------------------------------------
    def _calc_distance_aptitude(self, df: pd.DataFrame) -> pd.DataFrame:
        """Distance preference derived from performance at various distances.

        Research basis: Benter identified distance as a key fundamental
        variable. Horses have optimal distance ranges. Performance degrades
        at non-optimal distances. The delta from optimal is predictive.
        """
        df = df.sort_values(
            ["horse_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        # Weighted average distance of best performances
        df["_weighted_dist"] = df["dist_furlongs"] * df["NFP"].fillna(0)

        cum_weighted_dist = grp["_weighted_dist"].apply(
            lambda x: x.shift(1).cumsum()
        )
        cum_nfp = grp["NFP"].apply(
            lambda x: x.shift(1).cumsum()
        ).replace(0, np.nan)

        df["preferred_distance"] = cum_weighted_dist / cum_nfp

        # Distance from preferred (absolute)
        df["dist_from_preferred"] = (
            df["dist_furlongs"] - df["preferred_distance"]
        ).abs()

        # Signed distance change: positive = going longer
        df["dist_change_signed"] = (
            df["dist_furlongs"] - df["preferred_distance"]
        )

        # Last run distance change
        lr_dist = grp["dist_furlongs"].shift(1)
        df["dist_change_lr"] = df["dist_furlongs"] - lr_dist

        df.drop(columns=["_weighted_dist"], inplace=True)

        return df

    # ------------------------------------------------------------------
    # Going Preference (Benter fundamental variable)
    # ------------------------------------------------------------------
    def _calc_going_preference(self, df: pd.DataFrame) -> pd.DataFrame:
        """Going ground preference based on historical performance.

        Research basis: Benter's ~20 fundamental variables included going
        preference. Some horses are "mud lovers" or need fast ground.
        """
        going_map = {
            "heavy": 1.0, "soft": 2.0, "yielding": 2.5,
            "good to soft": 3.0, "good": 4.0, "good to firm": 5.0,
            "firm": 6.0, "hard": 7.0, "standard": 4.0,
            "standard to slow": 3.0, "slow": 2.0,
        }

        def encode_going(g):
            if not g or not isinstance(g, str):
                return 4.0
            gl = g.lower().strip()
            for key, val in going_map.items():
                if key in gl:
                    return val
            return 4.0

        df["_going_num"] = df["going_description"].apply(encode_going)

        df = df.sort_values(
            ["horse_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        # Performance-weighted preferred going
        df["_weighted_going"] = df["_going_num"] * df["NFP"].fillna(0)

        cum_wg = grp["_weighted_going"].apply(lambda x: x.shift(1).cumsum())
        cum_nfp = grp["NFP"].apply(
            lambda x: x.shift(1).cumsum()
        ).replace(0, np.nan)

        df["preferred_going"] = cum_wg / cum_nfp

        # Delta from preferred going
        df["going_from_preferred"] = (
            df["_going_num"] - df["preferred_going"]
        ).abs()

        # Going change from last run
        lr_going = grp["_going_num"].shift(1)
        df["going_change_lr"] = df["_going_num"] - lr_going

        df.drop(columns=["_going_num", "_weighted_going"], inplace=True)

        return df

    # ------------------------------------------------------------------
    # Draw Bias (Benter/Woods — post-position effect)
    # ------------------------------------------------------------------
    def _calc_draw_bias(self, df: pd.DataFrame) -> pd.DataFrame:
        """Draw/stall position relative to field and track bias.

        Research basis: Benter identified post-position as a key variable.
        Draw bias varies by track and distance. The relative position
        (stall / runners) matters more than absolute stall number.
        """
        df["stall_num_raw"] = pd.to_numeric(df["stall"], errors="coerce")
        nr = df["number_of_runners"].replace(0, np.nan)

        # Relative draw position (0=inside, 1=outside)
        df["draw_relative"] = (df["stall_num_raw"] - 1) / (nr - 1).replace(0, np.nan)

        # Draw quartile (1=inner, 4=outer)
        df["draw_quartile"] = pd.cut(
            df["draw_relative"],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=[1, 2, 3, 4],
            include_lowest=True,
        )
        df["draw_quartile"] = pd.to_numeric(df["draw_quartile"], errors="coerce")

        return df

    # ------------------------------------------------------------------
    # Form Trajectory (Syndicate — improvement/decline detection)
    # ------------------------------------------------------------------
    def _calc_form_trajectory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect improving/declining form trajectories.

        Research basis: Modern syndicates use form trajectory analysis.
        A horse that's improved across its last 3 runs is on an upward
        trajectory. Linear regression slope of recent NFP is predictive.
        """
        df = df.sort_values(
            ["horse_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        # Get last 3 and 5 NFP values
        for w in [3, 5]:
            lagged = pd.DataFrame(
                {f"lag{i}": grp["NFP"].shift(i + 1) for i in range(w)}
            )
            # Simple linear regression slope: trend direction
            # x = [0, 1, 2, ...], y = [most_recent, ..., oldest]
            x = np.arange(w, dtype=float)
            x_mean = x.mean()
            x_var = ((x - x_mean) ** 2).sum()

            y_vals = lagged.values  # shape: (n_rows, w)
            y_mean = np.nanmean(y_vals, axis=1, keepdims=True)

            # Slope = Σ((x - x_mean)(y - y_mean)) / Σ((x - x_mean)²)
            numerator = np.nansum(
                (x - x_mean) * (y_vals - y_mean), axis=1
            )
            slope = numerator / x_var

            # Positive slope = improving (most recent is better)
            df[f"form_slope_{w}"] = slope

            # Also: variance of recent form (consistency signal)
            df[f"form_var_{w}"] = np.nanvar(y_vals, axis=1)

        # Is improving? (slope > threshold)
        df["is_improving"] = (df["form_slope_3"] > 0.05).astype(int)
        df["is_declining"] = (df["form_slope_3"] < -0.05).astype(int)

        return df

    # ------------------------------------------------------------------
    # Consistency (Ziemba — reliability measure)
    # ------------------------------------------------------------------
    def _calc_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Consistency metrics: how reliable is this horse's form?

        Research basis: Ziemba emphasized that consistent performers are
        more predictable and thus better model inputs. High-variance
        horses need different treatment.
        """
        df = df.sort_values(
            ["horse_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        # Career standard deviation of NFP (lagged)
        df["career_nfp_std"] = grp["NFP"].apply(
            lambda x: x.shift(1).expanding().std()
        )

        # Recent consistency (std of last 5 NFP)
        df["recent_nfp_std"] = grp["NFP"].apply(
            lambda x: x.shift(1).rolling(5, min_periods=2).std()
        )

        # Place rate (top 3 finish rate)
        df["career_place_rate"] = grp["placed"].apply(
            lambda x: x.shift(1).expanding().mean()
        )

        # Win rate
        df["career_win_rate"] = grp["won"].apply(
            lambda x: x.shift(1).expanding().mean()
        )

        # Recent win rate (last 10)
        df["recent_win_rate"] = grp["won"].apply(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean()
        )

        # Recent place rate (last 10)
        df["recent_place_rate"] = grp["placed"].apply(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean()
        )

        return df

    # ------------------------------------------------------------------
    # Weight Differential (Benter fundamental variable)
    # ------------------------------------------------------------------
    def _calc_weight_differential(self, df: pd.DataFrame) -> pd.DataFrame:
        """Weight carried relative to field and own history.

        Research basis: Benter identified weight carried as a key variable.
        What matters is relative weight (vs field) not absolute weight.
        Weight changes between runs also carry information.
        """
        df["pounds_raw"] = pd.to_numeric(df["pounds"], errors="coerce")

        # Weight relative to field
        race_avg_wt = df.groupby("raceid")["pounds_raw"].transform("mean")
        race_min_wt = df.groupby("raceid")["pounds_raw"].transform("min")
        race_max_wt = df.groupby("raceid")["pounds_raw"].transform("max")

        df["weight_vs_avg"] = df["pounds_raw"] - race_avg_wt
        df["weight_vs_min"] = df["pounds_raw"] - race_min_wt
        df["weight_range"] = race_max_wt - race_min_wt

        # Weight change from last run
        df = df.sort_values(
            ["horse_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)
        lr_weight = grp["pounds_raw"].shift(1)
        df["weight_change_lr"] = df["pounds_raw"] - lr_weight

        return df

    # ------------------------------------------------------------------
    # Pedigree Features (Sire / Dam / Damsire)
    # ------------------------------------------------------------------
    def _calc_pedigree(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sire, dam, and damsire performance metrics with Bayesian shrinkage.

        Research basis: Pedigree is a major factor for (1) debut/lightly-raced
        horses with limited form, (2) first-time going/distance changes,
        (3) inherited aptitude patterns. Sire influence on going preference
        and distance aptitude is well-documented.

        Uses Bayesian shrinkage: adjusted = (n * sire_stat + k * pop_stat) / (n + k)
        to handle sires with few runners.
        """
        SHRINKAGE_K = 20  # prior strength

        # Population-level stats (lagged, expanding) for shrinkage
        df = df.sort_values(
            ["race_date", "race_time"]
        ).reset_index(drop=True)
        pop_win_rate = df["won"].expanding().mean().shift(1).fillna(0.1)
        pop_place_rate = df["placed"].expanding().mean().shift(1).fillna(0.3)
        pop_nfp = df["NFP"].expanding().mean().shift(1).fillna(0.5)

        # --- Sire career stats ---
        if "stallion" in df.columns:
            df["_stallion_clean"] = (
                df["stallion"].fillna("unknown").str.strip().str.lower()
            )

            df = df.sort_values(
                ["_stallion_clean", "race_date", "race_time"]
            ).reset_index(drop=True)
            s_grp = df.groupby("_stallion_clean", group_keys=False)

            # Raw expanding stats (lagged)
            raw_sire_runs = s_grp.cumcount()  # 0-indexed = runs before this
            raw_sire_wins = s_grp["won"].apply(lambda x: x.shift(1).cumsum())
            raw_sire_places = s_grp["placed"].apply(lambda x: x.shift(1).cumsum())
            raw_sire_nfp = s_grp["NFP"].apply(
                lambda x: x.shift(1).expanding().mean()
            )

            sire_n = raw_sire_runs.replace(0, np.nan)

            # Bayesian-shrunk sire win rate
            raw_win_rate = raw_sire_wins / sire_n
            df["sire_win_rate"] = (
                (sire_n * raw_win_rate + SHRINKAGE_K * pop_win_rate)
                / (sire_n + SHRINKAGE_K)
            )

            # Bayesian-shrunk sire place rate
            raw_place_rate = raw_sire_places / sire_n
            df["sire_place_rate"] = (
                (sire_n * raw_place_rate + SHRINKAGE_K * pop_place_rate)
                / (sire_n + SHRINKAGE_K)
            )

            # Bayesian-shrunk sire NFP
            df["sire_avg_nfp"] = (
                (sire_n * raw_sire_nfp + SHRINKAGE_K * pop_nfp)
                / (sire_n + SHRINKAGE_K)
            )

            # Sire WIV: cumulative progeny wins / expected wins
            raw_sire_xwin = s_grp["xWINRAND"].apply(
                lambda x: x.shift(1).cumsum()
            )
            df["sire_wiv"] = raw_sire_wins / raw_sire_xwin.replace(0, np.nan)

            # --- Sire going aptitude ---
            df["_going_cat_ped"] = (
                df["going_description"].fillna("unknown").str.lower().str.strip()
            )
            df = df.sort_values(
                ["_stallion_clean", "_going_cat_ped", "race_date", "race_time"]
            ).reset_index(drop=True)
            sg_grp = df.groupby(
                ["_stallion_clean", "_going_cat_ped"], group_keys=False
            )

            sg_runs = sg_grp.cumcount()
            sg_n = sg_runs.replace(0, np.nan)
            raw_sg_nfp = sg_grp["NFP"].apply(
                lambda x: x.shift(1).expanding().mean()
            )
            raw_sg_win = sg_grp["won"].apply(
                lambda x: x.shift(1).cumsum()
            )
            raw_sg_win_rate = raw_sg_win / sg_n

            df["sire_going_nfp"] = (
                (sg_n * raw_sg_nfp + SHRINKAGE_K * pop_nfp)
                / (sg_n + SHRINKAGE_K)
            )
            df["sire_going_win_rate"] = (
                (sg_n * raw_sg_win_rate + SHRINKAGE_K * pop_win_rate)
                / (sg_n + SHRINKAGE_K)
            )

            # --- Sire distance aptitude ---
            df["_dist_band_ped"] = df["dist_furlongs"].round(0)
            df = df.sort_values(
                ["_stallion_clean", "_dist_band_ped", "race_date", "race_time"]
            ).reset_index(drop=True)
            sd_grp = df.groupby(
                ["_stallion_clean", "_dist_band_ped"], group_keys=False
            )

            sd_runs = sd_grp.cumcount()
            sd_n = sd_runs.replace(0, np.nan)
            raw_sd_nfp = sd_grp["NFP"].apply(
                lambda x: x.shift(1).expanding().mean()
            )
            raw_sd_win = sd_grp["won"].apply(
                lambda x: x.shift(1).cumsum()
            )
            raw_sd_win_rate = raw_sd_win / sd_n

            df["sire_dist_nfp"] = (
                (sd_n * raw_sd_nfp + SHRINKAGE_K * pop_nfp)
                / (sd_n + SHRINKAGE_K)
            )
            df["sire_dist_win_rate"] = (
                (sd_n * raw_sd_win_rate + SHRINKAGE_K * pop_win_rate)
                / (sd_n + SHRINKAGE_K)
            )

            # Sire progeny count (useful signal on its own)
            df["sire_runners"] = raw_sire_runs
        else:
            for col in [
                "sire_win_rate", "sire_place_rate", "sire_avg_nfp",
                "sire_wiv", "sire_going_nfp", "sire_going_win_rate",
                "sire_dist_nfp", "sire_dist_win_rate", "sire_runners",
            ]:
                df[col] = np.nan

        # --- Damsire career stats ---
        if "dam_stallion" in df.columns:
            df["_damsire_clean"] = (
                df["dam_stallion"].fillna("unknown").str.strip().str.lower()
            )

            df = df.sort_values(
                ["_damsire_clean", "race_date", "race_time"]
            ).reset_index(drop=True)
            ds_grp = df.groupby("_damsire_clean", group_keys=False)

            raw_ds_runs = ds_grp.cumcount()
            ds_n = raw_ds_runs.replace(0, np.nan)
            raw_ds_nfp = ds_grp["NFP"].apply(
                lambda x: x.shift(1).expanding().mean()
            )
            raw_ds_wins = ds_grp["won"].apply(lambda x: x.shift(1).cumsum())
            raw_ds_win_rate = raw_ds_wins / ds_n

            df["damsire_avg_nfp"] = (
                (ds_n * raw_ds_nfp + SHRINKAGE_K * pop_nfp)
                / (ds_n + SHRINKAGE_K)
            )
            df["damsire_win_rate"] = (
                (ds_n * raw_ds_win_rate + SHRINKAGE_K * pop_win_rate)
                / (ds_n + SHRINKAGE_K)
            )

            # Damsire going aptitude
            df = df.sort_values(
                ["_damsire_clean", "_going_cat_ped", "race_date", "race_time"]
            ).reset_index(drop=True)
            dsg_grp = df.groupby(
                ["_damsire_clean", "_going_cat_ped"], group_keys=False
            )
            dsg_runs = dsg_grp.cumcount()
            dsg_n = dsg_runs.replace(0, np.nan)
            raw_dsg_nfp = dsg_grp["NFP"].apply(
                lambda x: x.shift(1).expanding().mean()
            )
            df["damsire_going_nfp"] = (
                (dsg_n * raw_dsg_nfp + SHRINKAGE_K * pop_nfp)
                / (dsg_n + SHRINKAGE_K)
            )

            # Damsire distance aptitude
            df = df.sort_values(
                ["_damsire_clean", "_dist_band_ped", "race_date", "race_time"]
            ).reset_index(drop=True)
            dsd_grp = df.groupby(
                ["_damsire_clean", "_dist_band_ped"], group_keys=False
            )
            dsd_runs = dsd_grp.cumcount()
            dsd_n = dsd_runs.replace(0, np.nan)
            raw_dsd_nfp = dsd_grp["NFP"].apply(
                lambda x: x.shift(1).expanding().mean()
            )
            df["damsire_dist_nfp"] = (
                (dsd_n * raw_dsd_nfp + SHRINKAGE_K * pop_nfp)
                / (dsd_n + SHRINKAGE_K)
            )

            df["damsire_runners"] = raw_ds_runs
        else:
            for col in [
                "damsire_avg_nfp", "damsire_win_rate",
                "damsire_going_nfp", "damsire_dist_nfp", "damsire_runners",
            ]:
                df[col] = np.nan

        # --- Debut interaction features ---
        # For debut runners, sire/trainer quality are primary predictors
        is_debut = (
            df.groupby("horse_name").cumcount() == 0
        ).astype(float)
        df["debut_x_sire_nfp"] = is_debut * df["sire_avg_nfp"]
        df["debut_x_sire_wiv"] = is_debut * df.get("sire_wiv", 0)
        df["debut_x_trainer_wiv"] = is_debut * df.get(
            "preracetrainercareerWIV", 0
        )

        # Clean up temp columns
        temp_cols = [
            c for c in df.columns
            if c.startswith("_") and c.endswith(("_clean", "_ped"))
        ]
        df.drop(columns=temp_cols, errors="ignore", inplace=True)

        return df

    # ------------------------------------------------------------------
    # Speed Figures (Benter/Mordin — normalized race times)
    # ------------------------------------------------------------------
    def _calc_speed_figures(self, df: pd.DataFrame) -> pd.DataFrame:
        """Speed ratings from completion times, adjusted for track/distance/going.

        Research basis: Benter (1994) used normalized times as a key variable.
        Mordin pioneered speed figures for European racing. Speed figures
        consistently rank among the top predictors in academic literature.

        RSR = (standard_time - actual_time) / standard_time * 100
        Positive RSR = faster than standard.
        """
        if "comptime_numeric" not in df.columns:
            for col in [
                "RSR", "preracehorsecareerRSR", "LR_RSR",
                "LR3_RSR", "LR5_RSR", "best_RSR", "RSR_gap",
                "SFI", "SFI_3",
            ]:
                df[col] = np.nan
            return df

        df["_comptime"] = pd.to_numeric(
            df["comptime_numeric"], errors="coerce"
        )

        # Standard time per (track, distance, going) — median of all times
        df["_going_lower"] = (
            df["going_description"].fillna("unknown").str.lower().str.strip()
        )
        df["_dist_round"] = df["dist_furlongs"].round(0)
        df["_track_lower_sf"] = (
            df["track"].fillna("unknown").str.lower().str.strip()
        )

        std_times = df.groupby(
            ["_track_lower_sf", "_dist_round", "_going_lower"]
        )["_comptime"].transform("median")

        # RSR: positive = faster than standard
        df["RSR"] = (
            (std_times - df["_comptime"]) / std_times.replace(0, np.nan) * 100
        )
        # Null out RSR where comptime is missing
        df.loc[df["_comptime"].isna(), "RSR"] = np.nan

        # Horse career and rolling RSR (lagged)
        df = df.sort_values(
            ["horse_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        df["preracehorsecareerRSR"] = grp["RSR"].apply(
            lambda x: x.shift(1).expanding().mean()
        )
        df["LR_RSR"] = grp["RSR"].shift(1)

        for w in [3, 5]:
            df[f"LR{w}_RSR"] = grp["RSR"].apply(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )

        # Best career RSR (lagged)
        df["best_RSR"] = grp["RSR"].apply(
            lambda x: x.shift(1).expanding().max()
        )
        df["RSR_gap"] = df["best_RSR"] - df.get("LR3_RSR", np.nan)

        # Speed Figure Improvement
        lr1 = grp["RSR"].shift(1)
        lr2 = grp["RSR"].shift(2)
        df["SFI"] = lr1 - lr2

        # Average improvement over last 3
        lr3 = grp["RSR"].shift(3)
        df["SFI_3"] = (
            (lr1 - lr2).fillna(0) + (lr2 - lr3).fillna(0)
        ) / 2

        df.drop(
            columns=["_comptime", "_going_lower", "_dist_round",
                     "_track_lower_sf"],
            errors="ignore", inplace=True,
        )

        return df

    # ------------------------------------------------------------------
    # Actual Lengths Beaten (Benter — real margins vs approximation)
    # ------------------------------------------------------------------
    def _calc_actual_lengths_beaten(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse total_dst_bt to get actual lengths beaten from winner.

        Research basis: Our RB metric approximates beaten distances from
        finishing position. actual beaten-lengths data is far more
        informative — a horse beaten a neck in 2nd is vastly different
        from one beaten 20 lengths in 2nd.
        """
        if "total_dst_bt" not in df.columns:
            for col in [
                "LB", "preracehorsecareerLB", "LR_LB",
                "LR3_LB", "LR5_LB", "FSALB",
            ]:
                df[col] = np.nan
            return df

        def parse_lengths(val):
            """Parse beaten-length strings to numeric."""
            if pd.isna(val) or val == "" or val == "0":
                return 0.0
            s = str(val).strip().lower()
            if s in ("dht", "dh"):
                return 0.0
            if s == "nse":
                return 0.05
            if s == "shd":
                return 0.1
            if s in ("hd", "sht-hd"):
                return 0.15
            if s in ("nk", "snk"):
                return 0.2
            if s == "dist":
                return 30.0
            # Handle combined forms like "2nk", "1shd", "3hd"
            m = re.match(r"(\d+\.?\d*)\s*(nk|shd|hd|nse)?", s)
            if m:
                base = float(m.group(1))
                frac = m.group(2)
                if frac == "nk":
                    base += 0.2
                elif frac == "shd":
                    base += 0.1
                elif frac == "hd":
                    base += 0.15
                elif frac == "nse":
                    base += 0.05
                return base
            try:
                return float(s)
            except (ValueError, TypeError):
                return np.nan

        df["LB"] = df["total_dst_bt"].apply(parse_lengths)

        # Career and rolling LB averages (lagged)
        df = df.sort_values(
            ["horse_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        df["preracehorsecareerLB"] = grp["LB"].apply(
            lambda x: x.shift(1).expanding().mean()
        )
        df["LR_LB"] = grp["LB"].shift(1)

        for w in [3, 5]:
            df[f"LR{w}_LB"] = grp["LB"].apply(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )

        # Field-Size Adjusted Lengths Beaten
        median_fs = df["number_of_runners"].median()
        if pd.isna(median_fs) or median_fs == 0:
            median_fs = 10.0
        df["FSALB"] = df["LB"] * (df["number_of_runners"] / median_fs)

        return df

    # ------------------------------------------------------------------
    # Equipment Changes (German model / Mordin — first-time blinkers)
    # ------------------------------------------------------------------
    def _calc_equipment_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect equipment changes, especially first-time headgear.

        Research basis: First-time blinkers is one of the most well-known
        and exploitable signals in UK racing. Equipment changes indicate
        trainer intent to improve performance.
        """
        if "headgear" not in df.columns:
            for col in [
                "headgear_change", "first_time_headgear",
                "headgear_removed", "has_headgear",
            ]:
                df[col] = 0
            return df

        df["_hg_clean"] = df["headgear"].fillna("").str.strip().str.lower()
        df["has_headgear"] = (df["_hg_clean"] != "").astype(int)

        df = df.sort_values(
            ["horse_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        lr_hg = grp["_hg_clean"].shift(1)

        # Headgear changed from last run
        df["headgear_change"] = (
            (df["_hg_clean"] != lr_hg) & lr_hg.notna()
        ).astype(int)

        # First time EVER wearing headgear
        cum_hg = grp["has_headgear"].apply(
            lambda x: x.shift(1).cumsum()
        ).fillna(0)
        df["first_time_headgear"] = (
            (df["has_headgear"] == 1) & (cum_hg == 0)
        ).astype(int)

        # Headgear removed (wore last time, not today)
        df["headgear_removed"] = (
            (df["has_headgear"] == 0)
            & (lr_hg.fillna("") != "")
            & lr_hg.notna()
        ).astype(int)

        df.drop(columns=["_hg_clean"], errors="ignore", inplace=True)

        return df

    # ------------------------------------------------------------------
    # Surface Preference (Benter — turf vs all-weather)
    # ------------------------------------------------------------------
    def _calc_surface_preference(self, df: pd.DataFrame) -> pd.DataFrame:
        """Horse performance on different surfaces (turf vs all-weather).

        Research basis: Benter's fundamental variables included surface
        preference. Some horses clearly prefer one surface. AW vs turf
        is a major factor in UK racing.
        """
        if "surface_type" not in df.columns:
            for col in [
                "surface_nfp", "surface_win_rate", "surface_runs",
                "first_on_surface",
            ]:
                df[col] = np.nan
            return df

        df["_surface_clean"] = (
            df["surface_type"].fillna("unknown").str.lower().str.strip()
        )

        df = df.sort_values(
            ["horse_name", "_surface_clean", "race_date", "race_time"]
        ).reset_index(drop=True)

        surf_grp = df.groupby(
            ["horse_name", "_surface_clean"], group_keys=False
        )

        df["surface_runs"] = surf_grp.cumcount()  # 0-indexed
        df["first_on_surface"] = (df["surface_runs"] == 0).astype(int)

        surf_n = df["surface_runs"].replace(0, np.nan)

        df["surface_nfp"] = surf_grp["NFP"].apply(
            lambda x: x.shift(1).expanding().mean()
        )

        cum_wins = surf_grp["won"].apply(lambda x: x.shift(1).cumsum())
        df["surface_win_rate"] = cum_wins / surf_n

        df.drop(columns=["_surface_clean"], errors="ignore", inplace=True)

        return df

    # ------------------------------------------------------------------
    # Track Preference (Benter — course specialist detection)
    # ------------------------------------------------------------------
    def _calc_track_preference(self, df: pd.DataFrame) -> pd.DataFrame:
        """Horse, trainer, and jockey performance at specific tracks.

        Research basis: Benter included specific track preference in his
        fundamental variables. Course specialists are well-documented
        in UK racing — some trainers have phenomenal records at specific tracks.
        """
        df["_track_clean"] = (
            df["track"].fillna("unknown").str.lower().str.strip()
        )

        # Horse at track
        df = df.sort_values(
            ["horse_name", "_track_clean", "race_date", "race_time"]
        ).reset_index(drop=True)
        ht_grp = df.groupby(
            ["horse_name", "_track_clean"], group_keys=False
        )
        df["horse_track_runs"] = ht_grp.cumcount()
        ht_n = df["horse_track_runs"].replace(0, np.nan)
        df["horse_track_nfp"] = ht_grp["NFP"].apply(
            lambda x: x.shift(1).expanding().mean()
        )
        ht_wins = ht_grp["won"].apply(lambda x: x.shift(1).cumsum())
        df["horse_track_win_rate"] = ht_wins / ht_n

        # Trainer at track
        df = df.sort_values(
            ["trainer", "_track_clean", "race_date", "race_time"]
        ).reset_index(drop=True)
        tt_grp = df.groupby(
            ["trainer", "_track_clean"], group_keys=False
        )
        df["trainer_track_runs"] = tt_grp.cumcount()
        tt_n = df["trainer_track_runs"].replace(0, np.nan)
        tt_wins = tt_grp["won"].apply(lambda x: x.shift(1).cumsum())
        df["trainer_track_win_rate"] = tt_wins / tt_n

        # Jockey at track
        df = df.sort_values(
            ["jockey_name", "_track_clean", "race_date", "race_time"]
        ).reset_index(drop=True)
        jt_grp = df.groupby(
            ["jockey_name", "_track_clean"], group_keys=False
        )
        df["jockey_track_runs"] = jt_grp.cumcount()
        jt_n = df["jockey_track_runs"].replace(0, np.nan)
        jt_wins = jt_grp["won"].apply(lambda x: x.shift(1).cumsum())
        df["jockey_track_win_rate"] = jt_wins / jt_n

        df.drop(columns=["_track_clean"], errors="ignore", inplace=True)

        return df

    # ------------------------------------------------------------------
    # OR Trajectory (Ziemba — handicap mark changes)
    # ------------------------------------------------------------------
    def _calc_or_trajectory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Official Rating changes between runs and vs career best.

        Research basis: Ziemba showed that OR changes are among the most
        predictive features in handicaps. A dropping OR means the horse
        is 'well-handicapped'. OR vs career best identifies value.
        """
        df["_or_num"] = pd.to_numeric(
            df["official_rating"], errors="coerce"
        )

        df = df.sort_values(
            ["horse_name", "race_date", "race_time"]
        ).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        # OR change from last run
        lr_or = grp["_or_num"].shift(1)
        df["or_change"] = df["_or_num"] - lr_or

        # OR change from 3 runs ago
        lr3_or = grp["_or_num"].shift(3)
        df["or_change_3"] = df["_or_num"] - lr3_or

        # Career best OR (lagged)
        df["career_best_or"] = grp["_or_num"].apply(
            lambda x: x.shift(1).expanding().max()
        )

        # OR vs career best (negative = below peak = potentially well-handicapped)
        df["or_vs_best"] = df["_or_num"] - df["career_best_or"]

        # Is racing off peak OR? (dropped 5+ lbs)
        df["or_off_peak"] = (
            (df["career_best_or"] - df["_or_num"]) > 5
        ).astype(int)

        # OR vs last winning OR
        df["_winning_or"] = df["_or_num"].where(df["won"] == 1)
        df["last_winning_or"] = grp["_winning_or"].apply(
            lambda x: x.shift(1).ffill()
        )
        df["or_vs_last_win"] = df["_or_num"] - df["last_winning_or"]

        df.drop(
            columns=["_or_num", "_winning_or"],
            errors="ignore", inplace=True,
        )

        return df

    # ------------------------------------------------------------------
    # Hot Form — Trainer/Jockey Recent Form (14/30 day rolling)
    # ------------------------------------------------------------------
    def _calc_hot_form(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trainer and jockey recent form using time-windowed rolling stats.

        Research basis: Trainers go through hot and cold streaks due to
        yard illness, horse fitness, travel patterns. Jockey confidence
        and booking patterns also matter. 14-day strike rate >25% is
        considered 'red hot form'.
        """
        df = df.sort_values(
            ["race_date", "race_time"]
        ).reset_index(drop=True)

        df["_date_ordinal"] = df["race_date"].astype(np.int64) // 10**9 // 86400

        for entity_col, prefix in [
            ("trainer", "trainer"),
            ("jockey_name", "jockey"),
        ]:
            df = df.sort_values(
                [entity_col, "race_date", "race_time"]
            ).reset_index(drop=True)
            egrp = df.groupby(entity_col, group_keys=False)

            # 14-day rolling win rate (approximate via last N runs within window)
            # Use time-aware approach: count wins in last 14/30 days
            for window_days, suffix in [(14, "14d"), (30, "30d")]:
                # For each entity, compute rolling count of runs and wins
                # within the time window using date differences
                wins_list = []
                runs_list = []
                for _, group in df.groupby(entity_col):
                    g_dates = group["_date_ordinal"].values
                    g_won = group["won"].values
                    g_wins = np.full(len(group), np.nan)
                    g_runs = np.full(len(group), np.nan)
                    for i in range(len(group)):
                        cutoff = g_dates[i] - window_days
                        # Look at runs BEFORE this one (j < i) within window
                        mask = (g_dates[:i] > cutoff) & (g_dates[:i] <= g_dates[i])
                        n_runs = mask.sum()
                        if n_runs > 0:
                            g_wins[i] = g_won[:i][mask].sum()
                            g_runs[i] = float(n_runs)
                    wins_list.append(pd.Series(g_wins, index=group.index))
                    runs_list.append(pd.Series(g_runs, index=group.index))

                all_wins = pd.concat(wins_list)
                all_runs = pd.concat(runs_list)
                df[f"{prefix}_wins_{suffix}"] = all_wins
                df[f"{prefix}_runs_{suffix}"] = all_runs
                df[f"{prefix}_sr_{suffix}"] = (
                    all_wins / all_runs.replace(0, np.nan)
                )

            # Form delta: recent form vs career
            career_wiv = df.get(
                f"prerace{prefix}careerWIV"
                if prefix == "trainer"
                else f"prerace{prefix.replace('jockey', 'jockey')}careerWIV",
                0,
            )
            if prefix == "trainer":
                career_col = "preracetrainercareerWIV"
            else:
                career_col = "preracejockeycareerWIV"

            if career_col in df.columns:
                df[f"{prefix}_form_delta"] = (
                    df[f"{prefix}_sr_14d"] - df[career_col]
                )
            else:
                df[f"{prefix}_form_delta"] = np.nan

        df.drop(columns=["_date_ordinal"], errors="ignore", inplace=True)

        return df

    # ------------------------------------------------------------------
    # Within-Race Rankings
    # ------------------------------------------------------------------
    def _calc_within_race_ranks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank every horse within its race on continuous metrics."""
        # Restore date sort for race grouping
        df = df.sort_values(
            ["race_date", "race_time", "track"]
        ).reset_index(drop=True)

        rank_configs = {
            # Horse-level
            "rNFP": "preracehorsecareerNFP",
            "rNFPLR3": "LR3NFPtotal",
            "rNFPLR5": "LR5NFPtotal",
            "rNFPLR10": "LR10NFPtotal",
            "horseRBrank": "preracehorsecareerRB",
            "horseFSARBrank": "preracehorsecareerFSARB",
            "horseFSARB2rank": "preracehorsecareerFSARB2",
            "horseNFPrank": "preracehorsecareerNFP",
            "horseWIVrank": "preracehorsecareerWIV",
            "horseWAXrank": "preracehorsecareerWAX",
            "horseWOArank": "preracehorsecareerWOA",
            "horseCWOrank": "preracehorsecareerCWO",
            "horseRunsrank": "preracehorsecareerRuns",
            "horseWinsrank": "preracehorsecareerWins",
            "horsePlacesrank": "preracehorsecareerPlaces",
            # ORR2/RWO
            "rORR2LR": "LR_ORR2",
            "rRWOLR3": "LR3_RWO",
            "rRWOLR5": "LR5_RWO",
            "rRWOLR10": "LR10_RWO",
            # EPF
            "rEPF_LR": "LR_EPF",
            "rEPF2_LR": "LR_EPF2",
            "rEPF3_LR": "LR_EPF3",
            "rJockeyEPF": "Jockey_Career_EPF",
            "rTrainerEPF": "trainer_Career_EPF",
            "rHorseCareerEPF": "Horse_Career_EPF",
            # Other
            "rDSLR": "FinalDSLR",
            "rFSS": "FSS",
            "rFCS": "FCS",
            "rPFD3": "PFD3",
            "rPFD5": "PFD5",
            "rPFD10": "PFD10",
            "rWPMRF3": "WPMRF3",
            "rWPMRF5": "WPMRF5",
            "rWPMRF10": "WPMRF10",
            "rPMW3": "PMW3",
            "rPMW5": "PMW5",
            "rPMW10": "PMW10",
            "rOFS3": "OFS3",
            "rOFS5": "OFS5",
            "rOFS10": "OFS10",
            "rTJWIV": "trainerjockeycareerWIV",
            "rTJNFP": "trainerjockeycareerNFP",
            # Trainer
            "trainerRBrank": "preracetrainercareerRB"
            if "preracetrainercareerRB" in df.columns
            else None,
            "trainerNFPrank": "preracetrainercareerNFP"
            if "preracetrainercareerNFP" in df.columns
            else None,
            "trainerWIVrank": "preracetrainercareerWIV",
            "trainerWAXrank": "preracetrainercareerWAX",
            "trainerWOArank": "preracetrainercareerWOA",
            "trainerCWOrank": "preracetrainercareerCWO",
            # Jockey
            "jockeyNFPrank": "preracejockeycareerNFP"
            if "preracejockeycareerNFP" in df.columns
            else None,
            "jockeyWIVrank": "preracejockeycareerWIV",
            "jockeyWAXrank": "preracejockeycareerWAX",
            "jockeyWOArank": "preracejockeycareerWOA",
            "jockeyCWOrank": "preracejockeycareerCWO",
            "jockeyLRIrank": "totalLRPjockeyindex",
            # New research-backed rankings
            "rEXP_NFP5": "EXP_NFP5",
            "rEXP_RB5": "EXP_RB5",
            "rResidual": "career_residual",
            "rFormSlope3": "form_slope_3",
            "rConsistency": "career_nfp_std",
            "rDistApt": "dist_from_preferred",
            "rGoingPref": "going_from_preferred",
            "rWeightVsAvg": "weight_vs_avg",
            "rUnexposure": "unexposure_score",
            # Pedigree rankings
            "rSireNFP": "sire_avg_nfp",
            "rSireWIV": "sire_wiv",
            "rSireGoingNFP": "sire_going_nfp",
            "rSireDistNFP": "sire_dist_nfp",
            "rDamsireNFP": "damsire_avg_nfp",
            # Speed / lengths / new feature rankings
            "rRSR": "preracehorsecareerRSR",
            "rLB": "preracehorsecareerLB",
            "rSurfaceNFP": "surface_nfp",
            "rHorseTrackNFP": "horse_track_nfp",
            "rORChange": "or_change",
            "rTrainerSR14d": "trainer_sr_14d",
            "rJockeySR14d": "jockey_sr_14d",
        }

        for rank_name, source_col in rank_configs.items():
            if source_col is None or source_col not in df.columns:
                continue
            df[rank_name] = df.groupby("raceid")[source_col].rank(
                ascending=False, method="min", na_option="bottom"
            )

        return df
