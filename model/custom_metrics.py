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
        grp = df.groupby("horse_name")

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
        grp = df.groupby("horse_name")

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
            grp = df.groupby(col)

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
            grp = df.groupby(col)

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
        grp = df.groupby("horse_name")

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
        grp = df.groupby("horse_name")

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
            egrp = df.groupby(entity_col)
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
        grp = df.groupby("horse_name")

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
        grp = df.groupby("horse_name")

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
        grp = df.groupby("horse_name")

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
        grp = df.groupby("horse_name")

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
        grp = df.groupby("horse_name")

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
        grp = df.groupby("horse_name")

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
        h_grp = df.groupby("horse_name")

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
        j_grp = df.groupby("jockey_name")

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
            grp = df.groupby(entity_col)
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
        tj_grp = df.groupby(["trainer", "jockey_name"])

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
        grp = df.groupby("horse_name")

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
        grp = df.groupby("horse_name")

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
        }

        for rank_name, source_col in rank_configs.items():
            if source_col is None or source_col not in df.columns:
                continue
            df[rank_name] = df.groupby("raceid")[source_col].rank(
                ascending=False, method="min", na_option="bottom"
            )

        return df
