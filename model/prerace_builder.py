"""
Pre-Race Feature Vector Construction.

Takes declared runners for a future race and builds feature vectors
from historical data. At prediction time, we only know the declared
runners — everything else must be looked up from past performances.

This is the bridge between raw historical data (with custom metrics)
and the prediction model.
"""

import numpy as np
import pandas as pd


class PreRaceBuilder:
    """Build pre-race feature vectors by looking up each entity's history.

    At prediction time we know ONLY:
    - horse_name, age, sex
    - jockey, trainer
    - course, distance, going, race_class
    - weight_carried, draw_position
    - number_of_runners (field size)
    - (optionally) current Betfair price / morning price

    We must look up everything else from the historical results database.

    Args:
        history_df: Complete historical results with custom metrics
                    already calculated (from CustomMetricsEngine).
    """

    # Horse features to extract from most recent historical appearance
    HORSE_FEATURES = [
        "preracehorsecareerRuns",
        "preracehorsecareerWins",
        "preracehorsecareerPlaces",
        "preracehorsecareerNFP",
        "preracehorsecareerRB",
        "preracehorsecareerFSARB",
        "preracehorsecareerWIV",
        "preracehorsecareerWAX",
        "preracehorsecareerWOA",
        "preracehorsecareerCWO",
        "Horse_Career_EPF",
        "horsepaceindex",
        "LRNFP",
        "LR3NFPtotal",
        "LR5NFPtotal",
        "LR10NFPtotal",
        "LR3_RWO",
        "LR5_RWO",
        "LR10_RWO",
        "FSS",
        "FCS",
        "PFD3",
        "PFD5",
        "PFD10",
        "WPMRF3",
        "WPMRF5",
        "WPMRF10",
        "PMW3",
        "PMW5",
        "PMW10",
        "OFS3",
        "OFS5",
        "OFS10",
        "FinalDSLR",
        "CIL3",
        "CIL5",
        "CIL10",
        "LR_ORR2",
        "preracehorsecareerORR2",
    ]

    # Jockey features
    JOCKEY_FEATURES = [
        "preracejockeycareerWins",
        "preracejockeycareerRuns",
        "preracejockeycareerWIV",
        "preracejockeycareerNFP",
        "preracejockeycareerRB"
        if False
        else "preracejockeycareerWAX",  # RB not computed for jockey
        "preracejockeycareerWAX",
        "preracejockeycareerWOA",
        "Jockey_Career_EPF",
        "jockeypaceindex",
        "totalLRPjockeyindex",
    ]

    # Trainer features
    TRAINER_FEATURES = [
        "preracetrainercareerWins",
        "preracetrainercareerRuns",
        "preracetrainercareerWIV",
        "preracetrainercareerNFP",
        "preracetrainercareerWAX",
        "preracetrainercareerWOA",
        "trainer_Career_EPF",
        "trainerpaceindex",
    ]

    # Trainer-jockey combination features
    TJ_FEATURES = [
        "trainerjockeycareerWIV",
        "trainerjockeycareerNFP",
    ]

    # Today's race features (known from declarations)
    RACE_FEATURES = [
        "number_of_runners",
        "dist_furlongs",
        "going_numeric",
        "race_class_numeric",
        "weight_lbs",
        "draw_position",
        "age",
        "is_debut",
    ]

    # Derived features computed at prediction time
    DERIVED_FEATURES = [
        "going_preference",
        "distance_preference",
        "class_change",
        "weight_change",
        "course_win_pct",
        "course_runs",
        "days_since_last_run",
    ]

    def __init__(self, history_df: pd.DataFrame):
        self.history_df = history_df.copy()
        self.history_df["race_date"] = pd.to_datetime(
            self.history_df["race_date"]
        )

        # Pre-compute population medians for debut runners
        all_features = (
            self.HORSE_FEATURES + self.JOCKEY_FEATURES + self.TRAINER_FEATURES
        )
        self._medians = {}
        for col in all_features:
            if col in self.history_df.columns:
                self._medians[col] = self.history_df[col].median()

        # Build horse/jockey/trainer lookup tables (most recent row per entity)
        self._build_lookups()

    def _build_lookups(self):
        """Build lookup dictionaries indexed by entity name -> most recent row."""
        hdf = self.history_df.sort_values("race_date")

        # Horse: last row per horse
        self._horse_lookup = {}
        for name, group in hdf.groupby("horse_name"):
            self._horse_lookup[name] = group.iloc[-1]

        # Jockey: last row per jockey
        self._jockey_lookup = {}
        if "jockey_name" in hdf.columns:
            for name, group in hdf.groupby("jockey_name"):
                self._jockey_lookup[name] = group.iloc[-1]

        # Trainer: last row per trainer
        self._trainer_lookup = {}
        if "trainer" in hdf.columns:
            for name, group in hdf.groupby("trainer"):
                self._trainer_lookup[name] = group.iloc[-1]

        # Trainer-jockey combos
        self._tj_lookup = {}
        if "trainer" in hdf.columns and "jockey_name" in hdf.columns:
            for (t, j), group in hdf.groupby(["trainer", "jockey_name"]):
                self._tj_lookup[(t, j)] = group.iloc[-1]

        # Horse at track
        self._horse_track = {}
        if "track" in hdf.columns:
            for (h, t), group in hdf.groupby(["horse_name", "track"]):
                wins = group["won"].sum() if "won" in group.columns else 0
                self._horse_track[(h, t)] = {
                    "runs": len(group),
                    "win_pct": wins / len(group) if len(group) > 0 else 0,
                }

    def build_features(
        self, declared_runners: pd.DataFrame, race_date: str
    ) -> pd.DataFrame:
        """Build pre-race feature vectors for declared runners.

        Args:
            declared_runners: DataFrame with columns:
                horse_name, jockey_name, trainer, track, dist_furlongs,
                going_description, race_class, weight_lbs (pounds),
                draw_position (stall), age (horse_age),
                number_of_runners
            race_date: The date of the race (YYYY-MM-DD).

        Returns:
            DataFrame with one row per runner and all feature columns.
        """
        race_dt = pd.Timestamp(race_date)
        rows = []

        for _, runner in declared_runners.iterrows():
            row = self._build_runner_features(runner, race_dt)
            rows.append(row)

        result = pd.DataFrame(rows)

        # Compute field-level features
        result = self._add_field_features(result)

        # Compute within-race ranks
        result = self._add_within_race_ranks(result)

        return result

    def _build_runner_features(
        self, runner: pd.Series, race_dt: pd.Timestamp
    ) -> dict:
        """Build feature vector for a single runner."""
        horse_name = runner.get("horse_name", "")
        jockey_name = runner.get("jockey_name", "")
        trainer = runner.get("trainer", "")
        track = runner.get("track", "")

        features = {"horse_name": horse_name}
        is_debut = horse_name not in self._horse_lookup

        # --- Horse features ---
        if not is_debut:
            horse_row = self._horse_lookup[horse_name]
            for col in self.HORSE_FEATURES:
                features[col] = (
                    horse_row[col] if col in horse_row.index else np.nan
                )

            # Days since last run (computed fresh)
            last_date = pd.Timestamp(horse_row["race_date"])
            features["days_since_last_run"] = (race_dt - last_date).days

            # Last BFSP and finishing position
            features["last_bsp"] = (
                horse_row.get("bfsp", np.nan)
            )
            features["last_finish_pos"] = (
                horse_row.get("placing_numerical", np.nan)
            )
            features["last_3_avg_position"] = (
                horse_row.get("placing_numerical", np.nan)
            )

            # Derived: going/distance/class preferences
            features["going_preference"] = self._going_preference(
                runner, horse_row
            )
            features["distance_preference"] = self._distance_preference(
                runner, horse_row
            )
            features["class_change"] = self._class_change(runner, horse_row)
            features["weight_change"] = self._weight_change(runner, horse_row)

        else:
            # Debut runner: fill with population medians
            for col in self.HORSE_FEATURES:
                features[col] = self._medians.get(col, 0)
            features["days_since_last_run"] = 0
            features["last_bsp"] = np.nan
            features["last_finish_pos"] = np.nan
            features["last_3_avg_position"] = np.nan
            features["going_preference"] = 0
            features["distance_preference"] = 0
            features["class_change"] = 0
            features["weight_change"] = 0

        features["is_debut"] = 1 if is_debut else 0

        # --- Jockey features ---
        if jockey_name in self._jockey_lookup:
            jockey_row = self._jockey_lookup[jockey_name]
            for col in self.JOCKEY_FEATURES:
                features[col] = (
                    jockey_row[col] if col in jockey_row.index else np.nan
                )
        else:
            for col in self.JOCKEY_FEATURES:
                features[col] = self._medians.get(col, 0)

        # --- Trainer features ---
        if trainer in self._trainer_lookup:
            trainer_row = self._trainer_lookup[trainer]
            for col in self.TRAINER_FEATURES:
                features[col] = (
                    trainer_row[col] if col in trainer_row.index else np.nan
                )
        else:
            for col in self.TRAINER_FEATURES:
                features[col] = self._medians.get(col, 0)

        # --- Trainer-Jockey combo ---
        tj_key = (trainer, jockey_name)
        if tj_key in self._tj_lookup:
            tj_row = self._tj_lookup[tj_key]
            for col in self.TJ_FEATURES:
                features[col] = (
                    tj_row[col] if col in tj_row.index else np.nan
                )
        else:
            for col in self.TJ_FEATURES:
                features[col] = 0

        # --- Course-specific stats ---
        ht_key = (horse_name, track)
        if ht_key in self._horse_track:
            features["course_win_pct"] = self._horse_track[ht_key]["win_pct"]
            features["course_runs"] = self._horse_track[ht_key]["runs"]
        else:
            features["course_win_pct"] = 0
            features["course_runs"] = 0

        # --- Today's race features ---
        features["number_of_runners"] = runner.get("number_of_runners", np.nan)
        features["dist_furlongs"] = pd.to_numeric(
            runner.get("dist_furlongs", np.nan), errors="coerce"
        )
        features["going_numeric"] = self._encode_going(
            runner.get("going_description", "")
        )
        features["race_class_numeric"] = pd.to_numeric(
            runner.get("race_class", np.nan), errors="coerce"
        )
        features["weight_lbs"] = pd.to_numeric(
            runner.get("pounds", runner.get("weight_lbs", np.nan)),
            errors="coerce",
        )
        features["draw_position"] = pd.to_numeric(
            runner.get("stall", runner.get("draw_position", np.nan)),
            errors="coerce",
        )
        features["age"] = pd.to_numeric(
            runner.get("horse_age", runner.get("age", np.nan)),
            errors="coerce",
        )

        return features

    def _add_field_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add field-level aggregate features."""
        for col, name in [
            ("preracehorsecareerNFP", "field_avg_career_nfp"),
            ("preracehorsecareerWIV", "field_avg_career_wiv"),
            ("preracehorsecareerRB", "field_avg_career_rb"),
        ]:
            if col in df.columns:
                df[name] = df[col].mean()
        return df

    def _add_within_race_ranks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank runners within the race on continuous metrics."""
        rank_cols = {
            "rNFP": "preracehorsecareerNFP",
            "rNFPLR5": "LR5NFPtotal",
            "horseWIVrank": "preracehorsecareerWIV",
            "horseRBrank": "preracehorsecareerRB",
            "rRWOLR5": "LR5_RWO",
            "rRWOLR10": "LR10_RWO",
            "rPFD5": "PFD5",
            "rFSS": "FSS",
            "rFCS": "FCS",
        }
        for rank_name, source_col in rank_cols.items():
            if source_col in df.columns:
                df[rank_name] = df[source_col].rank(
                    ascending=False, method="min", na_option="bottom"
                )
        return df

    @staticmethod
    def _encode_going(going: str) -> float:
        """Encode going description to numeric scale (1=Heavy, 7=Hard)."""
        if not going or not isinstance(going, str):
            return 4.0
        g = going.lower().strip()
        going_map = {
            "heavy": 1.0,
            "soft": 2.0,
            "yielding": 2.5,
            "good to soft": 3.0,
            "good": 4.0,
            "good to firm": 5.0,
            "firm": 6.0,
            "hard": 7.0,
            "standard": 4.0,
            "standard to slow": 3.0,
            "slow": 2.0,
        }
        for key, val in going_map.items():
            if key in g:
                return val
        return 4.0

    def _going_preference(
        self, runner: pd.Series, horse_row: pd.Series
    ) -> float:
        """How much today's going differs from the horse's historical average."""
        today_going = self._encode_going(
            runner.get("going_description", "")
        )
        hist_going = self._encode_going(
            horse_row.get("going_description", "")
        )
        return today_going - hist_going

    @staticmethod
    def _distance_preference(
        runner: pd.Series, horse_row: pd.Series
    ) -> float:
        """How much today's distance differs from horse's last distance."""
        today_dist = pd.to_numeric(
            runner.get("dist_furlongs", np.nan), errors="coerce"
        )
        hist_dist = pd.to_numeric(
            horse_row.get("dist_furlongs", np.nan), errors="coerce"
        )
        if pd.isna(today_dist) or pd.isna(hist_dist):
            return 0
        return today_dist - hist_dist

    @staticmethod
    def _class_change(runner: pd.Series, horse_row: pd.Series) -> float:
        """Today's race class minus horse's last race class."""
        today_class = pd.to_numeric(
            runner.get("race_class", np.nan), errors="coerce"
        )
        hist_class = pd.to_numeric(
            horse_row.get("race_class", np.nan), errors="coerce"
        )
        if pd.isna(today_class) or pd.isna(hist_class):
            return 0
        return today_class - hist_class

    @staticmethod
    def _weight_change(runner: pd.Series, horse_row: pd.Series) -> float:
        """Today's weight minus horse's last weight."""
        today_wt = pd.to_numeric(
            runner.get("pounds", runner.get("weight_lbs", np.nan)),
            errors="coerce",
        )
        hist_wt = pd.to_numeric(
            horse_row.get("pounds", np.nan), errors="coerce"
        )
        if pd.isna(today_wt) or pd.isna(hist_wt):
            return 0
        return today_wt - hist_wt

    def get_feature_columns(self) -> list[str]:
        """Return the full list of feature column names for the model."""
        cols = (
            self.HORSE_FEATURES
            + self.JOCKEY_FEATURES
            + self.TRAINER_FEATURES
            + self.TJ_FEATURES
            + self.RACE_FEATURES
            + self.DERIVED_FEATURES
            + [
                "last_bsp",
                "last_finish_pos",
                "last_3_avg_position",
                "field_avg_career_nfp",
                "field_avg_career_wiv",
                "field_avg_career_rb",
                "rNFP",
                "rNFPLR5",
                "horseWIVrank",
                "horseRBrank",
                "rRWOLR5",
                "rRWOLR10",
                "rPFD5",
                "rFSS",
                "rFCS",
            ]
        )
        return cols
