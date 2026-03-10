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
        # Career stats
        "preracehorsecareerRuns",
        "preracehorsecareerWins",
        "preracehorsecareerPlaces",
        "preracehorsecareerNFP",
        "preracehorsecareerRB",
        "preracehorsecareerFSARB",
        "preracehorsecareerFSARB2",
        "preracehorsecareerWIV",
        "preracehorsecareerWAX",
        "preracehorsecareerWOA",
        "preracehorsecareerCWO",
        "preracehorsecareerORR2",
        "Horse_Career_EPF",
        "horsepaceindex",
        # Recent form
        "LRNFP",
        "LR3NFPtotal",
        "LR5NFPtotal",
        "LR10NFPtotal",
        "LR3_RWO",
        "LR5_RWO",
        "LR10_RWO",
        "LR_ORR2",
        # EPF lags
        "LR_EPF",
        "LR2_EPF",
        "LR3_EPF",
        "LR_EPF2",
        "LR_EPF3",
        # Volatility
        "FSS",
        "FCS",
        # Market / probability
        "PFD3",
        "PFD5",
        "PFD10",
        "OFS1",
        "OFS3",
        "OFS5",
        "OFS10",
        # Prize money
        "WPMRF3",
        "WPMRF5",
        "WPMRF10",
        "PMW3",
        "PMW5",
        "PMW10",
        # Timing / freshness
        "FinalDSLR",
        "DSLR1",
        # Confidence / data completeness
        "CIL3",
        "CIL5",
        "CIL10",
        "LR3COUNT",
        "LR5COUNT",
        "LR10COUNT",
        # LRP momentum
        "LRPTotalScore",
        # Last race opposition strength
        "LR_RACE_RB",
        "LR_RACE_WIV",
        "LR_RACE_NFP",
        "LR_RACE_Wins",
        "LR_RACE_WOA",
    ]

    # Jockey features
    JOCKEY_FEATURES = [
        "preracejockeycareerWins",
        "preracejockeycareerRuns",
        "preracejockeycareerWIV",
        "preracejockeycareerNFP",
        "preracejockeycareerWAX",
        "preracejockeycareerWOA",
        "preracejockeycareerCWO",
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
        "preracetrainercareerCWO",
        "trainer_Career_EPF",
        "trainerpaceindex",
    ]

    # Trainer-jockey combination features
    TJ_FEATURES = [
        "trainerjockeycareerWIV",
        "trainerjockeycareerNFP",
        "trainerjockeyWAX",
        "trainerjockeyWOA",
        "trainerjockeyCWO",
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
        "official_rating",
        "jockeys_claim",
        "headgear_flag",
        "sex_numeric",
        "surface_numeric",
        "race_type_numeric",
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
        "or_vs_race_max",
        "or_vs_race_median",
        "stallion_win_rate",
        "stallion_runs",
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

        # Build horse/jockey/trainer/stallion lookup tables
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

        # Stallion aggregate stats
        self._stallion_lookup = {}
        if "stallion" in hdf.columns:
            for name, group in hdf.groupby("stallion"):
                wins = group["won"].sum() if "won" in group.columns else 0
                self._stallion_lookup[name] = {
                    "runs": len(group),
                    "win_rate": wins / len(group) if len(group) > 0 else 0,
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

        # --- Official rating features ---
        or_val = pd.to_numeric(
            runner.get("official_rating", np.nan), errors="coerce"
        )
        features["official_rating"] = or_val

        # OR relative to race (computed later in _add_field_features if needed)
        features["or_vs_race_max"] = np.nan  # placeholder, set in field features
        features["or_vs_race_median"] = np.nan

        # --- Equipment / metadata features ---
        headgear = str(runner.get("headgear", "")).strip()
        features["headgear_flag"] = 1 if headgear and headgear != "" else 0

        sex = str(runner.get("horse_sex", "")).lower().strip()
        features["sex_numeric"] = self._encode_sex(sex)

        features["jockeys_claim"] = pd.to_numeric(
            runner.get("jockeys_claim", 0), errors="coerce"
        ) or 0

        surface = str(runner.get("surface_type", "")).lower().strip()
        features["surface_numeric"] = 1 if "aw" in surface else 0

        race_type = str(runner.get("race_type", "")).lower().strip()
        features["race_type_numeric"] = self._encode_race_type(race_type)

        # --- Stallion features ---
        stallion = runner.get("stallion", "")
        if stallion and stallion in self._stallion_lookup:
            features["stallion_win_rate"] = self._stallion_lookup[stallion]["win_rate"]
            features["stallion_runs"] = self._stallion_lookup[stallion]["runs"]
        else:
            features["stallion_win_rate"] = 0
            features["stallion_runs"] = 0

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

        # Official rating relative to field
        if "official_rating" in df.columns:
            or_col = df["official_rating"]
            race_max = or_col.max()
            race_median = or_col.median()
            df["or_vs_race_max"] = or_col - race_max
            df["or_vs_race_median"] = or_col - race_median

        return df

    def _add_within_race_ranks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank runners within the race on continuous metrics."""
        rank_cols = {
            # Horse career ranks
            "rNFP": "preracehorsecareerNFP",
            "horseRBrank": "preracehorsecareerRB",
            "horseFSARBrank": "preracehorsecareerFSARB",
            "horseWIVrank": "preracehorsecareerWIV",
            "horseWAXrank": "preracehorsecareerWAX",
            "horseWOArank": "preracehorsecareerWOA",
            "horseCWOrank": "preracehorsecareerCWO",
            "horseRunsrank": "preracehorsecareerRuns",
            "horseWinsrank": "preracehorsecareerWins",
            # Recent form ranks
            "rNFPLR3": "LR3NFPtotal",
            "rNFPLR5": "LR5NFPtotal",
            "rNFPLR10": "LR10NFPtotal",
            "rRWOLR3": "LR3_RWO",
            "rRWOLR5": "LR5_RWO",
            "rRWOLR10": "LR10_RWO",
            "rORR2LR": "LR_ORR2",
            # Market / probability ranks
            "rPFD3": "PFD3",
            "rPFD5": "PFD5",
            "rPFD10": "PFD10",
            "rOFS3": "OFS3",
            "rOFS5": "OFS5",
            "rOFS10": "OFS10",
            # Prize money ranks
            "rWPMRF5": "WPMRF5",
            "rPMW5": "PMW5",
            # EPF / pace ranks
            "rEPF_LR": "LR_EPF",
            "rHorseCareerEPF": "Horse_Career_EPF",
            "rJockeyEPF": "Jockey_Career_EPF",
            "rTrainerEPF": "trainer_Career_EPF",
            # Volatility
            "rFSS": "FSS",
            "rFCS": "FCS",
            "rDSLR": "FinalDSLR",
            # Jockey/trainer ranks
            "jockeyWIVrank": "preracejockeycareerWIV",
            "jockeyWAXrank": "preracejockeycareerWAX",
            "trainerWIVrank": "preracetrainercareerWIV",
            "trainerWAXrank": "preracetrainercareerWAX",
            # Trainer-jockey
            "rTJWIV": "trainerjockeycareerWIV",
            "rTJNFP": "trainerjockeycareerNFP",
            # Jockey momentum
            "jockeyLRIrank": "totalLRPjockeyindex",
            # OR rank
            "orRank": "official_rating",
        }
        for rank_name, source_col in rank_cols.items():
            if source_col in df.columns:
                df[rank_name] = df[source_col].rank(
                    ascending=False, method="min", na_option="bottom"
                )
        return df

    @staticmethod
    def _encode_sex(sex: str) -> float:
        """Encode horse sex to numeric."""
        sex_map = {
            "gelding": 1.0, "colt": 2.0, "filly": 3.0,
            "mare": 4.0, "horse": 5.0, "rig": 6.0,
        }
        return sex_map.get(sex, 0.0)

    @staticmethod
    def _encode_race_type(race_type: str) -> float:
        """Encode race type to numeric."""
        rt_map = {
            "flat": 1.0, "hurdle": 2.0, "chase": 3.0,
            "nh flat": 4.0, "bumper": 4.0,
        }
        for key, val in rt_map.items():
            if key in race_type:
                return val
        return 0.0

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
        # Within-race rank features
        rank_features = [
            # Horse career ranks
            "rNFP", "horseRBrank", "horseFSARBrank",
            "horseWIVrank", "horseWAXrank", "horseWOArank",
            "horseCWOrank", "horseRunsrank", "horseWinsrank",
            # Recent form ranks
            "rNFPLR3", "rNFPLR5", "rNFPLR10",
            "rRWOLR3", "rRWOLR5", "rRWOLR10", "rORR2LR",
            # Market / probability ranks
            "rPFD3", "rPFD5", "rPFD10",
            "rOFS3", "rOFS5", "rOFS10",
            # Prize money ranks
            "rWPMRF5", "rPMW5",
            # EPF / pace ranks
            "rEPF_LR", "rHorseCareerEPF", "rJockeyEPF", "rTrainerEPF",
            # Volatility
            "rFSS", "rFCS", "rDSLR",
            # Jockey/trainer ranks
            "jockeyWIVrank", "jockeyWAXrank",
            "trainerWIVrank", "trainerWAXrank",
            # Trainer-jockey
            "rTJWIV", "rTJNFP",
            # Other
            "jockeyLRIrank", "orRank",
        ]

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
            ]
            + rank_features
        )
        return cols
