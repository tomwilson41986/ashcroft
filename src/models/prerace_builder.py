"""Pre-race feature vector construction from historical data.

Takes declared runners for a future race and builds feature vectors by looking
up each entity's full history. At prediction time we only know the declared
fields (horse, jockey, trainer, course, distance, going, class, weight, draw,
field size). Everything else must be derived from past performances.
"""

import numpy as np
import pandas as pd


# Columns we extract from the horse's most recent historical appearance.
# These are cumulative/lagged metrics already computed by CustomMetricsEngine.
HORSE_CAREER_COLS = [
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
]

HORSE_RECENT_COLS = [
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
]

JOCKEY_COLS = [
    "preracejockeycareerWins",
    "preracejockeycareerRuns",
    "preracejockeycareerWIV",
    "preracejockeycareerNFP",
    "preracejockeycareerRB",
    "preracejockeycareerWAX",
    "preracejockeycareerWOA",
    "Jockey_Career_EPF",
    "jockeypaceindex",
    "totalLRPjockeyindex",
]

TRAINER_COLS = [
    "preracetrainercareerWins",
    "preracetrainercareerRuns",
    "preracetrainercareerWIV",
    "preracetrainercareerNFP",
    "preracetrainercareerRB",
    "preracetrainercareerWAX",
    "preracetrainercareerWOA",
    "trainer_Career_EPF",
    "trainerpaceindex",
]

TJ_COMBO_COLS = [
    "trainerjockeycareerWIV",
    "trainerjockeycareerNFP",
]

ALL_HISTORY_COLS = HORSE_CAREER_COLS + HORSE_RECENT_COLS + JOCKEY_COLS + TRAINER_COLS + TJ_COMBO_COLS


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
    """

    def __init__(self, history_df: pd.DataFrame):
        """
        Args:
            history_df: Complete historical results with custom metrics already
                        calculated (from CustomMetricsEngine). This is our
                        "database" of past performances.
        """
        self.history_df = history_df.copy()
        self.history_df["date"] = pd.to_datetime(self.history_df["date"])

        # Build lookup indices: most recent row per entity sorted by date
        self._horse_history = self._build_entity_index("horse_name")
        self._jockey_history = self._build_entity_index("jockey")
        self._trainer_history = self._build_entity_index("trainer")
        self._tj_history = self._build_combo_index("trainer", "jockey")

        # Course-specific horse stats
        self._horse_course_stats = self._build_course_stats()

        # Compute population medians for filling unknowns
        self._medians = self._compute_medians()

    def _build_entity_index(self, entity_col: str) -> dict[str, pd.DataFrame]:
        """Build a dict mapping entity name -> sorted history DataFrame."""
        index = {}
        for name, grp in self.history_df.groupby(entity_col):
            index[name] = grp.sort_values("date").reset_index(drop=True)
        return index

    def _build_combo_index(self, col_a: str, col_b: str) -> dict[tuple, pd.DataFrame]:
        """Build index for entity combinations (trainer-jockey)."""
        index = {}
        for key, grp in self.history_df.groupby([col_a, col_b]):
            index[key] = grp.sort_values("date").reset_index(drop=True)
        return index

    def _build_course_stats(self) -> dict[tuple, dict]:
        """Pre-compute horse course-specific win stats."""
        stats = {}
        if "course" not in self.history_df.columns:
            return stats
        for (horse, course), grp in self.history_df.groupby(["horse_name", "course"]):
            wins = grp["placing_numerical"].eq(1).sum() if "placing_numerical" in grp.columns else 0
            runs = len(grp)
            stats[(horse, course)] = {
                "course_win_pct": wins / runs if runs > 0 else 0.0,
                "course_runs": runs,
            }
        return stats

    def _compute_medians(self) -> dict[str, float]:
        """Compute population medians for all history columns (for filling unknowns)."""
        medians = {}
        for col in ALL_HISTORY_COLS:
            if col in self.history_df.columns:
                medians[col] = self.history_df[col].median()
            else:
                medians[col] = 0.0
        return medians

    def _get_last_row(self, entity_index: dict, key, race_date: pd.Timestamp) -> pd.Series | None:
        """Get the most recent row for an entity before race_date."""
        if key not in entity_index:
            return None
        hist = entity_index[key]
        prior = hist[hist["date"] < race_date]
        if prior.empty:
            return None
        return prior.iloc[-1]

    def build_features(self, declared_runners: pd.DataFrame, race_date: str) -> pd.DataFrame:
        """Build pre-race feature vectors for all declared runners.

        Args:
            declared_runners: DataFrame with columns:
                horse_name, jockey, trainer, course, distance_furlongs,
                going, race_class, weight_lbs, draw_position, number_of_runners,
                age. Optionally: bfsp (current Betfair price).
            race_date: Date string (YYYY-MM-DD) for the upcoming race.

        Returns:
            DataFrame with one row per runner and all feature columns.
        """
        race_dt = pd.Timestamp(race_date)
        rows = []

        for _, runner in declared_runners.iterrows():
            features = self._build_runner_features(runner, race_dt)
            rows.append(features)

        result = pd.DataFrame(rows)

        # Compute within-race field-level features
        result = self._add_field_features(result)
        result = self._add_within_race_ranks(result)

        return result

    def _build_runner_features(self, runner: pd.Series, race_dt: pd.Timestamp) -> dict:
        """Build feature dict for a single runner."""
        features = {}

        # Copy today's known race features
        features["horse_name"] = runner.get("horse_name", "")
        features["jockey"] = runner.get("jockey", "")
        features["trainer"] = runner.get("trainer", "")
        features["number_of_runners"] = runner.get("number_of_runners", 0)
        features["distance_furlongs"] = runner.get("distance_furlongs", 0)
        features["going_numeric"] = self._encode_going(runner.get("going", ""))
        features["race_class_numeric"] = pd.to_numeric(runner.get("race_class", 0), errors="coerce") or 0
        features["weight_lbs"] = runner.get("weight_lbs", 0)
        features["draw_position"] = runner.get("draw_position", 0)
        features["age"] = runner.get("age", 0)

        # If BFSP / current price is available
        if "bfsp" in runner.index and pd.notna(runner["bfsp"]) and runner["bfsp"] > 1.0:
            features["bfsp"] = runner["bfsp"]
        else:
            features["bfsp"] = np.nan

        # --- Horse history lookup ---
        horse_name = runner.get("horse_name", "")
        horse_last = self._get_last_row(self._horse_history, horse_name, race_dt)

        is_debut = horse_last is None
        features["is_debut"] = int(is_debut)

        if horse_last is not None:
            for col in HORSE_CAREER_COLS + HORSE_RECENT_COLS:
                features[col] = horse_last.get(col, self._medians.get(col, 0.0))

            # Last finish position and last BFSP
            features["last_finish_pos"] = horse_last.get("placing_numerical", np.nan)
            features["last_bsp"] = horse_last.get("bfsp", np.nan)

            # Last 3/5 average positions
            horse_hist = self._horse_history[horse_name]
            prior = horse_hist[horse_hist["date"] < race_dt]
            if "placing_numerical" in prior.columns:
                features["last_3_avg_position"] = prior["placing_numerical"].tail(3).mean()
                features["last_5_avg_position"] = prior["placing_numerical"].tail(5).mean()
            else:
                features["last_3_avg_position"] = np.nan
                features["last_5_avg_position"] = np.nan

            # Days since last run
            features["days_since_last_run"] = (race_dt - horse_last["date"]).days

            # Derived: going/distance/class/weight preference
            if "going_numeric" in prior.columns:
                features["going_preference"] = features["going_numeric"] - prior["going_numeric"].mean()
            else:
                features["going_preference"] = 0.0

            if "distance_furlongs" in prior.columns:
                features["distance_preference"] = features["distance_furlongs"] - prior["distance_furlongs"].mean()
            elif "distance" in prior.columns:
                features["distance_preference"] = 0.0
            else:
                features["distance_preference"] = 0.0

            features["class_change"] = features["race_class_numeric"] - (
                horse_last.get("race_class", features["race_class_numeric"])
                if pd.notna(horse_last.get("race_class"))
                else features["race_class_numeric"]
            )
            features["weight_change"] = features["weight_lbs"] - (
                horse_last.get("weight_lbs", features["weight_lbs"])
                if pd.notna(horse_last.get("weight_lbs"))
                else features["weight_lbs"]
            )

            # Course-specific stats
            course = runner.get("course", "")
            cs = self._horse_course_stats.get((horse_name, course), {})
            features["course_win_pct"] = cs.get("course_win_pct", 0.0)
            features["course_runs"] = cs.get("course_runs", 0)
        else:
            # Debut runner — fill with medians
            for col in HORSE_CAREER_COLS + HORSE_RECENT_COLS:
                features[col] = self._medians.get(col, 0.0)
            features["last_finish_pos"] = np.nan
            features["last_bsp"] = np.nan
            features["last_3_avg_position"] = np.nan
            features["last_5_avg_position"] = np.nan
            features["days_since_last_run"] = np.nan
            features["going_preference"] = 0.0
            features["distance_preference"] = 0.0
            features["class_change"] = 0.0
            features["weight_change"] = 0.0
            features["course_win_pct"] = 0.0
            features["course_runs"] = 0

        # --- Jockey history lookup ---
        jockey = runner.get("jockey", "")
        jockey_last = self._get_last_row(self._jockey_history, jockey, race_dt)
        if jockey_last is not None:
            for col in JOCKEY_COLS:
                features[col] = jockey_last.get(col, self._medians.get(col, 0.0))
        else:
            for col in JOCKEY_COLS:
                features[col] = self._medians.get(col, 0.0)

        # --- Trainer history lookup ---
        trainer = runner.get("trainer", "")
        trainer_last = self._get_last_row(self._trainer_history, trainer, race_dt)
        if trainer_last is not None:
            for col in TRAINER_COLS:
                features[col] = trainer_last.get(col, self._medians.get(col, 0.0))
        else:
            for col in TRAINER_COLS:
                features[col] = self._medians.get(col, 0.0)

        # --- Trainer-Jockey combo ---
        tj_key = (trainer, jockey)
        tj_last = self._get_last_row(self._tj_history, tj_key, race_dt)
        if tj_last is not None:
            for col in TJ_COMBO_COLS:
                features[col] = tj_last.get(col, 0.0)
        else:
            for col in TJ_COMBO_COLS:
                features[col] = 0.0

        return features

    def _add_field_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute field-level aggregate features (averages across the race)."""
        for col, agg_name in [
            ("preracehorsecareerNFP", "field_avg_career_nfp"),
            ("preracehorsecareerWIV", "field_avg_career_wiv"),
            ("preracehorsecareerRB", "field_avg_career_rb"),
        ]:
            if col in df.columns:
                df[agg_name] = df[col].mean()
            else:
                df[agg_name] = 0.0
        return df

    def _add_within_race_ranks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank every runner within the race on continuous metrics."""
        rank_cols = {
            "preracehorsecareerNFP": "rNFP",
            "LR5NFPtotal": "rNFPLR5",
            "preracehorsecareerWIV": "horseWIVrank",
            "preracehorsecareerRB": "horseRBrank",
            "LR5_RWO": "rRWOLR5",
            "LR10_RWO": "rRWOLR10",
            "PFD5": "rPFD5",
            "FSS": "rFSS",
            "FCS": "rFCS",
        }

        for src_col, rank_name in rank_cols.items():
            if src_col in df.columns:
                df[rank_name] = df[src_col].rank(ascending=False, method="min")
            else:
                df[rank_name] = np.nan

        return df

    @staticmethod
    def _encode_going(going: str) -> float:
        """Encode going description to numeric scale (firm=1 to heavy=7)."""
        if not isinstance(going, str):
            return 4.0  # default to good
        going_lower = going.lower().strip()
        # Order matters: check compound terms before simple ones
        ordered_mapping = [
            ("good to firm", 3.0),
            ("good to soft", 5.0),
            ("yielding to soft", 6.0),
            ("standard to slow", 5.0),
            ("hard", 1.0),
            ("firm", 2.0),
            ("good", 4.0),
            ("soft", 6.0),
            ("heavy", 7.0),
            ("yielding", 5.5),
            ("standard", 4.0),
            ("slow", 5.5),
        ]
        for key, val in ordered_mapping:
            if key in going_lower:
                return val
        return 4.0

    def get_feature_names(self) -> list[str]:
        """Return the list of feature column names produced by build_features."""
        # Today's race features
        today_features = [
            "number_of_runners", "distance_furlongs", "going_numeric",
            "race_class_numeric", "weight_lbs", "draw_position", "age",
            "is_debut",
        ]

        # Derived features
        derived = [
            "going_preference", "distance_preference", "class_change",
            "weight_change", "course_win_pct", "course_runs",
            "days_since_last_run", "last_finish_pos", "last_bsp",
            "last_3_avg_position", "last_5_avg_position",
        ]

        # Field-level
        field = ["field_avg_career_nfp", "field_avg_career_wiv", "field_avg_career_rb"]

        # Ranks
        ranks = [
            "rNFP", "rNFPLR5", "horseWIVrank", "horseRBrank",
            "rRWOLR5", "rRWOLR10", "rPFD5", "rFSS", "rFCS",
        ]

        return (
            HORSE_CAREER_COLS + HORSE_RECENT_COLS
            + JOCKEY_COLS + TRAINER_COLS + TJ_COMBO_COLS
            + today_features + derived + field + ranks
        )
