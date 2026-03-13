"""
Pace Prediction & Running Position Feature Engineering.

Extends the existing EPF-based pace metrics with a comprehensive framework
for predicting in-race pace scenarios and running positions. Features are
designed to capture:

1. **Running Style Profiles** — Granular multi-phase position extraction from
   comments (early, mid, late) with progression/regression tracking.
2. **Pace Scenario Prediction** — Race-level pace forecasts based on the
   assembled field's historical running styles.
3. **Track/Distance Pace Bias** — Course-specific front-runner advantage with
   expanding-mean priors (lag-safe).
4. **Pace Fit** — How well a horse's preferred running style matches the
   predicted pace scenario and track bias.
5. **Tactical Metrics** — Keenness, trouble-in-running, position sustainability.

All features are strictly lag-safe (shift-1 pattern, no lookahead).
"""

import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Comment parsing — multi-phase position extraction
# ---------------------------------------------------------------------------

def parse_run_style(comment: str) -> dict:
    """Extract detailed running narrative from race comment.

    Returns dict with:
        early_pos       : 1.0–6.0 early race position
        mid_move        : -3 to +3 mid-race movement
        late_move       : -3 to +3 finishing movement
        finishing_effort: -3 to +3 effort level at finish
        was_keen        : 0/1 raced keenly / pulled hard
        had_trouble     : 0/1 hampered / denied clear run
        led_at_furlong  : int or None — when horse took lead
    """
    result = {
        "early_pos": 3.0,
        "mid_move": 0.0,
        "late_move": 0.0,
        "finishing_effort": 0.0,
        "was_keen": 0,
        "had_trouble": 0,
        "led_at_furlong": np.nan,
    }

    if not comment or not isinstance(comment, str):
        return result

    c = comment.lower().strip()

    # --- Early position (first phrase) ---
    if re.search(r"made virtually all|made all|made most", c):
        result["early_pos"] = 6.0
    elif re.search(r"soon led|led,|led early|led after|led before", c):
        result["early_pos"] = 5.8
    elif re.search(r"disputed|with leader", c):
        result["early_pos"] = 5.5
    elif re.search(r"chased leader|tracked leader|chased winner", c):
        result["early_pos"] = 5.0
    elif re.search(r"chased leaders|tracked leaders|tracked front pair|tracked leading pair", c):
        result["early_pos"] = 4.5
    elif re.search(
        r"pressed leader|prominent|close up|in touch(?! midfield)|in-touch"
        r"|pressing leaders|chasing leaders|tracking leaders",
        c,
    ):
        result["early_pos"] = 4.0
    elif re.search(r"front of mid", c):
        result["early_pos"] = 3.5
    elif re.search(r"held up in touch", c):
        result["early_pos"] = 3.5
    elif re.search(r"held up in midfield|held up in mid-division|held up in mid division", c):
        result["early_pos"] = 2.5
    elif re.search(r"mid-division|midfield|mid division", c):
        result["early_pos"] = 3.0
    elif re.search(r"held up towards rear|towards rear", c):
        result["early_pos"] = 1.5
    elif re.search(
        r"held up behind|held up|behind|last pair|in rear|always rear|always behind",
        c,
    ):
        result["early_pos"] = 1.0

    # --- Mid-race movement (with furlong markers) ---
    m = re.search(r"(rapid |good |smooth )?headway.*?(\d+)f out", c)
    if m:
        qualifier = m.group(1) or ""
        f_out = int(m.group(2))
        base = 2.5 if "rapid" in qualifier or "good" in qualifier else 1.5
        if "smooth" in qualifier:
            base = 2.0
        if f_out >= 3:
            result["mid_move"] = base
        else:
            result["late_move"] = base
    elif "headway" in c:
        result["mid_move"] = 1.5

    # --- Late movement ---
    if re.search(r"ran on|stayed on strongly|stayed on well", c):
        result["late_move"] = max(result["late_move"], 3.0)
    elif re.search(r"stayed on|kept on well", c):
        result["late_move"] = max(result["late_move"], 2.0)
    elif re.search(r"kept on", c):
        result["late_move"] = max(result["late_move"], 1.5)

    if "weakened" in c:
        result["late_move"] = min(result["late_move"], -3.0)
    elif re.search(r"no extra|one pace", c):
        result["late_move"] = min(result["late_move"], -1.5)
    elif "faded" in c:
        result["late_move"] = min(result["late_move"], -2.5)

    # --- Finishing effort ---
    if re.search(r"ridden out|driven out", c):
        result["finishing_effort"] = 3.0
    elif re.search(r"pushed out|pushed clear", c):
        result["finishing_effort"] = 2.0
    elif re.search(r"unchallenged|easily|comfortably", c):
        result["finishing_effort"] = 1.0  # positive = won easily
    elif re.search(r"never dangerous|always behind|tailed off", c):
        result["finishing_effort"] = -3.0

    # --- Keenness ---
    if re.search(r"pulled hard|raced freely|raced keenly|keen\b", c):
        result["was_keen"] = 1

    # --- Trouble in running ---
    if re.search(
        r"hampered|squeezed|short of room|bumped|checked"
        r"|denied clear run|not clear run|badly hampered|fell|brought down"
        r"|unseated|slipped",
        c,
    ):
        result["had_trouble"] = 1

    # --- Led at furlong marker ---
    m = re.search(r"led (?:over |approaching )?(\d+)f out", c)
    if m:
        result["led_at_furlong"] = float(m.group(1))
    elif re.search(r"led inside final|led close home", c):
        result["led_at_furlong"] = 0.5

    return result


# ---------------------------------------------------------------------------
# PaceMetricsEngine — integrates into CustomMetricsEngine pipeline
# ---------------------------------------------------------------------------

class PaceMetricsEngine:
    """Advanced pace feature engineering.

    Call ``calculate(df)`` after EPF and NFP have been computed.
    All features are lag-safe.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all pace feature calculations in order."""
        df = df.copy()

        # Step 1: Parse detailed run style from comments
        df = self._parse_run_styles(df)

        # Step 2: Horse running style profiles (historical)
        df = self._calc_horse_style_profiles(df)

        # Step 3: Jockey & trainer style profiles
        df = self._calc_entity_style_profiles(df)

        # Step 4: Track/distance pace bias
        df = self._calc_track_distance_pace_bias(df)

        # Step 5: Race-level pace scenario prediction
        df = self._calc_predicted_pace_scenario(df)

        # Step 6: Pace fit — style vs scenario vs track
        df = self._calc_pace_fit(df)

        # Step 7: Tactical / behavioural features
        df = self._calc_tactical_features(df)

        # Step 8: Position sustainability
        df = self._calc_position_sustainability(df)

        # Step 9: Within-race pace ranks
        df = self._calc_pace_ranks(df)

        return df

    # ------------------------------------------------------------------
    # Step 1: Parse multi-phase running style from comments
    # ------------------------------------------------------------------
    def _parse_run_styles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse comments into early_pos, mid_move, late_move, etc."""
        comment_col = "comment" if "comment" in df.columns else None
        if comment_col is None:
            for col in ["early_pos", "mid_move", "late_move", "finishing_effort",
                        "was_keen", "had_trouble", "led_at_furlong"]:
                df[col] = np.nan if col != "was_keen" and col != "had_trouble" else 0
            return df

        parsed = df[comment_col].apply(parse_run_style).apply(pd.Series)
        for col in parsed.columns:
            df[col] = parsed[col]

        # Composite: total position movement
        df["total_move"] = df["mid_move"] + df["late_move"]

        # Normalised early position (0–1 scale, field-adjusted)
        nr = df["number_of_runners"].replace(0, np.nan)
        df["early_pos_pct"] = (df["early_pos"] - 1) / 5.0  # 0=back, 1=front

        return df

    # ------------------------------------------------------------------
    # Step 2: Horse running style profiles
    # ------------------------------------------------------------------
    def _calc_horse_style_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Historical running style per horse — lag-safe expanding means."""
        df = df.sort_values(["horse_name", "race_date", "race_time"]).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        # Career average early position (lag-safe)
        df["horse_career_early_pos"] = grp["early_pos"].apply(
            lambda x: x.shift(1).expanding().mean()
        )
        # Career average late movement
        df["horse_career_late_move"] = grp["late_move"].apply(
            lambda x: x.shift(1).expanding().mean()
        )
        # Career average total movement
        df["horse_career_total_move"] = grp["total_move"].apply(
            lambda x: x.shift(1).expanding().mean()
        )
        # Career keenness rate
        df["horse_keen_rate"] = grp["was_keen"].apply(
            lambda x: x.shift(1).expanding().mean()
        )
        # Career trouble rate
        df["horse_trouble_rate"] = grp["had_trouble"].apply(
            lambda x: x.shift(1).expanding().mean()
        )

        # Running style consistency (std of early_pos — is the horse a fixed style?)
        df["horse_style_consistency"] = grp["early_pos"].apply(
            lambda x: x.shift(1).expanding().std()
        )

        # Last-run detailed metrics
        for i in range(1, 4):
            sfx = "" if i == 1 else str(i)
            df[f"LR{sfx}_early_pos"] = grp["early_pos"].shift(i)
            df[f"LR{sfx}_late_move"] = grp["late_move"].shift(i)
            df[f"LR{sfx}_total_move"] = grp["total_move"].shift(i)
            df[f"LR{sfx}_was_keen"] = grp["was_keen"].shift(i)

        # 3-run and 5-run rolling averages
        df["horse_early_pos_3r"] = grp["early_pos"].apply(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
        df["horse_late_move_3r"] = grp["late_move"].apply(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
        df["horse_early_pos_5r"] = grp["early_pos"].apply(
            lambda x: x.shift(1).rolling(5, min_periods=2).mean()
        )
        df["horse_late_move_5r"] = grp["late_move"].apply(
            lambda x: x.shift(1).rolling(5, min_periods=2).mean()
        )

        # Style trend: is horse being ridden differently recently vs career?
        df["style_shift"] = df["horse_early_pos_3r"] - df["horse_career_early_pos"]

        return df

    # ------------------------------------------------------------------
    # Step 3: Jockey & trainer running style profiles
    # ------------------------------------------------------------------
    def _calc_entity_style_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Jockey and trainer pace tendencies — lag-safe."""
        for entity_col, prefix in [
            ("jockey_name", "jockey"),
            ("trainer", "trainer"),
        ]:
            df = df.sort_values(
                [entity_col, "race_date", "race_time"]
            ).reset_index(drop=True)
            egrp = df.groupby(entity_col, group_keys=False)

            # Career average early position
            df[f"{prefix}_career_early_pos"] = egrp["early_pos"].apply(
                lambda x: x.shift(1).expanding().mean()
            )
            # Career average late movement
            df[f"{prefix}_career_late_move"] = egrp["late_move"].apply(
                lambda x: x.shift(1).expanding().mean()
            )
            # Front-runner rate (% of rides EPF > 4)
            df[f"{prefix}_front_rate"] = egrp["early_pos"].apply(
                lambda x: (x.shift(1) > 4).expanding().mean()
            )
            # Hold-up rate (% of rides EPF < 2)
            df[f"{prefix}_holdup_rate"] = egrp["early_pos"].apply(
                lambda x: (x.shift(1) < 2).expanding().mean()
            )
            # Keenness rate
            df[f"{prefix}_keen_rate"] = egrp["was_keen"].apply(
                lambda x: x.shift(1).expanding().mean()
            )

        return df

    # ------------------------------------------------------------------
    # Step 4: Track/distance pace bias
    # ------------------------------------------------------------------
    def _calc_track_distance_pace_bias(self, df: pd.DataFrame) -> pd.DataFrame:
        """Course-specific front-runner advantage — lag-safe expanding means.

        For each track+distance bucket, compute:
        - front_win_share: % of winners that were front-runners
        - holdup_win_share: % of winners that were hold-up
        - avg_winning_early_pos: average early position of winners
        """
        df = df.sort_values(["race_date", "race_time"]).reset_index(drop=True)

        # Create track-distance key (round distance to nearest furlong)
        df["_td_key"] = df["track"].astype(str) + "_" + df["dist_furlongs"].round(0).astype(str)

        # Binary: was this a front-runner win?
        df["_front_won"] = ((df["early_pos"] >= 5.0) & (df["placing_numerical"] == 1)).astype(float)
        df["_back_won"] = ((df["early_pos"] <= 2.0) & (df["placing_numerical"] == 1)).astype(float)
        df["_winner_early_pos"] = np.where(
            df["placing_numerical"] == 1, df["early_pos"], np.nan
        )

        # Lag-safe expanding mean per track-distance
        df = df.sort_values(["_td_key", "race_date", "race_time"]).reset_index(drop=True)
        td_grp = df.groupby("_td_key", group_keys=False)

        # Front-runner win share at this track+distance
        df["td_front_win_share"] = td_grp["_front_won"].apply(
            lambda x: x.shift(1).expanding().mean()
        )
        # Hold-up win share at this track+distance
        df["td_holdup_win_share"] = td_grp["_back_won"].apply(
            lambda x: x.shift(1).expanding().mean()
        )

        # Average winning early position at this track+distance
        # Only compute over winners
        df["td_avg_winner_pos"] = td_grp["_winner_early_pos"].apply(
            lambda x: x.shift(1).expanding().mean()
        )

        # Track-level (all distances) front bias
        df = df.sort_values(["track", "race_date", "race_time"]).reset_index(drop=True)
        t_grp = df.groupby("track", group_keys=False)
        df["track_front_win_share"] = t_grp["_front_won"].apply(
            lambda x: x.shift(1).expanding().mean()
        )

        # Cleanup
        df.drop(columns=["_td_key", "_front_won", "_back_won", "_winner_early_pos"],
                inplace=True)

        return df

    # ------------------------------------------------------------------
    # Step 5: Predicted pace scenario (race-level)
    # ------------------------------------------------------------------
    def _calc_predicted_pace_scenario(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict the pace scenario of a race from the assembled field.

        Uses each horse's historical running style to predict how fast
        the early pace will be. Key insight: pace is predictable because
        horses (and their connections) tend to repeat their preferred style.
        """
        df = df.sort_values(["race_date", "race_time"]).reset_index(drop=True)

        # Predicted early position for this horse (career avg as best predictor)
        df["pred_early_pos"] = df["horse_career_early_pos"].fillna(3.0)

        # Race-level pace prediction: average predicted early position of field
        df["pred_race_pace"] = df.groupby("raceid")["pred_early_pos"].transform("mean")

        # Predicted number of front-runners in race (EPF > 4)
        df["pred_n_front"] = df.groupby("raceid")["pred_early_pos"].transform(
            lambda x: (x > 4).sum()
        )
        # Predicted % of field that are front-runners
        nr = df["number_of_runners"].replace(0, np.nan)
        df["pred_front_pct"] = df["pred_n_front"] / nr * 100

        # Predicted number of hold-up runners
        df["pred_n_back"] = df.groupby("raceid")["pred_early_pos"].transform(
            lambda x: (x < 2.5).sum()
        )
        df["pred_back_pct"] = df["pred_n_back"] / nr * 100

        # Pace scenario classification (numeric for model)
        # Higher = faster predicted pace
        df["pred_pace_scenario"] = pd.cut(
            df["pred_front_pct"],
            bins=[-1, 15, 30, 50, 101],
            labels=[1, 2, 3, 4],
        ).astype(float)

        # Pace variance: std of predicted early positions in race
        # High variance = mixed pace, low = field all same style
        df["pred_pace_spread"] = df.groupby("raceid")["pred_early_pos"].transform("std")

        # Max predicted front-runner in race (the 'pace setter')
        df["pred_max_front"] = df.groupby("raceid")["pred_early_pos"].transform("max")

        return df

    # ------------------------------------------------------------------
    # Step 6: Pace fit — how well does style match conditions?
    # ------------------------------------------------------------------
    def _calc_pace_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interaction features: horse style x race pace x track bias.

        Key features:
        - pace_advantage: Does this horse's style suit the predicted pace?
        - track_style_fit: Does this horse's style suit this track?
        - pace_mismatch: Is the horse out of its comfort zone?
        """
        # Pace advantage: front-runners benefit more in slow-pace races
        # Hold-up horses benefit more in fast-pace races
        # Quantify: style_pos * (1 / predicted_pace_intensity)
        pred_ep = df["pred_early_pos"].fillna(3.0)
        pred_pace = df["pred_race_pace"].fillna(3.0)

        # Relative position: how far ahead/behind the average pace
        df["pace_position_delta"] = pred_ep - pred_pace

        # Front-runner in slow pace = big advantage
        # Hold-up in fast pace = big advantage
        # Capture as interaction:
        df["pace_advantage"] = np.where(
            pred_ep >= 4.0,
            # Front-runner: advantage when pace is slow (few other fronts)
            (5 - pred_pace),
            np.where(
                pred_ep <= 2.5,
                # Hold-up: advantage when pace is fast (many fronts to tire)
                (pred_pace - 2.5),
                0  # Mid-field: neutral
            )
        )

        # Track style fit: horse's style matches track's winning profile
        td_front = df["td_front_win_share"].fillna(0.5)
        df["track_style_fit"] = np.where(
            pred_ep >= 4.0,
            td_front,  # Front-runner at front-runner-friendly track
            np.where(
                pred_ep <= 2.5,
                df["td_holdup_win_share"].fillna(0.2),  # Hold-up at hold-up track
                0.3  # Neutral
            )
        )

        # Pace mismatch: difference between horse's typical style and
        # what's optimal at this track+distance
        td_winner_pos = df["td_avg_winner_pos"].fillna(3.5)
        df["pace_mismatch"] = np.abs(pred_ep - td_winner_pos)

        # Competition for lead: if horse wants to lead but many others do too
        df["lead_competition"] = np.where(
            pred_ep >= 4.5,
            df["pred_n_front"].fillna(1) - 1,  # -1 to exclude self
            0
        )

        # Lone front-runner indicator (strong advantage historically)
        df["lone_front_runner"] = (
            (pred_ep >= 4.5) & (df["pred_n_front"] <= 1)
        ).astype(float)

        return df

    # ------------------------------------------------------------------
    # Step 7: Tactical / behavioural features
    # ------------------------------------------------------------------
    def _calc_tactical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keenness, trouble patterns, and their predictive value."""
        df = df.sort_values(["horse_name", "race_date", "race_time"]).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        # Keenness rate over last 3 runs
        df["keen_rate_3r"] = grp["was_keen"].apply(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )

        # Was keen last run AND front-runner style -> at risk of burning out
        lr_keen = grp["was_keen"].shift(1).fillna(0)
        lr_ep = grp["early_pos"].shift(1).fillna(3)
        df["keen_front_risk"] = (lr_keen * (lr_ep >= 4)).astype(float)

        # Trouble rate (career)
        # Already computed in horse profiles, just reference
        # But add: trouble last run -> may have been unlucky
        df["LR_had_trouble"] = grp["had_trouble"].shift(1).fillna(0)

        # Jockey style match: does jockey typically ride this horse's style?
        # Difference between jockey's preferred style and horse's style
        df["jockey_horse_style_gap"] = np.abs(
            df["jockey_career_early_pos"].fillna(3) - df["horse_career_early_pos"].fillna(3)
        )

        # Trainer style match with jockey
        df["trainer_jockey_style_gap"] = np.abs(
            df["trainer_career_early_pos"].fillna(3) - df["jockey_career_early_pos"].fillna(3)
        )

        return df

    # ------------------------------------------------------------------
    # Step 8: Position sustainability
    # ------------------------------------------------------------------
    def _calc_position_sustainability(self, df: pd.DataFrame) -> pd.DataFrame:
        """Can this horse sustain its early position? Or does it fade?

        Key concept: horses that lead early but fade (negative late_move)
        are unsustainable front-runners. Track this pattern.
        """
        df = df.sort_values(["horse_name", "race_date", "race_time"]).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        # Front-runner sustainability: for horses that run prominently,
        # what's their average late movement?
        # Computed as: career avg(late_move) only when early_pos >= 4
        shifted_ep = grp["early_pos"].shift(1)
        shifted_lm = grp["late_move"].shift(1)

        front_mask = shifted_ep >= 4
        front_lm = shifted_lm.where(front_mask)
        df["front_sustainability"] = front_lm.groupby(
            df["horse_name"]
        ).expanding().mean().droplevel(0).sort_index()

        # Hold-up finishing ability: for horses that run from behind,
        # how strong is their finish?
        back_mask = shifted_ep <= 2.5
        back_lm = shifted_lm.where(back_mask)
        df["holdup_finish_ability"] = back_lm.groupby(
            df["horse_name"]
        ).expanding().mean().droplevel(0).sort_index()

        # Position drop: how much does this horse typically lose from early to finish?
        # Approximated by: (early_pos_rank - NFP) trend
        # Higher = loses more positions, lower = sustains/improves
        if "NFP" in df.columns:
            df["_pos_drop_raw"] = df["early_pos_pct"] - df["NFP"]
            df["horse_pos_drop"] = grp["_pos_drop_raw"].apply(
                lambda x: x.shift(1).expanding().mean()
            )
            df.drop(columns=["_pos_drop_raw"], inplace=True)

        return df

    # ------------------------------------------------------------------
    # Step 9: Within-race pace ranks
    # ------------------------------------------------------------------
    def _calc_pace_ranks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank horses within race on pace-related features."""
        df = df.sort_values(["race_date", "race_time", "track"]).reset_index(drop=True)

        rank_cols = {
            "rPredEarlyPos": "pred_early_pos",
            "rHorseCareerEarlyPos": "horse_career_early_pos",
            "rHorseLateMoveCareer": "horse_career_late_move",
            "rPaceAdvantage": "pace_advantage",
            "rTrackStyleFit": "track_style_fit",
            "rFrontSustain": "front_sustainability",
            "rHoldupFinish": "holdup_finish_ability",
            "rStyleConsistency": "horse_style_consistency",
        }

        for rank_name, source_col in rank_cols.items():
            if source_col in df.columns:
                df[rank_name] = df.groupby("raceid")[source_col].rank(
                    ascending=False, method="min", na_option="bottom"
                )

        return df


# ---------------------------------------------------------------------------
# Feature list for train_bfsp.py integration
# ---------------------------------------------------------------------------

# Horse running style profile features
HORSE_STYLE_FEATURES = [
    "horse_career_early_pos",
    "horse_career_late_move",
    "horse_career_total_move",
    "horse_keen_rate",
    "horse_trouble_rate",
    "horse_style_consistency",
    "LR_early_pos",
    "LR_late_move",
    "LR_total_move",
    "LR_was_keen",
    "LR2_early_pos",
    "LR2_late_move",
    "LR2_total_move",
    "LR3_early_pos",
    "LR3_late_move",
    "LR3_total_move",
    "horse_early_pos_3r",
    "horse_late_move_3r",
    "horse_early_pos_5r",
    "horse_late_move_5r",
    "style_shift",
]

# Entity style features
ENTITY_STYLE_FEATURES = [
    "jockey_career_early_pos",
    "jockey_career_late_move",
    "jockey_front_rate",
    "jockey_holdup_rate",
    "jockey_keen_rate",
    "trainer_career_early_pos",
    "trainer_career_late_move",
    "trainer_front_rate",
    "trainer_holdup_rate",
    "trainer_keen_rate",
]

# Track/distance pace bias features
TRACK_PACE_BIAS_FEATURES = [
    "td_front_win_share",
    "td_holdup_win_share",
    "td_avg_winner_pos",
    "track_front_win_share",
]

# Predicted pace scenario features
PACE_SCENARIO_FEATURES = [
    "pred_early_pos",
    "pred_race_pace",
    "pred_n_front",
    "pred_front_pct",
    "pred_n_back",
    "pred_back_pct",
    "pred_pace_scenario",
    "pred_pace_spread",
    "pred_max_front",
]

# Pace fit / interaction features
PACE_FIT_FEATURES = [
    "pace_position_delta",
    "pace_advantage",
    "track_style_fit",
    "pace_mismatch",
    "lead_competition",
    "lone_front_runner",
]

# Tactical features
TACTICAL_FEATURES = [
    "keen_rate_3r",
    "keen_front_risk",
    "LR_had_trouble",
    "jockey_horse_style_gap",
    "trainer_jockey_style_gap",
]

# Position sustainability features
SUSTAINABILITY_FEATURES = [
    "front_sustainability",
    "holdup_finish_ability",
    "horse_pos_drop",
]

# Within-race pace rankings
PACE_RANK_FEATURES = [
    "rPredEarlyPos",
    "rHorseCareerEarlyPos",
    "rHorseLateMoveCareer",
    "rPaceAdvantage",
    "rTrackStyleFit",
    "rFrontSustain",
    "rHoldupFinish",
    "rStyleConsistency",
]

# Combined list for easy import
ALL_PACE_FEATURES = (
    HORSE_STYLE_FEATURES
    + ENTITY_STYLE_FEATURES
    + TRACK_PACE_BIAS_FEATURES
    + PACE_SCENARIO_FEATURES
    + PACE_FIT_FEATURES
    + TACTICAL_FEATURES
    + SUSTAINABILITY_FEATURES
    + PACE_RANK_FEATURES
)
