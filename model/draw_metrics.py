"""
Draw Bias & Stall Position Feature Engineering.

Extends the existing minimal draw features (draw_relative, draw_quartile)
with a comprehensive framework capturing:

1. **Track+Distance Draw Bias** — Lag-safe expanding mean of low vs high stall
   NFP at each track+distance combination.
2. **Going-Adjusted Draw Bias** — How ground conditions shift the draw advantage.
3. **Field-Size Scaling** — Bigger fields amplify draw effects.
4. **Stall x Pace Interaction** — Draw matters differently for front-runners
   vs hold-up horses. Low stalls enable front-running but the advantage is
   concentrated on prominent/midfield runners.
5. **Horse Personal Draw Preference** — Does this horse perform better from
   low or high stalls? (lag-safe career stats)
6. **Draw Advantage Score** — Composite: how much does this stall help/hurt
   given the track, distance, going, and field size?

All features are strictly lag-safe (shift-1 / expanding mean pattern).
Requires NFP and draw_relative/draw_quartile to be computed first.
"""

import numpy as np
import pandas as pd


class DrawMetricsEngine:
    """Advanced draw/stall feature engineering.

    Call ``calculate(df)`` after NFP, EPF, draw_relative and draw_quartile
    have been computed (i.e. after _calc_draw_bias in CustomMetricsEngine).
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all draw feature calculations in order."""
        df = df.copy()

        # Ensure prerequisites
        if "stall_num_raw" not in df.columns:
            df["stall_num_raw"] = pd.to_numeric(df.get("stall"), errors="coerce")
        if "draw_relative" not in df.columns:
            nr = df["number_of_runners"].replace(0, np.nan)
            df["draw_relative"] = (df["stall_num_raw"] - 1) / (nr - 1).replace(0, np.nan)

        # Identify flat races (stall data only meaningful for flat)
        df["_is_flat"] = (df["stall_num_raw"] > 0) & df["stall_num_raw"].notna()

        # Step 1: Track+distance draw bias
        df = self._calc_track_distance_draw_bias(df)

        # Step 2: Going-adjusted draw bias
        df = self._calc_going_draw_bias(df)

        # Step 3: Field-size draw scaling
        df = self._calc_field_size_draw(df)

        # Step 4: Stall x pace interaction
        df = self._calc_stall_pace_interaction(df)

        # Step 5: Horse personal draw preference
        df = self._calc_horse_draw_preference(df)

        # Step 6: Draw advantage composite
        df = self._calc_draw_advantage(df)

        # Step 7: Within-race draw rankings
        df = self._calc_draw_ranks(df)

        # Cleanup
        df.drop(columns=["_is_flat"], errors="ignore", inplace=True)

        return df

    # ------------------------------------------------------------------
    # Step 1: Track+Distance Draw Bias (lag-safe expanding mean)
    # ------------------------------------------------------------------
    def _calc_track_distance_draw_bias(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute historical draw bias per track+distance.

        For each track+distance, we track the expanding mean NFP
        for low-stall and high-stall runners separately. The difference
        gives the draw bias at that course configuration.
        """
        df = df.sort_values(["race_date", "race_time"]).reset_index(drop=True)

        # Track-distance key
        df["_td_key"] = (
            df["track"].astype(str) + "_" + df["dist_furlongs"].round(0).astype(str)
        )

        # Low/high stall NFP (only for flat races)
        is_low = (df["draw_relative"] <= 0.5) & df["_is_flat"]
        is_high = (df["draw_relative"] > 0.5) & df["_is_flat"]

        nfp = df.get("NFP", pd.Series(np.nan, index=df.index))
        df["_low_stall_nfp"] = np.where(is_low, nfp, np.nan)
        df["_high_stall_nfp"] = np.where(is_high, nfp, np.nan)

        df = df.sort_values(["_td_key", "race_date", "race_time"]).reset_index(drop=True)
        td_grp = df.groupby("_td_key", group_keys=False)

        # Lag-safe expanding mean NFP for each stall group at this track+distance
        df["td_low_stall_nfp"] = td_grp["_low_stall_nfp"].apply(
            lambda x: x.shift(1).expanding().mean()
        )
        df["td_high_stall_nfp"] = td_grp["_high_stall_nfp"].apply(
            lambda x: x.shift(1).expanding().mean()
        )

        # Draw bias at this track+distance (positive = low stalls favoured)
        df["td_draw_bias"] = df["td_low_stall_nfp"] - df["td_high_stall_nfp"]

        # This horse's stall vs the bias: positive = horse has the favoured draw
        df["draw_bias_alignment"] = np.where(
            is_low,
            df["td_draw_bias"],   # Low stall: positive bias helps
            np.where(is_high, -df["td_draw_bias"], 0)  # High stall: bias hurts
        )

        # Track-level (all distances) draw bias
        df = df.sort_values(["track", "race_date", "race_time"]).reset_index(drop=True)
        t_grp = df.groupby("track", group_keys=False)
        df["track_draw_bias"] = (
            t_grp["_low_stall_nfp"].apply(lambda x: x.shift(1).expanding().mean())
            - t_grp["_high_stall_nfp"].apply(lambda x: x.shift(1).expanding().mean())
        )

        # Cleanup intermediates
        df.drop(columns=["_td_key", "_low_stall_nfp", "_high_stall_nfp"],
                inplace=True)

        return df

    # ------------------------------------------------------------------
    # Step 2: Going-Adjusted Draw Bias
    # ------------------------------------------------------------------
    def _calc_going_draw_bias(self, df: pd.DataFrame) -> pd.DataFrame:
        """Draw bias modulated by going/ground conditions.

        Going shifts draw advantage: heavy ground nearly neutralises bias,
        soft amplifies it at some tracks, AW (all-weather) is consistent.
        """
        df = df.sort_values(["race_date", "race_time"]).reset_index(drop=True)

        # Going classification
        def _classify_going(g):
            if not isinstance(g, str):
                return "Unknown"
            g = g.lower()
            if "heavy" in g:
                return "Heavy"
            if "soft" in g and "good" not in g:
                return "Soft"
            if "good to soft" in g or "yielding" in g:
                return "GtS"
            if "good to firm" in g:
                return "GtF"
            if "firm" in g and "good" not in g:
                return "Firm"
            if "standard" in g or "slow" in g or "fast" in g:
                return "AW"
            return "Good"

        df["_going_cat"] = df["going_description"].apply(_classify_going)

        # Track+distance+going key
        df["_tdg_key"] = (
            df["track"].astype(str) + "_"
            + df["dist_furlongs"].round(0).astype(str) + "_"
            + df["_going_cat"]
        )

        is_low = (df["draw_relative"] <= 0.5) & df["_is_flat"]
        is_high = (df["draw_relative"] > 0.5) & df["_is_flat"]
        nfp = df.get("NFP", pd.Series(np.nan, index=df.index))

        df["_low_nfp_g"] = np.where(is_low, nfp, np.nan)
        df["_high_nfp_g"] = np.where(is_high, nfp, np.nan)

        df = df.sort_values(["_tdg_key", "race_date", "race_time"]).reset_index(drop=True)
        tdg_grp = df.groupby("_tdg_key", group_keys=False)

        df["tdg_low_stall_nfp"] = tdg_grp["_low_nfp_g"].apply(
            lambda x: x.shift(1).expanding().mean()
        )
        df["tdg_high_stall_nfp"] = tdg_grp["_high_nfp_g"].apply(
            lambda x: x.shift(1).expanding().mean()
        )
        df["tdg_draw_bias"] = df["tdg_low_stall_nfp"] - df["tdg_high_stall_nfp"]

        # Going-adjusted alignment: use tdg if enough data, fallback to td
        df["going_draw_alignment"] = np.where(
            is_low,
            df["tdg_draw_bias"].fillna(df.get("td_draw_bias", 0)),
            np.where(
                is_high,
                -df["tdg_draw_bias"].fillna(df.get("td_draw_bias", 0)),
                0,
            )
        )

        # Going draw shift: difference between going-specific and overall bias
        td_bias = df.get("td_draw_bias", pd.Series(0, index=df.index))
        df["going_draw_shift"] = df["tdg_draw_bias"] - td_bias

        df.drop(
            columns=["_going_cat", "_tdg_key", "_low_nfp_g", "_high_nfp_g",
                     "tdg_low_stall_nfp", "tdg_high_stall_nfp"],
            inplace=True,
        )

        return df

    # ------------------------------------------------------------------
    # Step 3: Field-Size Draw Scaling
    # ------------------------------------------------------------------
    def _calc_field_size_draw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Draw bias amplified by field size.

        Data shows: 13-16 runners: bias +0.019, 6-8 runners: +0.009.
        Scale the draw bias by field size bucket.
        """
        nr = df["number_of_runners"].replace(0, np.nan)

        # Field-size weight: bigger fields = stronger draw effect
        # Normalise: ~1.0 at typical field of 10
        df["draw_field_weight"] = np.clip(nr / 10.0, 0.5, 2.0)

        # Field-adjusted draw advantage
        td_bias = df.get("td_draw_bias", pd.Series(0, index=df.index)).fillna(0)
        is_low = (df["draw_relative"] <= 0.5) & df["_is_flat"]

        raw_alignment = np.where(is_low, td_bias, -td_bias)
        df["draw_advantage_field_adj"] = raw_alignment * df["draw_field_weight"]

        # Edge position: is the horse on the extreme inside (stall 1) or
        # extreme outside (last stall)? These have distinct profiles.
        df["is_stall_1"] = (df["stall_num_raw"] == 1).astype(float)
        df["is_widest_stall"] = (
            df["stall_num_raw"] == df["number_of_runners"]
        ).astype(float)

        return df

    # ------------------------------------------------------------------
    # Step 4: Stall x Pace Interaction
    # ------------------------------------------------------------------
    def _calc_stall_pace_interaction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Draw effects interact with running style.

        Key insight from data: draw bias primarily affects Prominent/Midfield
        runners (NFP delta ~0.03-0.04), not Leaders or Held-up horses.
        Low stalls enable front-running (23.9% vs 21.3% Leader rate).
        """
        dr = df["draw_relative"].fillna(0.5)

        # Predicted early position from pace features (use if available)
        pred_ep = df.get("pred_early_pos", df.get("horse_career_early_pos",
                         df.get("Horse_Career_EPF", pd.Series(3.0, index=df.index))))
        pred_ep = pred_ep.fillna(3.0)

        # Front-runner from low stall advantage
        # Low stall + front-runner style = rail advantage
        df["draw_front_advantage"] = np.where(
            pred_ep >= 4.0,
            (0.5 - dr) * 2,  # Scales -1 (widest) to +1 (innermost)
            0,
        )

        # Prominent runner draw effect (strongest bias in the data)
        df["draw_prominent_effect"] = np.where(
            (pred_ep >= 3.0) & (pred_ep < 5.0),
            (0.5 - dr) * 2,
            0,
        )

        # Hold-up runner: draw matters less (near-zero bias in data)
        df["draw_holdup_effect"] = np.where(
            pred_ep < 2.5,
            (0.5 - dr) * 0.5,  # Dampened effect
            0,
        )

        # Draw enables front-running: probability of leading given stall
        # Low stall -> more likely to get to the front
        df["draw_front_enable"] = np.clip(1.0 - dr, 0, 1)

        return df

    # ------------------------------------------------------------------
    # Step 5: Horse Personal Draw Preference
    # ------------------------------------------------------------------
    def _calc_horse_draw_preference(self, df: pd.DataFrame) -> pd.DataFrame:
        """Does this horse historically perform better from low or high stalls?

        Some horses need the rail, others cope fine wide. Track this
        as a personal preference metric (lag-safe expanding mean).
        """
        df = df.sort_values(["horse_name", "race_date", "race_time"]).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        nfp = df.get("NFP", pd.Series(np.nan, index=df.index))
        is_flat = df["_is_flat"]

        # Low-stall NFP for this horse
        is_low = (df["draw_relative"] <= 0.5) & is_flat
        is_high = (df["draw_relative"] > 0.5) & is_flat

        df["_h_low_nfp"] = np.where(is_low, nfp, np.nan)
        df["_h_high_nfp"] = np.where(is_high, nfp, np.nan)

        # Career average NFP from low/high stalls (lag-safe)
        df["horse_low_stall_nfp"] = grp["_h_low_nfp"].apply(
            lambda x: x.shift(1).expanding().mean()
        )
        df["horse_high_stall_nfp"] = grp["_h_high_nfp"].apply(
            lambda x: x.shift(1).expanding().mean()
        )

        # Personal draw preference (positive = prefers low stalls)
        df["horse_draw_pref"] = df["horse_low_stall_nfp"] - df["horse_high_stall_nfp"]

        # Current draw matches preference?
        df["draw_pref_match"] = np.where(
            is_low,
            df["horse_draw_pref"].fillna(0),     # Low stall: positive pref = good match
            np.where(
                is_high,
                -df["horse_draw_pref"].fillna(0), # High stall: negative pref = good match
                0,
            )
        )

        # Last-run draw relative (was horse wide or inside last time?)
        df["LR_draw_relative"] = grp["draw_relative"].shift(1)

        # Draw change from last run
        df["draw_change"] = df["draw_relative"] - df["LR_draw_relative"]

        df.drop(columns=["_h_low_nfp", "_h_high_nfp"], inplace=True)

        return df

    # ------------------------------------------------------------------
    # Step 6: Draw Advantage Composite
    # ------------------------------------------------------------------
    def _calc_draw_advantage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Composite draw advantage score combining all factors.

        Combines: track bias, going adjustment, field size scaling,
        pace interaction, and personal preference.
        """
        # Component weights (based on data analysis signal strength)
        td_bias = df.get("draw_bias_alignment", pd.Series(0, index=df.index)).fillna(0)
        going_adj = df.get("going_draw_alignment", pd.Series(0, index=df.index)).fillna(0)
        field_adj = df.get("draw_advantage_field_adj", pd.Series(0, index=df.index)).fillna(0)
        pace_front = df.get("draw_front_advantage", pd.Series(0, index=df.index)).fillna(0)
        pref_match = df.get("draw_pref_match", pd.Series(0, index=df.index)).fillna(0)

        # Weighted composite
        df["draw_advantage_composite"] = (
            0.30 * td_bias
            + 0.25 * going_adj
            + 0.20 * field_adj
            + 0.15 * pace_front
            + 0.10 * pref_match
        )

        return df

    # ------------------------------------------------------------------
    # Step 7: Within-race draw rankings
    # ------------------------------------------------------------------
    def _calc_draw_ranks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank horses within race on draw-related features."""
        df = df.sort_values(["race_date", "race_time", "track"]).reset_index(drop=True)

        rank_cols = {
            "rDrawBiasAlign": "draw_bias_alignment",
            "rGoingDrawAlign": "going_draw_alignment",
            "rDrawAdvComposite": "draw_advantage_composite",
            "rDrawFieldAdj": "draw_advantage_field_adj",
            "rHorseDrawPref": "horse_draw_pref",
        }

        for rank_name, source_col in rank_cols.items():
            if source_col in df.columns:
                df[rank_name] = df.groupby("raceid")[source_col].rank(
                    ascending=False, method="min", na_option="bottom"
                )

        return df


# ---------------------------------------------------------------------------
# Feature lists for train_bfsp.py integration
# ---------------------------------------------------------------------------

# Track/distance draw bias
TRACK_DRAW_FEATURES = [
    "td_draw_bias",
    "draw_bias_alignment",
    "track_draw_bias",
    "td_low_stall_nfp",
    "td_high_stall_nfp",
]

# Going-adjusted draw
GOING_DRAW_FEATURES = [
    "tdg_draw_bias",
    "going_draw_alignment",
    "going_draw_shift",
]

# Field-size draw features
FIELD_DRAW_FEATURES = [
    "draw_field_weight",
    "draw_advantage_field_adj",
    "is_stall_1",
    "is_widest_stall",
]

# Stall x pace interaction
STALL_PACE_FEATURES = [
    "draw_front_advantage",
    "draw_prominent_effect",
    "draw_holdup_effect",
    "draw_front_enable",
]

# Horse personal draw preference
HORSE_DRAW_FEATURES = [
    "horse_low_stall_nfp",
    "horse_high_stall_nfp",
    "horse_draw_pref",
    "draw_pref_match",
    "LR_draw_relative",
    "draw_change",
]

# Composite and rankings
DRAW_COMPOSITE_FEATURES = [
    "draw_advantage_composite",
    "rDrawBiasAlign",
    "rGoingDrawAlign",
    "rDrawAdvComposite",
    "rDrawFieldAdj",
    "rHorseDrawPref",
]

# Combined list for easy import
ALL_DRAW_FEATURES = (
    TRACK_DRAW_FEATURES
    + GOING_DRAW_FEATURES
    + FIELD_DRAW_FEATURES
    + STALL_PACE_FEATURES
    + HORSE_DRAW_FEATURES
    + DRAW_COMPOSITE_FEATURES
)
