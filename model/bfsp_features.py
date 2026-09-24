"""The BFSP model's feature list, and the pre-race context columns it needs.

Lifted out of `train_bfsp.py` for two reasons.

`model/feature_cache.py` hashes the import graph of its entry points to decide
whether a cached feature matrix is still valid, and `train_bfsp.py` was one of
them. Every edit to the trainer -- the objective, a CLI flag, a log line --
therefore invalidated hours of cached feature building that the edit could not
possibly have changed. The entry point is this module now, so the cache turns
over when the features turn over and not before.

And the categorical encoding lives here, where it can be given a fixed
vocabulary. `astype("category").cat.codes` numbers the categories present in
whatever frame it is handed, so `track_cat` for Ascot was one integer in
training and another on a live card with a different set of meetings --
`race_type_cat` is the 11th most important feature in the model by gain. The
vocabulary is now built once, stored with the model, and passed back in at
prediction time; a track the model never saw encodes to -1, which LightGBM
treats as missing, instead of silently taking some other track's code.
"""

from __future__ import annotations

import pandas as pd

from model.draw_curve import DRAW_CURVE_FEATURES, DRAW_STYLE_FEATURES
from model.draw_metrics import ALL_DRAW_FEATURES, GP_DRAW_FEATURES
from model.financial_features import FINANCIAL_FEATURES, FINANCIAL_RANK_FEATURES
from model.pace_metrics import (
    ALL_PACE_FEATURES,
    ENTITY_STYLE_FEATURES,
    HORSE_STYLE_FEATURES,
    PACE_FIT_FEATURES,
    PACE_RANK_FEATURES,
    PACE_SCENARIO_FEATURES,
    SUSTAINABILITY_FEATURES,
    TACTICAL_FEATURES,
    TRACK_PACE_BIAS_FEATURES,
)
from model.race_shape import RACE_SHAPE_FEATURES

# ---------------------------------------------------------------------------
# Feature columns: all custom metrics + engineered features
# ---------------------------------------------------------------------------

# Horse career metrics (from CustomMetricsEngine)
HORSE_CAREER_FEATURES = [
    "preracehorsecareerNFP",
    "preracehorsecareerRB",
    "preracehorsecareerFSARB",
    "preracehorsecareerFSARB2",
    "preracehorsecareerWins",
    "preracehorsecareerRuns",
    "preracehorsecareerPlaces",
    "preracehorsecareerWIV",
    "preracehorsecareerWAX",
    "preracehorsecareerWOA",
    "preracehorsecareerCWO",
    "preracehorsecareerORR2",
    "Horse_Career_EPF",
    "horsepaceindex",
]

# Last-run and rolling-window metrics
ROLLING_FEATURES = [
    "LRNFP",
    "LR3NFPtotal",
    "LR5NFPtotal",
    "LR10NFPtotal",
    "LR_ORR2",
    "LR3_ORR2",
    "LR5_ORR2",
    "LR10_ORR2",
    "LR3_RWO",
    "LR5_RWO",
    "LR10_RWO",
]

# EPF (Early Position Figure) features
EPF_FEATURES = [
    "LR_EPF",
    "LR2_EPF",
    "LR3_EPF",
    "LR4_EPF",
    "LR5_EPF",
    "LR_EPF2",
    "LR2_EPF2",
    "LR3_EPF2",
    "LR4_EPF2",
    "LR5_EPF2",
    "LR_EPF3",
    "LR2_EPF3",
    "LR3_EPF3",
    "LR4_EPF3",
    "LR5_EPF3",
    "RPS",
    "pace_pressure",
    "prom_runner",
]

# Stability and class metrics
STABILITY_FEATURES = [
    "FSS",
    "FCS",
]

# Probability-Field Difference
PFD_FEATURES = [
    "PFD3",
    "PFD5",
    "PFD10",
]

# Prize money metrics
PRIZE_FEATURES = [
    "WPMRF3",
    "WPMRF5",
    "WPMRF10",
    "PMW3",
    "PMW5",
    "PMW10",
    "RACE_WPMRF",
]

# Odds x Field Size
OFS_FEATURES = [
    "OFS1",
    "OFS3",
    "OFS5",
    "OFS10",
]

# Days Since Last Run (enhanced)
DSLR_FEATURES = [
    "DSLR1",
    "DSLR2",
    "DSLR3",
    "DSLR4",
    "DSLR12diff",
    "DSLR23diff",
    "DSLR34diff",
    "WgtDSLR",
    "FinalDSLR",
]

# Jockey momentum
LRP_FEATURES = [
    "LRPTotalScore",
    "totaljockeyLRPscore",
    "totaljockeyrides",
    "totalLRPjockeyindex",
]

# Pace metrics
PACE_FEATURES = [
    "racepacescore",
    "racepaceindex",
    "trainerpaceindex",
    "jockeypaceindex",
]

# Jockey career metrics
JOCKEY_FEATURES = [
    "preracejockeycareerWins",
    "preracejockeycareerRuns",
    "preracejockeycareerPlaces",
    "preracejockeycareerWIV",
    "preracejockeycareerWAX",
    "preracejockeycareerWOA",
    "preracejockeycareerCWO",
    "Jockey_Career_EPF",
]

# Trainer career metrics
TRAINER_FEATURES = [
    "preracetrainercareerWins",
    "preracetrainercareerRuns",
    "preracetrainercareerPlaces",
    "preracetrainercareerWIV",
    "preracetrainercareerWAX",
    "preracetrainercareerWOA",
    "preracetrainercareerCWO",
    "trainer_Career_EPF",
]

# Trainer-jockey combination metrics
TJ_FEATURES = [
    "trainerjockeycareerWIV",
    "trainerjockeycareerNFP",
    "trainerjockeyWAX",
    "trainerjockeyWOA",
    "trainerjockeyCWO",
]

# Race strength (per-race averages)
RACE_STRENGTH_FEATURES = [
    "RACE_RB",
    "RACE_WIV",
    "RACE_NFP",
    "RACE_Wins",
    "RACE_WOA",
    "LR_RACE_RB",
    "LR_RACE_WIV",
    "LR_RACE_NFP",
    "LR_RACE_Wins",
    "LR_RACE_WOA",
]

# Recency and confidence intervals
RECENCY_FEATURES = [
    "LR3COUNT",
    "LR5COUNT",
    "LR10COUNT",
    "LR3wsum",
    "LR5wsum",
    "LR10wsum",
    "CIL3",
    "CIL5",
    "CIL10",
]

# --- NEW RESEARCH-BACKED FEATURES (Benter/Woods/Ziemba/Syndicate) ---

# Exponential decay form (Benter/Woods: superior to harmonic weights)
EXPONENTIAL_DECAY_FEATURES = [
    "EXP_NFP3", "EXP_NFP5", "EXP_NFP10",
    "EXP_RB3", "EXP_RB5", "EXP_RB10",
    "EXP_ORR23", "EXP_ORR25", "EXP_ORR210",
]

# Expectation residuals (Woods/Ziemba: market-expected vs actual)
RESIDUAL_FEATURES = [
    # "NFP_residual" removed: this race's finishing position minus what the market implied. Its lagged forms remain.
    "career_residual",
    "residual_exp3",
    "residual_exp5",
    # "win_surprise" removed: won, multiplied by the price: it IS the result. Its lagged forms remain.
    "career_win_surprise",
]

# Unexposure (Syndicate: novel conditions detection)
UNEXPOSURE_FEATURES = [
    "is_debut",
    "dist_experience",
    "first_at_distance",
    "going_experience",
    "first_at_going",
    "course_experience",
    "first_at_course",
    "cd_experience",
    "first_at_cd",
    "unexposure_score",
    "dist_avg_nfp",
    "going_avg_nfp",
    "course_avg_nfp",
]

# Class movement (Ziemba: class drops are strong signals)
CLASS_MOVEMENT_FEATURES = [
    "class_change",
    "avg_class_3",
    "class_vs_avg",
    "is_class_drop",
    "is_class_rise",
]

# Distance aptitude (Benter fundamental variable)
DISTANCE_APTITUDE_FEATURES = [
    "preferred_distance",
    "dist_from_preferred",
    "dist_change_signed",
    "dist_change_lr",
]

# Going preference (Benter fundamental variable)
GOING_PREFERENCE_FEATURES = [
    "preferred_going",
    "going_from_preferred",
    "going_change_lr",
]

# Draw bias (Benter/Woods: post-position effect)
DRAW_BIAS_FEATURES = [
    "draw_relative",
    "draw_quartile",
]

# Form trajectory (Syndicate: improvement/decline detection)
FORM_TRAJECTORY_FEATURES = [
    "form_slope_3",
    "form_slope_5",
    "form_var_3",
    "form_var_5",
    "is_improving",
    "is_declining",
]

# Consistency (Ziemba: reliability measure)
CONSISTENCY_FEATURES = [
    "career_nfp_std",
    "recent_nfp_std",
    "career_place_rate",
    "career_win_rate",
    "recent_win_rate",
    "recent_place_rate",
]

# Weight differential (Benter fundamental variable)
WEIGHT_FEATURES = [
    "weight_vs_avg",
    "weight_vs_min",
    "weight_range",
    "weight_change_lr",
]

# Pedigree features (sire / damsire)
PEDIGREE_FEATURES = [
    # Sire career stats (Bayesian-shrunk)
    "sire_win_rate",
    "sire_place_rate",
    "sire_avg_nfp",
    "sire_wiv",
    "sire_runners",
    # Sire going aptitude
    "sire_going_nfp",
    "sire_going_win_rate",
    # Sire distance aptitude
    "sire_dist_nfp",
    "sire_dist_win_rate",
    # Damsire stats (Bayesian-shrunk)
    "damsire_avg_nfp",
    "damsire_win_rate",
    "damsire_going_nfp",
    "damsire_dist_nfp",
    "damsire_runners",
    # Debut interactions
    "debut_x_sire_nfp",
    "debut_x_sire_wiv",
    "debut_x_trainer_wiv",
]

# Speed figures (Benter/Mordin — from comptime_numeric)
SPEED_FEATURES = [
    "preracehorsecareerRSR",
    "LR_RSR",
    "LR3_RSR",
    "LR5_RSR",
    "best_RSR",
    "RSR_gap",
    "SFI",
    "SFI_3",
]

# Actual lengths beaten (from total_dst_bt)
LENGTHS_BEATEN_FEATURES = [
    "preracehorsecareerLB",
    "LR_LB",
    "LR3_LB",
    "LR5_LB",
    # "FSALB" removed: this race's beaten lengths, scaled by field size. Its lagged forms remain.
]

# Equipment changes (first-time headgear signals)
EQUIPMENT_FEATURES = [
    "headgear_change",
    "first_time_headgear",
    "headgear_removed",
    "has_headgear",
]

# Surface preference (turf vs all-weather)
SURFACE_FEATURES = [
    "surface_nfp",
    "surface_win_rate",
    "surface_runs",
    "first_on_surface",
]

# Track preference (course specialist detection)
TRACK_PREF_FEATURES = [
    "horse_track_runs",
    "horse_track_nfp",
    "horse_track_win_rate",
    "trainer_track_runs",
    "trainer_track_win_rate",
    "jockey_track_runs",
    "jockey_track_win_rate",
]

# OR trajectory (Ziemba — handicap mark changes)
OR_TRAJECTORY_FEATURES = [
    "or_change",
    "or_change_3",
    "career_best_or",
    "or_vs_best",
    "or_off_peak",
    "or_vs_last_win",
]

# Trainer/jockey hot form (14/30 day rolling)
HOT_FORM_FEATURES = [
    "trainer_sr_14d",
    "trainer_sr_30d",
    "trainer_runs_14d",
    "trainer_form_delta",
    "jockey_sr_14d",
    "jockey_sr_30d",
    "jockey_runs_14d",
    "jockey_form_delta",
]

# Within-race rankings
RANK_FEATURES = [
    "rNFP",
    "rNFPLR3",
    "rNFPLR5",
    "rNFPLR10",
    "horseRBrank",
    "horseFSARBrank",
    "horseFSARB2rank",
    "horseNFPrank",
    "horseWIVrank",
    "horseWAXrank",
    "horseWOArank",
    "horseCWOrank",
    "horseRunsrank",
    "horseWinsrank",
    "horsePlacesrank",
    "rORR2LR",
    "rRWOLR3",
    "rRWOLR5",
    "rRWOLR10",
    "rEPF_LR",
    "rEPF2_LR",
    "rEPF3_LR",
    "rJockeyEPF",
    "rTrainerEPF",
    "rHorseCareerEPF",
    "rDSLR",
    "rFSS",
    "rFCS",
    "rPFD3",
    "rPFD5",
    "rPFD10",
    "rWPMRF3",
    "rWPMRF5",
    "rWPMRF10",
    "rPMW3",
    "rPMW5",
    "rPMW10",
    "rOFS3",
    "rOFS5",
    "rOFS10",
    "rTJWIV",
    "rTJNFP",
    "trainerWIVrank",
    "trainerWAXrank",
    "trainerWOArank",
    "trainerCWOrank",
    "jockeyWIVrank",
    "jockeyWAXrank",
    "jockeyWOArank",
    "jockeyCWOrank",
    "jockeyLRIrank",
    # New research-backed rankings
    "rEXP_NFP5",
    "rEXP_RB5",
    "rResidual",
    "rFormSlope3",
    "rConsistency",
    "rDistApt",
    "rGoingPref",
    "rWeightVsAvg",
    "rUnexposure",
    # Pedigree rankings
    "rSireNFP",
    "rSireWIV",
    "rSireGoingNFP",
    "rSireDistNFP",
    "rDamsireNFP",
    # Speed / lengths / new feature rankings
    "rRSR",
    "rLB",
    "rSurfaceNFP",
    "rHorseTrackNFP",
    "rORChange",
    "rTrainerSR14d",
    "rJockeySR14d",
]

# Race context features (known pre-race)
CONTEXT_FEATURES = [
    "number_of_runners",
    "dist_furlongs",
    "race_class_num",
    "going_numeric",
    "surface_type_cat",
    "race_type_cat",
    "track_cat",
    "horse_sex_cat",
    "headgear_cat",
    "horse_age_num",
    "pounds_num",
    "stall_num",
    "days_since_lr_num",
    "career_runs_num",
    "or_num",
    "max_or_race",
    "median_or_num",
    "jockeys_claim_num",
    "or_vs_max",
    "or_vs_median",
]

# Run style projected from the comments on each runner's earlier runs, the
# race's shape from the field's projected styles, what the projected position
# has been worth in that shape at this course and trip (model/race_shape.py),
# and what the draw has been worth at this course, trip and field size, in
# finishing position and in pounds, overall and for the runner's style
# (model/draw_curve.py). Built by CustomMetricsEngine.calculate_all.
SHAPE_DRAW_FEATURES = RACE_SHAPE_FEATURES + DRAW_CURVE_FEATURES + DRAW_STYLE_FEATURES


def needs_race_shape(feature_cols) -> bool:
    """Whether a model reads the shape and draw-curve block. A model trained
    before the block does not, and serving it need not build the block."""
    return bool(set(feature_cols) & set(SHAPE_DRAW_FEATURES))


# Opt-in feature blocks appended by CLI flags (see main): ABM simulation
# features, Betfair market-movement features, performance-figure features.
EXTRA_FEATURE_COLS: list[str] = []

# All feature columns combined
ALL_FEATURE_COLS = (
    HORSE_CAREER_FEATURES
    + ROLLING_FEATURES
    + EPF_FEATURES
    + STABILITY_FEATURES
    + PFD_FEATURES
    + PRIZE_FEATURES
    + OFS_FEATURES
    + DSLR_FEATURES
    + LRP_FEATURES
    + PACE_FEATURES
    + JOCKEY_FEATURES
    + TRAINER_FEATURES
    + TJ_FEATURES
    + RACE_STRENGTH_FEATURES
    + RECENCY_FEATURES
    + RANK_FEATURES
    + CONTEXT_FEATURES
    + EXPONENTIAL_DECAY_FEATURES
    + RESIDUAL_FEATURES
    + UNEXPOSURE_FEATURES
    + CLASS_MOVEMENT_FEATURES
    + DISTANCE_APTITUDE_FEATURES
    + GOING_PREFERENCE_FEATURES
    + DRAW_BIAS_FEATURES
    + FORM_TRAJECTORY_FEATURES
    + CONSISTENCY_FEATURES
    + WEIGHT_FEATURES
    + PEDIGREE_FEATURES
    + SPEED_FEATURES
    + LENGTHS_BEATEN_FEATURES
    + EQUIPMENT_FEATURES
    + SURFACE_FEATURES
    + TRACK_PREF_FEATURES
    + OR_TRAJECTORY_FEATURES
    + HOT_FORM_FEATURES
    + ALL_PACE_FEATURES
    + ALL_DRAW_FEATURES
    + FINANCIAL_FEATURES
    + FINANCIAL_RANK_FEATURES
    + SHAPE_DRAW_FEATURES
)


def assert_no_post_race_features(feature_cols) -> None:
    """Refuse to train on anything that describes the race being predicted.

    The pace block once shipped five of these: RPS, pace_pressure, prom_runner,
    racepacescore and racepaceindex were aggregates of an EPF parsed from the
    horse's own in-running comment for that day's race. They carried real gain
    in training and collapsed to a constant when a live card was priced,
    because a card has no comments yet.

    The performance primitives add a second family of the same kind: a beaten
    margin, a cluster gap, a normalised finishing position and everything built
    from them are facts about how the race finished. They are legitimate inputs
    to a lagged feature about a horse's previous runs, and never inputs
    themselves."""
    from model.custom_metrics import POST_RACE_ONLY
    from model.draw_metrics import DRAW_POST_RACE_ONLY
    from model.pace_metrics import PACE_POST_RACE_ONLY
    from model.primitives import POST_RACE_PRIMITIVES
    from model.race_shape import RACE_SHAPE_POST_RACE

    # Five modules describe the race being predicted, so the guard covers all
    # five. The union lives here rather than in any one of them so none has to
    # import the others just to be checked.
    banned = (set(POST_RACE_ONLY) | set(POST_RACE_PRIMITIVES)
              | set(PACE_POST_RACE_ONLY) | set(DRAW_POST_RACE_ONLY)
              | set(RACE_SHAPE_POST_RACE))
    bad = sorted(set(feature_cols) & banned)
    if bad:
        raise ValueError(
            "These feature columns describe the race being predicted and cannot be "
            f"model inputs: {bad}. Use their lagged form instead."
        )


assert_no_post_race_features(ALL_FEATURE_COLS)


# ---------------------------------------------------------------------------
# Categorical vocabulary
# ---------------------------------------------------------------------------

CATEGORICAL_COLS = ["surface_type", "race_type", "track", "horse_sex", "headgear"]


def _levels(s: pd.Series) -> list[str]:
    """Sorted distinct values of a column, as strings, missing excluded."""
    return sorted(pd.Series(s).astype("string").dropna().unique().tolist())


def categorical_vocab(df: pd.DataFrame) -> dict[str, list[str]]:
    """The category levels to encode against, for storing with a model."""
    return {c: _levels(df[c]) for c in CATEGORICAL_COLS if c in df.columns}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_context_features(df: pd.DataFrame, vocab: dict | None = None) -> pd.DataFrame:
    """Add race context features that are known pre-race.

    `vocab` fixes the categorical encoding. Pass the vocabulary stored with the
    model so a live card encodes the way the training frame did; leave it None
    to derive one from `df` (what every caller did before, and still correct
    when the frame is the whole history)."""

    # Numeric conversions
    df["race_class_num"] = (
        df["race_class"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .pipe(pd.to_numeric, errors="coerce")
    )
    df["horse_age_num"] = pd.to_numeric(df["horse_age"], errors="coerce")
    df["pounds_num"] = pd.to_numeric(df["pounds"], errors="coerce")
    df["stall_num"] = pd.to_numeric(df["stall"], errors="coerce")
    df["days_since_lr_num"] = pd.to_numeric(
        df.get("days_since_lr", pd.Series(dtype=float)), errors="coerce"
    )
    df["career_runs_num"] = pd.to_numeric(
        df.get("career_runs", pd.Series(dtype=float)), errors="coerce"
    )
    df["or_num"] = pd.to_numeric(df["official_rating"], errors="coerce")
    df["max_or_race"] = pd.to_numeric(
        df.get("max_or_in_race", pd.Series(dtype=float)), errors="coerce"
    )
    df["median_or_num"] = pd.to_numeric(
        df.get("median_or", pd.Series(dtype=float)), errors="coerce"
    )
    df["jockeys_claim_num"] = pd.to_numeric(
        df.get("jockeys_claim", pd.Series(dtype=float)), errors="coerce"
    )

    # OR relative to field
    df["or_vs_max"] = df["or_num"] - df["max_or_race"]
    df["or_vs_median"] = df["or_num"] - df["median_or_num"]

    # Encode going description to numeric scale
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

    df["going_numeric"] = df["going_description"].apply(encode_going)

    # Encode categoricals against a fixed vocabulary.
    #
    # `astype("category").cat.codes` numbers whatever categories this frame
    # happens to contain. Training on a full history and predicting on one
    # afternoon's card are different frames, so the same track got different
    # codes in each, and the model read a feature that meant something else.
    # An unseen category encodes to -1, which LightGBM treats as missing --
    # the honest answer for a track the model has never run on.
    if vocab is None:
        vocab = categorical_vocab(df)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            cats = vocab.get(col)
            if cats is None:
                cats = _levels(df[col])
            df[f"{col}_cat"] = pd.Categorical(
                df[col].astype("string"), categories=cats
            ).codes
        else:
            df[f"{col}_cat"] = -1
    df.attrs["categorical_vocab"] = {k: list(v) for k, v in vocab.items()}

    return df
