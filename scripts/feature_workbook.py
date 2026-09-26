"""The feature workbook: every feature, where it comes from, whether it is served, what it is worth.

    python scripts/feature_workbook.py                       # reports/feature_inventory.xlsx
    python scripts/feature_workbook.py --importance PATH     # another model's gain file

Reads the feature lists from the code (model/bfsp_features.py and the block modules)
and the served model's split gain (data/models/bfsp_feature_importance.csv), so it
is current whenever it is re-run. Sheets:

    Summary            counts by status and block, share of gain, the top 15
    Features           one row per feature: block, group, description, module,
                       status, gain, share of gain and rank (formulas), card-safe
    Blocks             one row per block, counts and gain by formula, the evidence
    Not yet developed  proposals and ideas not built, with what each needs
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model.bfsp_features as B  # noqa: E402
from model.form_windows import FORM_WINDOW_FEATURES, MEASURES, WINDOWS  # noqa: E402
from model.shape_form import SHAPE_FORM_FEATURES  # noqa: E402

FONT = "Arial"
HEADER_FILL = PatternFill("solid", fgColor="1F3864")
BLOCK_FILL = PatternFill("solid", fgColor="D9E1F2")
INPUT_FONT = Font(name=FONT, size=10, color="0000FF")

# ---------------------------------------------------------------------------
# Blocks: which feature groups make each, where they are built, what they showed
# ---------------------------------------------------------------------------

#: group constant -> (block, readable group, module)
GROUPS = {
    "HORSE_CAREER_FEATURES": ("Custom metrics", "Horse career", "model/custom_metrics.py"),
    "ROLLING_FEATURES": ("Custom metrics", "Last run and rolling windows", "model/custom_metrics.py"),
    "EPF_FEATURES": ("Custom metrics", "Early position figure (EPF)", "model/custom_metrics.py"),
    "STABILITY_FEATURES": ("Custom metrics", "Field-size and class stability", "model/custom_metrics.py"),
    "PFD_FEATURES": ("Custom metrics", "Probability-field difference (PFD)", "model/custom_metrics.py"),
    "PRIZE_FEATURES": ("Custom metrics", "Prize money (WPMRF, PMW)", "model/custom_metrics.py"),
    "OFS_FEATURES": ("Custom metrics", "Odds x field size (OFS)", "model/custom_metrics.py"),
    "DSLR_FEATURES": ("Custom metrics", "Days since last run (DSLR)", "model/custom_metrics.py"),
    "LRP_FEATURES": ("Custom metrics", "Last-run placing index (LRP)", "model/custom_metrics.py"),
    "PACE_FEATURES": ("Custom metrics", "Race pace indices", "model/custom_metrics.py"),
    "JOCKEY_FEATURES": ("Custom metrics", "Jockey career", "model/custom_metrics.py"),
    "TRAINER_FEATURES": ("Custom metrics", "Trainer career", "model/custom_metrics.py"),
    "TJ_FEATURES": ("Custom metrics", "Trainer-jockey combination", "model/custom_metrics.py"),
    "RACE_STRENGTH_FEATURES": ("Custom metrics", "Race strength", "model/custom_metrics.py"),
    "RECENCY_FEATURES": ("Custom metrics", "Recency weights and data counts", "model/custom_metrics.py"),
    "RANK_FEATURES": ("Custom metrics", "Within-race ranks", "model/custom_metrics.py"),
    "CONTEXT_FEATURES": ("Card fields", "Today's card", "train_bfsp.py build_context_features"),
    "EXPONENTIAL_DECAY_FEATURES": ("Research-backed metrics", "Exponentially decayed form", "model/custom_metrics.py"),
    "RESIDUAL_FEATURES": ("Research-backed metrics", "Results against the market", "model/custom_metrics.py"),
    "UNEXPOSURE_FEATURES": ("Research-backed metrics", "Unexposure (new conditions)", "model/custom_metrics.py"),
    "CLASS_MOVEMENT_FEATURES": ("Research-backed metrics", "Class movement", "model/custom_metrics.py"),
    "DISTANCE_APTITUDE_FEATURES": ("Research-backed metrics", "Distance aptitude", "model/custom_metrics.py"),
    "GOING_PREFERENCE_FEATURES": ("Research-backed metrics", "Going preference", "model/custom_metrics.py"),
    "DRAW_BIAS_FEATURES": ("Research-backed metrics", "Draw position", "model/custom_metrics.py"),
    "FORM_TRAJECTORY_FEATURES": ("Research-backed metrics", "Form trajectory", "model/custom_metrics.py"),
    "CONSISTENCY_FEATURES": ("Research-backed metrics", "Consistency and strike rates", "model/custom_metrics.py"),
    "WEIGHT_FEATURES": ("Research-backed metrics", "Weight carried", "model/custom_metrics.py"),
    "PEDIGREE_FEATURES": ("Research-backed metrics", "Pedigree (sire, damsire)", "model/custom_metrics.py"),
    "SPEED_FEATURES": ("Research-backed metrics", "Speed ratings (RSR)", "model/custom_metrics.py"),
    "LENGTHS_BEATEN_FEATURES": ("Research-backed metrics", "Lengths beaten", "model/custom_metrics.py"),
    "EQUIPMENT_FEATURES": ("Research-backed metrics", "Headgear", "model/custom_metrics.py"),
    "SURFACE_FEATURES": ("Research-backed metrics", "Surface form", "model/custom_metrics.py"),
    "TRACK_PREF_FEATURES": ("Research-backed metrics", "Course form", "model/custom_metrics.py"),
    "OR_TRAJECTORY_FEATURES": ("Research-backed metrics", "Official-rating trajectory", "model/custom_metrics.py"),
    "HOT_FORM_FEATURES": ("Research-backed metrics", "Trainer and jockey recent form", "model/custom_metrics.py"),
    "ALL_PACE_FEATURES": ("Pace and running style", "Pace, style, position", "model/pace_metrics.py"),
    "ALL_DRAW_FEATURES": ("Draw and stall", "Draw bias and stall position", "model/draw_metrics.py"),
    "FINANCIAL_FEATURES": ("Financial-style form", "Momentum, volatility, drawdown", "model/financial_features.py"),
    "FINANCIAL_RANK_FEATURES": ("Financial-style form", "Within-race ranks of the financial features",
                                "model/financial_features.py"),
    "INTENT_SERVED_FEATURES": ("Intent (card-safe)", "Connections' choices and the yard's record with them",
                               "model/intent_features.py"),
    "SERVED_FRESHNESS_FEATURES": ("Freshness", "Days since the last run, in context",
                                  "model/freshness_features.py"),
    "SERVED_FORM_WINDOW_FEATURES": ("Form windows", "Every measure over seven windows", "model/form_windows.py"),
}

#: Drop-in blocks (model/blocks/): (module, block label, group, status)
DROP_IN = [
    ("form_variants", "Form variants", "NFP and lengths beaten in every variant", "candidate"),
    ("race_relative", "Within-race readings", "Strongest form measures against this field", "candidate"),
    ("race_relative_wide", "Wide within-race readings", "Every ranked input as a distance from this field",
     "candidate"),
    ("time_figure", "Time figure", "The horse's own time against standard, going-adjusted", "candidate"),
    ("exposure", "Exposure", "Exposure, improvement, and what horses like it went on to do", "candidate"),
    ("head_to_head", "Collateral form", "Each runner's earlier meetings with today's rivals", "candidate"),
    ("shrunk_rates", "Shrunk records", "Small-sample records pulled toward the level above", "built"),
    ("rank_fix", "Rank fix", "The four misdirected ranks, lowest first, missing last", "built"),
    ("bookings", "Bookings", "The jockey and the yard as the market rated their runners", "candidate"),
    ("comments", "Comments", "What the in-running comments of earlier runs say", "built"),
    ("stable", "Stablemates", "A yard's runners today, their order, and whose jockey it books", "built"),
    ("dam_line", "Dam line", "The dam's produce record, the siblings and the female family", "built"),
    ("sire_apt", "Sire aptitudes", "Sire and damsire aptitudes against their own level, and the nick", "built"),
    ("elo", "Finishing-order rating", "Each horse's strength from whom it beat, across the race network", "built"),
    ("cond_form", "Condition form", "Form at today's trip, going, course, race type and headgear, against form "
     "everywhere", "built"),
    ("form_lines", "Form lines", "How the rivals from its recent races have done since", "candidate"),
    ("form_lines_ab", "Form lines split", "Rivals ahead of it and behind it, and form lines against today's field",
     "built"),
    ("form_uplift", "Form uplift", "What the rivals from its recent races have run since, in figures, marks and "
     "prices, against what they ran there", "built"),
    ("race_relative_new", "Within-race readings (newer blocks)", "Time figure, exposure and three-run windows "
     "against this field", "built"),
    ("draw_v2", "Draw v2", "Draw by course, trip, going and stall placement", "built"),
    ("pace_v2", "Pace v2", "Early position and race shape from sharper projections", "built"),
]

BLOCK_EVIDENCE = {
    "Custom metrics": "The 19 proprietary metrics and their ranks (the original model). Within-race ranks carry about "
                      "40% of gain; rPMW3 alone about 14%.",
    "Card fields": "Today's card as fetched at 06:00 (card fill: model/card_enrich.py, parity checked).",
    "Research-backed metrics": "Benter/Woods/Ziemba-style additions to the engine (served since the first retrain).",
    "Pace and running style": "Style and pace from in-running comments (rebuilt parser, 23 Sep).",
    "Draw and stall": "Draw bias cells and stall position (track_draw_bias is blank on every row: open defect).",
    "Financial-style form": "RSI, MACD, z-score, Sharpe, drawdown of finishing-position series.",
    "Intent (card-safe)": "Iteration 24: price-forecast error -0.0052 on the 501; iteration 28 (with freshness, as "
                          "served): -0.0085, Brier skill vs market +0.0013. Served from 25 Sep.",
    "Freshness": "Iteration 26: -0.0049, Brier skill vs market +0.0010 (resolved); leak probe clean. Served from 25 Sep.",
    "Form windows": "Iteration 29: -0.0073 on the 535 (rank 1 -0.0092), Brier skill vs market +0.0022 (resolved), "
                    "early-price rule +6.01% -> +6.50%; iteration 32 (with shape form, 6000 rounds): -0.0128, Brier "
                    "skill +0.0032 (resolved); leak probe clean. Served from 26 Sep (the 615).",
    "Shape form (shape/draw remodel)": "Iteration 30: -0.0051 on the 535, Brier skill vs market +0.0018 (resolved); "
                                       "iteration 34: -0.0001 on top of the form windows, which carry what it found. "
                                       "Not served.",
    "Form variants": "Iteration 34: -0.0021 (-0.0037 to -0.0006) beyond the form windows, rank 1 -0.0046. Iteration "
                     "54 (steadier screen, on the 853): -0.0022 (-0.0036 to -0.0008), five times the placebo. "
                     "Iteration 56 (served recipe, on the 853): -0.0025 (-0.0034 to -0.0018); iteration 59: -0.0013 "
                     "beyond form lines. In the 944 candidate.",
    "Within-race readings": "Iteration 35: -0.0074 (-0.0092 to -0.0057), every rank band resolved (Lessmann, Sung "
                            "and Johnson 2009, eq. 14). Iteration 36 (served recipe, 6000 rounds): -0.0078 beyond "
                            "the 615. Trained and verified as the 675 (train-bfsp run 33); not served.",
    "Wide within-race readings": "Iteration 37: -0.0012 alone (unresolved); with the time figure and exposure "
                                 "-0.0038 (resolved), -0.0056 at another seed (iteration 39), -0.0063 on the 615 + "
                                 "within-race readings (iteration 41). Iteration 42 (served recipe, 6000 rounds): "
                                 "-0.0045 with the time figure and exposure, -0.0053 with collateral form as well.",
    "Time figure": "Iteration 37: -0.0005 alone, ranks 1-3 better and outsiders worse; Brier skill vs market "
                   "+0.0021 and concordance +0.0035 (both resolved). Part of iteration 37's -0.0038.",
    "Exposure": "Iteration 37: -0.0003 alone; part of iteration 37's -0.0038 with the time figure and wide readings.",
    "Shrunk records": "Iteration 38: -0.00002 on the price. Iteration 54 (steadier screen): -0.0013 (-0.0028 to "
                      "+0.0001): marginal.",
    "Collateral form": "Iteration 41: -0.0019 alone and -0.0020 on top of all3 (all3 + collateral form -0.0083 on the "
                       "615 + within-race readings), Brier skill vs market +0.0036, beyond the seed floor. Iteration "
                       "42 (served recipe): all3 + collateral form -0.0053 (-0.0064 to -0.0041). Trained and "
                       "verified as the 853 (train-bfsp run 34), dry-run clean on a real card.",
    "Form lines": "Iteration 58 (steadier screen, on the 853): -0.0029 (-0.0044 to -0.0016), Brier skill vs "
                  "market +0.0018 and concordance +0.0026 (both resolved), early-price rule +6.35% -> +7.41%. "
                  "Iteration 59 (served recipe, on the 859): -0.0038 (-0.0047 to -0.0028), Brier skill +0.0017 "
                  "(resolved), rule +8.09% -> +8.81%; with form variants -0.0051, rule +8.94%. In the 944 candidate.",
    "Form lines split": "Iteration 60 (steadier screen): -0.0007 (-0.0020 to +0.0007) beyond form lines, Brier skill "
                        "and concordance worse (resolved): the split and the field readings add nothing. Retired.",
    "Condition form": "Iteration 57 (steadier screen, on the 853): -0.0002 (-0.0017 to +0.0013), unresolved: the "
                      "horse's form in today's conditions adds nothing the engine's aptitude features lack. Retired.",
    "Form uplift": "Iteration 62 (steadier screen, on the 853): being screened, alone and beside form lines.",
    "Rank fix": "Iteration 41: -0.0000; iteration 54 (steadier screen): +0.0006, with the four originals withheld: "
                "the trees had worked round the defect. Hygiene for the next engine rebuild.",
    "Bookings": "Quick recipe: -0.0027, -0.0028, -0.0038 on three feature sets. Iteration 51 (steadier screen, on "
                "the 853): -0.0044 (-0.0060 to -0.0029), concordance +0.0029, the placebo -0.0004. Iteration 48 "
                "(served recipe, on the 853): -0.0041 (-0.0051 to -0.0032), early-price rule +7.83% -> +8.09%. "
                "Trained and verified as the 859 (train-bfsp run 35), dry-run clean.",
    "Comments": "Iteration 45 (quick): +0.0019, inside the quick recipe's placebo bar; iteration 52 (steadier "
                "screen): -0.0001 (-0.0014 to +0.0011). Nothing the market and the pace features do not read. Retired.",
    "Stablemates": "Iteration 47 (quick): +0.0018, inside the placebo bar; iteration 52 (steadier screen): -0.0007 "
                   "(-0.0020 to +0.0007). Retired; iteration 48 also carries it with bookings on the served recipe.",
    "Dam line": "Iteration 49 (quick): +0.0008; iteration 52 (steadier screen): -0.0001, -0.0002 in maiden, novice "
                "and bumper races. Retired.",
    "Sire aptitudes": "Iteration 49 (quick): +0.0002, -0.0032 in maiden, novice and bumper races; iteration 52 "
                      "(steadier screen): -0.0006 overall, -0.0001 in those races: the split was the reshuffle. Retired.",
    "Finishing-order rating": "Iteration 55 (steadier screen): +0.0001 (-0.0013 to +0.0014), -0.0015 in maiden, "
                              "novice and bumper races: official ratings and the form blocks already carry it. Retired.",
    "Within-race readings (newer blocks)": "Iteration 44: +0.0007 (-0.0009 to +0.0022), the outsiders worse: the "
                                           "trees already read these measures against the field. Retired.",
    "Draw v2": "Iteration 35: -0.0004; iteration 54 (steadier screen): -0.0009 (-0.0023 to +0.0003): nothing.",
    "Pace v2": "Iteration 35: +0.0009; iteration 54 (steadier screen): -0.0007 (-0.0021 to +0.0005): nothing.",
    "Shape and draw (old)": "Iterations 18 and 25: no gain to the price forecast (-0.0000), a little concordance lost. "
                            "Built by the engine, not served; the inputs shape form reads.",
    "Intent (card-unsafe)": "The 06:00 card cannot know these (a gelding since the last run; the jockey or claim as "
                            "ridden). The post-race guard refuses them as inputs.",
}

SERVED, CANDIDATE, BUILT, EXCLUDED = ("Served", "Candidate (built, tested, promotion pending)",
                                      "Built, not served", "Excluded (not knowable at 06:00)")

# ---------------------------------------------------------------------------
# Descriptions
# ---------------------------------------------------------------------------

METRIC = {
    "NFP": "normalised finishing position (1 = won, 0 = last)",
    "RB": "races-beaten score (identical to NFP in this data)",
    "FSARB": "field-size-adjusted RB",
    "FSARB2": "mean squared field-size-adjusted RB",
    "Wins": "wins", "Runs": "runs", "Places": "first-three finishes",
    "WIV": "win index: wins / wins expected by chance (sum of 1/runners)",
    "WAX": "wins above chance per run (won - 1/runners)",
    "WOA": "wins over average (equal to WAX in this build)",
    "CWO": "cumulative wins above chance",
    "ORR2": "market rating: runners / BSP (the market's chance against a random runner's)",
    "RWO": "recency-weighted ORR2 mean (harmonic weights)",
    "EPF": "early position figure from the in-running comment (1 rear .. 6 led)",
    "EPF2": "early position figure adjusted for field size",
    "EPF3": "early position figure weighted by the finishing position",
    "RSR": "speed rating against the standard time (from the race's winning time)",
    "LB": "lengths beaten",
}

EXPLICIT = {
    # custom metrics
    "Horse_Career_EPF": "Horse's career mean EPF2 (early position figure) over earlier runs",
    "horsepaceindex": "Horse's pace index from its earlier early-position figures",
    "LR3NFPtotal": "Mean NFP over the last 3 runs", "LR5NFPtotal": "Mean NFP over the last 5 runs",
    "LR10NFPtotal": "Mean NFP over the last 10 runs", "LRNFP": "NFP in the last run",
    "LR_ORR2": "Market rating (runners / BSP) in the last run",
    "LR3_ORR2": "Harmonic-weighted sum of ORR2 over the last 3 runs",
    "LR5_ORR2": "Harmonic-weighted sum of ORR2 over the last 5 runs",
    "LR10_ORR2": "Harmonic-weighted sum of ORR2 over the last 10 runs",
    "LR3_RWO": "Harmonic-weighted mean ORR2 over the last 3 runs",
    "LR5_RWO": "Harmonic-weighted mean ORR2 over the last 5 runs",
    "LR10_RWO": "Harmonic-weighted mean ORR2 over the last 10 runs",
    "RPS": "Race pace score: sum of the field's expected early positions",
    "pace_pressure": "Share of the field expected to race prominently (expected EPF > 4)",
    "prom_runner": "Expected to race prominently (career EPF > 4)",
    "FSS": "Field-size stability: RMS change in field size over the last 10 runs",
    "FCS": "Field-class stability: RMS change in the race's mean OR over the last 10 runs",
    "PFD3": "Recency-weighted (BSP probability - 1/runners) over the last 3 runs",
    "PFD5": "Recency-weighted (BSP probability - 1/runners) over the last 5 runs",
    "PFD10": "Recency-weighted (BSP probability - 1/runners) over the last 10 runs",
    "WPMRF3": "Recency-weighted prize money raced for, last 3 runs",
    "WPMRF5": "Recency-weighted prize money raced for, last 5 runs",
    "WPMRF10": "Recency-weighted prize money raced for, last 10 runs",
    "PMW3": "Performance-adjusted prize money (prize x RB^2), recency-weighted, last 3 runs",
    "PMW5": "Performance-adjusted prize money (prize x RB^2), recency-weighted, last 5 runs",
    "PMW10": "Performance-adjusted prize money (prize x RB^2), recency-weighted, last 10 runs",
    "RACE_WPMRF": "Today's race's total prize money",
    "OFS1": "Odds x field size in the last run (runners / BSP; equals LR_ORR2)",
    "OFS3": "Recency-weighted odds x field size, last 3 runs",
    "OFS5": "Recency-weighted odds x field size, last 5 runs",
    "OFS10": "Recency-weighted odds x field size, last 10 runs",
    "DSLR1": "Days since the last run (dates held)", "DSLR2": "Days between the 2nd- and last-but-one runs",
    "DSLR3": "Days since the 3rd-last run", "DSLR4": "Days since the 4th-last run",
    "DSLR12diff": "Change in spacing between the last two gaps", "DSLR23diff": "Change in spacing (gaps 2-3), half weight",
    "DSLR34diff": "Change in spacing (gaps 3-4), third weight", "WgtDSLR": "Weighted acceleration of run spacing",
    "FinalDSLR": "Days since the last run adjusted by the spacing trend",
    "LRPTotalScore": "Last-run placing index (placed = first three)",
    "totaljockeyLRPscore": "Jockey's cumulative last-run placing score", "totaljockeyrides": "Jockey's cumulative rides",
    "totalLRPjockeyindex": "Jockey's last-run placing index",
    "racepacescore": "Race pace score (equals RPS)", "racepaceindex": "Race pace index from the field's expected pace",
    "trainerpaceindex": "Trainer's pace index (runners' early positions)",
    "jockeypaceindex": "Jockey's pace index (mounts' early positions)",
    "Jockey_Career_EPF": "Jockey's career mean EPF2 of mounts", "trainer_Career_EPF": "Trainer's career mean EPF2 of runners",
    "RACE_RB": "Field's mean career RB (race strength)", "RACE_WIV": "Field's mean career WIV",
    "RACE_NFP": "Field's mean career NFP", "RACE_Wins": "Field's mean career wins", "RACE_WOA": "Field's mean career WOA",
    "LR_RACE_RB": "Strength (mean career RB) of the field in the last run", "LR_RACE_WIV": "Mean career WIV of the last run's field",
    "LR_RACE_NFP": "Mean career NFP of the last run's field", "LR_RACE_Wins": "Mean career wins of the last run's field",
    "LR_RACE_WOA": "Mean career WOA of the last run's field",
    "LR3COUNT": "Runs held in the last 3 slots", "LR5COUNT": "Runs held in the last 5 slots",
    "LR10COUNT": "Runs held in the last 10 slots", "LR3wsum": "Recency weight sum, last 3",
    "LR5wsum": "Recency weight sum, last 5", "LR10wsum": "Recency weight sum, last 10",
    "CIL3": "Data count index (last 3)", "CIL5": "Data count index (last 5)", "CIL10": "Data count index (last 10)",
    # card fields
    "number_of_runners": "Declared runners", "dist_furlongs": "Race distance (furlongs)",
    "race_class_num": "Race class (1 = highest)", "going_numeric": "Going on the card, as a number",
    "surface_type_cat": "Surface (turf / all-weather), category code", "race_type_cat": "Race type, category code",
    "track_cat": "Course, category code", "horse_sex_cat": "Sex, category code (from the last run on the card)",
    "headgear_cat": "Headgear worn today, category code", "horse_age_num": "Age", "pounds_num": "Weight carried (lbs)",
    "stall_num": "Stall (draw)", "days_since_lr_num": "HRB's days since the last run",
    "career_runs_num": "Career runs before today", "or_num": "Official rating",
    "max_or_race": "Highest official rating in the race", "median_or_num": "Median official rating in the race",
    "jockeys_claim_num": "Jockey's claim (lbs)", "or_vs_max": "Official rating less the race's highest",
    "or_vs_median": "Official rating less the race's median",
    # research-backed
    "career_residual": "Career mean of (NFP - NFP expected from the market's order)",
    "residual_exp3": "Exponentially decayed NFP residual vs the market, last 3 runs",
    "residual_exp5": "Exponentially decayed NFP residual vs the market, last 5 runs",
    "career_win_surprise": "Career mean of wins at long prices (won x log BSP)",
    "is_debut": "First run", "dist_experience": "Earlier runs at today's trip", "first_at_distance": "First run at the trip",
    "going_experience": "Earlier runs on today's going", "first_at_going": "First run on the going",
    "course_experience": "Earlier runs at the course", "first_at_course": "First run at the course",
    "cd_experience": "Earlier runs at the course and distance", "first_at_cd": "First run at the course and distance",
    "unexposure_score": "Combined novelty of today's conditions", "dist_avg_nfp": "Mean NFP at the trip",
    "going_avg_nfp": "Mean NFP on the going", "course_avg_nfp": "Mean NFP at the course",
    "class_change": "Class change from the last run", "avg_class_3": "Mean class of the last 3 runs",
    "class_vs_avg": "Today's class against the recent mean", "is_class_drop": "Dropping in class",
    "is_class_rise": "Rising in class",
    "preferred_distance": "Trip of the horse's best runs", "dist_from_preferred": "Today's trip less the preferred",
    "dist_change_signed": "Trip change from the last run (signed)", "dist_change_lr": "Absolute trip change from the last run",
    "preferred_going": "Going of the horse's best runs", "going_from_preferred": "Today's going less the preferred",
    "going_change_lr": "Going change from the last run", "draw_relative": "Stall relative to the field",
    "draw_quartile": "Stall quartile",
    "form_slope_3": "Slope of NFP over the last 3 runs (+ improving)", "form_slope_5": "Slope of NFP over the last 5 runs",
    "form_var_3": "Variance of NFP over the last 3 runs", "form_var_5": "Variance of NFP over the last 5 runs",
    "is_improving": "Form slope over 3 runs > 0.05", "is_declining": "Form slope over 3 runs < -0.05",
    "career_nfp_std": "Career standard deviation of NFP", "recent_nfp_std": "Standard deviation of NFP, last 5 runs",
    "career_place_rate": "Horse's career place (top-three) rate over earlier runs",
    "career_win_rate": "Horse's career win rate over earlier runs",
    "recent_win_rate": "Win rate over the last 10 runs", "recent_place_rate": "Place rate over the last 10 runs",
    "weight_vs_avg": "Weight carried less the field's mean", "weight_vs_min": "Weight carried less the field's lowest",
    "weight_range": "The field's weight range", "weight_change_lr": "Weight change from the last run",
    "sire_win_rate": "Sire's progeny win rate (shrunk)", "sire_place_rate": "Sire's progeny place rate",
    "sire_avg_nfp": "Sire's progeny mean NFP", "sire_wiv": "Sire's progeny WIV", "sire_runners": "Sire's progeny runs",
    "sire_going_nfp": "Sire's progeny mean NFP on today's going", "sire_going_win_rate": "Sire's win rate on the going",
    "sire_dist_nfp": "Sire's progeny mean NFP at the trip", "sire_dist_win_rate": "Sire's win rate at the trip",
    "damsire_avg_nfp": "Damsire's progeny mean NFP", "damsire_win_rate": "Damsire's progeny win rate",
    "damsire_going_nfp": "Damsire's mean NFP on the going", "damsire_dist_nfp": "Damsire's mean NFP at the trip",
    "damsire_runners": "Damsire's progeny runs", "debut_x_sire_nfp": "Debutant x sire's mean NFP",
    "debut_x_sire_wiv": "Debutant x sire's WIV", "debut_x_trainer_wiv": "Debutant x trainer's WIV",
    "best_RSR": "Best career speed rating", "RSR_gap": "Best speed rating less the mean of the last 3",
    "SFI": "Speed-figure improvement (last run less the one before)", "SFI_3": "Mean speed-figure improvement, last 3",
    "headgear_change": "Headgear changed from the last run", "first_time_headgear": "First-time headgear",
    "headgear_removed": "Headgear taken off", "has_headgear": "Wears headgear today",
    "surface_nfp": "Mean NFP on today's surface", "surface_win_rate": "Win rate on the surface",
    "surface_runs": "Runs on the surface", "first_on_surface": "First run on the surface",
    "horse_track_runs": "Runs at the course", "horse_track_nfp": "Mean NFP at the course",
    "horse_track_win_rate": "Win rate at the course", "trainer_track_runs": "Trainer's runners at the course",
    "trainer_track_win_rate": "Trainer's strike rate at the course", "jockey_track_runs": "Jockey's rides at the course",
    "jockey_track_win_rate": "Jockey's strike rate at the course",
    "or_change": "Official rating change from the last run", "or_change_3": "OR change over the last 3 runs",
    "career_best_or": "Highest official rating held", "or_vs_best": "Official rating less the career best",
    "or_off_peak": "Points below the peak rating", "or_vs_last_win": "Official rating less the rating at the last win",
    "trainer_sr_14d": "Trainer's strike rate, last 14 days", "trainer_sr_30d": "Trainer's strike rate, last 30 days",
    "trainer_runs_14d": "Trainer's runners, last 14 days", "trainer_form_delta": "Trainer's 14-day less 30-day strike rate",
    "jockey_sr_14d": "Jockey's strike rate, last 14 days", "jockey_sr_30d": "Jockey's strike rate, last 30 days",
    "jockey_runs_14d": "Jockey's rides, last 14 days", "jockey_form_delta": "Jockey's 14-day less 30-day strike rate",
    # financial
    "form_rsi_5": "RSI of NFP over 5 runs", "form_rsi_10": "RSI of NFP over 10 runs",
    "form_z_score": "Last NFP against the career mean, in standard deviations", "nfp_macd": "MACD of the NFP series",
    "nfp_macd_signal": "MACD signal line", "mean_reversion_score": "Distance of recent form from the long-run mean",
    "form_sharpe_3": "Mean / sd of NFP over 3 runs", "form_sharpe_5": "Mean / sd of NFP over 5 runs",
    "nfp_acceleration": "Change in the NFP slope", "prize_momentum": "Trend in prize money raced for",
    "class_momentum": "Trend in class", "win_density_30d": "Wins in the last 30 days",
    "win_density_60d": "Wins in the last 60 days", "layoff_adjusted_form": "Recent form discounted by days off",
    "or_momentum": "Trend in official rating", "nfp_skew_5": "Skew of NFP over 5 runs",
    "drawdown_from_peak_nfp": "Recent NFP less the career-best NFP",
    # freshness
    "fr_usual_gap": "The horse's usual spacing: recency-weighted mean of log(1 + gap) before earlier runs",
    "fr_gap_vs_usual": "Today's log gap less the usual (+ longer break than usual)",
    "fr_gap_vs_trainer": "Today's log gap less the yard's usual gap",
    "fr_gap_rel_race": "Today's log gap less the field's mean",
    "fr_runs_since_break": "0 = first run back from 60+ days (or debut), 1 = second run back, ... (capped at 8)",
    "fr_break_len": "log(1 + days) of that break", "fr_runs_30d": "Runs in the last 30 days",
    "fr_runs_60d": "Runs in the last 60 days", "fr_runs_90d": "Runs in the last 90 days",
    "fr_quick_after_win": "Back within 14 days of a win", "fr_days_since_win": "Days since the last win held",
    "fr_runs_since_win": "Runs since the last win", "fr_days_since_placed": "Days since the last top-three finish",
    "fr_days_since_code": "Days since the last run in today's code (flat, AW, hurdle, chase, bumper)",
    # intent
    "it_new_yard": "First run for a new trainer", "it_hcap_debut": "Handicap debut",
    "it_ft_headgear": "First-time headgear item", "it_layoff": "Return from a 90+ day break",
    "it_class_drop": "Dropping in class", "it_code_switch": "First run in a code (hurdles, fences, AW)",
    "it_debut": "Debut", "it_runs_for_yard": "Runs for the current yard", "it_seen_runs": "Runs held for the horse",
    "it_class_change": "Class change from the last run", "it_n_angles": "Number of intent angles today",
    "it_ae_sum": "Sum of the trainer's A-E records for today's angles", "it_ae_best": "Best of the trainer's A-E records for today's angles",
    "it_gelded": "Gelded since the last run (not knowable at 06:00)",
    "it_ae_gelded": "Trainer's A-E with first run after gelding (not knowable at 06:00)",
    "it_jockey_upgrade": "Jockey as ridden stronger than usual (the 06:00 card has the booking)",
    "it_claim_change": "Claim as ridden changed (the 06:00 card has the booking)",
    # shape form
    "sf_speed_inside": "Expected leaders (sum of p_lead) drawn in lower stalls",
    "sf_speed_outside": "Expected leaders drawn in higher stalls",
    "sf_speed_near": "Expected leaders within two stalls either side",
    "sf_draw_ae": "Market's miss (won - BSP chance) for runners drawn in this cell, earlier days",
    "sf_pos_ae": "Market's miss for runners projected to race where this one is, in this projected shape",
}

EXPLICIT.update({
    # draw and stall (model/draw_metrics.py)
    "draw_adj": "Stall renumbered after non-runners are taken out", "draw_pct_adj": "Adjusted stall as a share of the field (0 = lowest)",
    "draw_vacancy_inside": "Empty stalls inside this runner", "draw_nr_shift": "Places the non-runners moved the stall in",
    "track_handed": "Course handedness", "is_left_handed": "Left-handed course", "is_right_handed": "Right-handed course",
    "rail_move_yards": "Rail movement today (yards)", "rail_moved": "Rail moved today",
    "rail_trust": "Trust in the draw bias given the rail movement", "draw_x_handed": "Stall x handedness (inside for the direction)",
    "draw_x_direction": "Stall x course direction", "draw_bias_ev": "Draw bias for the stall band here: NFP beyond what ratings expected",
    "draw_bias_ev_adj": "The same on the adjusted stall", "draw_bias_ev_stall": "Per-stall draw surface, shrunk toward coarser cells",
    "draw_bias_ev_course": "Course-level draw bias", "draw_bias_ev_n_eff": "Effective runners behind the draw bias",
    "draw_bias_stall_n_eff": "Effective runners behind the per-stall surface",
    "draw_bias_confident": "Draw bias is confident (enough races, rail not moved)",
    "draw_bias_gradient": "Slope of the bias from low to high stalls", "draw_bias_spread": "Best less worst stall bias in the race",
    "td_draw_screen": "Split-half stability of the course x distance draw bias", "track_draw_screen": "Split-half stability of the course draw bias",
    "td_dsh_level": "Draw bias level from the split-half screen", "draw_x_style": "Stall x projected running style",
    "draw_x_pace": "Stall x race pace", "draw_x_fieldsize": "Stall x field size", "draw_pos_fit": "Fit of the stall to the projected position",
    "draw_lead_fit": "Fit of the stall to leading", "td_draw_bias": "Course x distance: low-stall less high-stall NFP",
    "draw_bias_alignment": "Stall on the favoured side of the course x distance bias", "track_draw_bias": "Course low-less-high stall bias (blank on every row: open defect)",
    "td_low_stall_nfp": "Mean NFP from low stalls at course x distance", "td_high_stall_nfp": "Mean NFP from high stalls at course x distance",
    "tdg_draw_bias": "Course x distance x going draw bias", "going_draw_alignment": "Stall on the favoured side for today's going",
    "going_draw_shift": "How today's going shifts the draw bias", "draw_field_weight": "Field-size weight on the draw effect",
    "draw_advantage_field_adj": "Draw advantage scaled by field size", "is_stall_1": "Drawn in stall 1", "is_widest_stall": "Drawn widest",
    "draw_front_advantage": "Draw advantage for front-runners here", "draw_prominent_effect": "Draw effect for prominent racers",
    "draw_holdup_effect": "Draw effect for hold-up horses", "draw_front_enable": "Stall that makes front-running easy",
    "horse_low_stall_nfp": "Horse's mean NFP from low stalls", "horse_high_stall_nfp": "Horse's mean NFP from high stalls",
    "horse_draw_pref": "Horse's low-less-high stall NFP", "draw_pref_match": "Today's stall matches the horse's preference",
    "LR_draw_relative": "Relative draw in the last run", "draw_change": "Change in relative draw from the last run",
    "draw_advantage_composite": "Composite draw advantage (course, trip, going, field size)",
    # pace and style (model/pace_metrics.py)
    "horse_career_early_pos": "Horse's career mean early position", "horse_career_late_move": "Career mean places gained late",
    "horse_career_total_move": "Career mean places gained through the race", "horse_keen_rate": "Share of runs described as keen",
    "horse_trouble_rate": "Share of runs with trouble in running", "horse_style_consistency": "How consistent the horse's early position is",
    "LR_early_pos": "Early position in the last run", "LR_late_move": "Places gained late in the last run",
    "LR_total_move": "Places gained through the last run", "LR_was_keen": "Keen in the last run",
    "LR2_early_pos": "Early position two runs back", "LR2_late_move": "Late move two runs back", "LR2_total_move": "Total move two runs back",
    "LR3_early_pos": "Early position three runs back", "LR3_late_move": "Late move three runs back", "LR3_total_move": "Total move three runs back",
    "horse_early_pos_3r": "Mean early position, last 3 runs", "horse_late_move_3r": "Mean late move, last 3 runs",
    "horse_early_pos_5r": "Mean early position, last 5 runs", "horse_late_move_5r": "Mean late move, last 5 runs",
    "style_shift": "Recent early position less the career mean (style change)", "horse_epf_norm_mean": "Mean normalised early position",
    "horse_epf_norm_recent": "Recent normalised early position", "horse_epf_norm_sd": "Spread of normalised early position",
    "horse_pos_gain_mean": "Mean positions gained from early to finish", "horse_modal_style": "Most frequent running style",
    "horse_style_mode_share": "Share of runs in the most frequent style", "horse_style_inflexible": "Always races the same way",
    "LR_epf_norm": "Normalised early position in the last run", "jockey_career_early_pos": "Jockey's mean early position of mounts",
    "jockey_career_late_move": "Jockey's mean late move of mounts", "jockey_front_rate": "Jockey's share of mounts ridden prominently",
    "jockey_holdup_rate": "Jockey's share of mounts held up", "jockey_keen_rate": "Jockey's share of mounts keen",
    "trainer_career_early_pos": "Trainer's mean early position of runners", "trainer_career_late_move": "Trainer's mean late move of runners",
    "trainer_front_rate": "Trainer's share of runners prominent", "trainer_holdup_rate": "Trainer's share of runners held up",
    "trainer_keen_rate": "Trainer's share of runners keen", "td_front_win_share": "Course x distance: share of winners that raced prominently",
    "td_holdup_win_share": "Course x distance: share of winners that were held up", "td_avg_winner_pos": "Course x distance: winners' mean early position",
    "track_front_win_share": "Course: share of winners that raced prominently", "pred_early_pos": "Projected early position today",
    "pred_race_pace": "Projected pace of today's race", "pred_n_front": "Projected number of front-runners",
    "pred_front_pct": "Projected share of the field on the pace", "pred_n_back": "Projected number of hold-up horses",
    "pred_back_pct": "Projected share held up", "pred_pace_scenario": "Projected pace scenario (slow / even / fast)",
    "pred_pace_spread": "Spread of projected early positions", "pred_max_front": "Strongest front-runner's projected position",
    "pred_epf_norm": "Projected normalised early position", "predicted_lead_prob": "Projected chance of leading",
    "lead_prob_rank": "Rank of the chance of leading in the field", "lead_prob_top": "Most likely leader",
    "lead_prob_entropy": "How open the lead is (entropy of lead chances)", "pred_pace_pressure": "Projected pace pressure",
    "pred_pace_share": "Share of the projected pace", "pred_pace_contested": "Projected contested lead",
    "pred_race_pace_class": "Projected pace class", "pace_position_delta": "Projected position less the course's winning position",
    "pace_advantage": "Advantage of the projected position in the projected pace", "track_style_fit": "Fit of the style to the course's winners",
    "pace_mismatch": "Mismatch of style and projected pace", "lead_competition": "Rivals for the lead",
    "lone_front_runner": "The only projected front-runner", "pace_suit": "How well the projected pace suits the style",
    "course_front_edge": "Course edge for front-runners", "lead_prob_edge": "Lead chance less the next best",
    "keen_rate_3r": "Keen rate, last 3 runs", "keen_front_risk": "Keen and drawn to race prominently",
    "LR_had_trouble": "Trouble in running last time", "jockey_horse_style_gap": "Jockey's usual style less the horse's",
    "trainer_jockey_style_gap": "Trainer's usual style less the jockey's", "front_sustainability": "Ability to sustain a lead",
    "holdup_finish_ability": "Ability to finish from off the pace", "horse_pos_drop": "Positions usually lost late",
    "td_pcs_mean": "Course x distance mean pace-change score", "track_pcs_mean": "Course mean pace-change score",
    # old shape and draw (model/race_shape.py, model/draw_curve.py)
    "p_lead": "Chance of leading (projected style)", "p_prom": "Chance of racing prominently", "p_mid": "Chance of racing mid-division",
    "p_rear": "Chance of racing in rear", "pred_epf": "Projected early position (0 led .. 1 last)", "style_n_eff": "Runs behind the style projection",
    "shape_exp_leaders": "Expected number of leaders in the race", "shape_p_no_leader": "Chance nobody takes the lead",
    "shape_p_contested": "Chance the lead is contested", "shape_front_share": "Share of the field expected on the pace",
    "shape_lead1": "Largest chance of leading in the race", "shape_lead2": "Second largest chance of leading",
    "shape_lead_clarity": "How clear the likely leader is", "shape_mean_epf": "Field's mean projected early position",
    "shape_bin": "Race shape: 0 no natural leader, 1 one, 2 contested", "lead_share": "Share of the field's claims on the lead",
    "lead_rank": "Rank of the chance of leading", "rel_epf": "Projected early position less the field's mean",
    "pv_exp_nfp": "Value of the projected position in this shape here (NFP)", "pv_exp_lbs": "Value of the projected position in this shape here (lbs)",
    "pv_act_nfp": "Chance-weighted value of each actual position here (NFP)", "pv_act_lbs": "Chance-weighted value of each actual position here (lbs)",
    "pv_front_bias_lbs": "Lead less rear value in this shape here (lbs)", "pv_n_eff": "Runners behind the position value",
    "dc_draw_pct": "Stall position, 0 lowest .. 1 highest", "dc_edge_nfp": "Draw curve: the stall's worth in NFP here",
    "dc_edge_lbs": "Draw curve: the stall's worth in pounds", "dc_edge_rel_lbs": "Stall's worth less the race's mean",
    "dc_race_spread_lbs": "Best less worst stall in the race (lbs)", "dc_n_eff": "Runners behind the draw curve",
    "dc_edge_style_lbs": "Stall's worth for the projected style (lbs)", "dc_edge_style_rel_lbs": "The same less the race's mean",
})

#: Within-race ranks: rank column -> source column, read from the engines' rank tables
RANK_SOURCE: dict[str, str] = {}
for _mod in ("custom_metrics", "pace_metrics", "draw_metrics"):
    _src = (ROOT / "model" / f"{_mod}.py").read_text()
    RANK_SOURCE.update(dict(re.findall(r'"((?:r[A-Z]|horse|trainer|jockey)[A-Za-z0-9_]*)":\s*"([A-Za-z0-9_]+)"', _src)))
for _a, _b in re.findall(r'"([a-z][a-z0-9_]*)":\s*"(r[A-Z][A-Za-z0-9_]*)"', (ROOT / "model" / "financial_features.py").read_text()):
    RANK_SOURCE[_b] = _a

INTENT_ANGLE = {"new_yard": "a new yard", "hcap_debut": "a handicap debut", "ft_headgear": "first-time headgear",
                "layoff": "a return from a break", "class_drop": "a class drop", "code_switch": "a code switch",
                "debut": "a debut"}

FW_MEASURE = {
    "win": "wins", "plc": "places (first three)", "wax": "wins above chance (won - 1/N)",
    "nfp": "normalised finishing position", "lbs": "pounds beaten at the trip (capped 20)",
    "lbsc": "pounds better than the race's average runner", "perf": "rating-scale performance figure (lbs)",
    "rsr": "speed rating (RSR)", "mkt": "the market's view, log(N x BSP chance)",
    "ae": "wins above the market (won - BSP chance)", "nres": "finishing position against the market's order",
}
FW_WINDOW = {"car": "career mean", "l1": "last run", "m3": "mean of the last 3 runs", "m5": "mean of the last 5 runs",
             "w3": "last 3 runs weighted 3-2-1 by recency", "w5": "last 5 runs weighted 5..1 by recency",
             "w10": "last 10 runs weighted 10..1 by recency"}

PACE_TOKENS = [
    ("early_pos", "early position"), ("late_move", "late move (places gained late)"),
    ("total_move", "total move through the race"), ("keen", "keen (pulled hard)"), ("trouble", "trouble in running"),
    ("style", "running style"), ("epf_norm", "normalised early position"), ("pos_gain", "positions gained"),
    ("front", "front-running"), ("holdup", "held-up"), ("pace", "pace"), ("lead", "lead"),
    ("pcs", "pace-change score"), ("winner_pos", "winners' early position"), ("sustain", "sustaining a lead"),
    ("finish", "finishing from off the pace"), ("pos_drop", "positions lost"),
]


def _metric(x: str) -> str:
    return METRIC.get(x, x)


def _upper_first(d: str) -> str:
    """Capitalise a description's first letter only (str.capitalize lower-cases BSP, NFP, OR)."""
    return d[:1].upper() + d[1:]


def _lower_first(d: str) -> str:
    """Lower-case a description's first letter unless it opens with an acronym (RSI, NFP, OR)."""
    first = d.split(" ", 1)[0]
    return d if len(first) > 1 and first[:2].isupper() else d[:1].lower() + d[1:]


COMMENT_CLASS = {
    "trouble": "trouble in running (hampered, no clear run, checked)", "switched": "switched to find room",
    "slow_start": "a slow start (slowly away, dwelt)", "keen": "racing keen", "wide": "racing wide",
    "finished_well": "running on late", "tender": "a kind ride or greenness (eased, not knocked about)",
    "weakened": "weakening (faded, no extra)", "no_finish": "not finishing (pulled up, fell, unseated)",
    "problem": "a physical problem (lost action, bled, lost a shoe)", "easy_win": "an easy win",
    "excuse": "any excuse (trouble, slow start, keen, wide or a problem)",
}
CM_WINDOW = {"lr": "in the last run", "l3": "share of the last 3 runs", "l6": "share of the last 6 runs"}
EXPLICIT_BLOCKS = {
    "cm_excuse_close": "An excuse last time and beaten 5 lengths or less",
    "cm_excuse_nfp": "An excuse last time x its normalised finishing position",
    "cm_noexcuse_poor": "No excuse last time and in the bottom half of the field",
    "cm_tender_close": "A kind ride last time and beaten 5 lengths or less",
    "cm_finished_well_nfp": "Ran on last time x its normalised finishing position",
    "cm_easy_win": "Won easily last time",
    "cm_runs_since_trouble": "Runs since the last one with trouble in running",
    "cm_runs_since_excuse": "Runs since the last one with an excuse",
    "bk_jk_mkt": "Jockey's rides as the market rated them (decayed, shrunk to 0)",
    "bk_jk_upgrade": "Today's jockey against the jockeys of the horse's last three runs, as the market rated them",
    "bk_jk_same": "Today's jockey rode the last run",
    "bk_jk_rides_on_horse": "Earlier rides of this horse by today's jockey",
    "bk_tr_mkt": "Trainer's runners as the market rated them (decayed, shrunk to 0)",
    "bk_tr_mkt_trend": "Trainer's runners as the market rated them over the last fortnight, less the long view",
    "st_n_race": "Trainer's runners in this race", "st_n_meeting": "Trainer's runners at this meeting today",
    "st_n_day": "Trainer's runners anywhere today",
    "st_or_rank": "Rank by official rating among the trainer's runners in the race (1 = highest)",
    "st_or_gap": "Official rating less the best of the trainer's runners in the race",
    "st_jk_share": "Today's jockey's share of the trainer's rides on earlier days (decayed)",
    "st_jk_first": "Gets the yard's first-choice jockey among its stablemates in the race",
    "st_jk_gap": "Today's jockey's share of the yard's rides less the best among its stablemates' jockeys",
    "elo": "Rating from finishing orders (Elo, Flat and jumps apart), as it stood before the day",
    "elo_runs": "Finishes against other finishers behind the rating",
    "elo_z": "Rating against today's field (z-score over rated runners)",
    "elo_gap": "Rating less the best in today's field",
    "elo_trend": "Rating now less its value three finishes ago",
    "elo_field": "Today's field's mean rating",
    "h2h_rivals_met": "Today's rivals met before", "h2h_meetings": "Earlier meetings with today's rivals",
    "h2h_win_share": "Share of meetings with today's rivals finished ahead (shrunk to a half)",
    "h2h_net": "Meetings with today's rivals finished ahead less behind",
    "h2h_lbs": "Pounds ahead of today's rivals over earlier meetings (shrunk to 0)",
    "h2h_last_lbs": "Pounds ahead of today's rivals at the latest meeting",
    "h2h_last_net": "Rivals ahead of less behind at the latest meeting",
}
RR_PREFIX = {"rr": "Within-race reading", "rw": "Wide within-race reading", "rn": "Within-race reading (newer blocks)"}
CF_COND = {"trip": "today's trip band", "going": "today's going group", "course": "this course",
           "hcap": "today's race type (handicap or not)", "gear": "today's headgear status (on or off)"}
CF_MEASURE = {"perf": "performance figure", "nfp": "normalised finishing position", "mkt": "the market's view (BSP)"}
CF_STAT = {"car": "career mean", "m3": "mean of the last three",
           "d": "career mean less its mean everywhere, shrunk by n / (n + 3)"}
FL_WINDOW = {"l1": "its last race", "l3": "its last three races"}
FL_STAT = {"n": "rival runs since (each rival's next three, before today)", "wins": "rival wins since",
           "wr": "rivals' win rate since, shrunk to 1 in 10 by five runs",
           "plc": "rivals' place rate since, shrunk to 3 in 10 by five runs",
           "ae": "rivals' wins less their BSP chances since, per run, shrunk to 0"}
FU_STAT = {"perf": "rivals' performance figures since less theirs in the race (lb), per run, shrunk to 0",
           "or": "rivals' official ratings since less theirs that day (the handicapper's reassessment), "
                 "per run, shrunk to 0",
           "mkt": "the market's view of the rivals since less its view there, per run, shrunk to 0",
           "nres": "rivals' finishing positions since against the market's order, per run, shrunk to 0"}


def describe(f: str, group: str = "") -> str:
    if f in EXPLICIT:
        return EXPLICIT[f]
    if f in EXPLICIT_BLOCKS:
        return EXPLICIT_BLOCKS[f]
    m = re.fullmatch(r"cm_(lr|l3|l6)_([a-z_]+)", f)
    if m and m.group(2) in COMMENT_CLASS:
        return f"In-running comment: {COMMENT_CLASS[m.group(2)]}, {CM_WINDOW[m.group(1)]} (earlier days)"
    m = re.fullmatch(r"(rr|rw|rn)_(\w+)_(z|gap)", f)
    if m:
        how = "z-score against today's field" if m.group(3) == "z" else "gap to the best in today's field"
        return f"{RR_PREFIX[m.group(1)]}, {how}: {_lower_first(describe(m.group(2)))}"
    m = re.fullmatch(r"cf_(trip|going|course|hcap|gear)_n", f)
    if m:
        return f"Earlier runs at {CF_COND[m.group(1)]}"
    m = re.fullmatch(r"cf_(trip|going|course|hcap|gear)_(perf|nfp|mkt)_(car|m3|d)", f)
    if m:
        return f"At {CF_COND[m.group(1)]}: {CF_MEASURE[m.group(2)]}, {CF_STAT[m.group(3)]} (earlier days)"
    m = re.fullmatch(r"fl_(l1|l3)_(n|wins|wr|plc|ae)", f)
    if m:
        return f"Form lines of {FL_WINDOW[m.group(1)]}: {FL_STAT[m.group(2)]}"
    m = re.fullmatch(r"fa_(ahead|behind)_(n|wins|ae)", f)
    if m:
        side = "finished ahead of it" if m.group(1) == "ahead" else "finished behind it (or did not finish)"
        what = {"n": "runs since", "wins": "wins since", "ae": "wins less BSP chances since, per run, shrunk to 0"}
        return f"Form lines of its last race, rivals that {side}: {what[m.group(2)]}"
    if f == "fa_beat_winner":
        return "1 if a rival it beat in its last race has won since, else 0"
    m = re.fullmatch(r"fa_(l1|l3)_(ae|wr)_(z|gap)", f)
    if m:
        how = "z-score against today's field" if m.group(3) == "z" else "gap to the best in today's field"
        return f"Form lines of {FL_WINDOW[m.group(1)]} ({'wins less chances' if m.group(2) == 'ae' else 'win rate'}), {how}"
    m = re.fullmatch(r"fu_(l1|l3)_(perf|or|mkt|nres)", f)
    if m:
        return f"Form uplift of {FL_WINDOW[m.group(1)]}: {FU_STAT[m.group(2)]}"
    if f == "fu_l1_adj_vs_or":
        return ("Its performance figure in its last race, revised by the rivals' figures since, "
                "less today's official rating")
    m = re.fullmatch(r"rfix_(\w+)", f)
    if m:
        return f"Within-race rank of {m.group(1)}, lowest first, missing last"
    m = re.fullmatch(r"fw_([a-z]+)_(car|l1|m3|m5|w3|w5|w10)", f)
    if m:
        return f"{_upper_first(FW_MEASURE.get(m.group(1), m.group(1)))}: {FW_WINDOW[m.group(2)]} (earlier days only)"
    m = re.fullmatch(r"fw_perf_(l1|w5|car)_vs_or", f)
    if m:
        return f"Performance figure ({FW_WINDOW[m.group(1)]}) less today's official rating (+ = well in)"
    m = re.fullmatch(r"sf_(adj|bias)_(car|l1|m3|m5|w3|w5|w10)", f)
    if m:
        what = ("Form net of the pace and draw each run met (lbs better than the average runner less the credit)"
                if m.group(1) == "adj" else "Credit the pace and draw handed each run (+ = helped, lbs)")
        return f"{what}: {FW_WINDOW[m.group(2)]}"
    m = re.fullmatch(r"it_ae_([a-z_]+)", f)
    if m and m.group(1) in INTENT_ANGLE:
        return f"Trainer's shrunk A-E (won - BSP chance) with {INTENT_ANGLE[m.group(1)]}, earlier days"
    m = re.fullmatch(r"preracehorsecareer(\w+)", f)
    if m:
        return f"Horse's career {_metric(m.group(1))} over earlier runs"
    m = re.fullmatch(r"prerace(trainer|jockey)career(\w+)", f)
    if m:
        return f"{m.group(1).capitalize()}'s career {_metric(m.group(2))} over earlier days"
    m = re.fullmatch(r"trainerjockey(?:career)?(\w+)", f)
    if m:
        return f"Trainer-jockey combination's career {_metric(m.group(1))}"
    m = re.fullmatch(r"LR(\d?)_(EPF\d?)", f)
    if m:
        n = m.group(1)
        return f"{_upper_first(_metric(m.group(2)))} in the {'last' if not n else ('%s-back' % n)} run"
    m = re.fullmatch(r"(LR\d*_)?(\w+?)_?(RSR|LB)", f)
    if m and f.startswith(("LR", "preracehorse")):
        n = re.match(r"LR(\d*)", f).group(1)
        return f"{_upper_first(_metric(m.group(3)))}: {'last run' if not n else 'mean of the last %s runs' % n}"
    m = re.fullmatch(r"EXP_(NFP|RB|ORR2)(\d+)", f)
    if m:
        return f"Exponentially decayed (0.85) mean {_metric(m.group(1))}, last {m.group(2)} runs"
    if f in RANK_SOURCE and RANK_SOURCE[f] != f:
        src = RANK_SOURCE[f]
        d = describe(src)
        return f"Within-race rank (1 = highest in the field) of {src}: {_lower_first(d)}"
    if f.startswith("r") and f[1:2].isupper() or f.endswith("rank"):
        base = f[1:] if f.startswith("r") and f[1:2].isupper() else f[:-4]
        return f"Within-race rank of {base} (1 = highest in the field)"
    if group == "ALL_DRAW_FEATURES":
        return "Draw: " + f.replace("_", " ")
    if group == "ALL_PACE_FEATURES":
        words = [w for t, w in PACE_TOKENS if t in f]
        who = ("Jockey's " if f.startswith("jockey") else "Trainer's " if f.startswith("trainer") else
               "Track x distance " if f.startswith("td_") else "Course " if f.startswith("track") else
               "Projected, today's race: " if f.startswith(("pred_", "predicted", "lead_prob")) else
               "Last run's " if f.startswith("LR_") else "Two runs back: " if f.startswith("LR2") else
               "Three runs back: " if f.startswith("LR3") else "Horse's ")
        return who + (", ".join(words) if words else f.replace("_", " "))
    return f.replace("_", " ")


def card_safe(f: str) -> str:
    return "No" if f in B.INTENT_CARD_UNSAFE else "Yes"


# ---------------------------------------------------------------------------
# Not yet developed
# ---------------------------------------------------------------------------

NOT_BUILT = [
    # (id, feature, category, what it measures, data, notes, priority)
    ("B2", "Field-size-adjusted lengths beaten (lagged)", "Form", "Lengths beaten scaled by field size, over past runs",
     "Available", "The unlagged version leaked the result and was removed. fw_lbs / fw_lbsc now cover pounds beaten.",
     "Low"),
    ("B3", "Lengths per position", "Form", "Lengths beaten per place behind the winner (how strung out the field was)",
     "Available", "Not built", "Medium"),
    ("B4", "Closing-sectional proxy", "Form", "Speed over the final furlongs", "Not available",
     "No sectional or per-horse times in the data", "Blocked"),
    ("D2", "Headgear type flags", "Equipment", "One flag per headgear item (blinkers, visor, cheekpieces, hood, tongue-tie)",
     "Available", "Partial: one category column", "Low"),
    ("D3", "Form with / without headgear", "Equipment", "The horse's form when wearing today's headgear vs without",
     "Available", "Not built", "Medium"),
    ("F1", "SP-to-BSP spread (lagged)", "Market", "How far industry SP differed from BSP in past runs",
     "Available", "The same-race version added nothing (markets block)", "Low"),
    ("F2", "Win/place price ratio (lagged)", "Market", "Place-market view relative to the win market in past runs",
     "Partial (Betfair files from Jan 2026)", "Opt-in behind --market-features", "Medium"),
    ("G2", "Horse handicap / non-handicap form", "Form", "The horse's form split by handicap and non-handicap",
     "Available", "Only trainer and jockey splits exist (opt-in)", "Medium"),
    ("G4", "Age restriction", "Race conditions", "Age band of the race (2yo, 3yo only, 3yo+, 4yo+)", "Available",
     "race_restrictions_age is never read", "Low"),
    ("H3", "Claimer flag", "Connections", "Whether the jockey claims an allowance", "Available",
     "Partial: raw claim only; is_claimer is opt-in", "Low"),
    ("I1", "OR percentile in the field", "Ratings", "Official rating's percentile within the race", "Available",
     "Partial: difference to the max and median only", "Low"),
    ("I2", "OR trajectory slope", "Ratings", "Slope of the official rating over recent runs", "Available",
     "Partial: changes only", "Low"),
    ("J1", "Month / season", "Context", "Seasonal effects", "Available",
     "Surveyed on 24 Sep: nothing for the price to miss", "Low"),
    ("J2", "Day of week", "Context", "Weekday effects (strength of meetings)", "Available", "Surveyed: nothing", "Low"),
    ("J3", "Horse seasonality", "Form", "The horse's form by time of year", "Available", "Surveyed: nothing", "Low"),
    ("J4", "Campaign stage", "Form", "Run number within the current campaign", "Available",
     "Written in model/primitives.py, never called; fr_runs_since_break now covers most of it", "Low"),
    ("K1", "Major-race experience", "Class", "Runs and form in Group / Listed / major races", "Available",
     "`major` feeds only the opt-in connections block", "Medium"),
    ("K2", "Class ladder vs career median", "Class", "Today's class against the horse's career median class",
     "Available", "Partial: against a three-run mean", "Low"),
    ("L2", "Class x speed", "Speed", "Speed figures scaled by the class they were run in", "Available",
     "Not built", "Medium"),
    ("L4", "Weight per OR point", "Handicap", "Weight carried relative to the official rating", "Available",
     "The handicap block's hc_wt_vs_mark is close (opt-in, added nothing against the result)", "Medium"),
    ("R1", "Trainer and jockey career NFP / RB", "Connections", "Career finishing-position quality of runners and rides",
     "Available", "Never built, so five spec rank columns are skipped: trainerRBrank, trainerNFPrank, jockeyRBrank, "
     "jockeyNFPrank, horsexRBMARrank", "Medium"),
    ("N1", "Connection form windows", "Connections",
     "The window ladder (career, last 3/5, weighted 3/5/10) for trainer and jockey: A-E, strike rate, pounds beaten",
     "Available", "New idea: form windows gave the largest gain of any block for the horse", "High"),
    ("N2", "Condition-specific form windows", "Form",
     "Form windows restricted to runs at today's trip band, going, course and code", "Available",
     "New idea: extends form windows and the unexposure features", "High"),
    ("N3", "Re-test research blocks on the price forecast", "Research",
     "Perf figures, Kalman ratings, pedigree suite, connections, comments, handicap angles, A-E by entity",
     "Available", "Each was tested only against the result (nothing beyond BSP); freshness, intent and form windows "
     "all failed that test yet sharpened the BSP forecast", "High"),
    ("N4", "Head-to-head ratings", "Form", "Pairwise results between today's rivals (who has beaten whom, by how much)",
     "Available (h2h fetch)", "data/h2h exists; not a model feature", "Medium"),
    ("N5", "Jockey booking strength (card-safe)", "Connections",
     "Jockey booked at 06:00 against the horse's usual jockey quality", "Available",
     "The ridden version is card-unsafe; the booked version is safe", "Medium"),
    ("N6", "Betfair price-history windows", "Market", "Past runs' morning-to-BSP drift and volume, windowed",
     "Partial (Betfair files from Jan 2026)", "Needs more history before it can train", "Low"),
    ("N7", "Race shape x horse style fit, learned per horse", "Pace",
     "How this horse has run in each pace scenario (from shape form's per-run readings)", "Available",
     "Builds on shape form", "Medium"),
    ("N8", "Model hyperparameters", "Recipe", "Leaves, learning rate, min child samples at the 6000-round cap",
     "n/a", "Iteration 31: the recipe was under-fitted at 3000 rounds", "High"),
    ("X1", "Fix: track_draw_bias", "Defect", "Blank on every row (draw_metrics.py:643)", "n/a",
     "Zero gain; fix or drop. Its code shifts by row within the track, not by day: filled, it would read earlier "
     "races on the same day", "Low"),
    ("X2", "Fix: FCS without official ratings", "Defect", "FCS is 0 when today's race has no ratings", "n/a",
     "custom_metrics.py:656", "Low"),
    ("X3", "Fix: duplicate columns", "Defect", "RB = NFP (9), WOA = WAX (7), OFS = ORR2 (7), racepacescore = RPS (1)",
     "n/a", "Harmless to accuracy; splits importance readings. WOA needs the owner's definition of the average it "
     "is measured against (reports/metric_audit.md)", "Low"),
    ("X4", "Fix: rank direction for missing values", "Defect",
     "rLB, rConsistency, rDistApt, rGoingPref rank highest first with missing values last, which for a "
     "lower-is-better measure puts a runner with no figure beside the best", "n/a",
     "reports/metric_audit.md; the wide within-race readings read the four sources without the defect", "Medium"),
    ("X5", "Fix: void races read as non-finishes", "Defect", "154 VOI rows count as a run that did not finish", "n/a",
     "reports/metric_audit.md", "Low"),
    ("M1", "The rest of the racing2 master framework", "Research",
     "RESEARCH_FRAMEWORK.md s14 counted 277 items: 75 implemented, 80 partial, 122 missing", "Mixed",
     "The framework document is not in the repository, so the items cannot be listed one by one", "Review"),
]


# ---------------------------------------------------------------------------
# Workbook
# ---------------------------------------------------------------------------

def feature_rows(importance: pd.DataFrame) -> list[dict]:
    gain = dict(zip(importance["feature"], importance["importance"]))
    src = (ROOT / "model" / "bfsp_features.py").read_text()
    order = re.findall(r"\+?\s*([A-Z_]+FEATURES)", re.search(r"ALL_FEATURE_COLS = \((.*?)\n\)", src, re.S).group(1))
    rows, seen = [], set()
    for g in order:
        block, group, module = GROUPS.get(g, ("Other", g, "model/bfsp_features.py"))
        for f in getattr(B, g):
            if f in seen:
                continue
            seen.add(f)
            rows.append(dict(feature=f, block=block, group=group, module=module, status=SERVED, gain=gain.get(f),
                             card_safe=card_safe(f), constant=g))
    extra = [(FORM_WINDOW_FEATURES, "Form windows", "Every measure over seven windows", "model/form_windows.py",
              CANDIDATE)]
    for mod, label, group, status in DROP_IN:
        cols = importlib.import_module(f"model.blocks.{mod}").FEATURES
        extra.append((cols, label, group, f"model/blocks/{mod}.py", CANDIDATE if status == "candidate" else BUILT))
    extra += [(SHAPE_FORM_FEATURES, "Shape form (shape/draw remodel)", "Past runs read against pace and draw; "
              "speed drawn near; market misses", "model/shape_form.py", CANDIDATE),
             (B.SHAPE_DRAW_FEATURES, "Shape and draw (old)", "Run style, race shape, position value, draw curves",
              "model/race_shape.py, model/draw_curve.py", BUILT),
             (B.INTENT_CARD_UNSAFE, "Intent (card-unsafe)", "What the 06:00 card cannot know",
              "model/intent_features.py", EXCLUDED)]
    for cols, block, group, module, status in extra:
        for f in cols:
            if f in seen:
                continue
            seen.add(f)
            rows.append(dict(feature=f, block=block, group=group, module=module, status=status,
                             gain=gain.get(f), card_safe=card_safe(f), constant=""))
    for r in rows:
        r["description"] = describe(r["feature"], r["constant"])
    return rows


def style_header(ws, row: int, ncol: int):
    for c in range(1, ncol + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = Font(name=FONT, size=10, bold=True, color="FFFFFF")
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(vertical="center", wrap_text=True)


def build(importance_path: Path, out: Path, model_note: str):
    importance = pd.read_csv(importance_path)
    rows = feature_rows(importance)
    served = [r for r in rows if r["status"] == SERVED]
    served.sort(key=lambda r: -(r["gain"] or 0))
    others = [r for r in rows if r["status"] != SERVED]
    rows = served + others
    n = len(rows)
    last = n + 1

    wb = Workbook()
    base = Font(name=FONT, size=10)

    # --- Features ------------------------------------------------------------------------
    ws = wb.active
    ws.title = "Features"
    head = ["Feature", "Block", "Group", "Description", "Module", "Status", "Gain (split gain)",
            "Share of served gain", "Rank (served)", "Card-safe at 06:00"]
    ws.append(head)
    style_header(ws, 1, len(head))
    for i, r in enumerate(rows, start=2):
        ws.append([r["feature"], r["block"], r["group"], r["description"], r["module"], r["status"],
                   r["gain"] if r["status"] == SERVED else None,
                   f'=IF(G{i}="","",G{i}/SUMIFS($G$2:$G${last},$F$2:$F${last},"{SERVED}"))',
                   f'=IF(G{i}="","",RANK(G{i},$G$2:$G${last},0))',
                   r["card_safe"]])
    for row in ws.iter_rows(min_row=2, max_row=last):
        for cell in row:
            cell.font = base
        row[6].number_format = "#,##0"
        row[7].number_format = "0.00%"
        row[3].alignment = Alignment(wrap_text=False)
    ws["G1"].comment = Comment(f"Split gain of the served model. Source: {model_note}. Candidates and unserved "
                               "features have no gain in this model.", "feature_workbook.py")
    for col, w in zip("ABCDEFGHIJ", (28, 26, 34, 80, 30, 36, 16, 12, 10, 12)):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = f"A1:J{last}"

    # --- Blocks --------------------------------------------------------------------------
    wb_blocks = wb.create_sheet("Blocks")
    head = ["Block", "Status", "Features", "Served", "Share of served gain", "Module", "Evidence"]
    wb_blocks.append(head)
    style_header(wb_blocks, 1, len(head))
    blocks = []
    for r in rows:
        if r["block"] not in [b[0] for b in blocks]:
            blocks.append((r["block"], r["status"], r["module"]))
    fr = f"Features!$B$2:$B${last}"
    for i, (b, status, module) in enumerate(blocks, start=2):
        wb_blocks.append([b, status, f'=COUNTIFS({fr},A{i})',
                          f'=COUNTIFS({fr},A{i},Features!$F$2:$F${last},"{SERVED}")',
                          f'=SUMIFS(Features!$H$2:$H${last},{fr},A{i})', module, BLOCK_EVIDENCE.get(b, "")])
    nb = len(blocks) + 1
    wb_blocks.append(["Total", "", f"=SUM(C2:C{nb})", f"=SUM(D2:D{nb})", f"=SUM(E2:E{nb})", "", ""])
    tot = nb + 1
    research = [
        ("Performance figures (lbs)", "model/perf_figures.py", 7, "iter8: nothing beyond BSP against the result"),
        ("Kalman rating", "model/state_space.py", 6, "iter8: nothing beyond BSP"),
        ("Handicap angles", "model/handicap_features.py", 10, "iter8: nothing beyond BSP"),
        ("A-E vs the market by entity", "model/ae_features.py", 16, "iter11: -0.43 mnats, no persistent mispricing"),
        ("The race's other markets", "model/market_block.py", 7, "iter5: -0.26 mnats"),
        ("In-day (earlier races today)", "model/inday_features.py", 21, "iter12-16; rejected on the locked holdout"),
        ("Betfair price-file features", "model/market_features.py", 30, "not scored as a block; data from Jan 2026"),
        ("Timeform feed", "model/blandford_features.py", 18, "same-day fields banned (leak); lagged not a candidate"),
        ("Odds metrics", "model/odds_metrics.py", 16, "not scored: uses the race's own BSP (leaks)"),
        ("ABM race simulator", "model/abm/", 16, "not run at scale"),
    ]
    wb_blocks.append([])
    wb_blocks.append(["Research blocks (opt-in, never served)", "", "Features", "", "", "Module", "Evidence"])
    style_header(wb_blocks, tot + 2, 7)
    for name, module, k, ev in research:
        wb_blocks.append([name, "Research only (opt-in)", k, 0, None, module, ev])
        wb_blocks.cell(row=wb_blocks.max_row, column=3).font = INPUT_FONT
    first_r = tot + 3
    wb_blocks.cell(row=first_r, column=3).comment = Comment(
        "Feature counts of the opt-in research blocks, typed in from reports/feature_inventory.md s4 "
        "(they are not in the feature lists this workbook reads).", "feature_workbook.py")
    for row in wb_blocks.iter_rows(min_row=2, max_row=wb_blocks.max_row):
        for cell in row:
            if cell.font != INPUT_FONT:
                cell.font = base
        row[4].number_format = "0.00%"
    for c in range(1, 8):
        wb_blocks.cell(row=tot, column=c).font = Font(name=FONT, size=10, bold=True)
    for col, w in zip("ABCDEFG", (36, 38, 10, 9, 14, 40, 110)):
        wb_blocks.column_dimensions[col].width = w
    wb_blocks.freeze_panes = "A2"

    # --- Not yet developed ---------------------------------------------------------------
    nd = wb.create_sheet("Not yet developed")
    head = ["ID", "Feature or idea", "Category", "What it would measure", "Data", "Status / notes", "Priority"]
    nd.append(head)
    style_header(nd, 1, len(head))
    for item in NOT_BUILT:
        nd.append(list(item))
    for row in nd.iter_rows(min_row=2, max_row=nd.max_row):
        for cell in row:
            cell.font = base
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    for col, w in zip("ABCDEFG", (7, 38, 16, 60, 22, 70, 10)):
        nd.column_dimensions[col].width = w
    nd.freeze_panes = "B2"
    nd.auto_filter.ref = f"A1:G{nd.max_row}"
    nlast = nd.max_row

    # --- Summary -------------------------------------------------------------------------
    sm = wb.create_sheet("Summary", 0)
    sm["A1"] = "Ashcroft price model: feature inventory"
    sm["A1"].font = Font(name=FONT, size=14, bold=True)
    sm["A2"] = f"Gain from {model_note}. Regenerate with: python scripts/feature_workbook.py"
    sm["A2"].font = Font(name=FONT, size=9, italic=True)
    sm.append([])
    sm.append(["Features by status", "Count"])
    style_header(sm, 4, 2)
    statuses = [SERVED, CANDIDATE, BUILT, EXCLUDED]
    for s in statuses:
        sm.append([s, f'=COUNTIF(Features!$F$2:$F${last},A{sm.max_row + 1})'])
    sm.append(["All features listed", f"=SUM(B5:B{4 + len(statuses)})"])
    r_all = sm.max_row
    sm.append(["Served features with zero gain", f'=COUNTIFS(Features!$F$2:$F${last},"{SERVED}",Features!$G$2:$G${last},0)'])
    sm.append(["Share of gain check (should be 100%)", f"=SUM(Features!$H$2:$H${last})"])
    sm.cell(row=sm.max_row, column=2).number_format = "0.00%"
    sm.append([])
    sm.append(["Not yet developed, by priority", "Count"])
    style_header(sm, sm.max_row, 2)
    for p in ("High", "Medium", "Low", "Blocked", "Review"):
        sm.append([p, f"=COUNTIF('Not yet developed'!$G$2:$G${nlast},A{sm.max_row + 1})"])
    sm.append([])
    top_hdr = sm.max_row + 1
    sm.append(["Top 15 served features by gain", "Share of gain", "Block", "Description"])
    style_header(sm, top_hdr, 4)
    for k in range(1, 16):
        r = sm.max_row + 1
        m = f"MATCH({k},Features!$I$2:$I${last},0)"
        sm.append([f"=INDEX(Features!$A$2:$A${last},{m})", f"=INDEX(Features!$H$2:$H${last},{m})",
                   f"=INDEX(Features!$B$2:$B${last},{m})", f"=INDEX(Features!$D$2:$D${last},{m})"])
        sm.cell(row=r, column=2).number_format = "0.00%"
    for row in sm.iter_rows(min_row=5, max_row=sm.max_row):
        for cell in row:
            if not (cell.fill and cell.fill.fgColor and cell.fill.fgColor.rgb == HEADER_FILL.fgColor.rgb):
                cell.font = base
    sm.cell(row=r_all, column=1).font = Font(name=FONT, size=10, bold=True)
    sm.cell(row=r_all, column=2).font = Font(name=FONT, size=10, bold=True)
    for col, w in zip("ABCD", (44, 14, 30, 80)):
        sm.column_dimensions[col].width = w

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--importance", default=str(ROOT / "data/models/bfsp_feature_importance.csv"))
    ap.add_argument("--model-note", default="the served model, data/models/bfsp_model_meta.json")
    ap.add_argument("--out", default=str(ROOT / "reports/feature_inventory.xlsx"))
    args = ap.parse_args(argv)
    note = args.model_note
    meta = Path(args.importance).with_name("bfsp_model_meta.json")
    if meta.exists() and note.startswith("the served model"):
        import json
        m = json.load(open(meta))
        note = (f"the served model ({m.get('n_features')} features, trained through {m.get('trained_through')}, "
                f"feature hash {m.get('feature_code_hash')})")
    n = build(Path(args.importance), Path(args.out), note)
    print(f"wrote {args.out}: {n} features")


if __name__ == "__main__":
    main()
