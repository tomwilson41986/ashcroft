"""The dam's produce record, the siblings and the female family: breeding where a horse has little form of its own.

The price forecaster's largest errors are in the races of lightly raced horses:
maiden hurdles, bumpers, maidens and novices (about 0.50-0.57 mean absolute
log error against 0.34-0.37 in handicaps, a quarter of the total). There the
market prices breeding, and the engine reads only the sire's and damsire's
levels. The dam is the other half: what her other foals have done, how early,
at what trip and on what ground, and what she did herself. This block serves
model/pedigree.py's dam and family suites (racing-squared framework, Part 5.4),
which were only ever tested beside the BSP, never in the price forecast:

    dam_runners, dam_winners, dam_sr_shrunk   the dam's other foals: runs, wins,
                                              strike rate shrunk to the population
    dam_progeny_mean_nmfp, dam_best_progeny_rating, dam_black_type_progeny
    sib_count, sib_mean_nmfp                  the siblings (the horse itself excluded)
    sib_dist_profile, sib_going_profile       the trip and ground they did best at
    sib_precocity, sib_improvement_curve      how early they ran well, and how they improved
    family_bt_density, family_class_rating    the female family's black type and class

Every figure reads progeny runs on strictly earlier days (model/pedigree.py,
"Lag safety"), and a horse is not its own sibling. The module subtracts the
horse's own share from running totals, which leaves the last bit of a sibling
mean depending on the order sums were taken in, today's rows among them; the
block rounds to 12 decimal places so that no feature moves, even by rounding,
when a day's own results do (tests/test_feature_blocks.py), while any real
leak would move it by far more.
"""

from __future__ import annotations

import pandas as pd

from model.pedigree import add_pedigree_features

FEATURES = ["dam_runners", "dam_winners", "dam_sr_shrunk", "sib_count", "sib_mean_nmfp", "dam_progeny_mean_nmfp",
            "dam_black_type_progeny", "dam_best_progeny_rating", "sib_dist_profile", "sib_going_profile",
            "sib_precocity", "sib_improvement_curve", "family_bt_density", "family_class_rating"]
#: the dam's own record, which the module also builds, is left out: a dam of today's runners
#: raced a decade ago, before the history starts, so it is almost always empty
NOT_SERVED = ["dam_own_runs", "dam_own_nmfp", "dam_own_best_or", "dam_own_black_type"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "stallion", "dam", "dam_stallion",
         "placing_numerical", "number_of_runners", "official_rating", "horse_age", "dist_furlongs",
         "going_description", "prize_money", "major", "race_name", "race_class", "race_type", "career_runs",
         "surface_type", "won"]


def build(df: pd.DataFrame) -> pd.DataFrame:
    out, feats = add_pedigree_features(df.drop(columns=[c for c in FEATURES + NOT_SERVED if c in df.columns]),
                                       sire=False, damsire=False, dam=True, nick=False, family=True)
    if sorted(feats) != sorted(FEATURES + NOT_SERVED):
        raise ValueError(f"dam_line: the pedigree module now makes {feats}, not {FEATURES + NOT_SERVED}")
    keep = [c for c in df.columns if c not in FEATURES]
    return pd.concat([df[keep], out[FEATURES].astype(float).round(12)], axis=1)
