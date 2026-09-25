"""The sire's and damsire's aptitudes against their own level, and the nick: breeding read as a residual.

The engine serves sire and damsire *levels* (shrunk win rate and NFP, overall
and by going or trip), which is the part of pedigree the market prices best:
"is this a good sire". The racing-squared framework (Part 5.3-5.4) puts the
edge in the residual: how a sire's runners do on today's going, at today's
trip, on this surface, at this course, as two-year-olds and first time out,
relative to that same sire's overall level (the cell mean shrunk toward the
sire's own mean, minus it), with the evidence behind each. model/pedigree.py
builds them; they were tested only beside the BSP, never in the price
forecast:

    sire_prog_runs, sire_prog_wins, sire_sr_shrunk, sire_nmfp_shrunk,
    sire_prb2_fsa_shrunk, sire_going_apt/_n, sire_dist_apt/_n,
    sire_surface_apt/_n, sire_course_apt/_n, sire_2yo_apt/_n,
    sire_first_time_out_apt/_n, sire_improvement, sire_elite_share,
    sire_class_geo, sire_class_rating, sire_track_strength,
    sire_median_hcap_debut_or
    damsire_prog_runs ... damsire_surface_apt/_n   (the same, without the
                                                   two-year-old and debut cells)
    nick_runs, nick_apt, nick_sr_apt               the sire x damsire cross

Every figure reads progeny runs on strictly earlier days.
"""

from __future__ import annotations

import pandas as pd

from model.pedigree import add_pedigree_features

FEATURES = ["damsire_prog_runs", "damsire_prog_wins", "damsire_sr_shrunk", "damsire_nmfp_shrunk",
            "damsire_prb2_fsa_shrunk", "damsire_going_apt", "damsire_going_n", "damsire_dist_apt",
            "damsire_dist_n", "damsire_surface_apt", "damsire_surface_n",
            "sire_prog_runs", "sire_prog_wins", "sire_sr_shrunk", "sire_nmfp_shrunk", "sire_prb2_fsa_shrunk",
            "sire_going_apt", "sire_going_n", "sire_dist_apt", "sire_dist_n", "sire_surface_apt",
            "sire_surface_n", "sire_course_apt", "sire_course_n", "sire_2yo_apt", "sire_2yo_n",
            "sire_first_time_out_apt", "sire_first_time_out_n", "sire_improvement", "sire_elite_share",
            "sire_class_geo", "sire_class_rating", "sire_track_strength", "sire_median_hcap_debut_or",
            "nick_runs", "nick_apt", "nick_sr_apt"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "stallion", "dam", "dam_stallion",
         "placing_numerical", "number_of_runners", "official_rating", "horse_age", "dist_furlongs",
         "going_description", "prize_money", "major", "race_name", "race_class", "race_type", "career_runs",
         "surface_type", "won"]


def build(df: pd.DataFrame) -> pd.DataFrame:
    out, feats = add_pedigree_features(df.drop(columns=[c for c in FEATURES if c in df.columns]),
                                       sire=True, damsire=True, dam=False, nick=True, family=False)
    if list(feats) != FEATURES:
        raise ValueError(f"sire_apt: the pedigree module now makes {feats}, not {FEATURES}")
    keep = [c for c in df.columns if c not in FEATURES]
    return pd.concat([df[keep], out[FEATURES]], axis=1)
