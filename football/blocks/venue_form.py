"""Venue-specific form: the home side's form at home and the away side's form away (points, goal difference, and
goal difference against the close's expectation), over earlier matches at that venue, lagged to the collection
day. The team features pool both venues. ``vf_``."""

from __future__ import annotations

import pandas as pd

from football.blocks._util import home_minus_away, lagged, sides

MEASURES = ["pts", "gd", "gd_vs_mkt"]
WINS = ("l5", "l10", "ewm")
FEATURES = [f"vf_{c}_{w}" for c in MEASURES for w in WINS]


def build(m: pd.DataFrame) -> pd.DataFrame:
    s = sides(m, {"xgf": ("mclam_h", "mclam_a"), "xga": ("mclam_a", "mclam_h")})
    s["gd"] = s["gf"] - s["ga"]
    s["pts"] = (s["gd"] > 0) * 3.0 + (s["gd"] == 0) * 1.0
    s["gd_vs_mkt"] = s["gd"] - (s["xgf"] - s["xga"])
    s["team_venue"] = s["team"].astype(str) + "|" + s["venue"]
    w = lagged(s, MEASURES, key="team_venue")
    cols = [f"{c}_{x}" for c in MEASURES for x in WINS]
    return home_minus_away(s, w, cols, "vf_")
