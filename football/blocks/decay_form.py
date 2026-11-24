"""Form weighted by how long ago it was, not by how many matches ago. ``dk_``.

racing² football research, 24 Nov 2026 (the owner: recency, weighting, decaying; rolling form). The window ladder
counts matches, so a match from before the summer break weighs the same as last week's. Here each earlier match
weighs exp(-ln2 x days ago / half-life), at three half-lives (30, 120, 365 days):

- ``gae``: goal difference less the closing market's expected one;
- ``pae``: points less the closing market's expected points;
- ``sotd``: shots on target for less against.

``dk_<m>_h<half>_diff``: home less away. ``dk_w120``: the two sides' effective number of matches at the 120-day
half-life, summed: how much recent evidence there is (small after a break).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.blocks._util import decayed, per_match, sides

MEASURES = ["gae", "pae", "sotd"]
HALVES = (30, 120, 365)
FEATURES = [f"dk_{c}_h{h}_diff" for c in MEASURES for h in HALVES] + ["dk_w120"]


def side_table(m: pd.DataFrame) -> pd.DataFrame:
    s = sides(m, {"xg": ("mclam_h", "mclam_a"), "xga": ("mclam_a", "mclam_h"), "qw": ("qc_h", "qc_a"),
                  "qd": ("qc_d", "qc_d"), "sot_f": ("sot_home", "sot_away"), "sot_a": ("sot_away", "sot_home")})
    s["gae"] = (s["gf"] - s["ga"]) - (s["xg"] - s["xga"])
    pts = np.where(s["gf"] > s["ga"], 3.0, np.where(s["gf"] == s["ga"], 1.0, 0.0))
    s["pae"] = np.where(s["gf"].notna() & s["ga"].notna(), pts, np.nan) - (3 * s["qw"] + s["qd"])
    s["sotd"] = s["sot_f"] - s["sot_a"]
    return s


def build(m: pd.DataFrame) -> pd.DataFrame:
    s = side_table(m)
    cols = {}
    for h in HALVES:
        v, w = decayed(s, MEASURES, float(h))
        for c in MEASURES:
            cols[f"{c}_h{h}"] = v[c].to_numpy()
        if h == 120:
            cols["w"] = w
    p = per_match(s, cols)
    out = p[["match_id"]].copy()
    for c in MEASURES:
        for h in HALVES:
            out[f"dk_{c}_h{h}_diff"] = p[f"{c}_h{h}_h"] - p[f"{c}_h{h}_a"]
    out["dk_w120"] = p["w_h"] + p["w_a"]
    return out
