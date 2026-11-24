"""The market's own expected goals and points for each side, as form. ``mx_``.

racing² football research, 24 Nov 2026 (the owner: market-implied xPts, xG, xGA, xGDiff). Every closing price holds
a view of both sides: the goal rates it implies (``mclam``) and the points it expects (3 x win + draw). A side's
recent closes, averaged with a time decay, are the market's running opinion of it, schedule and venue included:

- ``mx_<m>_lvl_diff``: the 60-day half-life mean of the measure, home less away;
- ``mx_<m>_trend_diff``: the 30-day mean less the 180-day one (is the market warming to the side?), home less away.

Measures: ``xg`` (implied goals for), ``xga`` (against), ``xgd`` (difference), ``xpts`` (expected points). Only closes
of matches before the collection day enter.
"""

from __future__ import annotations

import pandas as pd

from football.blocks._util import decayed, per_match, sides

MEASURES = ["xg", "xga", "xgd", "xpts"]
FEATURES = [f"mx_{c}_{k}_diff" for c in MEASURES for k in ("lvl", "trend")]


def side_table(m: pd.DataFrame) -> pd.DataFrame:
    s = sides(m, {"xg": ("mclam_h", "mclam_a"), "xga": ("mclam_a", "mclam_h"), "qw": ("qc_h", "qc_a"),
                  "qd": ("qc_d", "qc_d")})
    s["xgd"] = s["xg"] - s["xga"]
    s["xpts"] = 3 * s["qw"] + s["qd"]
    return s


def build(m: pd.DataFrame) -> pd.DataFrame:
    s = side_table(m)
    lvl, _ = decayed(s, MEASURES, 60.0)
    fast, _ = decayed(s, MEASURES, 30.0)
    slow, _ = decayed(s, MEASURES, 180.0)
    cols = {}
    for c in MEASURES:
        cols[f"{c}_lvl"] = lvl[c].to_numpy()
        cols[f"{c}_trend"] = (fast[c] - slow[c]).to_numpy()
    p = per_match(s, cols)
    out = p[["match_id"]].copy()
    for k in cols:
        out[f"mx_{k}_diff"] = p[f"{k}_h"] - p[f"{k}_a"]
    return out
