"""Form normalised within the league's round. ``nz_``.

racing² football research, 24 Nov 2026 (the owner: normalised features). A goal difference of +0.3 a match means one
thing in a league of even sides and another in a lopsided one. Each side's lagged form is placed among the sides of
its competition playing in the same collection window (all known at the early price):

- ``nz_<m>_pct_diff``: the side's percentile there, home less away;
- ``nz_<m>_z_diff``: its z-score there, home less away.

Measures (120-day half-life means): ``gae`` (goal difference less the market's expected one), ``xgd`` (the market's
implied goal difference), ``sotd`` (shots on target for less against). Blank where under 6 sides are placed.
"""

from __future__ import annotations

import pandas as pd

from football.blocks._util import collection_day, decayed, per_match, sides

MEASURES = ["gae", "xgd", "sotd"]
FEATURES = [f"nz_{c}_{k}_diff" for c in MEASURES for k in ("pct", "z")]
MIN_SIDES = 6


def build(m: pd.DataFrame) -> pd.DataFrame:
    s = sides(m, {"xg": ("mclam_h", "mclam_a"), "xga": ("mclam_a", "mclam_h"), "sot_f": ("sot_home", "sot_away"),
                  "sot_a": ("sot_away", "sot_home")})
    s["xgd"] = s["xg"] - s["xga"]
    s["gae"] = (s["gf"] - s["ga"]) - s["xgd"]
    s["sotd"] = s["sot_f"] - s["sot_a"]
    v, _ = decayed(s, MEASURES, 120.0)
    key = [s["competition_id"].to_numpy(), collection_day(s["date"]).to_numpy()]
    cols = {}
    for c in MEASURES:
        g = v[c].groupby(key)
        n = g.transform("count")
        pct = g.rank(pct=True).where(n >= MIN_SIDES)
        sd = g.transform("std")
        z = ((v[c] - g.transform("mean")) / sd.where(sd > 0)).where(n >= MIN_SIDES)
        cols[f"{c}_pct"], cols[f"{c}_z"] = pct.to_numpy(), z.to_numpy()
    p = per_match(s, cols)
    out = p[["match_id"]].copy()
    for k in cols:
        out[f"nz_{k}_diff"] = p[f"{k}_h"] - p[f"{k}_a"]
    return out
