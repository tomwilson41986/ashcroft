"""Finishing, goalkeeping and luck from Sportmonks' xG, shrunk to the average. ``sl_``.

racing² football research, 24 Nov 2026 (the owner: "statistical transformations ... more advanced xG"). Results that
beat the chances tend not to last, and a market that reads results can overrate a side that has been finishing above
its xG. Per past match, from the side's point of view:

- finishing: goals less xG (penalties in both);
- shot quality after the strike: xG on target less xG (placement);
- goalkeeping: the opponents' xG on target less goals conceded (positive: goals kept out);
- luck: points less Sportmonks' expected points;
- big chances missed.

Each is a 120-day half-life mean over earlier matches, lagged to the collection day, and shrunk towards zero by its
evidence (mean x W / (W + 8), W the effective number of matches): one or two matches say little. Home less away:
``sl_fin_diff``, ``sl_shotq_diff``, ``sl_keep_diff``, ``sl_luck_diff``, ``sl_bcm_diff``. Blank before 2024/25.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.blocks._util import decayed, per_match, sides
from football.sportmonks_stage import COLUMNS

FEATURES = ["sl_fin_diff", "sl_shotq_diff", "sl_keep_diff", "sl_luck_diff", "sl_bcm_diff"]
MEASURES = ["fin", "shotq", "keep", "luck", "bcm"]
HALF, K = 120.0, 8.0


def build(m: pd.DataFrame) -> pd.DataFrame:
    m = m.assign(**{c: np.nan for c in COLUMNS if c not in m.columns})
    s = sides(m, {"xg_f": ("smx_xg_xg_h", "smx_xg_xg_a"), "xgot_f": ("smx_xg_xgot_h", "smx_xg_xgot_a"),
                  "xgot_o": ("smx_xg_xgot_a", "smx_xg_xgot_h"), "xpts_f": ("smx_xg_xpts_h", "smx_xg_xpts_a"),
                  "bcm_f": ("smx_st_big_chances_missed_h", "smx_st_big_chances_missed_a")})
    gf, ga = pd.to_numeric(s["gf"], errors="coerce"), pd.to_numeric(s["ga"], errors="coerce")
    pts = np.where(gf > ga, 3.0, np.where(gf == ga, 1.0, 0.0))
    pts = np.where(gf.notna() & ga.notna(), pts, np.nan)
    s["fin"] = gf - s["xg_f"]
    s["shotq"] = s["xgot_f"] - s["xg_f"]
    s["keep"] = s["xgot_o"] - ga
    s["luck"] = pts - s["xpts_f"]
    s["bcm"] = s["bcm_f"]
    v, w = decayed(s, MEASURES, HALF)
    shrink = w / (w + K)
    p = per_match(s, {c: v[c].to_numpy() * shrink for c in MEASURES})
    out = p[["match_id"]].copy()
    for f, c in zip(FEATURES, MEASURES):
        out[f] = p[f"{c}_h"] - p[f"{c}_a"]
    return out
