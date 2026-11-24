"""More of Sportmonks' match statistics as style and strength. ``sy_``.

racing² football research, 24 Nov 2026 (the owner: derive new variables from the Sportmonks data). The statistics no
other block reads, each a 120-day half-life mean over earlier matches lagged to the collection day, home less away:

- ``sy_sot_diff``: shots on target for less against;
- ``sy_saves_diff``: saves made (a busy keeper is a defence under pressure);
- ``sy_cross_diff``: crosses for less against;
- ``sy_defact_diff``: tackles and interceptions;
- ``sy_duels_diff``: duels won for less against;
- ``sy_long_diff``: the share of passes played long;
- ``sy_corners_diff``: corners for less against; ``sy_cornerxg_diff``: corner xG for less against.

Blank before 2024/25.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.blocks._util import decayed, per_match, sides
from football.sportmonks_stage import COLUMNS

FEATURES = ["sy_sot_diff", "sy_saves_diff", "sy_cross_diff", "sy_defact_diff", "sy_duels_diff", "sy_long_diff",
            "sy_corners_diff", "sy_cornerxg_diff"]
MEASURES = ["sot", "saves", "cross", "defact", "duels", "long", "corners", "cornerxg"]
HALF = 120.0


def build(m: pd.DataFrame) -> pd.DataFrame:
    m = m.assign(**{c: np.nan for c in COLUMNS if c not in m.columns})
    pair = {}
    for c in ("st_shots_on_target", "st_saves", "st_total_crosses", "st_tackles", "st_interceptions", "st_duels_won",
              "st_long_passes", "st_passes", "st_corners", "xg_xgc"):
        pair[f"{c}_f"] = (f"smx_{c}_h", f"smx_{c}_a")
        pair[f"{c}_o"] = (f"smx_{c}_a", f"smx_{c}_h")
    s = sides(m, pair)
    d = lambda c: s[f"{c}_f"] - s[f"{c}_o"]                                               # noqa: E731
    s["sot"] = d("st_shots_on_target")
    s["saves"] = s["st_saves_f"]
    s["cross"] = d("st_total_crosses")
    s["defact"] = (s["st_tackles_f"] + s["st_interceptions_f"]) - (s["st_tackles_o"] + s["st_interceptions_o"])
    s["duels"] = d("st_duels_won")
    s["long"] = s["st_long_passes_f"] / s["st_passes_f"].where(s["st_passes_f"] > 0)
    s["corners"] = d("st_corners")
    s["cornerxg"] = d("xg_xgc")
    v, _ = decayed(s, MEASURES, HALF)
    p = per_match(s, {c: v[c].to_numpy() for c in MEASURES})
    out = p[["match_id"]].copy()
    for f, c in zip(FEATURES, MEASURES):
        out[f] = p[f"{c}_h"] - p[f"{c}_a"]
    return out
