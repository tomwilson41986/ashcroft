"""Sportmonks' xG and match statistics as form. ``sx_``.

racing² football research, 24 Nov 2026 (the owner subscribed to Sportmonks' xG bundle). Each side's 120-day half-life
means over its earlier matches, lagged to the collection day (``_util.decayed``), home less away:

- ``sx_npxgd_diff``: non-penalty xG for less against;
- ``sx_xgd_mkt_diff``: xG difference less the closing market's expected goal difference (what the price missed);
- ``sx_xgot_diff``: xG on target for less against (shot quality after the strike);
- ``sx_xpts_mkt_diff``: Sportmonks' expected points less the market's (3 x win + draw at the close);
- ``sx_openplay_diff``: open-play xG for less against; ``sx_setplay_diff``: set-play xG for less against;
- ``sx_bigch_diff``: big chances created less conceded; ``sx_danger_diff``: dangerous attacks for less against;
- ``sx_box_diff``: share of shots from inside the box, for less against; ``sx_keyp_diff``: key passes for less against;
- ``sx_pass_diff``: pass completion, home less away;
- ``sx_w``: the two sides' effective number of matches behind them, summed.

Sportmonks' xG starts in 2024/25: blank before.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.blocks._util import decayed, per_match, sides
from football.sportmonks_stage import COLUMNS

FEATURES = ["sx_npxgd_diff", "sx_xgd_mkt_diff", "sx_xgot_diff", "sx_xpts_mkt_diff", "sx_openplay_diff",
            "sx_setplay_diff", "sx_bigch_diff", "sx_danger_diff", "sx_box_diff", "sx_keyp_diff", "sx_pass_diff",
            "sx_w"]
MEASURES = ["npxgd", "xgd_mkt", "xgotd", "xpts_mkt", "opd", "spd", "bigchd", "dangerd", "boxd", "keypd", "passp"]
HALF = 120.0


def build(m: pd.DataFrame) -> pd.DataFrame:
    m = m.assign(**{c: np.nan for c in COLUMNS if c not in m.columns})
    pair = {}
    for c in ("xg_npxg", "xg_xg", "xg_xgot", "xg_xpts", "xg_xgop", "xg_xgsp", "st_big_chances_created",
              "st_dangerous_attacks", "st_shots_insidebox", "st_shots_total", "st_key_passes", "st_successful_passes",
              "st_passes"):
        pair[f"{c}_f"] = (f"smx_{c}_h", f"smx_{c}_a")
        pair[f"{c}_o"] = (f"smx_{c}_a", f"smx_{c}_h")
    s = sides(m, {**pair, "mxg": ("mclam_h", "mclam_a"), "mxga": ("mclam_a", "mclam_h"), "qw": ("qc_h", "qc_a"),
                  "qd": ("qc_d", "qc_d")})
    s["npxgd"] = s["xg_npxg_f"] - s["xg_npxg_o"]
    s["xgd_mkt"] = (s["xg_xg_f"] - s["xg_xg_o"]) - (s["mxg"] - s["mxga"])
    s["xgotd"] = s["xg_xgot_f"] - s["xg_xgot_o"]
    s["xpts_mkt"] = s["xg_xpts_f"] - (3 * s["qw"] + s["qd"])
    s["opd"] = s["xg_xgop_f"] - s["xg_xgop_o"]
    s["spd"] = s["xg_xgsp_f"] - s["xg_xgsp_o"]
    s["bigchd"] = s["st_big_chances_created_f"] - s["st_big_chances_created_o"]
    s["dangerd"] = s["st_dangerous_attacks_f"] - s["st_dangerous_attacks_o"]
    box = lambda a, b: s[a] / s[b].where(s[b] > 0)                                       # noqa: E731
    s["boxd"] = box("st_shots_insidebox_f", "st_shots_total_f") - box("st_shots_insidebox_o", "st_shots_total_o")
    s["keypd"] = s["st_key_passes_f"] - s["st_key_passes_o"]
    s["passp"] = s["st_successful_passes_f"] / s["st_passes_f"].where(s["st_passes_f"] > 0)
    v, w = decayed(s, MEASURES + ["xg_xg_f"], HALF)
    have = np.where(np.isfinite(v["xg_xg_f"].to_numpy()), w, np.nan)
    p = per_match(s, {**{c: v[c].to_numpy() for c in MEASURES}, "w": have})
    out = p[["match_id"]].copy()
    for f, c in zip(FEATURES[:-1], MEASURES):
        out[f] = p[f"{c}_h"] - p[f"{c}_a"]
    out["sx_w"] = p["w_h"] + p["w_a"]
    return out
