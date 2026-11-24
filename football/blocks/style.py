"""How each side plays, and where two styles meet. ``st_``.

racing² football research, 24 Nov 2026 (the owner: team style metrics; tactical fit, tactical mismatches). From the
match counts football-data holds in every league (shots, shots on target, corners, cards), each side's 120-day
half-life means for and against, before the collection day. A matchup joins one side's output to the other's
concession:

- ``st_shots_exp_diff``, ``st_sot_exp_diff``, ``st_cor_exp_diff``: (home for + away against) / 2 less
  (away for + home against) / 2: the shot, shot-on-target and corner balance the two styles imply;
- ``st_acc_diff``: shots on target a shot, home less away (shot quality); ``st_acc_allowed_diff``: the same conceded;
- ``st_tempo``: the two sides' shots for and against a match, summed (an open or a closed game);
- ``st_tempo_gap``: the absolute difference of the two sides' tempos (a clash of styles);
- ``st_cards_diff``: yellow cards plus two a red, home less away;
- ``st_vol_leak``: the home side's shot volume against the quality the away side allows, less the reverse (a
  high-volume side meeting one that concedes good chances).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.blocks._util import decayed, per_match, sides

FEATURES = ["st_shots_exp_diff", "st_sot_exp_diff", "st_cor_exp_diff", "st_acc_diff", "st_acc_allowed_diff",
            "st_tempo", "st_tempo_gap", "st_cards_diff", "st_vol_leak"]
RAW = ["sh_f", "sh_a", "sot_f", "sot_a", "cor_f", "cor_a", "cards"]


def build(m: pd.DataFrame) -> pd.DataFrame:
    need = ["shots_home", "shots_away", "sot_home", "sot_away", "corners_home", "corners_away", "yellow_home",
            "yellow_away", "red_home", "red_away"]
    m = m.assign(**{c: np.nan for c in need if c not in m.columns})
    s = sides(m, {"sh_f": ("shots_home", "shots_away"), "sh_a": ("shots_away", "shots_home"),
                  "sot_f": ("sot_home", "sot_away"), "sot_a": ("sot_away", "sot_home"),
                  "cor_f": ("corners_home", "corners_away"), "cor_a": ("corners_away", "corners_home"),
                  "yel": ("yellow_home", "yellow_away"), "red": ("red_home", "red_away")})
    s["cards"] = s["yel"] + 2 * s["red"].fillna(0)
    v, _ = decayed(s, RAW, 120.0)
    acc = v["sot_f"] / v["sh_f"].where(v["sh_f"] > 0)
    acc_allowed = v["sot_a"] / v["sh_a"].where(v["sh_a"] > 0)
    p = per_match(s, {**{c: v[c].to_numpy() for c in RAW}, "acc": acc.to_numpy(), "acca": acc_allowed.to_numpy(),
                      "tempo": (v["sh_f"] + v["sh_a"]).to_numpy()})
    out = p[["match_id"]].copy()
    for name, c in (("shots", "sh"), ("sot", "sot"), ("cor", "cor")):
        out[f"st_{name}_exp_diff"] = (p[f"{c}_f_h"] + p[f"{c}_a_a"]) / 2 - (p[f"{c}_f_a"] + p[f"{c}_a_h"]) / 2
    out["st_acc_diff"] = p["acc_h"] - p["acc_a"]
    out["st_acc_allowed_diff"] = p["acca_h"] - p["acca_a"]
    out["st_tempo"] = p["tempo_h"] + p["tempo_a"]
    out["st_tempo_gap"] = (p["tempo_h"] - p["tempo_a"]).abs()
    out["st_cards_diff"] = p["cards_h"] - p["cards_a"]
    out["st_vol_leak"] = p["sh_f_h"] * p["acca_a"] - p["sh_f_a"] * p["acca_h"]
    return out
