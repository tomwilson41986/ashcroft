"""Sportmonks' xG adjusted for the opposition. ``sa_``.

racing² football research, 24 Nov 2026 (the owner: "strength adjustments ... more advanced xG"). ``sm_xg`` averages
each side's raw non-penalty xG; a side that met weak defences looks better than it is. Here each past match's xG is
read against the opponent: the xG a side made less what that opponent usually concedes (its 120-day half-life mean as
it stood before that match), and the xG it conceded less what that opponent usually makes. Those adjusted values are
then averaged the same way (120-day half-life) and lagged to the collection day:

- ``sa_att_diff``: adjusted attack (xG made over what the opponents allow), home less away;
- ``sa_def_diff``: adjusted defence (xG conceded over what the opponents make; lower is better), home less away;
- ``sa_net_diff``: adjusted attack less adjusted defence, home less away;
- ``sa_w``: the two sides' effective number of adjusted matches, summed.

A past match whose opponent had no earlier Sportmonks match is not counted. Blank before 2024/25.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.blocks._util import decayed, per_match, sides
from football.sportmonks_stage import COLUMNS

FEATURES = ["sa_att_diff", "sa_def_diff", "sa_net_diff", "sa_w"]
HALF = 120.0


def opponent_values(s: pd.DataFrame, v: pd.DataFrame) -> pd.DataFrame:
    """For each side-row, the opponent's row of ``v`` in the same match (its values before that match)."""
    key = pd.Series(np.arange(len(s)), index=pd.MultiIndex.from_arrays([s["match_id"], s["team"]]))
    j = key.reindex(pd.MultiIndex.from_arrays([s["match_id"], s["opp"]])).to_numpy()
    out = pd.DataFrame(np.nan, index=s.index, columns=v.columns)
    ok = ~np.isnan(j)
    out.loc[ok] = v.to_numpy()[j[ok].astype(int)]
    return out


def build(m: pd.DataFrame) -> pd.DataFrame:
    m = m.assign(**{c: np.nan for c in COLUMNS if c not in m.columns})
    s = sides(m, {"xf": ("smx_xg_npxg_h", "smx_xg_npxg_a"), "xa": ("smx_xg_npxg_a", "smx_xg_npxg_h")})
    v, _ = decayed(s, ["xf", "xa"], HALF)                    # each side's usual xG made and conceded, before the match
    o = opponent_values(s, v)
    s["af"] = s["xf"] - o["xa"]                               # made, over what this opponent usually concedes
    s["ad"] = s["xa"] - o["xf"]                               # conceded, over what this opponent usually makes
    s["an"] = s["af"] - s["ad"]
    v2, w = decayed(s, ["af", "ad", "an"], HALF)
    have = np.where(np.isfinite(v2["af"].to_numpy()), w, np.nan)
    p = per_match(s, {"af": v2["af"].to_numpy(), "ad": v2["ad"].to_numpy(), "an": v2["an"].to_numpy(), "w": have})
    out = p[["match_id"]].copy()
    out["sa_att_diff"] = p["af_h"] - p["af_a"]
    out["sa_def_diff"] = p["ad_h"] - p["ad_a"]
    out["sa_net_diff"] = p["an_h"] - p["an_a"]
    out["sa_w"] = p["w_h"] + p["w_a"]
    return out
