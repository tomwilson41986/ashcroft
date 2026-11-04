"""Short and recency-weighted form: the last 2 and 3 matches, and the last 5 and 10 weighted linearly by recency, of
results, results against expectation and strength of schedule. ``fs_``.

racing² football research, 4 Nov 2026 (the owner: "rolling form: form last 2, 3, 5, 10 games ... form strength,
strength of schedule, home/away advantage, metrics adjusted by all the above"). The team features already carry the
last 5 and 10, an EWM and the season to date; this adds the shorter and recency-weighted windows of:

- ``pts`` and ``gd``: points and goal difference;
- ``sotd``: shots on target, for less against;
- ``pae_mkt`` / ``gae_mkt``: points and goal difference above the closing market's expectation - form adjusted
  for the opponent's strength and the venue at once (the market prices both);
- ``pae_dc`` / ``gae_dc``: the same against our Dixon-Coles walk-forward prices;
- ``opp_r``: the opponents' market-implied strength (the schedule faced).

Every window is over the side's earlier matches only and lagged to the collection day (the side's matches in one
collection window share the windows of its first there). The xG versions (and xG adjusted by schedule) are in the xG
stage (``football.xg_features``: ``x_short``, ``x_sos``). ``fs_<measure>_<window>_diff``: home less away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.blocks._util import sides
from football.xg_features import short_windows

MEASURES = ["pts", "gd", "sotd", "pae_mkt", "gae_mkt", "pae_dc", "gae_dc", "opp_r"]
WINS = ["l2", "l3", "lw5", "lw10"]
FEATURES = [f"fs_{c}_{w}_diff" for c in MEASURES for w in WINS]


def build(m: pd.DataFrame) -> pd.DataFrame:
    s = sides(m, {"sot_f": ("sot_home", "sot_away"), "sot_a": ("sot_away", "sot_home"),
                  "pw": ("p_h", "p_a"), "qw": ("qc_h", "qc_a"), "lf": ("lam_h", "lam_a"), "la": ("lam_a", "lam_h"),
                  "mf": ("mclam_h", "mclam_a"), "ma": ("mclam_a", "mclam_h"), "opp_r": ("r_a", "r_h")})
    pd_ = s["match_id"].map(m.set_index("match_id")["p_d"])
    qd = s["match_id"].map(m.set_index("match_id")["qc_d"])
    d = s["gf"] - s["ga"]
    s["pts"] = np.where(d.isna(), np.nan, np.select([d > 0, d == 0], [3.0, 1.0], 0.0))
    s["gd"] = d
    s["sotd"] = s["sot_f"] - s["sot_a"]
    s["pae_mkt"] = s["pts"] - (3 * s["qw"] + qd)
    s["gae_mkt"] = d - (s["mf"] - s["ma"])
    s["pae_dc"] = s["pts"] - (3 * s["pw"] + pd_)
    s["gae_dc"] = d - (s["lf"] - s["la"])
    w = short_windows(s, MEASURES)
    w["venue"], w["match_id"] = s["venue"].to_numpy(), s["match_id"].to_numpy()
    h = w[w["venue"] == "H"].set_index("match_id")
    a = w[w["venue"] == "A"].set_index("match_id")
    idx = h.index.union(a.index)
    out = pd.DataFrame({"match_id": idx})
    for c in MEASURES:
        for k in WINS:
            out[f"fs_{c}_{k}_diff"] = (h[f"{c}_{k}"].reindex(idx) - a[f"{c}_{k}"].reindex(idx)).to_numpy()
    return out[["match_id"] + FEATURES]
