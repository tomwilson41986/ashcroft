"""Move memory: how each side's price, and prices like this one, have moved before. ``mm_``.

racing² football research, 10 Nov 2026 (the model review: the horse model's "steamers" and the quant block's record
against the closing price, for football). From matches before each match's collection day only (their closes are
known by then), each a sum over (count + k) so a thin record stays near zero:

- ``mm_team_h`` / ``mm_team_a`` / ``mm_team_diff``: the side's own chance's early-to-close move (log) over its last
  10 matches (k = 5): a side the market keeps backing late, or keeps opposing;
- ``mm_band_h`` / ``mm_band_a``: the same move over the last 3,000 home (away) sides whose early chance was in the
  same band (k = 200): the drift by price level (the favourite-longshot pattern of the move, and whatever the
  margin's removal leaves in the target: the review's finding A4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.blocks._util import collection_day, prior_sum, sides

FEATURES = ["mm_team_diff", "mm_team_h", "mm_team_a", "mm_band_h", "mm_band_a"]
EDGES = [0.2, 0.35, 0.5, 0.65]
K_TEAM, K_BAND = 5.0, 200.0


def build(m: pd.DataFrame) -> pd.DataFrame:
    mv = m.assign(y_h=np.log(m["qc_h"] / m["qo_h"]), y_a=np.log(m["qc_a"] / m["qo_a"]))
    s = sides(mv, {"y": ("y_h", "y_a")})
    y = np.array(s["y"], dtype=float)
    y[~np.isfinite(y)] = np.nan
    tot, n = prior_sum(s["team"], s["date"], collection_day(s["date"]).to_numpy(), y, 10)
    s["mem"] = tot / (n + K_TEAM)
    h = s[s["venue"] == "H"].set_index("match_id")["mem"]
    a = s[s["venue"] == "A"].set_index("match_id")["mem"]
    out = pd.DataFrame({"match_id": m["match_id"].to_numpy()})
    out["mm_team_h"], out["mm_team_a"] = h.reindex(out["match_id"]).to_numpy(), a.reindex(out["match_id"]).to_numpy()
    out["mm_team_diff"] = out["mm_team_h"] - out["mm_team_a"]
    date = pd.to_datetime(m["date"])
    coll = collection_day(date).to_numpy()
    for side in ("h", "a"):
        q = m[f"qo_{side}"].to_numpy(float)
        band = np.where(np.isfinite(q), np.digitize(q, EDGES), -1)
        v = np.array(mv[f"y_{side}"], dtype=float)
        v[~np.isfinite(v)] = np.nan
        tot, n = prior_sum(band, date, coll, v, 3000)
        out[f"mm_band_{side}"] = np.where(band >= 0, tot / (n + K_BAND), np.nan)
    return out[["match_id"] + FEATURES]
