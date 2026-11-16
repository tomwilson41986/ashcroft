"""League effects, shrunk: what the league's prices and results did lately, in place of the league as a bare label.
``lf_``.

racing² football research, 10 Nov 2026 (the model review, finding F: the league enters LightGBM as a 38-level
category and is its largest feature by gain, free to memorise). From the league's matches before each match's
collection day only, each a sum over (count + ``K``) so a thin record stays near zero:

- ``lf_move_380`` / ``lf_move_1900``: the home side's early-to-close move of its chance (log), over the league's last
  380 and 1,900 matches (about one and five seasons): does this league's home price shorten or drift;
- ``lf_haerr_380`` / ``lf_haerr_1900``: home goal difference less the closing market's expected one: the market's
  error on home advantage in this league (the review's finding E: what moves prices is the market's miss, not the
  raw level);
- ``lf_absmove_380``: the mean size of the home move: how much this league's prices move at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.blocks._util import collection_day, prior_sum

FEATURES = ["lf_move_380", "lf_move_1900", "lf_haerr_380", "lf_haerr_1900", "lf_absmove_380"]
K = 100.0


def build(m: pd.DataFrame) -> pd.DataFrame:
    date = pd.to_datetime(m["date"])
    coll = collection_day(date).to_numpy()
    y = np.array(np.log(m["qc_h"] / m["qo_h"]), dtype=float)
    y[~np.isfinite(y)] = np.nan
    err = ((m["ft_home"] - m["ft_away"]) - (m["mclam_h"] - m["mclam_a"])).to_numpy(float)
    out = pd.DataFrame({"match_id": m["match_id"].to_numpy()})
    for w in (380, 1900):
        s, n = prior_sum(m["competition_id"], date, coll, y, w)
        out[f"lf_move_{w}"] = s / (n + K)
        s, n = prior_sum(m["competition_id"], date, coll, err, w)
        out[f"lf_haerr_{w}"] = s / (n + K)
    s, n = prior_sum(m["competition_id"], date, coll, np.abs(y), 380)
    out["lf_absmove_380"] = np.where(n >= 50, s / np.maximum(n, 1), np.nan)
    return out[["match_id"] + FEATURES]
