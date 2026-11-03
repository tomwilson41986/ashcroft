"""Cross-market consistency at the early price: the goal supremacy the Asian handicap line implies against the one
the 1X2 prices imply, the handicap's price skew, and the over/under total against the 1X2-implied total. A
disagreement between markets is one the close may resolve. Opening information only. ``ah_``."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["ah_sup_gap", "ah_skew", "ah_line"]


def build(m: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"match_id": m["match_id"]})
    sup_1x2 = m["molam_h"] - m["molam_a"]
    ih, ia = 1 / m["open_ah_h"], 1 / m["open_ah_a"]
    p_home = ih / (ih + ia)
    # the line plus a price adjustment: each 0.1 of handicap chance away from 0.5 is worth about a quarter goal
    out["ah_sup_gap"] = (-m["open_ah_line"] + 2.5 * (p_home - 0.5)) - sup_1x2
    out["ah_skew"] = np.log(m["open_ah_a"] / m["open_ah_h"])
    out["ah_line"] = m["open_ah_line"]
    return out
