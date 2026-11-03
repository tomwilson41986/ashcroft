"""The market at the early price: the book's margin, the favourite's strength, the draw, the goal total it implies,
and Betfair's early price against the benchmark open where both exist. Opening information only. ``ms_``."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["ms_overround", "ms_fav", "ms_draw", "ms_total", "ms_ou_skew", "ms_bfe_gap_h"]


def build(m: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"match_id": m["match_id"]})
    out["ms_overround"] = 1 / m["open_h"] + 1 / m["open_d"] + 1 / m["open_a"]
    out["ms_fav"] = np.log(m["qo_h"] / m["qo_a"])
    out["ms_draw"] = m["qo_d"]
    out["ms_total"] = m["molam_h"] + m["molam_a"]
    out["ms_ou_skew"] = m["qo_o"] - 0.5
    bh = 1 / m["bfe_open_h"]
    bfair = bh / (1 / m["bfe_open_h"] + 1 / m["bfe_open_d"] + 1 / m["bfe_open_a"])
    out["ms_bfe_gap_h"] = np.log(bfair / m["qo_h"])
    return out
