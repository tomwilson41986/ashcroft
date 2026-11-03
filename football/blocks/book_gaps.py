"""Bookmakers against each other at the early price: the market average's, the best price's and Bet365's fair home
chance against the benchmark open, and the spread between the best and the average (dispersion). A soft book away
from the sharp one, or a wide spread, is a disagreement the close may settle. Opening information only. ``bk_``."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["bk_gap_avg", "bk_gap_b365", "bk_gap_max", "bk_disp_h", "bk_disp_a", "bk_avg_over"]


def _fair_h(m, pre):
    inv = 1 / m[f"{pre}_h"] + 1 / m[f"{pre}_d"] + 1 / m[f"{pre}_a"]
    return (1 / m[f"{pre}_h"]) / inv, inv


def build(m: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"match_id": m["match_id"]})
    for pre, name in (("bk_avg", "bk_gap_avg"), ("bk_b365", "bk_gap_b365"), ("bk_max", "bk_gap_max")):
        if f"{pre}_h" in m.columns:
            fh, _ = _fair_h(m, pre)
            out[name] = np.log(fh / m["qo_h"])
        else:
            out[name] = np.nan
    if "bk_max_h" in m.columns:
        out["bk_disp_h"] = np.log(m["bk_max_h"] / m["bk_avg_h"])
        out["bk_disp_a"] = np.log(m["bk_max_a"] / m["bk_avg_a"])
        out["bk_avg_over"] = 1 / m["bk_avg_h"] + 1 / m["bk_avg_d"] + 1 / m["bk_avg_a"]
    else:
        out["bk_disp_h"] = out["bk_disp_a"] = out["bk_avg_over"] = np.nan
    return out
