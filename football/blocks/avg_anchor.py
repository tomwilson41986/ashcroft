"""The early-price features anchored on football-data's market average (Avg), the one early price present from 2012 to
now (Pinnacle's early prices are stale from 23 Jul 2025 and absent from February 2026; Betfair's start in 2024/25), so
a model trained on history sees the same features the paper trial computes live. Opening information only. ``av_``:
the average book's margin, favourite and draw; the goal total its 1X2 and over/under imply; the best price's, Bet365's
and Betfair's fair home chance against the average's; the dispersion between the best and the average; the Asian
handicap line against the average's implied supremacy; our model against the average."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["av_overround", "av_fav", "av_draw", "av_total", "av_sup", "av_gap_max", "av_gap_b365", "av_gap_bfe",
            "av_disp_h", "av_disp_a", "av_ah_gap", "av_dis_model"]


def _fair(m, pre):
    inv = 1 / m[f"{pre}_h"] + 1 / m[f"{pre}_d"] + 1 / m[f"{pre}_a"]
    return (1 / m[f"{pre}_h"]) / inv, (1 / m[f"{pre}_d"]) / inv, (1 / m[f"{pre}_a"]) / inv, inv


def build(m: pd.DataFrame) -> pd.DataFrame:
    from football.team_features import implied_rates
    out = pd.DataFrame({"match_id": m["match_id"]})
    h, d, a, over = _fair(m, "bk_avg")
    io, iu = 1 / m["bk_avg_o"], 1 / m["bk_avg_u"]
    o = io / (io + iu)
    lh, la = implied_rates(h, a, o)
    out["av_overround"] = over
    out["av_fav"] = np.log(h / a)
    out["av_draw"] = d
    out["av_total"] = lh + la
    out["av_sup"] = lh - la
    for pre, name in (("bk_max", "av_gap_max"), ("bk_b365", "av_gap_b365"), ("bfe_open", "av_gap_bfe")):
        fh = _fair(m, pre)[0]
        out[name] = np.log(fh / h)
    out["av_disp_h"] = np.log(m["bk_max_h"] / m["bk_avg_h"])
    out["av_disp_a"] = np.log(m["bk_max_a"] / m["bk_avg_a"])
    ih, ia = 1 / m["open_ah_h"], 1 / m["open_ah_a"]
    out["av_ah_gap"] = (-m["open_ah_line"] + 2.5 * (ih / (ih + ia) - 0.5)) - (lh - la)
    out["av_dis_model"] = np.log(m["p_h"] / m["p_a"]) - np.log(h / a)
    return out
