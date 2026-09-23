"""Handicap angles: penalty carriers and weight against mark."""

import numpy as np
import pandas as pd
import pytest

from model.handicap_features import ASSUMED_PENALTY_LBS, add_handicap_features
from model.perf_figures import lbs_per_length


def _card():
    rows = []
    # race 1 (1 Jan, 8f handicap): A wins by 3 lengths from B
    for h, plc, bt, orr, wt in [("A", 1, "0", 70, 130), ("B", 2, "3", 72, 132), ("C", 3, "4", 65, 125)]:
        rows.append(("2025-01-01", "2.00", "Ascot", "Handicap", h, plc, bt, orr, wt, 8.0))
    # race 2 (10 Jan, 8f handicap): A runs again off the SAME mark, under a penalty
    for h, plc, bt, orr, wt in [("A", 2, "1", 70, 136), ("D", 1, "0", 80, 140), ("E", 3, "2", 75, 135)]:
        rows.append(("2025-01-10", "3.00", "Ascot", "Handicap", h, plc, bt, orr, wt, 8.0))
    return pd.DataFrame(rows, columns=["race_date", "race_time", "track", "race_type", "horse_name", "placing_numerical",
                                       "total_dst_bt", "official_rating", "pounds", "dist_furlongs"]).assign(
        race_date=lambda d: pd.to_datetime(d["race_date"]))


def test_penalty_carrier_and_its_edge():
    out, names = add_handicap_features(_card())
    a2 = out[(out["horse_name"] == "A") & (out["race_date"] == pd.Timestamp("2025-01-10"))].iloc[0]
    assert a2["hc_lr_won"] == 1 and a2["hc_days_since_lr"] == 9
    assert a2["hc_mark_unchanged_after_win"] == 1
    margin_lbs = 3 * lbs_per_length(8.0)
    assert a2["hc_lr_win_margin_lbs"] == pytest.approx(margin_lbs)
    assert a2["hc_penalty_edge_lbs"] == pytest.approx(margin_lbs - ASSUMED_PENALTY_LBS)
    # weight against mark: D is top (140 at 80); A at 70 "should" carry 130 and carries 136 -> +6 (the penalty)
    assert a2["hc_wt_vs_mark"] == pytest.approx(6.0)
    d = out[out["horse_name"] == "D"].iloc[0]
    assert d["hc_wt_vs_mark"] == pytest.approx(0.0) and np.isnan(d["hc_lr_won"])


def test_a_horse_that_lost_last_time_is_no_penalty_carrier():
    out, _ = add_handicap_features(_card())
    b = out[out["horse_name"] == "B"].iloc[0]
    assert np.isnan(b["hc_penalty_edge_lbs"]) and np.isnan(b["hc_lr_won"])


def test_todays_result_never_reaches_todays_features():
    d = _card()
    out1, names = add_handicap_features(d)
    d2 = d.copy()
    d2.loc[d2["race_date"] == pd.Timestamp("2025-01-10"), "placing_numerical"] = [3, 1, 2]
    d2.loc[d2["race_date"] == pd.Timestamp("2025-01-10"), "total_dst_bt"] = ["9", "0", "5"]
    out2, _ = add_handicap_features(d2)
    m = out1["race_date"] == pd.Timestamp("2025-01-10")
    for c in names:
        assert np.array_equal(out1.loc[m, c].to_numpy(), out2.loc[m, c].to_numpy(), equal_nan=True), c


def test_best_winning_mark_is_remembered_across_losing_runs():
    rows = [("2025-01-01", 1, 80), ("2025-02-01", 5, 82), ("2025-03-01", 6, 78), ("2025-04-01", 4, 75)]
    d = pd.DataFrame([{"race_date": pd.Timestamp(dt), "race_time": "2.00", "track": "York", "race_type": "Handicap",
                       "horse_name": "H", "placing_numerical": p, "total_dst_bt": "0" if p == 1 else "5",
                       "official_rating": o, "pounds": 130, "dist_furlongs": 8.0} for dt, p, o in rows])
    out, _ = add_handicap_features(d)
    out = out.sort_values("race_date")
    # won off 80 in January; by April (mark 75) it has won off a higher mark, two losing runs later
    assert out["hc_wins_off_higher_mark"].tolist()[1:] == [0.0, 1.0, 1.0]
    assert out["hc_mark_vs_last_win"].tolist()[1:] == [2.0, -2.0, -5.0]
