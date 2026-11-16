"""The review's comparison of arms, and the helper its new blocks share."""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.blocks._util import prior_sum
from research.open_close import football_review_arms as ra


def test_prior_sum_reads_only_rows_before_the_collection_day_and_only_the_window():
    date = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-02"])
    group = ["a", "a", "a", "a", "b"]
    value = [1.0, 2.0, np.nan, 4.0, 10.0]
    coll = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-04", "2026-01-05", "2026-01-09"])
    s, n = prior_sum(group, date, coll, value, window=2)
    assert s.tolist() == [0.0, 1.0, 3.0, 6.0, 10.0]        # row 3: the last two with a value (2 and 4), not the first
    assert n.tolist() == [0.0, 1.0, 2.0, 2.0, 1.0]         # row 4 (group b) may read itself only because coll is later


def test_an_arm_identical_to_the_base_has_no_difference_and_a_better_one_is_positive():
    rng = np.random.default_rng(0)
    days = np.repeat(np.arange(200), 10).astype(str)
    y = rng.normal(0, 0.05, 2000)
    base, better = (y - 0.0) ** 2, (y - 0.5 * y) ** 2
    same = ra.paired(days, base, base, y ** 2)
    assert same["vs_base"] == 0 and same["lo95"] == 0 and same["hi95"] == 0
    up = ra.paired(days, base, better, y ** 2)
    assert abs(up["vs_base"] - 0.75) < 1e-9 and up["lo95"] > 0.7 and up["days"] == 200


def test_arm_columns():
    base = ["d_pts_l5", "d_gd_ewm", "d_pae_mkt_l5", "fs_pts_l2_diff", "dis_open"]
    bfeat = {"kalman_form": ["kf_gae_diff"], "league_fx": ["lf_move_380"]}
    have = base + ["lq_h", "lq_a"]
    assert ra.columns(base, ra.ARMS["price"], bfeat, have) == base + ["lq_h", "lq_a"]
    assert ra.columns(base, ra.ARMS["kalman_for_windows"], bfeat, have) == ["d_pae_mkt_l5", "dis_open", "kf_gae_diff"]
    # a column the matrix does not hold (no xG stage) is left out, not invented
    assert ra.columns(base, ra.ARMS["shrunk_ratings"], bfeat, have) == base


def test_a_logit_forecast_turns_back_into_the_log_chance_move():
    qo, qc = np.array([0.2, 0.5, 0.8]), np.array([0.25, 0.45, 0.82])
    p = ra.logit(qc) - ra.logit(qo)
    back = np.log(1 / (1 + np.exp(-(ra.logit(qo) + p))) / qo)
    assert np.allclose(back, np.log(qc / qo))


def test_the_kalman_block_filters_a_side_that_beats_the_market(tmp_path):
    from football.blocks import kalman_form as kf
    rows, d0 = [], pd.Timestamp("2020-08-01")
    for i in range(40):                                      # GOOD beats the market's expectation by a goal every week
        rows.append({"match_id": f"m{i}", "date": d0 + pd.Timedelta(days=7 * i), "season": 2020,
                     "competition_id": "X1", "country": "X", "home_team_id": "GOOD", "away_team_id": f"O{i % 5}",
                     "ft_home": 2.0, "ft_away": 0.0, "mclam_h": 1.5, "mclam_a": 0.5, "sot_home": 6.0, "sot_away": 3.0})
    b = kf.build(pd.DataFrame(rows))
    assert b["kf_gae_diff"].iloc[0] == 0                     # no history: the market's view
    assert b["kf_gae_diff"].iloc[-1] > 0.5                   # after 39 matches a goal better: most of it filtered in
    assert b["kf_gae_sd"].iloc[-1] < b["kf_gae_sd"].iloc[0]  # and surer of it
