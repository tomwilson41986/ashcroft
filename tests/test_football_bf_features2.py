"""The Betfair price-file features of 24 Nov: deltas into the checkpoint, the row's own change, pace."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.open_close import football_bf_features2 as f2


def test_each_checkpoint_reads_the_change_from_the_grid_point_before_it_and_nothing_after():
    imp = pd.DataFrame({"event_id": ["e"] * 4, "cp_min": [720, 360, 180, 120],
                        "p_home": [0.40, 0.45, 0.47, 0.50], "total": [2.5, 2.6, 2.6, 2.8]})
    got = f2.implied_deltas(imp).set_index("cp_min")
    assert got.loc[360, "idl_d_p_home"] == pytest.approx(0.05)
    assert got.loc[120, "idl_d_p_home"] == pytest.approx(0.03)
    assert got.loc[120, "idl_d_total"] == pytest.approx(0.2)
    assert 60 not in got.index                                       # no 60 or 90 rows: nothing borrowed


def test_the_own_change_follows_the_row_s_role():
    d = pd.DataFrame({"role": ["home", "draw", "away"], "idl_d_p_home": [0.1] * 3, "idl_d_p_draw": [0.2] * 3,
                      "idl_d_p_away": [0.3] * 3})
    assert f2.own_change(d).tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_pace_and_acceleration():
    d = pd.DataFrame({"path_n_ticks_15m": [9.0], "path_n_ticks_1h": [9.0], "path_ret_15m": [0.1],
                      "path_ret_1h": [0.2], "path_ret_6h": [0.6]})
    got = f2.pace(d).iloc[0]
    assert got["pace_ticks_15_over_1h"] == pytest.approx(0.9)
    assert got["pace_acc_15m"] == pytest.approx(0.05)
    assert got["pace_acc_1h"] == pytest.approx(0.1)
