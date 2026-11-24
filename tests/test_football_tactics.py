"""The tactics stage and the match chain: shapes from positions, game states from goal events, the chain's chances."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football import tactics
from football.blocks import formation, match_sim, sim


def eleven(fid, side, shape):
    pos = ["G"] + ["D"] * shape[0] + ["M"] * shape[1] + ["F"] * shape[2]
    return [{"apif_fixture_id": fid, "side": side, "role": "starter", "apif_player_id": 100 * fid + i +
             (50 if side == "away" else 0), "player": f"p{i}", "pos": p} for i, p in enumerate(pos)]


def test_shapes_count_the_starters_positions_and_need_a_whole_eleven():
    rows = eleven(1, "home", (4, 4, 2)) + eleven(1, "away", (3, 5, 2)) + eleven(2, "home", (4, 3, 3)) \
        + eleven(2, "away", (5, 4, 1))[:-1]                                  # ten starters: no shape
    rows.append({"apif_fixture_id": 1, "side": "home", "role": "substitute", "apif_player_id": 999, "player": "s",
                 "pos": "F"})
    s = tactics.shapes(pd.DataFrame(rows)).set_index("apif_fixture_id")
    assert s.loc[1, ["tq_def_h", "tq_mid_h", "tq_fwd_h", "tq_def_a", "tq_mid_a", "tq_fwd_a"]].tolist() == \
        [4, 4, 2, 3, 5, 2]
    assert s.loc[2, "tq_fwd_h"] == 3 and np.isnan(s.loc[2, "tq_def_a"])
    assert formation.kind([3, 4, 5, 6, np.nan], [1, 2, 3, 1, 2]).tolist()[:4] == [31, 42, 52, 51]


def test_goal_states_split_the_minutes_and_goals_by_the_home_sides_state():
    hm = pd.DataFrame({"apif_fixture_id": [1, 2], "apif_home_id": [10, 10], "ft_home": [2, 1], "ft_away": [1, 0],
                       "has_events": [True, True]})
    he = pd.DataFrame({"apif_fixture_id": [1, 1, 1, 1, 2], "minute": [20, 50, 60, 70, 30],
                       "type": ["Goal", "Goal", "Goal", "Goal", "Card"],
                       "detail": ["Normal Goal", "Normal Goal", "Missed Penalty", "Normal Goal", "Red Card"],
                       "apif_team_id": [10, 20, 10, 10, 10]})
    g = tactics.goal_states(hm, he).set_index("apif_fixture_id")
    r = g.loc[1]
    # home scores at 20 (level), away at 50 (home leading), home at 70 (level): level 20 + 20, lead 30 + 20
    assert (r["gs_min_level"], r["gs_min_lead"], r["gs_min_trail"]) == (40, 50, 0)
    assert (r["gs_hg_level"], r["gs_ag_lead"], r["gs_hg_lead"]) == (2, 1, 0)
    assert 2 not in g.index                                   # a 1-0 with no goal event: the events do not add up


def test_the_chain_with_no_state_effect_is_the_poisson_and_a_side_that_holds_a_lead_wins_more():
    lh, la = np.array([1.5, 1.0]), np.array([1.1, 1.6])
    ones = np.ones((2, 3))
    h, d, a, goals = match_sim.play(lh, la, ones, ones)
    ph, pd_, pa = sim.chances(lh, la)
    assert h + d + a == pytest.approx(1.0)
    assert h == pytest.approx(ph, abs=0.01) and d == pytest.approx(pd_, abs=0.01)   # a minute at a time, not Poisson
    assert goals == pytest.approx(lh + la, rel=1e-9)
    # the away side scores at half its rate while the home side leads: the home side wins more, the draw less
    hold = ones.copy()
    hold[:, 2] = 0.5
    h2, d2, a2, g2 = match_sim.play(lh, la, ones, hold)
    assert (h2 > h).all() and (d2 < d).all() and (g2 < goals).all()
