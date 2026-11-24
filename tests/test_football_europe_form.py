"""European form: teams mapped through linked domestic matches, and only European matches before the collection day."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football import europe_form as ef
from football.blocks import eu_form


def frames():
    gold = pd.DataFrame({"match_id": ["d1", "d2", "d3"],
                         "match_date": ["2025-09-13", "2025-09-20", "2025-09-20"],
                         "home_team_id": ["ENG:arsenal", "ENG:arsenal", "ENG:fulham"],
                         "away_team_id": ["ENG:fulham", "ENG:chelsea", "ENG:brentford"]})
    linked = pd.DataFrame({"match_id": ["d1"], "sm_home_id": [19], "sm_away_id": [11]})
    sm = pd.DataFrame({
        "competition_id": ["UCL", "UCL", "UEL"],
        "kickoff_utc": pd.to_datetime(["2025-09-16 19:00", "2025-09-19 19:00", "2025-09-10 19:00"], utc=True),
        "sm_home_id": [19, 19, 11], "sm_away_id": [500, 501, 502],
        "ft_home": [2, 0, 1], "ft_away": [0, 3, 1],
        "xg_npxg_h": [2.1, 0.5, 1.0], "xg_npxg_a": [0.4, 2.0, 1.2],
        "od_q_h": [0.6, 0.5, 0.4], "od_q_d": [0.25, 0.25, 0.3], "od_q_a": [0.15, 0.25, 0.3]})
    return gold, linked, sm


def test_teams_map_through_linked_matches_and_only_earlier_european_games_count():
    gold, linked, sm = frames()
    assert ef.team_map(linked, gold) == {19: "ENG:arsenal", 11: "ENG:fulham"}
    out = ef.build(gold, sm, linked, since="2025-09-01").set_index("match_id")
    # d2 is a Saturday (20 Sep): collection Friday 19 Sep, so Arsenal's 19 Sep match is not counted, the 16th is
    r = out.loc["d2"]
    assert r["eux_n90_h"] == 1 and r["eux_days_h"] == 4 and r["eux_xgd_h"] == pytest.approx(2.1 - 0.4)
    assert r["eux_pae_h"] == pytest.approx(3 - (3 * 0.6 + 0.25))
    assert r["eux_n90_a"] == 0 and np.isnan(r["eux_days_a"])              # Chelsea: no mapped European match
    # d1 (13 Sep, collection 12 Sep): Arsenal had none yet; Fulham's 10 Sep draw counts
    assert out.loc["d1", "eux_n90_h"] == 0 and out.loc["d1", "eux_n90_a"] == 1
    assert out.loc["d1", "eux_pae_a"] == pytest.approx(1 - (3 * 0.4 + 0.3))        # Fulham were at home
    f = eu_form.build(out.reset_index()).set_index("match_id")
    assert f.loc["d2", "eu_n90_diff"] == 1 and f.loc["d2", "eu_rest_min"] == 4 and f.loc["d2", "eu_any"] == 1
    assert f.loc["d2", "eu_xgd_diff"] == pytest.approx(1.7)                  # Chelsea counts as 0, not unknown
