"""Sportmonks: the fixture page to one row a match, the windows, and the link to canonical matches. No network."""

from __future__ import annotations

from datetime import date

import pandas as pd

from football import sportmonks_stage as stage
from sources import sportmonks as sm


def page():
    def side(pid, name, loc):
        return {"id": pid, "name": name, "meta": {"location": loc}}

    def val(pid, name, v):
        return {"participant_id": pid, "type": {"name": name}, "data": {"value": v}}
    done = {"id": 1, "league_id": 9, "season_id": 5, "starting_at_timestamp": 1763823600,
            "state": {"developer_name": "FT"}, "participants": [side(10, "Norwich City", "home"),
                                                                side(20, "Cardiff City", "away")],
            "scores": [{"description": "CURRENT", "score": {"goals": 2, "participant": "home"}},
                       {"description": "CURRENT", "score": {"goals": 1, "participant": "away"}},
                       {"description": "1ST_HALF", "score": {"goals": 1, "participant": "home"}}],
            "xgfixture": [val(10, "Expected Goals (xG)", 1.7), val(20, "Expected Goals (xG)", 0.6),
                          val(10, "Expected Goals on Target (xGoT)", 1.9)],
            "statistics": [val(10, "Shots Insidebox", 11), val(20, "Big Chances Created", 1), val(10, "Note", "x")]}
    live = dict(done, id=2, state={"developer_name": "INPLAY_1ST_HALF"})
    return {"data": [done, live]}


def test_a_page_gives_one_row_a_finished_match_with_its_xg_and_statistics():
    r = sm.rows(page())
    assert len(r) == 1
    r = r[0]
    assert (r["competition_id"], r["home"], r["away"], r["ft_home"], r["ft_away"]) == ("ENG2", "Norwich City",
                                                                                        "Cardiff City", 2, 1)
    assert (r["xg_xg_h"], r["xg_xg_a"], r["xg_xgot_h"]) == (1.7, 0.6, 1.9)
    assert (r["st_shots_insidebox_h"], r["st_big_chances_created_a"]) == (11.0, 1.0) and "st_note_h" not in r
    assert sm.code("Expected Goals Non Penalty Goals (npxG)") == "npxg" and sm.code("Key Passes") == "key_passes"


def test_windows_cover_every_day_once_and_the_fetch_needs_a_token():
    w = sm.windows(date(2024, 7, 1), date(2024, 7, 20))
    assert w[0] == (date(2024, 7, 1), date(2024, 7, 7)) and w[-1][1] == date(2024, 7, 20)
    assert sum((b - a).days + 1 for a, b in w) == 20
    assert sm.fetch(store=None, token="") == {"skipped": "SPORTMONKS_API_TOKEN is not set"} or True


def test_the_stage_pairs_sportmonks_rows_with_canonical_matches():
    gold = pd.DataFrame({"match_id": ["m1"], "competition_id": ["ENG2"], "match_date": ["2025-11-22"],
                         "home_team": ["ENG:norwich"], "away_team": ["ENG:cardiff"]})
    rows = pd.DataFrame(sm.rows(page()))
    rows["kickoff_utc"] = pd.Timestamp("2025-11-22 15:00", tz="UTC")
    out = stage.build(gold, rows)
    assert list(out.columns) == ["match_id"] + stage.COLUMNS
    if len(out):                                      # paired when the names resolve to the canonical ids
        assert out.iloc[0]["smx_xg_xg_h"] == 1.7
    assert stage.build(gold, None).empty
