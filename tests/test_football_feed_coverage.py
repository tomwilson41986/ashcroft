"""Feed coverage: linking and the summary's shares."""

from __future__ import annotations

import pandas as pd

from research.open_close import football_feed_coverage as fc


def test_link_and_summary_shares():
    m = pd.DataFrame({"market_id": ["1", "2"], "event_name": ["Man Utd v Wolves", "Nowhere FC v Ghost Town"],
                      "kickoff": pd.to_datetime(["2026-11-07T15:00:00Z", "2026-11-07T15:00:00Z"], utc=True),
                      "ev_home": ["Man Utd", "Nowhere FC"], "ev_away": ["Wolves", "Ghost Town"],
                      "matched": [900.0, 100.0], "country": ["GB", "XX"], "competition": ["EPL", "Nothing"]})
    fx = pd.DataFrame({"apif_fixture_id": [11], "apif_league_id": [39], "league": ["Premier League"],
                       "league_country": ["England"], "apif_season": [2026], "status": ["SCHEDULED"],
                       "kickoff_utc": ["2026-11-07T15:00:00+00:00"], "home": ["Manchester United"],
                       "away": ["Wolverhampton Wanderers"]})
    j = fc.link(m, fx)
    assert j["apif_fixture_id"].tolist()[0] == 11 and pd.isna(j["apif_fixture_id"].tolist()[1])
    cov = pd.DataFrame({"apif_league_id": [39], "apif_season": [2026], **{f: [True] for f in fc.FLAGS}})
    s = fc.summarise(j, cov, {39: True}, {39})
    assert s["linked_to_api_football"] == {"matches": 0.5, "matched_money": 0.9}
    assert s["feed_offers"]["lineups"]["matched_money"] == 0.9
    assert s["expected_goals_in_team_statistics"]["matches"] == 0.5


def test_leagues_to_hold_keeps_covered_leagues_only():
    j = pd.DataFrame({"market_id": ["1", "2"], "apif_league_id": [39.0, 99.0], "league": ["PL", "X"],
                      "league_country": ["England", "Y"], "matched": [10.0, 5.0]})
    cov = pd.DataFrame({"apif_league_id": [39, 99], "apif_season": [2026, 2026],
                        **{f: [True, False] for f in fc.FLAGS}})
    fx = pd.DataFrame({"apif_league_id": [39] * 3, "apif_season": [2026] * 3, "status": ["FINISHED"] * 3})
    out = fc.leagues_to_hold([j], cov, fx)
    assert [x["apif_league_id"] for x in out] == [39] and out[0]["finished_per_season"] == {2026: 3}
