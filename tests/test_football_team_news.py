"""football/team_news.py: the expected eleven, the absentees and their replacements, and the win probabilities."""
from __future__ import annotations

import pandas as pd
import pytest

from football import team_news as tn


def _app():
    """Team 1, five earlier matches: players 1-11 start each; 12 (a defender) comes on in some."""
    rows = []
    for k in range(5):
        fid, d = 100 + k, pd.Timestamp("2026-11-01") + pd.Timedelta(days=7 * k)
        for p in range(1, 12):
            rows.append({"apif_fixture_id": fid, "apif_team_id": 1, "apif_player_id": p, "player": f"P{p}",
                         "pos": "G" if p == 1 else "D" if p <= 5 else "M" if p <= 8 else "F", "role": "starter",
                         "date": d})
        if k % 2 == 0:
            rows.append({"apif_fixture_id": fid, "apif_team_id": 1, "apif_player_id": 12, "player": "P12", "pos": "D",
                         "role": "substitute", "date": d})
    return pd.DataFrame(rows)


def test_an_absent_regular_is_replaced_from_his_position_and_the_loss_is_the_rating_gap():
    ratings = {2: 0.20, 12: 0.05}
    out = pd.DataFrame([{"apif_player_id": 2, "player": "P2", "type": "Missing Fixture", "reason": "Knee"}])
    r = tn.side_news(_app(), 1, pd.Timestamp("2026-12-10"), out, ratings)
    a = r["absentees"][0]
    assert a["replacement"] == ["P12"] and a["start_share"] == 1.0
    assert r["lost_gd90"] == pytest.approx(0.15)


def test_a_doubt_counts_half_and_a_player_who_never_started_costs_nothing():
    ratings = {2: 0.20, 12: 0.05, 99: 0.5}
    out = pd.DataFrame([{"apif_player_id": 2, "player": "P2", "type": "Questionable", "reason": "Knock"},
                        {"apif_player_id": 99, "player": "P99", "type": "Missing Fixture", "reason": "Injury"}])
    r = tn.side_news(_app(), 1, pd.Timestamp("2026-12-10"), out, ratings)
    assert r["lost_gd90"] == pytest.approx(0.075)


def test_too_little_history_says_no_data_found():
    app = _app()
    r = tn.side_news(app[app["apif_fixture_id"] < 102], 1, pd.Timestamp("2026-12-10"), app.iloc[:0], {})
    assert "no data found" in r["note"]


def test_losses_lower_that_sides_win_chance_and_none_leave_the_market_unchanged():
    none = tn.with_strength(0.45, 0.28, None, 0.0, 0.0)
    assert none["home_win_pp_lost_to_absences"] == 0.0 and none["away_win_pp_lost_to_absences"] == 0.0
    hit = tn.with_strength(0.45, 0.28, None, 0.3, 0.0)
    assert hit["home_win_pp_lost_to_absences"] > 0 and hit["home_full_strength"]["p_h"] > hit["model_at_market_rates"]["p_h"]


def test_injuries_keep_the_latest_type_and_the_first_sighting():
    def doc(t, typ):
        return {"fetched_utc": t, "body": {"response": [{"player": {"id": 7, "name": "X", "type": typ, "reason": "r"},
                                                         "fixture": {"id": 5}, "team": {"id": 1}}]}}
    d = tn.injuries([doc("2026-11-26T06:00:00", "Questionable"), doc("2026-11-26T09:00:00", "Missing Fixture")])
    assert len(d) == 1 and d.iloc[0]["type"] == "Missing Fixture" and d.iloc[0]["first_seen_utc"].startswith("2026-11-26T06")
