"""Cross-league strength: point in time, and strength carried across competitions only when crossing is on."""

from __future__ import annotations

import pandas as pd

from football import cross_league as cl


def _fx(rows):
    return pd.DataFrame([{"apif_fixture_id": i, "kickoff_utc": f"2025-0{1 + i // 28}-{1 + i % 28:02d}T15:00:00Z",
                          "status": "FINISHED", "league": lg, "league_country": c, "apif_season": 2025,
                          "apif_home_id": h, "apif_away_id": a, "ft_home": gh, "ft_away": ga}
                         for i, (lg, c, h, a, gh, ga) in enumerate(rows)])


def test_pre_match_ratings_never_see_their_own_or_later_results():
    rows = [("Premier League", "England", 1, 2, 3, 0), ("Premier League", "England", 2, 1, 0, 0),
            ("Premier League", "England", 1, 2, 1, 1)]
    f = cl.senior_club_matches(_fx(rows))
    a = cl.elo(f)
    rows[2] = ("Premier League", "England", 1, 2, 9, 0)                    # rewrite the last result
    b = cl.elo(cl.senior_club_matches(_fx(rows)))
    assert list(a["r_home"]) == list(b["r_home"]) and a["r_home"].iloc[0] == cl.START


def test_a_continental_win_carries_into_the_domestic_rating_only_when_crossing():
    rows = [("Premier League", "England", 1, 2, 0, 0), ("La Liga", "Spain", 3, 4, 0, 0),
            ("UEFA Champions League", "World", 1, 3, 4, 0),                 # England's club 1 beats Spain's 3
            ("Premier League", "England", 1, 2, 0, 0)]
    f = cl.senior_club_matches(_fx(rows))
    cross, apart = cl.elo(f, cross=True), cl.elo(f, cross=False)
    assert cross.iloc[3]["r_home"] > cross.iloc[0]["r_home"]              # the UEFA win counts at home
    assert abs(apart.iloc[3]["r_home"] - 1500.0) < 30                     # only the drawn league match moved it


def test_friendlies_youth_and_womens_matches_are_left_out():
    rows = [("Friendlies Clubs", "World", 1, 2, 5, 0), ("FA Women's Cup", "England", 1, 2, 5, 0),
            ("UEFA Youth League", "World", 1, 2, 5, 0), ("FA Cup", "England", 1, 2, 1, 0)]
    f = cl.senior_club_matches(_fx(rows))
    assert list(f["league"]) == ["FA Cup"]
