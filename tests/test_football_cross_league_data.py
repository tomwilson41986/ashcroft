"""The cross-league data report: competition kinds and the links between pools."""

from __future__ import annotations

import pandas as pd

from research.open_close import football_cross_league_data as cl


def test_competition_kinds():
    assert cl.kind("UEFA Champions League", "World") == "uefa"
    assert cl.kind("UEFA Europa Conference League", "World") == "uefa"
    assert cl.kind("FA Cup", "England") == "cup" and cl.kind("Copa del Rey", "Spain") == "cup"
    assert cl.kind("DFB Pokal", "Germany") == "cup" and cl.kind("Premier League", "England") == "league"


def test_links_count_cup_ties_across_divisions_and_uefa_ties_across_countries():
    rows = [  # league matches place each club; a cup tie across divisions; a UEFA tie across countries
        (1, "Premier League", "England", 10, 11, "league"), (2, "Championship", "England", 20, 21, "league"),
        (3, "La Liga", "Spain", 30, 31, "league"), (4, "FA Cup", "England", 10, 20, "cup"),
        (5, "UEFA Champions League", "World", 11, 30, "uefa")]
    fx = pd.DataFrame([{"apif_fixture_id": i, "league": lg, "league_country": c, "apif_home_id": h,
                        "apif_away_id": a, "apif_season": 2025, "status": "FINISHED"} for i, lg, c, h, a, _ in rows])
    r = cl.fixtures_report(fx)
    assert r["links"]["cup"]["across_divisions_one_country"] == 1
    assert r["links"]["uefa"]["across_countries"] == 1
    assert r["by_kind"] == {"league": 3, "cup": 1, "uefa": 1}
