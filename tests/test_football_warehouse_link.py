"""The link stage: Betfair events to API-Football fixtures (exact, alias, fuzzy, the team map's second pass), the
settlements' veto, the team vote, canonical matches, the goal check (the clock across half time, the score, recall
and precision) and a report that fails soft and lists what to distrust (hand-written frames: a handful of matches
each)."""

from __future__ import annotations

import io
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from football.reference import normalise
from football.warehouse import gold_event, gold_team, link
from football.warehouse.schema import conform
from football.warehouse.store_io import part_key, read_manifest, read_single, write_part, write_single
from sources.common import Store


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def _events(rows) -> pd.DataFrame:
    """(event_id, home, away, kick-off, [home_id, away_id, result, country_code, over25])"""
    out = []
    for r in rows:
        r = tuple(r) + (None,) * (9 - len(r))
        out.append({"event_id": r[0], "home_name": r[1], "away_name": r[2], "kickoff": _ts(r[3]), "home_id": r[4],
                    "away_id": r[5], "result": r[6], "country_code": r[7], "over25": r[8]})
    return pd.DataFrame(out)


def _fixtures(rows) -> pd.DataFrame:
    """(fixture id, home, away, kick-off, home team id, away team id, [ft_home, ft_away, league id])"""
    out = []
    for r in rows:
        r = tuple(r) + (None,) * (9 - len(r))
        out.append({"apif_fixture_id": r[0], "home": r[1], "away": r[2], "kickoff_utc": _ts(r[3]),
                    "apif_home_id": r[4], "apif_away_id": r[5], "ft_home": r[6], "ft_away": r[7],
                    "apif_league_id": r[8] or 61, "apif_season": 2025, "league": "Ligue 1", "league_country": "France"})
    return pd.DataFrame(out)


def _row(links: pd.DataFrame, event_id: int) -> pd.Series:
    return links.set_index("event_id").loc[event_id]


def test_equal_names_link_and_the_row_says_how_far_apart_the_kick_offs_were():
    ev = _events([(1, "Lille", "Rennes", "2026-01-03 20:05", 44790, 44799, "A"),
                  (2, "Brest", "Auxerre", "2026-01-04 16:15", 669348, 44800, "H")])
    fx = _fixtures([(900, "Lille", "Rennes FC", "2026-01-03 20:00", 79, 94, 0, 2)])
    lk = link.link_events(ev, fx)
    assert lk["event_id"].tolist() == [1, 2] and lk["linked"].tolist() == [True, False]
    a = _row(lk, 1)
    assert (a["method"], a["score"], a["apif_fixture_id"], a["apif_league_id"], a["season"]) == ("exact", 1.0, 900, 61,
                                                                                                 2025)
    assert (a["apif_home_team_id"], a["apif_away_team_id"], a["apif_home_name"]) == (79, 94, "Lille")
    assert a["kickoff_diff_min"] == 5.0 and not a["swapped"] and a["result_agrees"] and pd.isna(a["over25_agrees"])
    assert (a["apif_ft_home"], a["apif_ft_away"], a["apif_kickoff"]) == (0, 2, _ts("2026-01-03 20:00"))
    b = _row(lk, 2)
    assert b["method"] == "none" and pd.isna(b["apif_fixture_id"]) and pd.isna(b["score"]) and pd.isna(b["swapped"]) \
        is False and pd.isna(b["result_agrees"]) and pd.isna(b["over25_agrees"])


def test_swapped_sides_link_and_the_ids_and_score_follow_the_events_sides():
    ev = _events([(1, "Rennes", "Lille", "2026-01-03 20:00", 44799, 44790, "H")])
    fx = _fixtures([(900, "Lille", "Rennes", "2026-01-03 20:00", 79, 94, 0, 2)])
    a = _row(link.link_events(ev, fx), 1)
    assert a["method"] == "exact" and a["swapped"]
    # the event's home side is Rennes: API-Football's 94, who scored the two
    assert (a["apif_home_team_id"], a["apif_away_team_id"], a["apif_home_name"]) == (94, 79, "Rennes")
    assert (a["apif_ft_home"], a["apif_ft_away"]) == (2, 0) and a["result_agrees"]


def test_a_kick_off_outside_the_tolerance_does_not_link():
    fx = _fixtures([(900, "Lille", "Rennes", "2026-01-03 20:00", 79, 94),
                    (901, "Paris Saint Germain", "Le Havre", "2026-01-03 20:00", 85, 111)])
    ev = _events([(1, "Lille", "Rennes", "2026-01-03 23:01"),          # equal names: 181 minutes is too far
                  (2, "Paris St-G", "Le Havre", "2026-01-03 20:31")])  # similar names: 31 minutes is too far
    assert link.link_events(ev, fx)["linked"].tolist() == [False, False]
    ev = _events([(1, "Lille", "Rennes", "2026-01-03 22:59"), (2, "Paris St-G", "Le Havre", "2026-01-03 20:29")])
    lk = link.link_events(ev, fx)
    assert lk["method"].tolist() == ["exact", "fuzzy"] and lk["kickoff_diff_min"].tolist() == [179.0, 29.0]


def test_two_events_competing_for_one_fixture_the_better_one_gets_it():
    fx = _fixtures([(900, "Paris Saint Germain", "Le Havre", "2026-01-03 20:00", 85, 111)])
    ev = _events([(1, "Paris St-G", "Le Havre", "2026-01-03 20:00"),          # similar: 0.9
                  (2, "Paris Saint Germain", "Le Havre", "2026-01-03 20:10"),  # equal, ten minutes off
                  (3, "Paris Saint Germain", "Le Havre", "2026-01-03 20:02")])  # equal, two minutes off: the nearest
    lk = link.link_events(ev, fx)
    assert lk["method"].tolist() == ["none", "none", "exact"]
    assert lk["apif_fixture_id"].notna().sum() == 1 and lk["apif_fixture_id"].dropna().is_unique
    # a match Betfair listed twice (the first listing void, the second settled): the settled one gets the fixture
    scored = _fixtures([(900, "Paris Saint Germain", "Le Havre", "2026-01-03 20:00", 85, 111, 2, 0)])
    twice = _events([(1, "Paris Saint Germain", "Le Havre", "2026-01-03 20:00"),
                     (2, "Paris Saint Germain", "Le Havre", "2026-01-03 20:00", 5, 6, "H")])
    assert link.link_events(twice, scored)["method"].tolist() == ["none", "exact"]
    assert link.link_events(twice.assign(result=None), scored)["method"].tolist() == ["exact", "none"]


def test_a_reviewed_alias_counts_as_the_name():
    ev = _events([(1, "AGF", "Brondby", "2026-03-01 15:00", 1, 2, None, "DK")])
    fx = _fixtures([(900, "Aarhus", "Brondby", "2026-03-01 15:00", 406, 407)])
    assert link.link_events(ev, fx)["method"].tolist() == ["none"]
    aliases = {("betfair", "DEN", normalise("AGF")): "DEN:aarhus"}
    a = _row(link.link_events(ev, fx, aliases), 1)
    assert (a["method"], a["score"], a["apif_home_team_id"]) == ("alias", 1.0, 406)
    # an alias is a country's: the same name on an event of another country is not Aarhus
    ev["country_code"] = "SE"
    assert link.link_events(ev, fx, aliases)["method"].tolist() == ["none"]


def test_fuzzy_names_link_only_within_a_squad_and_only_when_unique():
    ko = "2026-02-07 15:00"
    fx = _fixtures([(900, "Paris Saint Germain", "Notts County", ko, 85, 1356),
                    (901, "Wigan", "Wycombe", ko, 60, 1357),
                    (902, "Real Sociedad", "Celta Vigo", ko, 548, 538),
                    (903, "Real Sociedad II", "Celta Vigo II", ko, 9580, 9581),
                    (904, "Cagliari", "Lecce", ko, 490, 867)])
    ev = _events([(1, "Paris St-G", "Notts Co", ko),                # both a prefix of the fixture's names
                  (2, "Arsenal (W)", "Man Utd (W)", ko),             # a lone W is not Wigan, nor Wycombe
                  (3, "Sociedad B", "Celta Vigo B", ko),             # the B sides, not the first teams
                  (4, "A.C. Trento S.C.S.D.", "Lecco", ko)])         # lone letters are not Cagliari; Lecco is not Lecce
    lk = link.link_events(ev, fx)
    assert lk["method"].tolist() == ["fuzzy", "none", "fuzzy", "none"]
    assert lk["apif_fixture_id"].tolist()[0] == 900 and lk["apif_fixture_id"].tolist()[2] == 903
    assert lk["score"].tolist()[0] == 0.9 and lk["score"].tolist()[2] == 0.95
    # two fixtures that fit equally well: unique or nothing
    twins = _fixtures([(910, "Paris Saint Germain", "Notts County", ko, 85, 1356),
                       (911, "Paris Saint Germain", "Notts County FC", ko, 85, 1356)])
    assert link.link_events(ev.head(1), twins)["method"].tolist() == ["none"]
    # a fuzzy link whose result disagrees with the fixture's score is dropped; an agreeing one stays
    scored = _fixtures([(900, "Paris Saint Germain", "Notts County", ko, 85, 1356, 3, 0)])
    lost, won = (_events([(1, "Paris St-G", "Notts Co", ko, 5, 6, r)]) for r in ("A", "H"))
    stats: dict = {}
    assert link.link_events(lost, scored, stats=stats)["linked"].tolist() == [False]
    assert stats["vetoed"] == {"fuzzy": 1}
    assert link.link_events(won, scored, stats=stats)["method"].tolist() == ["fuzzy"] and stats["vetoed"] == {}
    # and so is one whose over/under 2.5 disagrees with the score's three goals, the result agreeing
    under, over = (_events([(1, "Paris St-G", "Notts Co", ko, 5, 6, "H", None, o)]) for o in (0, 1))
    assert link.link_events(under, scored, stats=stats)["linked"].tolist() == [False]
    assert stats["vetoed"] == {"fuzzy": 1}
    a = _row(link.link_events(over, scored), 1)
    assert a["method"] == "fuzzy" and a["result_agrees"] and a["over25_agrees"]
    # equal names keep their link whatever the settlements say: the row is flagged for review
    same = _events([(1, "Paris Saint Germain", "Notts County", ko, 5, 6, "A", None, 0)])
    a = _row(link.link_events(same, scored, stats=stats), 1)
    assert a["method"] == "exact" and bool(a["result_agrees"]) is False and bool(a["over25_agrees"]) is False
    assert stats["vetoed"] == {}


def test_a_name_of_initials_resembles_nothing_but_the_same_letters():
    # measured on 2026: 'A.E.L. v Aris FC Limassol' fuzzy-linked 'Atromitos v Aris', sides swapped (a lone letter
    # stood for any word that starts with it, and nothing was left to compare once the lone letters were dropped)
    ael, atromitos = link.name_key("A.E.L."), link.name_key("Atromitos")
    assert link.side_score(ael, atromitos) == link.side_score(atromitos, ael) == 0.0
    assert link.side_score(ael, link.name_key("AEL")) == 1.0
    assert link.side_score(ael, link.name_key("AEL Limassol")) == 0.0
    assert link.side_score(link.name_key("B 93"), link.name_key("B93")) == 1.0
    ko = "2026-10-03 16:00"
    ev = _events([(1, "A.E.L.", "Aris FC Limassol", ko), (2, "A.E.L.", "Omonia", ko)])
    fx = _fixtures([(900, "Aris", "Atromitos", "2026-10-03 15:30", 1, 2), (901, "AEL", "Omonia Nicosia", ko, 3, 4)])
    lk = link.link_events(ev, fx)
    assert lk["method"].tolist() == ["none", "fuzzy"] and lk["apif_fixture_id"].tolist()[1] == 901


def test_name_keys_and_side_scores():
    assert link.name_key("Arsenal (W)") == link.name_key("Arsenal Women") == "arsenal|w"
    assert link.name_key("Real Sociedad II") == "real sociedad|b" and link.name_key("Jong Ajax") == "ajax|b"
    assert link.name_key("England U-21") == link.name_key("England U21") == "england|u21"
    assert link.name_key("AIK") == "aik|" and link.name_key(None) == "|" and link.name_key("Mainz 05") == "mainz 05|"
    assert link.side_score("arsenal|w", "arsenal|") == 0.0 and link.side_score("arsenal|w", "arsenal|w") == 1.0
    assert link.side_score("|", "|") == 0.0
    # the floor only spares work: a score that reaches it is the same with or without it
    a, b = link.name_key("Paris St-G"), link.name_key("Paris Saint Germain")
    assert link.side_score(a, b) == link.side_score(a, b, 0.85) == 0.9
    assert link.side_score(link.name_key("OB"), link.name_key("Odense"), 0.85) == 0.0


def _danish():
    """Brondby and Midtjylland link by name three times; AGF and OB are 'Aarhus' and 'Odense' to API-Football."""
    ev = _events([(1, "Brondby", "Midtjylland", "2026-03-01 15:00", 11, 12, "H"),
                  (2, "Midtjylland", "Brondby", "2026-03-08 15:00", 12, 11, "D"),
                  (3, "Brondby", "Midtjylland", "2026-03-15 15:00", 11, 12, "A"),
                  (4, "AGF", "Brondby", "2026-03-22 15:00", 13, 11, "H"),
                  (5, "Midtjylland", "OB", "2026-03-22 17:00", 12, 14, "A"),
                  (6, "AGF", "OB", "2026-03-29 15:00", 13, 14, "D"),
                  (7, "OB", "AGF", "2026-04-05 15:00", 14, 13, "H")])
    fx = _fixtures([(901, "Brondby", "Midtjylland", "2026-03-01 15:00", 407, 397, 2, 0),
                    (902, "Midtjylland", "Brondby", "2026-03-08 15:00", 397, 407, 1, 1),
                    (903, "Brondby", "Midtjylland", "2026-03-15 15:00", 407, 397, 0, 1),
                    (904, "Aarhus", "Brondby", "2026-03-22 15:00", 406, 407, 3, 1),
                    (905, "Midtjylland", "Odense", "2026-03-22 17:00", 397, 405, 0, 2),
                    (906, "Aarhus", "Odense", "2026-03-29 15:00", 406, 405, 2, 2),
                    (907, "Odense", "Aarhus", "2026-04-05 15:00", 405, 406, 0, 1)])   # an away win: Betfair says home
    return ev, fx


def test_the_second_pass_links_through_the_team_map_what_the_names_could_not():
    ev, fx = _danish()
    names_only = link.link_events(ev, fx, team_pass=False)
    assert names_only["method"].tolist() == ["exact", "exact", "exact", "none", "none", "none", "none"]
    lk = link.link_events(ev, fx)
    # 4 and 5: one side mapped (three matches behind it), the team's only fixture at that kick-off; 6: both sides
    # mapped by then; 7: both mapped, but the fixture's score is not the result Betfair settled
    stats: dict = {}
    assert link.link_events(ev, fx, stats=stats)["method"].tolist() == lk["method"].tolist() == [
        "exact", "exact", "exact", "team_anchor", "team_anchor", "team_map", "none"]
    assert stats["vetoed"] == {"team_map": 1}                      # counted once, though every round meets it
    assert lk["apif_fixture_id"].tolist()[:6] == [901, 902, 903, 904, 905, 906]
    assert lk["score"].tolist()[3:6] == [1.0] * 3
    assert _row(lk, 4)["apif_home_team_id"] == 406 and _row(lk, 5)["apif_away_team_id"] == 405
    tm = link.team_map(lk, ev).set_index("team_id")
    assert tm["apif_team_id"].to_dict() == {11: 407, 12: 397, 13: 406, 14: 405}
    assert tm["n_linked"].to_dict() == {11: 4, 12: 4, 13: 2, 14: 2} and not tm["low_agreement"].any()
    # a map with fewer matches behind it than the anchor needs places nothing on its own
    assert link.link_events(ev, fx, anchor_min_agree=5)["linked"].sum() == 3
    # the other side mapped to another team than the fixture's, and its name nothing like it: no anchor (Betfair
    # gives one id to two clubs of a name); a name that resembles the fixture's side is enough
    ev2 = pd.concat([ev.head(5), _events([(8, "Brondby", "Midtjylland", "2026-04-12 15:00", 11, 12),
                                          (9, "Midtjylland", "Brondby", "2026-04-19 15:00", 12, 11)])],
                    ignore_index=True)
    fx2 = pd.concat([fx.head(5), _fixtures([(908, "Brondby", "Aalborg", "2026-04-12 15:00", 407, 400),
                                            (909, "Midtjylland", "Brondby Reserves", "2026-04-19 15:00", 397, 9407),
                                            (910, "Midtjylland", "Brondby IF Kobenhavn", "2026-04-19 15:20", 397, 8407)
                                            ])], ignore_index=True)
    lk2 = link.link_events(ev2, fx2)
    assert _row(lk2, 8)["method"] == "none"
    assert (_row(lk2, 9)["method"], _row(lk2, 9)["apif_fixture_id"]) == ("team_anchor", 910)   # not the reserves


def test_the_team_map_is_a_majority_vote_with_a_flag_when_the_matches_disagree():
    events = pd.DataFrame({"event_id": [1, 2, 3, 4, 5], "home_id": [7, 7, 8, 7, 8], "away_id": [8, 8, 7, 8, 9]})
    links = pd.DataFrame({"event_id": [1, 2, 3, 4, 5], "apif_fixture_id": [901, 902, 903, 904, None],
                          "apif_home_team_id": [70, 70, 80, 71, None], "apif_away_team_id": [80, 80, 70, 80, None]})
    tm = link.team_map(links, events).set_index("team_id")
    assert tm.index.tolist() == [7, 8]                              # team 9 has no linked match
    assert tm.loc[7].to_dict() == {"apif_team_id": 70, "n_linked": 4, "n_agree": 3, "share": 0.75, "n_apif_ids": 2,
                                   "low_agreement": True}
    assert tm.loc[8].to_dict() == {"apif_team_id": 80, "n_linked": 4, "n_agree": 4, "share": 1.0, "n_apif_ids": 1,
                                   "low_agreement": False}
    assert not link.team_map(links, events, min_share=0.7)["low_agreement"].any()
    # the team table: every Betfair team id, the unmapped ones blank
    teams = pd.DataFrame({"team_id": [7, 8, 9], "team_name": ["Seven", "Eight", "Nine"]})
    lk = links.assign(apif_home_name=["A", "A", "B", "A2", None], apif_away_name=["B", "B", "A", "B", None])
    tt = link.team_table(link.team_map(links, events), teams, lk, events).set_index("team_id")
    assert tt["apif_team_name"].tolist() == ["A", "B", None] and tt["n_linked"].tolist() == [4, 4, 0]
    assert tt["low_agreement"].tolist() == [True, False, False] and pd.isna(tt.loc[9, "apif_team_id"])
    assert link.to_arrow(link.TEAM_LINK, tt.reset_index()).column("apif_team_id").null_count == 1


def test_canonical_matches_come_through_the_fixture_id_or_the_rule_that_writes_it():
    ev = _events([(1, "Lille", "Rennes", "2026-01-03 20:00"), (2, "Leeds", "Southampton", "2026-05-24 14:00"),
                  (3, "Flamengo", "Santos", "2026-05-24 20:00"), (4, "Nobody", "Noone", "2026-05-24 20:00")])
    fx = _fixtures([(900, "Lille", "Rennes", "2026-01-03 20:00", 79, 94, 1, 0, 61),
                    (901, "Leeds", "Southampton", "2026-05-24 14:00", 63, 41, 1, 0, 40),     # a play-off: not canonical
                    (902, "Flamengo", "Santos", "2026-05-24 20:00", 127, 128, 1, 0, 9999)])  # a league we do not hold
    lk = link.link_events(ev, fx)
    cm = pd.DataFrame({"match_id": ["aaaa", "bbbb"], "competition_id": ["FRA1", "FRA1"],
                       "match_date": ["2026-01-03", "2026-01-03"], "home_team": ["Lille", "Lyon"],
                       "away_team": ["Rennes", "Nice"], "home_team_id": ["FRA:lille", "FRA:lyon"],
                       "away_team_id": ["FRA:rennes", "FRA:nice"], "apif_fixture_id": [900.0, None]})
    out = link.canonical(lk, cm)
    assert out["match_id"].tolist() == ["aaaa", None, None, None]
    assert out["competition_id"].tolist() == ["FRA1", "ENG2", None, None]
    # a canonical table built without API-Football: the same pairing rule finds the match from the fixture
    bare = cm.drop(columns=["apif_fixture_id"])
    assert link.canonical(lk, bare)["match_id"].isna().all()
    again = link.canonical(lk, bare, fx)
    assert again["match_id"].tolist() == ["aaaa", None, None, None] and again["competition_id"].tolist()[0] == "FRA1"
    assert link.canonical(lk, None)["competition_id"].tolist() == ["FRA1", "ENG2", None, None]
    # a canonical table without the day cannot be paired by the rule: the league's competition, no match
    assert link.canonical(lk, bare.drop(columns=["match_date"]), fx)["match_id"].isna().all()
    # the canonical team id by the same vote
    ev["home_id"], ev["away_id"] = [5, 6, 7, 8], [15, 16, 17, 18]
    tt = link.team_table(link.team_map(out, ev), None, out, ev, cm).set_index("team_id")
    assert tt.loc[5, "canonical_team_id"] == "FRA:lille" and tt.loc[15, "canonical_team_id"] == "FRA:rennes"
    assert tt.loc[5, "canonical_share"] == 1.0 and tt.loc[6, "canonical_team_id"] is None


def test_a_canonical_match_the_names_miss_is_found_through_the_teams_vote():
    # football-data calls them 'Paris SG': no name pairs the fixture with the canonical match, the teams do
    ev = _events([(1, "Lille", "Rennes", "2026-01-03 20:00", 5, 15), (2, "Rennes", "Lille", "2026-01-10 20:00", 15, 5),
                  (3, "Lille", "Lyon", "2026-01-24 20:00", 5, 25),
                  (4, "Paris St-G", "Lille", "2026-01-17 20:00", 20, 5),
                  (5, "Paris St-G", "Lyon", "2026-01-31 20:00", 20, 25)])
    fx = _fixtures([(900, "Lille", "Rennes", "2026-01-03 20:00", 79, 94),
                    (901, "Rennes", "Lille", "2026-01-10 20:00", 94, 79),
                    (902, "Lille", "Lyon", "2026-01-24 20:00", 79, 80),
                    (903, "Paris St-G", "Lille", "2026-01-17 20:00", 85, 79),
                    (904, "Paris St-G", "Lyon", "2026-01-31 20:00", 85, 80)])
    days = ["2026-01-03", "2026-01-10", "2026-01-24", "2026-01-17", "2026-01-31", "2026-01-17"]
    cm = pd.DataFrame({"match_id": ["m1", "m2", "m3", "m4", "m5", "m6"], "competition_id": ["FRA1"] * 6,
                       "match_date": days, "home_team": ["Lille", "Rennes", "Lille", "Paris SG", "Paris SG", "Nice"],
                       "away_team": ["Rennes", "Lille", "Lyon", "Lille", "Lyon", "Metz"],
                       "apif_fixture_id": [900, None, 902, None, None, None]})
    cm["home_team_id"] = "FRA:" + cm["home_team"].str.lower()
    cm["away_team_id"] = "FRA:" + cm["away_team"].str.lower()
    lk = link.link_events(ev, fx)
    assert link.canonical(lk, cm)["match_id"].tolist() == ["m1", None, "m3", None, None]
    # round 1: Rennes v Lille, both sides voted for (one match each is enough when both agree). Round 2: Paris v
    # Lille, on Lille alone, now three matches strong. Round 3: Paris v Lyon, both voted for by then
    out = link.canonical(lk, cm, events=ev)
    assert out["match_id"].tolist() == ["m1", "m2", "m3", "m4", "m5"] and out["competition_id"].tolist() == ["FRA1"] * 5
    tt = link.team_table(link.team_map(out, ev), None, out, ev, cm).set_index("team_id")
    assert tt["canonical_team_id"].to_dict() == {5: "FRA:lille", 15: "FRA:rennes", 20: "FRA:paris sg", 25: "FRA:lyon"}


def _goal_inputs():
    links = pd.DataFrame({
        "event_id": [1, 2, 3, 4, 5, 6], "apif_fixture_id": [901, 902, 903, 904, 905, None],
        "apif_home_team_id": [10, 10, 10, 10, 10, None], "apif_away_team_id": [20, 20, 20, 20, 20, None],
        "apif_kickoff": [_ts("2026-03-01 15:00")] * 6,
        "kickoff_actual": [_ts("2026-03-01 15:01"), _ts("2026-03-01 15:01"), None, _ts("2026-03-01 15:00"), None, None],
        "apif_ft_home": [3, 1, 1, 1, None, None], "apif_ft_away": [1, 1, 0, 0, None, None],
        "went_in_play": [True] * 6})
    goals = pd.DataFrame(
        [(1, 1, "2026-03-01 15:11:30", "home"),    # 10.5 minutes in: minute 11
         (1, 2, "2026-03-01 15:47:40", "away"),    # 46.7 in, first-half stoppage: API-Football's 45+2 = 47
         (1, 3, "2026-03-01 16:08:30", "home"),    # 67.5 in: 18 minutes of break and stoppage make it minute 50
         (1, 4, "2026-03-01 16:51:00", "home"),    # 110 in: 90+2 = 92
         (2, 1, "2026-03-01 15:31:00", "home"),    # 30 in; API-Football has goals in minutes 30 and 80
         (3, 1, "2026-03-01 15:13:00", "H"),       # no actual kick-off: 13 after the scheduled one, minute 11
         (4, 1, "2026-03-01 15:20:00", "away")],   # the right minute, the wrong side
        columns=["event_id", "goal_no", "pt", "side"])
    goals["pt"] = pd.to_datetime(goals["pt"], utc=True)
    goals["elapsed_min"] = None
    goals["match_min"] = [10.5, 45.0, 50.5, 90.0, 30.0, None, 20.5]    # gold/goal's football minute (45 and 90 at most)
    apif = pd.DataFrame(
        # the own goal is listed under team 10, the side it counts for (as API-Football lists them), and the home
        # side's third goal it is; the missed penalty is no goal
        [(901, 11, "Goal", "Normal Goal", 10), (901, 47, "Goal", "Penalty", 20), (901, 50, "Goal", "Own Goal", 10),
         (901, 92, "Goal", "Normal Goal", 10), (901, 60, "Goal", "Missed Penalty", 20),
         (901, 70, "Card", "Red Card", 20),
         (902, 30, "Goal", "Normal Goal", 10), (902, 80, "Goal", "Normal Goal", 20),
         (903, 11, "Goal", "Normal Goal", 10),
         (904, 20, "Goal", "Normal Goal", 10), (904, 105, "Goal", "Normal Goal", 20),   # extra time: not Betfair's
         (905, 33, "subst", "Substitution 1", 10)],
        columns=["apif_fixture_id", "minute", "type", "detail", "apif_team_id"])
    return goals, links, apif


def test_the_goal_check_lines_the_wall_clock_up_with_the_match_clock_across_half_time():
    goals, links, apif = _goal_inputs()
    per_event, per_goal = link.goal_check(goals, links, apif)
    e = per_event.set_index("event_id")
    assert e.index.tolist() == [1, 2, 3, 4, 5]                    # 6 is not linked
    a = e.loc[1]
    assert (a["clock"], a["n_inferred"], a["n_apif"], a["n_ft"], a["count_diff"]) == ("actual", 4, 4, 4, 0)
    assert a["count_agrees"] and a["score_agrees"] and pd.isna(a["usable"])     # 3-1, the own goal the home side's
    assert (a["n_matched"], a["n_within_tol"], a["n_side_known"], a["n_side_agree"]) == (4, 4, 4, 4)
    g = per_goal[per_goal["event_id"] == 1].set_index("goal_no")
    assert g["apif_minute"].tolist() == [11, 47, 50, 92] and g["half"].tolist() == [1, 1, 2, 2]
    assert g["lag_min"].round(3).tolist() == [0.0, 0.167, 18.0, 18.5]
    assert g["minute_err"].round(3).tolist() == [0.0, 0.167, 0.0, 0.0]     # the second half's lag is allowed for
    assert a["ht_lag_min"] == 18.25 and round(a["minute_mae"], 3) == 0.042
    # gold/goal's match_min against the minute's middle, the minute capped as match_min is: 10.5, 44.5, 49.5, 89.5
    assert g["match_min_err"].tolist() == [0.0, 0.5, 1.0, 0.5] and a["match_min_mae"] == 0.5
    # one goal inferred of two: the count is short, the one pairs with the 30th-minute goal
    b = e.loc[2]
    assert (b["n_inferred"], b["count_diff"], b["count_agrees"], b["n_matched"]) == (1, -1, False, 1)
    assert bool(b["score_agrees"]) is False
    assert per_goal[per_goal["event_id"] == 2]["apif_minute"].tolist() == [30]
    # the scheduled kick-off only: a start up to five minutes late is allowed for; no match_min, no error of it
    c = e.loc[3]
    assert (c["clock"], c["n_matched"], c["minute_mae"], c["n_side_agree"]) == ("fixture", 1, 0.0, 1)
    assert c["score_agrees"] and pd.isna(c["match_min_mae"])
    # extra-time goals are not counted; a pair may disagree on the side, and then the score is not the true one
    d = e.loc[4]
    assert (d["n_apif"], d["n_ft"], d["count_agrees"], d["n_matched"], d["n_side_known"], d["n_side_agree"]) == \
        (1, 1, True, 1, 1, 0)
    assert bool(d["score_agrees"]) is False
    # events held and no goal: a goalless match the prices agree with
    f = e.loc[5]
    assert (f["has_events"], f["n_inferred"], f["n_apif"], f["count_agrees"]) == (True, 0, 0, True)
    assert pd.isna(f["n_ft"]) and f["score_agrees"]
    s = link.goal_summary(per_event, per_goal)
    assert (s["matches_checked"], s["count_agrees"], s["goals_matched"], s["goal_events"]) == (5, 4, 7, 8)
    assert (s["score_known"], s["score_agrees"], s["score_agrees_share"]) == (5, 3, 0.6)
    # seven of the eight real goals have an inferred goal within two minutes; every inferred goal is a real one
    assert (s["goals_inferred"], s["goals_within_tol"], s["recall_within_tol"], s["precision_within_tol"]) == \
        (7, 7, 0.875, 1.0)
    assert s["tolerance_min"] == 2.0 and s["side_agrees_share"] == round(6 / 7, 4) and "usable" not in s
    assert (s["match_min_abs_err_median"], s["match_min_within_tol_share"]) == (0.5, 1.0)
    # a side an inferred goal does not name: the count agrees, the score cannot be said to
    blank = goals.assign(side=goals["side"].where(goals["event_id"] != 3, ""))
    assert pd.isna(link.goal_check(blank, links, apif)[0].set_index("event_id").loc[3, "score_agrees"])
    # without gold/goal's match_min and side columns the check still runs
    bare, _ = link.goal_check(goals[["event_id", "goal_no", "pt"]], links, apif)
    assert bare["n_within_tol"].tolist() == per_event["n_within_tol"].tolist() and bare["match_min_mae"].isna().all()


def test_the_goal_check_keeps_to_the_matches_the_timeline_holds_and_carries_usable():
    goals, links, apif = _goal_inputs()
    timeline = pd.DataFrame({"event_id": [1, 2, 3, 99], "usable": [True, False, True, True]})
    per_event, per_goal = link.goal_check(goals, links, apif, timeline)
    # 4 and 5 went in play and are linked, but gold/goal never looked at them: not goalless, not checked
    assert per_event["event_id"].tolist() == [1, 2, 3] and per_event["usable"].tolist() == [True, False, True]
    assert sorted(set(per_goal["event_id"])) == [1, 2, 3]
    s = link.goal_summary(per_event, per_goal)
    assert (s["matches_checked"], s["goal_events"], s["goals_within_tol"], s["recall_within_tol"]) == (3, 7, 6, 0.8571)
    u = s["usable"]                                                # the match with half its goals missed is not usable
    assert (u["matches_checked"], u["count_agrees"], u["score_agrees"], u["goal_events"], u["goals_matched"]) == \
        (2, 2, 2, 5, 5)
    assert (u["recall_within_tol"], u["precision_within_tol"]) == (1.0, 1.0)
    t = link.to_arrow(link.GOAL_CHECK, per_event)
    assert t.column("usable").to_pylist() == [True, False, True] and t.column("score_agrees").null_count == 0
    assert link.goal_summary(per_event.head(0), per_goal.head(0)) == {"matches_checked": 0}


def test_a_goal_too_far_from_any_goal_event_is_left_unmatched():
    goals, links, apif = _goal_inputs()
    goals = goals[goals["event_id"] == 2].assign(pt=_ts("2026-03-01 15:56:00"))   # 55 minutes in: neither 30 nor 80
    per_event, per_goal = link.goal_check(goals, links[links["event_id"] == 2], apif)
    assert per_event["n_matched"].tolist() == [0] and per_goal.empty
    # a first-half goal read off the second half's clock would be 18 minutes out: the halves are not confused
    assert link._minute_err(67.5, 50, 0.0) == (0.0, 18.0, 2) and link._minute_err(47.0, 47, 0.0)[2] == 1
    assert link._minute_err(80.0, 50, 0.0)[0] == 5.5 and link._minute_err(40.0, 30, 0.0)[0] == 10.5


def _wide(spec, rows) -> pa.Table:
    """Rows of a few columns as a table of the spec (the other columns blank)."""
    df = pd.DataFrame(rows).reindex(columns=spec.names)
    return link.to_arrow(spec, df)


def _warehouse(tmp_path) -> Store:
    store = Store(root=tmp_path)
    ev, _ = _danish()
    ev = ev.assign(event_name=ev["home_name"] + " v " + ev["away_name"], n_markets=1, went_in_play=True, void=False,
                   country_code="DK", kickoff_actual=ev["kickoff"], category="senior")
    write_part(store, gold_event.SPEC, 2026, 3, _wide(gold_event.SPEC, ev[ev["event_id"] < 7]))
    write_part(store, gold_event.SPEC, 2026, 4, _wide(gold_event.SPEC, ev[ev["event_id"] == 7]))
    teams = pd.DataFrame({"team_id": [11, 12, 13, 14, 15],
                          "team_name": ["Brondby", "Midtjylland", "AGF", "OB", "Vejle"],
                          "n_names": 1, "n_events": 1, "n_home": 1, "n_played": 1})
    write_single(store, gold_team.SPEC, _wide(gold_team.SPEC, teams))
    return store


def test_the_run_fails_soft_naming_each_input_it_did_not_find(tmp_path, capsys):
    empty = link.run(Store(root=tmp_path / "nothing"))
    assert empty["missing"] == ["no data found: football/warehouse/gold/event"] and empty["written"] == {}
    store = _warehouse(tmp_path)
    rep = link.run(store)
    assert rep["missing"] == ["no data found: api_football/fixtures.parquet",
                              "no data found: api_football/history_matches.parquet"]
    assert rep["status"] == "no fixtures: nothing linked" and rep["written"] == {}
    assert read_single(store, "gold/link") is None

    _, fx = _danish()
    store.put_parquet(link.FIXTURES_KEY, fx.assign(apif_league_id=119))
    assert link.main(["run", "--root", str(tmp_path), "--dry-run"]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["status"] == "ok" and dry["written"] == {} and read_single(store, "gold/link") is None
    assert dry["missing"] == ["no data found: api_football/history_matches.parquet",
                              "no data found: football/gold/matches.parquet",
                              "no data found: football/warehouse/gold/goal"]

    rep = link.run(store)
    assert rep["events"] == 7 and rep["linked"] == {"n": 6, "share": round(6 / 7, 4)}
    assert rep["by_method"] == {"exact": 3, "team_map": 1, "team_anchor": 2, "none": 1}
    assert rep["by_month"] == {"2026-03": {"events": 6, "linked": 6, "share": 1.0},
                               "2026-04": {"events": 1, "linked": 0, "share": 0.0}}
    assert rep["by_country"] == {"DK": {"events": 7, "linked": 6, "share": round(6 / 7, 4)}}
    assert rep["unlinked_top_countries"] == {"DK": 1} and rep["result_check"] == {"known": 6, "agree": 6}
    # how far to trust it: the settlements by method, what they vetoed, the weakest links with the names side by side
    assert rep["over25_check"] == {"known": 0, "agree": 0} and rep["disagreements"] == []
    assert rep["vetoed"] == {"team_map": 1}                        # OB v AGF: the fixture's score is an away win
    assert rep["checks_by_method"]["team_anchor"] == {"linked": 2, "result": {"known": 2, "agree": 2},
                                                      "over25": {"known": 0, "agree": 0}, "min_score": 1.0,
                                                      "max_kickoff_gap_min": 0.0}
    assert sorted(rep["weakest_links"]) == ["exact", "team_anchor", "team_map"]
    w = rep["weakest_links"]["team_anchor"]
    assert [(x["event_id"], x["betfair"], x["api_football"]) for x in w] == [
        (4, "AGF v Brondby", "Aarhus v Brondby"), (5, "Midtjylland v OB", "Midtjylland v Odense")]   # least alike first
    assert w[1]["name_similarity"] == [1.0, 0.25] and w[1]["apif_score"] == "0-2" and w[1]["result_agrees"] is True
    assert w[1]["over25_agrees"] is None and w[0]["name_similarity"][0] < 0.25
    assert rep["teams"]["several_apif_ids"] == 0
    assert rep["teams"]["team_ids"] == 5 and rep["teams"]["mapped"] == 4 and rep["teams"]["low_agreement"] == 0
    assert rep["written"] == {"gold/link": 7, "gold/team_link": 5} and "goal_check" not in rep
    lk = read_single(store, "gold/link").to_pandas()
    assert lk["event_id"].is_unique and lk["competition_id"].tolist()[:6] == ["DEN1"] * 6
    assert pd.isna(lk["competition_id"].tolist()[6])
    assert lk["match_id"].isna().all() and lk["apif_fixture_id"].isna().sum() == 1
    tl = read_single(store, "gold/team_link").to_pandas().set_index("team_id")
    assert tl["apif_team_id"].dropna().astype(int).to_dict() == {11: 407, 12: 397, 13: 406, 14: 405}
    assert tl.loc[13, "apif_team_name"] == "Aarhus" and tl.loc[15, "n_linked"] == 0
    assert read_manifest(store, "gold/link")["parts"]["all"]["rows"] == 7

    # with gold/goal and API-Football's events the goal check is written and summarised
    goals = pd.DataFrame({"event_id": [1, 1], "goal_no": [1, 2], "side": ["home", "home"], "elapsed_min": [20.0, 70.0],
                          "pt": pd.to_datetime(["2026-03-01 15:20:10", "2026-03-01 16:28:00"], utc=True)})

    def put_part(table: str, df: pd.DataFrame) -> None:
        buf = io.BytesIO()
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf)
        store.put(part_key(table, 2026, 3), buf.getvalue())

    put_part("gold/goal", goals)
    store.put_parquet(link.HISTORY_EVENTS_KEY, pd.DataFrame(
        {"apif_fixture_id": [901, 901, 555], "minute": [21, 71, 9], "type": ["Goal"] * 3, "detail": ["Normal Goal"] * 3,
         "apif_team_id": [407, 407, 1]}))
    rep = link.run(store)
    assert rep["missing"] == ["no data found: api_football/history_matches.parquet",
                              "no data found: football/gold/matches.parquet",
                              "no data found: football/warehouse/gold/timeline"]
    assert rep["written"]["gold/goal_check"] == 6                  # six linked matches with a score to count against
    # the five matches with no goal inferred have goals in their scores: only the first agrees on the count
    assert rep["goal_check"]["count_agrees"] == 1 and rep["goal_check"]["goals_matched"] == 2
    assert rep["goal_check"]["score_agrees"] == 1 and rep["goal_check"]["recall_within_tol"] == 1.0
    gc = read_single(store, "gold/goal_check").to_pandas().set_index("event_id")
    assert gc.loc[1, "n_matched"] == 2 and gc.loc[1, "ht_lag_min"] == 17.5 and gc.loc[1, "minute_mae"] < 0.2
    assert gc["usable"].isna().all() and gc.loc[1, "score_agrees"]

    # with gold/timeline only the matches it holds are checked, and its usable flag is carried
    put_part("gold/timeline", pd.DataFrame({"event_id": [1, 2], "usable": [True, False]}))
    rep = link.run(store)
    assert "no data found: football/warehouse/gold/timeline" not in rep["missing"]
    assert rep["written"]["gold/goal_check"] == 2 and rep["goal_check"]["usable"]["matches_checked"] == 1
    assert rep["goal_check"]["usable"]["count_agrees_share"] == 1.0
    assert read_single(store, "gold/goal_check").to_pandas()["usable"].tolist() == [True, False]

    # a gold/goal without the columns the check needs is named, and nothing is checked or written for it
    put_part("gold/goal", goals.drop(columns=["pt"]))
    rep = link.run(store, write=False)
    assert rep["goal_check"] == {"error": "gold/goal lacks columns ['pt']"} and rep["status"] == "ok"


def test_a_stored_table_is_cut_to_the_kick_offs_and_fixtures_in_reach(tmp_path):
    store = Store(root=tmp_path)
    assert link._frame(store, "nothing.parquet", ["a"]) is None
    fx = _fixtures([(900, "Lille", "Rennes", "2016-01-03 20:00", 79, 94),
                    (901, "Lille", "Rennes", "2026-01-03 20:00", 79, 94),
                    (902, "Lille", "Rennes", "2026-01-05 20:00", 79, 94)])
    since, until = _ts("2026-01-02 20:00"), _ts("2026-01-04 20:00")
    for key, frame in (("aware.parquet", fx),                                               # a UTC timestamp column
                       ("naive.parquet", fx.assign(kickoff_utc=fx["kickoff_utc"].dt.tz_localize(None))),
                       ("text.parquet", fx.assign(kickoff_utc=fx["kickoff_utc"].astype(str)))):
        store.put_parquet(key, frame)
        got = link._frame(store, key, ["apif_fixture_id", "kickoff_utc", "home", "not_a_column"], since, until)
        assert got["apif_fixture_id"].tolist() == [901]
        assert list(got.columns) == ["apif_fixture_id", "kickoff_utc", "home"]
        assert len(link._frame(store, key, ["apif_fixture_id"])) == 3
    # the events table: only the linked fixtures' rows, whatever number type the ids were stored as
    ev = pd.DataFrame({"apif_fixture_id": [900.0, 901.0, 901.0, 902.0], "minute": [1, 2, 3, 4]})
    store.put_parquet("events.parquet", ev)
    ids = pd.Series([901, 902, 77], dtype="Int64")
    assert link._frame(store, "events.parquet", ["apif_fixture_id", "minute"], ids=ids)["minute"].tolist() == [2, 3, 4]
    # (a table read without its id column is not cut)
    assert link._frame(store, "events.parquet", ["minute"], ids=[901])["minute"].tolist() == [1, 2, 3, 4]
    # the run reads no fixture further than a day from the first and last Betfair kick-off
    ev, fx = _danish()
    far = _fixtures([(999, "Brondby", "Midtjylland", "2025-03-01 15:00", 407, 397, 2, 0)])
    wh = _warehouse(tmp_path / "wh")
    wh.put_parquet(link.FIXTURES_KEY, pd.concat([fx, far], ignore_index=True).assign(apif_league_id=119))
    assert link.run(wh, write=False)["fixtures"] == {"rows": 7, "from_fixtures_table": 7, "from_history": 0}


def test_every_column_is_documented_and_the_tables_meet_their_contracts():
    for spec in link.SPECS:
        assert all(c.doc.strip() for c in spec.columns) and set(spec.key) <= set(spec.names)
    ev, fx = _danish()
    lk = link.canonical(link.link_events(ev, fx), None)
    t = conform(link.LINK, link.to_arrow(link.LINK, lk))
    assert t.num_rows == 7 and t.column("apif_fixture_id").null_count == 1 and t.column("event_id").null_count == 0
    assert t.schema.field("apif_kickoff").type == pa.timestamp("ms", tz="UTC")
    # no events, or no fixtures: an empty or an all-unlinked table, never an error
    none = link.link_events(ev, _fixtures([]).reindex(columns=["apif_fixture_id", "kickoff_utc", "home", "away"]))
    assert not none["linked"].any() and none["method"].eq("none").all()
    assert conform(link.LINK, link.to_arrow(link.LINK, link.canonical(none, None))).num_rows == 7
    assert len(link.link_events(ev.head(0), fx)) == 0
