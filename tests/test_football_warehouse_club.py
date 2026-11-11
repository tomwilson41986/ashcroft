"""The club marts: a team's matches as the market valued them (hand-written inputs, exact values)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from football.warehouse import club
from football.warehouse.schema import TS, conform

T0 = pd.Timestamp("2026-01-03 15:00", tz="UTC")
logit = lambda p: math.log(p / (1 - p))                             # noqa: E731
lean = lambda xpts: club.PERF_LINE[0] + club.PERF_LINE[1] * xpts    # noqa: E731  the reading's lean at an expectation


def ts(days: float):
    return T0 + pd.Timedelta(days=days)


def table(rows: list[dict], types: dict) -> pa.Table:
    return pa.table({k: pa.array([r.get(k) for r in rows], t) for k, t in types.items()})


def inputs(events=None, implied=None, inplay=None, timeline=None, rating=None, teams=None) -> dict:
    """Two sides, United (1) and City (2), meet home and away; United's third match (v Town, 3) has no result."""
    events = events if events is not None else [
        dict(event_id=10, home_id=1, away_id=2, home_name="United", away_name="City", country_code="GB",
             category="senior", kickoff=ts(0), result="H", over25=1, went_in_play=True),
        dict(event_id=11, home_id=2, away_id=1, home_name="City", away_name="United", country_code="GB",
             category="senior", kickoff=ts(7), result="D", over25=0, went_in_play=True),
        dict(event_id=12, home_id=1, away_id=3, home_name="United", away_name="Town", country_code="GB",
             category="senior", kickoff=ts(14), result=None, over25=None, went_in_play=False)]
    implied = implied if implied is not None else [
        dict(event_id=10, cp_min=0, p_home=0.5, p_draw=0.25, p_away=0.25, lam_home=1.6, lam_away=1.0, total=2.6),
        dict(event_id=10, cp_min=1440, p_home=0.4, p_draw=0.3, p_away=0.3),
        dict(event_id=10, cp_min=60, p_home=0.5, p_draw=0.25, p_away=0.25),
        dict(event_id=11, cp_min=0, p_home=0.6, p_draw=0.2, p_away=0.2, lam_home=1.8, lam_away=0.9, total=2.7),
        dict(event_id=12, cp_min=0, p_home=0.7, p_draw=0.2, p_away=0.1, lam_home=2.0, lam_away=0.7, total=2.7)]
    inplay = inplay if inplay is not None else [
        dict(event_id=10, filled=True, xpts_inplay_home=2.4, xpts_inplay_away=0.4, ht_p_home=0.8, ht_p_draw=0.15,
             ht_p_away=0.05, max_p_home=0.99, min_p_home=0.45, max_p_away=0.3, min_p_away=0.0, travel=1.2,
             decided_match_min=64.0),
        dict(event_id=11, filled=False, xpts_inplay_home=2.9, xpts_inplay_away=0.1, ht_p_home=0.9, ht_p_draw=0.08,
             ht_p_away=0.02, max_p_home=0.95, min_p_home=0.2, max_p_away=0.3, min_p_away=0.01, travel=2.0,
             decided_match_min=None)]
    timeline = timeline if timeline is not None else [
        dict(event_id=10, usable=True, consistent=True, reconciled=False, reliability=0.99, n_unplaced=0,
             home_goals=2, away_goals=0, min_home_leading=70.0, min_level=25.0, min_home_trailing=0.0),
        dict(event_id=11, usable=True, consistent=False, reconciled=False, reliability=None, n_unplaced=0,
             home_goals=2, away_goals=1, min_home_leading=60.0, min_level=30.0, min_home_trailing=5.0)]
    rating = rating if rating is not None else [
        dict(team_id=1, event_id=10, n_prior=0, strength=0.1, strength_sd=1.0, pool_teams=1, n_after=1,
             strength_after=0.3),
        dict(team_id=2, event_id=10, n_prior=4, strength=0.1, strength_sd=0.2, pool_teams=40, n_after=5,
             strength_after=0.05),
        dict(team_id=1, event_id=11, n_prior=1, strength=0.3, strength_sd=0.5, pool_teams=41, n_after=2,
             strength_after=0.2),
        dict(team_id=2, event_id=11, n_prior=5, strength=0.05, strength_sd=0.2, pool_teams=41, n_after=6,
             strength_after=0.15)]
    teams = teams if teams is not None else [
        dict(team_id=1, team_name="United", country_code="GB"), dict(team_id=2, team_name="City", country_code="GB"),
        dict(team_id=3, team_name="Town", country_code="GB")]
    f, i, s, b = pa.float64(), pa.int64(), pa.string(), pa.bool_()
    return {
        "gold/event": table(events, dict(event_id=i, home_id=i, away_id=i, home_name=s, away_name=s, country_code=s,
                                          category=s, kickoff=TS, result=s, over25=pa.int8(), went_in_play=b)),
        "gold/implied": table(implied, dict(event_id=i, cp_min=pa.int32(), p_home=f, p_draw=f, p_away=f, lam_home=f,
                                            lam_away=f, total=f)),
        "gold/inplay": table(inplay, dict(event_id=i, filled=b, xpts_inplay_home=f, xpts_inplay_away=f, ht_p_home=f,
                                          ht_p_draw=f, ht_p_away=f, max_p_home=f, min_p_home=f, max_p_away=f,
                                          min_p_away=f, travel=f, decided_match_min=f)),
        # every goal sided and nothing against the third goal's moment, unless a row says otherwise
        "gold/timeline": table([dict({"n_side_unknown": 0, "agrees_third": None}, **r) for r in timeline],
                               dict(event_id=i, usable=b, consistent=b, reconciled=b, reliability=f,
                                    n_unplaced=pa.int16(), n_side_unknown=pa.int16(), agrees_third=b,
                                    home_goals=pa.int16(), away_goals=pa.int16(),
                                    min_home_leading=f, min_level=f, min_home_trailing=f)),
        "gold/team_rating": table([dict({"rated": True}, **r) for r in rating],          # rated unless a row says not
                                  dict(team_id=i, event_id=i, n_prior=pa.int32(), strength=f, strength_sd=f,
                                       pool_teams=pa.int32(), rated=b, n_after=pa.int32(), strength_after=f)),
        "gold/team": table(teams, dict(team_id=i, team_name=s, country_code=s)),
    }


@pytest.fixture()
def built():
    out = club.build(inputs())
    return {s.name.split("/")[1]: conform(s, out[s.name]).to_pandas() for s in club.SPECS}


def row(frame, team, event):
    return frame[(frame["team_id"] == team) & (frame["event_id"] == event)].iloc[0]


def test_two_rows_a_match_each_from_its_own_side(built):
    tm = built["team_match"]
    assert len(tm) == 6 and not tm.duplicated(["team_id", "event_id"]).any()
    h, a = row(tm, 1, 10), row(tm, 2, 10)
    assert (h["is_home"], h["opponent_id"], h["team_name"], h["opponent_name"]) == (True, 2, "United", "City")
    assert (a["is_home"], a["opponent_id"], a["team_name"], a["opponent_name"]) == (False, 1, "City", "United")
    assert (h["p_win"], h["p_draw"], h["p_loss"], h["xpts_pre"]) == (0.5, 0.25, 0.25, 1.75)
    assert (a["p_win"], a["p_loss"], a["xpts_pre"]) == (0.25, 0.5, 1.0)
    assert (h["lam_for"], h["lam_against"], a["lam_for"], a["lam_against"]) == (1.6, 1.0, 1.0, 1.6)
    assert (h["result"], h["points"], a["result"], a["points"]) == ("W", 3, "L", 0)
    d = row(tm, 1, 11)
    assert (d["result"], d["points"], d["is_home"], d["p_win"]) == ("D", 1, False, 0.2)


def test_the_markets_move_on_a_team_is_in_its_own_win_chance(built):
    tm = built["team_match"]
    h, a = row(tm, 1, 10), row(tm, 2, 10)
    assert h["move_24h"] == pytest.approx(logit(0.5) - logit(0.4)) and h["move_24h"] > 0      # came for the home side
    assert a["move_24h"] == pytest.approx(logit(0.25) - logit(0.3)) and a["move_24h"] < 0
    assert h["move_60m"] == pytest.approx(0.0) and np.isnan(h["move_6h"])                      # no T-360 reading


def test_a_result_splits_into_performance_and_luck(built):
    tm = built["team_match"]
    h, a = row(tm, 1, 10), row(tm, 2, 10)
    # performance is measured from what sides of that expectation average on the raw difference (the reading leans
    # against strong sides); luck is the rest of the result's surprise, so the two always add up to it
    assert h["pts_vs_xpts"] == pytest.approx(1.25) and h["perf"] == pytest.approx(0.65 - lean(1.75))
    assert h["luck"] == pytest.approx(0.6 + lean(1.75)) and h["perf"] + h["luck"] == pytest.approx(h["pts_vs_xpts"])
    assert a["perf"] == pytest.approx(0.4 - 1.0 - lean(1.0)) and a["luck"] == pytest.approx(0 - 0.4 + lean(1.0))
    assert lean(0.3) > 0 > lean(2.5)                          # raw, an outsider reads high and a favourite low
    assert (h["ht_p_win"], h["max_p_win"], h["min_p_win"], a["ht_p_win"], a["max_p_win"]) == (0.8, 0.99, 0.45, 0.05,
                                                                                               0.3)
    assert (h["travel"], h["decided_min"], a["decided_min"]) == (1.2, 64.0, 64.0)     # the football minute
    # a match that traded too little in play keeps its expectation and result, not its in-play record
    thin = row(tm, 2, 11)
    assert np.isnan(thin["xpts_inplay"]) and np.isnan(thin["perf"]) and np.isnan(thin["luck"])
    assert thin["pts_vs_xpts"] == pytest.approx(1 - (3 * 0.6 + 0.2))


def test_inferred_goals_and_minutes_only_where_the_timeline_holds_together(built):
    tm = built["team_match"]
    h, a = row(tm, 1, 10), row(tm, 2, 10)
    assert (h["goals_for"], h["goals_against"], a["goals_for"], a["goals_against"]) == (2, 0, 0, 2)
    assert h["gd_vs_exp"] == pytest.approx(2 - 0.6) and a["gd_vs_exp"] == pytest.approx(-2 + 0.6)
    assert (h["min_leading"], h["min_level"], h["min_trailing"]) == (70.0, 25.0, 0.0)
    assert (a["min_leading"], a["min_level"], a["min_trailing"]) == (0.0, 25.0, 70.0)
    assert h["score_reliability"] == 0.99
    bad = row(tm, 2, 11)                                             # its timeline disagrees with the settlement
    assert np.isnan(bad["goals_for"]) and np.isnan(bad["min_leading"]) and np.isnan(bad["gd_vs_exp"])
    assert np.isnan(bad["score_reliability"])


def test_a_score_put_right_by_the_settlement_counts_and_a_goal_without_a_minute_blanks_the_minutes():
    tl = [dict(event_id=10, usable=True, consistent=False, reconciled=True, reliability=0.75, n_unplaced=1,
               home_goals=2, away_goals=0, min_home_leading=70.0, min_level=25.0, min_home_trailing=0.0)]
    tm = conform(club.TEAM_MATCH, club.build(inputs(timeline=tl))["club/team_match"]).to_pandas()
    h = row(tm, 1, 10)
    assert (h["goals_for"], h["goals_against"], h["score_reliability"]) == (2, 0, 0.75)
    assert np.isnan(h["min_leading"]) and np.isnan(h["min_level"]) and np.isnan(h["min_trailing"])
    # no score from a list with a goal of no side, or whose third goal is not where the over/under market stopped
    for flaw in (dict(n_side_unknown=1), dict(agrees_third=False)):
        bad = [dict(event_id=10, usable=True, consistent=True, reconciled=False, reliability=None, n_unplaced=0,
                    home_goals=2, away_goals=0, min_home_leading=70.0, min_level=25.0, min_home_trailing=0.0, **flaw)]
        tm = conform(club.TEAM_MATCH, club.build(inputs(timeline=bad))["club/team_match"]).to_pandas()
        h = row(tm, 1, 10)
        assert np.isnan(h["goals_for"]) and np.isnan(h["goals_against"]) and np.isnan(h["min_leading"])
        assert h["points"] == 3 and h["pts_vs_xpts"] == pytest.approx(1.25)       # the result needs no score


def test_ratings_are_the_ones_going_into_the_match_and_only_once_a_side_is_rated(built):
    tm = built["team_match"]
    first, away = row(tm, 1, 10), row(tm, 2, 10)
    # United's first match: the row carries its opponent's rating (n_prior 0), which is not United's
    assert np.isnan(first["strength"]) and first["n_prior"] == 0 and first["strength_after"] == 0.3
    assert np.isnan(first["pool_teams"])
    assert first["opp_strength"] == 0.1 and np.isnan(away["opp_strength"]) and away["strength"] == 0.1
    second = row(tm, 1, 11)
    assert (second["strength"], second["strength_sd"], second["opp_strength"], second["pool_teams"]) == (0.3, 0.5,
                                                                                                         0.05, 41)
    assert np.isnan(row(tm, 3, 12)["strength"]) and np.isnan(row(tm, 3, 12)["points"])            # no rating, no result


def test_the_summary_adds_up_a_teams_matches_with_a_result(built):
    s = built["team_summary"].set_index("team_id")
    assert s.index.tolist() == [1, 2]                                 # Town has no match with a result
    u = s.loc[1]
    assert (u["team_name"], u["country_code"], u["category"]) == ("United", "GB", "senior")
    assert (u["matches"], u["wins"], u["draws"], u["losses"], u["ppm"]) == (2, 1, 1, 0, 2.0)
    assert u["first_kickoff"] == ts(0) and u["last_kickoff"] == ts(7)
    assert u["xpts_pre_pm"] == pytest.approx((1.75 + 0.8) / 2)        # home at 0.5/0.25, away at 0.2/0.2
    over = [3 - 1.75, 1 - 0.8]
    assert u["pts_vs_xpts_pm"] == pytest.approx(np.mean(over))
    assert u["pts_vs_xpts_se"] == pytest.approx(np.std(over, ddof=1) / math.sqrt(2))
    assert (u["n_inplay"], u["perf_pm"], u["luck_pm"]) == (1, pytest.approx(0.65 - lean(1.75)),
                                                           pytest.approx(0.6 + lean(1.75)))
    assert np.isnan(u["perf_se"])                                     # one match: no standard error
    assert u["xgf_pm"] == pytest.approx((1.6 + 0.9) / 2) and u["xga_pm"] == pytest.approx((1.0 + 1.8) / 2)
    assert u["fav_share"] == 0.5 and u["move_24h_pm"] == pytest.approx(logit(0.5) - logit(0.4))
    assert u["opp_strength_pm"] == pytest.approx((0.1 + 0.05) / 2)
    # the rating now; its first settled value needs ten matches behind it, which neither row has
    assert u["strength"] == 0.2 and np.isnan(u["strength_first"]) and np.isnan(u["strength_change"])
    assert (u["ppm_last5"], u["ppm_last10"], u["perf_last5"]) == (2.0, 2.0, pytest.approx(0.65 - lean(1.75)))
    assert (u["pool_id"], u["pool_size"]) == (1, 2)                   # United and City have met: one pool of two
    assert not u["ranked"] and np.isnan(u["country_rank"])            # two matches: under the ten a ranking needs


def test_the_rating_change_is_measured_from_the_first_settled_rating(monkeypatch):
    monkeypatch.setattr(club, "MIN_MATCHES", 1)                       # a rating with one match behind it has settled
    s = conform(club.TEAM_SUMMARY, club.build(inputs())["club/team_summary"]).to_pandas().set_index("team_id")
    monkeypatch.setattr(club, "DRIFT_MIN", 1)
    s = conform(club.TEAM_SUMMARY, club.build(inputs())["club/team_summary"]).to_pandas().set_index("team_id")
    # both matches are in one month: the scale has not moved, the change is the plain difference
    assert s.loc[1, "strength_first"] == 0.3 and s.loc[1, "strength_change"] == pytest.approx(-0.1)
    assert s.loc[2, "strength_first"] == 0.1 and s.loc[2, "strength_change"] == pytest.approx(0.05)


def test_the_rating_change_is_net_of_the_drift_of_the_scale(monkeypatch):
    # Three sides play each other in January and again in February. Every rating is 0.5 higher in February (the
    # scale's centre moved as other teams joined); beyond that only side 3 changed, by 0.2.
    monkeypatch.setattr(club, "MIN_MATCHES", 1)
    monkeypatch.setattr(club, "DRIFT_MIN", 3)
    ev, rt, n = [], [], 0
    base = {1: 1.0, 2: 0.0, 3: -1.0}
    for month, shift in ((0, 0.0), (31, 0.5)):
        for h, a in ((1, 2), (2, 3), (3, 1)):
            n += 1
            ev.append(dict(event_id=n, home_id=h, away_id=a, home_name=f"T{h}", away_name=f"T{a}", country_code="GB",
                           category="senior", kickoff=ts(month + n), result="D", over25=0, went_in_play=True))
            for side in (h, a):
                now = base[side] + shift + (0.2 if side == 3 and shift else 0.0)
                rt.append(dict(team_id=side, event_id=n, n_prior=5, strength=now, strength_sd=0.1, pool_teams=3,
                               n_after=6, strength_after=now))
    teams = [dict(team_id=k, team_name=f"T{k}", country_code="GB") for k in base]
    out = club.build(inputs(events=ev, implied=[], inplay=[], timeline=[], rating=rt, teams=teams))
    s = conform(club.TEAM_SUMMARY, out["club/team_summary"]).to_pandas().set_index("team_id")
    assert s["strength"].tolist() == pytest.approx([1.5, 0.5, -0.3])                 # as rated now, drift and all
    assert s["strength_change"].tolist() == pytest.approx([0.0, 0.0, 0.2])           # the median side held still
    assert s["country_change"].isna().all() and s["rerating"].isna().all()           # three sides: under the five
    monkeypatch.setattr(club, "COUNTRY_MIN", 3)
    monkeypatch.setattr(club, "POOL_MIN", 3)
    out = club.build(inputs(events=ev, implied=[], inplay=[], timeline=[], rating=rt, teams=teams))
    s = conform(club.TEAM_SUMMARY, out["club/team_summary"]).to_pandas().set_index("team_id")
    assert s["country_change"].tolist() == pytest.approx([0.0, 0.0, 0.0])            # the country's median side
    assert s["rerating"].tolist() == pytest.approx([0.0, 0.0, 0.2])
    # too few sides rated in both months to read the drift: no figure, not a wrong one
    monkeypatch.setattr(club, "DRIFT_MIN", 4)
    out = club.build(inputs(events=ev, implied=[], inplay=[], timeline=[], rating=rt, teams=teams))
    assert conform(club.TEAM_SUMMARY, out["club/team_summary"]).to_pandas()["strength_change"].isna().all()


def test_a_thin_month_starts_a_new_run_of_the_scales_drift_and_does_not_blank_the_later_ones(monkeypatch):
    """Three sides, all rated in January, March and April and one of them in February: the steps into and out of
    February cannot be read, the step from March to April can. Before, every level after January was blank, and
    with it the rating change of every side of a small pool."""
    monkeypatch.setattr(club, "DRIFT_MIN", 3)
    monkeypatch.setattr(club, "MIN_MATCHES", 1)
    rows = []
    for month, sides, shift in ((1, (1, 2, 3), 0.0), (2, (1,), 0.1), (3, (1, 2, 3), 0.3), (4, (1, 2, 3), 0.7)):
        rows += [dict(team_id=k, kickoff=pd.Timestamp(2026, month, 10, tz="UTC"), strength=float(k) + shift,
                      n_prior=5) for k in sides]
    pool = pd.DataFrame({"team_id": [1, 2, 3], "pool_id": [1, 1, 1]})
    got = club.scale_drift(pd.DataFrame(rows), pool).set_index("ym")
    jan = 2026 * 12
    assert got["run"].to_dict() == {jan: 0, jan + 1: 1, jan + 2: 2, jan + 3: 2}
    assert got["level"].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.4])       # March to April: every side up 0.4


def test_ranking_within_a_country_and_overall(monkeypatch):
    monkeypatch.setattr(club, "MIN_MATCHES", 1)
    monkeypatch.setattr(club, "POOL_MIN", 2)
    out = club.build(inputs())
    s = conform(club.TEAM_SUMMARY, out["club/team_summary"]).to_pandas().set_index("team_id")
    assert s["ranked"].all() and s["country_teams"].tolist() == [2, 2]
    assert (s.loc[1, "country_rank"], s.loc[2, "country_rank"]) == (1, 2)          # 0.2 against 0.15
    assert (s.loc[1, "pct_overall"], s.loc[2, "pct_overall"]) == (1.0, 0.0)
    c = conform(club.COUNTRY, out["club/country_strength"]).to_pandas()
    assert len(c) == 1
    g = c.iloc[0]
    assert (g["country_code"], g["category"], g["teams"], g["matches"]) == ("GB", "senior", 2, 2)
    assert g["strength_top"] == 0.2 and g["strength_median"] == pytest.approx(0.175) and np.isnan(g["strength_top5"])
    assert g["total_mean"] == pytest.approx((2.6 + 2.7) / 2) and g["home_edge_mean"] == pytest.approx((0.6 + 0.9) / 2)
    assert g["p_home_mean"] == pytest.approx(0.55) and (g["home_rate"], g["draw_rate"], g["over25_rate"]) == (0.5,
                                                                                                             0.5, 0.5)


def test_pools_are_the_teams_recent_matches_join_and_rankings_stay_inside_one(monkeypatch):
    monkeypatch.setattr(club, "MIN_MATCHES", 1)
    monkeypatch.setattr(club, "POOL_MIN", 2)
    base = inputs()
    # two more sides, Rovers (4) and Athletic (5), who only ever meet each other: a pool of their own
    ev = base["gold/event"].to_pylist() + [
        dict(event_id=20, home_id=4, away_id=5, home_name="Rovers", away_name="Athletic", country_code="GB",
             category="senior", kickoff=ts(3), result="A", over25=1, went_in_play=True)]
    rt = base["gold/team_rating"].to_pylist() + [
        dict(team_id=4, event_id=20, n_prior=2, strength=0.9, strength_sd=0.3, pool_teams=2, n_after=3,
             strength_after=0.8),
        dict(team_id=5, event_id=20, n_prior=2, strength=0.7, strength_sd=0.3, pool_teams=2, n_after=3,
             strength_after=0.9)]
    teams = base["gold/team"].to_pylist() + [dict(team_id=4, team_name="Rovers", country_code="GB"),
                                             dict(team_id=5, team_name="Athletic", country_code="GB")]
    out = club.build(inputs(events=ev, rating=rt, teams=teams))
    s = conform(club.TEAM_SUMMARY, out["club/team_summary"]).to_pandas().set_index("team_id")
    assert s["pool_id"].to_dict() == {1: 1, 2: 1, 4: 4, 5: 4} and (s["pool_size"] == 2).all()
    # Athletic's 0.9 is the highest number in the table and says nothing about United's 0.2: each pool ranks alone
    assert s["country_rank"].to_dict() == {1: 1, 2: 2, 4: 2, 5: 1} and (s["country_teams"] == 2).all()
    assert s["pct_overall"].to_dict() == {1: 1.0, 2: 0.0, 4: 0.0, 5: 1.0}
    c = conform(club.COUNTRY, out["club/country_strength"]).to_pandas().iloc[0]
    assert (c["teams"], c["matches"]) == (2, 3) and c["strength_top"] in (0.2, 0.9)       # one pool's sides only
    # a tie between the pools that rates nobody (settled, but it never went in play) leaves them two scales; one
    # that does rate its sides makes them one
    tie = dict(event_id=21, home_id=1, away_id=4, home_name="United", away_name="Rovers", country_code="GB",
               category="senior", kickoff=ts(5), result="H", over25=1, went_in_play=False)
    row_of = lambda team, rated: dict(team_id=team, event_id=21, n_prior=3, strength=0.5, strength_sd=0.3,  # noqa: E731
                                      pool_teams=2, n_after=3, strength_after=0.5, rated=rated)
    for rated, want in ((False, {1: 1, 2: 1, 4: 4, 5: 4}), (True, {1: 1, 2: 1, 4: 1, 5: 1})):
        joined = club.build(inputs(events=ev + [tie], rating=rt + [row_of(1, rated), row_of(4, rated)], teams=teams))
        got = conform(club.TEAM_SUMMARY, joined["club/team_summary"]).to_pandas().set_index("team_id")
        assert got["pool_id"].to_dict() == want
    # a side that has not played for too long drops out of the ranking, and a pool too small ranks nobody
    monkeypatch.setattr(club, "ACTIVE_DAYS", 3)                # the last match held is on day 7, theirs on day 3
    stale = conform(club.TEAM_SUMMARY, club.build(inputs(events=ev, rating=rt, teams=teams))["club/team_summary"])
    assert stale.to_pandas().set_index("team_id")["ranked"].to_dict() == {1: True, 2: True, 4: False, 5: False}
    monkeypatch.setattr(club, "POOL_MIN", 3)
    assert not conform(club.TEAM_SUMMARY, club.build(inputs())["club/team_summary"]).to_pandas()["ranked"].any()


def test_empty_inputs_give_empty_tables_and_the_build_is_pure():
    empty = club.build(inputs(events=[], implied=[], inplay=[], timeline=[], rating=[], teams=[]))
    for spec in club.SPECS:
        assert empty[spec.name].num_rows == 0 and empty[spec.name].schema.names == spec.names
    a, b = club.build(inputs()), club.build(inputs())
    for spec in club.SPECS:
        assert a[spec.name].equals(b[spec.name])
        conform(spec, a[spec.name])
