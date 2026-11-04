"""Expected-goals features: state accounting, the clean-xG arithmetic and lag safety (synthetic rows only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football import xg_features as xf
from football.team_features import windows


def test_match_states_from_goals_and_a_red_card():
    hm = pd.DataFrame({"apif_fixture_id": [1, 2], "apif_home_id": [10, 30]})
    he = pd.DataFrame([
        {"apif_fixture_id": 1, "minute": 30, "type": "Goal", "detail": "Normal Goal", "apif_team_id": 10},
        {"apif_fixture_id": 1, "minute": 60, "type": "Card", "detail": "Red Card", "apif_team_id": 20},
        {"apif_fixture_id": 1, "minute": 75, "type": "Goal", "detail": "Missed Penalty", "apif_team_id": 20}])
    s = xf.match_states(hm, he).set_index("apif_fixture_id")
    assert s.loc[1, "h_level"] == pytest.approx(30 / 90) and s.loc[1, "h_lead1"] == pytest.approx(60 / 90)
    assert s.loc[1, "h_up"] == pytest.approx(30 / 90) and s.loc[1, "h_down"] == 0
    assert s.loc[2, "h_level"] == 1.0                       # no events: level throughout


def test_state_factor_directions():
    one, zero = pd.Series([1.0]), pd.Series([0.0])
    assert xf.state_factor(zero, zero, one, zero, zero, zero, zero).iloc[0] == xf.STATE_MULT["trail1"]   # trailing: down
    assert xf.state_factor(zero, one, zero, zero, zero, zero, zero).iloc[0] == xf.STATE_MULT["lead2"]    # leading: up
    assert xf.state_factor(zero, zero, zero, zero, one, one, zero).iloc[0] == pytest.approx(1 / 0.35)   # a man down


def match_frame(days, xg_h, xg_a, pens_h=0):
    n = len(days)
    teams = [("ENG:a", "ENG:b"), ("ENG:c", "ENG:d")]
    rows = []
    for i, d in enumerate(days):
        h, a = teams[i % 2] if i % 4 < 2 else teams[i % 2][::-1]
        rows.append({"match_id": f"m{i}", "date": pd.Timestamp(d), "season": 2024, "country": "ENG",
                     "competition_id": "ENG1", "home_team_id": h, "away_team_id": a, "ft_home": 1, "ft_away": 1,
                     "mclam_h": 1.4, "mclam_a": 1.1, "apif_fixture_id": i, "xg_h": xg_h[i], "xg_a": xg_a[i],
                     "goals_prevented_h": 0.1, "goals_prevented_a": -0.1, "shots_h": 12, "shots_a": 9,
                     "shots_on_h": 5, "shots_on_a": 3, "shots_inside_h": 8, "shots_inside_a": 5,
                     "pens_scored_h": pens_h, "pens_missed_h": 0, "pens_scored_a": 0, "pens_missed_a": 0,
                     "red_cards_h": 0, "red_cards_a": 0, **{f"h_{k}": (1.0 if k == "level" else 0.0) for k in
                                                         ("level", "lead1", "lead2", "trail1", "trail2", "down", "up")}})
    return pd.DataFrame(rows)


def test_non_penalty_xg_takes_the_penalties_out():
    m = match_frame(["2024-09-07"], [1.5], [0.8], pens_h=1)
    t = xf.side_rows(m).set_index("venue")
    assert t.loc["H", "npxg"] == pytest.approx(1.5 - xf.PEN_XG) and t.loc["H", "npxga"] == pytest.approx(0.8)
    assert t.loc["H", "finishing"] == pytest.approx((1 - 1) - (1.5 - xf.PEN_XG))   # the penalty goal is not finishing
    assert t.loc["H", "npxgd_vs_mkt"] == pytest.approx((1.5 - xf.PEN_XG - 0.8) - (1.4 - 1.1))


def test_a_saturday_xg_rewrite_leaves_sunday_windows_and_ratings_alone():
    rng = np.random.default_rng(0)
    days = []
    for w in range(30):
        sat = pd.Timestamp("2024-08-03") + pd.Timedelta(days=7 * w)
        days += [sat, sat, sat + pd.Timedelta(days=1), sat + pd.Timedelta(days=1)]
    m = match_frame(days, rng.gamma(2, 0.7, len(days)), rng.gamma(2, 0.6, len(days)))
    # more sides so the ratings have a pool to fit
    m2 = m.copy()
    m2["home_team_id"] = m2["home_team_id"] + "x"
    m2["away_team_id"] = m2["away_team_id"] + "x"
    m2["match_id"] = m2["match_id"] + "x"
    m = pd.concat([m, m2], ignore_index=True)
    sat, sun = days[-4], days[-1]

    def feats(frame):
        t = xf.side_rows(frame)
        w = windows(t, xf.XG_MEASURES)
        sw = xf.short_windows(t, xf.SHORT_MEASURES)
        w = pd.concat([w, sw], axis=1).set_index(["match_id", "venue"]).sort_index()
        mm = frame.assign(npxg_h=frame["xg_h"], npxg_a=frame["xg_a"])
        r = xf.xg_ratings(mm).set_index("match_id").sort_index()
        return w, r
    wa, ra = feats(m)
    b = m.copy()
    on = b["date"] == sat
    b.loc[on, ["xg_h", "xg_a"]] = b.loc[on, ["xg_a", "xg_h"]].to_numpy() * 3
    wb, rb = feats(b)
    sunday = m.loc[m["date"] == sun, "match_id"]
    cols = [c for c in wa.columns if c not in ("team", "date", "own_r")]
    pd.testing.assert_frame_equal(wa.loc[wa.index.get_level_values(0).isin(sunday), cols],
                                  wb.loc[wb.index.get_level_values(0).isin(sunday), cols])
    pd.testing.assert_frame_equal(ra.loc[ra.index.isin(sunday)], rb.loc[rb.index.isin(sunday)])


def test_short_windows_weight_the_latest_match_most_and_never_see_the_match_itself():
    t = pd.DataFrame({"team": ["a"] * 6, "date": pd.date_range("2024-08-03", periods=6, freq="7D"),
                      "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    w = xf.short_windows(t, ["v"], equal=(2, 3), linear=(5,))
    last = w.iloc[-1]
    assert last["v_l2"] == 4.5 and last["v_l3"] == 4.0                  # matches 4 and 5; 3, 4 and 5
    assert abs(last["v_lw5"] - (5 * 5 + 4 * 4 + 3 * 3 + 2 * 2 + 1 * 1) / 15) < 1e-12
    assert np.isnan(w.iloc[0]["v_l2"])


def test_sos_xg_is_measured_against_the_opponents_earlier_matches():
    # b concedes 2 npxg a match before meeting a; a makes 3 against b: +1 above what b usually allows
    rows = []
    for i, d in enumerate(pd.date_range("2024-08-03", periods=4, freq="7D")):
        rows.append({"match_id": f"m{i}", "date": d, "team": "b", "opp": f"x{i}", "npxg": 1.0, "npxga": 2.0})
        rows.append({"match_id": f"m{i}", "date": d, "team": f"x{i}", "opp": "b", "npxg": 2.0, "npxga": 1.0})
    d = pd.Timestamp("2024-09-07")
    rows += [{"match_id": "z", "date": d, "team": "a", "opp": "b", "npxg": 3.0, "npxga": 0.5},
             {"match_id": "z", "date": d, "team": "b", "opp": "a", "npxg": 0.5, "npxga": 3.0}]
    t = xf.add_sos(pd.DataFrame(rows).sort_values(["team", "date"]))
    a = t[(t["match_id"] == "z") & (t["team"] == "a")].iloc[0]
    assert a["npxg_sos"] == 1.0 and a["npxga_sos"] == -0.5 and a["npxgd_sos"] == 1.5
