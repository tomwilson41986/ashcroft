"""Player ratings and the line-up surprise: mechanics on synthetic matches only (never a reported figure)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from football import player_ratings as pr


def synthetic(n_teams=10, squad=16, weeks=60, seed=0):
    """Sides with squads of rated players; each match starts eleven drawn from the squad (the best more often); the
    close's supremacy is the starters' true sum (home minus away) plus home advantage and noise."""
    rng = np.random.default_rng(seed)
    truth = {}
    squads = {}
    for t in range(n_teams):
        squads[t] = [1000 * t + i for i in range(squad)]
        for p in squads[t]:
            truth[p] = rng.normal(0, 0.05)
    matches, starts = [], []
    day = pd.Timestamp("2020-08-01")
    fid = 0
    for w in range(weeks):
        order = rng.permutation(n_teams)
        for i in range(0, n_teams, 2):
            h, a = order[i], order[i + 1]
            xi = {}
            for side, t in (("home", h), ("away", a)):
                pick = rng.choice(squads[t], 11, replace=False)
                xi[side] = pick
                starts += [{"apif_fixture_id": fid, "side": side, "apif_team_id": t, "apif_player_id": p} for p in pick]
            sup = 0.25 + sum(truth[p] for p in xi["home"]) - sum(truth[p] for p in xi["away"]) + rng.normal(0, 0.05)
            matches.append({"apif_fixture_id": fid, "date": day, "competition_id": "ENG1", "sup_close": sup})
            fid += 1
        day += pd.Timedelta(days=7)
    return pd.DataFrame(matches), pd.DataFrame(starts), truth


def test_market_plus_minus_recovers_the_players():
    m, st, truth = synthetic()
    r = pr.fit_mpm(m, st, m["date"].max() + pd.Timedelta(days=1), ridge=1.0, min_matches=100)
    ps = list(truth)
    assert np.corrcoef([r[p] for p in ps], [truth[p] for p in ps])[0, 1] > 0.8


def test_stints_cut_at_substitutions_and_red_cards():
    hp = pd.DataFrame(
        [{"apif_fixture_id": 1, "side": "home", "apif_team_id": 10, "role": "starter", "apif_player_id": i} for i in range(11)]
        + [{"apif_fixture_id": 1, "side": "away", "apif_team_id": 20, "role": "starter", "apif_player_id": 100 + i}
           for i in range(11)]
        + [{"apif_fixture_id": 1, "side": "home", "apif_team_id": 10, "role": "substitute", "apif_player_id": 50}])
    he = pd.DataFrame([
        {"apif_fixture_id": 1, "minute": 20, "type": "Goal", "detail": "Normal Goal", "apif_team_id": 10,
         "apif_player_id": 3, "apif_assist_id": None},
        {"apif_fixture_id": 1, "minute": 60, "type": "subst", "detail": "Substitution 1", "apif_team_id": 10,
         "apif_player_id": 3, "apif_assist_id": 50},
        {"apif_fixture_id": 1, "minute": 70, "type": "Card", "detail": "Red Card", "apif_team_id": 20,
         "apif_player_id": 105, "apif_assist_id": None},
        {"apif_fixture_id": 1, "minute": 80, "type": "Goal", "detail": "Own Goal", "apif_team_id": 10,
         "apif_player_id": 4, "apif_assist_id": None}])
    s = pr.stints(hp, he).sort_values("start").reset_index(drop=True)
    assert list(zip(s["start"], s["end"])) == [(0, 60), (60, 70), (70, 90)]
    assert list(zip(s["goals_home"], s["goals_away"])) == [(1, 0), (0, 0), (0, 1)]   # the own goal counts for the away
    assert 3 in s.loc[0, "home"] and 3 not in s.loc[1, "home"] and 50 in s.loc[1, "home"]
    assert len(s.loc[2, "away"]) == 10 and 105 not in s.loc[2, "away"]


def test_the_surprise_uses_earlier_matches_and_never_the_matchs_own_close():
    m, st, _ = synthetic(weeks=40)
    fits = pr.monthly(m, lambda as_of: pr.fit_mpm(m, st, as_of, ridge=1.0, min_matches=50), "2021-01")
    a = pr.surprise(m, st, fits, "mpm")
    last = m[m["date"] == m["date"].max()]["apif_fixture_id"]
    m2 = m.copy()
    m2.loc[m2["apif_fixture_id"].isin(last), "sup_close"] += 5.0          # rewrite the last day's closes
    fits2 = pr.monthly(m2, lambda as_of: pr.fit_mpm(m2, st, as_of, ridge=1.0, min_matches=50), "2021-01")
    b = pr.surprise(m2, st, fits2, "mpm")
    x = a[a["apif_fixture_id"].isin(last)].set_index("apif_fixture_id").sort_index()
    y = b[b["apif_fixture_id"].isin(last)].set_index("apif_fixture_id").sort_index()
    pd.testing.assert_frame_equal(x, y)
    first_week = m[m["date"] == m["date"].min()]["apif_fixture_id"]
    assert a[a["apif_fixture_id"].isin(first_week)]["mpm_exp_home"].isna().all()   # no earlier matches, no expectation
