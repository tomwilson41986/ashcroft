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
        {"apif_fixture_id": 1, "minute": 80, "type": "Goal", "detail": "Own Goal", "apif_team_id": 20,
         "apif_player_id": 4, "apif_assist_id": None}])     # a home player's own goal, listed under the away side
    s = pr.stints(hp, he).sort_values("start").reset_index(drop=True)
    assert list(zip(s["start"], s["end"])) == [(0, 60), (60, 70), (70, 90)]
    assert list(zip(s["goals_home"], s["goals_away"])) == [(1, 0), (0, 0), (0, 1)]   # listed under the away side, it counts for the away
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


def test_team_news_reads_only_matches_before_the_collection_day():
    """A side's team news at a Thursday match is fixed by its matches before Tuesday's collection: a match it played
    on the Tuesday evening (after the 13:00 early price) must not enter."""
    import numpy as np
    import pandas as pd
    from football.player_ratings import team_news
    days = pd.to_datetime(["2026-10-03", "2026-10-10", "2026-10-17", "2026-10-24", "2026-11-03", "2026-11-05"])
    fixtures = list(range(1, 7))
    m = pd.DataFrame({"apif_fixture_id": fixtures, "date": days})
    rows = []
    for f, d in zip(fixtures, days):
        for p in range(1, 12):
            pid = p if not (f == 4 and p == 1) else 99         # the regular player 1 misses fixture 4
            rows.append({"apif_fixture_id": f, "apif_team_id": 7, "side": "home", "apif_player_id": float(pid),
                         "role": "starter", "minutes_events": 90.0, "minutes_stats": 90.0})
        for p in range(1, 12):
            rows.append({"apif_fixture_id": f, "apif_team_id": 8, "side": "away", "apif_player_id": float(100 + p),
                         "role": "starter", "minutes_events": 90.0, "minutes_stats": 90.0})
    hp = pd.DataFrame(rows)
    he = pd.DataFrame({"apif_fixture_id": [5], "type": ["Card"], "detail": ["Red Card"], "apif_team_id": [7],
                       "apif_player_id": [2.0]})
    ratings = {"2026-11": {1.0: 0.2, 2.0: 0.1}}
    a = team_news(m, hp, he, ratings, ratings).set_index("apif_fixture_id")
    # fixture 6 (Thursday 5 Nov): collection Tuesday 3 Nov 13:00; fixture 5 (Tuesday) is not yet known
    # player 1 (started 3 of the 4 matches before: share 0.75, rating 0.2) missed fixture 4
    assert a.loc[6, "tn_absent_n_diff"] == 1 and abs(a.loc[6, "tn_absent_mpm_diff"] - 0.15) < 1e-9
    assert a.loc[6, "tn_red_mpm_diff"] == 0                                    # fixture 5's red card unseen
    # minutes load: nothing in the 7 days before 3 Nov; fixture 4 (24 Oct) in the 14: home's expected eleven
    # (ten players at share 1, player 99 at 0.25 played it) 10.25 matches of minutes, away's 11
    assert a.loc[6, "ld_load7_diff"] == 0 and abs(a.loc[6, "ld_load14_diff"] - (-0.75)) < 1e-9
    hp2 = hp.copy()
    hp2.loc[hp2["apif_fixture_id"] == 5, "apif_player_id"] = 555.0             # rewrite the Tuesday match entirely
    b = team_news(m, hp2, he, ratings, ratings).set_index("apif_fixture_id")
    assert a.loc[6].equals(b.loc[6])
