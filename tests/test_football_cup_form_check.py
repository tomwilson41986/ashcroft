"""Cup form check: form is point in time (a match never sees its own statistics)."""

from __future__ import annotations

import pandas as pd

from research.open_close import football_cup_form_check as cf


def test_team_form_uses_only_earlier_matches():
    rows, fx = [], []
    for i in range(6):
        fid = 100 + i
        fx.append({"apif_fixture_id": fid, "kickoff_utc": f"2026-01-{i + 1:02d}T15:00:00Z"})
        rows.append({"apif_fixture_id": fid, "apif_team_id": 1, "shots_on": i, "shots": 10, "xg": 1.0})
        rows.append({"apif_fixture_id": fid, "apif_team_id": 2, "shots_on": 0, "shots": 10, "xg": 1.0})
    f = cf.team_form(pd.DataFrame(rows), pd.DataFrame(fx))
    t1 = f[f["apif_team_id"] == 1].sort_values("apif_fixture_id")
    assert t1["sotd"].isna().tolist()[:3] == [True, True, True]          # fewer than 3 earlier matches
    assert t1["sotd"].tolist()[3] == (0 + 1 + 2) / 3                      # matches 0-2 only, not match 3 itself
    assert t1["n_prior"].tolist() == list(range(6))


def test_linked_matches_carry_their_own_fixture_id():
    from research.open_close import football_cross_league_price as cp
    rows = []
    for mid, home, away, t in (("1.1", "Alpha", "Beta", "2026-09-01T15:00:00Z"),
                               ("1.2", "Gamma", "Delta", "2026-09-01T18:00:00Z")):
        for name, leg, won in ((home, "h", 1.0), (away, "a", 0.0), ("The Draw", "d", 0.0)):
            rows.append({"market_id": mid, "event_id": mid, "event_name": f"{home} v {away}",
                         "market_type": "MATCH_ODDS", "country": "GB", "market_time": t, "runner_name": name,
                         "ltp_t_1440": 3.0, "ltp_first": 3.0, "ltp_close": 3.0, "won": won})
    # the fixtures out of kick-off order, so the link's sorted copy re-indexes them
    d = pd.DataFrame({"kickoff": pd.to_datetime(["2026-09-01T18:00:00Z", "2026-09-01T15:00:00Z"], utc=True),
                      "home": ["Gamma", "Alpha"], "away": ["Delta", "Beta"], "group": "league_matches",
                      "diff_v2": 0.0, "diff_league_only": 0.0, "gd": [1, 1], "competition": "X",
                      "apif_fixture_id": [202, 101]})
    lk = cp.link(cp.wide(pd.DataFrame(rows)), d)
    got = dict(zip(lk["market_id"], lk["apif_fixture_id"]))
    assert got == {"1.1": 101, "1.2": 202}
