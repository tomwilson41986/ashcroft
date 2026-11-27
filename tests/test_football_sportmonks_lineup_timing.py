"""research/open_close/football_sportmonks_lineup_timing.py on synthetic replies (no figure here is a reported one)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from research.open_close import football_sportmonks_lineup_timing as st


def test_starters_count_type_11_a_team():
    f = {"lineups": [{"type_id": 11, "team_id": 1}] * 11 + [{"type_id": 12, "team_id": 1}] * 7
         + [{"type_id": 11, "team_id": 2}] * 10}
    n, types = st.starters(f)
    assert n == {1: 11, 2: 10} and types == {11: 21, 12: 7}


def test_watch_keeps_the_first_poll_with_both_elevens(monkeypatch):
    t0 = datetime(2026, 11, 26, 17, 0, tzinfo=timezone.utc)
    ko = int((t0 + timedelta(minutes=45)).timestamp())
    clock = {"t": t0}
    monkeypatch.setattr(st, "discover", lambda s, day: {7: {"id": 7, "kickoff": ko, "league": "L", "home": "A",
                                                             "away": "B", "home_id": 1, "away_id": 2}})

    class R:
        def __init__(self, body):
            self.body = body

        def json(self):
            return self.body

    def get(s, url, params=None):
        m = (clock["t"] - t0).total_seconds() / 60
        if m < 3:
            lu = []
        else:                                                    # at T+5 one home starter is swapped
            home = list(range(1, 12)) if m < 5 else list(range(1, 11)) + [99]
            lu = ([{"type_id": 11, "team_id": 1, "player_id": p} for p in home]
                  + [{"type_id": 11, "team_id": 2, "player_id": 100 + p} for p in range(11)])
        return R({"data": [{"id": 7, "lineups": lu}]})
    import sources.common
    monkeypatch.setattr(sources.common, "get", get)

    def sleep(s):
        clock["t"] += timedelta(seconds=s)
    rows, meta = st.watch(None, t0 + timedelta(minutes=6), now=lambda: clock["t"], sleep=sleep)
    r = rows[7]
    assert r["minutes_before_ko"] == 42.0 and r["polls"] == 6        # first seen at T+3, polled on to the end
    assert r["changes"] == [(40.0, 1)] and r["last_change_min"] == 40.0
    assert r["first_xi_kept"] == round(21 / 22, 3)


def test_pairing_by_kickoff_and_names():
    sm = pd.DataFrame([{"kickoff_utc": "2026-11-26 17:45:00+00:00", "home": "Ajax", "away": "FC Thun",
                        "minutes_before_ko": 70.0}])
    af = pd.DataFrame([{"kick": pd.Timestamp("2026-11-26 17:45", tz="UTC"), "home": "Ajax", "away": "Thun",
                        "minutes_before_ko": 28.0}])
    p = st.pair(sm, af)
    assert len(p) == 1 and p.iloc[0]["api_football_min"] == 28.0


def test_a_checkpoint_after_each_polling_cycle_keeps_the_result(monkeypatch):
    t0 = datetime(2026, 11, 26, 17, 0, tzinfo=timezone.utc)
    ko = int((t0 + timedelta(minutes=45)).timestamp())
    clock = {"t": t0}
    monkeypatch.setattr(st, "discover", lambda s, day: {7: {"id": 7, "kickoff": ko, "league": "L", "home": "A",
                                                             "away": "B", "home_id": 1, "away_id": 2}})

    class R:
        def json(self):
            return {"data": [{"id": 7, "lineups": [{"type_id": 11, "team_id": 1}] * 11
                              + [{"type_id": 11, "team_id": 2}] * 11}]}
    import sources.common
    monkeypatch.setattr(sources.common, "get", lambda s, url, params=None: R())
    saved = []

    def sleep(s):
        clock["t"] += timedelta(seconds=s)
    st.watch(None, t0 + timedelta(minutes=3), now=lambda: clock["t"], sleep=sleep,
             checkpoint=lambda rows, meta: saved.append(st.summarise(rows, meta, "2026-11-26", False)))
    assert saved and saved[0]["with_both_elevens"] == 1 and saved[0]["sportmonks_minutes_before_ko"][0.5] == 45.0
