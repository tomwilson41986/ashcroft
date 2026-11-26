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
        full = clock["t"] >= t0 + timedelta(minutes=3)
        lu = ([{"type_id": 11, "team_id": 1}] * 11 + [{"type_id": 11, "team_id": 2}] * 11) if full else []
        return R({"data": [{"id": 7, "lineups": lu}]})
    import sources.common
    monkeypatch.setattr(sources.common, "get", get)

    def sleep(s):
        clock["t"] += timedelta(seconds=s)
    rows, meta = st.watch(None, t0 + timedelta(minutes=6), now=lambda: clock["t"], sleep=sleep)
    assert rows[7]["minutes_before_ko"] == 42.0 and rows[7]["polls"] == 4      # seen at T+3; not polled again


def test_pairing_by_kickoff_and_names():
    sm = pd.DataFrame([{"kickoff_utc": "2026-11-26 17:45:00+00:00", "home": "Ajax", "away": "FC Thun",
                        "minutes_before_ko": 70.0}])
    af = pd.DataFrame([{"kick": pd.Timestamp("2026-11-26 17:45", tz="UTC"), "home": "Ajax", "away": "Thun",
                        "minutes_before_ko": 28.0}])
    p = st.pair(sm, af)
    assert len(p) == 1 and p.iloc[0]["api_football_min"] == 28.0
