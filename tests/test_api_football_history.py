"""The API-Football history backfill: minutes from events, resuming, the quota floor and the tables, on synthetic
replies only (no figure here is a reported one)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from sources import api_football_history as h
from sources.common import Store


def xi(base):
    return [{"player": {"id": base + i, "name": f"p{base + i}", "pos": "M"}} for i in range(11)]


def ev(minute, kind, detail, a, b=None, team=1):
    return {"time": {"elapsed": minute, "extra": None}, "team": {"id": team}, "player": {"id": a},
            "assist": {"id": b}, "type": kind, "detail": detail}


def test_minutes_from_events_reads_who_was_on_the_pitch():
    lu = [{"team": {"id": 1}, "startXI": xi(0), "substitutes": [{"player": {"id": 50}}, {"player": {"id": 51}}]},
          {"team": {"id": 2}, "startXI": xi(100), "substitutes": []}]
    events = [ev(60, "subst", "Substitution 1", 3, 50),          # player off, assist on (today's order)
              ev(70, "subst", "Substitution 2", 51, 4),          # the other order: still resolved
              ev(80, "Card", "Red Card", 100, team=2),
              ev(85, "subst", "Substitution 3", 998, 999)]       # neither on the pitch: unresolved
    mins, unresolved = h.minutes_from_events(lu, events)
    assert mins[3] == 60 and mins[50] == 30 and mins[4] == 70 and mins[51] == 20
    assert mins[100] == 80 and mins[0] == 90 and unresolved == 1


def season_listing(ids, status="FT"):
    return {"errors": [], "response": [{"fixture": {"id": i, "status": {"short": status}}} for i in ids]}


def full(fid):
    return {"fixture": {"id": fid, "timestamp": 1_700_000_000, "status": {"short": "FT"}},
            "league": {"id": 39, "season": 2025}, "teams": {"home": {"id": 1, "name": "H"}, "away": {"id": 2, "name": "A"}},
            "score": {"fulltime": {"home": 1, "away": 0}},
            "lineups": [{"team": {"id": 1}, "startXI": xi(0), "substitutes": []},
                        {"team": {"id": 2}, "startXI": xi(100), "substitutes": []}],
            "events": [ev(30, "Goal", "Normal Goal", 5)], "players": []}


class Fake:
    def __init__(self, n=45, quota=7000):
        self.n, self.quota, self.calls = n, quota, []

    def __call__(self, path, params):
        self.calls.append(dict(params))
        self.quota -= 1
        if "league" in params:
            return season_listing(range(1, self.n + 1)) if params["season"] == 2025 else season_listing([]), self.quota
        ids = [int(x) for x in params["ids"].split("-")]
        return {"errors": [], "response": [full(i) for i in ids]}, self.quota


def test_fetch_batches_resumes_and_builds(tmp_path):
    st = Store(root=tmp_path)
    f = Fake()
    out = h.fetch(st, first_season=2025, fetcher=f, today=date(2025, 11, 2), leagues={39})
    assert out["matches_fetched"] == 45 and out["season_lists"] == 1
    assert [len(c["ids"].split("-")) for c in f.calls if "ids" in c] == [20, 20, 5]
    again = Fake()
    out2 = h.fetch(st, first_season=2025, fetcher=again, today=date(2025, 11, 2), leagues={39})
    assert out2["matches_fetched"] == 0 and not any("ids" in c for c in again.calls)   # all held: listing only
    rep = h.build(st)
    assert rep["matches"] == 45 and rep["with_lineups"] == 45 and rep["player_rows"] == 45 * 22
    assert rep["duplicate_player_ids"] == 0 and rep["event_rows"] == 45
    p = pd.read_parquet(tmp_path / "api_football/history_players.parquet")
    assert set(p["minutes_events"]) == {90}


def test_a_past_season_list_is_not_fetched_twice(tmp_path):
    st = Store(root=tmp_path)
    h.fetch(st, first_season=2024, fetcher=Fake(n=3), today=date(2026, 11, 2), leagues={39})
    f = Fake(n=3)
    h.fetch(st, first_season=2024, fetcher=f, today=date(2026, 11, 2), leagues={39})
    assert [c["season"] for c in f.calls if "league" in c] == [2026]                 # the current season only


def test_stops_at_the_quota_floor(tmp_path):
    f = Fake(n=200, quota=1504)
    out = h.fetch(Store(root=tmp_path), first_season=2025, fetcher=f, today=date(2025, 11, 2), leagues={39},
                  quota_floor=1500)
    assert out["stopped"] == "quota at 1500" and out["requests"] == 4


def test_no_key_fetches_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    assert "no API_FOOTBALL_KEY" in h.fetch(Store(root=tmp_path))["error"]
