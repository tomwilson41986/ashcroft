"""The API-Football line-up collector: polling, timing and the tables, on synthetic replies only (no figure here is
a reported one)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from sources import api_football_lineups as al
from sources.common import Store

KO = datetime(2026, 11, 7, 15, 0, tzinfo=timezone.utc)
LEAGUE = 39


def team(tid, name, n_xi=11, base=0):
    return {"team": {"id": tid, "name": name}, "formation": "4-4-2", "coach": {"id": 1, "name": "c"},
            "startXI": [{"player": {"id": base + i, "name": f"p{base + i}", "number": i, "pos": "M",
                                    "grid": "2:1"}} for i in range(n_xi)],
            "substitutes": [{"player": {"id": base + 50, "name": "s", "number": 50, "pos": "D", "grid": None}}]}


def fixture(fid, ko=KO, status="NS", lineups=None, league=LEAGUE):
    return {"fixture": {"id": fid, "timestamp": int(ko.timestamp()), "status": {"short": status}},
            "league": {"id": league, "season": 2026}, "teams": {"home": {"id": 10 * fid, "name": f"H{fid}"},
                                                                "away": {"id": 10 * fid + 1, "name": f"A{fid}"}},
            "lineups": lineups or []}


class Api:
    """A fake API-Football: the day's fixtures, and line-ups that appear at ``out_at``."""

    def __init__(self, clock, n=3, out_at=KO - timedelta(minutes=60), quota=7000, league=LEAGUE):
        self.clock, self.n, self.out_at, self.quota, self.league = clock, n, out_at, quota, league
        self.calls = []

    def __call__(self, path, params):
        self.calls.append((path, dict(params)))
        self.quota -= 1
        if path == "injuries":
            body = {"errors": [], "results": 1, "response": [
                {"player": {"id": 7, "name": "x", "type": "Missing Fixture", "reason": "Knee Injury"},
                 "team": {"id": 10, "name": "H1"}, "fixture": {"id": 1}, "league": {"id": self.league}},
                {"player": {"id": 8, "name": "y", "type": "Questionable", "reason": "Illness"},
                 "team": {"id": 99, "name": "Z"}, "fixture": {"id": 999}, "league": {"id": 1}}]}
        elif "date" in params:
            body = {"errors": [], "response": [fixture(i + 1, league=self.league) for i in range(self.n)]
                    + [fixture(500, league=1)]}
        else:
            ids = [int(x) for x in params["ids"].split("-")]
            lu = [team(1, "h"), team(2, "a", base=100)] if self.clock() >= self.out_at else []
            body = {"errors": [], "response": [fixture(i, lineups=lu, league=self.league) for i in ids]}
        return body, self.quota


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def collector(tmp_path, clock, api, **kw):
    return al.Collector(Store(root=tmp_path), api, "2026-11-07", {LEAGUE}, now=clock, **kw)


def test_only_the_divisions_matches_are_tracked(tmp_path):
    clock = Clock(KO - timedelta(hours=3))
    c = collector(tmp_path, clock, Api(clock))
    assert c.refresh_fixtures() == 3 and 500 not in c.matches


def test_polling_starts_at_the_lead_and_records_the_first_sight(tmp_path):
    clock = Clock(KO - timedelta(minutes=90))
    api = Api(clock)
    c = collector(tmp_path, clock, api)
    c.refresh_fixtures()
    assert c.due() == []                                        # 90 minutes out: before the lead
    clock.t = KO - timedelta(minutes=70)
    assert c.due() == [1, 2, 3]
    assert c.poll(c.due()) == 0                                 # not out yet
    clock.t += timedelta(minutes=1)
    assert c.due() == []                                        # polled a minute ago
    clock.t = KO - timedelta(minutes=59)
    assert c.poll(c.due()) == 3
    assert all(m.seen_at == KO - timedelta(minutes=59) for m in c.matches.values())
    clock.t = KO - timedelta(minutes=30)
    assert c.due() == []                                        # seen; the confirming poll waits for T-10
    clock.t = KO - timedelta(minutes=9)
    assert c.due() == [1, 2, 3]
    c.poll(c.due())
    assert all(m.confirmed for m in c.matches.values())
    clock.t = KO - timedelta(minutes=5)
    assert c.due() == []
    clock.t = KO + timedelta(minutes=1)
    assert c.due() == []                                        # started


def test_batches_of_twenty(tmp_path):
    clock = Clock(KO - timedelta(minutes=70))
    api = Api(clock, n=45)
    c = collector(tmp_path, clock, api)
    c.refresh_fixtures()
    c.poll(c.due())
    sizes = [len(p["ids"].split("-")) for path, p in api.calls if "ids" in p]
    assert sizes == [20, 20, 5]


def test_stops_at_the_quota_floor(tmp_path):
    clock = Clock(KO - timedelta(minutes=70))
    api = Api(clock, n=45, quota=102)
    c = collector(tmp_path, clock, api)
    c.refresh_fixtures()                                        # quota now 101
    c.poll(c.due())                                             # one batch (100 left), then it stops
    assert c.stopped == "quota at 100" and len([1 for _, p in api.calls if "ids" in p]) == 1


def test_build_times_players_and_injuries(tmp_path):
    clock = Clock(KO - timedelta(minutes=70))
    api = Api(clock, n=2)
    c = collector(tmp_path, clock, api)
    c.refresh_fixtures()
    c.injuries()
    c.poll(c.due())                                             # T-70: nothing
    clock.t = KO - timedelta(minutes=58)
    c.poll(c.due())                                             # T-58: both out
    clock.t = KO - timedelta(minutes=8)
    c.poll(c.due())                                             # T-8: confirmed
    rep = al.build(Store(root=tmp_path), {LEAGUE})
    assert rep["matches"] == 2 and rep["with_both_elevens"] == 2 and rep["xi_changed"] == 0
    assert rep["player_rows"] == 2 * 2 * 12 and rep["duplicate_player_rows"] == 0
    assert rep["sides_without_eleven_starters"] == 0
    assert rep["injury_rows_in"] == 2 and rep["injury_rows_out"] == 1          # the other league's dropped
    t = pd.read_parquet(tmp_path / "api_football/lineup_times.parquet")
    r = t[t.apif_fixture_id == 1].iloc[0]
    assert r.first_seen_utc == pd.Timestamp(KO - timedelta(minutes=58))
    assert r.last_unseen_utc == pd.Timestamp(KO - timedelta(minutes=70))
    assert r.minutes_before_ko == 58 and r.polls == 3
    inj = pd.read_parquet(tmp_path / "api_football/injuries.parquet")
    assert inj.iloc[0]["reason"] == "Knee Injury"


def test_collect_loop_ends_at_until(tmp_path):
    clock = Clock(datetime(2026, 11, 7, 13, 0, tzinfo=timezone.utc))
    api = Api(clock)

    def sleep(s):
        clock.t += timedelta(seconds=s)
    out = al.collect(Store(root=tmp_path), api, "15:05", leagues={LEAGUE}, tick_s=60, now=clock, sleep=sleep)
    assert out["lineups_seen"] == 3 and out["confirmed"] == 3 and out["stopped"] is None
    # a polling day, not a flood: fixtures hourly, injuries 3-hourly, line-ups every 2 minutes until out
    assert out["calls"] < 40


def test_no_key_collects_nothing(monkeypatch, capsys):
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    assert al.main(["--collect"]) == 0
    assert "no API_FOOTBALL_KEY" in capsys.readouterr().out


def test_a_rate_limit_reply_does_not_end_the_day(tmp_path):
    clock = Clock(KO - timedelta(minutes=70))
    api = Api(clock)
    c = collector(tmp_path, clock, api)
    c.refresh_fixtures()
    real = api.__call__
    state = {"n": 0}

    def limited(path, params):
        state["n"] += 1
        if state["n"] == 1:
            return {"errors": {"rateLimit": "Too many requests."}, "response": []}, 7000
        return real(path, params)
    c.fetch = limited
    c.poll(c.due())                                             # the limited reply: nothing seen, not stopped
    assert c.stopped is None and c.rate_limited == 1
    clock.t = KO - timedelta(minutes=58)
    assert c.poll(c.due()) == 3                                 # and the next poll works
