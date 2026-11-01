"""sources/parse_odds.py: the client's credit floor and retries, the raw store, the tables and the panels.

The payloads are hand-made in the shapes the three Parse APIs returned on 1 Nov 2026; they test the mechanics only."""
import gzip
import json

import pandas as pd
import pytest
import requests

import sources.parse_odds as po
from sources.common import Store


class Resp:
    def __init__(self, body, status=200, charged=1, remaining=1000):
        self.status_code = status
        self._body = body
        self.content = json.dumps(body).encode()
        self.text = self.content.decode()
        self.headers = {"x-credits-charged": str(charged), "x-credits-remaining": str(remaining)}

    def json(self):
        return self._body


class FakeHTTP:
    """Answers by endpoint name; records the calls."""

    def __init__(self, answers, fail_first=0):
        self.answers = answers
        self.calls = []
        self.fail_first = fail_first

    def _answer(self, url, payload):
        self.calls.append((url, payload))
        if self.fail_first:
            self.fail_first -= 1
            raise requests.ConnectionError("dropped")
        ep = url.rsplit("/", 1)[-1]
        a = self.answers[ep]
        return a(payload) if callable(a) else a

    def post(self, url, json=None, headers=None, timeout=None):
        assert headers["X-API-Key"] == "k"
        return self._answer(url, json)

    def get(self, url, params=None, headers=None, timeout=None):
        assert headers["X-API-Key"] == "k"
        return self._answer(url, params)


def ok(data, **kw):
    return Resp({"status": "success", "data": data}, **kw)


FEED = {"matches": [
    {"match_id": "AAAAAAAA", "status": "finished", "start_time": "2026-10-31T14:10:00+00:00",
     "home_team": {"name": "Home P.", "id": "h1"}, "away_team": {"name": "Away P.", "id": "a1"},
     "score": {"home": "2", "away": "0"},
     "competition": {"name": "ATP - SINGLES: Basel (Switzerland), hard (indoor)", "id": "c1", "country": "ATP - Singles"},
     "odds": {"home_win": "3.4", "draw": None, "away_win": "1.3", "home_win_opening": "2.9", "draw_opening": None,
              "away_win_opening": "1.4"}},
    {"match_id": "BBBBBBBB", "status": "finished", "start_time": "2026-10-31T15:00:00+00:00",
     "home_team": {"name": "X", "id": "x1"}, "away_team": {"name": "Y", "id": "y1"}, "score": {"home": "1", "away": "1"},
     "competition": {"name": "Premier League", "id": "c2", "country": "England"},
     "odds": {"home_win": "2.1", "draw": "3.6", "away_win": "3.2", "home_win_opening": "2.0", "draw_opening": "3.5",
              "away_win_opening": "3.5"}},
    {"match_id": "CCCCCCCC", "status": "finished", "start_time": "2026-10-31T16:00:00+00:00",
     "home_team": {"name": "No", "id": "n1"}, "away_team": {"name": "Odds", "id": "n2"}, "score": {"home": "0", "away": "1"},
     "competition": {"name": "ITF", "id": "c3", "country": "ITF"}, "odds": None}]}

MATCH_ODDS = {"match_id": "AAAAAAAA", "match_date": "2026-10-31T14:10:00+00:00", "home_team": "Home P.",
              "away_team": "Away P.", "competition": "ATP", "home_score": 2, "away_score": 0, "status": "finished",
              "markets": [{"market_type": "HOME_AWAY", "scope": "FULL_TIME", "bookmakers": [
                  {"bookmaker_id": 1, "bookmaker_name": "BookA", "selections": [
                      {"value": "3.5", "opening": "3.0", "selection": "home"},
                      {"value": "1.3", "opening": "1.4", "selection": "away"}]}]},
                  {"market_type": "OVER_UNDER", "scope": "FIRST_SET", "bookmakers": [
                      {"bookmaker_id": 1, "bookmaker_name": "BookA", "selections": [
                          {"value": "2.1", "opening": "2.1", "selection": "over", "handicap": "10.5"}]}]}]}


def test_prices():
    assert po.fractional("7/4") == 2.75
    assert po.fractional("2") == 3.0
    assert po.fractional("EVS") == 2.0
    assert po.fractional("x") is None
    assert po.american(-133) == pytest.approx(1 + 100 / 133)
    assert po.american(104) == pytest.approx(2.04)
    assert po.american(50) is None


def test_client_floor_retry_and_key(monkeypatch):
    monkeypatch.delenv("PARSE_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        po.Parse()
    http = FakeHTTP({"get_sports": ok({"items": []}, charged=2, remaining=1500)}, fail_first=1)
    monkeypatch.setattr(po.time, "sleep", lambda s: None)
    p = po.Parse(key="k", floor=1000, http=http)
    got = p.call("pinnacle", "get_sports")
    assert got["status_code"] == 200 and p.spent == 2 and p.remaining == 1500 and len(http.calls) == 2
    p.floor = 2000
    with pytest.raises(po.OutOfCredits):
        p.call("pinnacle", "get_sports")


def test_feed_tables_panel_and_skip(tmp_path, monkeypatch):
    monkeypatch.setattr(po.time, "sleep", lambda s: None)
    store = Store(root=tmp_path)
    http = FakeHTTP({"get_fixtures_with_odds": ok(FEED), "get_match_odds": ok(MATCH_ODDS)})
    p = po.Parse(key="k", floor=0, http=http)
    po.flashscore_days(store, p, ["tennis"], [-1])
    n_calls = len(http.calls)
    po.flashscore_days(store, p, ["tennis"], [-1])           # a finished day held: not fetched again
    assert len(http.calls) == n_calls
    keys = list(store.listing("parse/raw/flashscore/get_fixtures_with_odds/"))
    env = json.loads(gzip.decompress(store.get(keys[0])))
    assert env["params"] == {"sport": "tennis", "day_offset": -1} and env["credits"] == 1 and "k" not in json.dumps(env)

    mt = po.table_flashscore_matches(store)
    assert len(mt) == 3 and mt.set_index("match_id").loc["AAAAAAAA", "home_open"] == 2.9

    assert po.flashscore_match_odds(store, p, ["AAAAAAAA", "AAAAAAAA"], workers=1) == 1
    assert po.flashscore_match_odds(store, p, ["AAAAAAAA"], workers=1) == 0   # held
    od = po.table_flashscore_odds(store)
    assert len(od) == 3 and set(od["market_type"]) == {"HOME_AWAY", "OVER_UNDER"}

    pan = po.panel_flashscore(store)
    feed = pan[pan["source"] == "flashscore_feed"]
    # the tennis match (2 selections) and the football match (3); the match with no odds is left out
    assert len(feed) == 5
    t = feed[feed["event_id"] == "AAAAAAAA"].set_index("selection_id")
    assert t.loc["home", "open_price"] == 2.9 and t.loc["home", "close_price"] == 3.4 and t.loc["home", "result"] == 1
    assert t.loc["away", "result"] == 0
    f = feed[feed["event_id"] == "BBBBBBBB"].set_index("selection_id")
    assert f.loc["draw", "result"] == 1 and f.loc["home", "result"] == 0
    book = pan[pan["source"] == "flashscore:BookA"]
    assert len(book) == 2 and set(book["sport"]) == {"tennis"}
    assert book.set_index("selection_id").loc["home", "open_price"] == 3.0

    counts = po.build(store)
    assert counts["flashscore_matches"] == 3 and counts["flashscore_odds"] == 3
    assert store.get_parquet("parse/flashscore_matches.parquet") is not None


def test_pinnacle_table_and_panel(tmp_path, monkeypatch):
    monkeypatch.setattr(po.time, "sleep", lambda s: None)
    store = Store(root=tmp_path)
    mu = ok({"items": [{"id": 7, "startTime": "2099-01-01T12:00:00Z", "isLive": False,
                        "league": {"id": 3, "name": "ATP Basel"},
                        "participants": [{"alignment": "home", "name": "A"}, {"alignment": "away", "name": "B"}]}]})
    prices = iter([(-150, 130), (-170, 150)])

    def odds(_):
        h, a = next(prices)
        return ok({"items": [{"matchupId": 7, "type": "moneyline", "period": 0, "isAlternate": False, "status": "open",
                              "key": "s;0;m", "version": 1, "cutoffAt": "2099-01-01T12:00:00Z",
                              "limits": [{"type": "maxRiskStake", "amount": 250}],
                              "prices": [{"designation": "home", "price": h}, {"designation": "away", "price": a}]}]},
                  charged=2)

    http = FakeHTTP({"get_matchups": mu, "get_odds": odds})
    p = po.Parse(key="k", floor=0, http=http)
    from datetime import datetime, timedelta, timezone
    clock = iter(datetime(2026, 11, 1, 12, tzinfo=timezone.utc) + timedelta(hours=h) for h in range(10))
    monkeypatch.setattr(po, "_now", lambda: next(clock))     # each snapshot under its own time
    po.pinnacle_snapshot(store, p, [33])
    po.pinnacle_snapshot(store, p, [33])
    t = po.table_pinnacle(store)
    assert len(t) == 4 and set(t["home"]) == {"A"} and t["max_risk"].iloc[0] == 250
    pan = po.panel_pinnacle(store).set_index("selection_id")
    assert pan.loc["home", "open_price"] == pytest.approx(1 + 100 / 150)
    assert pan.loc["home", "close_price"] == pytest.approx(1 + 100 / 170)
    assert pan.loc["away", "close_price"] == pytest.approx(2.5) and pan.loc["home", "n_runners"] == 2


def test_oddschecker_races_and_table(tmp_path, monkeypatch):
    monkeypatch.setattr(po.time, "sleep", lambda s: None)
    store = Store(root=tmp_path)
    meetings = ok({"meetings": [
        {"venue": "Carlisle", "races": [{"time": "23:59", "url": "https://www.oddschecker.com/horse-racing/carlisle/23:59/winner"}]},
        {"venue": "Gavea", "races": [{"time": "23:58", "url": "https://www.oddschecker.com/horse-racing/gavea/23:58/winner"}]}]})
    race = ok({"runners": [{"name": "Horse A", "odds": {"B3": "7/4", "BF": "19/10"}},
                           {"name": "Horse B", "odds": {"B3": "EVS"}}]}, charged=3)
    http = FakeHTTP({"get_horse_racing_meetings": meetings, "get_horse_race_odds": race})
    p = po.Parse(key="k", floor=0, http=http)
    races = po.oddschecker_races(store, p)
    assert [r["venue"] for r in races] == ["Carlisle"]       # GB/IE only
    po.oddschecker_snapshot(store, p, races, "first")
    t = po.table_oddschecker(store)
    assert len(t) == 3 and set(t["course"]) == {"carlisle"} and t["race_time_uk"].iloc[0] == "23:59"
    assert t.set_index(["horse", "bookmaker"]).loc[("Horse A", "BF"), "price"] == pytest.approx(2.9)
