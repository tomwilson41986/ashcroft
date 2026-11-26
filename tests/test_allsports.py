"""allsports/: the all-sports census (every market with its start time, every call under Betfair's 200 weight points,
a full window split until none is) and the stream recorder of the main markets (at most 190 markets a connection, at
most the connections asked for, the overflow left out and counted, never the order stream). The catalogues here are
made up for the tests; nothing is read as data."""
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from allsports import census, reader as rd, stream as st

T0 = datetime(2026, 11, 28, 12, 0, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[1]


def _m(i, start, mtype="MATCH_ODDS", country="GB", comp="1", et="1"):
    return {"marketId": f"1.{i}", "marketName": mtype, "marketStartTime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "eventType": {"id": et, "name": "Soccer"}, "event": {"id": str(i), "name": "A v B", "countryCode": country},
            "competition": {"id": comp, "name": f"C{comp}"}, "totalMatched": 0.0,
            "description": {"marketType": mtype, "turnInPlayEnabled": True, "bspMarket": False}}


class FakeAPI:
    """Betfair's listing calls over a made-up catalogue: at most maxResults markets an answer, as Betfair does."""

    def __init__(self, markets):
        self.markets, self.calls = markets, []

    def _match(self, flt):
        out = []
        for m in self.markets:
            t = m["marketStartTime"]
            win = flt.get("marketStartTime") or {}
            if win and not (win["from"] <= t <= win["to"]):
                continue
            if "marketTypeCodes" in flt and m["description"]["marketType"] not in flt["marketTypeCodes"]:
                continue
            if "marketCountries" in flt and m["event"]["countryCode"] not in flt["marketCountries"]:
                continue
            if "competitionIds" in flt and m["competition"]["id"] not in flt["competitionIds"]:
                continue
            if "eventTypeIds" in flt and m["eventType"]["id"] not in flt["eventTypeIds"]:
                continue
            out.append(m)
        return out

    def read(self, method, params):
        self.calls.append((method, params))
        got = self._match(params["filter"])
        if method == "listMarketCatalogue":
            assert params["maxResults"] <= 200
            return sorted(got, key=lambda m: m["marketStartTime"])[:params["maxResults"]]
        if method == "listMarketTypes":
            return [{"marketType": t} for t in sorted({m["description"]["marketType"] for m in got})]
        if method == "listCountries":
            return [{"countryCode": c} for c in sorted({m["event"]["countryCode"] for m in got})]
        if method == "listCompetitions":
            return [{"competition": {"id": c}} for c in sorted({m["competition"]["id"] for m in got})]
        raise AssertionError(method)


def test_the_census_reads_every_market_even_when_hundreds_start_at_once():
    kick_off = T0 + timedelta(hours=3)
    mk = []
    # a Saturday 15:00: 3 market types x 2 countries x 3 competitions x 60 matches at one instant (1,080 markets)
    i = 0
    for t in ("MATCH_ODDS", "OVER_UNDER_25", "CORRECT_SCORE"):
        for c in ("GB", "ES"):
            for comp in ("1", "2", "3"):
                for _ in range(60):
                    mk.append(_m(i, kick_off, t, c, comp))
                    i += 1
    mk += [_m(i + k, T0 + timedelta(minutes=k)) for k in range(300)]        # and a spread of others
    api = FakeAPI(mk)
    rows, reading = census.read_markets(api, T0, T0 + timedelta(hours=24))
    assert len(rows) == len(mk) and len({r["market_id"] for r in rows}) == len(mk)
    assert reading["truncated_windows"] == 0 and not reading["call_budget_ran_out"]
    assert reading["splits"]["time"] > 0 and reading["splits"]["marketTypeCodes"] == 1
    assert reading["splits"]["marketCountries"] == 3                       # 360 a type: still full, so by country
    assert "competitionIds" not in reading["splits"]                       # 180 a country: answered whole
    assert all(m in rd.READS for m, _ in api.calls)
    assert rows[0]["start_utc"] <= rows[-1]["start_utc"] and rows[0]["sport"] == "Soccer"


def test_a_window_nothing_can_split_is_counted_as_truncated():
    api = FakeAPI([_m(i, T0) for i in range(250)])                          # one type, country, competition
    rows, reading = census.read_markets(api, T0, T0 + timedelta(hours=1))
    assert len(rows) == 200 and reading["truncated_windows"] == 1


def test_the_call_budget_stops_the_reading_and_says_so():
    api = FakeAPI([_m(i, T0 + timedelta(seconds=i)) for i in range(1000)])
    _, reading = census.read_markets(api, T0, T0 + timedelta(hours=1), max_calls=3)
    assert reading["call_budget_ran_out"] and reading["calls"] == 3


def test_the_peak_counts_markets_open_at_once_with_the_assumed_close():
    rows = [census.row(_m(1, T0)), census.row(_m(2, T0 + timedelta(minutes=60))),
            census.row(_m(3, T0 + timedelta(hours=10)))]
    p = census.peak_open(rows, lead_minutes=120)
    assert p["peak_open"] == 2 and p["connections_at_190"] == 1           # football open 130 min after the off
    not_in_play = {**rows[1], "in_play_enabled": False}
    assert census.live_minutes(not_in_play) == census.NOT_IN_PLAY_MINUTES
    s = census.sizing(rows + [census.row(_m(4, T0, "CORRECT_SCORE"))])
    assert s["all"]["markets"] == 4 and s["core"]["markets"] == 3 and "assumed" in s["assumption"]
    assert sum(s["starts_by_hour_utc"].values()) == 4


def test_the_reader_lists_only_and_paces_its_calls():
    class Client:
        app_key, session_token = "K", "T"

        def __init__(self):
            self.n = 0

        def login(self):
            self.n += 1

        def _api_call(self, method, params):
            return []

    t = [0.0]
    slept = []
    r = rd.Reader(Client(), per_second=5.0, clock=lambda: t[0], sleep=lambda s: (slept.append(s), t.__setitem__(0, t[0] + s)))
    for m in ("placeOrders", "cancelOrders", "listCurrentOrders", "getAccountFunds", "listMarketBook"):
        with pytest.raises(ValueError):
            r.read(m, {})
    r.read("listMarketCatalogue", {})
    r.read("listMarketCatalogue", {})
    assert slept and abs(slept[0] - 0.2) < 1e-9 and r.calls == 2
    assert r.credentials() == ("K", "T") and r.client.n == 1                  # logs in on first use
    r.credentials()
    assert r.client.n == 1
    t[0] += rd.SESSION_HOURS * 3600 + 1
    r.credentials()
    assert r.client.n == 2                                                    # and again before the session lapses


def _rows(n, start, step_s=0, et="1"):
    return [census.row(_m(i, start + timedelta(seconds=i * step_s), et=et)) for i in range(n)]


def test_the_plan_never_puts_more_than_190_on_a_connection_and_counts_the_rest():
    rows = _rows(500, T0 + timedelta(minutes=30), step_s=1)
    p = st.Planner(lambda a, b: rows, shards=2)
    plan = p.plan(T0)
    assert all(len(v) <= st.SHARD_CAP for v in plan.values())
    assert sum(len(v) for v in plan.values()) == 2 * st.SHARD_CAP
    s = p.state()
    assert s["wanted"] == 500 and s["left_out"] == 500 - 2 * st.SHARD_CAP and s["per_connection"] == [190, 190]
    # the nearest to their start kept: the latest 120 left out
    kept = {m for v in plan.values() for m in v}
    assert "1.0" in kept and "1.499" not in kept


def test_the_plan_keeps_each_market_on_its_connection_and_frees_closed_ones():
    rows = _rows(10, T0 + timedelta(minutes=30))
    p = st.Planner(lambda a, b: rows, shards=3)
    first = p.plan(T0)
    p.mark_closed("1.0")
    again = p.plan(T0 + timedelta(minutes=5))
    where = {m: k for k, v in first.items() for m in v}
    assert all(where[m] == k for k, v in again.items() for m in v)
    assert "1.0" not in {m for v in again.values() for m in v}
    later = p.plan(T0 + timedelta(hours=6))                                   # long closed (130 min after the off)
    assert not any(later.values())


def test_the_plan_waits_for_the_lead_and_leaves_out_excluded_sports():
    rows = _rows(3, T0 + timedelta(hours=5)) + [census.row(_m(9, T0, et="7"))]
    p = st.Planner(lambda a, b: rows, shards=1, lead_minutes=120, exclude_event_types=["7"])
    assert p.plan(T0) == {0: []}
    assert len(p.plan(T0 + timedelta(hours=3, minutes=1))[0]) == 3


def test_a_closed_definition_frees_the_market():
    line = json.dumps({"op": "mcm", "mc": [{"id": "1.5", "marketDefinition": {"status": "CLOSED"}},
                                           {"id": "1.6", "marketDefinition": {"status": "OPEN"}}]})
    assert st.closed_markets(line) == ["1.5"]
    assert st.closed_markets(json.dumps({"op": "mcm", "ct": "HEARTBEAT"})) == []


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t


class FakeConn:
    def __init__(self, clock):
        self.clock, self.sent = clock, []
        self.q = [json.dumps({"op": "connection", "connectionId": "x"})]

    def send(self, msg):
        self.sent.append(msg)
        if msg["op"] == "authentication":
            self.q.append(json.dumps({"op": "status", "id": msg["id"], "statusCode": "SUCCESS"}))
        elif msg["op"] == "marketSubscription":
            self.q += [json.dumps({"op": "mcm", "id": msg["id"], "clk": str(i)}) for i in range(3)]

    def line(self):
        time.sleep(0.001)                                  # real time, so both connections' threads get turns
        self.clock.t += timedelta(seconds=1)
        return self.q.pop(0) if self.q else json.dumps({"op": "mcm", "ct": "HEARTBEAT"})

    def close(self):
        pass


def test_each_connection_records_its_own_share_under_its_own_name_and_never_orders(tmp_path):
    clock, conns, puts = Clock(), [], {}
    rows = _rows(300, T0 + timedelta(minutes=30), step_s=1)
    planner = st.Planner(lambda a, b: rows, shards=2)

    def connect():
        conns.append(FakeConn(clock))
        return conns[-1]

    writers = [st.ShardWriter(date(2026, 11, 28), k, put=lambda key, b: puts.update({key: b}), root=tmp_path,
                              clock=lambda: 0.0, prefix=st.PREFIX, keep_local=False) for k in range(2)]
    st.run_shards(date(2026, 11, 28), T0 + timedelta(minutes=5), planner, lambda: ("K", "T"), connect, writers,
                  now=clock, sleep=lambda s: None)
    for w in writers:
        w.maybe_roll(force=True)
    subs = [m for c in conns for m in c.sent if m["op"] == "marketSubscription"]
    assert subs and all(len(s["marketFilter"]["marketIds"]) <= st.SHARD_CAP for s in subs)
    assert {i for s in subs for i in s["marketFilter"]["marketIds"]} == {r["market_id"] for r in rows}
    assert all(s["conflateMs"] == st.CONFLATE_MS for s in subs)
    assert not any(m["op"] == "orderSubscription" for c in conns for m in c.sent)
    assert sorted(puts) == ["betfair_stream/sports/2026-11-28/s0-part-000.jsonl.gz",
                            "betfair_stream/sports/2026-11-28/s1-part-000.jsonl.gz"]


def test_the_package_has_no_route_to_an_order_and_never_touches_the_live_horse_system():
    for p in (ROOT / "allsports").glob("*.py"):
        src = p.read_text()
        for word in ("placeOrders", "cancelOrders", "replaceOrders", "updateOrders", "auto_trade", "LiveExchange",
                     "from trading", "import trading", "orderSubscription"):
            assert word not in src, (p.name, word)
    assert st.MAX_CONNECTIONS <= 10 and st.CONNECTIONS == 2 and st.SHARD_CAP < 200


def test_the_workflow_records_on_its_own_runner_by_hand_only():
    """sports-stream.yml: its own runner (never the horse, football, greyhound or data runner), started by hand until
    the owner has Betfair's limits for the key, the live key to this recorder alone, connections from STREAM_CONNECTIONS."""
    import yaml
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "sports-stream.yml").read_text())
    on = wf.get(True) or wf.get("on")
    assert set(on) == {"workflow_dispatch"}
    job = wf["jobs"]["record"]
    assert "STREAM_RUNNER" in job["runs-on"] and "'streams'" in job["runs-on"]
    guard = job["steps"][0]
    assert guard["name"] == "Not on another job's runner"
    for v in ("BETFAIR_RUNNER", "FOOTBALL_RUNNER", "PAPER_MODELS_RUNNER", "DATA_RUNNER", "MORNING_RUNNER"):
        assert v in str(guard["env"])
    code = [ln.strip() for s in job["steps"] for ln in str(s.get("run", "")).splitlines()
            if ln.strip() and not ln.strip().startswith("#")]
    py = [ln for ln in code if "python" in ln and "pip" not in ln and "venv" not in ln and "sys.version" not in ln]
    assert len(py) == 1 and "-m allsports.stream" in py[0] and '"$BETFAIR_LIVE_APP_KEY"' in py[0]
    assert "STREAM_CONNECTIONS" in str(job)
