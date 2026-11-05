"""greyhound/stream_record.py: the market window, authentication then subscription, every line written with its
receive time, re-subscription as the window moves, a failure remade into a new connection, the parts uploaded, and
nothing that could place or cancel an order."""
import gzip
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from greyhound import stream_record as sr

T0 = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)


def test_the_window_takes_the_races_near_their_off_nearest_first():
    offs = {"a": T0 + timedelta(minutes=30), "b": T0 + timedelta(minutes=5), "c": T0 + timedelta(minutes=120),
            "d": T0 - timedelta(minutes=10), "e": T0 - timedelta(minutes=40)}
    assert sr.window(offs, T0) == ["d", "b", "a"]
    many = {f"m{i}": T0 + timedelta(seconds=i) for i in range(500)}
    assert len(sr.window(many, T0)) == sr.MAX_MARKETS


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t


class FakeConn:
    """A connection message, the reply to authentication, then market changes; each line read moves the clock."""

    def __init__(self, clock, auth="SUCCESS", lines=5, fail_after=None):
        self.clock, self.sent, self.closed = clock, [], False
        self.q = [json.dumps({"op": "connection", "connectionId": "x"})]
        self.auth, self.n, self.fail_after = auth, lines, fail_after

    def send(self, msg):
        self.sent.append(msg)
        if msg["op"] == "authentication":
            self.q.append(json.dumps({"op": "status", "id": msg["id"], "statusCode": self.auth}))
        elif msg["op"] == "marketSubscription":
            self.q += [json.dumps({"op": "mcm", "id": msg["id"], "clk": str(i), "mc": [{"id": "1.1"}]})
                       for i in range(self.n)]

    def line(self):
        self.clock.t += timedelta(seconds=30)
        if self.fail_after is not None and len([m for m in self.sent if m["op"] == "marketSubscription"]) and \
                self.fail_after <= 0:
            raise ConnectionError("dropped")
        if self.fail_after is not None:
            self.fail_after -= 1
        if not self.q:
            return json.dumps({"op": "mcm", "ct": "HEARTBEAT"})
        return self.q.pop(0)

    def close(self):
        self.closed = True


def test_authenticates_subscribes_and_writes_every_line(tmp_path):
    clock = Clock()
    conns = []

    def connect():
        conns.append(FakeConn(clock))
        return conns[-1]

    puts = {}
    w = sr.Writer(date(2026, 11, 5), put=lambda k, b: puts.update({k: b}), root=tmp_path, clock=lambda: 0.0)
    cat = [{"marketId": "1.1", "marketStartTime": (T0 + timedelta(minutes=20)).isoformat()},
           {"marketId": "1.2", "marketStartTime": (T0 + timedelta(minutes=200)).isoformat()}]
    out = sr.run(date(2026, 11, 5), T0 + timedelta(minutes=5), lambda: ("KEY", "TOKEN"), lambda: cat, connect, w,
                 now=clock, sleep=lambda s: None)
    c = conns[0]
    assert c.sent[0] == {"op": "authentication", "id": 1, "appKey": "KEY", "session": "TOKEN"}
    sub = c.sent[1]
    assert sub["op"] == "marketSubscription" and sub["marketFilter"] == {"marketIds": ["1.1"]}
    assert set(sub["marketDataFilter"]["fields"]) == set(sr.FIELDS)
    assert not any(m["op"] == "orderSubscription" for m in c.sent)          # never the order stream
    assert out["connects"] == 1 and out["lines"] == w.lines and c.closed
    assert list(puts) == ["betfair_stream/greyhound/2026-11-05/part-000.jsonl.gz"]
    rows = [json.loads(x) for x in gzip.decompress(puts["betfair_stream/greyhound/2026-11-05/part-000.jsonl.gz"])
            .decode().splitlines()]
    assert len(rows) == w.lines and json.loads(rows[0]["m"])["op"] == "connection" and rows[0]["rt"].endswith("Z")


def test_a_failed_login_or_a_dropped_connection_is_remade(tmp_path):
    clock = Clock()
    conns = []

    def connect():
        conns.append(FakeConn(clock, auth="FAILURE") if not conns else FakeConn(clock, fail_after=4))
        return conns[-1]

    w = sr.Writer(date(2026, 11, 5), put=None, root=tmp_path, clock=lambda: 0.0)
    cat = [{"marketId": "1.1", "marketStartTime": (T0 + timedelta(minutes=20)).isoformat()}]
    out = sr.run(date(2026, 11, 5), T0 + timedelta(minutes=6), lambda: ("K", "T"), lambda: cat, connect, w,
                 now=clock, sleep=lambda s: None)
    assert out["auth_failures"] == 1 and out["errors"] >= 2 and out["connects"] >= 3
    assert all(c.closed for c in conns)


def test_resubscribes_as_the_window_moves(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "RESUB_SECONDS", 60.0)
    clock = Clock()
    conns = []

    def connect():
        conns.append(FakeConn(clock, lines=0))
        return conns[-1]

    w = sr.Writer(date(2026, 11, 5), put=None, root=tmp_path, clock=lambda: 0.0)
    cat = [{"marketId": "1.1", "marketStartTime": (T0 + timedelta(minutes=20)).isoformat()},
           {"marketId": "1.2", "marketStartTime": (T0 + timedelta(minutes=80)).isoformat()}]
    sr.run(date(2026, 11, 5), T0 + timedelta(minutes=12), lambda: ("K", "T"), lambda: cat, connect, w, now=clock,
           sleep=lambda s: None)
    subs = [m["marketFilter"]["marketIds"] for m in conns[0].sent if m["op"] == "marketSubscription"]
    assert subs[0] == ["1.1"] and ["1.1", "1.2"] in subs


def test_a_restart_never_rewrites_a_part(tmp_path):
    (tmp_path / "part-000.jsonl.gz").write_bytes(gzip.compress(b"x\n"))
    w = sr.Writer(date(2026, 11, 5), put=None, root=tmp_path)
    assert w.part == 1
