"""research/horse_stream_record.py: the greyhounds' stream recorder pointed at the horse WIN markets: every market not
yet off, conflated, the parts under their own prefix and off the server once uploaded, never the order stream, and
started only beside the horse trader, to 10:30 UK (the owner, 10 Nov 2026)."""
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from greyhound import stream_record as sr
from research import horse_stream_record as hs

T0 = datetime(2026, 11, 10, 6, 0, tzinfo=timezone.utc)


def test_every_market_not_yet_off_nearest_first():
    offs = {"a": T0 + timedelta(hours=7), "b": T0 + timedelta(minutes=5), "c": T0 + timedelta(hours=12),
            "d": T0 - timedelta(minutes=10), "e": T0 - timedelta(minutes=40)}
    assert hs.not_yet_off(offs, T0) == ["d", "b", "a", "c"]          # the greyhounds' window would drop a and c
    many = {f"m{i}": T0 + timedelta(seconds=i) for i in range(500)}
    assert len(hs.not_yet_off(many, T0)) == hs.MAX_MARKETS


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
        self.clock.t += timedelta(seconds=30)
        return self.q.pop(0) if self.q else json.dumps({"op": "mcm", "ct": "HEARTBEAT"})

    def close(self):
        pass


def test_subscribes_to_the_whole_day_conflated_and_never_to_orders(tmp_path):
    clock, conns, puts = Clock(), [], {}

    def connect():
        conns.append(FakeConn(clock))
        return conns[-1]

    w = sr.Writer(date(2026, 11, 10), put=lambda k, b: puts.update({k: b}), root=tmp_path, clock=lambda: 0.0,
                  prefix=hs.PREFIX, keep_local=False)
    cat = [{"marketId": "1.1", "marketStartTime": (T0 + timedelta(hours=6)).isoformat()},
           {"marketId": "1.2", "marketStartTime": (T0 + timedelta(hours=9)).isoformat()}]
    out = sr.run(date(2026, 11, 10), T0 + timedelta(minutes=5), lambda: ("KEY", "TOKEN"), lambda: cat, connect, w,
                 now=clock, sleep=lambda s: None, pick=hs.not_yet_off, subscribe=hs.subscription())
    sub = [m for m in conns[0].sent if m["op"] == "marketSubscription"][0]
    assert sub["marketFilter"] == {"marketIds": ["1.1", "1.2"]} and sub["conflateMs"] == hs.CONFLATE_MS
    assert set(sub["marketDataFilter"]["fields"]) == set(sr.FIELDS)
    assert not any(m["op"] == "orderSubscription" for c in conns for m in c.sent)
    assert out["lines"] > 0 and list(puts) == ["betfair_stream/horse/2026-11-10/part-000.jsonl.gz"]
    assert not list(tmp_path.glob("part-*.jsonl.gz"))               # uploaded, so off the server's disk


def test_the_greyhounds_recorder_is_unchanged_by_default(tmp_path):
    assert sr.subscription(["1.1"], 1)["conflateMs"] == 0
    w = sr.Writer(date(2026, 11, 10), put=lambda k, b: None, root=tmp_path, clock=lambda: 0.0)
    assert w.prefix == sr.PREFIX and w.keep_local


def test_only_the_trading_job_starts_it_beside_the_trader_to_10_30():
    """The owner, 10 Nov 2026: beside the horse trader in live-trade.yml, the morning the rule trades, on the live key
    (no delayed key opens the stream), backgrounded under a timeout so it can never hold up or stop the trader, and
    only when the live key's secret exists. No other workflow starts it."""
    wf = Path(__file__).resolve().parents[1] / ".github" / "workflows"
    starts = {p.name for p in wf.glob("*.yml") if "horse_stream_record" in p.read_text()}
    assert starts == {"live-trade.yml"}, starts
    text = (wf / "live-trade.yml").read_text()
    code = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    line = [ln for ln in code if "research.horse_stream_record" in ln]
    assert len(line) == 1
    ln = line[0]
    assert ln.startswith('BETFAIR_APP_KEY="$BETFAIR_LIVE_APP_KEY" timeout ') and "--until 10:30" in ln
    assert ln.endswith("&") and "auto_trade" not in ln and "echo" not in ln
    i = code.index(ln)
    assert code[i - 1] == 'if [ -n "$BETFAIR_LIVE_APP_KEY" ]; then' and code[i + 1] == "STREAM=$!"
    # in the trade branch, started before the trader and waited on after it
    assert code.index(ln) < next(k for k, c in enumerate(code) if "--trade-until" in c and "auto_trade.py" in c)
    assert 'wait "$STREAM"; echo "horse stream record exited $?"' in code


def test_the_recorder_has_no_route_to_an_order():
    """Read-only: neither module names an order call, and the stream is never asked for the order subscription."""
    root = Path(__file__).resolve().parents[1]
    for rel in ("research/horse_stream_record.py", "greyhound/stream_record.py"):
        src = (root / rel).read_text()
        for word in ("placeOrders", "cancelOrders", "replaceOrders", "updateOrders", "auto_trade", "LiveExchange"):
            assert word not in src, (rel, word)
        assert '"orderSubscription"' not in src and "'orderSubscription'" not in src, rel
