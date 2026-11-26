"""Betfair's Exchange Stream for the main market of every event in every sport, recorded raw (the owner, 26 Nov 2026:
"record prices in depth ... for all sports on betfair", "stay under betfair limits").

Read-only: it logs in, subscribes to market data and writes what arrives; the stream's order subscription is never
sent and no order call exists in allsports/. The connection, the subscription and the raw writer are the greyhound
recorder's (greyhound/stream_record.py), each line kept as received: the full ladder (EX_ALL_OFFERS), the volume
traded at each price (EX_TRADED), the last traded price, the market's definition and state.

Under Betfair's limits:
- each connection subscribes to at most ``SHARD_CAP`` (190) markets, under the default subscription limit of 200;
- at most ``--connections`` connections (default 2, the repository variable STREAM_CONNECTIONS), so the account's
  other streams (the horse and greyhound recorders) keep room: raise it only once Betfair confirms the key's limits;
- the markets wanted at once over the connections' total are left out, the furthest from their start first, and
  counted in the status (``left_out``), never subscribed;
- the catalogue is read with the paced listing reader (allsports/reader.py: 5 calls a second, 200 weight points a call).

Which markets: ``--market-types`` (default the main market: MATCH_ODDS, WIN, MONEY_LINE) of every sport, each from
``--lead-minutes`` before its start until it closes (the stream says so) or its expected close passes
(allsports/census.py LIVE_MINUTES, assumed). A market keeps its connection while it is wanted (no churn); a new one
goes to the connection with the most room.

Output: gzip JSON lines, a part every ten minutes per connection, to
``s3://$CAPTURE_BUCKET/betfair_stream/sports/<UTC day the run started>/s<k>-part-NNN.jsonl.gz``, deleted from the
machine once uploaded; the state each minute to ``betfair_live/<day>/status_sports_stream.json``.

    BETFAIR_APP_KEY=<the live key> python -m allsports.stream --hours 23.5 --connections 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from allsports import census
from greyhound import stream_record as sr

log = logging.getLogger(__name__)

PREFIX = "betfair_stream/sports"
SHARD_CAP = census.SHARD_CAP
CONNECTIONS = 2
MAX_CONNECTIONS = 10                   # a hard ceiling whatever is asked
CORE_TYPES = census.CORE_TYPES
CONFLATE_MS = 0                        # every change as Betfair sends it
CATALOGUE_SECONDS = 600.0


class Planner:
    """Which markets each connection subscribes to. ``catalogue(start, end)`` -> rows (census.row) of the markets
    starting in that window."""

    def __init__(self, catalogue, shards: int, cap: int = SHARD_CAP, lead_minutes: float = census.LEAD_MINUTES,
                 exclude_event_types=(), refresh_seconds: float = CATALOGUE_SECONDS):
        self.catalogue, self.shards, self.cap = catalogue, shards, cap
        self.lead = timedelta(minutes=lead_minutes)
        self.exclude = {str(x) for x in exclude_event_types}
        self.refresh_seconds = refresh_seconds
        self.rows: dict[str, dict] = {}
        self.closed: set[str] = set()
        self.assigned: dict[str, int] = {}
        self.next_read: datetime | None = None
        self.left_out = 0
        self.wanted = 0
        self.lock = threading.Lock()

    def _refresh(self, now: datetime) -> None:
        if self.next_read is not None and now < self.next_read:
            return
        self.next_read = now + timedelta(seconds=self.refresh_seconds)
        longest = timedelta(minutes=max([census.LIVE_DEFAULT, *census.LIVE_MINUTES.values()]))
        try:
            got = self.catalogue(now - longest, now + self.lead + timedelta(seconds=self.refresh_seconds))
        except Exception as exc:                            # keep the last catalogue; try again next time
            log.warning("catalogue not read (%s)", exc)
            return
        for r in got:
            if r.get("market_id") and str(r.get("event_type_id")) not in self.exclude:
                self.rows[str(r["market_id"])] = r

    def mark_closed(self, market_id: str) -> None:
        with self.lock:
            self.closed.add(str(market_id))

    def plan(self, now: datetime) -> dict[int, list[str]]:
        with self.lock:
            self._refresh(now)
            wanted = []
            for mid, r in self.rows.items():
                iv = census.open_interval(r, self.lead.total_seconds() / 60)
                if iv and iv[0] <= now <= iv[1] and mid not in self.closed:
                    start = census._utc(r.get("start_utc"))
                    wanted.append((abs((start - now).total_seconds()), mid))
            wanted.sort()
            room = self.shards * self.cap
            self.wanted = len(wanted)
            self.left_out = max(0, len(wanted) - room)
            keep = {mid for _, mid in wanted[:room]}
            self.assigned = {mid: k for mid, k in self.assigned.items() if mid in keep}
            load = [0] * self.shards
            for k in self.assigned.values():
                load[k] += 1
            for _, mid in wanted[:room]:
                if mid not in self.assigned:
                    k = min(range(self.shards), key=lambda i: load[i])
                    self.assigned[mid] = k
                    load[k] += 1
            out = {k: [] for k in range(self.shards)}
            for mid, k in self.assigned.items():
                out[k].append(mid)
            return {k: sorted(v) for k, v in out.items()}

    def pick_for(self, shard: int):
        """The pick function greyhound.stream_record.run calls for one connection."""
        return lambda _offs, now: self.plan(now)[shard]

    def state(self) -> dict:
        with self.lock:
            load = [0] * self.shards
            for k in self.assigned.values():
                load[k] += 1
            return {"catalogue_markets": len(self.rows), "wanted": self.wanted, "subscribed": sum(load),
                    "per_connection": load, "left_out": self.left_out, "closed_seen": len(self.closed)}


def closed_markets(line: str) -> list[str]:
    """The markets a stream line says are CLOSED (their definition's status), so their slot is freed."""
    if '"CLOSED"' not in line:
        return []
    try:
        msg = json.loads(line)
    except Exception:
        return []
    return [str(mc["id"]) for mc in msg.get("mc") or []
            if mc.get("id") and (mc.get("marketDefinition") or {}).get("status") == "CLOSED"]


class ShardWriter(sr.Writer):
    """The raw writer with the connection's number in each part's name, so the shards never overwrite each other."""

    def __init__(self, day: date, shard: int, on_line=None, **kw):
        self.shard, self.on_line = shard, on_line
        super().__init__(day, **kw)
        self.part = len(list(self.root.glob(f"s{shard}-part-*.jsonl.gz")))

    def _path(self) -> Path:
        return self.root / f"s{self.shard}-part-{self.part:03d}.jsonl.gz"

    def write(self, received: datetime, line: str) -> None:
        super().write(received, line)
        if self.on_line is not None:
            self.on_line(line)


def subscription(conflate_ms: int = CONFLATE_MS):
    return lambda ids, n: sr.subscription(ids, n, conflate_ms=conflate_ms)


def run_shards(day: date, until: datetime, planner: Planner, credentials, connect, writers: list[ShardWriter],
               conflate_ms: int = CONFLATE_MS, status=None, now=lambda: datetime.now(timezone.utc),
               sleep=time.sleep) -> list[dict]:
    """One greyhound.stream_record.run per connection, side by side, each on its share of the plan."""
    results: list[dict] = [{} for _ in writers]

    def one(k: int) -> None:
        results[k] = sr.run(day, until, credentials, lambda: [], connect, writers[k], now=now,
                            status=(lambda body, k=k: status(k, body)) if status else None, sleep=sleep,
                            pick=planner.pick_for(k), subscribe=subscription(conflate_ms))

    threads = [threading.Thread(target=one, args=(k,), daemon=True, name=f"stream-s{k}") for k in range(len(writers))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def main(argv=None) -> int:
    from betfair_recorder import PREFIX as BL, bucket, s3_client
    from allsports.reader import Reader
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hours", type=float, default=23.5)
    ap.add_argument("--connections", type=int, default=int(os.getenv("STREAM_CONNECTIONS") or CONNECTIONS))
    ap.add_argument("--market-types", default=",".join(CORE_TYPES))
    ap.add_argument("--lead-minutes", type=float, default=census.LEAD_MINUTES)
    ap.add_argument("--exclude-event-types", default="", help="Betfair event type ids left out, comma-separated")
    ap.add_argument("--conflate-ms", type=int, default=CONFLATE_MS)
    a = ap.parse_args(argv)
    shards = max(1, min(a.connections, MAX_CONNECTIONS))
    started = datetime.now(timezone.utc)
    day, until = started.date(), started + timedelta(hours=a.hours)
    types = [t for t in a.market_types.split(",") if t]
    reader = Reader()
    reader.login()
    s3 = s3_client()

    def put(key: str, body: bytes) -> None:
        s3.put_object(Bucket=bucket(), Key=key, Body=body)

    def catalogue(t0: datetime, t1: datetime) -> list[dict]:
        rows, reading = census.read_markets(reader, t0, t1, market_types=types, max_calls=500)
        if reading["truncated_windows"] or reading["call_budget_ran_out"]:
            log.warning("catalogue incomplete: %s", reading)
        return rows

    planner = Planner(catalogue, shards, lead_minutes=a.lead_minutes,
                      exclude_event_types=[x for x in a.exclude_event_types.split(",") if x])
    shard_state: dict[int, dict] = {}
    beat_lock = threading.Lock()

    def status(k: int, body: dict) -> None:
        with beat_lock:
            shard_state[k] = body
            doc = {"time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "connections": shards,
                   "market_types": types, "plan": planner.state(), "reader_calls": reader.calls,
                   "shards": {str(i): s for i, s in sorted(shard_state.items())}}
        try:
            put(f"{BL}/{day:%Y-%m-%d}/status_sports_stream.json", json.dumps(doc).encode())
        except Exception as exc:
            log.warning("stream status not written (%s)", exc)

    root = Path(os.getenv("SPORTS_STREAM_DIR") or (Path.home() / ".ashcroft" / "sports")) / "stream" / f"{day:%Y-%m-%d}"

    def on_line(line: str) -> None:
        for mid in closed_markets(line):
            planner.mark_closed(mid)

    writers = [ShardWriter(day, k, on_line=on_line, put=put, root=root, prefix=PREFIX, keep_local=False)
               for k in range(shards)]
    try:
        out = run_shards(day, until, planner, reader.credentials, sr.Connection, writers, a.conflate_ms, status)
    finally:
        for w in writers:
            w.maybe_roll(force=True)
    print(json.dumps({"day": f"{day}", "connections": shards, "plan": planner.state(), "shards": out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
