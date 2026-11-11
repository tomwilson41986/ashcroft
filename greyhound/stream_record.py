"""Betfair's Exchange Stream for the GB/IE greyhound markets, recorded raw (the owner, 5 Nov 2026, on the audit:
"Work on the next items"; reports/greyhound_audit_1105.md C1). Read-only: it logs in, subscribes to market data and
writes what arrives; nothing here can place or cancel an order (the stream's order subscription is never sent).

Why: the polled recorders hold the best 3 prices only, the delayed key's every 60 s (about 90 s stale) and the live
key's every 5 s from 15 minutes out (greyhound/book_record.py). Every honest backtest of a resting order needs the full
ladder, the volume traded at each price and each change as it happened; the stream carries all of it, and each day
recorded is a day of data the research never had.

What: every ``mcm`` message (market changes: ``EX_ALL_OFFERS``, ``EX_TRADED``, ``EX_TRADED_VOL``, ``EX_LTP``,
``EX_MARKET_DEF``, ``SP_TRADED``, ``SP_PROJECTED``), each line as received with its local receive time (UTC), in gzip
JSON lines, a part every ``UPLOAD_SECONDS`` to ``s3://$CAPTURE_BUCKET/betfair_stream/greyhound/<day>/part-NNN.jsonl.gz``.
A subscription takes at most 200 markets (Betfair's default), and a GB/IE day has more WIN and PLACE markets than that,
so the recorder subscribes to the markets off within ``WINDOW_MINUTES`` (and those off in the last 15 minutes, a race
off late), refreshed every ``RESUB_SECONDS``; a new subscription replaces the last and starts with a full image.
A dropped connection is reconnected with a fresh login check; the state each minute to
``betfair_live/<day>/status_greyhound_stream.json``.

    BETFAIR_APP_KEY=<the live key> python -m greyhound.stream_record --until 21:30
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import socket
import ssl
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)

HOST, PORT = "stream-api.betfair.com", 443
FIELDS = ["EX_ALL_OFFERS", "EX_TRADED", "EX_TRADED_VOL", "EX_LTP", "EX_MARKET_DEF", "SP_TRADED", "SP_PROJECTED"]
GREYHOUNDS = "4339"
WINDOW_MINUTES = 75.0
LATE_MINUTES = 15.0
MAX_MARKETS = 190                       # under Betfair's default subscription limit of 200
RESUB_SECONDS = 300.0
CATALOGUE_SECONDS = 600.0
UPLOAD_SECONDS = 600.0
HEARTBEAT_MS = 5000
READ_TIMEOUT = 20.0                     # four heartbeats: no line in this long and the connection is remade
PREFIX = "betfair_stream/greyhound"


class Connection:
    """One SSL connection to the stream: CRLF-delimited JSON both ways."""

    def __init__(self, host: str = HOST, port: int = PORT, timeout: float = READ_TIMEOUT):
        raw = socket.create_connection((host, port), timeout=timeout)
        self.sock = ssl.create_default_context().wrap_socket(raw, server_hostname=host)
        self.sock.settimeout(timeout)
        self.buf = b""

    def send(self, msg: dict) -> None:
        self.sock.sendall((json.dumps(msg) + "\r\n").encode())

    def line(self) -> str:
        while b"\n" not in self.buf:
            got = self.sock.recv(1 << 16)
            if not got:
                raise ConnectionError("the stream closed the connection")
            self.buf += got
        out, self.buf = self.buf.split(b"\n", 1)
        return out.decode("utf-8", "replace").rstrip("\r")

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


class Writer:
    """The raw lines, one gzip part at a time, each part uploaded to S3 when closed."""

    def __init__(self, day: date, put=None, root: Path | None = None, clock=time.monotonic, prefix: str = PREFIX,
                 keep_local: bool = True):
        self.day, self.put, self.clock, self.prefix, self.keep_local = day, put, clock, prefix, keep_local
        self.root = root or Path(os.getenv("GREYHOUND_LEDGER_DIR") or (Path.home() / ".ashcroft" / "greyhound")) \
            / "stream" / f"{day:%Y-%m-%d}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.part = len(list(self.root.glob("part-*.jsonl.gz")))      # a restart starts a new part, never rewrites
        self.f = None
        self.opened = None
        self.lines = 0
        self.uploaded: list[str] = []

    def _path(self) -> Path:
        return self.root / f"part-{self.part:03d}.jsonl.gz"

    def write(self, received: datetime, line: str) -> None:
        if self.f is None:
            self.f = gzip.open(self._path(), "at", encoding="utf-8")
            self.opened = self.clock()
        self.f.write(json.dumps({"rt": received.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), "m": line}) + "\n")
        self.lines += 1

    def maybe_roll(self, force: bool = False) -> None:
        if self.f is None or (not force and self.clock() - self.opened < UPLOAD_SECONDS):
            return
        self.f.close()
        path, self.f = self._path(), None
        self.part += 1
        if self.put is not None:
            key = f"{self.prefix}/{self.day:%Y-%m-%d}/{path.name}"
            try:
                self.put(key, path.read_bytes())
                self.uploaded.append(key)
                if not self.keep_local:                    # uploaded: the part need not fill the server's disk
                    path.unlink()
            except Exception as exc:                       # the local part stays; the next run's upload can retry
                log.warning("%s not uploaded (%s)", path.name, exc)


def window(offs: dict, now: datetime) -> list[str]:
    """The markets off within ``WINDOW_MINUTES`` or in the last ``LATE_MINUTES``, nearest first, at most
    ``MAX_MARKETS``."""
    lo, hi = now - timedelta(minutes=LATE_MINUTES), now + timedelta(minutes=WINDOW_MINUTES)
    ids = sorted((off, mid) for mid, off in offs.items() if lo <= off <= hi)
    return [mid for _, mid in ids][:MAX_MARKETS]


def subscription(ids: list[str], n: int, conflate_ms: int = 0) -> dict:
    return {"op": "marketSubscription", "id": n, "heartbeatMs": HEARTBEAT_MS, "conflateMs": int(conflate_ms),
            "marketFilter": {"marketIds": list(ids)}, "marketDataFilter": {"fields": FIELDS}}


def _utc(s) -> datetime | None:
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def run(day: date, until: datetime, credentials, catalogue, connect, writer: Writer,
        now=lambda: datetime.now(timezone.utc), status=None, sleep=time.sleep, pick=None, subscribe=None) -> dict:
    """Record until ``until``. ``credentials()`` -> (app key, session token); ``catalogue()`` -> the day's markets
    (marketId, marketStartTime); ``connect()`` -> a Connection; ``status(dict)`` writes the state. ``pick(offs, now)``
    chooses the markets to subscribe to (default ``window``) and ``subscribe(ids, n)`` builds the subscription
    (default ``subscription``): another recorder of the same stream passes its own."""
    pick = window if pick is None else pick
    subscribe = subscription if subscribe is None else subscribe
    offs: dict[str, datetime] = {}
    st = {"connects": 0, "auth_failures": 0, "errors": 0, "last_error": "", "markets": 0, "lines": 0,
          "subscribed": 0}
    next_cat = next_beat = now()
    msg_id = 0

    def refresh():
        for m in catalogue():
            off = _utc(m.get("marketStartTime"))
            if m.get("marketId") and off:
                offs[str(m["marketId"])] = off
        st["markets"] = len(offs)

    def beat(force=False):
        nonlocal next_beat
        if status is None or (not force and now() < next_beat):
            return
        next_beat = now() + timedelta(seconds=60)
        status({"time_utc": now().strftime("%Y-%m-%dT%H:%M:%SZ"), **st, "lines": writer.lines,
                "parts_uploaded": len(writer.uploaded)})

    while now() < until:
        conn = None
        try:
            if now() >= next_cat:
                refresh()
                next_cat = now() + timedelta(seconds=CATALOGUE_SECONDS)
            ids = pick(offs, now())
            if not ids:                                     # nothing near its off: wait for the next window
                beat()
                sleep(30)
                continue
            conn = connect()
            st["connects"] += 1
            writer.write(now(), conn.line())                 # the 'connection' message
            key, token = credentials()
            msg_id += 1
            conn.send({"op": "authentication", "id": msg_id, "appKey": key, "session": token})
            reply = conn.line()
            writer.write(now(), reply)
            r = json.loads(reply)
            if r.get("statusCode") != "SUCCESS":
                st["auth_failures"] += 1
                raise ConnectionError(f"authentication: {r.get('errorCode')} {r.get('errorMessage')}")
            msg_id += 1
            conn.send(subscribe(ids, msg_id))
            st["subscribed"] = len(ids)
            resub_at = now() + timedelta(seconds=RESUB_SECONDS)
            while now() < until:
                line = conn.line()
                writer.write(now(), line)
                if '"op":"status"' in line.replace(" ", "") and '"FAILURE"' in line:
                    raise ConnectionError(f"stream status: {line[:300]}")
                if now() >= resub_at:
                    if now() >= next_cat:
                        refresh()
                        next_cat = now() + timedelta(seconds=CATALOGUE_SECONDS)
                    new = pick(offs, now())
                    if new and new != ids:
                        ids = new
                        msg_id += 1
                        conn.send(subscribe(ids, msg_id))
                        st["subscribed"] = len(ids)
                    resub_at = now() + timedelta(seconds=RESUB_SECONDS)
                writer.maybe_roll()
                beat()
        except Exception as exc:                            # remake the connection; never stop the day for one error
            st["errors"] += 1
            st["last_error"] = str(exc)[:300]
            log.warning("stream: %s", exc)
            sleep(5)
        finally:
            if conn is not None:
                conn.close()
    writer.maybe_roll(force=True)
    beat(force=True)
    return {"day": f"{day}", **st, "lines": writer.lines, "parts_uploaded": len(writer.uploaded)}


def main(argv=None) -> int:
    from betfair_recorder import PREFIX as BL, bucket, read_catalogue, s3_client, uk_time_on, uk_today
    from trading.exchange import BetfairData, uk_day_window
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--until", default="21:30", help="UK time to stop")
    ap.add_argument("--date", default=None)
    ap.add_argument("--countries", default="GB,IE")
    a = ap.parse_args(argv)
    day = date.fromisoformat(a.date) if a.date else uk_today()
    data = BetfairData(source="greyhound_stream")            # read-only: no order call exists on it
    data.login()
    start, end = uk_day_window(day)
    s3 = s3_client()

    def put(key: str, body: bytes) -> None:
        s3.put_object(Bucket=bucket(), Key=key, Body=body)

    def credentials():
        if not data.client.session_token:
            data.login()
        return data.client.app_key, data.client.session_token

    def catalogue():
        return read_catalogue(data, start, end, ("WIN", "PLACE"), tuple(a.countries.split(",")), event_type=GREYHOUNDS)

    def status(body: dict) -> None:
        try:
            put(f"{BL}/{day:%Y-%m-%d}/status_greyhound_stream.json", json.dumps(body).encode())
        except Exception as exc:
            log.warning("stream status not written (%s)", exc)

    writer = Writer(day, put=put)
    try:
        out = run(day, uk_time_on(day, a.until), credentials, catalogue, Connection, writer, status=status)
    finally:
        writer.maybe_roll(force=True)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
