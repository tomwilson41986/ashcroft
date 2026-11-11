"""Betfair's Exchange Stream for the GB/IE horse racing WIN markets, recorded raw (the consultant review, 9 Nov 2026).

Read-only: it logs in, subscribes to market data and writes what arrives. Nothing here can place or cancel an order
(the stream's order subscription is never sent). It runs beside the horse trader in live-trade.yml from the session's
start to 10:30 UK, on the live key (the owner, 10 Nov 2026: "wire the horse ladder recorder"; see "Where it runs").

What differs from the greyhounds' recorder:
- the markets: horse racing (event type 7), WIN, GB and IE. A day's WIN markets fit one subscription (at most 190),
  so every market not yet off is subscribed from the start: the rule's edge is hours before the off, not minutes;
- ``conflateMs`` 500 by default: finer than anything a rule read once a minute decides on, and it bounds the volume
  (the traded volumes are running totals, so nothing traded is lost to it);
- a part is deleted from the server once it is in S3 (the horse runner's disk is the live trader's);
- to ``s3://$CAPTURE_BUCKET/betfair_stream/horse/<day>/part-NNN.jsonl.gz``; the state each minute to
  ``betfair_live/<day>/status_horse_stream.json``.

Where it runs: live-trade.yml, a background process beside the trader, under a timeout, to 10:30 UK, the morning the
rule trades (the recorder's own job, live-record.yml, waits all day for the runner while the trader is live). The
stream needs the live key: Betfair offers it on no delayed key, which carries no full ladder or traded volume either,
so this is the one read-only process beside the trader on the live key (the owner's decision of 10 Nov 2026). Without
the secret it does not run; its failing never touches the trader. By hand:

    BETFAIR_APP_KEY=<the live key> python -m research.horse_stream_record --until 10:30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path

from greyhound import stream_record as sr

log = logging.getLogger(__name__)

HORSE_RACING = "7"
PREFIX = "betfair_stream/horse"
CONFLATE_MS = 500
LATE_MINUTES = 15.0
MAX_MARKETS = sr.MAX_MARKETS


def not_yet_off(offs: dict, now: datetime) -> list[str]:
    """Every market not yet off, and those off in the last ``LATE_MINUTES`` (a race off late), nearest first, at
    most ``MAX_MARKETS``."""
    lo = now - timedelta(minutes=LATE_MINUTES)
    ids = sorted((off, mid) for mid, off in offs.items() if off >= lo)
    return [mid for _, mid in ids][:MAX_MARKETS]


def subscription(conflate_ms: int = CONFLATE_MS):
    """The subscription builder ``greyhound.stream_record.run`` calls, at this conflation."""
    return lambda ids, n: sr.subscription(ids, n, conflate_ms=conflate_ms)


def main(argv=None) -> int:
    from betfair_recorder import PREFIX as BL, bucket, read_catalogue, s3_client, uk_time_on, uk_today
    from trading.exchange import BetfairData, uk_day_window
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--until", default="21:30", help="UK time to stop")
    ap.add_argument("--date", default=None)
    ap.add_argument("--countries", default="GB,IE")
    ap.add_argument("--conflate-ms", type=int, default=CONFLATE_MS)
    a = ap.parse_args(argv)
    day = date.fromisoformat(a.date) if a.date else uk_today()
    data = BetfairData(source="horse_stream")                # read-only: no order call exists on it
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
        return read_catalogue(data, start, end, ("WIN",), tuple(a.countries.split(",")), event_type=HORSE_RACING)

    def status(body: dict) -> None:
        try:
            put(f"{BL}/{day:%Y-%m-%d}/status_horse_stream.json", json.dumps(body).encode())
        except Exception as exc:
            log.warning("stream status not written (%s)", exc)

    root = Path(os.getenv("HORSE_STREAM_DIR") or (Path.home() / ".ashcroft" / "horse")) / "stream" / f"{day:%Y-%m-%d}"
    writer = sr.Writer(day, put=put, root=root, prefix=PREFIX, keep_local=False)
    try:
        out = sr.run(day, uk_time_on(day, a.until), credentials, catalogue, sr.Connection, writer, status=status,
                     pick=not_yet_off, subscribe=subscription(a.conflate_ms))
    finally:
        writer.maybe_roll(force=True)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
