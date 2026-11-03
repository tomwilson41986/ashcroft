"""The greyhound books on the live key, dense near the off (the owner, 2 Nov 2026: "use the live key, just don't
execute bets yet"). Read-only: everything goes through ``trading.exchange.BetfairData``, which refuses any call but
the reads of catalogues, books and settled bets, so nothing here can place an order.

Why its own record (reports/greyhound_roadmap.md QA C1, J1, Phase 0.2): the recorder's greyhound books
(``betfair_recorder.py``, the live horse system's, not changed for another sport) are the delayed key's, polled a
minute apart, about 90 seconds old near the off and without each runner's traded volume; the greyhound trader reads
books only when it decides. A greyhound market's money arrives in its last three minutes (a median GBP380 matched
at T-3, GBP1,993 at T-1: reports/open_close/data.md 4.2), so the trade-out needs the book as it stands, every few
seconds, through those minutes.

Each GB/IE greyhound WIN and PLACE market of the day: every ``EVERY_FAR`` seconds from ``START_MINUTES`` before the
off, every ``EVERY_NEAR`` seconds in the last ``NEAR_MINUTES``, and on past the scheduled off until the market closes
or turns in play (a race goes off late; the true last book is wanted), at most ``AFTER_MINUTES`` past it. Each read
asks for the three best prices each side, each runner's traded volume (``EX_TRADED``: ``runner_total_matched``) and
the projected SP (``SP_AVAILABLE``: near and far). The rows are ``betfair_recorder.BOOK_FIELDS`` (source "live") in
``betfair_live/<day>/books_greyhound_live.csv`` with its catalogue in ``markets_greyhound_live.csv``, copied to S3
every 15 minutes and at the end (``betfair_recorder.DayRecorder``, tag ``greyhound_live``).

    BETFAIR_APP_KEY=<the live key> python -m greyhound.book_record --until 21:30
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from datetime import date, datetime, timedelta, timezone

log = logging.getLogger(__name__)

TAG = "greyhound_live"
GREYHOUNDS = "4339"
START_MINUTES = 15.0
NEAR_MINUTES = 5.0
EVERY_FAR = 60.0
EVERY_NEAR = 5.0
AFTER_MINUTES = 10.0
#: the results (the owner, 3 Nov 2026: the running table settles each race within minutes): each market read again
#: with its BSP once it is in play or past its off, and until it is CLOSED, every SETTLE_EVERY seconds for at most
#: SETTLE_MINUTES past the off; the closed book (winner, losers) is written as ``source="final"`` with the BSP read
#: at the off filled in (Betfair sends no BSP in a closed book), as betfair_recorder.final writes the horses'
SETTLE_EVERY = 30.0
SETTLE_MINUTES = 30.0
#: Betfair's request weight: EX_BEST_OFFERS (depth 3) 5, EX_TRADED 17, SP_AVAILABLE 3 a market, at most 200 a call.
#: The projected SP (near and far) comes with SP_AVAILABLE: "SP_PROJECTED" is no Betfair price-data type, and asking
#: for it had every read refused (DSC-0008) from the recorder's first day, 3 Nov 2026, to the fix that afternoon
PRICE_DATA = ["EX_BEST_OFFERS", "EX_TRADED", "SP_AVAILABLE"]
BATCH = 4                                     # 25 a market: 100 a call, well under the limit (3 Nov: at 8, 200,
                                              # the S3 day file held the settled reads and no book before the off)
REFRESH_MINUTES = 30.0


class TradedLadder:
    """The money traded at each price, runner by runner (Betfair's ``tradedVolume``, cumulative), for the queue-aware
    fills (the owner, 3 Nov 2026: "count a lay as matched only when the money traded at our price exceeds what was
    queued ahead of us"). The books' own file holds each runner's total only (``betfair_recorder.BOOK_FIELDS``, the
    horse system's, not changed), so this is a file of its own, ``traded_greyhound_live.csv`` beside it: a row each
    time a price's traded money changes (polled_utc, market_id, selection_id, price, traded), copied to S3 with the
    books. Never raises into the recording."""

    FIELDS = ["polled_utc", "market_id", "selection_id", "price", "traded"]

    def __init__(self, rec, upload_every: float = 900.0, clock=time.monotonic):
        self.rec, self.upload_every, self.clock = rec, upload_every, clock
        self.path = Path(rec.dir) / "traded_greyhound_live.csv"
        self.last: dict = {}
        self._uploaded = None
        self.rows = 0

    def write(self, raw_books, polled_at: datetime) -> int:
        try:
            out = []
            t = polled_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            for b in raw_books or []:
                mid = str(b.get("marketId"))
                for r in b.get("runners") or []:
                    sid = r.get("selectionId")
                    for lv in ((r.get("ex") or {}).get("tradedVolume") or []):
                        k = (mid, sid, float(lv["price"]))
                        v = round(float(lv["size"]), 2)
                        if self.last.get(k) != v:
                            self.last[k] = v
                            out.append({"polled_utc": t, "market_id": mid, "selection_id": sid,
                                        "price": k[2], "traded": v})
            if out:
                new = not self.path.exists()
                with self.path.open("a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=self.FIELDS, lineterminator="\n")
                    if new:
                        w.writeheader()
                    w.writerows(out)
                self.rows += len(out)
            return len(out)
        except Exception as exc:
            log.warning("traded ladder not written (%s)", exc)
            return 0

    def maybe_upload(self, force: bool = False) -> None:
        now = self.clock()
        if not force and self._uploaded is not None and now - self._uploaded < self.upload_every:
            return
        self._uploaded = now
        try:
            import gzip
            from betfair_recorder import PREFIX, bucket, s3_client
            if self.path.exists() and self.path.stat().st_size:
                s3_client().put_object(Bucket=bucket(), Key=f"{PREFIX}/{self.rec.day:%Y-%m-%d}/{self.path.name}.gz",
                                       Body=gzip.compress(self.path.read_bytes()))
        except Exception as exc:
            log.warning("traded ladder not copied to S3 (%s)", exc)


def read_books(data, market_ids: list[str]) -> list[dict]:
    """The books of the markets, with traded volume and the projected SP, in batches under the weight limit."""
    out = []
    for i in range(0, len(market_ids), BATCH):
        out += data._read("listMarketBook", {
            "marketIds": market_ids[i:i + BATCH],
            "priceProjection": {"priceData": PRICE_DATA, "exBestOffersOverrides": {"bestPricesDepth": 3}}}) or []
    return out


def settle_reads(data, rec, market_ids: list[str], bsp: dict, polled_at) -> set[str]:
    """One settled read of the markets (the book with the BSP: betfair_recorder.read_books ``settled``): each runner's
    BSP kept in ``bsp`` as it appears; each CLOSED market written as ``final`` with its BSP filled in. Returns the
    markets closed."""
    from betfair_recorder import read_books
    raw = read_books(data, market_ids, settled=True)
    closed, out = set(), []
    for b in raw or []:
        mid = str(b.get("marketId"))
        for r in b.get("runners") or []:
            sp = r.get("sp") if isinstance(r.get("sp"), dict) else {}
            v = sp.get("actualSP")
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = None
            if v is not None and v > 1:
                bsp[(mid, str(r.get("selectionId")))] = v
        if b.get("status") == "CLOSED":
            for r in b.get("runners") or []:
                if not isinstance(r.get("sp"), dict):
                    r["sp"] = {}
                k = (mid, str(r.get("selectionId")))
                if r["sp"].get("actualSP") is None and k in bsp:
                    r["sp"]["actualSP"] = bsp[k]
            out.append(b)
            closed.add(mid)
    if out:
        rec.record_books(out, source="final", polled_at=polled_at)
        rec.maybe_upload(force=True)
    return closed


def interval(minutes_to_off: float) -> float | None:
    """Seconds between reads of a market this far from its scheduled off; None when it is not read yet."""
    if minutes_to_off > START_MINUTES:
        return None
    return EVERY_NEAR if minutes_to_off <= NEAR_MINUTES else EVERY_FAR


def _utc(s) -> datetime | None:
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


def record(day: date, until: datetime, data, rec, countries=("GB", "IE"), market_types=("WIN", "PLACE"),
           now=lambda: datetime.now(timezone.utc), sleep=time.sleep, ladder=None) -> dict:
    """Read the day's greyhound markets until ``until`` (see the module's note); returns the counts."""
    from betfair_recorder import read_catalogue
    from trading.exchange import uk_day_window
    start, end = uk_day_window(day)
    offs: dict[str, datetime] = {}
    last: dict[str, datetime] = {}
    done: set[str] = set()
    settled: set[str] = set()
    last_settle: dict[str, datetime] = {}
    bsp: dict[tuple, float] = {}
    polls = rows = 0
    state = {"read_errors": 0, "last_error": "", "beat": None}

    def heartbeat(force: bool = False) -> None:
        """The recorder's state each minute to S3 (``status_greyhound_live.json`` beside its day files), read by the
        running table: the job's own log is printed only when the day ends."""
        t = now()
        if not force and state["beat"] and (t - state["beat"]).total_seconds() < 60:
            return
        state["beat"] = t
        try:
            import json
            from betfair_recorder import PREFIX, bucket, s3_client
            body = {"time_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "markets": len(offs), "polls": polls,
                    "rows": rows, "read_errors": state["read_errors"], "last_error": state["last_error"][:300],
                    "settled": len(settled), "traded_rows": ladder.rows if ladder is not None else 0}
            s3_client().put_object(Bucket=bucket(), Key=f"{PREFIX}/{day:%Y-%m-%d}/status_greyhound_live.json",
                                   Body=json.dumps(body).encode())
        except Exception as exc:
            log.warning("recorder status not written (%s)", exc)

    def refresh() -> None:
        cat = read_catalogue(data, start, end, market_types, countries, event_type=GREYHOUNDS)
        rec.record_catalogue(cat)
        for m in cat:
            off = _utc(m.get("marketStartTime"))
            if m.get("marketId") and off:
                offs[m["marketId"]] = off

    refresh()
    next_refresh = now() + timedelta(minutes=REFRESH_MINUTES)
    log.info("Reading %d greyhound markets on %s until %s UTC", len(offs), day, until.strftime("%H:%M"))
    while now() < until:
        t = now()
        due = []
        for mid, off in offs.items():
            if mid in done:
                continue
            mto = (off - t).total_seconds() / 60.0
            if mto < -AFTER_MINUTES:
                done.add(mid)
                continue
            step = interval(max(mto, 0.0))
            if step is not None and (mid not in last or (t - last[mid]).total_seconds() >= step - 0.5):
                due.append(mid)
        if due:
            try:
                raw = read_books(data, due)
            except Exception as exc:                           # one failed read: the next poll tries again
                log.warning("books not read (%s)", exc)
                state["read_errors"] += 1
                state["last_error"] = str(exc)
                raw = []
            rows += rec.record_books(raw, source="live", polled_at=t)
            if ladder is not None:
                ladder.write(raw, t)
                ladder.maybe_upload()
            polls += 1
            for b in raw:
                mid = b.get("marketId")
                last[mid] = t
                if b.get("inplay") or b.get("status") == "CLOSED":
                    done.add(mid)
        due_settle = [mid for mid, off in offs.items()
                      if mid not in settled and (mid in done or t >= off)
                      and (t - off).total_seconds() <= SETTLE_MINUTES * 60
                      and (mid not in last_settle or (t - last_settle[mid]).total_seconds() >= SETTLE_EVERY - 0.5)]
        if due_settle:
            try:
                settled |= settle_reads(data, rec, due_settle, bsp, t)
            except Exception as exc:                           # one failed read: the next tries again
                log.warning("settled books not read (%s)", exc)
            for mid in due_settle:
                last_settle[mid] = t
        if t >= next_refresh:
            try:
                refresh()
            except Exception as exc:
                log.warning("catalogue not read again (%s)", exc)
            next_refresh = t + timedelta(minutes=REFRESH_MINUTES)
        if offs and all(m in done for m in offs) and t >= max(offs.values()) and all(
                m in settled or (t - off).total_seconds() > SETTLE_MINUTES * 60 for m, off in offs.items()):
            break
        heartbeat()
        sleep(1.0)
    heartbeat(force=True)
    rec.maybe_upload(force=True)
    if ladder is not None:
        ladder.maybe_upload(force=True)
    return {"markets": len(offs), "polls": polls, "rows": rows, "done": len(done), "settled": len(settled),
            "traded_rows": ladder.rows if ladder is not None else 0}


def main(argv=None) -> int:
    from betfair_recorder import DayRecorder, uk_time_on, uk_today
    from trading.exchange import BetfairData
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--until", default="21:30", help="UK time to stop")
    ap.add_argument("--date", default=None)
    ap.add_argument("--countries", default="GB,IE")
    a = ap.parse_args(argv)
    day = date.fromisoformat(a.date) if a.date else uk_today()
    rec = DayRecorder(day, tag=TAG)
    data = BetfairData(source="greyhound_live")                # read-only by construction: no order call exists
    try:
        data.login()
        print(record(day, uk_time_on(day, a.until), data, rec, tuple(a.countries.split(",")),
                     ladder=TradedLadder(rec)))
    finally:
        rec.maybe_upload(force=True)
    return 0 if rec.errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
