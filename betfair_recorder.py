"""Record Betfair's live win and place markets for research: every book the trader reads, and snapshots all day.

The owner's ask (30 Sep 2026): now that the UK server reaches Betfair, keep all of it so the models can learn from
it. What the delayed application key's feed carries, as the trader sees it: for each GB/IE market, every runner's
best three back and lay prices with their sizes, the last traded price, the market's matched money (the delayed key
reports no runner's), the market's status; the catalogue (cloth number, stall, jockey, trainer, age, weight, rating,
form, headgear, the forecast price); and once a market is settled, each runner's BSP and result. Nothing here can
place an order (trading.exchange.BetfairData reads catalogue and books only).

Where it goes. Append-only files for each UK racing day on the UK server ($BETFAIR_LIVE_DIR, default
~/.ashcroft/betfair_live/<day>/): books.csv, one row a runner a snapshot, and markets.csv, one row a runner of each
market's catalogue. They are copied gzipped to s3://$CAPTURE_BUCKET/betfair_live/<day>/ through the day. The nightly
job (daily-results.yml, the one job that writes the database) loads them into horse_racing.db:

    betfair_live_markets   the catalogue, matched to race_results
    betfair_live_marks     for each runner, the book nearest to each clock mark of the morning (08:00-12:00 UK)
                           and to each mark before the off (T-120 ... T-1 minutes), the last book before the off,
                           and the settled one (BSP, result)
    live_orders            the live trader's ledger, every order and its settlement

Usage:
    python betfair_recorder.py --record --until 21:30          # snapshots to 21:30 UK (live-record.yml)
    python betfair_recorder.py --final --date 2026-09-30       # that day's settled books (BSP, winners)
    python betfair_recorder.py --record --date 2026-10-06 --tag evening --market-types WIN --evening 17:00-21:30
                                                               # 6 Oct's win markets, on the evening of 5 Oct
    python betfair_recorder.py --upload --date 2026-09-30      # copy the day's files to S3 now
    python betfair_recorder.py --load-db --days 3              # the nightly load, from S3 into horse_racing.db
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import json
import io
import logging
import math
import os
import signal
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(os.getenv("BETFAIR_LIVE_DIR") or (Path.home() / ".ashcroft" / "betfair_live"))
PREFIX = "betfair_live"
DEFAULT_DB = str(Path(__file__).resolve().parent / "horse_racing.db")

BOOK_FIELDS = [
    "polled_utc", "source", "market_id", "status", "inplay", "complete", "bsp_reconciled", "number_of_runners",
    "number_of_active_runners", "market_total_matched", "last_match_utc", "is_delayed", "bet_delay", "version",
    "selection_id", "runner_status", "adjustment_factor", "removal_utc", "last_traded", "runner_total_matched",
    "back1", "back1_size", "back2", "back2_size", "back3", "back3_size",
    "lay1", "lay1_size", "lay2", "lay2_size", "lay3", "lay3_size",
    "sp_near", "sp_far", "sp_actual", "sp_back_taken", "sp_lay_taken",
    "number_of_winners",                                    # the places the market pays (from 7 Oct 2026)
]
MARKET_FIELDS = [
    "market_id", "market_type", "market_name", "market_start_utc", "venue", "country", "event_id", "event_name",
    "race_type", "selection_id", "runner_name", "sort_priority", "cloth_number", "stall_draw", "jockey", "trainer",
    "age", "weight_value", "official_rating", "form", "days_since_last_run", "wearing", "forecast_num",
    "forecast_den", "recorded_utc",
    "each_way_divisor",                                     # an each-way market's place fraction, 1/divisor (7 Oct)
]
#: the catalogue the recorder asks for (the trader's own catalogue call is left as it is)
CATALOGUE_PROJECTION = ["EVENT", "MARKET_START_TIME", "RUNNER_DESCRIPTION", "RUNNER_METADATA", "MARKET_DESCRIPTION"]
#: marks of the morning, UK clock time; marks before the off, minutes, each with how far a snapshot may stray
CLOCK_MARKS = ("08:00", "09:00", "10:00", "11:00", "12:00")
OFF_MARKS = (120, 60, 30, 15, 10, 5, 3, 1)
#: a market's BSP is read once it has reconciled at the off (Betfair's book of a closed market carries none): up to
#: BSP_TRIES reads, BSP_EVERY seconds apart, within 15 minutes of the off. On 7 Oct none found one (189 WIN markets):
#: the delayed key sends no Starting Price data after the off at all (1-7 Oct), and the live key's reads (the trader's
#: settling, SP_TRADED) brought the money taken at SP but no BSP (ledger bsp-at-off-1007). Both ask SP_AVAILABLE too
#: from 8 Oct; until a BSP is read the day's come with Betfair's price files the next day
BSP_TRIES, BSP_EVERY = 3, 15.0
#: Betfair's event type ids: the record is of horse racing unless asked otherwise (greyhounds: --event-type 4339)
HORSE_RACING, GREYHOUNDS = "7", "4339"


def bucket() -> str:
    return os.getenv("CAPTURE_BUCKET") or os.getenv("ULTRA_BETTING_S3_BUCKET") or "ashcroft"


def s3_client():
    import boto3
    from botocore.config import Config
    # short timeouts: an upload that hangs must not hold up whoever called it
    return boto3.client("s3", region_name=os.getenv("CAPTURE_REGION") or "eu-west-2",
                        config=Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 3}))


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc(text) -> datetime | None:
    if not text or (isinstance(text, float) and math.isnan(text)):
        return None
    try:
        return datetime.strptime(str(text)[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _num(x):
    """A price or size as a float; Betfair's "NaN" and "Infinity" projections, and blanks, as None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _taken(levels) -> float | None:
    """The total of a BSP-taken ladder ([{price, size}]), or None when Betfair sends none."""
    if not levels:
        return None
    return round(sum(float(o.get("size") or 0.0) for o in levels), 2)


def uk_today(now: datetime | None = None) -> date:
    from zoneinfo import ZoneInfo
    return (now or datetime.now(timezone.utc)).astimezone(ZoneInfo("Europe/London")).date()


def uk_time_on(day: date, hhmm: str) -> datetime:
    from zoneinfo import ZoneInfo
    h, m = (int(x) for x in hhmm.split(":"))
    return datetime(day.year, day.month, day.day, h, m, tzinfo=ZoneInfo("Europe/London")).astimezone(timezone.utc)


def evening_window(day: date, spec: str) -> tuple[datetime, datetime]:
    """The evening before ``day``, between two UK times ("17:00-21:30"), in UTC: when ``day``'s markets are
    recorded the evening before."""
    try:
        a, b = (uk_time_on(day - timedelta(days=1), s.strip()) for s in spec.split("-"))
    except ValueError:
        raise ValueError(f"an evening is two UK times, HH:MM-HH:MM, not {spec!r}") from None
    if not a < b:
        raise ValueError(f"the evening {spec!r} ends before it starts")
    return a, b


# --------------------------------------------------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------------------------------------------------

def book_rows(raw_books, polled_at: datetime, source: str) -> list[dict]:
    """listMarketBook's reply as one row a runner."""
    ts = _iso(polled_at)
    rows = []
    for b in raw_books or []:
        base = {"polled_utc": ts, "source": source, "market_id": b.get("marketId"), "status": b.get("status"),
                "inplay": int(bool(b.get("inplay"))), "complete": int(bool(b.get("complete"))),
                "bsp_reconciled": int(bool(b.get("bspReconciled"))),
                "number_of_runners": b.get("numberOfRunners"),
                "number_of_active_runners": b.get("numberOfActiveRunners"),
                "market_total_matched": _num(b.get("totalMatched")), "last_match_utc": b.get("lastMatchTime"),
                "is_delayed": int(bool(b.get("isMarketDataDelayed"))), "bet_delay": b.get("betDelay"),
                "version": b.get("version"), "number_of_winners": b.get("numberOfWinners")}
        for r in b.get("runners") or []:
            ex, sp = r.get("ex") or {}, r.get("sp") or {}
            row = dict(base, selection_id=r.get("selectionId"), runner_status=r.get("status"),
                       adjustment_factor=_num(r.get("adjustmentFactor")), removal_utc=r.get("removalDate"),
                       last_traded=_num(r.get("lastPriceTraded")), runner_total_matched=_num(r.get("totalMatched")),
                       sp_near=_num(sp.get("nearPrice")), sp_far=_num(sp.get("farPrice")),
                       sp_actual=_num(sp.get("actualSP")), sp_back_taken=_taken(sp.get("backStakeTaken")),
                       sp_lay_taken=_taken(sp.get("layLiabilityTaken")))
            for side, key in (("back", "availableToBack"), ("lay", "availableToLay")):
                levels = ex.get(key) or []
                for i in range(3):
                    lv = levels[i] if i < len(levels) else {}
                    row[f"{side}{i + 1}"], row[f"{side}{i + 1}_size"] = _num(lv.get("price")), _num(lv.get("size"))
            rows.append(row)
    return rows


def catalogue_rows(raw_catalogue, recorded_at: datetime) -> list[dict]:
    """listMarketCatalogue's reply as one row a runner (the metadata only when RUNNER_METADATA was asked for)."""
    ts = _iso(recorded_at)
    rows = []
    for m in raw_catalogue or []:
        ev, desc = m.get("event") or {}, m.get("description") or {}
        for r in m.get("runners") or []:
            md = r.get("metadata") or {}
            rows.append({
                "market_id": m.get("marketId"), "market_type": desc.get("marketType") or "",
                "market_name": m.get("marketName"), "market_start_utc": m.get("marketStartTime"),
                "venue": ev.get("venue"), "country": ev.get("countryCode"), "event_id": ev.get("id"),
                "event_name": ev.get("name"), "race_type": desc.get("raceType"),
                "selection_id": r.get("selectionId"), "runner_name": r.get("runnerName"),
                "sort_priority": r.get("sortPriority"), "cloth_number": md.get("CLOTH_NUMBER"),
                "stall_draw": md.get("STALL_DRAW"), "jockey": md.get("JOCKEY_NAME"), "trainer": md.get("TRAINER_NAME"),
                "age": md.get("AGE"), "weight_value": md.get("WEIGHT_VALUE"),
                "official_rating": md.get("OFFICIAL_RATING"), "form": md.get("FORM"),
                "days_since_last_run": md.get("DAYS_SINCE_LAST_RUN"), "wearing": md.get("WEARING"),
                "forecast_num": md.get("FORECASTPRICE_NUMERATOR"), "forecast_den": md.get("FORECASTPRICE_DENOMINATOR"),
                "recorded_utc": ts, "each_way_divisor": _num(desc.get("eachWayDivisor"))})
    return rows


def _append(path: Path, fields: list[str], rows: list[dict]) -> None:
    if not rows:
        return
    new = not path.exists() or path.stat().st_size == 0
    if not new:                                 # a file begun by an older recorder keeps its own columns
        with path.open(newline="") as f:
            head = next(csv.reader(f), None)
        if head:
            fields = head
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        if new:
            w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------------------------------------------------
# The day's files
# --------------------------------------------------------------------------------------------------------------------

class DayRecorder:
    """The append-only record of one UK racing day. It never raises into its caller: a record that cannot be written
    is logged and counted, and the trading (or recording) goes on."""

    def __init__(self, day: date, root: Path | str | None = None, s3=None, upload_every: float = 900.0,
                 clock=time.monotonic, background: bool = True, tag: str = ""):
        self.day = day
        self.tag = tag
        self.dir = Path(root or ROOT) / f"{day:%Y-%m-%d}"
        self.dir.mkdir(parents=True, exist_ok=True)
        # a tag gives a writer its own files (books_trader.csv): the trader and the recorder run side by side
        suffix = f"_{tag}" if tag else ""
        self.books_path = self.dir / f"books{suffix}.csv"
        self.markets_path = self.dir / f"markets{suffix}.csv"
        self._s3 = s3
        self.upload_every = upload_every
        self._clock = clock
        self.background = background
        self._uploader = None
        self._last_upload: float | None = None
        self.rows = 0
        self.errors = 0
        self._seen = self._markets_on_file()

    def _markets_on_file(self) -> dict[str, bool]:
        """The markets already catalogued, and whether with the runners' metadata (jockey, draw, ...)."""
        if not self.markets_path.exists():
            return {}
        try:
            seen: dict[str, bool] = {}
            with self.markets_path.open() as f:
                for r in csv.DictReader(f):
                    md = any(r.get(k) for k in ("cloth_number", "stall_draw", "jockey", "trainer"))
                    seen[r["market_id"]] = seen.get(r["market_id"], False) or md
            return seen
        except Exception as exc:                       # a torn last line: the catalogue is simply recorded again
            log.warning("recorder: %s unreadable (%s)", self.markets_path, exc)
            return {}

    def record_books(self, raw_books, source: str, polled_at: datetime | None = None) -> int:
        try:
            rows = book_rows(raw_books, polled_at or datetime.now(timezone.utc), source)
            _append(self.books_path, BOOK_FIELDS, rows)
            self.rows += len(rows)
        except Exception as exc:
            self.errors += 1
            log.warning("recorder: %d books not recorded (%s)", len(raw_books or []), exc)
            return 0
        self.maybe_upload()
        return len(rows)

    def record_catalogue(self, raw_catalogue, recorded_at: datetime | None = None) -> int:
        """The catalogue of markets not yet on file, and again of those on file without the runners' metadata when
        this call carries it (the trader's catalogue has none; the recorder's has). The load keeps the last row."""
        try:
            def has_md(m) -> bool:
                return any(r.get("metadata") for r in m.get("runners") or [])
            fresh = [m for m in raw_catalogue or []
                     if m.get("marketId") not in self._seen or (has_md(m) and not self._seen[m.get("marketId")])]
            rows = catalogue_rows(fresh, recorded_at or datetime.now(timezone.utc))
            _append(self.markets_path, MARKET_FIELDS, rows)
            for m in fresh:
                self._seen[m.get("marketId")] = self._seen.get(m.get("marketId"), False) or has_md(m)
            return len(rows)
        except Exception as exc:
            self.errors += 1
            log.warning("recorder: catalogue not recorded (%s)", exc)
            return 0

    def maybe_upload(self, force: bool = False) -> bool:
        """Copy the day's files to S3 every ``upload_every`` seconds, in the background so a slow S3 never holds up
        the caller; ``force`` waits for any copy under way and copies now (the end of a session)."""
        now = self._clock()
        if force:
            if self._uploader is not None:
                self._uploader.join(timeout=120)
            self._last_upload = now
            return self._upload_logged()
        if self._last_upload is not None and now - self._last_upload < self.upload_every:
            return False
        if self._uploader is not None and self._uploader.is_alive():
            return False
        self._last_upload = now
        if not self.background:
            return self._upload_logged()
        import threading
        self._uploader = threading.Thread(target=self._upload_logged, name="recorder-upload", daemon=True)
        self._uploader.start()
        return True

    def _upload_logged(self) -> bool:
        try:
            self.upload()
            return True
        except Exception as exc:                        # S3 unreachable: the files stay on the server, sent later
            self.errors += 1
            log.warning("recorder: upload to S3 failed (%s); kept on the server", exc)
            return False

    def upload(self) -> None:
        s3 = self._s3 or s3_client()
        for p in (self.books_path, self.markets_path):
            if p.exists() and p.stat().st_size:
                s3.put_object(Bucket=bucket(), Key=f"{PREFIX}/{self.day:%Y-%m-%d}/{p.name}.gz",
                              Body=gzip.compress(p.read_bytes()))


#: the day's writers: the recorder's files, and the trader's (books_trader.csv) when the two ran side by side
DAY_FILE_TAGS = ("", "trader")


def market_ids_on_file(day: date, root: Path | str | None = None) -> list[str]:
    ids: set[str] = set()
    for tag in DAY_FILE_TAGS:
        path = Path(root or ROOT) / f"{day:%Y-%m-%d}" / f"markets{'_' + tag if tag else ''}.csv"
        if path.exists():
            with path.open() as f:
                ids |= {r["market_id"] for r in csv.DictReader(f)}
    return sorted(ids)


# --------------------------------------------------------------------------------------------------------------------
# Reading Betfair (read-only)
# --------------------------------------------------------------------------------------------------------------------

def read_catalogue(data, start: datetime, end: datetime, market_types=("WIN", "PLACE"),
                   countries=("GB", "IE"), window_hours: float = 4.0, event_type: str = HORSE_RACING) -> list[dict]:
    """The catalogue, one market type and a few hours at a time: with MARKET_DESCRIPTION and RUNNER_METADATA (a
    weight of 2 a market) Betfair answers at most 100 markets a call, and a big Saturday has more than that."""
    out, seen = [], set()
    for mtype in market_types:
        t = start
        while t < end:
            t2 = min(end, t + timedelta(hours=window_hours))
            for m in data._read("listMarketCatalogue", {
                    "filter": {"eventTypeIds": [event_type], "marketCountries": list(countries), "marketTypeCodes": [mtype],
                               "marketStartTime": {"from": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                                   "to": t2.strftime("%Y-%m-%dT%H:%M:%SZ")}},
                    "marketProjection": CATALOGUE_PROJECTION, "maxResults": 100, "sort": "FIRST_TO_START"}) or []:
                if m.get("marketId") not in seen:
                    seen.add(m.get("marketId"))
                    out.append(m)
            t = t2
    return out


def census(data, day: date, countries=("GB", "IE"), window_hours: float = 2.0) -> dict:
    """Every market Betfair lists on a day's racing, by type, with a few names of each (read-only): which markets a
    recording could add beside win and place (match bets, the other place markets, forecasts). One call a window of
    hours, any market type; a weight of 1 a market lets Betfair answer 200 a call."""
    from trading.exchange import uk_day_window
    start, end = uk_day_window(day)
    by_type: dict[str, list[str]] = {}
    seen, full = set(), 0
    t = start
    while t < end:
        t2 = min(end, t + timedelta(hours=window_hours))
        got = data._read("listMarketCatalogue", {
            "filter": {"eventTypeIds": ["7"], "marketCountries": list(countries),
                       "marketStartTime": {"from": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "to": t2.strftime("%Y-%m-%dT%H:%M:%SZ")}},
            "marketProjection": ["MARKET_DESCRIPTION", "EVENT"], "maxResults": 200, "sort": "FIRST_TO_START"}) or []
        full += len(got) >= 200                           # a window Betfair may have cut short
        for m in got:
            if m.get("marketId") in seen:
                continue
            seen.add(m.get("marketId"))
            kind = (m.get("description") or {}).get("marketType") or "?"
            by_type.setdefault(kind, []).append(f"{(m.get('event') or {}).get('venue', '')}: {m.get('marketName', '')}")
        t = t2
    out = {k: {"markets": len(v), "e.g.": v[:4]} for k, v in sorted(by_type.items(), key=lambda kv: -len(kv[1]))}
    if full:
        out["_note"] = f"{full} window(s) returned 200 markets, the most a call answers: some may be missing"
    return out


def census_sports(data, hours: float = 24.0, top: int = 200, now: datetime | None = None) -> dict:
    """Every sport Betfair lists open markets on that start in the next ``hours``, with its market types and the money
    matched so far on its busiest markets (read-only): the liquidity behind the choice of which sport to model next
    (the owner's ask, 3 Oct 2026). One listEventTypes, then for each sport a listMarketTypes and one catalogue of its
    ``top`` markets by matched money. Read once a day at the same time, the matched money compares the sports like
    with like; it is the money matched by then, not each market's final total."""
    start = now or datetime.now(timezone.utc)
    end = start + timedelta(hours=hours)
    window = {"marketStartTime": {"from": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "to": end.strftime("%Y-%m-%dT%H:%M:%SZ")}}
    out: dict = {"read_utc": _iso(start), "hours": hours, "sports": []}
    for et in data._read("listEventTypes", {"filter": window}) or []:
        sport = et.get("eventType") or {}
        sid = str(sport.get("id"))
        flt = {**window, "eventTypeIds": [sid]}
        row = {"event_type_id": sid, "sport": sport.get("name"), "markets": et.get("marketCount")}
        try:
            types = data._read("listMarketTypes", {"filter": flt}) or []
            row["market_types"] = {t.get("marketType"): t.get("marketCount")
                                   for t in sorted(types, key=lambda t: -(t.get("marketCount") or 0))}
            cat = data._read("listMarketCatalogue", {"filter": flt, "maxResults": top, "sort": "MAXIMUM_TRADED",
                                                     "marketProjection": ["EVENT", "MARKET_DESCRIPTION"]}) or []
            matched = sorted((float(m.get("totalMatched") or 0.0) for m in cat), reverse=True)
            row["top_matched_total"] = round(sum(matched), 2)
            row["top_matched_median"] = round(matched[len(matched) // 2], 2) if matched else 0.0
            by_type: dict[str, float] = {}
            for m in cat:
                k = (m.get("description") or {}).get("marketType") or "?"
                by_type[k] = by_type.get(k, 0.0) + float(m.get("totalMatched") or 0.0)
            row["top_matched_by_type"] = {k: round(v, 2) for k, v in sorted(by_type.items(), key=lambda kv: -kv[1])}
            row["busiest"] = [f"{(m.get('event') or {}).get('name', '')}: {m.get('marketName', '')} "
                              f"({float(m.get('totalMatched') or 0):,.0f})" for m in cat[:5]]
        except Exception as exc:                          # one sport unread: the census goes on
            row["error"] = str(exc)[:200]
        out["sports"].append(row)
    out["sports"].sort(key=lambda r: -(r.get("top_matched_total") or 0.0))
    return out


def save_census(result: dict, day: date, s3=None, root: Path | str | None = None) -> str:
    """The census beside the day's record: on the server and in S3 (betfair_live/<day>/census_sports.json)."""
    body = json.dumps(result, indent=1, default=str).encode()
    path = Path(root or ROOT) / f"{day:%Y-%m-%d}" / "census_sports.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    key = f"{PREFIX}/{day:%Y-%m-%d}/census_sports.json"
    try:
        (s3 or s3_client()).put_object(Bucket=bucket(), Key=key, Body=body)
    except Exception as exc:
        log.warning("census not copied to S3 (%s); kept at %s", exc, path)
        return str(path)
    return f"s3://{bucket()}/{key}"


def antepost(data, rec: "DayRecorder", countries=("GB", "IE"), days_ahead: int = 400, now=None) -> dict:
    """One snapshot a day of every ante-post win market on GB/IE racing (the big races weeks or months ahead), its
    catalogue and its book, for the ante-post model (read-only)."""
    t = now or datetime.now(timezone.utc)
    cat = read_catalogue(data, t, t + timedelta(days=days_ahead), ("ANTEPOST_WIN",), countries,
                         window_hours=24.0 * 30)
    rec.record_catalogue(cat, t)
    raw = read_books(data, [m["marketId"] for m in cat if m.get("marketId")])
    n = rec.record_books(raw, source="antepost", polled_at=t)
    rec.maybe_upload(force=True)
    return {"markets": len(cat), "rows": n}


def read_books(data, market_ids: list[str], settled: bool = False) -> list[dict]:
    """The books of the markets; ``settled`` asks for the BSP with the prices (SP_AVAILABLE and SP_TRADED: SP_TRADED
    alone brought the money taken at SP but no BSP, ledger bsp-at-off-1007). Batches keep each call under Betfair's
    weight limit of 200 (a market: EX_BEST_OFFERS 5, SP_AVAILABLE 3, SP_TRADED 7)."""
    data_kinds = ["EX_BEST_OFFERS"] + (["SP_AVAILABLE", "SP_TRADED"] if settled else [])
    batch = 12 if settled else 25                      # 180 and 125: some room under the limit
    out = []
    for i in range(0, len(market_ids), batch):
        out += data._read("listMarketBook", {
            "marketIds": market_ids[i:i + batch],
            "priceProjection": {"priceData": data_kinds, "exBestOffersOverrides": {"bestPricesDepth": 3}}}) or []
    return out


def _interval(minutes_to_off: float, every_far: float, every_near: float, near_minutes: float) -> float:
    return every_near if minutes_to_off <= near_minutes else every_far


def record(day: date, until: datetime, data, rec: DayRecorder, every_far: float = 300.0, every_near: float = 60.0,
           near_minutes: float = 60.0, market_types=("WIN", "PLACE"), countries=("GB", "IE"),
           now=lambda: datetime.now(timezone.utc), sleep=time.sleep, refresh_minutes: float = 30.0,
           event_type: str = HORSE_RACING, wait_for_markets: bool = False) -> dict:
    """Snapshot the day's markets until ``until``: each market every ``every_far`` seconds, and every ``every_near``
    in its last ``near_minutes`` before the off; a market is left once it is in play or closed. The catalogue is read
    again every ``refresh_minutes`` for markets added in the day. With ``wait_for_markets`` a catalogue with nothing
    in it yet does not end the record: tomorrow's markets, recorded the evening before, are listed during the day."""
    from trading.exchange import uk_day_window
    start, end = uk_day_window(day)
    offs: dict[str, datetime] = {}
    last: dict[str, datetime] = {}
    done: set[str] = set()
    await_bsp: dict[str, int] = {}                        # markets at the off whose BSP is still to read: tries
    last_bsp: dict[str, datetime] = {}
    polls = books = 0

    def refresh() -> None:
        # a greyhound race every few minutes at each track: an hour's window keeps each call under its 100 markets
        cat = read_catalogue(data, start, end, market_types, countries, event_type=event_type,
                             window_hours=1.0 if event_type == GREYHOUNDS else 4.0)
        rec.record_catalogue(cat, now())
        for m in cat:
            t = _utc(m.get("marketStartTime"))
            if t is not None:
                offs.setdefault(m["marketId"], t)

    for attempt in range(3):                              # the first read: a passing fault is tried again
        try:
            refresh()
            break
        except Exception as exc:
            if attempt == 2:
                raise
            log.warning("catalogue not read (%s); trying again in 30 s", exc)
            sleep(30.0)
    next_refresh = now() + timedelta(minutes=refresh_minutes)
    log.info("Recording %d markets on %s until %s UTC", len(offs), day, until.strftime("%H:%M"))
    while now() < until:
        t = now()
        due = []
        for mid, off in offs.items():
            if mid in done:
                continue
            if t > off + timedelta(minutes=15):             # long past the off, whatever its status said
                done.add(mid)
                continue
            step = _interval((off - t).total_seconds() / 60.0, every_far, every_near, near_minutes)
            if mid not in last or (t - last[mid]).total_seconds() >= step - 1.0:
                due.append(mid)
        if due:
            try:
                raw = read_books(data, due)
            except Exception as exc:                        # one failed read: the next poll tries again
                log.warning("books not read (%s)", exc)
                raw = []
            books += rec.record_books(raw, source="recorder", polled_at=t)
            polls += 1
            for b in raw:
                mid = b.get("marketId")
                last[mid] = t
                if b.get("inplay") or b.get("status") == "CLOSED" or (
                        b.get("status") == "SUSPENDED" and t >= offs.get(mid, t)):
                    if mid not in done:
                        await_bsp.setdefault(mid, 0)
                    done.add(mid)
        # the BSP, read once the market has reconciled it at the off (a closed market's book carries none)
        bsp_due = [m for m, n in await_bsp.items()
                   if n < BSP_TRIES and (m not in last_bsp or (t - last_bsp[m]).total_seconds() >= BSP_EVERY - 1.0)]
        if bsp_due:
            try:
                raw_sp = read_books(data, bsp_due, settled=True)
            except Exception as exc:                        # the next pass tries again, BSP_TRIES in all
                log.warning("BSPs not read (%s)", exc)
                raw_sp = []
            for m in bsp_due:
                await_bsp[m] += 1
                last_bsp[m] = t
            got = {b.get("marketId") for b in raw_sp
                   if any(_num((r.get("sp") or {}).get("actualSP")) is not None for r in b.get("runners") or [])}
            if got:
                books += rec.record_books([b for b in raw_sp if b.get("marketId") in got], source="bsp", polled_at=t)
            for m in list(await_bsp):
                if m in got or await_bsp[m] >= BSP_TRIES or t > offs.get(m, t) + timedelta(minutes=15):
                    del await_bsp[m]
        if t >= next_refresh:
            try:
                refresh()
            except Exception as exc:
                log.warning("catalogue not read again (%s)", exc)
            next_refresh = t + timedelta(minutes=refresh_minutes)
        pending = [m for m in offs if m not in done]
        if not pending and not await_bsp and not (wait_for_markets and not offs):
            break
        nxt = min(((last[m] + timedelta(seconds=_interval((offs[m] - t).total_seconds() / 60.0, every_far, every_near,
                                                          near_minutes)) - t).total_seconds()
                   if m in last else 0.0 for m in pending), default=(next_refresh - t).total_seconds())
        if await_bsp:
            nxt = min(nxt, BSP_EVERY)
        sleep(min(60.0, max(5.0, nxt)))
    rec.maybe_upload(force=True)
    return {"markets": len(offs), "polls": polls, "rows": books, "left": len(offs) - len(done)}


def _bsp_on_file(rec: DayRecorder) -> dict:
    """(market id, selection id) -> the BSP read as its market reconciled at the off, from the day file's "bsp" rows."""
    out = {}
    try:
        with rec.books_path.open(newline="") as f:
            for r in csv.DictReader(f):
                if r.get("source") == "bsp" and _num(r.get("sp_actual")) is not None:
                    out[(str(r.get("market_id")), str(r.get("selection_id")))] = _num(r.get("sp_actual"))
    except OSError:
        pass
    return out


def final(day: date, data, rec: DayRecorder, market_ids: list[str] | None = None, now=None) -> int:
    """The settled books of the day's markets (status CLOSED: WINNER/LOSER/REMOVED, reduction factors). Betfair sends
    no BSP in a closed market's book, so each runner's is the one read as its market reconciled at the off (record()),
    where the day file has it."""
    if market_ids is not None:
        ids = market_ids
    elif getattr(rec, "tag", "") not in DAY_FILE_TAGS:     # a record of its own (other place markets, greyhounds)
        ids = sorted(rec._markets_on_file())
    else:
        ids = market_ids_on_file(day, rec.dir.parent)
    if not ids:
        return 0
    raw = read_books(data, ids, settled=True)
    known = _bsp_on_file(rec)
    for b in raw if known else []:
        for r in b.get("runners") or []:
            if not isinstance(r.get("sp"), dict):
                r["sp"] = {}
            if _num(r["sp"].get("actualSP")) is None:
                v = known.get((str(b.get("marketId")), str(r.get("selectionId"))))
                if v is not None:
                    r["sp"]["actualSP"] = v
    n = rec.record_books(raw, source="final", polled_at=(now or datetime.now(timezone.utc)))
    rec.maybe_upload(force=True)
    return n


# --------------------------------------------------------------------------------------------------------------------
# The nightly load into horse_racing.db
# --------------------------------------------------------------------------------------------------------------------

MARK_FIELDS = ["race_date", "market_id", "selection_id", "mark", "polled_utc", "minutes_to_off", "market_status",
               "runner_status", "back1", "back1_size", "back2", "back2_size", "back3", "back3_size", "lay1",
               "lay1_size", "lay2", "lay2_size", "lay3", "lay3_size", "last_traded", "market_total_matched",
               "active_runners", "bsp", "source"]
LIVE_MARKET_FIELDS = ["race_date"] + [f for f in MARKET_FIELDS if f != "recorded_utc"] + ["horse_norm",
                                                                                          "race_results_id"]


def ensure_tables(conn: sqlite3.Connection) -> None:
    from trading.session import LEDGER_FIELDS
    real = {"minutes_to_off", "back1", "back1_size", "back2", "back2_size", "back3", "back3_size", "lay1", "lay1_size",
            "lay2", "lay2_size", "lay3", "lay3_size", "last_traded", "market_total_matched", "bsp"}
    cols = ", ".join(f"{c} {'REAL' if c in real else 'INTEGER' if c in ('selection_id', 'active_runners') else 'TEXT'}"
                     for c in MARK_FIELDS)
    mcols = ", ".join(f"{c} {'INTEGER' if c in ('selection_id', 'sort_priority', 'race_results_id') else 'TEXT'}"
                      for c in LIVE_MARKET_FIELDS)
    ocols = ", ".join(f'"{c}" TEXT' for c in LEDGER_FIELDS)
    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS betfair_live_marks ({cols}, loaded_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (market_id, selection_id, mark));
        CREATE INDEX IF NOT EXISTS idx_blm_date ON betfair_live_marks(race_date);
        CREATE TABLE IF NOT EXISTS betfair_live_markets ({mcols}, loaded_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (market_id, selection_id));
        CREATE INDEX IF NOT EXISTS idx_blmk_date ON betfair_live_markets(race_date);
        CREATE INDEX IF NOT EXISTS idx_blmk_rr ON betfair_live_markets(race_results_id);
        CREATE TABLE IF NOT EXISTS live_orders (race_date TEXT, seq INTEGER, {ocols},
            loaded_at TEXT DEFAULT (datetime('now')), PRIMARY KEY (race_date, mode, seq));
    """)
    have = {r[1] for r in conn.execute("PRAGMA table_info(betfair_live_markets)")}
    for c in LIVE_MARKET_FIELDS:                # columns added to the record since the table was made
        if c not in have:
            kind = "INTEGER" if c in ("selection_id", "sort_priority", "race_results_id") else "TEXT"
            conn.execute(f"ALTER TABLE betfair_live_markets ADD COLUMN {c} {kind}")
    conn.commit()


def marks(books, markets, day: date):
    """For each runner: the book nearest to each clock mark of the morning (within 5 minutes, before its off), to each
    mark before the off (within a quarter of the mark, at least a minute), the last book before the off, and the
    settled one."""
    import pandas as pd
    if books is None or books.empty or markets is None or markets.empty:
        return pd.DataFrame(columns=MARK_FIELDS)
    offs = markets.drop_duplicates("market_id").set_index("market_id")["market_start_utc"].map(_utc)
    b = books.copy()
    b["polled"] = pd.to_datetime(b["polled_utc"], utc=True, errors="coerce")
    b["off"] = pd.to_datetime(b["market_id"].map(offs), utc=True, errors="coerce")
    b = b.dropna(subset=["polled", "off"])
    b["minutes_to_off"] = (b["off"] - b["polled"]).dt.total_seconds() / 60.0
    pre = b[(b["source"] != "final") & (b["status"] == "OPEN") & (b["inplay"].astype(str).isin(["0", "False", "false"]))
            & (b["minutes_to_off"] > 0)]
    picks = []
    for hhmm in CLOCK_MARKS:
        target = pd.Timestamp(uk_time_on(day, hhmm))
        d = pre.assign(gap=(pre["polled"] - target).abs().dt.total_seconds() / 60.0)
        d = d[d["gap"] <= 5.0]
        if len(d):
            picks.append(d.sort_values("gap").drop_duplicates(["market_id", "selection_id"]).assign(mark=f"clock_{hhmm}"))
    for m in OFF_MARKS:
        tol = max(1.0, 0.25 * m)
        d = pre.assign(gap=(pre["minutes_to_off"] - m).abs())
        d = d[d["gap"] <= tol]
        if len(d):
            picks.append(d.sort_values("gap").drop_duplicates(["market_id", "selection_id"]).assign(mark=f"off_{m}"))
    if len(pre):
        picks.append(pre.sort_values("polled").drop_duplicates(["market_id", "selection_id"], keep="last")
                     .assign(mark="last"))
    fin = b[(b["source"] == "final") & (b["status"] == "CLOSED")]      # read before a market settled: not final
    if len(fin):
        picks.append(fin.sort_values("polled").drop_duplicates(["market_id", "selection_id"], keep="last")
                     .assign(mark="final"))
    if not picks:
        return pd.DataFrame(columns=MARK_FIELDS)
    out = pd.concat(picks, ignore_index=True)
    out["race_date"] = f"{day:%Y-%m-%d}"
    out["market_status"] = out["status"]
    out["active_runners"] = pd.to_numeric(out["number_of_active_runners"], errors="coerce")
    out["bsp"] = pd.to_numeric(out["sp_actual"], errors="coerce")
    # the settled book of a closed market carries no BSP: the one read as the market reconciled it at the off, by the
    # recorder or by the trader settling its races
    sp = b[b["source"].isin(["bsp", "final", "trader"])].copy()
    sp["_sp"] = pd.to_numeric(sp["sp_actual"], errors="coerce")
    sp = sp.dropna(subset=["_sp"]).sort_values("polled").groupby(["market_id", "selection_id"])["_sp"].last()
    fill = out["mark"].eq("final") & out["bsp"].isna()
    if fill.any() and len(sp):
        out.loc[fill, "bsp"] = [sp.get((m, s)) for m, s in zip(out.loc[fill, "market_id"],
                                                               out.loc[fill, "selection_id"])]
    return out[MARK_FIELDS]


def match_markets(markets, conn: sqlite3.Connection, day: date):
    """Each catalogue runner's race_results row, by track, UK time and horse (then track and horse)."""
    import pandas as pd
    from betfair_prices import db_time_to_24h, normalise_horse, normalise_track, utc_iso_to_uk_hhmm
    m = markets.copy()
    for c in MARKET_FIELDS:
        if c not in m.columns:
            m[c] = None
    # one row a runner: the one with the card's metadata when the market was catalogued twice, else the last
    m["_md"] = m[["cloth_number", "stall_draw", "jockey", "trainer"]].notna().any(axis=1)
    m = m.sort_values("_md", kind="stable").drop_duplicates(["market_id", "selection_id"], keep="last")
    m = m.drop(columns="_md")
    m["race_date"] = f"{day:%Y-%m-%d}"
    m["horse_norm"] = m["runner_name"].map(normalise_horse)
    try:
        rr = pd.read_sql_query("SELECT id, race_time, track, horse_name FROM race_results WHERE race_date = ?", conn,
                               params=[f"{day:%Y-%m-%d}"])
    except Exception:                                   # no race_results table (a test database)
        rr = pd.DataFrame(columns=["id", "race_time", "track", "horse_name"])
    m["race_results_id"] = None
    if len(rr):
        rr["k"] = list(zip(rr["track"].map(normalise_track), rr["race_time"].map(db_time_to_24h),
                           rr["horse_name"].map(normalise_horse)))
        rr["k2"] = list(zip(rr["track"].map(normalise_track), rr["horse_name"].map(normalise_horse)))
        full = rr.drop_duplicates("k").set_index("k")["id"]
        two = rr.groupby("k2")["id"].agg(["first", "size"])
        two = two[two["size"] == 1]["first"]
        keys = list(zip(m["venue"].map(normalise_track), m["market_start_utc"].map(utc_iso_to_uk_hhmm), m["horse_norm"]))
        keys2 = list(zip(m["venue"].map(normalise_track), m["horse_norm"]))
        m["race_results_id"] = [full.get(k, two.get(k2)) for k, k2 in zip(keys, keys2)]
    return m[LIVE_MARKET_FIELDS]


def _upsert(conn: sqlite3.Connection, table: str, df) -> int:
    if df is None or len(df) == 0:
        return 0
    cols = list(df.columns)
    rows = df.astype(object).where(df.notna(), None).values.tolist()
    conn.executemany(f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", rows)
    return len(rows)


def _read_day_csv(day: date, name: str, s3=None, root: Path | None = None):
    """A day file: the server's copy when there is one, else the S3 copy (gzipped); None when neither exists."""
    import pandas as pd
    local = Path(root or ROOT) / f"{day:%Y-%m-%d}" / name
    if local.exists():
        return pd.read_csv(local, dtype={"market_id": str}, on_bad_lines="skip")
    try:
        obj = (s3 or s3_client()).get_object(Bucket=bucket(), Key=f"{PREFIX}/{day:%Y-%m-%d}/{name}.gz")
    except Exception as exc:
        if "NoSuchKey" not in str(exc) and "404" not in str(exc):
            log.warning("%s for %s not read from S3 (%s)", name, day, exc)
        return None
    return pd.read_csv(io.BytesIO(gzip.decompress(obj["Body"].read())), dtype={"market_id": str}, on_bad_lines="skip")


def _read_ledger(day: date, mode: str = "live", s3=None):
    import pandas as pd
    try:
        obj = (s3 or s3_client()).get_object(Bucket=bucket(), Key=f"trading/{mode}/{day:%Y-%m-%d}/ledger.csv")
    except Exception:
        return None
    return pd.read_csv(io.BytesIO(obj["Body"].read()), dtype=str, keep_default_na=False)


def _read_day_files(day: date, stem: str, s3=None, root: Path | None = None):
    """A day's file from every writer (the recorder's and the trader's), as one frame; None when there is none."""
    import pandas as pd
    parts = [f for tag in DAY_FILE_TAGS
             if (f := _read_day_csv(day, f"{stem}{'_' + tag if tag else ''}.csv", s3, root)) is not None and len(f)]
    return pd.concat(parts, ignore_index=True) if parts else None


def load_day(conn: sqlite3.Connection, day: date, s3=None, root: Path | None = None) -> dict:
    from trading.session import LEDGER_FIELDS
    out = {"day": f"{day:%Y-%m-%d}", "markets": 0, "marks": 0, "orders": 0}
    markets = _read_day_files(day, "markets", s3, root)
    books = _read_day_files(day, "books", s3, root)
    if markets is not None and len(markets):
        out["markets"] = _upsert(conn, "betfair_live_markets", match_markets(markets, conn, day))
        out["marks"] = _upsert(conn, "betfair_live_marks", marks(books, markets, day))
    ledger = _read_ledger(day, "live", s3)
    if ledger is not None and len(ledger):
        rows = ledger.reindex(columns=LEDGER_FIELDS)
        rows["mode"] = "live"
        rows.insert(0, "seq", range(len(rows)))
        rows.insert(0, "race_date", f"{day:%Y-%m-%d}")
        conn.execute("DELETE FROM live_orders WHERE race_date = ? AND mode = 'live'", [f"{day:%Y-%m-%d}"])
        out["orders"] = _upsert(conn, "live_orders", rows)
    conn.commit()
    return out


def load_db(db_path: str, days: int = 3, day_to: date | None = None, s3=None, root: Path | None = None) -> list[dict]:
    day_to = day_to or uk_today()
    conn = sqlite3.connect(db_path)
    try:
        ensure_tables(conn)
        done = []
        for k in range(days - 1, -1, -1):
            d = day_to - timedelta(days=k)
            r = load_day(conn, d, s3, root)
            log.info("live record %s: %d catalogue rows, %d marks, %d orders", r["day"], r["markets"], r["marks"],
                     r["orders"])
            done.append(r)
        return done
    finally:
        conn.close()


# --------------------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------------------

class Stopped(SystemExit):
    """The record was asked to stop (SIGTERM): the recorder giving way to the trader (scripts/give_way.py), or the job
    cancelled. A clean exit."""


@contextlib.contextmanager
def _stopped_by_sigterm():
    """While recording, SIGTERM ends the record at once and cleanly: the signal raises in the main thread between two
    statements, so no row is cut in half, and the day's files are copied to S3 on the way out (main)."""
    def stop(signum, frame):
        raise Stopped(0)
    try:
        old = signal.signal(signal.SIGTERM, stop)
    except ValueError:                                   # not the main thread: the default stands
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, old)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record", action="store_true", help="snapshot the day's markets until --until (UK)")
    ap.add_argument("--final", action="store_true", help="record the day's settled books (BSP, results)")
    ap.add_argument("--upload", action="store_true", help="copy the day's files to S3 now")
    ap.add_argument("--load-db", action="store_true", help="load the last --days of the record into --db")
    ap.add_argument("--census", action="store_true",
                    help="print every market type Betfair lists on the --date's racing, with examples (read-only)")
    ap.add_argument("--census-sports", action="store_true",
                    help="every sport's markets and matched money on the --date, saved beside the day's record")
    ap.add_argument("--antepost", action="store_true", help="snapshot every GB/IE ante-post win market once")
    ap.add_argument("--event-type", default=HORSE_RACING, help="Betfair event type: 7 horse racing, 4339 greyhounds")
    ap.add_argument("--date", default=None, help="UK racing day, YYYY-MM-DD (default today)")
    ap.add_argument("--until", default="21:30", help="stop recording at this UK time")
    ap.add_argument("--for-minutes", type=float, default=None,
                    help="stop recording this many minutes from now instead (--until is read on --date, so an "
                         "evening record of tomorrow's markets needs this)")
    ap.add_argument("--tag", default=None,
                    help="write the record to its own files (books_<tag>.csv): tomorrow's markets recorded the "
                         "evening before (--tag evening), the other place markets (other_place), the greyhounds "
                         "(greyhound, the default with --event-type 4339); the nightly load reads the untagged and "
                         "the trader's files only")
    ap.add_argument("--evening", default=None, metavar="HH:MM-HH:MM",
                    help="record --date's markets the evening before, between these UK times on the day before "
                         "(--date tomorrow --tag evening --evening 17:00-21:30): waits for the start, and goes on "
                         "while nothing is listed yet")
    ap.add_argument("--every", type=float, default=300.0, help="seconds between snapshots of a market")
    ap.add_argument("--every-near", type=float, default=60.0, help="... in its last --near minutes")
    ap.add_argument("--near", type=float, default=60.0)
    ap.add_argument("--market-types", default="WIN,PLACE")
    ap.add_argument("--countries", default="GB,IE")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--db", default=DEFAULT_DB)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    day = date.fromisoformat(a.date) if a.date else uk_today()

    if a.load_db:
        for r in load_db(a.db, a.days, day):
            print(r)
        return 0
    if a.census:
        from trading.exchange import BetfairData
        data = BetfairData()
        data.login()
        for kind, v in census(data, day, tuple(a.countries.split(","))).items():
            print(kind, v)
        return 0
    if a.census_sports:
        from trading.exchange import BetfairData
        data = BetfairData()
        data.login()
        got = census_sports(data)
        for r in got["sports"]:
            print(r.get("sport"), {k: r.get(k) for k in ("markets", "top_matched_total", "top_matched_median")},
                  dict(list((r.get("market_types") or {}).items())[:8]))
        print("saved:", save_census(got, day))
        return 0
    if a.antepost:
        from trading.exchange import BetfairData
        data = BetfairData()
        data.login()
        rec = DayRecorder(day, tag=a.tag or "antepost")
        print(antepost(data, rec, tuple(a.countries.split(","))))
        return 0 if rec.errors == 0 else 2
    if not (a.record or a.final or a.upload):
        ap.print_help()
        return 1
    window = None
    if a.evening:
        if not (a.record and a.tag) or a.final:
            ap.error("--evening records (--record) to its own files (--tag), and the markets are not settled yet")
        try:
            window = evening_window(day, a.evening)
        except ValueError as exc:
            ap.error(str(exc))
        now = datetime.now(timezone.utc)
        if now >= window[1]:
            print({"evening": "over", "until": window[1].isoformat()})
            return 0
        wait = (window[0] - now).total_seconds()
        if wait > 0:                                   # before the login: an idle session would lapse
            log.info("Waiting %.0f minutes for the evening's record of %s", wait / 60.0, day)
            time.sleep(wait)
    tag = a.tag if a.tag is not None else ("greyhound" if a.event_type == GREYHOUNDS else "")
    if a.event_type != HORSE_RACING and not tag:
        ap.error("a sport other than horse racing needs its own --tag: the nightly load reads the untagged files")
    rec = DayRecorder(day, tag=tag)
    if a.upload:
        rec.maybe_upload(force=True)
    if a.record or a.final:
        from trading.exchange import BetfairData
        with _stopped_by_sigterm():
            try:
                data = BetfairData()
                data.login()
                if a.record:
                    if window:
                        until = window[1]
                    elif a.for_minutes:
                        until = datetime.now(timezone.utc) + timedelta(minutes=a.for_minutes)
                    else:
                        until = uk_time_on(day, a.until)
                    sport = {} if a.event_type == HORSE_RACING else {"event_type": a.event_type}
                    print(record(day, until, data, rec, a.every, a.every_near, a.near,
                                 tuple(a.market_types.split(",")), tuple(a.countries.split(",")),
                                 wait_for_markets=window is not None, **sport))
                if a.final:
                    print({"final": final(day, data, rec)})
            except Stopped:
                log.info("Asked to stop (the trader comes first, or the job was cancelled): the day's files go to S3")
                rec.maybe_upload(force=True)
                raise
    return 0 if rec.errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
