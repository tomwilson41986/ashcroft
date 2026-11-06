"""Every football match on Betfair, recorded and paper-traded from the feed the football trader reads.

The football trader's own process (``python -m football.live --trade --all-games``) runs this beside its decisions,
on the live application key (the owner, 4 Nov 2026: "use live key"; until then the delayed key unless
``FOOTBALL_LIVE=yes``). Every data request goes through ``Throttle``, under Betfair's limits (the owner: "do not pass
betfair thresholds"). This module itself places nothing; real bets stay with ``FOOTBALL_LIVE=yes``.

What it does, every loop:

1. **Catalogue** (every 15 minutes): every soccer market of the chosen types (MATCH_ODDS, OVER_UNDER_25 and
   TO_QUALIFY, DOUBLE_CHANCE, DRAW_NO_BET by default) starting in the next ``hours`` hours, every country and competition. Windows that hit Betfair's 1,000
   results are split in two until none does.
2. **Books**, each market on its own cadence, tighter towards the off:

   | Minutes to the off | Every |
   |---|---|
   | over 360 | 30 min |
   | 120-360 | 10 min |
   | 60-120 | 5 min |
   | 15-60 | 2 min |
   | 0-15 | 1 min |

   Each read asks for ``depth`` levels a side (EX_BEST_OFFERS) and the traded ladder (EX_TRADED), batched under
   Betfair's 200-point weight per request. After the off: one read when the market first shows in play (the close is
   the last book before it) and reads every 30 minutes from 100 minutes after the start until Betfair shows the
   result (each runner WINNER or LOSER), or 6 hours pass.
3. **The record**: one row a runner a read. Status, in play, market and runner matched money, last traded price,
   ``depth`` back and lay levels with sizes, and the traded money at each price since the previous read
   (``traded_delta``, "price:GBP|..."). Written to ``~/.ashcroft/football/exchange/<match day>/`` and to
   ``s3://$CAPTURE_BUCKET/sources/football/exchange/<match day>/`` (``books_<UTC hour>.csv.gz`` chunks and
   ``markets.csv.gz``). The match day is the UK date of the market's start.
4. **The paper book** (``football.paper_book``): every selection of every market, backed and laid at the executable
   price (walked down the depth) for several stakes at fixed checkpoints before the off, then valued at the close and
   settled on Betfair's own result.

    python -m football.live --trade --all-games --until 21:30
    python -m football.exchange_feed --summary --days 3
"""

from __future__ import annotations

import csv
import gzip
import io
import logging
import math
import os
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)

PREFIX = "football/exchange"
LOCAL = Path(os.path.expanduser("~/.ashcroft/football/exchange"))
SOCCER = "1"
#: TO_QUALIFY: cup ties and two-legged European ties (the owner, 6 Nov 2026: "record cup markets, and games in Europe").
#: Cups and UEFA matches' MATCH_ODDS and OVER_UNDER_25 were already in: the catalogue filters on soccer alone, every
#: country and competition
#: DOUBLE_CHANCE and DRAW_NO_BET: the cross-market read (6 Nov 2026, reports/football_model_rethink.md) found them
#: following match odds on the last traded prices; their books (quotes and depth) say whether that lag can be traded
DEFAULT_TYPES = ("MATCH_ODDS", "OVER_UNDER_25", "TO_QUALIFY", "DOUBLE_CHANCE", "DRAW_NO_BET")
#: (minutes to the off at or above which, seconds between reads)
CADENCE = ((360.0, 1800.0), (120.0, 600.0), (60.0, 300.0), (15.0, 120.0), (0.0, 60.0))
RESULT_FROM_MIN = 100.0          # minutes after the start before the result is asked for
RESULT_EVERY_S = 1800.0
RESULT_GIVE_UP_H = 6.0
WEIGHT_LIMIT = 200.0
CATALOGUE_MAX = 1000


#: our own ceiling on the feed's Betfair data requests (the owner, 4 Nov 2026: "use live key ... do not pass betfair
#: thresholds"). Betfair's published data limits are 200 weight points a request (batch_size keeps under them) and at
#: most 5 listMarketBook calls a second to one market (the cadence reads a market at most once a minute); it publishes
#: no account-wide data rate, so the feed holds itself to MAX_CALLS_PER_SECOND overall and MAX_CALLS_PER_HOUR.
MAX_CALLS_PER_SECOND = 5.0
MAX_CALLS_PER_HOUR = 12000


class Throttle:
    """A data-request wrapper that never lets the feed exceed ``per_second`` calls a second or ``per_hour`` in any
    rolling hour (it waits instead), and counts the calls (``hourly``: {UTC hour: calls}) for the run's report."""

    def __init__(self, read, per_second: float = MAX_CALLS_PER_SECOND, per_hour: int = MAX_CALLS_PER_HOUR,
                 clock=time.monotonic, sleep=time.sleep, wall=lambda: datetime.now(timezone.utc)):
        self.read, self.gap, self.per_hour = read, 1.0 / per_second, per_hour
        self.clock, self.sleep, self.wall = clock, sleep, wall
        self.last = None
        self.window: deque = deque()
        self.calls, self.waited_s, self.hourly = 0, 0.0, {}

    def __call__(self, *args, **kwargs):
        while True:
            now = self.clock()
            while self.window and self.window[0] <= now - 3600.0:
                self.window.popleft()
            wait = 0.0
            if self.last is not None:
                wait = max(wait, self.last + self.gap - now)
            if len(self.window) >= self.per_hour:
                wait = max(wait, self.window[0] + 3600.0 - now)
            if wait <= 0:
                break
            wait += 1e-3                     # past the edge, so the oldest call has left the hour when we look again
            self.sleep(wait)
            self.waited_s += wait
        self.last = now
        self.window.append(now)
        self.calls += 1
        h = self.wall().strftime("%Y-%m-%dT%H")
        self.hourly[h] = self.hourly.get(h, 0) + 1
        return self.read(*args, **kwargs)

    def report(self) -> dict:
        return {"calls": self.calls, "waited_s": round(self.waited_s, 1), "max_per_hour": max(self.hourly.values(), default=0),
                "limit_per_second": round(1.0 / self.gap, 2), "limit_per_hour": self.per_hour}


def cadence_seconds(minutes_to_off: float, cadence=CADENCE) -> float:
    for floor, every in cadence:
        if minutes_to_off >= floor:
            return every
    return cadence[-1][1]


def weight_per_market(depth: int, traded: bool) -> float:
    """Betfair's listMarketBook weight a market: EX_BEST_OFFERS 5, with EX_TRADED 20; scaled by depth / 3 when the
    depth is overridden (Market Data Request Limits)."""
    return (20.0 if traded else 5.0) * max(1.0, depth / 3.0)


def batch_size(depth: int, traded: bool, limit: float = WEIGHT_LIMIT) -> int:
    return max(1, int(math.floor(limit / weight_per_market(depth, traded) + 1e-9)))


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc(text) -> datetime | None:
    try:
        return datetime.strptime(str(text)[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def uk_date(t: datetime) -> str:
    from zoneinfo import ZoneInfo
    return t.astimezone(ZoneInfo("Europe/London")).date().isoformat()


# --------------------------------------------------------------------------------------------------------------------
# Catalogue
# --------------------------------------------------------------------------------------------------------------------

#: Betfair's listMarketCatalogue weights (Market Data Request Limits): sum(weight) x maxResults must stay within 200
#: points a request. MARKET_DESCRIPTION (1) at 1,000 results was 1,000 points: every call refused TOO_MUCH_DATA (the
#: first run on the football runner, 4 Nov 2026). The type comes from asking one type a call instead.
CATALOGUE_WEIGHTS = {"EVENT": 0, "COMPETITION": 0, "MARKET_START_TIME": 0, "RUNNER_DESCRIPTION": 0,
                     "EVENT_TYPE": 0, "MARKET_DESCRIPTION": 1, "RUNNER_METADATA": 1}
CATALOGUE_PROJECTION = ["EVENT", "COMPETITION", "MARKET_START_TIME", "RUNNER_DESCRIPTION"]


def catalogue_limit(projection=CATALOGUE_PROJECTION, wanted: int = CATALOGUE_MAX,
                    limit: float = WEIGHT_LIMIT) -> int:
    """The most results a catalogue call may ask for under Betfair's 200 points (and its 1,000 cap)."""
    w = sum(CATALOGUE_WEIGHTS.get(p, 1) for p in projection)
    return min(wanted, CATALOGUE_MAX) if w == 0 else max(1, min(wanted, CATALOGUE_MAX, int(limit // w)))


def catalogue_all(read, start: datetime, end: datetime, types=DEFAULT_TYPES, max_results: int = CATALOGUE_MAX,
                  min_window: timedelta = timedelta(minutes=10)) -> list[dict]:
    """Every soccer market of ``types`` starting in [start, end), one type a call (each market carries its
    ``marketType``): windows answered with ``max_results`` markets are split in two (Betfair answers at most 1,000 a
    call), down to ``min_window``. Each call is held under Betfair's 200 points. ``read(method, params)``."""
    out, seen = [], set()
    max_results = catalogue_limit(CATALOGUE_PROJECTION, max_results)

    def window(a: datetime, b: datetime, mtype: str) -> None:
        raw = read("listMarketCatalogue", {
            "filter": {"eventTypeIds": [SOCCER], "marketTypeCodes": [mtype], "inPlayOnly": False,
                       "marketStartTime": {"from": _iso(a), "to": _iso(b)}},
            "marketProjection": list(CATALOGUE_PROJECTION),
            "maxResults": max_results, "sort": "FIRST_TO_START"}) or []
        if len(raw) >= max_results and b - a > min_window:
            mid = a + (b - a) / 2
            window(a, mid, mtype)
            window(mid, b, mtype)
            return
        if len(raw) >= max_results:
            log.warning("catalogue: %d %s markets in %s-%s, the most Betfair answers; some may be missing",
                        len(raw), mtype, _iso(a), _iso(b))
        for m in raw:
            if m.get("marketId") not in seen:
                seen.add(m.get("marketId"))
                m.setdefault("marketType", mtype)
                out.append(m)

    for mtype in types:
        window(start, end, mtype)
    return out


MARKET_FIELDS = ["recorded_utc", "market_id", "market_type", "market_name", "market_start_utc", "event_id",
                 "event_name", "country", "competition_id", "competition", "selection_id", "runner_name",
                 "sort_priority", "handicap"]


def market_rows(raw: list[dict], recorded: datetime) -> list[dict]:
    """The catalogue, one row a runner (the columns the football paper trials read: market_type, event_name,
    country, selection_id, runner_name, market_start_utc)."""
    rows = []
    for m in raw:
        ev, comp, desc = m.get("event") or {}, m.get("competition") or {}, m.get("description") or {}
        for r in m.get("runners") or []:
            rows.append({"recorded_utc": _iso(recorded), "market_id": m.get("marketId"),
                         "market_type": desc.get("marketType") or m.get("marketType") or "",
                         "market_name": m.get("marketName"), "market_start_utc": m.get("marketStartTime"),
                         "event_id": ev.get("id"), "event_name": ev.get("name"), "country": ev.get("countryCode"),
                         "competition_id": comp.get("id"), "competition": comp.get("name"),
                         "selection_id": r.get("selectionId"), "runner_name": r.get("runnerName"),
                         "sort_priority": r.get("sortPriority"), "handicap": r.get("handicap")})
    return rows


# --------------------------------------------------------------------------------------------------------------------
# Books
# --------------------------------------------------------------------------------------------------------------------

def book_fields(depth: int) -> list[str]:
    f = ["polled_utc", "market_id", "status", "inplay", "complete", "is_delayed", "market_total_matched",
         "selection_id", "runner_status", "last_traded", "runner_total_matched", "traded_total", "traded_delta"]
    for side in ("back", "lay"):
        for i in range(1, depth + 1):
            f += [f"{side}{i}", f"{side}{i}_size"]
    return f


def traded_delta(prev: dict[float, float], now: dict[float, float]) -> dict[float, float]:
    """The money traded at each price since the previous read (Betfair's ladder is cumulative)."""
    out = {}
    for p, v in now.items():
        d = v - prev.get(p, 0.0)
        if d > 0.005:
            out[p] = round(d, 2)
    return out


def _fmt_ladder(lad: dict[float, float]) -> str:
    return "|".join(f"{p:g}:{v:.2f}" for p, v in sorted(lad.items()))


def parse_ladder(text) -> dict[float, float]:
    if not isinstance(text, str) or not text:
        return {}
    out = {}
    for part in text.split("|"):
        p, _, v = part.partition(":")
        try:
            out[float(p)] = float(v)
        except ValueError:
            continue
    return out


def book_rows(raw_books: list[dict], polled: datetime, depth: int, prev: dict) -> list[dict]:
    """listMarketBook's reply as one row a runner; ``prev`` ((market, selection) -> traded ladder) is updated."""
    rows = []
    for b in raw_books or []:
        base = {"polled_utc": _iso(polled), "market_id": b.get("marketId"), "status": b.get("status"),
                "inplay": int(bool(b.get("inplay"))), "complete": int(bool(b.get("complete"))),
                "is_delayed": int(bool(b.get("isMarketDataDelayed"))),
                "market_total_matched": _num(b.get("totalMatched"))}
        for r in b.get("runners") or []:
            ex = r.get("ex") or {}
            key = (str(b.get("marketId")), int(r.get("selectionId")))
            lad = {float(o["price"]): float(o["size"]) for o in ex.get("tradedVolume") or []
                   if _num(o.get("price")) and _num(o.get("size")) is not None}
            delta = traded_delta(prev.get(key, {}), lad) if key in prev else {}
            if lad:
                prev[key] = lad
            row = dict(base, selection_id=r.get("selectionId"), runner_status=r.get("status"),
                       last_traded=_num(r.get("lastPriceTraded")), runner_total_matched=_num(r.get("totalMatched")),
                       traded_total=round(sum(lad.values()), 2) if lad else None, traded_delta=_fmt_ladder(delta))
            for side, k in (("back", "availableToBack"), ("lay", "availableToLay")):
                lv = ex.get(k) or []
                for i in range(depth):
                    o = lv[i] if i < len(lv) else {}
                    row[f"{side}{i + 1}"], row[f"{side}{i + 1}_size"] = _num(o.get("price")), _num(o.get("size"))
            rows.append(row)
    return rows


# --------------------------------------------------------------------------------------------------------------------
# Day files
# --------------------------------------------------------------------------------------------------------------------

class DayWriter:
    """Rows appended to local CSVs by match day (books in UTC-hour chunks), copied gzipped to S3 on ``upload``."""

    def __init__(self, store=None, root: Path = LOCAL, depth: int = 5):
        self.store, self.root, self.depth = store, Path(root), depth
        self.dirty: set[Path] = set()

    def _append(self, path: Path, fields: list[str], rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
            if new:
                w.writeheader()
            w.writerows(rows)
        self.dirty.add(path)

    def books(self, rows: list[dict], day_of: dict[str, str], polled: datetime) -> None:
        by_day: dict[str, list] = {}
        for r in rows:
            by_day.setdefault(day_of.get(str(r["market_id"]), uk_date(polled)), []).append(r)
        for day, part in by_day.items():
            self._append(self.root / day / f"books_{polled:%Y%m%d%H}.csv", book_fields(self.depth), part)

    def markets(self, rows: list[dict], day_of: dict[str, str]) -> None:
        by_day: dict[str, list] = {}
        for r in rows:
            by_day.setdefault(day_of.get(str(r["market_id"])), []).append(r)
        for day, part in by_day.items():
            if day:
                self._append(self.root / day / "markets.csv", MARKET_FIELDS, part)

    def put(self, day: str, name: str, data: bytes) -> None:
        p = self.root / day / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        self.dirty.add(p)

    def upload(self) -> int:
        if self.store is None:
            self.dirty.clear()
            return 0
        n = 0
        for p in sorted(self.dirty):
            key = f"{PREFIX}/{p.parent.name}/{p.name}"
            try:
                data = p.read_bytes()
                self.store.put(key + ".gz" if p.suffix == ".csv" else key,
                               gzip.compress(data) if p.suffix == ".csv" else data)
                n += 1
            except Exception as exc:              # the server's copy stays; the next upload retries
                log.warning("S3: %s not copied (%s)", key, str(exc)[:200])
                continue
            self.dirty.discard(p)
        return n


def read_day(store, day: str, stem: str, root: Path | None = None):
    """Every chunk of one stem ("books" or "markets") of a match day, as one frame: the server's copy, else S3."""
    import pandas as pd
    parts = []
    local = Path(root or LOCAL) / day
    if local.exists():
        parts = [pd.read_csv(p, dtype={"market_id": str}) for p in sorted(local.glob(f"{stem}*.csv"))]
    elif store is not None:
        for key in sorted(store.listing(f"{PREFIX}/{day}/")):
            if not Path(key).name.startswith(stem):
                continue
            raw = store.get(key)
            if raw:
                data = gzip.decompress(raw) if key.endswith(".gz") else raw
                parts.append(pd.read_csv(io.BytesIO(data), dtype={"market_id": str}))
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else None


# --------------------------------------------------------------------------------------------------------------------
# The feed
# --------------------------------------------------------------------------------------------------------------------

class Feed:
    """The all-games record: which markets are due, the reads, the rows, the hand-off to the paper book."""

    def __init__(self, read, writer: DayWriter | None = None, types=DEFAULT_TYPES, hours: float = 24.0,
                 depth: int = 5, traded: bool = True, paper=None, catalogue_seconds: float = 900.0):
        self.read, self.writer, self.types, self.hours = read, writer, tuple(types), hours
        self.depth, self.traded, self.paper = depth, traded, paper
        self.catalogue_seconds = catalogue_seconds
        self.markets: dict[str, dict] = {}       # market_id -> {start, day, meta}
        self.next_due: dict[str, datetime] = {}
        self.inplay_seen: set[str] = set()
        self.settled: set[str] = set()
        self.prev: dict = {}
        self.last_catalogue: datetime | None = None
        self.reads = 0
        self.rows = 0

    def refresh(self, now: datetime) -> int:
        raw = catalogue_all(self.read, now - timedelta(hours=2), now + timedelta(hours=self.hours), self.types)
        new = [m for m in raw if m.get("marketId") not in self.markets]
        day_of = {}
        for m in new:
            st = _utc(m.get("marketStartTime"))
            if st is None:
                continue
            mid = m["marketId"]
            day_of[mid] = uk_date(st)
            self.markets[mid] = {"start": st, "day": day_of[mid], "meta": m}
            self.next_due[mid] = now
        if self.writer is not None and new:
            self.writer.markets(market_rows([m for m in new if m["marketId"] in day_of], now), day_of)
        if self.paper is not None:
            for m in new:
                if m.get("marketId") in self.markets:
                    self.paper.add_market(m)
        self.last_catalogue = now
        return len(new)

    def due(self, now: datetime) -> tuple[list[str], list[str]]:
        """(markets due a priced read, markets due a result read)."""
        priced, result = [], []
        for mid, mk in self.markets.items():
            if mid in self.settled or self.next_due.get(mid, now) > now:
                continue
            mins = (mk["start"] - now).total_seconds() / 60.0
            if mid not in self.inplay_seen and mins > -15:
                priced.append(mid)
            elif mins <= -RESULT_FROM_MIN:
                if -mins > RESULT_GIVE_UP_H * 60:
                    self.settled.add(mid)
                else:
                    result.append(mid)
        return priced, result

    def _schedule(self, mid: str, now: datetime) -> None:
        mk = self.markets[mid]
        mins = (mk["start"] - now).total_seconds() / 60.0
        if mid in self.inplay_seen or mins <= -15:
            nxt = max(now + timedelta(seconds=RESULT_EVERY_S), mk["start"] + timedelta(minutes=RESULT_FROM_MIN))
        else:
            nxt = now + timedelta(seconds=cadence_seconds(mins))
        self.next_due[mid] = nxt

    def poll(self, now: datetime) -> list[dict]:
        priced, result = self.due(now)
        raw_all = []
        n = batch_size(self.depth, self.traded)
        data = ["EX_BEST_OFFERS"] + (["EX_TRADED"] if self.traded else [])
        for i in range(0, len(priced), n):
            raw = self.read("listMarketBook", {
                "marketIds": priced[i:i + n],
                "priceProjection": {"priceData": data, "exBestOffersOverrides": {"bestPricesDepth": self.depth}}})
            self.reads += 1
            raw_all += raw or []
        for i in range(0, len(result), 40):              # no price projection: the status and the runners' result
            raw = self.read("listMarketBook", {"marketIds": result[i:i + 40]})
            self.reads += 1
            raw_all += raw or []
        for b in raw_all:
            mid = b.get("marketId")
            if mid not in self.markets:
                continue
            if b.get("inplay"):
                self.inplay_seen.add(mid)
            if b.get("status") == "CLOSED" or any(r.get("status") in ("WINNER", "LOSER")
                                                   for r in b.get("runners") or []):
                if b.get("status") == "CLOSED":
                    self.settled.add(mid)
        for mid in set(priced) | set(result):
            self._schedule(mid, now)
        rows = book_rows(raw_all, now, self.depth, self.prev)
        if self.writer is not None and rows:
            self.writer.books(rows, {k: v["day"] for k, v in self.markets.items()}, now)
        if self.paper is not None:
            for b in raw_all:
                if b.get("marketId") in self.markets:
                    self.paper.on_book(b, self.markets[b["marketId"]]["start"], now)
        self.rows += len(rows)
        return rows

    def step(self, now: datetime | None = None) -> dict:
        now = now or datetime.now(timezone.utc)
        new = 0
        if self.last_catalogue is None or (now - self.last_catalogue).total_seconds() >= self.catalogue_seconds:
            new = self.refresh(now)
        rows = self.poll(now)
        return {"markets": len(self.markets), "new": new, "rows": len(rows), "reads": self.reads}

    def save(self) -> None:
        if self.paper is not None and self.writer is not None:
            for day, name, data in self.paper.files():
                self.writer.put(day, name, data)
        if self.writer is not None:
            self.writer.upload()


# --------------------------------------------------------------------------------------------------------------------
# Summary (S3 or the server's copy)
# --------------------------------------------------------------------------------------------------------------------

def summary(store, days: int, today=None, root: Path | None = None) -> dict:
    """What the record holds a match day: markets, matches, countries, reads, the spread and the money at the best
    price a minute before the off, and the paper book's settled lines."""
    import numpy as np
    from datetime import date
    today = today or date.today()
    out = {}
    for k in range(days):
        day = (today - timedelta(days=k)).isoformat()
        mk = read_day(store, day, "markets", root)
        bk = read_day(store, day, "books", root)
        if mk is None and bk is None:
            out[day] = "no data found"
            continue
        rec = {}
        if mk is not None:
            mk1 = mk.drop_duplicates("market_id")
            rec.update({"markets": int(len(mk1)), "matches": int(mk1["event_id"].nunique()),
                        "countries": int(mk1["country"].nunique()), "competitions": int(mk1["competition"].nunique()),
                        "by_type": mk1["market_type"].value_counts().to_dict()})
        if bk is not None:
            rec["rows"] = int(len(bk))
            rec["reads_per_market_median"] = float(bk.groupby("market_id")["polled_utc"].nunique().median())
            pre = bk[(bk["status"] == "OPEN") & (bk["inplay"] == 0) & (bk["back1"] > 1) & (bk["lay1"] > 1)]
            if len(pre):
                last = pre.sort_values("polled_utc").groupby(["market_id", "selection_id"]).tail(1)
                from trading.exchange import tick_up
                ticks = []
                for b, l in zip(last["back1"], last["lay1"]):
                    n, p = 0, float(b)
                    while p < float(l) - 1e-9 and n < 50:
                        p, n = tick_up(p + 1e-6), n + 1
                    ticks.append(n)
                rec["last_book_spread_ticks_median"] = float(np.median(ticks))
                rec["last_book_best_back_gbp_median"] = float(last["back1_size"].median())
                rec["last_book_depth_back_gbp_median"] = float(
                    last[[c for c in last.columns if c.startswith("back") and c.endswith("_size")]].sum(axis=1).median())
                if "traded_total" in last and last["traded_total"].notna().any():
                    rec["traded_gbp_median_per_selection"] = float(last["traded_total"].median())
        lines = read_day(store, day, "paper_lines", root)
        if lines is not None and "pnl" in lines:
            s = lines[lines["pnl"].notna()]
            rec["paper_lines_settled"] = int(len(s))
            if len(s):
                rec["paper_by_side_checkpoint"] = (s.groupby(["side", "checkpoint_min"])
                                                   .apply(lambda g: round(float(g["pnl"].sum() / g["stake"].sum()), 4),
                                                          include_groups=False)
                                                   .rename("roi").reset_index().to_dict("records"))
        out[day] = rec
    return out


def main(argv=None) -> int:
    import argparse
    import json
    from sources.common import Store
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--root", default=None, help="a local sources folder in place of S3")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if a.summary:
        print(json.dumps(summary(Store(root=a.root), a.days), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
