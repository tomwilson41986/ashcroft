"""The cricket market recorder (build spec 5.3): every cricket market's book, in play and before, read-only.

    python -m cricket.market.recorder --hours 6 --inplay-seconds 5
    python -m cricket.market.recorder --market-types ALL --until 22:15        # cricket-record.yml

Each loop:
1. **Catalogue** (every 10 minutes): cricket (event type 4) markets of ``--market-types`` (MATCH_ODDS by default;
   ALL for every type) that started in the last 12 hours or start in the next ``--lookahead`` hours, with the
   competition, event, start and runners; a window that hits Betfair's 1,000-result cap is split in two. Once a
   run, the census: every cricket market type Betfair lists with its market count, so the innings-runs markets'
   codes are read from Betfair, not guessed.
2. **Books**, each market on its own cadence: in play every ``--inplay-seconds``; before the off every 60 s inside two
   hours of the start, else every 10 minutes; a closed market once more for its result. Each read asks for three
   levels a side (EX_BEST_OFFERS) and the traded ladder (EX_TRADED): weight 20 a market against Betfair's 200 a request,
   so ten markets a call.
3. **The record**: one row a runner each time anything about it changes (its first read, then changes only, so a
   5-second cadence does not repeat a still book): pt (our UTC poll time, when the reply came back), market and
   runner, status, in play, ``betDelay`` (Betfair's own, read from every book), total matched, last traded price,
   three back and lay levels with sizes, and the traded ladder (``price:size|...``). Written by UTC day to
   ``~/.ashcroft/cricket/books/<day>/`` and ``s3://$CAPTURE_BUCKET/sources/cricket/books/<day>/``
   (``books_<UTC hour>_<run>.csv.gz``, ``markets_<run>.csv.gz``, ``market_types.json``; ``<run>`` is the run's
   start, so a restart never overwrites an earlier run's rows).

The REST book has no ``betDelayModels`` (the Stream API's market definition carries it): recorded as no data found
until the Stream recorder exists. The delayed key's prices are themselves delayed, and it leaves out the traded
volume the live key restores (CLAUDE.md): this record measures the books, prices, delays and results, not the money
matched or the speed a live key would see.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

from cricket.market.betfair_read import CRICKET, ReadOnly

log = logging.getLogger(__name__)

PREFIX = "cricket/books"
LOCAL = Path(os.path.expanduser("~/.ashcroft/cricket/books"))
DEPTH = 3
WEIGHT = 20          # EX_BEST_OFFERS with EX_TRADED, a market (Betfair's weight table; football/exchange_feed.py)
PER_CALL = 200 // WEIGHT
FIELDS = (["pt", "market_id", "selection_id", "status", "inplay", "bet_delay", "market_matched", "runner_status",
           "ltp", "runner_matched"]
          + [f"{s}{k}_{v}" for s in ("back", "lay") for k in range(1, DEPTH + 1) for v in ("p", "s")] + ["traded"])
CATALOGUE_MAX = 1000
MARKET_FIELDS = ["recorded", "market_id", "market_type", "market_name", "event_id", "event_name", "competition",
                 "country", "start", "runners"]


def iso(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def utc(text: str) -> dt.datetime:
    return dt.datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)


def book_rows(raw: list[dict], pt: dt.datetime) -> list[dict]:
    rows = []
    for b in raw or []:
        for r in b.get("runners", []):
            ex = r.get("ex", {}) or {}
            row = {"pt": iso(pt), "market_id": b["marketId"], "selection_id": r["selectionId"],
                   "status": b.get("status"), "inplay": b.get("inplay"), "bet_delay": b.get("betDelay"),
                   "market_matched": b.get("totalMatched"), "runner_status": r.get("status"),
                   "ltp": r.get("lastPriceTraded"), "runner_matched": r.get("totalMatched")}
            for side, key in (("back", "availableToBack"), ("lay", "availableToLay")):
                lv = ex.get(key, []) or []
                for k in range(DEPTH):
                    row[f"{side}{k + 1}_p"] = lv[k]["price"] if k < len(lv) else None
                    row[f"{side}{k + 1}_s"] = lv[k]["size"] if k < len(lv) else None
            row["traded"] = "|".join(f"{x['price']}:{x['size']}" for x in ex.get("tradedVolume", []) or [])
            rows.append(row)
    return rows


def market_rows(raw: list[dict], recorded: dt.datetime) -> list[dict]:
    return [{"recorded": iso(recorded), "market_id": m["marketId"],
             "market_type": (m.get("description") or {}).get("marketType"), "market_name": m.get("marketName"),
             "event_id": (m.get("event") or {}).get("id"), "event_name": (m.get("event") or {}).get("name"),
             "competition": (m.get("competition") or {}).get("name"),
             "country": (m.get("event") or {}).get("countryCode"), "start": m.get("marketStartTime"),
             "runners": json.dumps({str(r["selectionId"]): r.get("runnerName") for r in m.get("runners", [])})}
            for m in raw or []]


class Recorder:
    def __init__(self, api: ReadOnly, store=None, root: Path = LOCAL, market_types=("MATCH_ODDS",),
                 lookahead_h: float = 24.0, inplay_s: float = 5.0, clock=None):
        self.api, self.store, self.root = api, store, root
        # None or ("ALL",): every cricket market type; else the codes given
        self.market_types = None if not market_types or list(market_types) == ["ALL"] else list(market_types)
        self.lookahead_h, self.inplay_s = lookahead_h, inplay_s
        self.clock = clock or (lambda: dt.datetime.now(dt.timezone.utc))
        self.markets: dict[str, dict] = {}
        self.next_read: dict[str, dt.datetime] = {}
        self.closed: set[str] = set()
        self.rows: dict[tuple[str, str], list[dict]] = {}     # (day, hour) -> rows
        self.market_record: dict[str, list[dict]] = {}
        self.last_catalogue: dt.datetime | None = None
        self.run_id = self.clock().strftime("%Y%m%dT%H%M%SZ")
        self.last_sig: dict[tuple, tuple] = {}
        self.polled_rows = 0

    def census(self) -> dict:
        raw = self.api.read("listMarketTypes", {"filter": {"eventTypeIds": [CRICKET]}})
        out = {r["marketType"]: r.get("marketCount") for r in raw or []}
        self._put(self.clock().strftime("%Y-%m-%d"), "market_types.json", json.dumps(out, indent=1).encode())
        return out

    def _catalogue_window(self, a: dt.datetime, b: dt.datetime, depth: int = 0) -> list[dict]:
        """One window of the catalogue; a window that hits Betfair's 1,000-result cap is split in two (to 6 levels)."""
        filt = {"eventTypeIds": [CRICKET],
                "marketStartTime": {"from": iso(a)[:19] + "Z", "to": iso(b)[:19] + "Z"}}
        if self.market_types:
            filt["marketTypeCodes"] = self.market_types
        raw = self.api.read("listMarketCatalogue", {
            "filter": filt,
            "marketProjection": ["EVENT", "COMPETITION", "MARKET_START_TIME", "RUNNER_DESCRIPTION",
                                 "MARKET_DESCRIPTION"],
            "maxResults": CATALOGUE_MAX, "sort": "FIRST_TO_START"}) or []
        if len(raw) >= CATALOGUE_MAX and depth < 6:
            mid = a + (b - a) / 2
            return self._catalogue_window(a, mid, depth + 1) + self._catalogue_window(mid, b, depth + 1)
        return raw

    def catalogue(self) -> int:
        now = self.clock()
        raw = self._catalogue_window(now - dt.timedelta(hours=12), now + dt.timedelta(hours=self.lookahead_h))
        rows = market_rows(raw, now)
        new = 0
        for r in rows:
            if r["market_id"] not in self.markets:
                new += 1
                self.next_read.setdefault(r["market_id"], now)
            self.markets[r["market_id"]] = r
        day = now.strftime("%Y-%m-%d")
        self.market_record.setdefault(day, []).extend(rows)
        self.last_catalogue = now
        return new

    def _cadence(self, mid: str, inplay: bool, now: dt.datetime) -> float:
        if inplay:
            return self.inplay_s
        start = utc(self.markets[mid]["start"])
        return 60.0 if (start - now).total_seconds() < 7200 else 600.0

    def _changed(self, rows: list[dict]) -> list[dict]:
        """Only the rows that differ from the runner's last kept row (anything but the poll time): every change is
        kept, the identical repeats of a 5-second cadence are not."""
        out = []
        for r in rows:
            key = (r["market_id"], r["selection_id"])
            sig = tuple(v for k, v in r.items() if k != "pt")
            if self.last_sig.get(key) != sig:
                self.last_sig[key] = sig
                out.append(r)
        self.polled_rows += len(rows)
        return out

    def due(self, now: dt.datetime) -> list[str]:
        return [m for m, t in self.next_read.items() if t <= now and m not in self.closed]

    def poll(self) -> int:
        now = self.clock()
        ids = self.due(now)
        n = 0
        for i in range(0, len(ids), PER_CALL):
            raw = self.api.read("listMarketBook", {
                "marketIds": ids[i:i + PER_CALL],
                "priceProjection": {"priceData": ["EX_BEST_OFFERS", "EX_TRADED"],
                                    "exBestOffersOverrides": {"bestPricesDepth": DEPTH}}})
            pt = self.clock()
            rows = self._changed(book_rows(raw, pt))
            self.rows.setdefault((pt.strftime("%Y-%m-%d"), pt.strftime("%H")), []).extend(rows)
            n += len(rows)
            for b in raw or []:
                mid = b["marketId"]
                if b.get("status") == "CLOSED":
                    self.closed.add(mid)
                    continue
                self.next_read[mid] = pt + dt.timedelta(seconds=self._cadence(mid, bool(b.get("inplay")), pt))
        return n

    # ----------------------------------------------------------------------------------------------------------
    def _put(self, day: str, name: str, data: bytes) -> None:
        p = self.root / day / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        if self.store is not None:
            try:
                self.store.put(f"{PREFIX}/{day}/{name}", data)
            except Exception as exc:          # the local copy stands; the next save tries the store again
                log.warning("store: %s/%s not kept (%s)", day, name, exc)

    @staticmethod
    def _csv(fields, rows) -> bytes:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
        return gzip.compress(buf.getvalue().encode(), mtime=0)

    def save(self) -> None:
        for (day, hour), rows in self.rows.items():
            self._put(day, f"books_{hour}_{self.run_id}.csv.gz", self._csv(FIELDS, rows))
        for day, rows in self.market_record.items():
            self._put(day, f"markets_{self.run_id}.csv.gz", self._csv(MARKET_FIELDS, rows))
        now = self.clock()
        # an hour fully written is kept on disk and in the store: drop it from memory
        self.rows = {k: v for k, v in self.rows.items() if k >= (now.strftime("%Y-%m-%d"), now.strftime("%H"))}

    def run(self, hours: float, save_every_s: float = 120.0, sleep=time.sleep, end: dt.datetime | None = None) -> dict:
        end = end or self.clock() + dt.timedelta(hours=hours)
        types = self.census()
        last_save = self.clock()
        n_rows = 0
        while self.clock() < end:
            now = self.clock()
            if self.last_catalogue is None or (now - self.last_catalogue).total_seconds() >= 600:
                self.catalogue()
            n_rows += self.poll()
            if (self.clock() - last_save).total_seconds() >= save_every_s:
                self.save()
                last_save = self.clock()
            waits = [(t - self.clock()).total_seconds() for m, t in self.next_read.items() if m not in self.closed]
            sleep(max(0.5, min(waits + [60.0])))
        self.save()
        return {"markets": len(self.markets), "rows_kept": n_rows, "rows_read": self.polled_rows,
                "calls": self.api.calls, "market_types": types}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--lookahead", type=float, default=24.0)
    ap.add_argument("--inplay-seconds", type=float, default=5.0)
    ap.add_argument("--market-types", default="MATCH_ODDS", help="comma-separated codes, or ALL for every type")
    ap.add_argument("--until", help="stop at this UTC time today (HH:MM) instead of after --hours")
    ap.add_argument("--local-only", action="store_true", help="keep the record on this host only")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    end = None
    if a.until:
        now = dt.datetime.now(dt.timezone.utc)
        hh, mm = (int(x) for x in a.until.split(":"))
        end = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if end <= now:
            print(json.dumps({"stopped": f"already past {a.until} UTC"}))
            return 0
    from sources.common import Store
    api = ReadOnly()
    api.login()
    rec = Recorder(api, None if a.local_only else Store(), market_types=a.market_types.split(","),
                   lookahead_h=a.lookahead, inplay_s=a.inplay_seconds)
    print(json.dumps(rec.run(a.hours, end=end)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
