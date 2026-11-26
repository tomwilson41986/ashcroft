"""Every Betfair market starting in the next hours, one row each, with its start time: the sizing for the all-sports
stream recorder (the owner, 26 Nov 2026: "size it", "stay under betfair limits").

The sports census (betfair_recorder.py --census-sports) counts each sport's markets but keeps no start times, and the
stream's limit bites on the markets open at the same moment, not on a day's total. This reads every market (a call
answers at most 200, so a window that comes back full is split: by time down to ``MIN_SPAN``, then by market type,
then by country, then by competition), and reports, for each tier of markets, the most open at once if each is
subscribed from ``LEAD_MINUTES`` before its start until it is expected to close.

How long a market stays open after its start is not in the catalogue. ``LIVE_MINUTES`` holds assumed figures per sport
(a market Betfair does not turn in play is taken to close at its start); they are assumptions, written into the
output as such, to be replaced by the closing times the recorder itself records.

Read-only, on the delayed key: listing calls only (allsports/reader.py). Output beside the day's record:
s3://$CAPTURE_BUCKET/betfair_live/<day>/census_markets.csv.gz (every market) and census_streams.json (the sizing).

    python -m allsports.census --hours 24
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import logging
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from allsports.reader import MAX_RESULTS, PROJECTION

log = logging.getLogger(__name__)

MIN_SPAN = timedelta(minutes=1)
MAX_CALLS = 4000
LEAD_MINUTES = 120.0
SHARD_CAP = 190                        # markets a connection subscribes to, under Betfair's default of 200
# The main market of each event: the first tier the recorder covers.
CORE_TYPES = ("MATCH_ODDS", "WIN", "MONEY_LINE")
# ASSUMED minutes a market stays open after its start, by Betfair event type id (not measured: see the docstring).
#   1 football, 2 tennis, 4 cricket, 7 horse racing, 4339 greyhounds, 7522 basketball, 6423 American football,
#   7524 ice hockey; every other sport LIVE_DEFAULT
LIVE_MINUTES = {"1": 130, "2": 210, "4": 480, "7": 20, "4339": 10, "7522": 150, "6423": 210, "7524": 180}
LIVE_DEFAULT = 180
NOT_IN_PLAY_MINUTES = 10
FIELDS = ["market_id", "event_type_id", "sport", "market_type", "market_name", "start_utc", "event_id", "event_name",
          "country", "competition", "in_play_enabled", "bsp_market", "total_matched"]


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc(s) -> datetime | None:
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def row(m: dict) -> dict:
    ev, et = m.get("event") or {}, m.get("eventType") or {}
    d, comp = m.get("description") or {}, m.get("competition") or {}
    return {"market_id": m.get("marketId"), "event_type_id": str(et.get("id") or ""), "sport": et.get("name"),
            "market_type": d.get("marketType"), "market_name": m.get("marketName"),
            "start_utc": m.get("marketStartTime"), "event_id": ev.get("id"), "event_name": ev.get("name"),
            "country": ev.get("countryCode"), "competition": comp.get("name"),
            "in_play_enabled": d.get("turnInPlayEnabled"), "bsp_market": d.get("bspMarket"),
            "total_matched": m.get("totalMatched")}


class Collector:
    """Every market matching a filter in a window, split until no call comes back full."""

    def __init__(self, reader, min_span: timedelta = MIN_SPAN, max_calls: int = MAX_CALLS):
        self.reader, self.min_span, self.max_calls = reader, min_span, max_calls
        self.seen: dict[str, dict] = {}
        self.calls = 0
        self.splits = Counter()
        self.truncated = 0                 # windows still full after every split: some markets may be missing
        self.stopped = False               # the call budget ran out

    def _call(self, method: str, params: dict):
        if self.calls >= self.max_calls:
            self.stopped = True
            return None
        self.calls += 1
        return self.reader.read(method, params) or []

    def _catalogue(self, flt: dict):
        return self._call("listMarketCatalogue", {"filter": flt, "marketProjection": PROJECTION,
                                                  "maxResults": MAX_RESULTS, "sort": "FIRST_TO_START"})

    def _keep(self, got) -> None:
        for m in got:
            if m.get("marketId"):
                self.seen.setdefault(str(m["marketId"]), m)

    def collect(self, base: dict, start: datetime, end: datetime) -> None:
        self._window(base, start, end, level=0)

    def _window(self, base: dict, t0: datetime, t1: datetime, level: int) -> None:
        flt = {**base, "marketStartTime": {"from": _iso(t0), "to": _iso(t1)}}
        got = self._catalogue(flt)
        if got is None:
            return
        self._keep(got)
        if len(got) < MAX_RESULTS:
            return
        if level == 0 and t1 - t0 > self.min_span:          # by time first
            self.splits["time"] += 1
            mid = (t0 + (t1 - t0) / 2).replace(microsecond=0)
            self._window(base, t0, mid - timedelta(seconds=1), 0)   # start times are whole seconds: no overlap
            self._window(base, mid, t1, 0)
            return
        # one instant (a Saturday 15:00 kick-off) holds more than a call answers: by type, then country, competition
        for lvl, (method, key, field) in enumerate(SPLITS, start=1):
            if lvl <= level or key in base:
                continue
            parts = self._call(method, {"filter": flt}) or []
            values = [((p.get(field) or {}).get("id") if isinstance(p.get(field), dict) else p.get(field))
                      for p in parts]
            values = [str(v) for v in values if v]
            if len(values) <= 1:
                continue                                   # this split cannot divide the window
            self.splits[key] += 1
            for v in values:
                self._window({**base, key: [v]}, t0, t1, lvl)
            return
        self.truncated += 1


# (the listing call, the filter key it splits on, the field of each answer that holds the value)
SPLITS = (("listMarketTypes", "marketTypeCodes", "marketType"),
          ("listCountries", "marketCountries", "countryCode"),
          ("listCompetitions", "competitionIds", "competition"))


def read_markets(reader, start: datetime, end: datetime, event_type_ids=None, market_types=None,
                 min_span: timedelta = MIN_SPAN, max_calls: int = MAX_CALLS) -> tuple[list[dict], dict]:
    """Every market starting in [start, end] (optionally of these sports and market types), one ``row`` each, and the
    reading's own counts (calls, splits, windows left full)."""
    c = Collector(reader, min_span, max_calls)
    base: dict = {}
    if event_type_ids:
        base["eventTypeIds"] = [str(x) for x in event_type_ids]
    if market_types:
        base["marketTypeCodes"] = list(market_types)
    c.collect(base, start, end)
    rows = sorted((row(m) for m in c.seen.values()), key=lambda r: (r["start_utc"] or "", r["market_id"]))
    return rows, {"calls": c.calls, "splits": dict(c.splits), "truncated_windows": c.truncated,
                  "call_budget_ran_out": c.stopped}


def live_minutes(r: dict) -> float:
    if r.get("in_play_enabled") is False:
        return NOT_IN_PLAY_MINUTES
    return LIVE_MINUTES.get(str(r.get("event_type_id")), LIVE_DEFAULT)


def open_interval(r: dict, lead_minutes: float = LEAD_MINUTES) -> tuple[datetime, datetime] | None:
    t = _utc(r.get("start_utc"))
    if t is None:
        return None
    return t - timedelta(minutes=lead_minutes), t + timedelta(minutes=live_minutes(r))


def peak_open(rows: list[dict], lead_minutes: float = LEAD_MINUTES) -> dict:
    """The most markets open at once (each from ``lead_minutes`` before its start to its expected close) and when."""
    edges = []
    for r in rows:
        iv = open_interval(r, lead_minutes)
        if iv:
            edges += [(iv[0], 1), (iv[1], -1)]
    edges.sort(key=lambda e: (e[0], e[1]))                 # a close before an open at the same instant
    now = peak = 0
    at = None
    for t, d in edges:
        now += d
        if now > peak:
            peak, at = now, t
    return {"markets": len(rows), "peak_open": peak, "peak_at_utc": _iso(at) if at else None,
            "connections_at_190": math.ceil(peak / SHARD_CAP) if peak else 0}


def sizing(rows: list[dict], lead_minutes: float = LEAD_MINUTES, core_types=CORE_TYPES) -> dict:
    core = [r for r in rows if r.get("market_type") in core_types]
    by_sport = defaultdict(list)
    for r in rows:
        by_sport[(r.get("event_type_id"), r.get("sport"))].append(r)
    sports = []
    for (sid, name), rs in sorted(by_sport.items(), key=lambda kv: -len(kv[1])):
        cs = [r for r in rs if r.get("market_type") in core_types]
        sports.append({"event_type_id": sid, "sport": name, "all": peak_open(rs, lead_minutes),
                       "core": peak_open(cs, lead_minutes),
                       "live_minutes_assumed": LIVE_MINUTES.get(str(sid), LIVE_DEFAULT),
                       "not_in_play_markets": sum(r.get("in_play_enabled") is False for r in rs)})
    starts = Counter((r.get("start_utc") or "")[:13] for r in rows)
    return {"lead_minutes": lead_minutes, "core_types": list(core_types), "shard_cap": SHARD_CAP,
            "all": peak_open(rows, lead_minutes), "core": peak_open(core, lead_minutes), "sports": sports,
            "starts_by_hour_utc": dict(sorted(starts.items())),
            "assumption": "a market's close is assumed (LIVE_MINUTES), not measured; markets not turned in play are "
                          f"taken to close {NOT_IN_PLAY_MINUTES} minutes after their start"}


def to_csv_gz(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)
    return gzip.compress(buf.getvalue().encode())


def main(argv=None) -> int:
    from betfair_recorder import PREFIX as BL, bucket, s3_client, uk_today
    from allsports.reader import Reader
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--lead-minutes", type=float, default=LEAD_MINUTES)
    ap.add_argument("--max-calls", type=int, default=MAX_CALLS)
    ap.add_argument("--no-s3", action="store_true", help="print only")
    a = ap.parse_args(argv)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    reader = Reader()
    reader.login()
    rows, reading = read_markets(reader, now, now + timedelta(hours=a.hours), max_calls=a.max_calls)
    out = {"read_utc": _iso(now), "hours": a.hours, "reading": reading, **sizing(rows, a.lead_minutes)}
    print(json.dumps({k: out[k] for k in ("read_utc", "reading", "all", "core")}, indent=1))
    for s in out["sports"]:
        print(f"{s['sport']}: {s['all']['markets']} markets, peak open {s['all']['peak_open']} "
              f"(core {s['core']['markets']}, peak {s['core']['peak_open']})")
    if not a.no_s3:
        s3, day = s3_client(), uk_today(now)
        s3.put_object(Bucket=bucket(), Key=f"{BL}/{day:%Y-%m-%d}/census_markets.csv.gz", Body=to_csv_gz(rows))
        s3.put_object(Bucket=bucket(), Key=f"{BL}/{day:%Y-%m-%d}/census_streams.json",
                      Body=json.dumps(out, indent=1).encode())
        print(f"saved: s3://{bucket()}/{BL}/{day:%Y-%m-%d}/census_markets.csv.gz, census_streams.json")
    return 0 if not reading["call_budget_ran_out"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
