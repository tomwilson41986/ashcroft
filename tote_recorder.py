"""Record the Tote's horse-racing pools through the day, beside the Betfair record (the owner's ask of 6 Oct 2026).

Read-only: no login, nothing posted. tote.co.uk serves no one outside the UK, so this runs on the owner's UK server,
beside the trader (live-trade.yml). It reads the racing API the site's own pages read (tote_probe.py found it: the
address and the public client key every visitor is served come from /config.js, read at start and kept out of every
file and log line), at most one request a second:

    every 10 minutes          race-card/pools/today      every pool open today with its total: win, place,
                                                         exacta, trifecta, swinger, quinella, placepot, jackpot
    near each GB/IE off       race-card/pool/<id>        win and place: each runner's approximate dividend
      win, place: T-60, -30, -15, -10, -5, -3, -2, -1,   (baseWinStake, basePlaceStake: the pool's own, per unit
      +2, +5 min; exacta, trifecta, swinger, quinella:    staked) and the figure the site shows with the Tote
      T-5, +5 min                                         Guarantee (winstake: at least the bookmakers' price),
                                                          the bookmakers' show price, scratched; the pool's total
    after each GB/IE off      race-card/race-results/<ids>  the declared dividends (T+10 and T+30 min)

Day files, JSON lines gzipped, under --out/<day>/: pools.jsonl.gz, detail.jsonl.gz and results.jsonl.gz, a line an
answer: {"polled_utc", "path", "status", "race_id", "pool", "mark", "answer"} (the runners without their silks and
write-ups, the pools without their stake limits). Copied to s3://$CAPTURE_BUCKET/tote/live/<day>/ every 15 minutes
and at the end. The comparison with Betfair (tote_compare.py) reads them.

Usage:
    python tote_recorder.py --record --until 21:30            # today, to 21:30 UK (live-trade.yml, beside the trader)
    python tote_recorder.py --record --minutes 4              # a short test
    python tote_recorder.py --results                         # today's declared dividends, once (after racing)
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import time
import urllib.parse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

from tote_probe import BASE, CONFIG_PATH, UA, api_config, redact

log = logging.getLogger("tote_recorder")

ROOT = Path(os.getenv("TOTE_LIVE_DIR") or (Path.home() / ".ashcroft" / "tote_live"))
PREFIX = "tote/live"
COUNTRIES = ("GB", "IE")
POOLS_EVERY = 600                                           # seconds between polls of every pool's total
WIN_PLACE_MARKS = (-60, -30, -15, -10, -5, -3, -2, -1, 2, 5)  # minutes from the off
EXOTIC_MARKS = (-5, 5)
RESULT_MARKS = (10, 30)
LATE = 3                                                    # minutes a mark may be taken late (a mark missed by
                                                            # more, at a restart, is skipped)
PAUSE = 1.0                                                 # seconds between requests
UPLOAD_EVERY = 900
WIN_PLACE = ("WIN", "PLACE")
EXOTICS = ("EXACTA", "TRIFECTA", "SWINGER", "QUINELLA")
RUNNER_KEEP = ("id", "name", "programNumber", "position", "scratched", "status", "spPrice", "decimalShowPrice",
               "fractionalShowPrice", "morningLineOdds", "winstake", "baseWinStake", "placestake", "basePlaceStake",
               "unnamedFavourite", "coupleIndicator", "jockeyName", "trainer", "raceId")


def uk_today(now: datetime | None = None) -> date:
    from zoneinfo import ZoneInfo
    return (now or datetime.now(timezone.utc)).astimezone(ZoneInfo("Europe/London")).date()


def uk_time_on(day: date, hhmm: str) -> datetime:
    from zoneinfo import ZoneInfo
    h, m = (int(x) for x in hhmm.split(":"))
    return datetime(day.year, day.month, day.day, h, m, tzinfo=ZoneInfo("Europe/London")).astimezone(timezone.utc)


def parse_utc(s) -> datetime | None:
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def pool_name(p: dict) -> str:
    return str((p.get("poolType") or {}).get("name", "")).replace(" ", "").upper()


def slim_pool(p: dict) -> dict:
    """A pool as kept: without its stake limits (betTypes), its runners without their silks and write-ups."""
    out = {k: v for k, v in p.items() if k != "betTypes"}
    legs = []
    for leg in p.get("legs") or []:
        leg = dict(leg)
        race = leg.get("race")
        if isinstance(race, dict):
            race = dict(race)
            runners = []
            for r in race.get("runners") or []:
                kept = {k: r.get(k) for k in RUNNER_KEEP if k in r}
                draw = (r.get("extraData") or {}).get("draw")
                if draw is not None:
                    kept["draw"] = draw
                runners.append(kept)
            race["runners"] = runners
            leg["race"] = race
        legs.append(leg)
    out["legs"] = legs
    return out


def races_from_pools(pools: list[dict], countries=COUNTRIES) -> dict[int, dict]:
    """Today's single-race pools by race: {race_id: {"post": UTC time, "meeting", "country", "pools": {name: id}}},
    for the countries asked (the GB and IE meetings by default)."""
    races: dict[int, dict] = {}
    for p in pools:
        legs = p.get("legs") or []
        name = pool_name(p)
        if len(legs) != 1 or name not in WIN_PLACE + EXOTICS:
            continue
        meeting = p.get("meeting") or {}
        if countries and str(meeting.get("countryCode", "")).upper() not in countries:
            continue
        leg = legs[0]
        rid = leg.get("raceId")
        post = parse_utc(leg.get("racePostTime"))
        if rid is None or post is None:
            continue
        r = races.setdefault(int(rid), {"post": post, "meeting": meeting.get("name"),
                                        "country": meeting.get("countryCode"), "pools": {}})
        r["post"] = post                                    # the latest list's post time (a race put back moves)
        r["pools"][name] = p.get("id")
    return races


class ToteRecorder:
    def __init__(self, day: date, out: Path, stop_utc: datetime, session=None, clock=None, sleep=None,
                 base: str = BASE, countries=COUNTRIES, pools_every: int = POOLS_EVERY, pause: float = PAUSE,
                 upload_every: int = UPLOAD_EVERY, s3=None):
        self.day, self.stop_utc, self.base, self.countries = day, stop_utc, base, tuple(countries)
        self.dir = Path(out) / f"{day:%Y-%m-%d}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": UA, "Accept": "application/json,*/*;q=0.5",
                                     "Accept-Language": "en-GB,en;q=0.9"})
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.sleep = sleep or time.sleep
        self.pools_every, self.pause, self.upload_every, self.s3 = pools_every, pause, upload_every, s3
        self.api_url: str | None = None
        self.key: str | None = None
        self.races: dict[int, dict] = {}
        self.done: set[tuple] = set()
        self.last_request = 0.0
        self.last_pools: datetime | None = None
        self.last_upload: datetime | None = None
        self.counts = {"pools": 0, "detail": 0, "results": 0, "errors": 0}

    # --- requests ---------------------------------------------------------------------------------------------
    def _get(self, url: str):
        wait = self.pause - (time.monotonic() - self.last_request)
        if wait > 0:
            self.sleep(wait)
        self.last_request = time.monotonic()
        try:
            return self.session.get(url, timeout=25)
        except requests.RequestException as exc:
            log.warning("request failed: %s", redact(f"{type(exc).__name__}: {exc}", self.key)[:300])
            self.counts["errors"] += 1
            return None

    def start(self) -> bool:
        r = self._get(urllib.parse.urljoin(self.base, CONFIG_PATH))
        if r is None or not r.ok:
            log.error("config.js not read (%s)", None if r is None else r.status_code)
            return False
        self.api_url, self.key = api_config(r.text)
        if not self.api_url or not self.key:
            log.error("no racing API address and key in config.js")
            return False
        log.info("racing API %s (client key read)", self.api_url)
        return True

    def ask(self, path: str, with_key: bool = True):
        """(status, JSON answer or None) for an API path; the key goes in the query unless with_key is False."""
        url = urllib.parse.urljoin(self.api_url, path)
        if with_key:
            url += ("&" if "?" in url else "?") + "key=" + urllib.parse.quote(self.key or "")
        r = self._get(url)
        if r is None:
            return None, None
        try:
            return r.status_code, (r.json() if r.ok else None)
        except ValueError:
            return r.status_code, None

    def write(self, kind: str, rec: dict) -> None:
        line = redact(json.dumps(rec, separators=(",", ":"), default=str), self.key)
        with gzip.open(self.dir / f"{kind}.jsonl.gz", "at", encoding="utf-8") as f:
            f.write(line + "\n")

    # --- the three records --------------------------------------------------------------------------------------
    def poll_pools(self, now: datetime) -> None:
        status, ans = self.ask("race-card/pools/today")
        pools = (ans or {}).get("pools") if isinstance(ans, dict) else (ans if isinstance(ans, list) else None)
        self.last_pools = now
        if pools is None:
            self.counts["errors"] += 1
            log.warning("pools/today: status %s, no pools", status)
            return
        pools = [p for p in pools if isinstance(p, dict)]
        self.write("pools", {"polled_utc": now.isoformat(), "path": "race-card/pools/today", "status": status,
                             "answer": {"pools": [slim_pool(p) for p in pools]}})
        self.counts["pools"] += 1
        for rid, r in races_from_pools(pools, self.countries).items():
            known = self.races.setdefault(rid, r)
            known["post"] = r["post"]
            known["pools"].update(r["pools"])

    def due(self, now: datetime) -> tuple[list, list]:
        """The pool marks and the result marks due now: [(race_id, pool name, pool id, mark)], [(race_id, mark)].
        A mark missed by more than LATE minutes (a restart) is passed over."""
        pools, results = [], []
        for rid, r in sorted(self.races.items(), key=lambda kv: kv[1]["post"]):
            for name, pid in r["pools"].items():
                marks = WIN_PLACE_MARKS if name in WIN_PLACE else EXOTIC_MARKS
                for m in marks:
                    key = ("pool", rid, name, m)
                    at = r["post"] + timedelta(minutes=m)
                    if key in self.done or at > now:
                        continue
                    self.done.add(key)
                    if now - at <= timedelta(minutes=LATE):
                        pools.append((rid, name, pid, m))
            for m in RESULT_MARKS:
                key = ("result", rid, m)
                at = r["post"] + timedelta(minutes=m)
                if key in self.done or at > now:
                    continue
                self.done.add(key)
                if now - at <= timedelta(minutes=LATE):
                    results.append((rid, m))
        return pools, results

    def take_pool(self, now: datetime, rid: int, name: str, pid, mark: int) -> None:
        path = f"race-card/pool/{pid}"
        status, ans = self.ask(path)
        self.write("detail", {"polled_utc": now.isoformat(), "path": path, "status": status, "race_id": rid,
                              "pool": name, "mark": mark, "answer": slim_pool(ans) if isinstance(ans, dict) else None})
        self.counts["detail" if isinstance(ans, dict) else "errors"] += 1

    def take_results(self, now: datetime, race_ids: list[int], mark=None) -> None:
        """The declared dividends of these races, in one request (the site asks without the key; asked again with
        it if refused)."""
        if not race_ids:
            return
        path = "race-card/race-results/" + ",".join(str(r) for r in race_ids)
        status, ans = self.ask(path, with_key=False)
        if status in (401, 403):
            status, ans = self.ask(path)
        self.write("results", {"polled_utc": now.isoformat(), "path": path, "status": status,
                               "race_id": race_ids, "mark": mark, "answer": ans})
        self.counts["results" if ans is not None else "errors"] += 1

    # --- the day -------------------------------------------------------------------------------------------------
    def upload(self, force: bool = False) -> int:
        now = self.clock()
        if not force and self.last_upload and (now - self.last_upload).total_seconds() < self.upload_every:
            return 0
        self.last_upload = now
        bucket = os.getenv("CAPTURE_BUCKET")
        if not bucket and self.s3 is None:
            return 0
        n = 0
        try:
            if self.s3 is None:
                import boto3
                self.s3 = boto3.client("s3", region_name=os.getenv("CAPTURE_REGION") or "eu-west-2")
            for f in sorted(self.dir.glob("*.jsonl.gz")):
                self.s3.put_object(Bucket=bucket or "ashcroft", Key=f"{PREFIX}/{self.day:%Y-%m-%d}/{f.name}",
                                   Body=f.read_bytes())
                n += 1
        except Exception as exc:                                 # the day files stay on the server either way
            log.warning("S3 copy failed: %s", str(exc)[:200])
        return n

    def step(self) -> None:
        now = self.clock()
        if self.last_pools is None or (now - self.last_pools).total_seconds() >= self.pools_every:
            self.poll_pools(now)
        pools, results = self.due(self.clock())
        for rid, name, pid, mark in pools:
            self.take_pool(self.clock(), rid, name, pid, mark)
        by_mark: dict[int, list[int]] = {}
        for rid, mark in results:
            by_mark.setdefault(mark, []).append(rid)
        for mark, rids in sorted(by_mark.items()):
            self.take_results(self.clock(), rids, mark)
        self.upload()

    def run(self, sample: int = 0) -> dict:
        """Record until the stop time; with sample, first take the next sample races' win and place pools once
        (a short test sees the pool answers before any mark is due)."""
        if not self.start():
            return {"started": False, **self.counts}
        if sample:
            self.poll_pools(self.clock())
            upcoming = sorted((r["post"], rid) for rid, r in self.races.items() if r["post"] > self.clock())
            for _, rid in upcoming[:sample]:
                for name in WIN_PLACE:
                    if name in self.races[rid]["pools"]:
                        self.take_pool(self.clock(), rid, name, self.races[rid]["pools"][name], "sample")
        while self.clock() < self.stop_utc:
            self.step()
            self.sleep(min(20.0, max(1.0, (self.stop_utc - self.clock()).total_seconds())))
        # the day's last results, every race since its off
        now = self.clock()
        ran = [rid for rid, r in self.races.items() if r["post"] <= now]
        for i in range(0, len(ran), 20):
            self.take_results(now, ran[i:i + 20], "end")
        self.upload(force=True)
        summary = {"started": True, "races": len(self.races), **self.counts}
        log.info("Tote record %s: %s", self.day, summary)
        return summary

    def results_once(self) -> dict:
        """Today's races' declared dividends, once (the cards give the race ids)."""
        if not self.start():
            return {"started": False, **self.counts}
        status, cards = self.ask("race-card/cards/today")
        ids = []
        for m in cards or []:
            if self.countries and str(m.get("countryCode", "")).upper() not in self.countries:
                continue
            ids += [r["id"] for r in m.get("races") or [] if r.get("id") is not None]
        now = self.clock()
        for i in range(0, len(ids), 20):
            self.take_results(now, ids[i:i + 20], "once")
        self.upload(force=True)
        return {"started": True, "races": len(ids), **self.counts}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record", action="store_true", help="record today's pools until --until (UK)")
    ap.add_argument("--results", action="store_true", help="today's declared dividends, once")
    ap.add_argument("--until", default="21:30", help="stop at this UK time")
    ap.add_argument("--minutes", type=float, default=None, help="stop this many minutes from now instead")
    ap.add_argument("--out", default=str(ROOT))
    ap.add_argument("--countries", default=",".join(COUNTRIES), help="meetings' countries ('' for every one)")
    ap.add_argument("--sample", type=int, default=0, help="first take the next N races' win and place pools once")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    day = uk_today()
    stop = (datetime.now(timezone.utc) + timedelta(minutes=args.minutes) if args.minutes is not None
            else uk_time_on(day, args.until))
    countries = tuple(c.strip().upper() for c in args.countries.split(",") if c.strip())
    rec = ToteRecorder(day, Path(args.out), stop, countries=countries)
    if args.results:
        s = rec.results_once()
    elif args.record:
        s = rec.run(sample=args.sample)
    else:
        ap.error("--record or --results")
    log.info("summary: %s", s)
    return 0 if s.get("started") else 1


if __name__ == "__main__":
    raise SystemExit(main())
