"""GreyhoundAPI (api.greyhoundapi.com), the owner's Live Plus subscription (the owner, 16 Nov 2026: "We have live
plus, get all history. Get all data.").

What it holds (``/v1/archive``, 16 Nov 2026): 2,037,815 races and 14,195,219 runners, GB from 16 Nov 2013 (667,458
races), Ireland from 2 May 1999 (190,238), Australia from 1 Jan 1990 (961,311). A result carries, a runner: trap,
position, SP, first sectional, run and adjusted time, the weight after the race with its change since the last run,
trainer, and in GB the last Betfair back and lay price before the off (``/races/<id>/market`` holds no more than
that: no price history); a race: going, prize money, forecast and tricast
dividends, the weather it ran in. No comment and no beaten distance (GBGB's results have both: sources/gbgb.py). Its
racecards carry no weight before the result (16 Nov 2026: every runner of the day's cards "awaiting_result", minutes
before the off), so this is not a source of the weigh-in.

**Fetch** (read-only to the API): ``/v1/results`` a region and a calendar month at a time, 50 races a page through
``next_cursor``, at most ``RATE`` calls a second (the plan allows 180 a minute), each month written raw to
``sources/greyhoundapi/raw/results/<region>/<YYYY-MM>.json.gz`` with a manifest (races, pages, the API's total, when).
A month held is not fetched again unless it is one of the last ``REFRESH_MONTHS`` (results are corrected for a while).
``--max-minutes`` stops cleanly between months; a rerun goes on where it stopped. The reference lists (tracks, sires,
dams, and each track's statistics) are paged the same way to ``raw/lists/<name>.json.gz``.

**Build**: one row a runner, ``sources/greyhoundapi/runs_<region>_<year>.parquet``: the race (id, meeting, track,
date, off, grade, distance, going, handicap, field size, prize money, dividends, weather) and the runner (dog id and
name, trap, position, SP, sectional, run time, adjusted time, weight, the weight change and days since its last run,
trainer, Betfair selection, back and lay before the off).

The API key is read from ``GREYHOUNDAPI_KEY`` and never written or logged.

    GREYHOUNDAPI_KEY=... python -m sources.greyhoundapi --fetch results --regions GB,IE,AU --max-minutes 300
    GREYHOUNDAPI_KEY=... python -m sources.greyhoundapi --fetch lists
    GREYHOUNDAPI_KEY=... python -m sources.greyhoundapi --fetch pedigree --max-minutes 300   # every sire's offspring
    GREYHOUNDAPI_KEY=... python -m sources.greyhoundapi --fetch runners --regions GB,IE,AU --max-minutes 320  # each race's runners
    GREYHOUNDAPI_KEY=... python -m sources.greyhoundapi --fetch cards                        # today's and upcoming cards
    python -m sources.greyhoundapi --build --regions GB,IE,AU
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import os
import time
from datetime import date, datetime, timezone

import pandas as pd
import requests

log = logging.getLogger(__name__)

API = "https://api.greyhoundapi.com/v1"
PREFIX = "greyhoundapi"
RATE = 2.8                       # calls a second: under the plan's 180 a minute
REFRESH_MONTHS = 2
FIRST = {"GB": date(2013, 11, 1), "IE": date(1999, 5, 1), "AU": date(1990, 1, 1)}
LISTS = {"tracks": "/tracks", "sires": "/sires", "dams": "/dams"}
#: per track (``/tracks/<id>/...``): its statistics (trap wins by distance, records)
TRACK_LISTS = ("stats",)


class Client:
    """The API, at most ``rate`` calls a second, retrying a 429 or a 5xx with backoff. The key never leaves the header."""

    def __init__(self, key: str | None = None, rate: float = RATE, session=None, sleep=time.sleep, clock=time.monotonic):
        self.key = key or os.environ.get("GREYHOUNDAPI_KEY", "")
        if not self.key:
            raise SystemExit("GREYHOUNDAPI_KEY is not set")
        self.s = session or requests.Session()
        self.interval, self.sleep, self.clock, self.next = 1.0 / rate, sleep, clock, None
        self.calls = 0

    def get(self, path: str, params: dict | None = None, tries: int = 6) -> dict:
        for attempt in range(tries):
            now = self.clock()
            if self.next is not None and now < self.next:
                self.sleep(self.next - now)
            self.next = max(self.clock(), self.next or 0.0) + self.interval
            try:
                r = self.s.get(f"{API}{path}", params=params or {}, headers={"X-API-Key": self.key}, timeout=90)
            except requests.RequestException as exc:
                log.warning("%s: %s (try %d)", path, type(exc).__name__, attempt + 1)
                self.sleep(2.0 * (attempt + 1))
                continue
            self.calls += 1
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                wait = float(r.headers.get("Retry-After") or 0) or 5.0 * (attempt + 1)
                log.warning("%s: HTTP %d, waiting %.0f s", path, r.status_code, wait)
                self.sleep(wait)
                continue
            raise RuntimeError(f"{path}: HTTP {r.status_code} {r.text[:200]}")
        raise RuntimeError(f"{path}: gave up after {tries} tries")

    def pages(self, path: str, params: dict | None = None):
        """Every page's ``data`` through ``next_cursor``; yields (data, meta)."""
        p = dict(params or {})
        while True:
            j = self.get(path, p)
            meta = j.get("meta", {})
            yield j.get("data") or [], meta
            cur = meta.get("next_cursor")
            if not cur:
                return
            p = {**(params or {}), "cursor": cur}


def gz(b: bytes) -> bytes:
    return gzip.compress(b, mtime=0)


def months(first: date, last: date) -> list[tuple[int, int]]:
    out, y, m = [], first.year, first.month
    while (y, m) <= (last.year, last.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def month_bounds(y: int, m: int) -> tuple[str, str]:
    nxt = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    return f"{y:04d}-{m:02d}-01", (nxt - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def raw_key(region: str, y: int, m: int) -> str:
    return f"{PREFIX}/raw/results/{region}/{y:04d}-{m:02d}.json.gz"


def fetch_month(c: Client, region: str, y: int, m: int) -> tuple[list, dict]:
    lo, hi = month_bounds(y, m)
    races, pages, total = [], 0, None
    for data, meta in c.pages("/results", {"date_from": lo, "date_to": hi, "region": region}):
        races += data
        pages += 1
        if total is None:
            total = meta.get("total")
    return races, {"region": region, "month": f"{y:04d}-{m:02d}", "races": len(races), "pages": pages,
                   "api_total": total, "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def fetch_results(store, c: Client, regions: list[str], today: date, max_minutes: float | None = None,
                  newest_first: bool = True) -> dict:
    t0 = time.monotonic()
    held = set(store.listing(f"{PREFIX}/raw/results/"))
    recent = set(months(date(today.year, today.month, 1) - pd.DateOffset(months=REFRESH_MONTHS - 1), today)) \
        if REFRESH_MONTHS else set()
    todo = []
    for reg in regions:
        for y, m in months(FIRST[reg], today):
            k = raw_key(reg, y, m)
            if k in held and (y, m) not in recent:
                continue
            todo.append((reg, y, m))
    todo.sort(key=lambda t: (t[1], t[2]), reverse=newest_first)
    done, races, stopped = 0, 0, False
    for reg, y, m in todo:
        if max_minutes and (time.monotonic() - t0) / 60.0 > max_minutes:
            stopped = True
            break
        rows, man = fetch_month(c, reg, y, m)
        store.put(raw_key(reg, y, m), gz(json.dumps(rows, separators=(",", ":")).encode()))
        store.put(raw_key(reg, y, m).replace(".json.gz", ".manifest.json"), json.dumps(man).encode())
        done += 1
        races += len(rows)
        log.info("%s %04d-%02d: %d races (API total %s), %d calls so far", reg, y, m, len(rows), man["api_total"], c.calls)
    return {"months_to_fetch": len(todo), "months_fetched": done, "races": races, "calls": c.calls,
            "stopped_for_time": stopped, "left": len(todo) - done}


def fetch_lists(store, c: Client, names: list[str] | None = None, max_minutes: float | None = None) -> dict:
    t0 = time.monotonic()
    out = {}
    for name in names or list(LISTS):
        if max_minutes and (time.monotonic() - t0) / 60.0 > max_minutes:
            break
        rows = []
        try:
            for data, _ in c.pages(LISTS[name]):
                rows += data
        except RuntimeError as exc:
            out[name] = f"not fetched ({str(exc)[:120]})"
            log.warning("list %s: %s", name, exc)
            continue
        store.put(f"{PREFIX}/raw/lists/{name}.json.gz", gz(json.dumps(rows, separators=(",", ":")).encode()))
        out[name] = len(rows)
        log.info("list %s: %d rows", name, len(rows))
        if name == "tracks":
            for sub in TRACK_LISTS:
                per = {}
                for t in rows:
                    tid = t.get("track_id")
                    if tid is None:
                        continue
                    try:
                        per[str(tid)] = c.get(f"/tracks/{int(tid)}/{sub}").get("data")
                    except RuntimeError as exc:
                        per[str(tid)] = {"error": str(exc)[:120]}
                store.put(f"{PREFIX}/raw/lists/track_{sub.replace('-', '_')}.json.gz",
                          gz(json.dumps(per, separators=(",", ":")).encode()))
                out[f"track_{sub}"] = sum(1 for v in per.values() if not (isinstance(v, dict) and "error" in v))
    return out


PROGENY_CHUNK = 200


def fetch_pedigree(store, c: Client, max_minutes: float | None = None) -> dict:
    """Every sire (``/sires?min_runs=0``) and each sire's offspring (``/sires/<id>/progeny``: dog id, name, dam), in
    chunks of ``PROGENY_CHUNK`` sires to ``raw/progeny/<n>.json.gz``; a chunk held is not fetched again."""
    t0 = time.monotonic()
    sires = []
    for data, _ in c.pages("/sires", {"min_runs": 0}):
        sires += data
    store.put(f"{PREFIX}/raw/lists/sires_all.json.gz", gz(json.dumps(sires, separators=(",", ":")).encode()))
    held = set(store.listing(f"{PREFIX}/raw/progeny/"))
    done, dogs, stopped = 0, 0, False
    for i in range(0, len(sires), PROGENY_CHUNK):
        key = f"{PREFIX}/raw/progeny/{i // PROGENY_CHUNK:04d}.json.gz"
        if key in held:
            continue
        if max_minutes and (time.monotonic() - t0) / 60.0 > max_minutes:
            stopped = True
            break
        out = []
        for sr in sires[i:i + PROGENY_CHUNK]:
            sid = sr.get("sire_id")
            if sid is None:
                continue
            for data, _ in c.pages(f"/sires/{int(sid)}/progeny"):
                for d in data:
                    out.append({"dog_id": d.get("dog_id"), "dog_name": d.get("name"), "dam": d.get("dam"),
                                "sire_id": sid, "sire": sr.get("name")})
        store.put(key, gz(json.dumps(out, separators=(",", ":")).encode()))
        done += 1
        dogs += len(out)
        log.info("progeny chunk %s: %d dogs, %d calls so far", key.rsplit("/", 1)[1], len(out), c.calls)
    return {"sires": len(sires), "chunks_fetched": done, "dogs": dogs, "stopped_for_time": stopped}


QUOTA_STOP = 450_000           # a month's calls at which the backfills stop, leaving room for the daily jobs


def month_used(c: Client) -> int | None:
    """Calls used this month (``/usage``), or None when it cannot be read."""
    try:
        return int(c.get("/usage")["data"]["month"]["used"])
    except Exception:                                    # noqa: BLE001  (a missing reading only skips the guard)
        return None


def runners_key(region: str, month: str) -> str:
    return f"{PREFIX}/raw/runners/{region}/{month}.json.gz"


def fetch_runners(store, c: Client, regions: list[str], max_minutes: float | None = None,
                  quota_stop: int = QUOTA_STOP) -> dict:
    """Each held results month's races, their full runner lists (``/races/<id>/runners``: every runner with its status,
    scratched and reserve dogs included, owner, sire, dam), a region and month a file, newest months first, the regions
    in the order given. A month held is skipped; stops at ``max_minutes`` or when the month's calls reach
    ``quota_stop``."""
    t0 = time.monotonic()
    held = set(store.listing(f"{PREFIX}/raw/runners/"))
    todo = []
    for reg in regions:
        for k in sorted(store.listing(f"{PREFIX}/raw/results/{reg}/"), reverse=True):
            if k.endswith(".json.gz") and not k.endswith(".manifest.json"):
                month = k.rsplit("/", 1)[1][:7]
                if runners_key(reg, month) not in held:
                    todo.append((reg, month, k))
    done, races, stop = 0, 0, None
    for reg, month, k in todo:
        if max_minutes and (time.monotonic() - t0) / 60.0 > max_minutes:
            stop = "time"
            break
        used = month_used(c)
        if used is not None and used >= quota_stop:
            stop = f"quota ({used:,} calls used this month)"
            break
        out = {}
        for rc in json.loads(gzip.decompress(store.get(k))):
            rid = rc.get("race_id")
            if rid is None:
                continue
            try:
                out[str(rid)] = c.get(f"/races/{int(rid)}/runners").get("data")
            except RuntimeError as exc:
                out[str(rid)] = {"error": str(exc)[:120]}
        store.put(runners_key(reg, month), gz(json.dumps(out, separators=(",", ":")).encode()))
        done += 1
        races += len(out)
        log.info("runners %s %s: %d races, %d calls so far", reg, month, len(out), c.calls)
    return {"months_to_fetch": len(todo), "months_fetched": done, "races": races, "stopped": stop,
            "left": len(todo) - done}


def comments_key(region: str, month: str) -> str:
    return f"{PREFIX}/raw/comments/{region}/{month}.json.gz"


def fetch_comments(store, c: Client, regions: list[str], first_month: str = "2011-01", max_minutes: float | None = None,
                   quota_stop: int = QUOTA_STOP) -> dict:
    """Each held results month's races (from ``first_month``), their in-running comments (``/races/<id>/comments``:
    each runner's events, e.g. 'fast_start', 'led', with the place in the race where given, and a summary), a region
    and month a file, newest months first (the owner, 17 Nov 2026: "fetch the in-running comments for Irish runs").
    A month held is skipped (the two newest are fetched again: results are still arriving); the vocabulary is saved
    once. Stops at ``max_minutes`` or when the month's calls reach ``quota_stop``."""
    t0 = time.monotonic()
    if store.get(f"{PREFIX}/raw/comments/vocabulary.json.gz") is None:
        try:
            store.put(f"{PREFIX}/raw/comments/vocabulary.json.gz",
                      gz(json.dumps(c.get("/comments/vocabulary")).encode()))
        except RuntimeError as exc:
            log.warning("comment vocabulary not read (%s)", exc)
    held = set(store.listing(f"{PREFIX}/raw/comments/"))
    todo = []
    for reg in regions:
        months = sorted((k for k in store.listing(f"{PREFIX}/raw/results/{reg}/")
                         if k.endswith(".json.gz") and k.rsplit("/", 1)[1][:7] >= first_month), reverse=True)
        for i, k in enumerate(months):
            month = k.rsplit("/", 1)[1][:7]
            if comments_key(reg, month) not in held or i < REFRESH_MONTHS:
                todo.append((reg, month, k))
    done, races, empty, stop = 0, 0, 0, None
    for reg, month, k in todo:
        if max_minutes and (time.monotonic() - t0) / 60.0 > max_minutes:
            stop = "time"
            break
        used = month_used(c)
        if used is not None and used >= quota_stop:
            stop = f"quota ({used:,} calls used this month)"
            break
        out = {}
        for rc in json.loads(gzip.decompress(store.get(k))):
            rid = rc.get("race_id")
            if rid is None:
                continue
            try:
                out[str(rid)] = c.get(f"/races/{int(rid)}/comments").get("data")
            except RuntimeError as exc:
                out[str(rid)] = {"error": str(exc)[:120]}
            if not (isinstance(out[str(rid)], dict) and out[str(rid)].get("runners")):
                empty += 1
        store.put(comments_key(reg, month), gz(json.dumps(out, separators=(",", ":")).encode()))
        done += 1
        races += len(out)
        log.info("comments %s %s: %d races, %d calls so far", reg, month, len(out), c.calls)
    return {"months_to_fetch": len(todo), "months_fetched": done, "races": races, "races_without_comments": empty,
            "stopped": stop, "left": len(todo) - done}


def comment_rows(store, region: str) -> pd.DataFrame:
    """One row a runner of each race with comments: race, dog, trap, position, the summary, the events in order
    (event codes joined by '|', their places by '|'), and the count."""
    rows = []
    for k in sorted(store.listing(f"{PREFIX}/raw/comments/{region}/")):
        if not k.endswith(".json.gz"):
            continue
        for rid, d in json.loads(gzip.decompress(store.get(k))).items():
            if not isinstance(d, dict):
                continue
            for p in d.get("runners") or []:
                ir = _g(p, "comments", "in_running") or {}
                ev = ir.get("events") or []
                rows.append({"race_id": int(rid), "dog_id": p.get("dog_id"), "dog_name": p.get("dog_name"),
                             "trap": p.get("trap"), "position": p.get("position"),
                             "start_utc": d.get("scheduled_start"), "track": _g(d, "track", "name"),
                             "summary": ir.get("summary"), "n_events": len(ev),
                             "events": "|".join(str(e.get("event") or "") for e in ev),
                             "places": "|".join(str(e.get("place") or "") for e in ev)})
    return pd.DataFrame(rows)


def build_comments(store, regions: list[str]) -> dict:
    out = {}
    for reg in regions:
        df = comment_rows(store, reg)
        if len(df):
            store.put_parquet(f"{PREFIX}/comments_{reg}.parquet", df)
        out[reg] = int(len(df))
    return out


def snapshot_cards(store, c: Client) -> dict:
    """Today's and the upcoming racecards for every region, raw, to ``raw/cards/<UTC day>/<HHMM>.json.gz`` (the entries
    as they stand: runners, reserves, scratchings, weights once weighed)."""
    now = datetime.now(timezone.utc)
    out = {}
    for path in ("/racecards/today", "/racecards/upcoming"):
        rows = []
        for data, _ in c.pages(path):
            rows += data
        out[path.rsplit("/", 1)[1]] = rows
    store.put(f"{PREFIX}/raw/cards/{now:%Y-%m-%d}/{now:%H%M}.json.gz", gz(json.dumps(out, separators=(",", ":")).encode()))
    return {k: len(v) for k, v in out.items()}


def build_pedigree(store) -> int:
    """``greyhoundapi/pedigree.parquet``: one row an API dog with its sire and dam names."""
    rows = []
    for k in sorted(k for k in store.listing(f"{PREFIX}/raw/progeny/") if k.endswith(".json.gz")):
        rows += json.loads(gzip.decompress(store.get(k)))
    df = pd.DataFrame(rows)
    if len(df):
        df = df.drop_duplicates("dog_id", keep="last")
        store.put_parquet(f"{PREFIX}/pedigree.parquet", df)
    return int(len(df))


def _g(d, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def rows(races: list) -> list[dict]:
    """One row a runner of each resulted race."""
    out = []
    for r in races:
        res = r.get("result") or {}
        wx = res.get("weather") or r.get("weather") or {}
        base = {"race_id": r.get("race_id"), "meeting_id": r.get("meeting_id"), "region": r.get("region"),
                "track_id": _g(r, "track", "track_id"), "track": _g(r, "track", "name"),
                "start_utc": _g(r, "scheduled_start", "utc"), "start_local": _g(r, "scheduled_start", "local"),
                "off_utc": _g(r, "off", "utc"), "race_number": r.get("race_number"), "grade": r.get("grade"),
                "class_letter": r.get("class_letter"), "class_number": r.get("class_number"),
                "distance_m": r.get("distance_m"), "race_type": r.get("race_type"), "is_handicap": r.get("is_handicap"),
                "going": r.get("going"), "field_size": r.get("field_size"), "status": r.get("status"),
                "result_status": res.get("result_status"), "prize_money": json.dumps(res.get("prize_money"))
                if res.get("prize_money") is not None else None,
                "forecast_dividend": json.dumps(res.get("forecast_dividend")) if res.get("forecast_dividend") is not None else None,
                "tricast_dividend": json.dumps(res.get("tricast_dividend")) if res.get("tricast_dividend") is not None else None,
                "weather": json.dumps(wx) if wx else None}
        for p in res.get("positions") or []:
            sp = p.get("sp") or {}
            w = p.get("weight") or {}
            bf = p.get("betfair") or {}
            out.append({**base, "trap": p.get("trap"), "rug": p.get("rug"), "position": p.get("position"),
                        "dog_id": p.get("dog_id"), "dog_name": p.get("dog_name"),
                        "trainer_id": _g(p, "trainer", "trainer_id"), "trainer": _g(p, "trainer", "name"),
                        "sp_fraction": sp.get("fraction") if isinstance(sp, dict) else None,
                        "sp_decimal": sp.get("decimal") if isinstance(sp, dict) else None,
                        "sectional_s": p.get("sectional_s"), "run_time_s": p.get("run_time_s"),
                        "adjusted_time_s": p.get("adjusted_time_s"), "weight_kg": p.get("weight_kg"),
                        "weight_status": w.get("status") if isinstance(w, dict) else None,
                        "weight_change_kg": w.get("change_kg") if isinstance(w, dict) else None,
                        "last_race_kg": _g(w, "last_race", "kg"), "last_race_days": _g(w, "last_race", "days_before"),
                        "bf_selection_id": bf.get("selection_id"), "bf_back": bf.get("back_price"),
                        "bf_lay": bf.get("lay_price"), "bf_last_update": bf.get("last_update")})
    return out


def build(store, regions: list[str]) -> dict:
    out = {}
    for reg in regions:
        keys = sorted(k for k in store.listing(f"{PREFIX}/raw/results/{reg}/") if k.endswith(".json.gz"))
        by_year: dict = {}
        for k in keys:
            by_year.setdefault(k.rsplit("/", 1)[1][:4], []).append(k)
        for y, ks in sorted(by_year.items()):
            recs = []
            for k in ks:
                recs += rows(json.loads(gzip.decompress(store.get(k))))
            df = pd.DataFrame(recs)
            if len(df):
                df = df.drop_duplicates(["race_id", "trap", "dog_id"], keep="last")
                store.put_parquet(f"{PREFIX}/runs_{reg}_{y}.parquet", df)
            out[f"{reg}_{y}"] = int(len(df))
            log.info("%s %s: %d runners", reg, y, len(df))
    return out


def main(argv=None) -> int:
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fetch", choices=["results", "lists", "both", "pedigree", "runners", "cards", "comments"],
                    default=None)
    ap.add_argument("--first-month", default="2011-01", help="comments: the first results month fetched")
    ap.add_argument("--build-comments", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--regions", default="GB,IE,AU")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--store", default=None)
    ap.add_argument("--out", default="out/greyhoundapi.json")
    a = ap.parse_args(argv)
    store = Store(root=a.store)
    regions = [r.strip().upper() for r in a.regions.split(",") if r.strip()]
    rep: dict = {}
    if a.fetch:
        c = Client()
        if a.fetch in ("results", "both"):
            rep["results"] = fetch_results(store, c, regions, date.today(), a.max_minutes)
        if a.fetch in ("lists", "both"):
            rep["lists"] = fetch_lists(store, c)
        if a.fetch == "pedigree":
            rep["pedigree"] = fetch_pedigree(store, c, a.max_minutes)
        if a.fetch == "runners":
            rep["runners"] = fetch_runners(store, c, regions, a.max_minutes)
        if a.fetch == "cards":
            rep["cards"] = snapshot_cards(store, c)
        if a.fetch == "comments":
            rep["comments"] = fetch_comments(store, c, regions, a.first_month, a.max_minutes)
    if a.build_comments:
        rep["comments_built"] = build_comments(store, regions)
    if a.build:
        rep["build"] = build(store, regions)
        rep["pedigree_dogs"] = build_pedigree(store)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
