"""Betfair's Historic Data service (historicdata.betfair.com): the exchange's own price history for every sport, the
benchmark a model outside horse racing is scored against (there is no Betfair SP there: the closing price is the last
price traded before the market turned in play).

The service sells data by sport and plan; the BASIC plan (the last traded price each minute, the market's definition
and its result) is free, but each sport has to be "bought" once on the website, at no charge, with the account's
login, before the API lists it. ``--my-data`` shows what the account holds. Then, per sport, one file a market:

    python betfair_historic.py --my-data
    python betfair_historic.py --fetch --sport Tennis --from 2019-01-01 --max-minutes 60
    python betfair_historic.py --fetch --max-minutes 60            # every sport in PLAN_SCOPE the account holds
    python betfair_historic.py --build --sport Tennis               # the closing-price table

Raw: ``sources/betfair_historic/raw/<sport>/<the service's own path>`` (bz2 stream files, as served). Table:
``sources/betfair_historic/markets_<sport>_<year>.parquet``, one row a runner a market: the market's definition, the
first price traded, the price traded at 24 h, 60, 15 and 1 minute before the scheduled off, the last before it went in
play (the close), and the result. Needs a Betfair login, and the service refuses some countries: it runs on the UK
server (betfair-prices.yml), read-only.
"""

from __future__ import annotations

import argparse
import bz2
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

from sources.common import Budget, Store

log = logging.getLogger("betfair_historic")

API = "https://historicdata.betfair.com/api"
PLAN = "Basic Plan"
#: what is fetched for each sport (the service's names), newest month first, in this order: the market types a model
#: there would price, and the countries with the liquidity (an empty list: every one). Horse racing is left out: the
#: price files (betfair_prices.py) already hold it with the BSP.
PLAN_SCOPE = {
    "Greyhound Racing": {"markets": ["WIN", "PLACE"], "countries": ["GB", "IE", "AU"]},
    "Tennis": {"markets": ["MATCH_ODDS"], "countries": []},
    "Soccer": {"markets": ["MATCH_ODDS", "OVER_UNDER_25", "ASIAN_HANDICAP"],
               "countries": ["GB", "ES", "DE", "IT", "FR", "NL", "PT", "BE", "TR", "GR"]},
    "Cricket": {"markets": ["MATCH_ODDS"], "countries": []},
    "Darts": {"markets": ["MATCH_ODDS"], "countries": []},
    "Snooker": {"markets": ["MATCH_ODDS"], "countries": []},
}
FIRST = date(2016, 1, 1)
MARKS_MIN = {"t_1440": 1440, "t_60": 60, "t_15": 15, "t_1": 1}


class Historic:
    def __init__(self, token: str, http=None):
        import requests
        self.http = http or requests.Session()
        self.http.headers.update({"ssoid": token, "Content-Type": "application/json"})

    def _post(self, method: str, body: dict | None = None):
        r = self.http.post(f"{API}/{method}", data=json.dumps(body or {}), timeout=120)
        r.raise_for_status()
        return r.json() if r.content else None

    def my_data(self) -> list[dict]:
        return self._post("GetMyData") or []

    @staticmethod
    def body(sport: str, d_from: date, d_to: date, markets=(), countries=(), plan: str = PLAN) -> dict:
        return {"sport": sport, "plan": plan, "fromDay": d_from.day, "fromMonth": d_from.month,
                "fromYear": d_from.year, "toDay": d_to.day, "toMonth": d_to.month, "toYear": d_to.year,
                "eventId": None, "eventName": None, "marketTypesCollection": list(markets),
                "countriesCollection": list(countries), "fileTypeCollection": ["M"]}

    def size(self, body: dict) -> dict:
        return self._post("GetAdvBasketDataSize", body) or {}

    def files(self, body: dict) -> list[str]:
        return self._post("DownloadListOfFiles", body) or []

    def download(self, path: str) -> bytes:
        r = self.http.get(f"{API}/DownloadFile", params={"filePath": path}, timeout=120)
        r.raise_for_status()
        return r.content


def login_token() -> str:
    from betfair_client import BetfairClient
    return BetfairClient().login()


def sport_dir(sport: str) -> str:
    return sport.lower().replace(" ", "_")


def raw_key(sport: str, path: str) -> str:
    return f"betfair_historic/raw/{sport_dir(sport)}/{path.lstrip('/')}"


def _plan_path(key: str) -> str:
    """A file's path from its plan on ('/xds_nfs/edp_processed/BASIC/2026/Sep/1/35/1.5.bz2' -> 'BASIC/2026/Sep/1/35/
    1.5.bz2'): the API's paths and a website bundle's name the same file alike from there."""
    for plan in ("BASIC/", "ADVANCED/", "PRO/"):
        i = key.find(plan)
        if i >= 0:
            return key[i:]
    return key


def path_year(key: str) -> int | None:
    """The year in the service's path ('.../BASIC/2023/Jan/1/32040445/1.207795034.bz2' -> 2023)."""
    m = re.search(r"/(\d{4})/[A-Za-z]{3}/\d{1,2}/", key)
    return int(m.group(1)) if m else None


def months_back(d_to: date, d_from: date):
    """(first, last) day of each month from d_to's back to d_from's, newest first."""
    y, m = d_to.year, d_to.month
    while (y, m) >= (d_from.year, d_from.month):
        first = date(y, m, 1)
        last = (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1))
        yield max(first, d_from), min(last, d_to)
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)


def held_sports(my_data: list[dict], plan: str = PLAN) -> list[str]:
    """The sports the account holds on the plan, in the PLAN_SCOPE order."""
    held = {str(x.get("sport")) for x in my_data if str(x.get("plan", "")).lower() == plan.lower()}
    return [s for s in PLAN_SCOPE if s in held]


def fetch(store: Store, hd: Historic, sports: list[str] | None = None, d_from: date = FIRST,
          d_to: date | None = None, max_minutes: float | None = None, workers: int = 4) -> dict:
    """Every file of the sports' scope the store lacks, month by month, newest first, sport by sport in turn (so a
    short night still reaches each sport's latest months)."""
    d_to = d_to or (date.today() - timedelta(days=1))
    if sports is None:
        sports = held_sports(hd.my_data())
        if not sports:
            log.warning("the account holds no %s sport on Betfair's Historic Data service: get each one (free) at "
                        "historicdata.betfair.com, signed in with the trading account", PLAN)
            return {"fetched": 0, "sports": []}
    have = store.listing("betfair_historic/raw/")
    held_paths = {_plan_path(k) for k in have}            # a file imported from a website bundle is held too
    budget = Budget(max_minutes)
    out = {"fetched": 0, "skipped": 0, "failed": 0, "stopped": False, "sports": {}}
    iters = {s: months_back(d_to, d_from) for s in sports}
    while iters and budget.left():
        for sport in list(iters):
            month = next(iters[sport], None)
            if month is None:
                del iters[sport]
                continue
            scope = PLAN_SCOPE.get(sport, {"markets": [], "countries": []})
            try:
                paths = hd.files(Historic.body(sport, month[0], month[1], scope["markets"], scope["countries"]))
            except Exception as exc:
                log.warning("%s %s: file list not read (%s)", sport, month[0], exc)
                out["failed"] += 1
                continue
            todo = [p for p in paths if raw_key(sport, p) not in have and _plan_path(p) not in held_paths]
            out["skipped"] += len(paths) - len(todo)

            def one(p):
                if not budget.left():
                    return 0
                try:
                    store.put(raw_key(sport, p), hd.download(p))
                    return 1
                except Exception as exc:
                    log.warning("%s not fetched (%s)", p, exc)
                    return -1
            with ThreadPoolExecutor(max_workers=workers) as ex:
                got = list(ex.map(one, todo))
            n = sum(1 for g in got if g == 1)
            out["fetched"] += n
            out["failed"] += sum(1 for g in got if g == -1)
            out["sports"].setdefault(sport, {"months": 0, "files": 0})
            out["sports"][sport]["months"] += 1
            out["sports"][sport]["files"] += n
            log.info("%s %s: %d listed, %d fetched", sport, month[0].strftime("%Y-%m"), len(paths), n)
            if not budget.left():
                break
    out["stopped"] = bool(iters) and not budget.left()
    return out


# --------------------------------------------------------------------------------------------------------------------
# Reading a market's stream file
# --------------------------------------------------------------------------------------------------------------------

def _ts(ms) -> datetime:
    return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc)


def _iso_utc(text) -> datetime | None:
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None


class _MarketState:
    """One market's reading of a stream file: its last definition, the prices standing at each mark, the close."""

    def __init__(self):
        self.defn: dict = {}
        self.first: dict[int, float] = {}
        self.last_pre: dict[int, float] = {}
        self.ltp_now: dict[int, float] = {}
        self.marks: dict[str, dict[int, float]] = {k: {} for k in MARKS_MIN}
        self.pending = sorted(MARKS_MIN.items(), key=lambda kv: -kv[1])
        self.off = None
        self.inplay_at = None

    def read(self, pt: datetime, mc: dict) -> None:
        d = mc.get("marketDefinition")
        # the marks pass before this message's prices: the price standing at the mark is the last one before it
        if self.off is not None and self.inplay_at is None:
            while self.pending and pt >= self.off - timedelta(minutes=self.pending[0][1]):
                self.marks[self.pending[0][0]] = dict(self.ltp_now)
                self.pending.pop(0)
        if d:
            self.defn = d
            self.off = _iso_utc(d.get("marketTime")) or self.off
            if d.get("inPlay") and self.inplay_at is None:
                self.inplay_at = pt
        for rc in mc.get("rc") or []:
            if "ltp" not in rc:
                continue
            sid, p = int(rc["id"]), float(rc["ltp"])
            self.ltp_now[sid] = p
            self.first.setdefault(sid, p)
            if self.inplay_at is None:
                self.last_pre[sid] = p

    def rows(self, market_id: str) -> list[dict]:
        defn, out = self.defn, []
        for r in defn.get("runners") or []:
            sid = int(r.get("id"))
            row = {
                "market_id": market_id, "event_id": defn.get("eventId"), "event_name": defn.get("eventName"),
                "event_type_id": defn.get("eventTypeId"), "market_type": defn.get("marketType"),
                "market_name": defn.get("name"), "country": defn.get("countryCode"), "venue": defn.get("venue"),
                "market_time": defn.get("marketTime"), "open_date": defn.get("openDate"),
                "turned_in_play": defn.get("turnInPlayEnabled"),
                "went_in_play_utc": self.inplay_at.isoformat() if self.inplay_at else None,
                "number_of_winners": defn.get("numberOfWinners"), "settled": defn.get("settledTime"),
                "bsp_market": defn.get("bspMarket"),
                "selection_id": sid, "runner_name": r.get("name"), "handicap": r.get("hc"),
                "sort_priority": r.get("sortPriority"), "runner_status": r.get("status"),
                "won": {"WINNER": 1, "LOSER": 0, "PLACED": 1}.get(r.get("status")),
                "bsp": r.get("bsp"), "ltp_first": self.first.get(sid), "ltp_close": self.last_pre.get(sid),
            }
            for k in MARKS_MIN:
                row[f"ltp_{k}"] = self.marks[k].get(sid)
            out.append(row)
        return out


def parse_market(raw: bytes) -> list[dict]:
    """A stream file (bz2 or plain JSON lines of 'mcm' messages) -> one row a runner a market: definition, the
    first traded price, the price traded at each mark before the scheduled off, the close (last before in play) and
    the result. A market's own file holds one market; an event's file (named by the event) may hold several."""
    text = bz2.decompress(raw) if raw[:3] == b"BZh" else raw
    markets: dict[str, _MarketState] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        msg = json.loads(line)
        pt = _ts(msg.get("pt", 0))
        for mc in msg.get("mc") or []:
            mid = mc.get("id")
            if mid is None:
                continue
            markets.setdefault(mid, _MarketState()).read(pt, mc)
    rows = []
    for mid, st in markets.items():
        if st.defn:
            rows += st.rows(mid)
    return rows


def _parse_logged(item: tuple[str, bytes]) -> tuple[str, list[dict] | None]:
    try:
        return item[0], parse_market(item[1])
    except Exception:
        return item[0], None


def import_tar(store: Store, path: str, sport: str, workers: int = 16, build_tables: bool = True) -> dict:
    """A bundle downloaded from the Historic Data website (data.tar: BASIC/<year>/<Mon>/<day>/<event>/<market>.bz2)
    into the store, each file under the sport's raw folder (the API's fetch knows it by its path from the plan on, and does
    not fetch it again), then the
    sport's tables for the years it touched, read from the bundle itself."""
    import tarfile
    from concurrent.futures import ProcessPoolExecutor
    have = store.listing(f"betfair_historic/raw/{sport_dir(sport)}/")
    import pandas as pd
    out = {"files": 0, "stored": 0, "already": 0, "unreadable": 0, "years": {}}
    by_year: dict[int, list] = {}                         # a frame a batch: a dict a runner would not fit in memory

    def members():
        with tarfile.open(path) as t:
            for m in t:
                if m.isfile() and m.name.endswith(".bz2"):
                    yield "/" + m.name.lstrip("./"), t.extractfile(m).read()

    def put(item):
        key = raw_key(sport, item[0])
        if key in have:
            return 0
        store.put(key, item[1])
        return 1

    batch: list[tuple[str, bytes]] = []

    def flush(pool, ex):
        nonlocal batch
        stored = list(ex.map(put, batch))
        out["stored"] += sum(stored)
        out["already"] += len(stored) - sum(stored)
        if build_tables:
            got: dict[int, list[dict]] = {}
            for name, rows in pool.map(_parse_logged, batch, chunksize=64):
                if rows is None:
                    out["unreadable"] += 1
                    continue
                y = path_year(name)
                if y is not None:
                    got.setdefault(y, []).extend(rows)
            for y, rows in got.items():
                by_year.setdefault(y, []).append(pd.DataFrame(rows))
        batch = []

    with ThreadPoolExecutor(max_workers=workers) as ex, ProcessPoolExecutor() as pool:
        for item in members():
            batch.append(item)
            out["files"] += 1
            if len(batch) >= 5000:
                flush(pool, ex)
                log.info("%s: %d files read, %d stored", path, out["files"], out["stored"])
        if batch:
            flush(pool, ex)
    held = [path_year(k) for k in store.listing(f"betfair_historic/raw/{sport_dir(sport)}/")]
    for y, frames in sorted(by_year.items()):
        key = f"betfair_historic/markets_{sport_dir(sport)}_{y}.parquet"
        old = store.get_parquet(key)
        df = pd.concat(([old] if old is not None and len(old) else []) + frames, ignore_index=True)
        df = df.drop_duplicates(["market_id", "selection_id", "handicap"], keep="last")
        store.put_parquet(key, df)
        _note_built(store, sport, y, held.count(y))
        out["years"][y] = {"markets": int(df["market_id"].nunique()) if len(df) else 0, "runners": len(df),
                           "with_close": int(df["ltp_close"].notna().sum()) if len(df) else 0}
    log.info("imported %s: %s", path, out)
    return out


def _manifest_key(sport: str, year: int) -> str:
    return f"betfair_historic/built_{sport_dir(sport)}_{year}.json"


def _note_built(store: Store, sport: str, year: int, files: int) -> None:
    store.put(_manifest_key(sport, year), json.dumps({"files": files}).encode())


def _built_files(store: Store, sport: str, year: int) -> int | None:
    raw = store.get(_manifest_key(sport, year))
    try:
        return int(json.loads(raw)["files"]) if raw else None
    except (ValueError, KeyError):
        return None


def build(store: Store, sports: list[str] | None = None, years: list[int] | None = None, force: bool = False,
          workers: int = 16) -> dict:
    """The closing-price tables, a sport and year each, from the raw files (the year in the service's path). A
    sport-year whose file count is the one its table was last built from is left as it is, unless ``force``."""
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    out = {}
    for sport in sports or list(PLAN_SCOPE):
        by_year: dict[int, list[str]] = {}
        for k in store.listing(f"betfair_historic/raw/{sport_dir(sport)}/"):
            y = path_year(k)
            if y is not None and (not years or y in years):
                by_year.setdefault(y, []).append(k)
        for y, keys in sorted(by_year.items()):
            name = f"{sport_dir(sport)}_{y}"
            if not force and _built_files(store, sport, y) == len(keys):
                out[name] = {"files": len(keys), "unchanged": True}
                continue
            frames, bad = [], 0
            keys = sorted(keys)
            with ThreadPoolExecutor(max_workers=workers) as ex, ProcessPoolExecutor() as pool:
                for i in range(0, len(keys), 5000):
                    chunk = keys[i:i + 5000]
                    raw = list(ex.map(store.get, chunk))
                    rows = []
                    for _, got in pool.map(_parse_logged, zip(chunk, raw), chunksize=64):
                        if got is None:
                            bad += 1
                        else:
                            rows += got
                    frames.append(pd.DataFrame(rows))
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if len(df):
                df = df.drop_duplicates(["market_id", "selection_id", "handicap"], keep="last")
            store.put_parquet(f"betfair_historic/markets_{name}.parquet", df)
            _note_built(store, sport, y, len(keys))
            out[name] = {"files": len(keys), "unreadable": bad,
                         "markets": int(df["market_id"].nunique()) if len(df) else 0, "runners": len(df),
                         "with_close": int(df["ltp_close"].notna().sum()) if len(df) else 0}
    log.info("betfair historic tables: %s", out)
    return out


IMPORT_QUEUE = "sources/betfair_imports.json"


def import_queue(store: Store, queue: str = IMPORT_QUEUE, dest: str = ".") -> dict:
    """The bundles listed in the queue file ([{"name", "url", "sport"}]), each imported once: a bundle the store
    notes as imported (betfair_historic/imports/<name>.json) is not downloaded again."""
    import subprocess
    from pathlib import Path
    out = {}
    items = json.loads(Path(queue).read_text()) if Path(queue).exists() else []
    for it in items:
        name, url, sport = it["name"], it["url"], it["sport"]
        marker = f"betfair_historic/imports/{name}.json"
        if store.get(marker) is not None:
            out[name] = "already imported"
            continue
        tar = Path(dest) / f"{name}.tar"
        subprocess.run(["curl", "-sSL", "--retry", "5", "-C", "-", "-o", str(tar), url], check=True)
        got = import_tar(store, str(tar), sport)
        store.put(marker, json.dumps({**it, **got}, default=str).encode())
        tar.unlink(missing_ok=True)
        out[name] = got
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--my-data", action="store_true", help="what the account holds on the service")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--sport", default=None, help="comma-separated service names, e.g. Tennis,Soccer")
    ap.add_argument("--from", dest="date_from", default=str(FIRST))
    ap.add_argument("--to", dest="date_to", default=None)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--root", default=None, help="a local folder in place of S3")
    ap.add_argument("--import-tar", default=None, help="a bundle downloaded from the website (data.tar), with --sport")
    ap.add_argument("--import-queue", nargs="?", const=IMPORT_QUEUE, default=None,
                    help=f"import each bundle the queue file lists, once (default {IMPORT_QUEUE})")
    ap.add_argument("--force", action="store_true", help="--build: rebuild every table, changed or not")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    store = Store(root=a.root)
    sports = [s.strip() for s in a.sport.split(",")] if a.sport else None
    out = {}
    if a.my_data or a.fetch:
        hd = Historic(login_token())
        if a.my_data:
            held = hd.my_data()
            out["my_data"] = held
            out["held_sports"] = held_sports(held)
        if a.fetch:
            out["fetch"] = fetch(store, hd, sports, date.fromisoformat(a.date_from),
                                 date.fromisoformat(a.date_to) if a.date_to else None, a.max_minutes)
    if a.import_tar:
        if not sports or len(sports) != 1:
            ap.error("--import-tar needs the bundle's one --sport, e.g. 'Greyhound Racing'")
        out["import"] = import_tar(store, a.import_tar, sports[0])
    if a.import_queue:
        out["import_queue"] = import_queue(store, a.import_queue)
    if a.build:
        out["build"] = build(store, sports, force=a.force)
    if not out:
        ap.print_help()
        return 1
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
