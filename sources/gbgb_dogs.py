"""Each declared dog's GBGB record, saved daily: a point-in-time history from 14 Nov 2026 forward (the owner, 14 Nov
2026: "Yes do these").

GBGB's results carry the dog's trainer, owner and season as they are now, not as they were on the day of the run
(sources/gbgb.py ``rows``: every past row shows the current values), so a feature read from them leaks. Saving the
dog's record on each day it is declared fixes that from today on.

The record: GBGB's public ``https://api.gbgb.org.uk/api/dogs/<dogId>`` (the API sources/gbgb.py reads; checked by hand
on 14 Nov 2026: dog 660182 gave {"dogId", "dogName", "dogSire", "dogDam", "dogOrigin", "dogBorn", "dogColour",
"dogSex", "dogSeason": "24.My.26", "trainerName", "ownerName"}, an unknown id a 404).

The day's dogs: the GBGB ids of the day's cards as the greyhound live job prices them
(``sources/greyhound/live/predictions/<day>.csv``, hourly from 07:21 UTC; a positive ``dog_id`` is a GBGB dog, a
negative one a debutant GBGB has no run for). When that file is not there yet: the recorder's Betfair catalogue
(``betfair_live/<day>/markets_greyhound.csv``) matched by name to the GBGB runs of this year and last year
(greyhound/card.py ``norm_name``, the most recent dog of the name). GBGB itself publishes no card through its API.

Stored: ``sources/gbgb_dogs/<day>/dogs.jsonl.gz``, one line a fetch: {"fetched_utc", "dog_id", "status", "body"}
(``status`` GBGB's HTTP status, 0 when no reply came; ``body`` the JSON as served, or the error text). A rerun the same
day skips the dogs already saved (a 200 or a 404) and tries the rest again. Polite: at most ``--rate`` requests a second
(default 2), retries with a growing wait on a fault, a 429 or a 5xx (Retry-After honoured), at most ``--max`` dogs a run.
Read-only to GBGB; writes the store only.

    python -m sources.gbgb_dogs --date 2026-11-14 --max 3000
    python -m sources.gbgb_dogs --ids 660182 --store /tmp/sources
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import time
from datetime import date, datetime, timedelta, timezone

import requests

log = logging.getLogger(__name__)

API = "https://api.gbgb.org.uk/api"
PREFIX = "gbgb_dogs"
PREDICTIONS = "greyhound/live/predictions/{day}.csv"
USER_AGENT = "ashcroft-research/1.0 (read-only greyhound record archiver)"
DONE = (200, 404)                      # a status a rerun does not ask again
RETRY = (429, 500, 502, 503, 504)
FLUSH_EVERY = 100
MAX_WAIT = 60.0
RAW_DAYS_BACK = 14                     # the GBGB raw days after the year tables read for names (the fallback)


def day_key(day: date) -> str:
    return f"{PREFIX}/{day:%Y-%m-%d}/dogs.jsonl.gz"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class RateLimiter:
    """At most ``rate`` calls a second: ``wait()`` sleeps until the next call may go."""

    def __init__(self, rate: float = 2.0, clock=time.monotonic, sleep=time.sleep):
        self.interval = 1.0 / float(rate)
        self.clock, self.sleep = clock, sleep
        self.next = None

    def wait(self) -> None:
        now = self.clock()
        if self.next is not None and now < self.next:
            self.sleep(self.next - now)
            now = self.clock()
        self.next = max(now, self.next or now) + self.interval


# --------------------------------------------------------------------------------------------------------------------
# The day's dogs
# --------------------------------------------------------------------------------------------------------------------

def ids_from_predictions(store, day: date) -> list[int]:
    import pandas as pd
    raw = store.get(PREDICTIONS.format(day=f"{day:%Y-%m-%d}"))
    if not raw:
        return []
    df = pd.read_csv(io.BytesIO(raw), usecols=lambda c: c == "dog_id")
    if "dog_id" not in df:
        return []
    ids = pd.to_numeric(df.dog_id, errors="coerce").dropna().astype("int64")
    return sorted({int(i) for i in ids if i > 0})


def ids_from_catalogue(store, day: date, markets=None, s3=None) -> list[int]:
    """The day's GB WIN runners in the recorder's catalogue, each matched by name to the most recent GBGB dog of that
    name in the runs of ``day``'s year and the year before (and the raw days held since the tables were built)."""
    import pandas as pd
    from greyhound.card import norm_name
    from sources import gbgb
    if markets is None:
        from betfair_recorder import _read_day_csv
        markets = _read_day_csv(day, "markets_greyhound.csv", s3=s3)
    if markets is None or not len(markets):
        return []
    w = markets
    if "market_type" in w:
        w = w[w.market_type.fillna("WIN") == "WIN"]
    if "country" in w:
        w = w[w.country.fillna("GB") == "GB"]
    names = set(w.runner_name.astype(str).str.replace(r"^\d+\.\s*", "", regex=True).map(norm_name))
    frames = []
    for y in (day.year - 1, day.year):
        f = store.get_parquet(f"gbgb/runs_{y}.parquet")
        if f is not None and len(f):
            frames.append(f[["dog_id", "dog_name", "race_date"]])
    extra = []
    for back in range(1, RAW_DAYS_BACK + 1):
        raw = store.get(gbgb.raw_key(day - timedelta(days=back)))
        if raw:
            extra += [{k: r[k] for k in ("dog_id", "dog_name", "race_date")}
                      for r in gbgb.rows(json.loads(gbgb.gunzip(raw)))]
    if extra:
        frames.append(pd.DataFrame(extra))
    if not frames:
        return []
    h = pd.concat(frames, ignore_index=True).dropna(subset=["dog_id"]).sort_values("race_date", kind="stable")
    last = h.assign(_n=h.dog_name.map(norm_name)).drop_duplicates("_n", keep="last")
    return sorted({int(i) for i in last[last._n.isin(names)].dog_id if int(i) > 0})


def day_dog_ids(store, day: date, s3=None) -> tuple[list[int], str]:
    ids = ids_from_predictions(store, day)
    if ids:
        return ids, "predictions"
    ids = ids_from_catalogue(store, day, s3=s3)
    return ids, ("catalogue" if ids else "none")


# --------------------------------------------------------------------------------------------------------------------
# Fetching and saving
# --------------------------------------------------------------------------------------------------------------------

def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return s


def fetch_dog(s, dog_id: int, limiter: RateLimiter, tries: int = 4, backoff: float = 2.0, timeout: float = 30.0,
              sleep=time.sleep) -> dict:
    """One dog's record as a stored line; never raises: a failure is a line with its status."""
    url = f"{API}/dogs/{int(dog_id)}"
    status, body, wait = 0, None, backoff
    for attempt in range(tries):
        limiter.wait()
        try:
            r = s.get(url, timeout=timeout)
        except requests.RequestException as exc:
            status, body = 0, f"{type(exc).__name__}: {str(exc)[:300]}"
        else:
            status = int(r.status_code)
            try:
                body = r.json()
            except ValueError:
                body = (r.text or "")[:2000]
            if status not in RETRY:
                break
            ra = r.headers.get("Retry-After") if getattr(r, "headers", None) else None
            try:
                wait = min(MAX_WAIT, float(ra)) if ra else wait
            except ValueError:
                pass
        if attempt < tries - 1:
            log.warning("gbgb dog %s: %s; again in %.0f s", dog_id, status or body, wait)
            sleep(wait)
            wait = min(MAX_WAIT, wait * 2)
    return {"fetched_utc": utc_now(), "dog_id": int(dog_id), "status": status, "body": body}


def read_lines(store, day: date) -> list[dict]:
    raw = store.get(day_key(day))
    if not raw:
        return []
    out = []
    for line in gzip.decompress(raw).decode().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def write_lines(store, day: date, lines: list[dict]) -> None:
    text = "".join(json.dumps(x, separators=(",", ":"), sort_keys=True) + "\n" for x in lines)
    store.put(day_key(day), gzip.compress(text.encode(), mtime=0))


def save_day(store, day: date, ids, max_dogs: int = 3000, rate: float = 2.0, s=None, limiter=None,
             sleep=time.sleep, flush_every: int = FLUSH_EVERY) -> dict:
    """Fetch and save each of ``ids`` not saved yet on ``day``, at most ``max_dogs`` of them."""
    lines = read_lines(store, day)
    done = {int(x["dog_id"]) for x in lines if x.get("status") in DONE}
    todo = [int(i) for i in dict.fromkeys(ids) if int(i) not in done]
    batch = todo[:max(0, int(max_dogs))]
    s = s or session()
    limiter = limiter or RateLimiter(rate)
    out = {"day": f"{day:%Y-%m-%d}", "key": store.where(day_key(day)), "ids": len(set(ids)),
           "already_saved": len(set(ids) & done), "asked": len(batch), "left_for_next_run": len(todo) - len(batch),
           "status": {}}
    new = 0
    for dog_id in batch:
        rec = fetch_dog(s, dog_id, limiter, sleep=sleep)
        lines.append(rec)
        new += 1
        out["status"][str(rec["status"])] = out["status"].get(str(rec["status"]), 0) + 1
        if new % flush_every == 0:
            write_lines(store, day, lines)
            log.info("gbgb dogs %s: %d of %d fetched", out["day"], new, len(batch))
    if new:
        write_lines(store, day, lines)
    out["fetched"] = new
    out["lines_in_file"] = len(lines)
    return out


# --------------------------------------------------------------------------------------------------------------------
# The point-in-time table
# --------------------------------------------------------------------------------------------------------------------

FIELDS = {"dog_name": "dogName", "trainer": "trainerName", "owner": "ownerName", "season": "dogSeason",
          "sex": "dogSex", "born": "dogBorn", "sire": "dogSire", "dam": "dogDam", "colour": "dogColour",
          "origin": "dogOrigin"}


def table(store, first: date | None = None, last: date | None = None):
    """Every saved record as a row (dog_id, day, fetched_utc, the record's fields), the first good fetch of a dog a
    day: what GBGB said of the dog on that day."""
    import pandas as pd
    rows = []
    for key in sorted(store.listing(f"{PREFIX}/")):
        if not key.endswith("/dogs.jsonl.gz"):
            continue
        d = date.fromisoformat(key.split("/")[-2])
        if (first and d < first) or (last and d > last):
            continue
        for x in read_lines(store, d):
            b = x.get("body")
            if x.get("status") != 200 or not isinstance(b, dict):
                continue
            row = {"dog_id": int(x["dog_id"]), "day": f"{d:%Y-%m-%d}", "fetched_utc": x.get("fetched_utc")}
            row.update({k: b.get(v) for k, v in FIELDS.items()})
            rows.append(row)
    cols = ["dog_id", "day", "fetched_utc", *FIELDS]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=cols)
    df["fetched_utc"] = pd.to_datetime(df.fetched_utc, utc=True, errors="coerce")
    return (df.sort_values(["dog_id", "fetched_utc"], kind="stable").drop_duplicates(["dog_id", "day"], keep="first")
            .reset_index(drop=True))


def as_of(snapshots, keys, when: str = "race_date"):
    """``keys`` (dog_id and a ``when`` day) with the record saved on or before that day (the latest such): a run's
    trainer, owner and season as GBGB gave them then; nothing when no record was saved by then."""
    import pandas as pd
    k = keys.copy()
    k["_t"] = pd.to_datetime(k[when]).astype("datetime64[ns]")
    snap = snapshots.copy()
    snap["_t"] = pd.to_datetime(snap["day"]).astype("datetime64[ns]")
    k["_i"] = range(len(k))
    m = pd.merge_asof(k.sort_values("_t"), snap.drop(columns=["day"]).sort_values("_t"), on="_t", by="dog_id",
                      direction="backward", suffixes=("", "_gbgb"))
    return m.sort_values("_i").drop(columns=["_t", "_i"]).reset_index(drop=True)


def main(argv=None) -> int:
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", default=None, help="the UK day, YYYY-MM-DD (default today, UK)")
    ap.add_argument("--max", type=int, default=3000, help="at most this many dogs fetched a run")
    ap.add_argument("--rate", type=float, default=2.0, help="requests a second, at most (capped at 2)")
    ap.add_argument("--ids", default=None, help="these GBGB dog ids instead of the day's (comma separated)")
    ap.add_argument("--store", default=None, help="the sources store root (default S3)")
    a = ap.parse_args(argv)
    if a.date:
        day = date.fromisoformat(a.date)
    else:
        from betfair_recorder import uk_today
        day = uk_today()
    store = Store(root=a.store)
    if a.ids:
        ids, source = [int(x) for x in a.ids.split(",") if x.strip()], "given"
    else:
        ids, source = day_dog_ids(store, day)
    out = {"source": source}
    if not ids:
        out.update({"day": f"{day:%Y-%m-%d}", "note": "no data found: no GBGB dog ids for the day"})
    else:
        out.update(save_day(store, day, ids, max_dogs=a.max, rate=min(2.0, max(0.1, a.rate))))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
