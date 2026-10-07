"""The tracks' own racecards from greyhounds.today, recorded through a race day (the owner's ask of 7 Oct 2026: find
a source for the weigh-in weights, going, prize money and handicap marks before the off).

greyhounds.today publishes the digital cards of twelve GB tracks as JSON: ``/api/tracks`` the tracks,
``/api/content/<track>`` the day's card files, ``/api/racecard/<file>`` a card. A card carries each runner's
handicap and a ``weight`` (filled in after kennelling, the GBGB weigh-in), the race's prizes and handicap flag, and
each past run's going and weight. This only records: every poll writes each card that changed since the last one,
raw, to ``sources/greyhounds_today/<day>/<HHMMSS>Z_<track>_<file>.json``, and a poll log to
``sources/greyhounds_today/<day>/polls.jsonl`` (when each track's cards appeared, how many runners have a weight).
Nothing here feeds the model or the trader; it is read-only and polite (one request at a time, a pause between).

    python -m greyhound.today_cards --until 22:30 --every 300
    python -m greyhound.today_cards --once --store /tmp/sources
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from urllib.parse import quote

import requests

log = logging.getLogger(__name__)

BASE = "https://greyhounds.today/api"
PREFIX = "greyhounds_today"
HEADERS = {"User-Agent": "ashcroft-research/1.0 (read-only racecard recorder)"}


def _get(session, path: str, timeout: float = 20.0):
    r = session.get(f"{BASE}/{path}", headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def card_files(content) -> list[str]:
    """The card file names in a ``/api/content/<track>`` reply (a list of names, or of objects carrying one)."""
    items = content.get("content", []) if isinstance(content, dict) else content or []
    out = []
    for it in items:
        if isinstance(it, str):
            out.append(it)
        elif isinstance(it, dict):
            for k in ("filename", "file", "name", "key", "id"):
                if it.get(k):
                    out.append(str(it[k]))
                    break
    return out


def weight_count(card) -> tuple[int, int]:
    """(runners, runners with a weight) anywhere in a card: every dict holding a ``weight`` object or number that
    sits beside a runner's name or trap. Counted loosely: the card's exact shape is what this recorder is for."""
    runners = weighted = 0

    def walk(x):
        nonlocal runners, weighted
        if isinstance(x, dict):
            if "weight" in x and any(k in x for k in ("trap", "name", "dog", "greyhound", "dogName")):
                runners += 1
                w = x.get("weight")
                v = w.get("weight") if isinstance(w, dict) else w
                if v not in (None, "", 0):
                    weighted += 1
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(card)
    return runners, weighted


def poll(store, session, seen: dict, pause: float = 1.0, now=None) -> dict:
    """One pass over every track: each card that changed is written; returns the pass's summary."""
    t = now or datetime.now(timezone.utc)
    day, stamp = t.strftime("%Y-%m-%d"), t.strftime("%H%M%S")
    summary = {"utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "tracks": {}}
    try:
        tracks = _get(session, "tracks")
    except Exception as e:                                   # the site down: say so, try again next poll
        summary["error"] = f"tracks: {e}"
        return summary
    for track in tracks:
        try:
            files = card_files(_get(session, f"content/{quote(str(track))}"))
        except Exception as e:
            summary["tracks"][track] = {"error": str(e)}
            continue
        row = {"cards": len(files), "written": 0, "runners": 0, "weighted": 0}
        for f in files:
            time.sleep(pause)
            try:
                card = _get(session, f"racecard/{quote(f)}")
            except Exception as e:
                row.setdefault("errors", []).append(f"{f}: {e}")
                continue
            raw = json.dumps(card, sort_keys=True).encode()
            h = hashlib.sha1(raw).hexdigest()
            n, w = weight_count(card)
            row["runners"] += n
            row["weighted"] += w
            if seen.get((track, f)) != h:
                seen[(track, f)] = h
                safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in f"{track}_{f}")
                store.put(f"{PREFIX}/{day}/{stamp}Z_{safe.removesuffix('.json')}.json", raw)
                row["written"] += 1
        summary["tracks"][track] = row
        time.sleep(pause)
    prev = store.get(f"{PREFIX}/{day}/polls.jsonl") or b""
    store.put(f"{PREFIX}/{day}/polls.jsonl", prev + (json.dumps(summary) + "\n").encode())
    return summary


def main(argv=None) -> int:
    from betfair_recorder import uk_time_on, uk_today
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None, help="the sources store root (default S3)")
    ap.add_argument("--once", action="store_true", help="one pass, then stop")
    ap.add_argument("--until", default="22:30", help="stop at this UK time")
    ap.add_argument("--every", type=float, default=300.0, help="seconds between passes")
    ap.add_argument("--max-minutes", type=float, default=345.0, help="stop after this long (a hosted job's limit)")
    a = ap.parse_args(argv)
    store, session, seen = Store(root=a.store), requests.Session(), {}
    end = min(uk_time_on(uk_today(), a.until),
              datetime.fromtimestamp(time.time() + 60 * a.max_minutes, tz=timezone.utc))
    while True:
        s = poll(store, session, seen)
        busy = {k: v for k, v in s["tracks"].items() if v.get("cards")}
        log.info("poll %s: %d tracks with cards %s", s["utc"], len(busy),
                 {k: (v["cards"], v["runners"], v["weighted"]) for k, v in busy.items()})
        if a.once or datetime.now(timezone.utc) >= end:
            break
        time.sleep(a.every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
