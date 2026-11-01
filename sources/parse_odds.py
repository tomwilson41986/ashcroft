"""Odds from three Parse (parse.bot) APIs the owner subscribed to (1 Nov 2026): Flashscore, Pinnacle and Oddschecker.

What each gives (probed 1 Nov 2026; reports/open_close/parse_odds.md):

- Flashscore ``get_fixtures_with_odds``: every match of one sport on one day with the opening and the final match-winner
  odds and the score, 1 credit a sport-day, the last 7 days only. ``get_match_odds``: every market of one match, each
  bookmaker's opening and final price, 1 credit a match, back to at least May 2026. The bookmakers are those Flashscore
  shows to Parse's (US) proxy: BetMGM, bet365.us, FanDuel.
- Pinnacle ``get_matchups`` + ``get_odds``: every pre-event line of a sport now (American prices, the line's version and
  its maximum stake), 3 + 2 credits a sport. No history: polling is what makes an open and a close.
- Oddschecker ``get_horse_racing_meetings`` + ``get_horse_race_odds``: every runner's price at about 25 UK bookmakers
  (Betfair Sportsbook and Matchbook among them) now, 1 + 3 credits a race. No history.

Raw responses go to ``sources/parse/raw/<source>/<endpoint>/<day>/<HHMMSS>_<tag>.json.gz`` in the capture store (S3, or
``--root``), each wrapped with when it was fetched (UTC), the call and the credits it cost. ``--build`` turns the raw
files into tables under ``sources/parse/`` (rebuilt whole from raw, so it is safe to rerun). ``panel_*`` turn the tables
into the standard open-close panel (``sources/open_close.py SCHEMA``) for ``research/open_close``.

The key is read from ``PARSE_API_KEY`` and never written anywhere. A floor on the account's remaining credits
(``PARSE_CREDIT_FLOOR``, default 2,000) stops every collector before it spends the last of the month's quota.

    python -m sources.parse_odds --flashscore-days 7 --sports all --root /tmp/parse      # the last 7 finished days
    python -m sources.parse_odds --pinnacle --pin-sports 33,29,8                          # one snapshot: tennis, soccer, cricket
    python -m sources.parse_odds --oddschecker                                            # one snapshot of the UK/IE races to come
    python -m sources.parse_odds --oddschecker --marks 120,60,15 --until 21:30            # each race at T-120/60/15 (a loop)
    python -m sources.parse_odds --build
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd
import requests

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sources.common import Store, gunzip, gz  # noqa: E402

API = "https://api.parse.bot"
SOURCE = "parse"
UK = ZoneInfo("Europe/London")

#: The owner's subscribed copies (1 Nov 2026); overridable. Ids, not credentials: a call needs the key as well.
SCRAPERS = {
    "flashscore": os.getenv("PARSE_FLASHSCORE_ID", "c11a9c7a-593c-4737-b02a-219fa3656c59"),
    "pinnacle": os.getenv("PARSE_PINNACLE_ID", "a93dafa7-1c95-4969-968b-12aa15c9f54f"),
    "oddschecker": os.getenv("PARSE_ODDSCHECKER_ID", "0085799c-319c-4b87-a3d5-687a633394ae"),
}
#: Endpoints that take a JSON body (POST); the rest take a query string (from each API's OpenAPI document).
POST = {"flashscore": {"get_fixtures_with_odds", "get_match_odds", "get_team_results", "get_daily_fixtures",
                       "get_match_detail", "get_match_lineups", "get_prioritized_fixtures", "search"}}
FLASHSCORE_SPORTS = ["football", "tennis", "basketball", "hockey", "american_football", "volleyball", "snooker",
                     "boxing", "golf", "table_tennis", "esports", "handball", "rugby_union", "rugby_league"]
#: Pinnacle's sport ids (get_sports, 1 Nov 2026).
PINNACLE_SPORTS = {33: "tennis", 29: "soccer", 8: "cricket", 4: "basketball", 19: "hockey", 17: "golf", 28: "snooker"}


class OutOfCredits(RuntimeError):
    pass


class Parse:
    """A thin client: one method, the call; it records the credits each call cost and what is left."""

    def __init__(self, key: str | None = None, floor: int | None = None, http=None, pause: float = 0.25):
        self.key = key or os.getenv("PARSE_API_KEY") or ""
        if not self.key:
            raise SystemExit("PARSE_API_KEY is not set")
        self.floor = int(os.getenv("PARSE_CREDIT_FLOOR", "2000")) if floor is None else floor
        self.http = http or requests.Session()
        self.pause = pause
        self.spent = 0
        self.remaining: int | None = None
        import threading
        self._lock = threading.Lock()

    def call(self, source: str, endpoint: str, params: dict | None = None, tries: int = 3) -> dict:
        if self.remaining is not None and self.remaining < self.floor:
            raise OutOfCredits(f"{self.remaining} credits left, under the floor of {self.floor}")
        url = f"{API}/scraper/{SCRAPERS[source]}/{endpoint}"
        head = {"X-API-Key": self.key}
        params = {k: v for k, v in (params or {}).items() if v is not None}
        r = None
        for attempt in range(tries):
            try:
                if endpoint in POST.get(source, set()):
                    r = self.http.post(url, json=params, headers=head, timeout=150)
                else:
                    r = self.http.get(url, params=params, headers=head, timeout=150)
            except requests.RequestException as exc:          # a dropped connection: wait and try again
                r = None
                last = str(exc)[:200]
                time.sleep(5 * (attempt + 1))
                continue
            charged = int(r.headers.get("x-credits-charged") or 0)
            with self._lock:
                self.spent += charged
                if r.headers.get("x-credits-remaining"):
                    rem = int(r.headers["x-credits-remaining"])
                    self.remaining = rem if self.remaining is None else min(self.remaining, rem)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            time.sleep(self.pause)
            body = r.json() if r.content else {}
            return {"status_code": r.status_code, "credits": charged, "body": body}
        if r is None:
            return {"status_code": 0, "credits": 0, "body": {"status": "error", "error": last}}
        return {"status_code": r.status_code, "credits": 0, "body": {"status": "error", "error": r.text[:300]}}


# --------------------------------------------------------------------------------------------------- raw store

def _now() -> datetime:
    return datetime.now(timezone.utc)


def raw_key(source: str, endpoint: str, fetched: datetime, tag: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", tag)[:80]
    # microseconds in the name: two snapshots in one second must not overwrite each other
    return f"{SOURCE}/raw/{source}/{endpoint}/{fetched:%Y-%m-%d}/{fetched:%H%M%S%f}_{tag}.json.gz"


def save(store: Store, p: Parse, source: str, endpoint: str, params: dict, tag: str) -> dict:
    fetched = _now()
    got = p.call(source, endpoint, params)
    env = {"fetched_utc": fetched.isoformat(), "source": source, "endpoint": endpoint, "params": params,
           "status_code": got["status_code"], "credits": got["credits"], "response": got["body"]}
    store.put(raw_key(source, endpoint, fetched, tag), gz(json.dumps(env).encode()))
    return env


def raw_files(store: Store, source: str, endpoint: str) -> list[dict]:
    out = []
    for key in sorted(store.listing(f"{SOURCE}/raw/{source}/{endpoint}/")):
        data = store.get(key)
        if data:
            try:
                out.append(json.loads(gunzip(data)))
            except (OSError, ValueError):
                continue
    return out


def _data(env: dict):
    body = env.get("response") or {}
    return body.get("data") if isinstance(body, dict) and body.get("status") == "success" else None


# --------------------------------------------------------------------------------------------------- collectors

def flashscore_days(store: Store, p: Parse, sports: list[str], offsets: list[int]) -> dict:
    """The daily feed for each sport and day (offset 0 today, -1 yesterday ... -7). A finished day's file holds the
    final odds; today's holds the odds as they stand."""
    done = {}
    held = held_feed_days(store)
    today = _now().date()
    for off in offsets:
        for sport in sports:
            if off < 0 and (sport, (today + timedelta(days=off)).isoformat()) in held:
                continue                                  # a finished day already held: its odds do not change
            env = save(store, p, "flashscore", "get_fixtures_with_odds", {"sport": sport, "day_offset": off},
                       f"{sport}_d{off}")
            d = _data(env) or {}
            done[f"{sport}:{off}"] = (len(d.get("matches") or []), d.get("matches_with_odds"))
    return done


def held_feed_days(store: Store) -> set[tuple[str, str]]:
    """(sport, match day) of every finished day of the daily feed already held, from the raw keys."""
    held = set()
    for key in store.listing(f"{SOURCE}/raw/flashscore/get_fixtures_with_odds/"):
        m = re.search(r"/(\d{4}-\d{2}-\d{2})/\d{6,12}_(.+)_d(-\d+)\.json\.gz$", key)
        if m:
            day = date.fromisoformat(m.group(1)) + timedelta(days=int(m.group(3)))
            held.add((m.group(2), day.isoformat()))
    return held


def flashscore_match_odds(store: Store, p: Parse, match_ids: list[str], skip_held: bool = True, workers: int = 8) -> int:
    held = set()
    if skip_held:
        for key in store.listing(f"{SOURCE}/raw/flashscore/get_match_odds/"):
            m = re.search(r"_([A-Za-z0-9]{8})\.json\.gz$", key)
            if m:
                held.add(m.group(1))
    todo = [m for m in dict.fromkeys(match_ids) if m and m not in held]

    def one(mid):
        try:
            save(store, p, "flashscore", "get_match_odds", {"match_id": mid}, mid)
            return 1
        except OutOfCredits:
            return 0

    if workers <= 1:
        return sum(one(m) for m in todo)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        return sum(ex.map(one, todo))


#: Tour-level singles in Flashscore's competition names (the matches with Betfair money: all_markets.md, tennis)
TOUR_SINGLES = re.compile(r"^(ATP|WTA) - SINGLES")


def flashscore_team_results(store: Store, p: Parse, teams: list[str], sport: str = "tennis", pages: int = 2) -> int:
    """Past results of each team or player (by Flashscore id; the slug is not checked, 1 Nov 2026), ``pages`` pages
    of about 40 each (two pages reach back about a year for a tour player): the route to match ids older than the
    daily feed's 7 days. Players already held at that page are skipped."""
    held = set(store.listing(f"{SOURCE}/raw/flashscore/get_team_results/"))
    held = {re.sub(r"^.*/\d{6,12}_", "", k) for k in held}
    n = 0
    for tid in teams:
        for page in range(1, pages + 1):
            tag = f"{sport}_{tid}_p{page}"
            if f"{tag}.json.gz" in held:
                continue
            env = save(store, p, "flashscore", "get_team_results",
                       {"team_id": tid, "team_slug": "x", "sport": sport, "page": page}, tag)
            n += 1
            if not (_data(env) or {}).get("has_next_page"):
                break
    return n


def table_flashscore_results(store: Store) -> pd.DataFrame:
    rows = []
    for env in raw_files(store, "flashscore", "get_team_results"):
        for r in (_data(env) or {}).get("results") or []:
            rows.append({"match_id": r.get("match_id"), "date_utc": r.get("date"), "competition": r.get("competition"),
                         "home": r.get("home_team"), "away": r.get("away_team"), "score": r.get("score"),
                         "sport": (env.get("params") or {}).get("sport")})
    df = pd.DataFrame(rows)
    return df.drop_duplicates("match_id").reset_index(drop=True) if not df.empty else df


def tennis_history(store: Store, p: Parse, pages: int = 2, odds_cap: int = 2000) -> dict:
    """Tour-level tennis history with each bookmaker's opening and final price: the players of the held feed's ATP and
    WTA tour singles, their last ``pages`` pages of results, then per-match odds (1 credit each) for up to
    ``odds_cap`` tour-level singles matches not yet held, newest first."""
    mt = table_flashscore_matches(store)
    tour = mt[(mt["sport"] == "tennis") & mt["competition"].fillna("").str.match(TOUR_SINGLES)]
    players = list(dict.fromkeys(list(tour["home_id"].dropna()) + list(tour["away_id"].dropna())))
    calls = flashscore_team_results(store, p, players, "tennis", pages)
    res = table_flashscore_results(store)
    res = res[res["competition"].fillna("").str.match(TOUR_SINGLES) & res["score"].fillna("").str.contains(":")]
    res = res.sort_values("date_utc", ascending=False)
    got = flashscore_match_odds(store, p, list(res["match_id"])[:odds_cap])
    return {"players": len(players), "result_calls": calls, "tour_matches_found": len(res), "match_odds_calls": got}


def pinnacle_snapshot(store: Store, p: Parse, sport_ids: list[int]) -> dict:
    out = {}
    for sid in sport_ids:
        a = save(store, p, "pinnacle", "get_matchups", {"sport_id": sid}, f"s{sid}")
        b = save(store, p, "pinnacle", "get_odds", {"sport_id": sid}, f"s{sid}")
        out[sid] = (len(_items(a)), len(_items(b)))
    return out


def _items(env: dict) -> list:
    d = _data(env)
    return (d.get("items") if isinstance(d, dict) else d) or []


def _norm_course(name: str) -> str:
    return re.sub(r"[^a-z]", "", re.sub(r"\(.*?\)", "", (name or "").lower()))


def uk_ie_courses() -> set[str]:
    from model.blocks.travel import COURSES
    return set(COURSES)


def oddschecker_races(store: Store, p: Parse) -> list[dict]:
    """Today's GB/IE races from the meetings list: venue, the race time (UK), its URL."""
    env = save(store, p, "oddschecker", "get_horse_racing_meetings", {}, "meetings")
    keep = uk_ie_courses()
    today = datetime.now(UK).date()
    races = []
    for m in (_data(env) or {}).get("meetings") or []:
        if _norm_course(m.get("venue")) not in keep:
            continue
        for r in m.get("races") or []:
            try:
                hh, mm = map(int, r["time"].split(":"))
            except (KeyError, ValueError):
                continue
            off = datetime(today.year, today.month, today.day, hh, mm, tzinfo=UK)
            races.append({"venue": m["venue"], "time": r["time"], "url": r["url"], "off_utc": off.astimezone(timezone.utc)})
    return races


def oddschecker_snapshot(store: Store, p: Parse, races: list[dict], tag: str = "snap") -> int:
    n = 0
    for r in races:
        save(store, p, "oddschecker", "get_horse_race_odds", {"race_url": r["url"]},
             f"{tag}_{_norm_course(r['venue'])}_{r['time'].replace(':', '')}")
        n += 1
    return n


def oddschecker_marks(store: Store, p: Parse, marks: list[int], until: str, first: bool = True,
                      poll_s: int = 60, clock=_now, sleep=time.sleep) -> dict:
    """Each GB/IE race at each mark (minutes before the off), from now until ``until`` (UK). ``first`` takes every
    race still to run once at the start: the earliest bookmaker book this record holds."""
    races = oddschecker_races(store, p)
    done: set[tuple[str, int]] = set()
    if first:
        todo = [r for r in races if r["off_utc"] > clock()]
        oddschecker_snapshot(store, p, todo, "first")
    hh, mm = map(int, until.split(":"))
    stop = datetime.now(UK).replace(hour=hh, minute=mm, second=0, microsecond=0).astimezone(timezone.utc)
    while clock() < stop and any(r["off_utc"] > clock() for r in races):
        now = clock()
        for r in races:
            for mk in marks:
                due = r["off_utc"] - timedelta(minutes=mk)
                if (r["url"], mk) not in done and due <= now < r["off_utc"]:
                    oddschecker_snapshot(store, p, [r], f"t{mk}")
                    done.add((r["url"], mk))
        sleep(poll_s)
    return {"races": len(races), "marks_taken": len(done), "spent": p.spent, "remaining": p.remaining}


# --------------------------------------------------------------------------------------------------- tables

def _f(x):
    try:
        v = float(x)
        return v if v > 1.0 else None
    except (TypeError, ValueError):
        return None


def fractional(x) -> float | None:
    """Oddschecker's fractional price to decimal: '7/4' 2.75, '2' 3.0 (2/1), 'EVS' 2.0."""
    if x is None:
        return None
    s = str(x).strip().upper()
    if s in ("EVS", "EVENS", "EVEN"):
        return 2.0
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", s)
    if m:
        den = float(m.group(2))
        return 1.0 + float(m.group(1)) / den if den else None
    try:
        return 1.0 + float(s)
    except ValueError:
        return None


def american(x) -> float | None:
    try:
        a = float(x)
    except (TypeError, ValueError):
        return None
    if a >= 100:
        return 1.0 + a / 100.0
    if a <= -100:
        return 1.0 + 100.0 / -a
    return None


def table_flashscore_matches(store: Store) -> pd.DataFrame:
    rows = []
    for env in raw_files(store, "flashscore", "get_fixtures_with_odds"):
        d = _data(env) or {}
        sport = (env.get("params") or {}).get("sport")
        for m in d.get("matches") or []:
            o = m.get("odds") or {}
            sc = m.get("score") or {}
            comp = m.get("competition") or {}
            rows.append({"fetched_utc": env["fetched_utc"], "sport": sport, "match_id": m.get("match_id"),
                         "status": m.get("status"), "start_utc": m.get("start_time"),
                         "competition": comp.get("name"), "competition_id": comp.get("id"),
                         "country": comp.get("country"),
                         "home": (m.get("home_team") or {}).get("name"), "away": (m.get("away_team") or {}).get("name"),
                         "home_id": (m.get("home_team") or {}).get("id"), "away_id": (m.get("away_team") or {}).get("id"),
                         "score_home": sc.get("home"), "score_away": sc.get("away"),
                         "home_open": _f(o.get("home_win_opening")), "draw_open": _f(o.get("draw_opening")),
                         "away_open": _f(o.get("away_win_opening")), "home_last": _f(o.get("home_win")),
                         "draw_last": _f(o.get("draw")), "away_last": _f(o.get("away_win"))})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # one row a match: the latest fetch, so a finished match's final odds win over the odds read before it was played
    df = df.sort_values("fetched_utc").drop_duplicates("match_id", keep="last").reset_index(drop=True)
    return df


def table_flashscore_odds(store: Store) -> pd.DataFrame:
    rows = []
    for env in raw_files(store, "flashscore", "get_match_odds"):
        d = _data(env) or {}
        for mk in d.get("markets") or []:
            for b in mk.get("bookmakers") or []:
                for s in b.get("selections") or []:
                    rows.append({"fetched_utc": env["fetched_utc"], "match_id": d.get("match_id"),
                                 "match_date": d.get("match_date"), "status": d.get("status"),
                                 "home": _name(d.get("home_team")), "away": _name(d.get("away_team")),
                                 "competition": _name(d.get("competition")),
                                 "home_score": d.get("home_score"), "away_score": d.get("away_score"),
                                 "market_type": mk.get("market_type"), "scope": mk.get("scope"),
                                 "bookmaker_id": b.get("bookmaker_id"), "bookmaker": b.get("bookmaker_name"),
                                 "selection": s.get("selection"), "handicap": s.get("handicap"),
                                 "opening": _f(s.get("opening")), "last": _f(s.get("value"))})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    keys = ["match_id", "market_type", "scope", "bookmaker_id", "selection", "handicap"]
    return df.sort_values("fetched_utc").drop_duplicates(keys, keep="last").reset_index(drop=True)


def _name(x):
    return x.get("name") if isinstance(x, dict) else x


def table_pinnacle(store: Store) -> pd.DataFrame:
    """Every price in every snapshot, with its matchup's league, participants and start."""
    mu: dict[int, dict] = {}
    for env in raw_files(store, "pinnacle", "get_matchups"):
        for m in _items(env):
            parts = m.get("participants") or (m.get("parent") or {}).get("participants") or []
            side = {pp.get("alignment"): pp.get("name") for pp in parts}
            mu[m.get("id")] = {"league": (m.get("league") or {}).get("name"),
                               "league_id": (m.get("league") or {}).get("id"),
                               "home": side.get("home"), "away": side.get("away"),
                               "start_utc": m.get("startTime"), "parent_id": (m.get("parent") or {}).get("id"),
                               "is_live": m.get("isLive")}
    rows = []
    for env in raw_files(store, "pinnacle", "get_odds"):
        sid = (env.get("params") or {}).get("sport_id")
        for o in _items(env):
            info = mu.get(o.get("matchupId"), {})
            lim = {l.get("type"): l.get("amount") for l in o.get("limits") or []}
            for pr in o.get("prices") or []:
                rows.append({"fetched_utc": env["fetched_utc"], "sport_id": sid,
                             "sport": PINNACLE_SPORTS.get(int(sid)) if sid else None,
                             "matchup_id": o.get("matchupId"), **info, "market_type": o.get("type"),
                             "period": o.get("period"), "key": o.get("key"), "is_alternate": o.get("isAlternate"),
                             "status": o.get("status"), "cutoff_utc": o.get("cutoffAt"), "version": o.get("version"),
                             "max_risk": lim.get("maxRiskStake"), "designation": pr.get("designation"),
                             "participant_id": pr.get("participantId"), "points": pr.get("points"),
                             "price_american": pr.get("price"), "price": american(pr.get("price"))})
    return pd.DataFrame(rows)


def table_oddschecker(store: Store) -> pd.DataFrame:
    rows = []
    for env in raw_files(store, "oddschecker", "get_horse_race_odds"):
        url = (env.get("params") or {}).get("race_url") or ""
        m = re.search(r"/horse-racing/([^/]+)/(\d{1,2}:\d{2})/", url)
        fetched = datetime.fromisoformat(env["fetched_utc"])
        day = fetched.astimezone(UK).date()
        off = None
        if m:
            hh, mm = map(int, m.group(2).split(":"))
            off = datetime(day.year, day.month, day.day, hh, mm, tzinfo=UK).astimezone(timezone.utc)
        for r in (_data(env) or {}).get("runners") or []:
            for bk, frac in (r.get("odds") or {}).items():
                rows.append({"fetched_utc": env["fetched_utc"], "race_date": day.isoformat(),
                             "course": m.group(1) if m else None, "race_time_uk": m.group(2) if m else None,
                             "off_utc": off.isoformat() if off else None,
                             "minutes_to_off": round((off - fetched).total_seconds() / 60, 1) if off else None,
                             "race_url": url, "horse": r.get("name"), "bookmaker": bk, "fractional": frac,
                             "price": fractional(frac)})
    return pd.DataFrame(rows)


TABLES = {"flashscore_matches": table_flashscore_matches, "flashscore_odds": table_flashscore_odds,
          "flashscore_results": table_flashscore_results,
          "pinnacle_prices": table_pinnacle, "oddschecker_prices": table_oddschecker}


def build(store: Store, out: Store | None = None) -> dict:
    out = out or store
    counts = {}
    for name, fn in TABLES.items():
        df = fn(store)
        counts[name] = len(df)
        if not df.empty:
            out.put_parquet(f"{SOURCE}/{name}.parquet", df)
    return counts


# --------------------------------------------------------------------------------------------------- panels

def _panel_row(**kw) -> dict:
    from sources.open_close import SCHEMA
    return {c: kw.get(c) for c in SCHEMA}


def _frame(rows: list[dict]) -> pd.DataFrame:
    from sources.open_close import conform, empty
    return conform(pd.DataFrame(rows)) if rows else empty()


def panel_flashscore(store: Store, use_match_odds: bool = True) -> pd.DataFrame:
    """The standard open-close panel from Flashscore: one row a selection of each match's winner market. With the
    per-match odds held, one row per bookmaker as well (``source`` names it); the feed's own figure otherwise."""
    rows = []
    mt = table_flashscore_matches(store)
    for _, m in mt.iterrows():
        if m["status"] != "finished":
            continue
        try:
            hs, as_ = int(m["score_home"]), int(m["score_away"])
        except (TypeError, ValueError):
            hs = as_ = None
        sels = [("home", m["home_open"], m["home_last"], m["home"]),
                ("draw", m["draw_open"], m["draw_last"], "Draw"),
                ("away", m["away_open"], m["away_last"], m["away"])]
        live = [s for s in sels if pd.notna(s[1]) and pd.notna(s[2])]
        if len(live) < 2:
            continue
        for sel, op, cl, name in live:
            res = None
            if hs is not None:
                res = int({"home": hs > as_, "draw": hs == as_, "away": as_ > hs}[sel])
            rows.append(_panel_row(sport=m["sport"], country=m["country"], market_type="MATCH_WINNER",
                                   market_id=f"fs:{m['match_id']}", event_id=m["match_id"],
                                   event_name=f"{m['home']} v {m['away']}", selection_id=sel, selection_name=name,
                                   off_time_utc=m["start_utc"], n_runners=len(live), open_price=op, close_price=cl,
                                   result=res, source="flashscore_feed"))
    df = _frame(rows)
    if use_match_odds:
        od = table_flashscore_odds(store)
        if not od.empty:
            win = od[(od["scope"] == "FULL_TIME") & od["market_type"].isin(["HOME_AWAY", "HOME_DRAW_AWAY"])
                     & od["status"].astype(str).str.lower().eq("finished")]
            rows = []
            for (mid, bk), g in win.groupby(["match_id", "bookmaker"]):
                g = g.dropna(subset=["opening", "last"])
                if len(g) < 2:
                    continue
                hs, as_ = _int(g["home_score"].iloc[0]), _int(g["away_score"].iloc[0])
                for _, s in g.iterrows():
                    sel = s["selection"]
                    res = None
                    if hs is not None and as_ is not None and sel in ("home", "draw", "away"):
                        res = int({"home": hs > as_, "draw": hs == as_, "away": as_ > hs}[sel])
                    rows.append(_panel_row(sport=None, market_type="MATCH_WINNER", market_id=f"fs:{mid}:{bk}",
                                           event_id=mid, event_name=f"{s['home']} v {s['away']}",
                                           selection_id=sel, selection_name=sel, off_time_utc=s["match_date"],
                                           n_runners=len(g), open_price=s["opening"], close_price=s["last"],
                                           result=res, source=f"flashscore:{bk}"))
            per_book = _frame(rows)
            if not per_book.empty:
                # the sport: from the daily feed where the match is in it, else from the results the id came from
                sport = pd.Series(dtype=object)
                res = table_flashscore_results(store)
                if not res.empty:
                    sport = res.set_index("match_id")["sport"]
                if not df.empty:
                    feed = df.drop_duplicates("event_id").set_index("event_id")["sport"]
                    sport = pd.concat([sport[~sport.index.isin(feed.index)], feed])
                per_book["sport"] = per_book["event_id"].map(sport)
            df = pd.concat([df, per_book], ignore_index=True) if not per_book.empty else df
    return df


def _int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def panel_pinnacle(store: Store, market_type: str = "moneyline") -> pd.DataFrame:
    """Pinnacle's full-match moneyline: the open is the first price this record saw (and when), the close the last
    seen before the matchup's start. A record polled every two hours sees the open within two hours of it."""
    t = table_pinnacle(store)
    if t.empty:
        return _frame([])
    t = t[(t["market_type"] == market_type) & (t["period"] == 0) & (~t["is_alternate"].fillna(False).astype(bool))]
    t = t.dropna(subset=["price", "start_utc"])
    t["fetched"] = pd.to_datetime(t["fetched_utc"], utc=True)
    t = t[t["fetched"] < pd.to_datetime(t["start_utc"], utc=True)]
    rows = []
    for (mid, des), g in t.sort_values("fetched").groupby(["matchup_id", "designation"]):
        first, last = g.iloc[0], g.iloc[-1]
        name = first["home"] if des == "home" else first["away"] if des == "away" else des
        rows.append(_panel_row(sport=first["sport"], country=first["league"], market_type="MONEYLINE",
                               market_id=f"pin:{mid}", event_id=str(mid),
                               event_name=f"{first['home']} v {first['away']}", selection_id=des,
                               selection_name=name, off_time_utc=first["start_utc"], open_price=first["price"],
                               open_time_utc=first["fetched"].isoformat(), open_matched=first["max_risk"],
                               close_price=last["price"], pre_play_matched=last["max_risk"],
                               n_runners=g["designation"].nunique(), source="pinnacle"))
    df = _frame(rows)
    if not df.empty:
        df["n_runners"] = df.groupby("market_id")["selection_id"].transform("count")
    return df


# --------------------------------------------------------------------------------------------------- CLI

def _store(root: str | None) -> Store:
    return Store(root=root) if root else Store()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="a local folder in place of S3")
    ap.add_argument("--flashscore-days", type=int, default=0, help="the last N finished days of the daily feed")
    ap.add_argument("--today", action="store_true", help="the daily feed for today as well")
    ap.add_argument("--sports", default="football,tennis", help="Flashscore sports, comma-separated, or all")
    ap.add_argument("--match-odds-from-feed", type=int, default=0,
                    help="per-match odds (1 credit each) for up to N finished matches in the held feed, newest first")
    ap.add_argument("--match-sports", default="tennis", help="sports whose matches get per-match odds")
    ap.add_argument("--tennis-history", type=int, default=0,
                    help="tour-level tennis history: per-match odds for up to N matches (players from the held feed)")
    ap.add_argument("--pages", type=int, default=2, help="pages of results a player (about 40 a page)")
    ap.add_argument("--pinnacle", action="store_true")
    ap.add_argument("--pin-sports", default="33,29,8")
    ap.add_argument("--oddschecker", action="store_true")
    ap.add_argument("--marks", default="", help="minutes before the off, e.g. 120,60,15 (a loop until --until)")
    ap.add_argument("--until", default="21:30")
    ap.add_argument("--no-first", action="store_true", help="with --marks: no first snapshot of every race to run")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args(argv)
    store = _store(a.root)
    p = None
    report: dict = {}
    try:
        if a.flashscore_days or a.today or a.match_odds_from_feed or a.pinnacle or a.oddschecker or a.tennis_history:
            p = Parse()
        if a.flashscore_days or a.today:
            sports = FLASHSCORE_SPORTS if a.sports == "all" else a.sports.split(",")
            offs = ([0] if a.today else []) + [-i for i in range(1, a.flashscore_days + 1)]
            report["flashscore_days"] = flashscore_days(store, p, sports, offs)
        if a.match_odds_from_feed:
            mt = table_flashscore_matches(store)
            want = mt[(mt["status"] == "finished") & mt["sport"].isin(a.match_sports.split(","))
                      & mt["home_open"].notna()]
            # tennis: tour-level singles only (the matches with Betfair money: reports/open_close/all_markets.md)
            want = want[(want["sport"] != "tennis") | want["competition"].fillna("").str.match(TOUR_SINGLES)]
            want = want.sort_values("start_utc", ascending=False)
            report["match_odds"] = flashscore_match_odds(store, p, list(want["match_id"])[: a.match_odds_from_feed])
        if a.tennis_history:
            report["tennis_history"] = tennis_history(store, p, a.pages, a.tennis_history)
        if a.pinnacle:
            report["pinnacle"] = pinnacle_snapshot(store, p, [int(x) for x in a.pin_sports.split(",") if x])
        if a.oddschecker:
            marks = [int(x) for x in a.marks.split(",") if x]
            if marks:
                report["oddschecker"] = oddschecker_marks(store, p, marks, a.until, first=not a.no_first)
            else:
                races = [r for r in oddschecker_races(store, p) if r["off_utc"] > _now()]
                report["oddschecker"] = {"races": oddschecker_snapshot(store, p, races, "snap")}
    except OutOfCredits as exc:
        report["stopped"] = str(exc)
    if a.build:
        report["tables"] = build(store)
    if p is not None:
        report["credits_spent"] = p.spent
        report["credits_remaining"] = p.remaining
    print(json.dumps(report, default=str, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
