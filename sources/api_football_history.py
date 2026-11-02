"""API-Football's finished matches of the 22 divisions from a season on: line-ups, events and player statistics.

The history behind player-level strength (reports/football_quant_approach.md s.6.2 F7: minutes-weighted player
ratings, plus-minus): who started, who came on and when, the goals and cards with their minutes, and, where the plan
covers the season, each player's minutes and rating. Coverage checked on 2 Nov 2026 (``/leagues``): line-ups and
events from 2010-2016 in all 22 divisions, player statistics from about 2015 in the top divisions, none in the
National League and Scotland's lower three.

Fetch (resumable, within a time budget and the plan's daily quota): for each division and season from
``first_season`` (newest first), the season's fixture list (``/fixtures?league=&season=``, fetched again while the
season is current), then the finished matches not yet held, 20 a request (``/fixtures?ids=``, which carries the
line-ups, events, statistics and players). Raw replies as they came, wrapped with the UTC fetch time:
``api_football/raw/season_fixtures/<league>/<season>/<stamp>.json.gz`` and
``api_football/raw/history/<season>/<league>/<stamp>_<n>.json.gz``; the ids held in
``api_football/history/done_ids.json`` (rebuilt from the raw files when missing).

Build (rebuilt whole from the raw files):

- ``api_football/history_matches.parquet``: one row a match (league, season, kick-off, the sides, the score,
  whether line-ups, events and player statistics came);
- ``api_football/history_players.parquet``: one row a player a match (side, starter or substitute, position,
  minutes: the plan's where it gives them, else from the line-up and the substitution and red-card events);
- ``api_football/history_events.parquet``: one row an event (goal, card, substitution, VAR) with its minute.

    python source_data.py --source api_football_history --fetch --build --from 2020
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone

from sources.api_football_lineups import BATCH, PREFIX, http_fetch, main_league_ids
from sources.common import Budget, Store, env, gunzip, gz

log = logging.getLogger(__name__)

FIRST_SEASON = 2020
RATE_TRIES, RATE_WAIT_S = 5, 61.0   # API-Football's per-minute limit (Pro: 300 a minute, its reply of 2 Nov 2026)
QUOTA_FLOOR = 1500          # the same reserve as the nightly fixtures backfill: the day's line-up polls need it
FINISHED = {"FT", "AET", "PEN"}
DONE_KEY = f"{PREFIX}/history/done_ids.json"


def current_season(today: date) -> int:
    """API-Football's season of the 22 divisions: the year it starts (July on)."""
    return today.year if today.month >= 7 else today.year - 1


def _wrap(path: str, params: dict, body: dict) -> bytes:
    return gz(json.dumps({"fetched_utc": datetime.now(timezone.utc).isoformat(), "path": path, "params": params,
                          "body": body}).encode())


def _read(store: Store, key: str) -> dict | None:
    try:
        return json.loads(gunzip(store.get(key)))
    except (ValueError, TypeError):
        log.warning("api_football_history: %s unreadable", key)
        return None


def done_ids(store: Store) -> set[int]:
    raw = store.get(DONE_KEY)
    if raw:
        return set(json.loads(raw))
    out: set[int] = set()
    for k in store.listing(f"{PREFIX}/raw/history/"):
        doc = _read(store, k) if k.endswith(".json.gz") else None
        for f in ((doc or {}).get("body") or {}).get("response") or []:
            fid = (f.get("fixture") or {}).get("id")
            if fid is not None:
                out.add(int(fid))
    return out


def fetch(store: Store, max_minutes: float | None = None, first_season: int = FIRST_SEASON, fetcher=None,
          today: date | None = None, quota_floor: int = QUOTA_FLOOR, leagues: set[int] | None = None,
          sleep=None) -> dict:
    out = {"season_lists": 0, "requests": 0, "matches_fetched": 0, "already_held": 0, "stopped": None,
           "quota_left": None}
    if fetcher is None:
        key = env("API_FOOTBALL_KEY")
        if not key:
            out["error"] = "no API_FOOTBALL_KEY; nothing fetched"
            log.warning("api_football_history: %s", out["error"])
            return out
        fetcher = http_fetch(key)
    import time
    sleep = sleep or time.sleep
    today = today or date.today()
    cur = current_season(today)
    leagues = sorted(leagues or main_league_ids())
    budget = Budget(max_minutes)
    done = done_ids(store)
    out["already_held"] = len(done)
    n = 0

    def call(path, params):
        nonlocal n
        if out["quota_left"] is not None and out["quota_left"] <= quota_floor:
            out["stopped"] = f"quota at {out['quota_left']}"
            return None
        if not budget.left():
            out["stopped"] = "time budget"
            return None
        for attempt in range(RATE_TRIES):
            body, left = fetcher(path, params)
            out["requests"] += 1
            n += 1
            if left is not None:
                out["quota_left"] = left
            errs = (body or {}).get("errors")
            if errs and "rateLimit" in str(errs) and attempt < RATE_TRIES - 1:
                out["rate_limited"] = out.get("rate_limited", 0) + 1     # the plan's per-minute limit: wait, again
                sleep(RATE_WAIT_S)
                continue
            break
        if body is not None and body.get("errors"):
            out["stopped"] = f"API error: {str(body['errors'])[:200]}"
            return None
        return body

    for season in range(cur, first_season - 1, -1):
        for lg in leagues:
            if out["stopped"]:
                break
            prefix = f"{PREFIX}/raw/season_fixtures/{lg}/{season}/"
            held = sorted(k for k in store.listing(prefix) if k.endswith(".json.gz"))
            if held and season < cur:
                listing = (_read(store, held[-1]) or {}).get("body") or {}
            else:
                params = {"league": lg, "season": season}
                listing = call("fixtures", params)
                if listing is None:
                    break
                store.put(f"{prefix}{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json.gz",
                          _wrap("fixtures", params, listing))
                out["season_lists"] += 1
            todo = [f["fixture"]["id"] for f in listing.get("response") or []
                    if (f.get("fixture") or {}).get("status", {}).get("short") in FINISHED
                    and f["fixture"]["id"] not in done]
            for i in range(0, len(todo), BATCH):
                chunk = todo[i:i + BATCH]
                params = {"ids": "-".join(str(x) for x in chunk)}
                body = call("fixtures", params)
                if body is None:
                    break
                store.put(f"{PREFIX}/raw/history/{season}/{lg}/{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}Z_{n}"
                          f".json.gz", _wrap("fixtures", params, body))
                got = {(f.get("fixture") or {}).get("id") for f in body.get("response") or []}
                done |= {x for x in chunk if x in got}
                out["matches_fetched"] += len(got & set(chunk))
                if out["requests"] % 10 == 0:
                    store.put(DONE_KEY, json.dumps(sorted(done)).encode())
        if out["stopped"]:
            break
    store.put(DONE_KEY, json.dumps(sorted(done)).encode())
    out["held_now"] = len(done)
    log.info("api_football_history: %s", out)
    return out


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _minute(e: dict) -> int | None:
    t = e.get("time") or {}
    return None if t.get("elapsed") is None else int(t["elapsed"]) + int(t.get("extra") or 0)


def minutes_from_events(lineups: list, events: list, full: int = 90) -> tuple[dict, int]:
    """Each player's minutes from the starting elevens and the substitution and red-card events, over ``full``
    minutes (extra time is not counted). Who went off is read from who was on the pitch, not from the event's field
    order (API-Football's ``player`` and ``assist`` have not always meant the same). Returns (minutes by player id,
    substitutions it could not resolve)."""
    on: dict[int, int] = {}                     # player -> minute he came on
    mins: dict[int, int] = {}
    for team in lineups or []:
        for p in team.get("startXI") or []:
            pid = (p.get("player") or {}).get("id")
            if pid is not None:
                on[pid] = 0
    unresolved = 0
    for e in sorted(events or [], key=lambda x: _minute(x) or 0):
        m = min(_minute(e) or 0, full)
        a, b = (e.get("player") or {}).get("id"), (e.get("assist") or {}).get("id")
        if e.get("type") == "subst":
            if a in on and b not in on:
                off, new = a, b
            elif b in on and a not in on:
                off, new = b, a
            else:
                unresolved += 1
                continue
            mins[off] = mins.get(off, 0) + m - on.pop(off)
            if new is not None:
                on[new] = m
        elif e.get("type") == "Card" and any(w in str(e.get("detail", "")).lower() for w in ("red", "second yellow")):
            if a in on:
                mins[a] = mins.get(a, 0) + m - on.pop(a)
    for pid, start in on.items():
        mins[pid] = mins.get(pid, 0) + full - start
    return mins, unresolved


def _num(x) -> float | None:
    """A player statistic as a number; None for API-Football's blanks and placeholders ('', '-', '\u2013')."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def build(store: Store) -> dict:
    import pandas as pd
    matches, players, events = [], [], []
    unresolved = 0
    for k in sorted(store.listing(f"{PREFIX}/raw/history/")):
        if not k.endswith(".json.gz"):
            continue
        doc = _read(store, k)
        for f in ((doc or {}).get("body") or {}).get("response") or []:
            fx, lg, tm = f.get("fixture") or {}, f.get("league") or {}, f.get("teams") or {}
            fid = fx.get("id")
            sc = (f.get("score") or {}).get("fulltime") or {}
            home_id = (tm.get("home") or {}).get("id")
            lu, ev, pl = f.get("lineups") or [], f.get("events") or [], f.get("players") or []
            matches.append({"apif_fixture_id": fid, "apif_league_id": lg.get("id"), "apif_season": lg.get("season"),
                            "kickoff_utc": pd.Timestamp(fx["timestamp"], unit="s", tz="UTC") if fx.get("timestamp")
                            else pd.NaT, "status_short": (fx.get("status") or {}).get("short"),
                            "apif_home_id": home_id, "home": (tm.get("home") or {}).get("name"),
                            "apif_away_id": (tm.get("away") or {}).get("id"), "away": (tm.get("away") or {}).get("name"),
                            "ft_home": sc.get("home"), "ft_away": sc.get("away"),
                            "has_lineups": len(lu) == 2, "has_events": bool(ev), "has_player_stats": bool(pl),
                            "fetched_utc": doc.get("fetched_utc")})
            stat_min = {}
            for t in pl:
                for p in t.get("players") or []:
                    g = ((p.get("statistics") or [{}])[0] or {}).get("games") or {}
                    stat_min[(p.get("player") or {}).get("id")] = (g.get("minutes"), g.get("rating"))
            ev_min, u = minutes_from_events(lu, ev)
            unresolved += u
            for j, t in enumerate(lu):
                tid = (t.get("team") or {}).get("id")
                side = "home" if tid == home_id else ("away" if tid is not None else ("home" if j == 0 else "away"))
                for role, key in (("starter", "startXI"), ("substitute", "substitutes")):
                    for p in t.get(key) or []:
                        q = p.get("player") or {}
                        pid = q.get("id")
                        sm, rating = stat_min.get(pid, (None, None))
                        players.append({"apif_fixture_id": fid, "apif_league_id": lg.get("id"),
                                        "apif_season": lg.get("season"), "side": side, "apif_team_id": tid,
                                        "role": role, "apif_player_id": pid, "player": q.get("name"),
                                        "pos": q.get("pos"), "minutes_stats": sm,
                                        "minutes_events": ev_min.get(pid, 0) if pid is not None else None,
                                        "rating": _num(rating)})
            for e in ev:
                events.append({"apif_fixture_id": fid, "minute": _minute(e), "type": e.get("type"),
                               "detail": e.get("detail"), "apif_team_id": (e.get("team") or {}).get("id"),
                               "apif_player_id": (e.get("player") or {}).get("id"),
                               "apif_assist_id": (e.get("assist") or {}).get("id")})
    m, p, e = pd.DataFrame(matches), pd.DataFrame(players), pd.DataFrame(events)
    rep = {"matches_in": int(len(m))}
    if len(m):
        dup = m.duplicated("apif_fixture_id", keep="last")
        rep["duplicate_matches_dropped"] = int(dup.sum())
        m = m[~dup]
        store.put_parquet(f"{PREFIX}/history_matches.parquet", m)
        rep.update({"matches": int(len(m)), "with_lineups": int(m["has_lineups"].sum()),
                    "with_events": int(m["has_events"].sum()), "with_player_stats": int(m["has_player_stats"].sum()),
                    "kickoff_range": [str(m["kickoff_utc"].min()), str(m["kickoff_utc"].max())],
                    "by_season": m.groupby("apif_season").size().to_dict()})
    if len(p):
        p = p.drop_duplicates(["apif_fixture_id", "side", "role", "apif_player_id", "player"], keep="last")
        store.put_parquet(f"{PREFIX}/history_players.parquet", p)
        both = p.dropna(subset=["minutes_stats"])
        rep.update({"player_rows": int(len(p)), "player_rows_without_id": int(p["apif_player_id"].isna().sum()),
                    "duplicate_player_ids": int(p.dropna(subset=["apif_player_id"]).duplicated(
                        ["apif_fixture_id", "apif_player_id"]).sum()),
                    "minutes_events_vs_stats_mean_abs_diff": float((both["minutes_events"] - both["minutes_stats"])
                                                                   .abs().mean()) if len(both) else None,
                    "substitutions_unresolved": unresolved})
    if len(e):
        store.put_parquet(f"{PREFIX}/history_events.parquet", e)
        rep["event_rows"] = int(len(e))
    log.info("api_football_history build: %s", rep)
    return rep
