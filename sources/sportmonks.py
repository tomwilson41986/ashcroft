"""Sportmonks: each finished match's xG measures and match statistics for the leagues our football models cover.

racing² football data, 24 Nov 2026 (the owner subscribed: Pro, the Pressure Index & xG bundle; trial to 8 Dec 2026).
What the plan returns was read from the API on 24 Nov: 13 xG measures a side (xG, xGA, xGoT, npxG, open-play, set-play,
corner, free-kick and penalty xG, xG prevented, xPTS, shooting performance) and 86 statistic types (big chances, key
passes, dangerous attacks, shots inside the box ...). xG goes back to 2024/25 (a Championship weekend of Nov 2024
answered; a Premier League weekend of Nov 2023 did not).

- ``fetch``: the fixtures of the mapped leagues (``LEAGUES``: our competition id -> Sportmonks league id) by windows of
  ``WINDOW_DAYS`` from ``first_day``, raw, to ``sportmonks/raw/<from>_<to>/p<page>.json.gz`` (under the store's ``sources/``). A window already
  held is not fetched again, except those ending in the last ``REFRESH_DAYS`` (results and xG still arriving).
- ``build``: ``sportmonks/matches.parquet``, one row a finished match: league, kick-off, both sides' names,
  the score, ``xg_<code>_h`` / ``_a`` for each xG measure and ``st_<code>_h`` / ``_a`` for each statistic.

The token is read from ``SPORTMONKS_API_TOKEN`` (a repository secret) and sent in the Authorization header (Sportmonks
refuses a 'Bearer' prefix). Never logged, never written.

    python source_data.py --source sportmonks --fetch --build --from 2024-07-01
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, timedelta

from sources.common import Budget, Store, env, get, gunzip, gz, session

log = logging.getLogger(__name__)

PREFIX = "sportmonks"
BASE = "https://api.sportmonks.com/v3/football"
FIRST_DAY = date(2024, 7, 1)
WINDOW_DAYS = 7
REFRESH_DAYS = 10
PER_PAGE = 50
INCLUDE = "participants;scores;xGFixture.type;statistics.type;state;predictions.type;odds"
#: the raw files' layout version: v2 (24 Nov 2026) adds Sportmonks' predictions and the 1X2 bookmaker odds (market 1),
#: so every window is fetched once more under raw/v2/
RAW = "raw/v2"
#: our competition id -> Sportmonks league id (read from /v3/football/leagues on 24 Nov 2026; the Scottish lower
#: divisions are not in the plan)
LEAGUES = {"ENG1": 8, "ENG2": 9, "ENG3": 12, "ENG4": 14, "ENG5": 17, "SCO1": 501, "GER1": 82, "GER2": 85,
           "ITA1": 384, "ITA2": 387, "ESP1": 564, "ESP2": 567, "FRA1": 301, "FRA2": 304, "NED1": 72, "BEL1": 208,
           "POR1": 462, "TUR1": 600, "GRE1": 325, "ARG1": 636, "AUT1": 181, "BRA1": 648, "CHN1": 989, "DEN1": 271,
           "FIN1": 292, "IRL1": 360, "JPN1": 968, "MEX1": 743, "NOR1": 444, "POL1": 453, "ROU1": 474, "RUS1": 486,
           "SWE1": 573, "SUI1": 591, "USA1": 779}
#: the owner, 24 Nov 2026 ("use these in our models and prices"): the plan's Euro Club Tournaments and International
#: Tournaments. Not canonical competitions of ours (football/reference.py), so they are kept under these ids and not
#: linked to the domestic models' matches
EUROPE_INTERNATIONAL = {"UCL": 2, "UEL": 5, "UECL": 2286, "USC": 1328, "UNL": 1538, "EURO": 1326, "EUROQ": 1325,
                        "WC": 732, "WCQ_EU": 720, "WCQ_SA": 726, "COPA": 1114, "AFCON": 1117, "ASIAN": 1105,
                        "GOLD": 1112, "LIBERTADORES": 1122, "SUDAMERICANA": 1116}
ALL_LEAGUES = {**LEAGUES, **EUROPE_INTERNATIONAL}
BY_LEAGUE = {v: k for k, v in ALL_LEAGUES.items()}
#: Sportmonks answers "Error parsing filters" to a fixtureLeagues filter of 51 ids (35 pass; 24 Nov 2026): the leagues
#: go in groups of at most this many a request
LEAGUE_GROUP = 30


def league_groups() -> list[str]:
    ids = [str(v) for v in ALL_LEAGUES.values()]
    return [",".join(ids[i:i + LEAGUE_GROUP]) for i in range(0, len(ids), LEAGUE_GROUP)]


def code(name: str) -> str:
    """'Expected Goals on Target (xGoT)' -> 'xgot'; 'Shots Insidebox' -> 'shots_insidebox'."""
    m = re.search(r"\(([^)]+)\)\s*$", name or "")
    s = m.group(1) if m else (name or "")
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def windows(first: date, last: date, width: int = WINDOW_DAYS) -> list[tuple[date, date]]:
    out, d = [], first
    while d <= last:
        e = min(d + timedelta(days=width - 1), last)
        out.append((d, e))
        d = e + timedelta(days=1)
    return out


def fetch(store: Store, max_minutes: float | None = None, first_day: date | None = None, today: date | None = None,
          token: str | None = None, http=None) -> dict:
    token = token or env("SPORTMONKS_API_TOKEN")
    if not token:
        return {"skipped": "SPORTMONKS_API_TOKEN is not set"}
    today = today or date.today()
    budget = Budget(max_minutes)
    s = http or session()
    s.headers.update({"Authorization": token, "Accept": "application/json"})
    held = {k.split("/")[-2] for k in store.listing(f"{PREFIX}/{RAW}/")}
    rep = {"windows": 0, "skipped_held": 0, "pages": 0, "fixtures": 0}
    for a, b in reversed(windows(first_day or FIRST_DAY, today - timedelta(days=1))):
        tag = f"{a.isoformat()}_{b.isoformat()}"
        if tag in held and (today - b).days > REFRESH_DAYS:
            rep["skipped_held"] += 1
            continue
        if not budget.left():
            rep["stopped"] = "time budget"
            break
        for gi, leagues in enumerate(league_groups()):
            page, more = 1, True
            while more:
                r = get(s, f"{BASE}/fixtures/between/{a.isoformat()}/{b.isoformat()}",
                        params={"include": INCLUDE, "filters": f"fixtureLeagues:{leagues};markets:1",
                                "per_page": PER_PAGE, "page": page})
                body = r.json() if r is not None else {}
                store.put(f"{PREFIX}/{RAW}/{tag}/g{gi}_p{page:03d}.json.gz", gz(json.dumps(body).encode()))
                rep["pages"] += 1
                rep["fixtures"] += len(body.get("data") or [])
                more = bool((body.get("pagination") or {}).get("has_more"))
                page += 1
        rep["windows"] += 1
    if budget.left():
        rep["snapshot"] = snapshot(store, s, today=today)
    return rep


SNAPSHOT_DAYS = 3


def snapshot(store: Store, s, today: date | None = None, now=None) -> dict:
    """The fixtures of the next ``SNAPSHOT_DAYS`` days with Sportmonks' predictions and the 1X2 odds as they stand now,
    stamped with the time they were read (``sportmonks/snapshots/<day>/<HHMM>_p<page>.json.gz``): the clean record of
    what the predictions said before a match, which the history cannot give. Run with each nightly fetch."""
    from datetime import datetime, timezone
    now = now or datetime.now(timezone.utc)
    today = today or now.date()
    end = today + timedelta(days=SNAPSHOT_DAYS)
    n = pages = 0
    stamp = now.strftime("%Y-%m-%dT%H%MZ")
    for gi, leagues in enumerate(league_groups()):
        page, more = 1, True
        while more:
            r = get(s, f"{BASE}/fixtures/between/{today.isoformat()}/{end.isoformat()}",
                    params={"include": "participants;predictions.type;odds",
                            "filters": f"fixtureLeagues:{leagues};markets:1", "per_page": PER_PAGE, "page": page})
            body = r.json() if r is not None else {}
            body["fetched_utc"] = now.isoformat(timespec="seconds")
            store.put(f"{PREFIX}/snapshots/{today.isoformat()}/{stamp}_g{gi}_p{page:03d}.json.gz",
                      gz(json.dumps(body).encode()))
            n += len(body.get("data") or [])
            more = bool((body.get("pagination") or {}).get("has_more"))
            page, pages = page + 1, pages + 1
    return {"fixtures": n, "pages": pages, "read_utc": now.isoformat(timespec="seconds")}


def snapshot_rows(body: dict) -> list[dict]:
    """One row a fixture of a snapshot page: who, when, and the predictions and odds as read at ``fetched_utc``."""
    import pandas as pd
    out = []
    for f in body.get("data") or []:
        side = {(p.get("meta") or {}).get("location"): p for p in f.get("participants") or []}
        if set(side) < {"home", "away"}:
            continue
        r = {"sm_fixture_id": f.get("id"), "sm_league_id": f.get("league_id"),
             "competition_id": BY_LEAGUE.get(f.get("league_id")), "fetched_utc": body.get("fetched_utc"),
             "kickoff_utc": pd.Timestamp(f["starting_at_timestamp"], unit="s", tz="UTC")
             if f.get("starting_at_timestamp") else pd.NaT,
             "home": side["home"].get("name"), "away": side["away"].get("name")}
        r.update(predictions(f.get("predictions") or []))
        r.update(odds_1x2(f.get("odds") or []))
        out.append(r)
    return out


def rows(body: dict) -> list[dict]:
    """One row a finished fixture from one page of the fixtures endpoint."""
    import pandas as pd
    out = []
    for f in body.get("data") or []:
        if (f.get("state") or {}).get("developer_name") not in ("FT", "AET", "FT_PEN"):
            continue
        side = {}
        for p in f.get("participants") or []:
            loc = (p.get("meta") or {}).get("location")
            if loc in ("home", "away"):
                side[loc] = p
        if set(side) != {"home", "away"}:
            continue
        r = {"sm_fixture_id": f.get("id"), "sm_league_id": f.get("league_id"),
             "competition_id": BY_LEAGUE.get(f.get("league_id")), "sm_season_id": f.get("season_id"),
             "kickoff_utc": pd.Timestamp(f["starting_at_timestamp"], unit="s", tz="UTC")
             if f.get("starting_at_timestamp") else pd.NaT,
             "home": side["home"].get("name"), "away": side["away"].get("name"),
             "sm_home_id": side["home"].get("id"), "sm_away_id": side["away"].get("id")}
        for sc in f.get("scores") or []:
            if sc.get("description") == "CURRENT":
                who = (sc.get("score") or {}).get("participant")
                if who in ("home", "away"):
                    r[f"ft_{who}"] = (sc.get("score") or {}).get("goals")
        ids = {side["home"].get("id"): "h", side["away"].get("id"): "a"}
        for key, pre in (("xgfixture", "xg"), ("statistics", "st")):
            for x in f.get(key) or []:
                tag = ids.get(x.get("participant_id"))
                if tag is None and x.get("location") in ("home", "away"):
                    tag = x["location"][0]
                v = (x.get("data") or {}).get("value")
                name = (x.get("type") or {}).get("name")
                if tag and name and isinstance(v, (int, float)):
                    r[f"{pre}_{code(name)}_{tag}"] = float(v)
        r.update(predictions(f.get("predictions") or []))
        r.update(odds_1x2(f.get("odds") or []))
        out.append(r)
    return out


#: Sportmonks' prediction types kept, by type id (read from the API on 24 Nov 2026): full-time result, over/under 2.5,
#: both teams to score. Sportmonks gives no time for a prediction: the historical value may have been revised up to
#: the kick-off, so a test on it is read with that caveat (the snapshots below carry their own time)
PREDICTION_TYPES = {"Fulltime Result Probability": "ft", "Over/Under 2.5 Probability": "o25",
                    "Both Teams To Score Probability": "btts"}


def predictions(items: list) -> dict:
    out = {}
    for x in items:
        tag = PREDICTION_TYPES.get((x.get("type") or {}).get("name"))
        v = x.get("predictions") or {}
        if tag == "ft":
            out.update({"pr_ft_h": v.get("home"), "pr_ft_d": v.get("draw"), "pr_ft_a": v.get("away")})
        elif tag in ("o25", "btts"):
            out[f"pr_{tag}"] = v.get("yes")
    return {k: (float(v) / 100.0 if isinstance(v, (int, float)) else None) for k, v in out.items()}


def odds_1x2(items: list) -> dict:
    """The bookmakers' 1X2 prices (market 1) as one consensus: the mean margin-free chance (each book's margin split in
    proportion), the best price, the number of books, and the time of the latest update among them. Sportmonks keeps
    the latest price only, so for a finished match this is the market near the kick-off: a closing benchmark, not an
    early price."""
    books: dict = {}
    last = None
    for o in items:
        if o.get("market_id") != 1:
            continue
        lab = str(o.get("label") or "").lower()
        side = {"home": "h", "1": "h", "draw": "d", "x": "d", "away": "a", "2": "a"}.get(lab)
        try:
            price = float(o.get("value"))
        except (TypeError, ValueError):
            continue
        if side is None or price <= 1.0:
            continue
        books.setdefault(o.get("bookmaker_id"), {})[side] = price
        t = o.get("latest_bookmaker_update")
        if t and (last is None or t > last):
            last = t
    full = [b for b in books.values() if set(b) == {"h", "d", "a"}]
    if not full:
        return {}
    probs = []
    for b in full:
        inv = {k: 1.0 / v for k, v in b.items()}
        tot = sum(inv.values())
        probs.append({k: v / tot for k, v in inv.items()})
    out = {f"od_q_{k}": sum(p[k] for p in probs) / len(probs) for k in ("h", "d", "a")}
    out.update({f"od_max_{k}": max(b[k] for b in full) for k in ("h", "d", "a")})
    out.update({"od_books": len(full), "od_last_update": last})
    return out


def build(store: Store) -> dict:
    import pandas as pd
    all_rows = []
    for k in sorted(store.listing(f"{PREFIX}/{RAW}/")):
        raw = store.get(k)
        if raw:
            all_rows += rows(json.loads(gunzip(raw)))
    m = pd.DataFrame(all_rows)
    rep = {"rows_in": int(len(m))}
    if m.empty:
        return rep
    m = m.sort_values("kickoff_utc").drop_duplicates("sm_fixture_id", keep="last")
    rep.update({"matches": int(len(m)), "duplicates_dropped": rep["rows_in"] - int(len(m)),
                "kickoff_range": [str(m["kickoff_utc"].min()), str(m["kickoff_utc"].max())],
                "with_xg": int(m["xg_xg_h"].notna().sum()) if "xg_xg_h" in m else 0,
                "null_team_names": int(m[["home", "away"]].isna().any(axis=1).sum()),
                "by_competition": m.groupby("competition_id").size().to_dict()})
    if "xg_xg_h" in m:
        rep["xg_share_by_competition"] = m.groupby("competition_id")["xg_xg_h"].apply(
            lambda s: round(float(s.notna().mean()), 3)).to_dict()
    store.put_parquet(f"{PREFIX}/matches.parquet", m)
    if "pr_ft_h" in m:
        rep["with_predictions"] = int(m["pr_ft_h"].notna().sum())
    if "od_books" in m:
        rep["with_odds"] = int(m["od_books"].notna().sum())
    snaps = []
    for k in sorted(store.listing(f"{PREFIX}/snapshots/")):
        raw = store.get(k)
        if raw:
            snaps += snapshot_rows(json.loads(gunzip(raw)))
    if snaps:
        sn = pd.DataFrame(snaps).drop_duplicates(["sm_fixture_id", "fetched_utc"])
        store.put_parquet(f"{PREFIX}/snapshots.parquet", sn)
        rep["snapshot_rows"] = int(len(sn))
    return rep
