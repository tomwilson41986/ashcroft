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
INCLUDE = "participants;scores;xGFixture.type;statistics.type;state"
#: our competition id -> Sportmonks league id (read from /v3/football/leagues on 24 Nov 2026; the Scottish lower
#: divisions are not in the plan)
LEAGUES = {"ENG1": 8, "ENG2": 9, "ENG3": 12, "ENG4": 14, "ENG5": 17, "SCO1": 501, "GER1": 82, "GER2": 85,
           "ITA1": 384, "ITA2": 387, "ESP1": 564, "ESP2": 567, "FRA1": 301, "FRA2": 304, "NED1": 72, "BEL1": 208,
           "POR1": 462, "TUR1": 600, "GRE1": 325, "ARG1": 636, "AUT1": 181, "BRA1": 648, "CHN1": 989, "DEN1": 271,
           "FIN1": 292, "IRL1": 360, "JPN1": 968, "MEX1": 743, "NOR1": 444, "POL1": 453, "ROU1": 474, "RUS1": 486,
           "SWE1": 573, "SUI1": 591, "USA1": 779}
BY_LEAGUE = {v: k for k, v in LEAGUES.items()}


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
    held = {k.split("/")[-2] for k in store.listing(f"{PREFIX}/raw/")}
    leagues = ",".join(str(v) for v in LEAGUES.values())
    rep = {"windows": 0, "skipped_held": 0, "pages": 0, "fixtures": 0}
    for a, b in reversed(windows(first_day or FIRST_DAY, today - timedelta(days=1))):
        tag = f"{a.isoformat()}_{b.isoformat()}"
        if tag in held and (today - b).days > REFRESH_DAYS:
            rep["skipped_held"] += 1
            continue
        if not budget.left():
            rep["stopped"] = "time budget"
            break
        page, more = 1, True
        while more:
            r = get(s, f"{BASE}/fixtures/between/{a.isoformat()}/{b.isoformat()}",
                    params={"include": INCLUDE, "filters": f"fixtureLeagues:{leagues}", "per_page": PER_PAGE,
                            "page": page})
            body = r.json() if r is not None else {}
            store.put(f"{PREFIX}/raw/{tag}/p{page:03d}.json.gz", gz(json.dumps(body).encode()))
            rep["pages"] += 1
            rep["fixtures"] += len(body.get("data") or [])
            more = bool((body.get("pagination") or {}).get("has_more"))
            page += 1
        rep["windows"] += 1
    return rep


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
        out.append(r)
    return out


def build(store: Store) -> dict:
    import pandas as pd
    all_rows = []
    for k in sorted(store.listing(f"{PREFIX}/raw/")):
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
    return rep
