"""API-Football (v3.football.api-sports.io): every fixture a day, in every competition it covers, with its status,
kick-off, venue, referee and score. The source of truth for a match's status (postponed, abandoned, awarded, which
football-data.co.uk never says) and the breadth football-data lacks (cups, UEFA, internationals); keyed by the
secret ``API_FOOTBALL_KEY`` (the Pro plan, US$19 a month, 7,500 requests a day; the free plan's 100 a day serves
the daily window only). Without a key it fetches nothing and says so.

Each run: the window D-3 to D+14 again (late corrections, kick-off changes), then the days the store lacks, newest
first, back to ``first_day``, while the plan's remaining daily quota (the reply's header) stays above ``QUOTA_FLOOR``.

Raw: ``api_football/raw/fixtures/<day>/<ingest stamp>.json.gz`` (bronze, append-only: a day fetched again is a new
file). Table: ``api_football/fixtures.parquet``, the newest reading of each day, one row a fixture.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone

from sources.common import Budget, Store, env, get, gunzip, gz, session

log = logging.getLogger(__name__)

API = "https://v3.football.api-sports.io"
FIRST_DAY = date(2016, 1, 1)
QUOTA_FLOOR = 300
STATUS = {"FT": "FINISHED", "AET": "FINISHED", "PEN": "FINISHED", "NS": "SCHEDULED", "TBD": "SCHEDULED",
          "PST": "POSTPONED", "CANC": "CANCELLED", "ABD": "ABANDONED", "AWD": "AWARDED", "WO": "AWARDED",
          "SUSP": "SUSPENDED", "INT": "SUSPENDED"}


def _day_key(d: date, stamp: str) -> str:
    return f"api_football/raw/fixtures/{d.isoformat()}/{stamp}.json.gz"


def fetch(store: Store, first_day: date = FIRST_DAY, max_minutes: float | None = None, s=None,
          today: date | None = None, key: str | None = None) -> dict:
    key = key or env("API_FOOTBALL_KEY")
    out = {"fetched": 0, "skipped": 0, "stopped": False, "quota_left": None}
    if not key:
        log.warning("api_football: no API_FOOTBALL_KEY; nothing fetched (an API-Football plan is the owner's to buy)")
        out["error"] = "no API_FOOTBALL_KEY"
        return out
    s = s or session()
    s.headers.update({"x-apisports-key": key})
    t = today or date.today()
    have = {k.split("/")[3] for k in store.listing("api_football/raw/fixtures/")}
    window = [t + timedelta(days=i) for i in range(14, -4, -1)]
    back = [t - timedelta(days=i) for i in range(4, (t - first_day).days + 1)]
    days = window + [d for d in back if d.isoformat() not in have]
    budget = Budget(max_minutes)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for d in days:
        if not budget.left() or (out["quota_left"] is not None and out["quota_left"] < QUOTA_FLOOR):
            out["stopped"] = True
            break
        r = get(s, f"{API}/fixtures", params={"date": d.isoformat()}, pause=0.25)
        if r is None:
            continue
        left = r.headers.get("x-ratelimit-requests-remaining")
        out["quota_left"] = int(left) if left and str(left).isdigit() else out["quota_left"]
        body = r.json() if hasattr(r, "json") else json.loads(r.content)
        if body.get("errors"):
            out["error"] = str(body["errors"])[:300]
            log.warning("api_football: %s", out["error"])
            break
        store.put(_day_key(d, stamp), gz(r.content))
        out["fetched"] += 1
    log.info("api_football: %s", out)
    return out


def rows(doc: dict) -> list[dict]:
    out = []
    for f in doc.get("response", []) or []:
        fx, lg, tm = f.get("fixture") or {}, f.get("league") or {}, f.get("teams") or {}
        gl, sc = f.get("goals") or {}, f.get("score") or {}
        st = (fx.get("status") or {}).get("short")
        ht = sc.get("halftime") or {}
        ft = sc.get("fulltime") or {}
        out.append({
            "apif_fixture_id": fx.get("id"), "kickoff_utc": fx.get("date"), "status_short": st,
            "status": STATUS.get(st, "IN_PLAY" if st else None), "referee": fx.get("referee"),
            "venue": (fx.get("venue") or {}).get("name"), "apif_league_id": lg.get("id"),
            "league": lg.get("name"), "league_country": lg.get("country"), "apif_season": lg.get("season"),
            "round": lg.get("round"), "home": (tm.get("home") or {}).get("name"),
            "away": (tm.get("away") or {}).get("name"), "apif_home_id": (tm.get("home") or {}).get("id"),
            "apif_away_id": (tm.get("away") or {}).get("id"),
            # the score after 90 minutes where the match went to extra time, as football-data records it
            "ft_home": ft.get("home") if ft.get("home") is not None else gl.get("home"),
            "ft_away": ft.get("away") if ft.get("away") is not None else gl.get("away"),
            "ht_home": ht.get("home"), "ht_away": ht.get("away")})
    return out


def build(store: Store) -> dict:
    import pandas as pd

    from football.reference import COMPETITIONS
    newest: dict[str, str] = {}
    for k in store.listing("api_football/raw/fixtures/"):
        day = k.split("/")[3]
        if k > newest.get(day, ""):
            newest[day] = k
    recs = []
    for day in sorted(newest):
        try:
            recs += rows(json.loads(gunzip(store.get(newest[day]))))
        except ValueError:
            continue
    if not recs:
        return {"fixtures": 0}
    df = pd.DataFrame(recs).drop_duplicates("apif_fixture_id", keep="last")
    comp = {c.apif_league_id: c.competition_id for c in COMPETITIONS if c.apif_league_id}
    df["competition_id"] = df.apif_league_id.map(comp)
    ko = pd.to_datetime(df.kickoff_utc, utc=True, errors="coerce")
    df["kickoff_utc"] = ko
    df["match_date"] = ko.dt.tz_convert("Europe/London").dt.strftime("%Y-%m-%d")
    store.put_parquet("api_football/fixtures.parquet", df)
    out = {"fixtures": len(df), "in_scope": int(df.competition_id.notna().sum()), "days": len(newest),
           "status": df.status.value_counts().to_dict()}
    log.info("api_football: %s", out)
    return out
