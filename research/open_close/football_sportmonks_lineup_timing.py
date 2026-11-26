"""When does Sportmonks hold both starting elevens, against API-Football's ~28 minutes before kick-off?

racing² football research, 26 Nov 2026 (the owner: "run the Sportmonks line-up timing check"). The news-timing study
(``football_news_timing``) found API-Football holds both elevens about 28 minutes before kick-off (389 matches,
10th-90th percentile 27.7-28.6), while Betfair's price moves fastest 60-45 and 90-60 minutes out, when the team sheets
are released (75 minutes before in the Premier League and UEFA competitions, 60 in the EFL). A line-up signal is only
tradeable from a source that has the elevens near the release. This measures Sportmonks (the owner's Pro plan):

- the day's fixtures in the plan's leagues (``sources.sportmonks.ALL_LEAGUES``), read hourly;
- every match kicking off in the next ``LEAD_MIN`` minutes polled every minute (``fixtures/multi``, up to 50 ids a
  call, ``include=lineups``) until both sides have eleven starters (lineup ``type_id`` 11), the first such poll's
  time kept;
- at the end, API-Football's times for the same day (the collector's raw polls) and the matches both feeds hold,
  paired by kick-off and team names.

Read-only; nothing is stored in S3 or traded. The token is read from ``SPORTMONKS_API_TOKEN``, never written.

    python -m research.open_close.football_sportmonks_lineup_timing --until 20:05 --out out/sm_lineup_timing.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

LEAD_MIN = 100.0
EVERY_S = 60
STARTER_TYPE = 11            # Sportmonks lineup type: 11 starting eleven, 12 bench (checked against each reply)


def _norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
    s = re.sub(r"\b(fc|cf|sc|ac|fk|sk|afc|cd|club|de|the)\b", " ", s)
    return " ".join(s.split())


def discover(s, day: str) -> dict[int, dict]:
    from sources.common import get
    from sources.sportmonks import BASE, PER_PAGE, league_groups
    out = {}
    for leagues in league_groups():
        page, more = 1, True
        while more:
            r = get(s, f"{BASE}/fixtures/date/{day}", params={"include": "participants;league",
                                                              "filters": f"fixtureLeagues:{leagues}",
                                                              "per_page": PER_PAGE, "page": page})
            body = r.json() if r is not None else {}
            for f in body.get("data") or []:
                parts = {(p.get("meta") or {}).get("location"): p for p in f.get("participants") or []}
                out[f["id"]] = {"id": f["id"], "kickoff": int(f.get("starting_at_timestamp") or 0),
                                "league": (f.get("league") or {}).get("name"),
                                "home": (parts.get("home") or {}).get("name"), "away": (parts.get("away") or {}).get("name"),
                                "home_id": (parts.get("home") or {}).get("id"), "away_id": (parts.get("away") or {}).get("id")}
            more = bool((body.get("pagination") or {}).get("has_more"))
            page += 1
    return out


def starters(f: dict) -> tuple[dict, dict]:
    """(starters a participant, every lineup type's count) from one fixture's reply."""
    n, types = {}, {}
    for x in f.get("lineups") or []:
        t = x.get("type_id")
        types[t] = types.get(t, 0) + 1
        if t == STARTER_TYPE:
            n[x.get("team_id")] = n.get(x.get("team_id"), 0) + 1
    return n, types


def watch(s, until: datetime, now=lambda: datetime.now(timezone.utc), sleep=time.sleep,
          checkpoint=None) -> tuple[dict, dict]:
    """Poll to ``until``; ``checkpoint(rows, meta)`` after each cycle that polled (the result kept if the job is
    stopped early: 26 Nov's run was ended by its time limit before it wrote anything)."""
    from sources.common import get
    from sources.sportmonks import BASE
    fx, seen, polls, types_seen = {}, {}, {}, {}
    last_disc = None
    while now() < until:
        t = now()
        if last_disc is None or (t - last_disc) >= timedelta(hours=1):
            fx.update(discover(s, t.date().isoformat()))
            last_disc = t
            log.info("sportmonks: %d fixtures today", len(fx))
        ts = t.timestamp()
        due = [i for i, f in fx.items() if i not in seen and 0 < f["kickoff"] - ts <= LEAD_MIN * 60]
        for k in range(0, len(due), 50):
            ids = due[k:k + 50]
            r = get(s, f"{BASE}/fixtures/multi/{','.join(str(i) for i in ids)}", params={"include": "lineups"})
            body = r.json() if r is not None else {}
            for f in body.get("data") or []:
                i = f["id"]
                polls[i] = polls.get(i, 0) + 1
                n, types = starters(f)
                for ty, c in types.items():
                    types_seen[ty] = types_seen.get(ty, 0) + c
                h, a = fx[i]["home_id"], fx[i]["away_id"]
                if n.get(h, 0) >= 11 and n.get(a, 0) >= 11:
                    seen[i] = t
                    log.info("sportmonks elevens: %s v %s (%s), %.1f min before kick-off", fx[i]["home"],
                             fx[i]["away"], fx[i]["league"], (fx[i]["kickoff"] - ts) / 60)
        if checkpoint is not None and due:
            checkpoint(_rows(fx, seen, polls), {"lineup_type_counts": types_seen})
        sleep(EVERY_S)
    return _rows(fx, seen, polls), {"lineup_type_counts": types_seen}


def _rows(fx: dict, seen: dict, polls: dict) -> dict:
    rows = {}
    for i, f in fx.items():
        ko = datetime.fromtimestamp(f["kickoff"], timezone.utc) if f["kickoff"] else None
        rows[i] = {**f, "kickoff_utc": str(ko), "polls": polls.get(i, 0),
                   "first_seen_utc": str(seen[i]) if i in seen else None,
                   "minutes_before_ko": round((ko - seen[i]).total_seconds() / 60, 1) if i in seen and ko else None}
    return rows


def api_football_times(store, day: str) -> pd.DataFrame:
    from research.open_close.football_lineup_release import lineup_tables
    times, _, _ = lineup_tables(store)
    if times is None or times.empty:
        return pd.DataFrame()
    t = times.copy()
    t["kick"] = pd.to_datetime(t["kickoff_utc"], utc=True)
    return t[t["kick"].dt.strftime("%Y-%m-%d") == day]


def pair(sm: pd.DataFrame, af: pd.DataFrame) -> pd.DataFrame:
    if sm.empty or af.empty:
        return pd.DataFrame()
    out = []
    for r in sm.itertuples(index=False):
        ko = pd.Timestamp(r.kickoff_utc)
        c = af[(af["kick"] - ko).abs() <= pd.Timedelta(minutes=10)]
        for q in c.itertuples(index=False):
            hn, an = _norm(getattr(q, "home", "")), _norm(getattr(q, "away", ""))
            if (_norm(r.home) in hn or hn in _norm(r.home)) and (_norm(r.away) in an or an in _norm(r.away)):
                out.append({"match": f"{r.home} v {r.away}", "kickoff": str(ko), "sportmonks_min": r.minutes_before_ko,
                            "api_football_min": q.minutes_before_ko})
                break
    return pd.DataFrame(out)


def summarise(rows: dict, meta: dict, day: str, final: bool) -> dict:
    sm = pd.DataFrame(list(rows.values()))
    watched = sm[sm["polls"] > 0] if len(sm) else sm
    got = watched.dropna(subset=["minutes_before_ko"]) if len(watched) else watched
    res = {"day": day, "final": final, "fixtures_today": int(len(sm)), "watched": int(len(watched)),
           "with_both_elevens": int(len(got)), **meta,
           "sportmonks_minutes_before_ko": {q: round(float(got["minutes_before_ko"].quantile(q)), 1)
                                            for q in (0.1, 0.5, 0.9)} if len(got) else "no data found",
           "by_league": got.groupby("league")["minutes_before_ko"].agg(["size", "median"]).round(1)
           .reset_index().to_dict("records") if len(got) else []}
    if final:
        try:
            from sources.common import Store
            af = api_football_times(Store(), day)
            p = pair(got, af)
            res["paired_with_api_football"] = int(len(p))
            if len(p):
                p["sportmonks_earlier_by_min"] = p["sportmonks_min"] - p["api_football_min"]
                res["paired_median"] = {"sportmonks_min": float(p["sportmonks_min"].median()),
                                        "api_football_min": float(p["api_football_min"].median()),
                                        "sportmonks_earlier_by_min": float(p["sportmonks_earlier_by_min"].median())}
                res["paired"] = p.to_dict("records")
        except Exception as exc:                                  # the pairing is a bonus; the timing stands alone
            res["paired_with_api_football"] = f"failed: {str(exc)[:200]}"
    res["matches"] = watched.sort_values("kickoff").to_dict("records") if len(watched) else []
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--until", required=True, help="UTC HH:MM to stop polling")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from sources.common import Store, env, session
    token = env("SPORTMONKS_API_TOKEN")
    if not token:
        print(json.dumps({"status": "no SPORTMONKS_API_TOKEN; nothing measured"}))
        return 0
    s = session()
    s.headers.update({"Authorization": token, "Accept": "application/json"})
    now = datetime.now(timezone.utc)
    h, m = (int(x) for x in a.until.split(":"))
    until = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if until <= now:                                              # past midnight UTC
        until += timedelta(days=1)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save(rows, meta, final=False):
        res = summarise(rows, meta, now.date().isoformat(), final)
        out.write_text(json.dumps(res, indent=1, default=str) + "\n")
        return res

    rows, meta = watch(s, until, checkpoint=save)
    res = save(rows, meta, final=True)
    print(json.dumps({k: v for k, v in res.items() if k not in ("matches", "paired")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
