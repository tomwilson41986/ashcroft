"""openfootball's football.json (github.com/openfootball/football.json): results of the main European leagues, public
domain (CC0), the latest season refreshed daily at 05:00 UTC. The licence-clean cross-check of football-data.co.uk's
scores (football.data's ``score_vs_openfootball`` check).

Raw: ``openfootball/raw/<season>/<file>.json`` (e.g. ``2025-26/en.1.json``), as published; the current and previous
seasons fetched again each run. Table: ``openfootball/matches.parquet``: competition_id, match_date, kickoff (local
as published), home, away, ft_home, ft_away, ht_home, ht_away, round, the source's names unresolved (the gold build
pairs them with fd's matches on the day).
"""

from __future__ import annotations

import json
import logging
from datetime import date

from sources.common import Budget, Store, get, session

log = logging.getLogger(__name__)

BASE = "https://raw.githubusercontent.com/openfootball/football.json/master"
FILES = {"en.1": "ENG1", "en.2": "ENG2", "en.3": "ENG3", "en.4": "ENG4", "sco.1": "SCO1", "de.1": "GER1",
         "de.2": "GER2", "es.1": "ESP1", "es.2": "ESP2", "it.1": "ITA1", "it.2": "ITA2", "fr.1": "FRA1",
         "fr.2": "FRA2", "nl.1": "NED1", "pt.1": "POR1", "be.1": "BEL1", "tr.1": "TUR1", "gr.1": "GRE1",
         "at.1": "AUT1"}
FIRST_SEASON = 2010


def season_dir(y: int) -> str:
    return f"{y}-{(y + 1) % 100:02d}"


def fetch(store: Store, first_season: int = FIRST_SEASON, max_minutes: float | None = None, s=None,
          today: date | None = None) -> dict:
    from sources.football import current_season
    s = s or session()
    have = store.listing("openfootball/raw/")
    now = current_season(today)
    budget = Budget(max_minutes)
    out = {"fetched": 0, "missing": 0, "skipped": 0, "stopped": False}
    for y in range(now, first_season - 1, -1):
        for f in FILES:
            if not budget.left():
                out["stopped"] = True
                return out
            key = f"openfootball/raw/{season_dir(y)}/{f}.json"
            if key in have and y < now - 1:
                out["skipped"] += 1
                continue
            r = get(s, f"{BASE}/{season_dir(y)}/{f}.json", pause=0.2)
            if r is None or not r.content.strip():
                out["missing"] += 1
                continue
            store.put(key, r.content)
            out["fetched"] += 1
    return out


def build(store: Store) -> dict:
    import pandas as pd
    rows = []
    for key in sorted(store.listing("openfootball/raw/")):
        if not key.endswith(".json"):
            continue
        comp = FILES.get(key.rsplit("/", 1)[1][:-5])
        if comp is None:
            continue
        try:
            doc = json.loads(store.get(key))
        except ValueError:
            continue
        for m in doc.get("matches", []):
            sc = m.get("score") or {}
            if isinstance(sc, list):                     # some files give the full-time score as a bare pair
                sc = {"ft": sc}
            ft, ht = sc.get("ft") or [None, None], sc.get("ht") or [None, None]
            rows.append({"competition_id": comp, "match_date": m.get("date"), "kickoff": m.get("time"),
                         "home": m.get("team1"), "away": m.get("team2"), "ft_home": ft[0], "ft_away": ft[1],
                         "ht_home": ht[0], "ht_away": ht[1], "round": m.get("round"), "source_file": key})
    if not rows:
        return {"matches": 0}
    df = pd.DataFrame(rows)
    store.put_parquet("openfootball/matches.parquet", df)
    out = {"matches": len(df), "with_score": int(df.ft_home.notna().sum()), "from": df.match_date.min(),
           "to": df.match_date.max()}
    log.info("openfootball: %s", out)
    return out
