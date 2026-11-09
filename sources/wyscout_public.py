"""The Wyscout public match event dataset (Pappalardo et al. 2019): every on-ball event of 1,941 matches, with pitch
positions. racing² football research, 6 Nov 2026 (the owner: "the most advanced set of football metrics").

Source: figshare collection "Soccer match event dataset" (doi:10.6084/m9.figshare.c.4415000), Pappalardo L., Massucco
E. et al., "A public data set of spatio-temporal match events in soccer competitions", Scientific Data 6, 236 (2019).
Licence: CC BY 4.0 (figshare's own record for each file, read 6 Nov 2026): commercial use allowed with attribution.
(StatsBomb's open data is NOT used: its user agreement forbids commercial use of the data or any analysis derived
from it.)

Coverage: 2017/18 Premier League, La Liga, Serie A, Bundesliga, Ligue 1, the 2018 World Cup and Euro 2016. Positions
are on a 0-100 grid in the direction the acting team attacks (x towards the opponent's goal, y across).

Raw files to ``sources/wyscout_public/raw/`` as served; tables ``events.parquet`` (one row an event: match, period,
second, team, player, event and sub-event ids, start and end positions, tags as a space-separated string,
competition) and ``matches.parquet`` (one row a match: teams, sides, scores, date, competition).

    python -m sources.wyscout_public --fetch --build [--root DIR]
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import zipfile

import pandas as pd

from sources.common import Store, get, session

log = logging.getLogger("wyscout_public")

PREFIX = "wyscout_public"
FILES = {                                   # name -> figshare file id (the collection's version 5)
    "events.zip": 14464685, "matches.zip": 14464622, "eventid2name.csv": 21385245, "tags2name.csv": 21385239,
    "players.json": 15073721, "teams.json": 15073697,
}
URL = "https://ndownloader.figshare.com/files/{}"
ATTRIBUTION = ("Pappalardo et al., A public data set of spatio-temporal match events in soccer competitions, "
               "Scientific Data 6, 236 (2019); data CC BY 4.0, doi:10.6084/m9.figshare.c.4415000")


def fetch(store: Store, force: bool = False, max_minutes: float | None = None) -> dict:
    have = store.listing(f"{PREFIX}/raw/")
    s, out = session(), {}
    # figshare answers a browser-like User-Agent with an empty 202 (a bot check): name ourselves plainly
    s.headers["User-Agent"] = "racing2-research/1.0"
    for name, fid in FILES.items():
        key = f"{PREFIX}/raw/{name}"
        if key in have and not force:
            out[name] = "held"
            continue
        r = get(s, URL.format(fid), timeout=600)
        if r is None or r.status_code != 200 or not r.content:
            out[name] = f"no data found (HTTP {getattr(r, 'status_code', None)})"
            continue
        store.put(key, r.content)
        out[name] = len(r.content)
    return out


def _events(raw: bytes, competition: str) -> pd.DataFrame:
    ev = json.loads(raw)
    rows = []
    for e in ev:
        pos = e.get("positions") or []
        p0 = pos[0] if pos else {}
        p1 = pos[1] if len(pos) > 1 else {}
        rows.append((e.get("matchId"), e.get("matchPeriod"), e.get("eventSec"), e.get("teamId"), e.get("playerId"),
                     e.get("eventId"), e.get("subEventId"), p0.get("x"), p0.get("y"), p1.get("x"), p1.get("y"),
                     " ".join(str(t["id"]) for t in e.get("tags") or []), e.get("id")))
    df = pd.DataFrame(rows, columns=["match_id", "period", "sec", "team_id", "player_id", "event_id", "sub_event_id",
                                     "x", "y", "x_end", "y_end", "tags", "id"])
    df["competition"] = competition
    df["sub_event_id"] = pd.to_numeric(df["sub_event_id"], errors="coerce")
    return df


def _matches(raw: bytes, competition: str) -> pd.DataFrame:
    rows = []
    for m in json.loads(raw):
        sides = {v.get("side"): v for v in (m.get("teamsData") or {}).values()}
        h, a = sides.get("home", {}), sides.get("away", {})
        rows.append({"match_id": m.get("wyId"), "date_utc": m.get("dateutc"), "competition": competition,
                     "home_id": h.get("teamId"), "away_id": a.get("teamId"), "home_score": h.get("score"),
                     "away_score": a.get("score"), "label": m.get("label"), "gameweek": m.get("gameweek")})
    return pd.DataFrame(rows)


def _unescape(v: str) -> str:
    if "\\u" in v and v.isascii():
        try:
            return v.encode("ascii").decode("unicode_escape")
        except UnicodeDecodeError:
            return v
    return v


def build(store: Store) -> dict:
    out = {}
    for stem, fn in (("events", _events), ("matches", _matches)):
        raw = store.get(f"{PREFIX}/raw/{stem}.zip")
        if raw is None:
            out[stem] = "no data found"
            continue
        frames = []
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for name in sorted(z.namelist()):
                comp = name.split("_", 1)[1].rsplit(".", 1)[0]
                frames.append(fn(z.read(name), comp))
        df = pd.concat(frames, ignore_index=True)
        store.put_parquet(f"{PREFIX}/{stem}.parquet", df)
        out[stem] = {"rows": int(len(df)), "matches": int(df["match_id"].nunique()),
                     "by_competition": df.groupby("competition")["match_id"].nunique().to_dict()}
    for stem, cols in (("players", {"wyId": "player_id", "shortName": "name", "currentTeamId": "team_id"}),
                       ("teams", {"wyId": "team_id", "name": "name", "officialName": "official_name"})):
        raw = store.get(f"{PREFIX}/raw/{stem}.json")
        if raw is None:
            out[stem] = "no data found"
            continue
        df = pd.DataFrame(json.loads(raw))
        df = df[[c for c in cols if c in df.columns]].rename(columns=cols)
        if "role" in pd.DataFrame(json.loads(raw)).columns and stem == "players":
            df["role"] = [r.get("code2") if isinstance(r, dict) else None for r in pd.DataFrame(json.loads(raw))["role"]]
        for c in ("name", "official_name"):           # the file double-escapes non-ASCII ('Mbapp\\u00e9')
            if c in df.columns:
                df[c] = df[c].astype(str).map(_unescape)
        for c in ("player_id", "team_id"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        store.put_parquet(f"{PREFIX}/{stem}.parquet", df)
        out[stem] = {"rows": int(len(df))}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--root", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    store = Store(root=a.root)
    res = {"attribution": ATTRIBUTION}
    if a.fetch:
        res["fetch"] = fetch(store)
    if a.build:
        res["build"] = build(store)
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
