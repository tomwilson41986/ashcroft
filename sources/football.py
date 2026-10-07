"""Football results and odds from football-data.co.uk, for the football model.

Two kinds of file:

- the main leagues (England 0-4 and the Conference, Scotland 0-3, Germany, Italy, Spain, France 1-2, the Netherlands,
  Belgium, Portugal, Turkey, Greece), one file a season from 1993/94: result, half-time, shots, corners, cards,
  referee, and the bookmakers' opening and closing 1X2, over/under 2.5 and Asian handicap prices, Pinnacle (PS*, PSC*)
  and Betfair Exchange (BFE*, BFEC*, from 2019/20) among them;
- the extra leagues (Argentina, Austria, Brazil, China, Denmark, Finland, Ireland, Japan, Mexico, Norway, Poland,
  Romania, Russia, Sweden, Switzerland, the USA), one file holding every season since 2012: result and closing 1X2
  (Pinnacle, Betfair Exchange, the market's maximum and average).

Raw: ``football/raw/<season>/<div>.csv`` (e.g. ``2526/E0.csv``) and ``football/raw/new/<code>.csv``, as published,
and the upcoming fixtures with their opening prices, ``football/raw/fixtures/<day>/fixtures.csv``, kept a day a folder.
The current season's files and the extra leagues are fetched again each run. Table: ``football/matches.parquet``, one
row a match, the result columns named alike across both kinds and every odds column kept as football-data names it
(its notes: https://www.football-data.co.uk/notes.txt).
"""

from __future__ import annotations

import io
import logging
from datetime import date

from sources.common import Budget, Store, get, session

log = logging.getLogger(__name__)

BASE = "https://www.football-data.co.uk"
MAIN = ("E0", "E1", "E2", "E3", "EC", "SC0", "SC1", "SC2", "SC3", "D1", "D2", "I1", "I2", "SP1", "SP2", "F1", "F2",
        "N1", "B1", "P1", "T1", "G1")
EXTRA = ("ARG", "AUT", "BRA", "CHN", "DNK", "FIN", "IRL", "JPN", "MEX", "NOR", "POL", "ROU", "RUS", "SWE", "SWZ", "USA")
FIRST_SEASON = 1993
FIXTURES = ("fixtures.csv", "new_league_fixtures.csv")
#: the extra leagues' result columns, named as in the main files
RENAME = {"Home": "HomeTeam", "Away": "AwayTeam", "HG": "FTHG", "AG": "FTAG", "Res": "FTR"}


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def current_season(today: date | None = None) -> int:
    t = today or date.today()
    return t.year if t.month >= 7 else t.year - 1


def fetch(store: Store, first_season: int = FIRST_SEASON, max_minutes: float | None = None, s=None,
          today: date | None = None) -> dict:
    s = s or session()
    have = store.listing("football/raw/")
    now = current_season(today)
    budget = Budget(max_minutes)
    out = {"fetched": 0, "missing": 0, "skipped": 0, "stopped": False}
    jobs = [(f"football/raw/new/{c}.csv", f"{BASE}/new/{c}.csv", True) for c in EXTRA]
    for y in range(now, first_season - 1, -1):
        for div in MAIN:
            jobs.append((f"football/raw/{season_code(y)}/{div}.csv", f"{BASE}/mmz4281/{season_code(y)}/{div}.csv",
                         y >= now - (1 if (today or date.today()).month in (7, 8) else 0)))
    # the upcoming fixtures with their opening prices (collected Friday and Tuesday afternoons): bronze, by day
    stamp = (today or date.today()).isoformat()
    for name in FIXTURES:
        jobs.insert(0, (f"football/raw/fixtures/{stamp}/{name}", f"{BASE}/{name}", True))
    for key, url, always in jobs:
        if not budget.left():
            out["stopped"] = True
            break
        if key in have and not always:
            out["skipped"] += 1
            continue
        r = get(s, url, pause=0.5)
        if r is None or not r.content.strip() or r.content.lstrip()[:1] == b"<":
            out["missing"] += 1                          # a division that had no file that season
            continue
        store.put(key, r.content)
        out["fetched"] += 1
    return out


def read_csv(raw: bytes):
    import pandas as pd
    text = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(io.StringIO(text), dtype=str, on_bad_lines="skip", keep_default_na=False)
    df = df.loc[:, [c for c in df.columns if c and not str(c).startswith("Unnamed")]]
    df.columns = [str(c).strip() for c in df.columns]
    return df.rename(columns=RENAME)


def build(store: Store) -> dict:
    import pandas as pd
    parts = []
    for key in sorted(store.listing("football/raw/")):
        if not key.endswith(".csv"):
            continue
        df = read_csv(store.get(key))
        if df.empty or "HomeTeam" not in df.columns:
            continue
        bits = key.split("/")
        if bits[2] == "fixtures":
            continue                                     # upcoming matches: the gold build reads them (football.data)
        if bits[2] == "new":
            df["source_file"] = f"new/{bits[3]}"
            df["Div"] = bits[3][:-4]
            df["season"] = df.get("Season")
        else:
            df["source_file"] = f"{bits[2]}/{bits[3]}"
            df["season"] = f"{2000 + int(bits[2][:2]) if int(bits[2][:2]) < 90 else 1900 + int(bits[2][:2])}"
        parts.append(df[df["HomeTeam"].astype(str).str.strip() != ""])
    if not parts:
        return {"matches": 0}
    m = pd.concat(parts, ignore_index=True, sort=False)
    m["match_date"] = pd.to_datetime(m["Date"], dayfirst=True, errors="coerce", format="mixed").dt.strftime("%Y-%m-%d")
    for c in ("FTHG", "FTAG", "HTHG", "HTAG"):
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")
    keep = ["source_file", "Div", "Country", "League", "season", "match_date", "Time", "HomeTeam", "AwayTeam",
            "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", "Referee"]
    odds = [c for c in m.columns if c not in keep and c not in ("Date", "Season")]
    m = m[[c for c in keep if c in m.columns] + odds]
    num = [c for c in odds if c not in ("Referee",)]
    m[num] = m[num].apply(pd.to_numeric, errors="coerce")
    m = m.dropna(axis=1, how="all")
    store.put_parquet("football/matches.parquet", m)
    out = {"matches": len(m), "files": len(parts), "leagues": int(m["Div"].nunique()),
           "from": m["match_date"].min(), "to": m["match_date"].max(),
           "with_betfair_close": int(m.filter(like="BFEC").notna().any(axis=1).sum())}
    log.info("football: %s", out)
    return out
