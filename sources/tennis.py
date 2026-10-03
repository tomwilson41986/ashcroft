"""Tennis match data for the tennis model: the match statistics Jeff Sackmann compiled, and the results with closing
odds from tennis-data.co.uk.

- Sackmann's own repositories (github.com/JeffSackmann/tennis_atp, tennis_wta) were taken down in 2026; their files
  are read from the archival mirror github.com/Aneeshers/tennis-sackmann-archive (snapshot of June 2026): every
  tour-level match since 1968 (ATP) and 1920 (WTA) with the serve and return statistics (aces, double faults, first
  serves in and won, second serves won, break points), ranking and points, seed, entry, height, hand and age; the
  ATP's Futures; the WTA's qualifying and ITF matches. The ATP's tour-level matches after the snapshot come from
  github.com/Tennismylife/TML-Database, kept up to date in the same layout. Licence of both: non-commercial (CC
  BY-NC-SA 4.0): check before the data trains a model that stakes money.
- tennis-data.co.uk: every ATP match since 2000 and WTA match since 2007, with the bookmakers' closing prices
  (Pinnacle, Bet365, the market's maximum and average; Betfair's own until 2014) and the match's sets. Its Cloudflare
  refuses GitHub's runners (3 Oct: HTTP 403 on every file), so this half is its own source, ``tennis_odds``, run from
  the UK server (betfair-prices.yml).

Raw: ``tennis/raw/sackmann/<file>.csv``, ``tennis/raw/tml/<year>.csv`` and ``tennis/raw/tennis_data/<tour>_<year>.
<xlsx|xls>``, as published; the current and the previous year are fetched again each run. Tables:
``tennis/sackmann_matches.parquet`` (one row a match, with the tour, level and source) and
``tennis/odds_matches.parquet``.
"""

from __future__ import annotations

import io
import logging
import re
from datetime import date

from sources.common import Budget, Store, get, session

log = logging.getLogger(__name__)

SACKMANN = "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main"
TML = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master"
TENNIS_DATA = "http://www.tennis-data.co.uk"
FIRST_YEAR = 2000


def sackmann_files(first_year: int, last_year: int) -> list[tuple[str, str]]:
    """(file name, url) for every Sackmann file the model reads, newest first."""
    out = []
    for name in ("atp_players.csv", "atp_rankings_current.csv"):
        out.append((name, f"{SACKMANN}/atp/{name}"))
    for name in ("wta_players.csv", "wta_rankings_current.csv"):
        out.append((name, f"{SACKMANN}/wta/{name}"))
    for y in range(last_year, first_year - 1, -1):
        for stem in ("atp_matches", "atp_matches_qual_chall", "atp_matches_futures"):
            out.append((f"{stem}_{y}.csv", f"{SACKMANN}/atp/{stem}_{y}.csv"))
        for stem in ("wta_matches", "wta_matches_qual_itf"):
            out.append((f"{stem}_{y}.csv", f"{SACKMANN}/wta/{stem}_{y}.csv"))
    return out


def tennis_data_files(first_year: int, last_year: int) -> list[tuple[str, list[str]]]:
    """(stem, [urls to try]) a tour and year: .xlsx from 2013, .xls before (the site's own layout)."""
    out = []
    for y in range(last_year, first_year - 1, -1):
        for tour, folder in (("atp", f"{y}"), ("wta", f"{y}w")):
            if tour == "wta" and y < 2007:
                continue
            out.append((f"{tour}_{y}", [f"{TENNIS_DATA}/{folder}/{y}.xlsx", f"{TENNIS_DATA}/{folder}/{y}.xls"]))
    return out


def _fetch_csvs(store: Store, s, jobs, have, budget, out, fresh) -> bool:
    """(key, url) jobs, each a CSV kept as served; False once the budget is spent."""
    for key, url in jobs:
        if not budget.left():
            out["stopped"] = True
            return False
        if key in have and not fresh(key):
            out["skipped"] += 1
            continue
        try:
            r = get(s, url, pause=0.2)
        except Exception as exc:
            log.warning("%s not fetched (%s)", url, exc)
            out["failed"] += 1
            continue
        if r is None or r.content.lstrip()[:1] == b"<":
            out["missing"] += 1                          # a file the source does not have (404)
            continue
        store.put(key, r.content)
        out["fetched"] += 1
    return True


def fetch(store: Store, first_year: int = FIRST_YEAR, max_minutes: float | None = None, s=None,
          today: date | None = None) -> dict:
    """The match statistics: the Sackmann mirror, then TML's ATP years (this year and last fetched again)."""
    s = s or session()
    last = (today or date.today()).year
    have = store.listing("tennis/raw/")
    budget = Budget(max_minutes)
    out = {"fetched": 0, "missing": 0, "skipped": 0, "failed": 0, "stopped": False}
    recent = tuple(str(y) for y in (last, last - 1))

    def fresh(key: str) -> bool:
        return not re.search(r"\d{4}", key.rsplit("/", 1)[-1]) or any(y in key.rsplit("/", 1)[-1] for y in recent)
    jobs = [(f"tennis/raw/sackmann/{n}", u) for n, u in sackmann_files(first_year, last)]
    if _fetch_csvs(store, s, jobs, have, budget, out, fresh):
        jobs = [(f"tennis/raw/tml/{y}.csv", f"{TML}/{y}.csv") for y in range(last, first_year - 1, -1)]
        _fetch_csvs(store, s, jobs, have, budget, out, fresh)
    return out


def fetch_odds(store: Store, first_year: int = FIRST_YEAR, max_minutes: float | None = None, s=None,
               today: date | None = None) -> dict:
    """tennis-data.co.uk's results with odds (this year and last fetched again). Stops at the first refusal: the
    site's Cloudflare refuses whole hosts, so one 403 means every file would be refused."""
    s = s or session()
    last = (today or date.today()).year
    have = store.listing("tennis/raw/tennis_data/")
    budget = Budget(max_minutes)
    out = {"fetched": 0, "missing": 0, "skipped": 0, "refused": False, "stopped": False}
    for stem, urls in tennis_data_files(first_year, last):
        if not budget.left():
            out["stopped"] = True
            return out
        year = int(stem.split("_")[1])
        if any(k.startswith(f"tennis/raw/tennis_data/{stem}.") for k in have) and year < last - 1:
            out["skipped"] += 1
            continue
        for url in urls:
            try:
                r = get(s, url, pause=0.5)
            except Exception as exc:
                if "403" in str(exc):
                    log.warning("tennis-data.co.uk refuses this host (%s): no file fetched", exc)
                    out["refused"] = True
                    return out
                log.warning("%s not fetched (%s)", url, exc)
                r = None
            if r is not None and r.content[:2] in (b"PK", b"\xd0\xcf"):     # an xlsx (zip) or an xls (OLE) file
                store.put(f"tennis/raw/tennis_data/{stem}.{url.rsplit('.', 1)[1]}", r.content)
                out["fetched"] += 1
                break
        else:
            out["missing"] += 1
    return out


def _tour_level(name: str) -> tuple[str, str]:
    stem = re.sub(r"_\d{4}$", "", name[:-4])                # atp_matches_qual_chall_2024 -> atp_matches_qual_chall
    tour = stem.split("_")[0]
    if stem.endswith("qual_chall"):
        return tour, "challenger_qualifying"
    if stem.endswith("futures"):
        return tour, "futures"
    if stem.endswith("qual_itf"):
        return tour, "qualifying_itf"
    return tour, "tour"


def build(store: Store) -> dict:
    import pandas as pd
    out = {}
    parts = []
    for key in sorted(store.listing("tennis/raw/sackmann/")):
        name = key.rsplit("/", 1)[-1]
        if "_matches" not in name or "doubles" in name or "amateur" in name:
            continue
        df = pd.read_csv(io.BytesIO(store.get(key)), dtype=str, encoding="latin-1", on_bad_lines="skip")
        df["tour"], df["level_file"] = _tour_level(name)
        df["source"] = "sackmann"
        parts.append(df)
    for key in sorted(store.listing("tennis/raw/tml/")):                # after the mirror: TML wins a duplicate
        df = pd.read_csv(io.BytesIO(store.get(key)), dtype=str, encoding="latin-1", on_bad_lines="skip")
        df["tour"], df["level_file"], df["source"] = "atp", "tour", "tml"
        parts.append(df)
    if parts:
        m = pd.concat(parts, ignore_index=True, sort=False)
        # the same match from both: by date, tournament, round and the two players' names
        m = m.drop_duplicates(["tour", "level_file", "tourney_date", "tourney_name", "round", "winner_name", "loser_name"],
                              keep="last")
        m["match_date"] = pd.to_datetime(m["tourney_date"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
        text = {"tourney_id", "tourney_name", "surface", "tourney_level", "tourney_date", "winner_seed", "winner_entry",
                "winner_name", "winner_hand", "winner_ioc", "loser_seed", "loser_entry", "loser_name", "loser_hand",
                "loser_ioc", "score", "round", "tour", "level_file", "match_date", "winner_id", "loser_id", "source",
                "indoor"}
        num = [c for c in m.columns if c not in text]
        m[num] = m[num].apply(pd.to_numeric, errors="coerce")
        store.put_parquet("tennis/sackmann_matches.parquet", m)
        out["sackmann"] = {"matches": len(m), "from": m["match_date"].min(), "to": m["match_date"].max(),
                           "with_serve_stats": int(m["w_svpt"].notna().sum()) if "w_svpt" in m else 0}
    out.update(build_odds(store))
    log.info("tennis: %s", out)
    return out


def build_odds(store: Store) -> dict:
    import pandas as pd
    out = {}
    parts = []
    for key in sorted(store.listing("tennis/raw/tennis_data/")):
        stem = key.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        try:
            df = pd.read_excel(io.BytesIO(store.get(key)), dtype=str)
        except Exception as exc:                          # an .xls without xlrd, or a damaged file
            log.warning("%s not read (%s)", key, exc)
            continue
        df["tour"] = stem.split("_")[0]
        parts.append(df)
    if parts:
        o = pd.concat(parts, ignore_index=True, sort=False)
        o["match_date"] = pd.to_datetime(o["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        text = {"Location", "Tournament", "Date", "Series", "Tier", "Court", "Surface", "Round", "Winner", "Loser",
                "Comment", "tour", "match_date"}
        num = [c for c in o.columns if c not in text]
        o[num] = o[num].apply(pd.to_numeric, errors="coerce")
        store.put_parquet("tennis/odds_matches.parquet", o)
        out["tennis_data"] = {"matches": len(o), "from": o["match_date"].min(), "to": o["match_date"].max(),
                              "with_pinnacle": int(o["PSW"].notna().sum()) if "PSW" in o else 0}
    return out
