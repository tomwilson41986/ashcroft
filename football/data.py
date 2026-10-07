"""The football tables the model and the trader read: football-data.co.uk's silver table made canonical, with its odds,
the closing-line benchmark, the data-quality checks and the link to Betfair's markets.

The store follows the bronze/silver/gold layout of the design (reports/football_model.md), in the sources store:

- bronze: ``sources/football/raw/...`` (football-data.co.uk, as published; ``sources.football``),
  ``sources/openfootball/raw/...``, ``sources/api_football/raw/...``, ``sources/betfair_historic/raw/soccer/...``;
- silver: ``sources/football/matches.parquet`` (one row a match, fd's own columns), the other sources' tables;
- gold (this module): ``sources/football/gold/``
    - ``matches.parquet``: one row a match, keyed by ``match_id``, kick-off in UTC, ``status``, scores, match stats,
      the opening and closing 1X2 / over-under 2.5 / Asian handicap prices chosen as the benchmark, a
      ``record_version`` raised whenever a score, status or kick-off changes;
    - ``matches_history.parquet``: every earlier version of a changed row (corrections stay auditable);
    - ``odds_long.parquet``: every bookmaker's price, one row a (match, bookmaker, market, line, outcome, phase);
    - ``betfair_market_map.parquet``: each Betfair Historic Data soccer market tied to its ``match_id``, with the
      exchange's first, T-60, T-1 and closing prices and the settled winner; ``team_aliases.parquet`` the names
      learned there; ``betfair_review.parquet`` the events that could not be tied uniquely (for a person);
- ``sources/football/dq/latest.json`` and ``dq/<run>.json``: the data-quality checks of each build.

``match_id`` is a hash of competition, season, the two canonical team ids and the meeting's number in that season
(the first or second time that home side hosts that away side: split leagues meet more than twice), not the date, so a
rescheduled match keeps its id. The gold build is idempotent: the same bronze gives the same gold, and a rebuild only
raises ``record_version`` where something really changed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from football.reference import BY_FD, BY_ID, MULTI_ROUND, load_aliases, pair_event, team_id
from sources.football import MAIN as MAIN_CODES

log = logging.getLogger(__name__)

GOLD = "football/gold"
PINNACLE_UNRELIABLE = "2025-07-23"     # football-data.co.uk: Pinnacle's public odds systematically stale since then
#: bookmaker codes as football-data names them (notes.txt); longest first, so 'BFE' wins over 'BF', 'BFD' (Betfred)
BOOKS = sorted(["B365", "BW", "BF", "BFD", "BFE", "BV", "PS", "P", "WH", "1XB", "Max", "Avg", "IW", "LB", "VC", "SJ",
                "GB", "BS", "SB", "SO", "BbMx", "BbAv", "PP", "SKB", "CL", "BMGM", "SY"], key=len, reverse=True)
SUFFIX = {"H": ("1X2", "H"), "D": ("1X2", "D"), "A": ("1X2", "A"), ">2.5": ("OU25", "O"), "<2.5": ("OU25", "U"),
          "AHH": ("AH", "H"), "AHA": ("AH", "A")}
#: the seasons cut short (COVID-19, 2019/20; the National League's 2020/21 too): short by design
CURTAILED = {("ENG3", 2019), ("ENG4", 2019), ("ENG5", 2019), ("ENG5", 2020), ("FRA1", 2019), ("FRA2", 2019),
             ("BEL1", 2019), ("NED1", 2019), ("SCO1", 2019), ("SCO2", 2019), ("SCO3", 2019), ("SCO4", 2019)}
KEY_FIELDS = ["status", "ft_home", "ft_away", "ht_home", "ht_away", "kickoff_utc"]
#: the benchmark price sources in order of preference; a close is the last price before kick-off
OPEN_1X2 = ["BFE", "PS", "Avg", "B365", "BbAv"]
OPEN_OU = ["BFE", "P", "Avg", "B365", "BbAv"]
OPEN_AH = ["BFE", "P", "Avg", "B365", "BbAv"]
PINNACLE = {"PS", "PSC", "P", "PC", "PAH", "PCAH"}
CLOSE_OLD = ["PSC", "BFEC", "AvgC"]     # to 2024/25: Pinnacle's close is the sharpest
CLOSE_NEW = ["BFEC", "AvgC", "PSC"]     # from 2025/26: Pinnacle's feed is unreliable


def season_label(y: int) -> str:
    return f"{y}-{(y + 1) % 100:02d}"


def make_match_id(comp: str, season: int, home: str, away: str, n: int) -> str:
    return hashlib.sha1(f"{comp}|{season}|{home}|{away}|{n}".encode()).hexdigest()[:16]


def kickoff_utc(dates: pd.Series, times: pd.Series) -> pd.Series:
    """football-data's date (YYYY-MM-DD) and UK local time (HH:MM) -> UTC timestamps, DST-aware; NaT where the time
    is missing (before 2019/20) or falls in the clocks' change."""
    t = times.fillna("").astype(str).str.strip()
    ok = t.str.match(r"^\d{1,2}:\d{2}$")
    local = pd.to_datetime(dates.where(ok) + " " + t.where(ok), errors="coerce", format="%Y-%m-%d %H:%M")
    return local.dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    v = pd.to_numeric(df[col], errors="coerce")
    return v.where(v > 1.0)                                  # a price at or below 1.0 is a missing price


def _pick(df: pd.DataFrame, books: list[str], cols: tuple[str, ...], lo: float, hi: float,
          drop_pinnacle_after: str | None = PINNACLE_UNRELIABLE) -> tuple[pd.DataFrame, pd.Series]:
    """Per row, the first bookmaker in ``books`` with every price of ``cols`` (suffixes) present and a plausible book
    (sum of 1/price in [lo, hi]). Returns the prices and the bookmaker chosen."""
    out = pd.DataFrame(np.nan, index=df.index, columns=list(cols))
    src = pd.Series(None, index=df.index, dtype=object)
    late = (df["match_date"] >= drop_pinnacle_after) if drop_pinnacle_after else pd.Series(False, index=df.index)
    for bk in books:
        prices = pd.DataFrame({c: _num(df, f"{bk}{c}") for c in cols})
        book = (1.0 / prices).sum(axis=1, min_count=len(cols))
        good = prices.notna().all(axis=1) & book.between(lo, hi) & src.isna()
        if bk in PINNACLE:
            good &= ~late
        out.loc[good] = prices.loc[good].to_numpy()
        src.loc[good] = bk
    return out, src


def benchmarks(m: pd.DataFrame) -> pd.DataFrame:
    """The opening and closing prices the model is scored against, per market, with their source."""
    out = pd.DataFrame(index=m.index)
    new = m["season"] >= 2025
    for phase, books in (("open", OPEN_1X2), ("close", None)):
        if books is None:
            a, sa = _pick(m, CLOSE_OLD, ("H", "D", "A"), 0.95, 1.35)
            b, sb = _pick(m, CLOSE_NEW, ("H", "D", "A"), 0.95, 1.35)
            p = a.where(~new, b)
            s = sa.where(~new, sb)
        else:
            p, s = _pick(m, books, ("H", "D", "A"), 0.95, 1.35)
        out[[f"{phase}_h", f"{phase}_d", f"{phase}_a"]] = p.to_numpy()
        out[f"{phase}_1x2_src"] = s
    # over/under 2.5
    p, s = _pick(m, OPEN_OU, (">2.5", "<2.5"), 0.95, 1.30)
    out[["open_o25", "open_u25"]] = p.to_numpy()
    out["open_ou_src"] = s
    a, sa = _pick(m, ["PC", "BFEC", "AvgC"], (">2.5", "<2.5"), 0.95, 1.30)
    b, sb = _pick(m, ["BFEC", "AvgC", "PC"], (">2.5", "<2.5"), 0.95, 1.30)
    out[["close_o25", "close_u25"]] = a.where(~new, b).to_numpy()
    out["close_ou_src"] = sa.where(~new, sb)
    # Asian handicap: the line is the home side's handicap, the same line for every bookmaker of a phase
    line_open = _num_line(m, "AHh").fillna(_num_line(m, "BbAHh"))
    line_close = _num_line(m, "AHCh")
    p, s = _pick(m, [f"{b}AH" for b in OPEN_AH], ("H", "A"), 0.95, 1.30)
    out["open_ah_line"], out[["open_ah_h", "open_ah_a"]], out["open_ah_src"] = line_open, p.to_numpy(), s
    a, sa = _pick(m, ["PCAH", "BFECAH", "AvgCAH"], ("H", "A"), 0.95, 1.30)
    b, sb = _pick(m, ["BFECAH", "AvgCAH", "PCAH"], ("H", "A"), 0.95, 1.30)
    out["close_ah_line"] = line_close
    out[["close_ah_h", "close_ah_a"]] = a.where(~new, b).to_numpy()
    out["close_ah_src"] = sa.where(~new, sb)
    # Betfair Exchange on its own: the venue the trader bets on
    for phase, bk in (("open", "BFE"), ("close", "BFEC")):
        for o in ("H", "D", "A"):
            out[f"bfe_{phase}_{o.lower()}"] = _num(m, f"{bk}{o}")
        out[f"bfe_{phase}_o25"] = _num(m, f"{bk}>2.5")
        out[f"bfe_{phase}_u25"] = _num(m, f"{bk}<2.5")
    return out


def _num_line(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


def parse_odds_column(col: str) -> tuple[str, str, str, str] | None:
    """'PSCH' -> ('PS', 'close', '1X2', 'H'); 'BFE>2.5' -> ('BFE', 'open', 'OU25', 'O'); 'BFDH' (Betfred) ->
    ('BFD', 'open', '1X2', 'H'); 'HST' -> None."""
    for bk in BOOKS:
        if not col.startswith(bk):
            continue
        rest = col[len(bk):]
        phase = "open"
        if rest.startswith("C") and rest[1:] in SUFFIX:
            phase, rest = "close", rest[1:]
        if rest in SUFFIX:
            market, outcome = SUFFIX[rest]
            return bk, phase, market, outcome
    return None


def odds_long(m: pd.DataFrame) -> pd.DataFrame:
    """Every bookmaker's price as rows: match_id, bookmaker, phase (open/close), market (1X2/OU25/AH), line,
    outcome, price."""
    parts = []
    line_open = _num_line(m, "AHh").fillna(_num_line(m, "BbAHh"))
    line_close = _num_line(m, "AHCh")
    for col in m.columns:
        got = parse_odds_column(col)
        if got is None:
            continue
        bk, phase, market, outcome = got
        price = pd.to_numeric(m[col], errors="coerce")
        keep = price > 1.0
        if not keep.any():
            continue
        line = (line_close if phase == "close" else line_open) if market == "AH" else (
            pd.Series(2.5 if market == "OU25" else np.nan, index=m.index))
        parts.append(pd.DataFrame({"match_id": m.loc[keep, "match_id"], "bookmaker": bk, "phase": phase,
                                   "market": market, "line": line[keep], "outcome": outcome,
                                   "price": price[keep]}))
    if not parts:
        return pd.DataFrame(columns=["match_id", "bookmaker", "phase", "market", "line", "outcome", "price"])
    return pd.concat(parts, ignore_index=True)


def canonical(silver: pd.DataFrame, fixtures: pd.DataFrame | None = None) -> pd.DataFrame:
    """football-data's silver table (and the upcoming fixtures, SCHEDULED) -> one canonical row a match."""
    m = silver.copy()
    if fixtures is not None and len(fixtures):
        f = fixtures.copy()
        f["status"] = "SCHEDULED"
        m["status"] = "FINISHED"
        m = pd.concat([m, f], ignore_index=True, sort=False)
    else:
        m["status"] = "FINISHED"
    m = m[m["Div"].isin(BY_FD) & m["HomeTeam"].notna() & m["AwayTeam"].notna() & m["match_date"].notna()].copy()
    m["competition_id"] = m["Div"].map(lambda d: BY_FD[d].competition_id)
    m["country"] = m["Div"].map(lambda d: BY_FD[d].country)
    m["tier"] = m["Div"].map(lambda d: BY_FD[d].tier)
    m["season"] = pd.to_numeric(m["season"].astype(str).str[:4], errors="coerce")
    if m["season"].isna().any():                               # a fixture: the season its date falls in
        d = pd.to_datetime(m["match_date"])
        cal = m["competition_id"].map(lambda c: BY_ID[c].fd_code in ("ARG", "BRA", "CHN", "FIN", "IRL", "JPN",
                                                                         "NOR", "SWE", "USA", "RUS"))
        guess = np.where(cal, d.dt.year, np.where(d.dt.month >= 7, d.dt.year, d.dt.year - 1))
        m["season"] = m["season"].fillna(pd.Series(guess, index=m.index))
    m["season"] = m["season"].astype(int)
    m["home_team"] = m["HomeTeam"].astype(str).str.strip()
    m["away_team"] = m["AwayTeam"].astype(str).str.strip()
    m["home_team_id"] = [team_id(c, n) for c, n in zip(m["country"], m["home_team"])]
    m["away_team_id"] = [team_id(c, n) for c, n in zip(m["country"], m["away_team"])]
    # a played match wins over its own fixture row (the same pairing, the same day)
    m["_rank"] = (m["status"] != "FINISHED").astype(int)
    m = m.sort_values(["_rank"]).drop_duplicates(
        ["competition_id", "match_date", "home_team_id", "away_team_id"], keep="first")
    m = m.sort_values(["competition_id", "season", "match_date", "Time" if "Time" in m else "match_date"],
                      kind="stable")
    m["meeting"] = m.groupby(["competition_id", "season", "home_team_id", "away_team_id"]).cumcount() + 1
    m["match_id"] = [make_match_id(*k) for k in zip(m["competition_id"], m["season"], m["home_team_id"],
                                                    m["away_team_id"], m["meeting"])]
    m["kickoff_utc"] = kickoff_utc(m["match_date"], m["Time"] if "Time" in m else pd.Series(index=m.index))
    num = {"FTHG": "ft_home", "FTAG": "ft_away", "HTHG": "ht_home", "HTAG": "ht_away", "HS": "shots_home",
           "AS": "shots_away", "HST": "sot_home", "AST": "sot_away", "HC": "corners_home", "AC": "corners_away",
           "HY": "yellow_home", "AY": "yellow_away", "HR": "red_home", "AR": "red_away"}
    out = pd.DataFrame({
        "match_id": m["match_id"], "competition_id": m["competition_id"], "fd_code": m["Div"],
        "country": m["country"], "tier": m["tier"], "season": m["season"],
        "season_label": m["season"].map(season_label), "meeting": m["meeting"], "match_date": m["match_date"],
        "kickoff_uk": m["Time"] if "Time" in m else None, "kickoff_utc": m["kickoff_utc"],
        "home_team": m["home_team"], "away_team": m["away_team"], "home_team_id": m["home_team_id"],
        "away_team_id": m["away_team_id"], "status": m["status"],
        "referee": m["Referee"] if "Referee" in m else None,
        "fd_couk_row_key": m.get("source_file", pd.Series("fixtures", index=m.index)).fillna("fixtures"),
    })
    for src, dst in num.items():
        out[dst] = pd.to_numeric(m[src], errors="coerce") if src in m.columns else np.nan
    # a played row without a score: abandoned, awarded or void in fd's file (no result the model can use)
    out.loc[(out["status"] == "FINISHED") & out[["ft_home", "ft_away"]].isna().any(axis=1), "status"] = "NO_RESULT"
    sched = out["status"] != "FINISHED"
    for c in ("ft_home", "ft_away", "ht_home", "ht_away"):
        out.loc[sched, c] = np.nan
    out = pd.concat([out, benchmarks(m.assign(match_date=m["match_date"]))], axis=1)
    out["source_of_truth"] = "fd_couk"
    return out.reset_index(drop=True)


def _same(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a == b) | (a.isna() & b.isna())


def version(prev: pd.DataFrame | None, new: pd.DataFrame, now: str | None = None
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The upsert: ``new`` against the last gold ``prev``. A row whose score, status or kick-off changed gets its
    ``record_version`` raised and its old version returned for the history; an unchanged row keeps its stamps; a row
    only in ``prev`` is kept (gold never loses a match). Returns (gold, the old versions of changed rows)."""
    now = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    new = new.copy()
    if prev is None or prev.empty:
        new["record_version"], new["first_seen_ts"], new["last_updated_ts"] = 1, now, now
        return new, new.iloc[0:0]
    p = prev.set_index("match_id")
    n = new.set_index("match_id")
    common = n.index.intersection(p.index)
    changed = pd.Series(False, index=common)
    for c in KEY_FIELDS:
        if c in n.columns and c in p.columns:
            a, b = n.loc[common, c], p.loc[common, c]
            if c == "kickoff_utc":
                a, b = pd.to_datetime(a, utc=True), pd.to_datetime(b, utc=True)
            changed |= ~_same(a, b)
    n["record_version"] = 1
    n["first_seen_ts"] = now
    n["last_updated_ts"] = now
    n.loc[common, "first_seen_ts"] = p.loc[common, "first_seen_ts"]
    n.loc[common, "record_version"] = p.loc[common, "record_version"] + changed.astype(int)
    n.loc[common, "last_updated_ts"] = np.where(changed, now, p.loc[common, "last_updated_ts"])
    gone = p.index.difference(n.index)
    out = pd.concat([n, p.loc[gone]], sort=False).reset_index()
    hist = p.loc[changed[changed].index].reset_index()
    hist["superseded_ts"] = now
    return out, hist


# --------------------------------------------------------------------------------------------------------------------
# Data quality
# --------------------------------------------------------------------------------------------------------------------

def dq_checks(g: pd.DataFrame, today: str | None = None) -> pd.DataFrame:
    """One row an issue: check_name, match_id, details, severity (high/medium/low)."""
    today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    issues = []

    def add(name, rows, detail, sev):
        for mid, d in zip(rows["match_id"], detail(rows) if callable(detail) else [detail] * len(rows)):
            issues.append({"check_name": name, "match_id": mid, "details": d, "severity": sev})

    fin = g[g.status == "FINISHED"]
    # duplicates: one row a match_id; one (competition, season, home, away, meeting)
    dup = g[g.duplicated("match_id", keep=False)]
    add("duplicate_match_id", dup, "match_id appears more than once", "high")
    # internal logic
    neg = fin[(fin[["ft_home", "ft_away"]] < 0).any(axis=1)]
    add("negative_score", neg, "a negative score", "high")
    noscore = fin[fin[["ft_home", "ft_away"]].isna().any(axis=1)]
    add("finished_without_score", noscore, "FINISHED with no full-time score", "high")
    add("no_result", g[g.status == "NO_RESULT"], "played row without a score (abandoned, awarded or void)", "low")
    ht = fin[(fin.ht_home > fin.ft_home) | (fin.ht_away > fin.ft_away)]
    add("ht_above_ft", ht, lambda r: [f"HT {a}-{b} FT {c}-{d}" for a, b, c, d in
                                      zip(r.ht_home, r.ht_away, r.ft_home, r.ft_away)], "high")
    sched_scores = g[(g.status == "SCHEDULED") & g[["ft_home", "ft_away"]].notna().any(axis=1)]
    add("scheduled_with_score", sched_scores, "SCHEDULED with a score", "medium")
    stale = g[(g.status == "SCHEDULED") & (g.match_date < today)]
    add("scheduled_in_past", stale, lambda r: [f"scheduled for {d}" for d in r.match_date], "low")
    # odds
    book = 1 / g[["open_h", "open_d", "open_a"]]
    over = book.sum(axis=1, min_count=3)
    add("open_overround", g[(over > 1.15)], lambda r: [f"open book {v:.3f}" for v in over[r.index]], "low")
    late_ps = g[(g.match_date >= PINNACLE_UNRELIABLE) & (g.close_1x2_src == "PSC")]
    add("pinnacle_after_unreliable", late_ps, "Pinnacle close used after 23/07/2025", "medium")
    # completeness: a main division's finished season is a double round-robin, n*(n-1) matches (the extra leagues
    # are left out: play-offs, splits and calendar seasons make no fixed count)
    seasons = fin[fin.fd_code.isin(MAIN_CODES)].groupby(["competition_id", "season"])
    last = int(g["season"].max()) if len(g) else 0
    for (comp, season), grp in seasons:
        if comp in MULTI_ROUND or season >= last:
            continue
        teams = pd.unique(pd.concat([grp.home_team_id, grp.away_team_id]))
        want = len(teams) * (len(teams) - 1)
        if len(grp) != want:
            known = (comp, int(season)) in CURTAILED
            issues.append({"check_name": "season_match_count", "match_id": None,
                           "details": f"{comp} {season}: {len(grp)} finished, {want} expected for {len(teams)} teams"
                                      + (" (curtailed, COVID-19)" if known else ""),
                           "severity": "low" if known else "medium" if abs(len(grp) - want) <= 2 else "high"})
    # freshness: the newest finished match is recent (in season)
    if len(fin):
        newest = fin.match_date.max()
        age = (pd.Timestamp(today) - pd.Timestamp(newest)).days
        if age > 10 and pd.Timestamp(today).month not in (6, 7):
            issues.append({"check_name": "freshness", "match_id": None,
                           "details": f"newest finished match {newest}, {age} days old", "severity": "high"})
    return pd.DataFrame(issues, columns=["check_name", "match_id", "details", "severity"])


def link_other(g: pd.DataFrame, other: pd.DataFrame, source: str, aliases=None) -> pd.DataFrame:
    """Another source's matches (columns competition_id, match_date, home, away, ...) with the canonical match_id,
    paired by competition and UK day, both team names agreeing (``football.reference.pair_event``); the rows that
    pair with nothing are left out."""
    if other is None or other.empty:
        return pd.DataFrame()
    o = other[other.competition_id.notna() & other.match_date.notna()].copy()
    cand = {k: v for k, v in g.groupby(["competition_id", "match_date"])}
    ids = []
    for r in o.itertuples(index=False):
        day = cand.get((r.competition_id, r.match_date))
        if day is None:
            ids.append(None)
            continue
        fx = list(zip(day.match_id, day.home_team, day.away_team))
        i, _ = pair_event(str(r.home), str(r.away), fx, BY_ID[r.competition_id].country, source, aliases)
        ids.append(fx[i][0] if i is not None else None)
    o["match_id"] = ids
    return o[o.match_id.notna()].drop_duplicates("match_id", keep="last")


def cross_check(g: pd.DataFrame, linked: pd.DataFrame, source: str) -> pd.DataFrame:
    """Full-time scores against another source's linked rows: the matches where both have a score and disagree."""
    if linked is None or linked.empty:
        return pd.DataFrame(columns=["check_name", "match_id", "details", "severity"])
    j = g[g.status == "FINISHED"].merge(linked[["match_id", "ft_home", "ft_away"]], on="match_id",
                                       suffixes=("", "_o"))
    j = j[j.ft_home_o.notna() & j.ft_away_o.notna()]
    bad = j[(j.ft_home != j.ft_home_o) | (j.ft_away != j.ft_away_o)]
    return pd.DataFrame({"check_name": f"score_vs_{source}", "match_id": bad.match_id,
                         "details": [f"fd {a}-{b}, {source} {c}-{d}" for a, b, c, d in
                                     zip(bad.ft_home, bad.ft_away, bad.ft_home_o, bad.ft_away_o)],
                         "severity": "high"})


def apply_api_football(g: pd.DataFrame, linked: pd.DataFrame) -> pd.DataFrame:
    """API-Football is the source of truth for a match's status and kick-off: a fixture fd lists as SCHEDULED takes
    its POSTPONED / CANCELLED / ABANDONED / AWARDED status, and every linked match its kick-off, venue and id."""
    if linked is None or linked.empty:
        return g
    g = g.copy()
    a = linked.set_index("match_id")
    idx = g.match_id.isin(a.index)
    mids = g.loc[idx, "match_id"]
    g.loc[idx, "apif_fixture_id"] = mids.map(a.apif_fixture_id).to_numpy()
    g.loc[idx, "venue"] = mids.map(a.venue).to_numpy()
    ko = pd.to_datetime(mids.map(a.kickoff_utc), utc=True)
    g["kickoff_utc"] = pd.to_datetime(g["kickoff_utc"], utc=True)
    g.loc[idx, "kickoff_utc"] = ko.where(ko.notna(), g.loc[idx, "kickoff_utc"]).to_numpy()
    st = mids.map(a.status)
    over = (g.loc[idx, "status"] == "SCHEDULED") & st.isin(["POSTPONED", "CANCELLED", "ABANDONED", "AWARDED",
                                                             "SUSPENDED"])
    g.loc[over[over].index, "status"] = st[over].to_numpy()
    return g


# --------------------------------------------------------------------------------------------------------------------
# Betfair's markets
# --------------------------------------------------------------------------------------------------------------------

BF_COUNTRY = {}
for _c in BY_FD.values():
    BF_COUNTRY.setdefault(_c.betfair_country, set()).add(_c.country)


def split_event(name: str) -> tuple[str, str] | None:
    parts = re.split(r"\s+v\s+|\s+vs\.?\s+|\s+@\s+", str(name or ""), maxsplit=1)
    return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else None


def betfair_map(g: pd.DataFrame, bf: pd.DataFrame, aliases=None, hours: float = 3.0
                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Betfair Historic Data soccer rows (betfair_historic's table: one row a runner a market) -> (the market map,
    the aliases learned, the events left for review). An event is tied to the one canonical match of that country
    whose kick-off is within ``hours`` of the market's start (the same UK day when fd has no time), both sides
    agreeing; anything not unique goes to review. The stored ids are reused: an event is matched by name once."""
    cols = ["match_id", "betfair_event_id", "market_id", "market_type", "market_start_time", "selection_id",
            "runner_name", "outcome", "handicap", "won", "ltp_first", "ltp_t_60", "ltp_t_1", "ltp_close", "bsp"]
    if bf is None or bf.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=["source", "country", "source_name", "team_id"]), \
            pd.DataFrame(columns=["betfair_event_id", "event_name", "country", "market_start_time", "best_score"])
    ev = bf.drop_duplicates("event_id")[["event_id", "event_name", "country", "market_time"]].copy()
    ev["start"] = pd.to_datetime(ev["market_time"], utc=True, errors="coerce")
    ev["uk_day"] = ev["start"].dt.tz_convert("Europe/London").dt.strftime("%Y-%m-%d")
    cand = g[["match_id", "country", "match_date", "kickoff_utc", "home_team", "away_team", "home_team_id",
              "away_team_id"]].copy()
    cand["ko"] = pd.to_datetime(cand["kickoff_utc"], utc=True)
    by_day = {k: v for k, v in cand.groupby("match_date")}
    links, learned, review = [], [], []
    for r in ev.itertuples(index=False):
        teams = split_event(r.event_name)
        day = by_day.get(r.uk_day)
        if teams is None or day is None or pd.isna(r.start):
            review.append({"betfair_event_id": r.event_id, "event_name": r.event_name, "country": r.country,
                           "market_start_time": r.market_time, "best_score": 0.0})
            continue
        day = day[day.country.isin(BF_COUNTRY.get(r.country, {r.country}))]
        near = day[day.ko.isna() | ((day.ko - r.start).abs() <= pd.Timedelta(hours=hours))]
        fixtures = list(zip(near.match_id, near.home_team, near.away_team))
        best, score = None, 0.0
        for country in sorted(set(near.country)):
            sub = [(k, h, a) for (k, h, a), c in zip(fixtures, near.country) if c == country]
            i, sc = pair_event(teams[0], teams[1], sub, country, "betfair", aliases)
            if i is not None and sc > score:
                best, score = (sub[i][0], country), sc
        if best is None:
            review.append({"betfair_event_id": r.event_id, "event_name": r.event_name, "country": r.country,
                           "market_start_time": r.market_time, "best_score": round(score, 3)})
            continue
        row = near[near.match_id == best[0]].iloc[0]
        links.append((r.event_id, best[0]))
        learned += [{"source": "betfair", "country": best[1], "source_name": teams[0], "team_id": row.home_team_id},
                    {"source": "betfair", "country": best[1], "source_name": teams[1], "team_id": row.away_team_id}]
    link = pd.DataFrame(links, columns=["event_id", "match_id"]).drop_duplicates("event_id")
    j = bf.merge(link, on="event_id")
    if j.empty:
        mp = pd.DataFrame(columns=cols)
    else:
        home_away = j["event_name"].map(lambda n: split_event(n) or ("", ""))
        rn = j["runner_name"].astype(str)
        outcome = np.where(rn.str.lower().eq("the draw"), "D",
                           np.where(rn == home_away.str[0], "H", np.where(rn == home_away.str[1], "A", "")))
        ou = j["market_type"].astype(str).str.startswith("OVER_UNDER")
        outcome = np.where(ou & rn.str.lower().str.startswith("over"), "O",
                           np.where(ou & rn.str.lower().str.startswith("under"), "U", outcome))
        mp = pd.DataFrame({"match_id": j["match_id"], "betfair_event_id": j["event_id"], "market_id": j["market_id"],
                           "market_type": j["market_type"], "market_start_time": j["market_time"],
                           "selection_id": j["selection_id"], "runner_name": rn, "outcome": outcome,
                           "handicap": j.get("handicap"), "won": j.get("won"), "ltp_first": j.get("ltp_first"),
                           "ltp_t_60": j.get("ltp_t_60"), "ltp_t_1": j.get("ltp_t_1"), "ltp_close": j.get("ltp_close"),
                           "bsp": j.get("bsp")})
    al = pd.DataFrame(learned, columns=["source", "country", "source_name", "team_id"]).drop_duplicates(
        ["source", "country", "source_name"])
    return mp, al, pd.DataFrame(review)


def betfair_closes(mp: pd.DataFrame) -> pd.DataFrame:
    """The market map -> one row a match: Betfair's own first and closing match-odds and over/under 2.5 prices and
    the settled 1X2 result (the check against fd's score)."""
    if mp.empty:
        return pd.DataFrame(columns=["match_id"])
    out = []
    for mt, outcomes in (("MATCH_ODDS", "HDA"), ("OVER_UNDER_25", "OU")):
        sub = mp[(mp.market_type == mt) & mp.outcome.isin(list(outcomes))]
        if sub.empty:
            continue
        for col in ("ltp_first", "ltp_t_60", "ltp_close"):
            w = sub.pivot_table(index="match_id", columns="outcome", values=col, aggfunc="first")
            w.columns = [f"bf_{col.replace('ltp_', '')}_{o.lower()}{'25' if mt != 'MATCH_ODDS' else ''}"
                         for o in w.columns]
            out.append(w)
        if mt == "MATCH_ODDS":
            win = sub[sub.won == 1].drop_duplicates("match_id").set_index("match_id")["outcome"].rename("bf_result")
            out.append(win.to_frame())
    return pd.concat(out, axis=1).reset_index() if out else pd.DataFrame(columns=["match_id"])


# --------------------------------------------------------------------------------------------------------------------
# The build
# --------------------------------------------------------------------------------------------------------------------

def read_fixtures(store) -> pd.DataFrame | None:
    """The newest fixtures files in bronze (``football/raw/fixtures/<day>/...``) as fd-style rows."""
    from sources.football import read_csv
    keys = sorted(k for k in store.listing("football/raw/fixtures/") if k.endswith(".csv"))
    if not keys:
        return None
    newest_day = keys[-1].split("/")[3]
    parts = []
    for k in keys:
        if k.split("/")[3] != newest_day:
            continue
        df = read_csv(store.get(k))
        if df.empty or "HomeTeam" not in df.columns or "Div" not in df.columns:
            continue                                     # the extra leagues' fixtures name no division
        parts.append(df)
    if not parts:
        return None
    f = pd.concat(parts, ignore_index=True, sort=False)
    f["match_date"] = pd.to_datetime(f["Date"], dayfirst=True, errors="coerce", format="mixed").dt.strftime("%Y-%m-%d")
    f["season"] = None
    return f


def build(store, today: str | None = None) -> dict:
    silver = store.get_parquet("football/matches.parquet")
    if silver is None or silver.empty:
        return {"matches": 0, "error": "no silver table: run source_data.py --source football --build first"}
    fixtures = read_fixtures(store)
    new = canonical(silver, fixtures)
    aliases = load_aliases(extra=store.get_parquet(f"{GOLD}/team_aliases.parquet"))
    apif = link_other(new, store.get_parquet("api_football/fixtures.parquet"), "api_football", aliases)
    new = apply_api_football(new, apif)
    gold, hist = version(store.get_parquet(f"{GOLD}/matches.parquet"), new)
    # Betfair Historic Data soccer: tie each market to its match, learn the names, keep the closes beside the match
    bf_keys = [k for k in store.listing("betfair_historic/") if re.search(r"markets_soccer_\d{4}\.parquet$", k)]
    bf = pd.concat([store.get_parquet(k) for k in sorted(bf_keys)], ignore_index=True) if bf_keys else None
    learned_prev = store.get_parquet(f"{GOLD}/team_aliases.parquet")
    mp, learned, review = betfair_map(gold, bf, aliases)
    if bf is not None:
        learned = pd.concat([learned_prev, learned]) if learned_prev is not None else learned
        learned = learned.drop_duplicates(["source", "country", "source_name"], keep="last")
        store.put_parquet(f"{GOLD}/betfair_market_map.parquet", mp)
        store.put_parquet(f"{GOLD}/team_aliases.parquet", learned)
        store.put_parquet(f"{GOLD}/betfair_review.parquet", review)
    closes = betfair_closes(mp)
    gold = gold.drop(columns=[c for c in gold.columns if c.startswith("bf_")], errors="ignore")
    if len(closes.columns) > 1:
        gold = gold.merge(closes, on="match_id", how="left")
        gold["betfair_event_id"] = gold["match_id"].map(mp.drop_duplicates("match_id").set_index(
            "match_id")["betfair_event_id"])
    gold = gold.sort_values(["match_date", "competition_id", "match_id"]).reset_index(drop=True)
    store.put_parquet(f"{GOLD}/matches.parquet", gold)
    if len(hist):
        old = store.get_parquet(f"{GOLD}/matches_history.parquet")
        store.put_parquet(f"{GOLD}/matches_history.parquet", pd.concat([old, hist]) if old is not None else hist)
    ol = odds_long(_with_ids(silver, gold))
    store.put_parquet(f"{GOLD}/odds_long.parquet", ol)
    # data quality, with the cross-checks against the other sources' tables when they are built
    issues = [dq_checks(gold, today)]
    opf = link_other(gold, store.get_parquet("openfootball/matches.parquet"), "openfootball", aliases)
    for src, linked in (("openfootball", opf), ("api_football", apif)):
        issues.append(cross_check(gold, linked, src))
    if "bf_result" in gold.columns:
        fin = gold[(gold.status == "FINISHED") & gold.bf_result.notna()]
        res = np.sign(fin.ft_home - fin.ft_away).map({1: "H", 0: "D", -1: "A"})
        bad = fin[res != fin.bf_result]
        issues.append(pd.DataFrame({"check_name": "score_vs_betfair_settlement", "match_id": bad.match_id,
                                    "details": [f"fd {a}-{b}, Betfair settled {r}" for a, b, r in
                                                zip(bad.ft_home, bad.ft_away, bad.bf_result)], "severity": "high"}))
    dq = pd.concat(issues, ignore_index=True)
    run = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")
    summary = {"run": run, "matches": int(len(gold)),
               "finished": int((gold.status == "FINISHED").sum()), "scheduled": int((gold.status == "SCHEDULED").sum()),
               "changed": int(len(hist)), "odds_rows": int(len(ol)),
               "betfair_markets_linked": int(mp.market_id.nunique()) if len(mp) else 0,
               "betfair_matches_linked": int(mp.match_id.nunique()) if len(mp) else 0,
               "betfair_events_for_review": int(len(review)),
               "openfootball_linked": int(len(opf)), "api_football_linked": int(len(apif)),
               "issues": dq.groupby(["check_name", "severity"]).size().rename("n").reset_index().to_dict("records")}
    store.put(f"football/dq/{run}.json", json.dumps({"summary": summary,
                                                     "issues": dq.head(5000).to_dict("records")},
                                                    default=str).encode())
    store.put("football/dq/latest.json", json.dumps(summary, default=str).encode())
    log.info("football gold: %s", summary)
    return summary


def _with_ids(silver: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """The silver rows with their canonical match_id (joined on competition, date and the two team ids)."""
    s = silver[silver["Div"].isin(BY_FD)].copy()
    s["competition_id"] = s["Div"].map(lambda d: BY_FD[d].competition_id)
    s["country"] = s["Div"].map(lambda d: BY_FD[d].country)
    s["home_team_id"] = [team_id(c, str(n).strip()) for c, n in zip(s["country"], s["HomeTeam"])]
    s["away_team_id"] = [team_id(c, str(n).strip()) for c, n in zip(s["country"], s["AwayTeam"])]
    key = ["competition_id", "match_date", "home_team_id", "away_team_id"]
    return s.merge(gold[key + ["match_id"]].drop_duplicates(key), on=key, how="inner")


def load_gold(store) -> pd.DataFrame:
    g = store.get_parquet(f"{GOLD}/matches.parquet")
    if g is None:
        raise SystemExit("no gold table: run python -m football.data --build first")
    return g


def main(argv=None) -> int:
    import argparse
    import sys

    from sources.common import Store
    ap = argparse.ArgumentParser(description="Build the football gold tables and their data-quality checks")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--root", default=None, help="a local sources folder in place of S3")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    store = Store(root=a.root)
    if a.build:
        print(json.dumps(build(store), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
