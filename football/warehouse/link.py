"""The link stage: Betfair's events and teams tied to API-Football's fixtures and teams and to our canonical matches.

Betfair's Historic Data names a match and its two sides and nothing else: no competition, no season, no line-up.
API-Football holds those for every fixture worldwide (``api_football/fixtures.parquet``) and, for the leagues our
markets trade in, line-ups, goal minutes and xG (``api_football/history_*.parquet``). This stage writes the bridge:

- ``gold/link``: one row a Betfair event: its API-Football fixture, league and season, both sides' API-Football
  team ids, how the link was made and how sure it is, and our canonical ``match_id`` and ``competition_id`` where
  the fixture is one of the 38 football-data competitions;
- ``gold/team_link``: one row a Betfair team id: the API-Football team it most often links to (a majority vote
  over its linked matches), the share of them that agree, and our canonical team id where there is one;
- ``gold/goal_check``: one row a linked match: the goals inferred from prices (``gold/goal``) against
  API-Football's goal events, by count and by minute.

How an event finds its fixture (``link_events``), strongest evidence first, a fixture never given to two events:

1. *exact*: both sides' names equal after ``football.reference.normalise`` (the squad marker made one: '(W)',
   'Women' and 'Ladies' are the same squad, 'B' and 'II' are), kick-offs within ``EXACT_TOL_MIN``. A reviewed alias
   (``football/team_aliases.csv``) counts as the name (*alias*).
2. *team_map*: both Betfair team ids already map to API-Football team ids (``team_map``: the vote over the links
   made so far) and that pair of teams has a fixture at that kick-off. *team_anchor*: one side maps, with at least
   ``ANCHOR_MIN_AGREE`` matches behind it, and that team has exactly one fixture within ``TOL_MIN`` of the
   kick-off: a team plays one match at a time, so the fixture is the event's whatever the other side is called
   ('AGF' against 'Aarhus'). Each round's links widen the map; the rounds repeat until nothing new links.
3. *fuzzy*: among the fixtures kicking off within ``TOL_MIN``, the one whose two sides both score at least
   ``FUZZY_MIN`` on ``football.reference.similarity``, and only when no other fixture comes within
   ``FUZZY_MARGIN``. Sides of different squads never match (a women's or under-21 side against the senior one).
   Then the team rounds again.

A link by anything but the names being equal is dropped when both sources hold a result and they disagree.

The tolerances are measured (11 Nov 2026, the owner's 2026 Betfair download against API-Football's 22 main
divisions): of 4,246 exact-name pairs within half a day, 3,961 were within 1 minute, 4,241 within 30, the other 5
within 90 (late kick-offs Betfair's market time followed), none between 90 minutes and 12 hours, and every one
whose result both sources hold agreed. So 30 minutes where the names are not proof, 180 where they are.

Not a gold module run by ``gold.build``: the API-Football and canonical tables live in the capture bucket, so the
stage runs where they are (a workflow), reads ``gold/event`` and ``gold/team`` from the warehouse and writes single
files beside them:

    python -m football.warehouse.link run [--root <local folder>] [--dry-run]
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from football.reference import COMPETITIONS, load_aliases, normalise, similarity, strip_accents
from football.warehouse.schema import TS, Column, TableSpec

log = logging.getLogger(__name__)

VERSION = 1
TOL_MIN = 30.0            # kick-off tolerance where the names are not proof (fuzzy, one mapped side)
EXACT_TOL_MIN = 180.0     # ... where they are (both names equal, or both team ids mapped)
FUZZY_MIN = 0.85          # the weaker side's similarity a fuzzy link needs
FUZZY_MARGIN = 0.05       # ... and its lead over the next fixture
MIN_SHARE = 0.8           # a team id's map is trusted when this share of its linked matches agree
ANCHOR_MIN_AGREE = 3      # matches behind a map before it may place an event on its own
MAX_ROUNDS = 8
METHODS = ("exact", "alias", "team_map", "team_anchor", "fuzzy")

#: the store keys read (relative to ``sources/`` in the capture bucket)
FIXTURES_KEY = "api_football/fixtures.parquet"
HISTORY_MATCHES_KEY = "api_football/history_matches.parquet"
HISTORY_EVENTS_KEY = "api_football/history_events.parquet"
CANONICAL_KEY = "football/gold/matches.parquet"

#: goal check: API-Football's goal details that count, and the clock's allowances (minutes)
GOAL_DETAILS = ("Normal Goal", "Penalty", "Own Goal")
HT_WALL_MIN = 60.0        # the second half cannot have begun before this many wall-clock minutes after kick-off
HT_LAG = (15.0, 25.0)     # wall clock less match clock in the second half: the break plus first-half stoppage
KO_SLACK_MIN = 5.0        # added to both allowances when only the scheduled kick-off is known (a late start)
MINUTE_TOL = 2.0          # a matched goal's minute agrees within this
GAP_COST = 8.0            # the alignment leaves a goal unmatched rather than pair it more than 2 x this away
SIDE_COST = 3.0           # ... and prefers a pairing whose scoring side agrees

LINK = TableSpec(
    "gold/link",
    "One row a Betfair event: the API-Football fixture it is, how the link was made, and our canonical match. "
    "Competition, season, line-ups, goal minutes and xG join to the prices through it. Events that found no "
    "fixture are kept, with method 'none'.",
    (
        Column("event_id", pa.int64(), "Betfair's event id.", False),
        Column("linked", pa.bool_(), "Whether a fixture was found.", False),
        Column("method", pa.string(), "How: exact (both names equal after normalisation), alias (a reviewed "
               "alias), team_map (both Betfair team ids mapped to the fixture's teams), team_anchor (one side "
               "mapped, the team's only fixture at that kick-off), fuzzy (both names similar, unique), none.",
               False),
        Column("score", pa.float64(), "How sure, 0 to 1: 1 for exact and alias; the weaker side's name similarity "
               "for fuzzy; the weaker mapped side's vote share for team_map and team_anchor."),
        Column("apif_fixture_id", pa.int64(), "API-Football's fixture id."),
        Column("apif_league_id", pa.int32(), "API-Football's league id: the competition."),
        Column("apif_league", pa.string(), "The league's name (where the fixtures table gives it)."),
        Column("apif_league_country", pa.string(), "The league's country or region as API-Football names it."),
        Column("season", pa.int16(), "API-Football's season: the year it starts."),
        Column("apif_home_team_id", pa.int64(), "API-Football's team id of the side Betfair names first (the "
               "event's home side, whichever way round the fixture lists them)."),
        Column("apif_away_team_id", pa.int64(), "API-Football's team id of the side Betfair names second."),
        Column("apif_home_name", pa.string(), "API-Football's name for the event's home side."),
        Column("apif_away_name", pa.string(), "API-Football's name for the event's away side."),
        Column("apif_kickoff", TS, "The fixture's kick-off."),
        Column("kickoff_diff_min", pa.float64(), "Betfair's scheduled kick-off less the fixture's, minutes."),
        Column("swapped", pa.bool_(), "True when the fixture lists the sides the other way round (its home side "
               "is Betfair's away side): flip API-Football's home/away tables before joining."),
        Column("apif_ft_home", pa.int16(), "API-Football's goals after 90 minutes for the event's home side."),
        Column("apif_ft_away", pa.int16(), "... for the event's away side."),
        Column("result_agrees", pa.bool_(), "Whether that score gives the result Betfair settled (blank when "
               "either is missing). False on an exact link marks a data error to review; other methods never "
               "keep one."),
        Column("match_id", pa.string(), "Our canonical match id (football.data.make_match_id) where the fixture "
               "is one of the 38 football-data competitions' matches."),
        Column("competition_id", pa.string(), "Our competition id (ENG1 ...): the canonical match's, else the "
               "one API-Football's league id maps to (a play-off football-data does not hold)."),
    ),
    key=("event_id",), sort=("event_id",),
)

TEAM_LINK = TableSpec(
    "gold/team_link",
    "One row a Betfair team id: the API-Football team it most often links to, by a vote over its linked matches, "
    "and our canonical team id where its matches are canonical ones. Teams with no linked match are kept, blank.",
    (
        Column("team_id", pa.int64(), "Betfair's selection id for the team (gold/team).", False),
        Column("team_name", pa.string(), "The name Betfair most often gives it."),
        Column("apif_team_id", pa.int64(), "The API-Football team id most of its linked matches give."),
        Column("apif_team_name", pa.string(), "API-Football's name for that team."),
        Column("n_linked", pa.int32(), "Its matches linked to a fixture.", False),
        Column("n_agree", pa.int32(), "Of those, the matches that give apif_team_id.", False),
        Column("share", pa.float64(), "n_agree / n_linked."),
        Column("n_apif_ids", pa.int32(), "Distinct API-Football team ids its matches gave (more than one: Betfair "
               "gives one id to two clubs of the same name, or a wrong link).", False),
        Column("low_agreement", pa.bool_(), f"The share is under {MIN_SHARE}: the map is not used to link events "
               "and should not be used to join team tables.", False),
        Column("canonical_team_id", pa.string(), "Our team id ('ENG:arsenal') most of its canonical matches give."),
        Column("canonical_share", pa.float64(), "The share of its canonical matches that give it."),
    ),
    key=("team_id",), sort=("team_id",),
)

GOAL_CHECK = TableSpec(
    "gold/goal_check",
    "One row a linked match that went in play and has API-Football's score or goal events: the goals inferred from "
    "prices (gold/goal) against them. Counts against the 90-minute score; minutes against the goal events, the "
    "half-time break allowed for (API-Football's minute is the match clock, an inferred goal's is the wall clock).",
    (
        Column("event_id", pa.int64(), "Betfair's event id.", False),
        Column("apif_fixture_id", pa.int64(), "API-Football's fixture id.", False),
        Column("clock", pa.string(), "What the wall clock was measured from: 'actual' (the market's turn in play) "
               "or 'fixture' (the fixture's scheduled kick-off: a late start is allowed for).", False),
        Column("has_events", pa.bool_(), "Whether API-Football's events are held for the fixture.", False),
        Column("n_inferred", pa.int16(), "Goals inferred from prices.", False),
        Column("n_apif", pa.int16(), "API-Football's goal events (normal, penalty, own goal) within the 90 "
               "minutes; blank without events."),
        Column("n_ft", pa.int16(), "Goals in API-Football's 90-minute score; blank without one."),
        Column("count_diff", pa.int16(), "n_inferred less the true count (n_ft, else n_apif).", False),
        Column("count_agrees", pa.bool_(), "count_diff is 0.", False),
        Column("n_matched", pa.int16(), "Inferred goals paired with a goal event (in order, within "
               f"{2 * GAP_COST:.0f} minutes).", False),
        Column("n_within_tol", pa.int16(), f"Of those, pairs whose minute agrees within {MINUTE_TOL:.0f}.", False),
        Column("n_side_known", pa.int16(), "Pairs where both sources name the scoring side.", False),
        Column("n_side_agree", pa.int16(), "Of those, pairs that name the same side.", False),
        Column("minute_mae", pa.float64(), "Mean absolute minute error of the pairs."),
        Column("minute_err_max", pa.float64(), "The largest absolute minute error."),
        Column("ht_lag_min", pa.float64(), "Median of wall clock less match clock over the second-half pairs: the "
               "half-time break plus first-half stoppage this match shows."),
    ),
    key=("event_id",), sort=("event_id",),
)
SPECS = (LINK, TEAM_LINK, GOAL_CHECK)

_BF_COUNTRY: dict[str, set[str]] = {}
for _comp in COMPETITIONS:
    _BF_COUNTRY.setdefault(_comp.betfair_country, set()).add(_comp.country)
_LEAGUE_COUNTRY = {c.apif_league_id: c.country for c in COMPETITIONS if c.apif_league_id}
_LEAGUE_COMPETITION = {c.apif_league_id: c.competition_id for c in COMPETITIONS if c.apif_league_id}


# --------------------------------------------------------------------------------------------------------------------
# names: football.reference's normaliser and similarity, plus the squad a name belongs to
# --------------------------------------------------------------------------------------------------------------------

_SQUAD = {"w": "w", "women": "w", "womens": "w", "ladies": "w", "fem": "w", "femenino": "w", "feminino": "w",
          "femenil": "w", "femminile": "w", "frauen": "w", "b": "b", "ii": "b", "2": "b", "iii": "c", "3": "c",
          "res": "res", "reserve": "res", "reserves": "res", "youth": "youth"}
_AGE = re.compile(r"u\d\d$")


@lru_cache(maxsize=None)
def name_key(name: str) -> str:
    """A side's name as the link compares it: '<normalised club name>|<squad>'. The squad is what the trailing
    marker says ('Arsenal (W)' -> 'arsenal|w', 'Sociedad B' and 'Real Sociedad II' -> '...|b', 'Ajax U21' ->
    'ajax|u21', 'Jong Ajax' -> 'ajax|b'), blank for a first team. A name ``normalise`` empties ('AIK': a suffix
    in its list) keeps its plain lower-cased form."""
    raw = "" if name is None or name != name else str(name)
    n = normalise(raw) or re.sub(r"[^a-z0-9 ]+", " ", strip_accents(raw).lower()).strip()
    toks, squad = n.split(), []
    while len(toks) > 1:
        last = toks[-1]
        if len(toks) > 2 and last.isdigit() and len(last) == 2 and toks[-2] in ("u", "sub", "under"):
            squad.append("u" + last)
            toks = toks[:-2]
        elif last in _SQUAD or _AGE.match(last):
            squad.append(_SQUAD.get(last, last))
            toks = toks[:-1]
        else:
            break
    if len(toks) > 1 and toks[0] == "jong":
        squad.append("b")
        toks = toks[1:]
    return " ".join(toks) + "|" + ",".join(sorted(set(squad)))


def _no_lone(core: str) -> str:
    return " ".join(t for t in core.split() if len(t) > 1 or t.isdigit())


def _related(ta: list[str], tb: list[str]) -> bool:
    return any(u.startswith(v) or v.startswith(u) for u in ta for v in tb)


def side_score(a: str, b: str, floor: float = 0.0) -> float:
    """How alike two name keys are, 0 to 1: 0 for different squads; else ``football.reference.similarity`` of the
    club names, and no more than their similarity with lone letters left out (its prefix rule lets a lone letter
    stand for any word: 'Arsenal (W)' against 'Wigan', 'A.C. Trento S.C.S.D.' against 'Cagliari', both measured
    false links). A score that cannot reach ``floor`` is returned as 0 without the costly comparison."""
    if a == b:
        return 1.0 if a.split("|", 1)[0] else 0.0
    ca, sa = a.split("|", 1)
    cb, sb = b.split("|", 1)
    if sa != sb or not ca or not cb:
        return 0.0
    if ca == cb:
        return 1.0
    if floor and 2 * min(len(ca), len(cb)) < floor * (len(ca) + len(cb)) and not _related(ca.split(), cb.split()):
        return 0.0                                   # no shared word start, and too unequal for the character ratio
    s = similarity(ca, cb)
    la, lb = _no_lone(ca), _no_lone(cb)
    if s and la and lb and (la != ca or lb != cb):
        s = min(s, similarity(la, lb))
    return s


def _forms(name, source: str, countries, aliases) -> tuple[tuple[str, ...], bool]:
    """(the keys a side answers to, whether an alias added one): its own name, and the canonical name a reviewed
    alias gives it in one of ``countries``."""
    key = name_key(name)
    out = [key]
    if aliases and countries:
        n = normalise("" if name is None or name != name else str(name))
        for c in sorted(countries):
            tid = aliases.get((source, c, n))
            if tid and ":" in tid:
                k = name_key(tid.split(":", 1)[1])
                if k not in out:
                    out.append(k)
    return tuple(out), len(out) > 1


# --------------------------------------------------------------------------------------------------------------------
# the inputs as arrays
# --------------------------------------------------------------------------------------------------------------------

def _minutes(s) -> np.ndarray:
    """A UTC time column as minutes since 1970 (blank: NaN), whatever its unit."""
    t = pd.to_datetime(pd.Series(s).reset_index(drop=True), utc=True, errors="coerce")
    return ((t - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(minutes=1)).to_numpy(dtype=float)


def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def fixture_table(fixtures: pd.DataFrame | None = None, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """API-Football's fixtures as the link reads them, one row a fixture id: ``api_football/fixtures.parquet``
    (every competition) with ``history_matches.parquet`` laid over it (a finished match as it was fetched with its
    events). Columns: apif_fixture_id, kickoff, home, away, apif_home_id, apif_away_id, apif_league_id, apif_season,
    ft_home, ft_away, league, league_country."""
    cols = ["apif_fixture_id", "kickoff", "home", "away", "apif_home_id", "apif_away_id", "apif_league_id",
            "apif_season", "ft_home", "ft_away", "league", "league_country"]
    parts = []
    for df in (fixtures, history):
        if df is None or not len(df):
            continue
        d = df.rename(columns={"kickoff_utc": "kickoff"})
        d = d.reindex(columns=cols)
        d["kickoff"] = pd.to_datetime(d["kickoff"], utc=True, errors="coerce")
        parts.append(d)
    if not parts:
        return pd.DataFrame(columns=cols)
    out = pd.concat(parts, ignore_index=True)
    out = out[out["apif_fixture_id"].notna() & out["kickoff"].notna()]
    if len(parts) == 2:                              # the history row wins, the fixtures row fills what it lacks
        out = out.groupby("apif_fixture_id", sort=True, as_index=False).last()
    else:
        out = out.drop_duplicates("apif_fixture_id", keep="last")
    return out.sort_values(["kickoff", "apif_fixture_id"]).reset_index(drop=True)


class _Side:
    """Events or fixtures as parallel arrays: kick-off minutes, each side's name keys, team ids, result sign."""

    def __init__(self, df: pd.DataFrame, source: str, aliases, home: str, away: str, home_id: str, away_id: str,
                 countries):
        self.n = len(df)
        self.t = _minutes(df["kickoff"]) if self.n else np.zeros(0)
        hf = [_forms(nm, source, c, aliases) for nm, c in zip(df[home], countries)]
        af = [_forms(nm, source, c, aliases) for nm, c in zip(df[away], countries)]
        self.hf, self.af = [x[0] for x in hf], [x[0] for x in af]
        self.h_alias = np.array([x[1] for x in hf], dtype=bool)
        self.a_alias = np.array([x[1] for x in af], dtype=bool)
        self.named = np.array([bool(h[0].split("|")[0]) and bool(a[0].split("|")[0])
                               for h, a in zip(self.hf, self.af)], dtype=bool)
        self.hid, self.aid = _num(df, home_id), _num(df, away_id)

    def long(self, idx: np.ndarray, col: str) -> pd.DataFrame:
        """One row a (row, home key, away key) the side answers to: the exact pass's join keys."""
        rows = [(i, h, a, h != self.hf[i][0] or a != self.af[i][0])
                for i in idx for h in self.hf[i] for a in self.af[i]]
        return pd.DataFrame(rows, columns=[col, "kh", "ka", f"alias_{col}"])


def _ix(c: pd.DataFrame, col: str) -> np.ndarray:
    """A candidate frame's row numbers as an index array (an empty frame's column has no integer type)."""
    return c[col].to_numpy(dtype=np.int64)


def _result_sign(result) -> np.ndarray:
    return pd.Series(result).map({"H": 1.0, "D": 0.0, "A": -1.0}).to_numpy(dtype=float)


def _agrees(res_e: np.ndarray, gd_f: np.ndarray, swapped: np.ndarray) -> np.ndarray:
    """1.0 where the fixture's score gives the event's result, 0.0 where it does not, NaN where either is blank."""
    sign = np.sign(gd_f) * np.where(swapped, -1.0, 1.0)
    return np.where(np.isnan(res_e) | np.isnan(gd_f), np.nan, (res_e == sign).astype(float))


# --------------------------------------------------------------------------------------------------------------------
# the passes
# --------------------------------------------------------------------------------------------------------------------

class _State:
    """The links so far: each event's fixture (-1: none), which way round, how, how sure."""

    def __init__(self, n_events: int, n_fixtures: int):
        self.f = np.full(n_events, -1, dtype=np.int64)
        self.swapped = np.zeros(n_events, dtype=bool)
        self.method = np.full(n_events, "none", dtype=object)
        self.score = np.full(n_events, np.nan)
        self.taken = np.zeros(n_fixtures, dtype=bool)

    def assign(self, cand: pd.DataFrame, method=None) -> int:
        """Give each event of ``cand`` (e, f, swapped, score, gap, bad, [method]) its best free fixture: best
        score first, then the result not disagreeing, the sides the same way round, the nearer kick-off, the
        lower ids. One event a fixture, one fixture an event."""
        if not len(cand):
            return 0
        c = cand.assign(absgap=cand["gap"].abs()).sort_values(
            ["score", "bad", "swapped", "absgap", "e", "f"], ascending=[False, True, True, True, True, True])
        methods = c["method"].to_numpy() if "method" in c.columns else None
        made = 0
        for i, (e, f, sw, sc) in enumerate(zip(_ix(c, "e"), _ix(c, "f"), c["swapped"].to_numpy(dtype=bool),
                                               c["score"].to_numpy(dtype=float))):
            if self.f[e] >= 0 or self.taken[f]:
                continue
            self.f[e], self.swapped[e], self.score[e], self.taken[f] = f, bool(sw), float(sc), True
            self.method[e] = methods[i] if methods is not None else method
            made += 1
        return made


def _exact(E: _Side, F: _Side, st: _State, res_e, gd_f, tol: float) -> int:
    free_e = np.flatnonzero((st.f < 0) & E.named & ~np.isnan(E.t))
    free_f = np.flatnonzero(~st.taken & F.named)
    el, fl = E.long(free_e, "e"), F.long(free_f, "f")
    if not len(el) or not len(fl):
        return 0
    straight = el.merge(fl, on=["kh", "ka"]).assign(swapped=False)
    flipped = el.merge(fl.rename(columns={"kh": "ka", "ka": "kh"}), on=["kh", "ka"]).assign(swapped=True)
    c = pd.concat([straight, flipped], ignore_index=True)
    if not len(c):
        return 0
    c["gap"] = E.t[_ix(c, "e")] - F.t[_ix(c, "f")]
    c = c[c["gap"].abs() <= tol].copy()
    alias = (c["alias_e"] | c["alias_f"]).to_numpy(dtype=bool)
    c["method"] = np.where(alias, "alias", "exact")
    c["score"] = 1.0
    agree = _agrees(res_e[_ix(c, "e")], gd_f[_ix(c, "f")], c["swapped"].to_numpy(dtype=bool))
    c["bad"] = (agree == 0.0) | alias                # among equals: a result that agrees first, a plain name first
    return st.assign(c)


def _fuzzy(E: _Side, F: _Side, st: _State, res_e, gd_f, tol: float, floor: float, margin: float) -> int:
    """Blocked by kick-off: each unlinked event against the free fixtures within ``tol`` of it."""
    pool = np.flatnonzero(~st.taken & F.named)       # in kick-off order: the fixtures are sorted by it
    ft = F.t[pool]
    cache: dict[tuple[str, str], float] = {}

    def side(xs: tuple[str, ...], ys: tuple[str, ...]) -> float:
        best = 0.0
        for x in xs:
            for y in ys:
                s = cache.get((x, y))
                if s is None:
                    s = cache[(x, y)] = side_score(x, y, floor)
                if s > best:
                    best = s
        return best

    rows = []
    for e in np.flatnonzero((st.f < 0) & E.named & ~np.isnan(E.t)):
        lo, hi = np.searchsorted(ft, E.t[e] - tol, "left"), np.searchsorted(ft, E.t[e] + tol, "right")
        eh, ea = E.hf[e], E.af[e]
        found = []                                   # (score, swapped, fixture)
        for f in pool[lo:hi]:
            fh, fa = F.hf[f], F.af[f]
            s = side(eh, fh)
            if s >= floor:
                s = min(s, side(ea, fa))
                if s >= floor:
                    found.append((s, False, f))
            s = side(eh, fa)
            if s >= floor:
                s = min(s, side(ea, fh))
                if s >= floor:
                    found.append((s, True, f))
        if not found:
            continue
        found.sort(key=lambda x: (-x[0], x[1], x[2]))
        s, sw, f = found[0]
        if len(found) > 1 and s - found[1][0] < margin:
            continue                                 # two fixtures fit: unique or nothing
        rows.append((e, f, sw, s, E.t[e] - F.t[f]))
    c = pd.DataFrame(rows, columns=["e", "f", "swapped", "score", "gap"])
    if not len(c):
        return 0
    agree = _agrees(res_e[_ix(c, "e")], gd_f[_ix(c, "f")], c["swapped"].to_numpy(dtype=bool))
    c = c[agree != 0.0].assign(bad=False)
    return st.assign(c, "fuzzy")


def _vote(pairs: pd.DataFrame, key: str, value: str) -> pd.DataFrame:
    """The majority vote: one row a ``key`` with the ``value`` most of its rows give (ties: the lowest), the rows
    (n_linked), the rows that agree (n_agree), their share, and the distinct values seen."""
    cols = [key, value, "n_linked", "n_agree", "share", "n_values"]
    p = pairs[pairs[key].notna() & pairs[value].notna()]
    if not len(p):
        return pd.DataFrame(columns=cols)
    cnt = p.groupby([key, value], sort=True).size().rename("n_agree").reset_index()
    tot = cnt.groupby(key)["n_agree"].agg(n_linked="sum", n_values="size").reset_index()
    top = cnt.sort_values([key, "n_agree", value], ascending=[True, False, True]).drop_duplicates(key)
    out = top.merge(tot, on=key)
    out["share"] = out["n_agree"] / out["n_linked"]
    return out[cols].sort_values(key).reset_index(drop=True)


def team_map(links: pd.DataFrame, events: pd.DataFrame, min_share: float = MIN_SHARE) -> pd.DataFrame:
    """One row a Betfair team id with a linked match: the API-Football team id most of its linked matches give
    (``apif_team_id``), the matches linked (``n_linked``), those that agree (``n_agree``), their ``share``, the
    distinct API-Football ids seen (``n_apif_ids``) and ``low_agreement`` (share under ``min_share``).
    ``links``: gold/link rows (the API-Football team ids already the event's way round); ``events``: gold/event
    rows (event_id, home_id, away_id)."""
    cols = ["team_id", "apif_team_id", "n_linked", "n_agree", "share", "n_apif_ids", "low_agreement"]
    lk = links[links["apif_fixture_id"].notna()][["event_id", "apif_home_team_id", "apif_away_team_id"]]
    j = lk.merge(events[["event_id", "home_id", "away_id"]], on="event_id")
    pairs = pd.concat([
        pd.DataFrame({"team_id": j["home_id"].to_numpy(), "apif_team_id": j["apif_home_team_id"].to_numpy()}),
        pd.DataFrame({"team_id": j["away_id"].to_numpy(), "apif_team_id": j["apif_away_team_id"].to_numpy()})],
        ignore_index=True)
    out = _vote(pairs, "team_id", "apif_team_id").rename(columns={"n_values": "n_apif_ids"})
    out["low_agreement"] = out["share"] < min_share
    for c in ("team_id", "apif_team_id", "n_linked", "n_agree", "n_apif_ids"):
        out[c] = out[c].astype("int64")
    return out[cols]


def _state_map(E: _Side, F: _Side, st: _State, min_share: float) -> pd.DataFrame:
    """``team_map`` over the links made so far."""
    e = np.flatnonzero(st.f >= 0)
    f, sw = st.f[e], st.swapped[e]
    links = pd.DataFrame({"event_id": e, "apif_fixture_id": f,
                          "apif_home_team_id": np.where(sw, F.aid[f], F.hid[f]),
                          "apif_away_team_id": np.where(sw, F.hid[f], F.aid[f])})
    events = pd.DataFrame({"event_id": np.arange(E.n), "home_id": E.hid, "away_id": E.aid})
    return team_map(links, events, min_share)


def _teams(E: _Side, F: _Side, st: _State, tm: pd.DataFrame, res_e, gd_f, tol: float, exact_tol: float,
           min_agree: int, floor: float) -> int:
    """The second pass: events whose Betfair team ids the map knows. Both sides mapped and that pair's fixture
    within ``exact_tol`` (team_map); else one side mapped, by at least ``min_agree`` matches, and that team's only
    free fixture within ``tol`` (team_anchor), unless the other side is mapped to another team and its name does
    not resemble the fixture's."""
    good = tm[~tm["low_agreement"]]
    if not len(good):
        return 0
    to_apif = dict(zip(good["team_id"], good["apif_team_id"]))
    share = dict(zip(good["team_id"], good["share"]))
    n_agree = dict(zip(good["team_id"], good["n_agree"]))
    free_e = np.flatnonzero((st.f < 0) & ~np.isnan(E.t))
    ev = pd.DataFrame({"e": free_e, "mh": pd.Series(E.hid[free_e]).map(to_apif).to_numpy(dtype=float),
                       "ma": pd.Series(E.aid[free_e]).map(to_apif).to_numpy(dtype=float),
                       "sh": pd.Series(E.hid[free_e]).map(share).to_numpy(dtype=float),
                       "sa": pd.Series(E.aid[free_e]).map(share).to_numpy(dtype=float),
                       "nh": pd.Series(E.hid[free_e]).map(n_agree).to_numpy(dtype=float),
                       "na": pd.Series(E.aid[free_e]).map(n_agree).to_numpy(dtype=float)})
    free_f = np.flatnonzero(~st.taken)
    fx = pd.DataFrame({"f": free_f, "fh": F.hid[free_f], "fa": F.aid[free_f]}).dropna()

    def finish(c: pd.DataFrame, limit: float) -> pd.DataFrame:
        c = c.assign(gap=E.t[_ix(c, "e")] - F.t[_ix(c, "f")])
        c = c[c["gap"].abs() <= limit]
        agree = _agrees(res_e[_ix(c, "e")], gd_f[_ix(c, "f")], c["swapped"].to_numpy(dtype=bool))
        return c[agree != 0.0].assign(bad=False)

    both = ev.dropna(subset=["mh", "ma"])
    c = pd.concat([both.merge(fx, left_on=["mh", "ma"], right_on=["fh", "fa"]).assign(swapped=False),
                   both.merge(fx, left_on=["mh", "ma"], right_on=["fa", "fh"]).assign(swapped=True)],
                  ignore_index=True)
    c["score"] = c[["sh", "sa"]].min(axis=1)
    made = st.assign(finish(c, exact_tol), "team_map")

    ev = ev[st.f[_ix(ev, "e")] < 0]
    fx = fx[~st.taken[_ix(fx, "f")]]
    parts = []
    for mine, n, s, other_is in (("mh", "nh", "sh", "away"), ("ma", "na", "sa", "home")):
        a = ev[ev[mine].notna() & (ev[n] >= min_agree)]
        for fcol in ("fh", "fa"):
            # the anchor is the event's home side: found as the fixture's home (straight) or away (swapped)
            swapped = (fcol == "fa") if mine == "mh" else (fcol == "fh")
            parts.append(a.merge(fx, left_on=mine, right_on=fcol).assign(swapped=swapped, score=lambda d: d[s],
                                                                         other=other_is))
    c = finish(pd.concat(parts, ignore_index=True), tol)
    if len(c):
        # the other side: where the map says another team, its name must still resemble the fixture's
        keep = np.ones(len(c), dtype=bool)
        for i, (e, f, sw, other, mh, ma, fh, fa) in enumerate(zip(c["e"], c["f"], c["swapped"], c["other"], c["mh"],
                                                                  c["ma"], c["fh"], c["fa"])):
            mapped = ma if other == "away" else mh
            theirs = (fh if sw else fa) if other == "away" else (fa if sw else fh)
            if mapped == mapped and mapped != theirs:
                mine_keys = E.af[e] if other == "away" else E.hf[e]
                their_keys = (F.hf[f] if sw else F.af[f]) if other == "away" else (F.af[f] if sw else F.hf[f])
                keep[i] = max(side_score(x, y) for x in mine_keys for y in their_keys) >= floor
        c = c[keep].drop_duplicates(["e", "f"])
        c = c[~c.duplicated("e", keep=False)]           # the team's only fixture at that kick-off, or nothing
    return made + st.assign(c, "team_anchor")


def link_events(events: pd.DataFrame, fixtures: pd.DataFrame, aliases: dict | None = None,
                tol_min: float = TOL_MIN, exact_tol_min: float = EXACT_TOL_MIN, fuzzy_min: float = FUZZY_MIN,
                fuzzy_margin: float = FUZZY_MARGIN, min_share: float = MIN_SHARE,
                anchor_min_agree: int = ANCHOR_MIN_AGREE, team_pass: bool = True) -> pd.DataFrame:
    """One row a Betfair event (gold/link's columns but match_id and competition_id): the API-Football fixture it
    is, or method 'none'. ``events``: gold/event rows (event_id, home_name, away_name, kickoff, and where held
    country_code, home_id, away_id, result). ``fixtures``: ``fixture_table``'s rows. ``aliases``:
    ``football.reference.load_aliases()``. ``team_pass`` False stops after the names (exact, alias, fuzzy)."""
    ev = events.sort_values("event_id").reset_index(drop=True)
    if not ev["event_id"].is_unique:
        raise ValueError("link_events: event_id repeats")
    fx = fixture_table(fixtures)
    k = _minutes(ev["kickoff"]) if len(ev) else np.zeros(0)
    if len(ev) and len(fx) and not np.isnan(k).all():   # only the fixtures an event could reach
        tf = _minutes(fx["kickoff"])
        fx = fx[(tf >= np.nanmin(k) - exact_tol_min) & (tf <= np.nanmax(k) + exact_tol_min)].reset_index(drop=True)
    cc = ev["country_code"] if "country_code" in ev.columns else pd.Series([None] * len(ev))
    E = _Side(ev, "betfair", aliases, "home_name", "away_name", "home_id", "away_id",
              [_BF_COUNTRY.get(c, ()) for c in cc])
    lg = _num(fx, "apif_league_id")
    F = _Side(fx, "api_football", aliases, "home", "away", "apif_home_id", "apif_away_id",
              [(_LEAGUE_COUNTRY[int(x)],) if x == x and int(x) in _LEAGUE_COUNTRY else () for x in lg])
    res_e = _result_sign(ev["result"]) if "result" in ev.columns else np.full(len(ev), np.nan)
    ft_h, ft_a = _num(fx, "ft_home"), _num(fx, "ft_away")
    gd_f = ft_h - ft_a
    st = _State(E.n, F.n)

    def team_rounds():
        for _ in range(MAX_ROUNDS):
            tm = _state_map(E, F, st, min_share)
            if not _teams(E, F, st, tm, res_e, gd_f, tol_min, exact_tol_min, anchor_min_agree, fuzzy_min):
                break

    if E.n and F.n:
        _exact(E, F, st, res_e, gd_f, exact_tol_min)
        if team_pass:
            team_rounds()
        if _fuzzy(E, F, st, res_e, gd_f, tol_min, fuzzy_min, fuzzy_margin) and team_pass:
            team_rounds()

    on = st.f >= 0
    f = np.where(on, st.f, 0)

    def take(values, blank=np.nan):
        v = np.asarray(values, dtype=object)[f] if F.n else np.full(E.n, blank, dtype=object)
        return np.where(on, v, blank)

    def sided(home, away):
        h, a = take(home), take(away)
        return np.where(st.swapped, a, h), np.where(st.swapped, h, a)

    hid, aid = sided(F.hid, F.aid)
    hname, aname = sided(fx["home"] if F.n else [], fx["away"] if F.n else [])
    fth, fta = sided(ft_h, ft_a)
    gap = np.where(on, E.t - (F.t[f] if F.n else np.nan), np.nan)
    agree = _agrees(res_e, np.where(on, gd_f[f] if F.n else np.nan, np.nan), st.swapped)
    out = pd.DataFrame({
        "event_id": ev["event_id"].to_numpy(), "linked": on, "method": st.method,
        "score": np.where(on, st.score, np.nan),
        "apif_fixture_id": take(fx["apif_fixture_id"] if F.n else []),
        "apif_league_id": take(fx["apif_league_id"] if F.n else []),
        "apif_league": take(fx["league"] if F.n else [], None),
        "apif_league_country": take(fx["league_country"] if F.n else [], None),
        "season": take(fx["apif_season"] if F.n else []),
        "apif_home_team_id": hid, "apif_away_team_id": aid, "apif_home_name": hname, "apif_away_name": aname,
        "apif_kickoff": pd.Series(take(fx["kickoff"] if F.n else [], pd.NaT)),
        "kickoff_diff_min": gap, "swapped": np.where(on, st.swapped, False),
        "apif_ft_home": fth, "apif_ft_away": fta,
        "result_agrees": np.where(np.isnan(agree), None, agree == 1.0),
    })
    out["apif_kickoff"] = pd.to_datetime(out["apif_kickoff"], utc=True)
    for c in ("apif_fixture_id", "apif_league_id", "season", "apif_home_team_id", "apif_away_team_id", "apif_ft_home",
              "apif_ft_away"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    for c in ("apif_league", "apif_league_country", "apif_home_name", "apif_away_name"):
        out[c] = out[c].where(out[c].notna(), None).astype(object)
    out["score"] = out["score"].astype(float)
    out["kickoff_diff_min"] = out["kickoff_diff_min"].astype(float)
    out["result_agrees"] = out["result_agrees"].astype("boolean")
    return out


# --------------------------------------------------------------------------------------------------------------------
# canonical matches and the team table
# --------------------------------------------------------------------------------------------------------------------

def canonical(links: pd.DataFrame, canonical_matches: pd.DataFrame | None, fixtures: pd.DataFrame | None = None,
              aliases: dict | None = None) -> pd.DataFrame:
    """``links`` with ``match_id`` and ``competition_id``. A fixture is a canonical match where the canonical table
    (``football/gold/matches.parquet``) carries its ``apif_fixture_id`` (``football.data.apply_api_football`` writes
    it on every build). For linked fixtures of the 38 competitions it does not carry, and when ``fixtures`` is
    given, the same rule that writes it is run here (``football.player_ratings.link_matches``: competition, UK
    day, both names). ``competition_id`` is the match's, else the one the fixture's league id maps to."""
    out = links.copy()
    mid = pd.Series([None] * len(out), index=out.index, dtype=object)
    comp = pd.Series([None] * len(out), index=out.index, dtype=object)
    cm = canonical_matches
    if cm is not None and len(cm) and "match_id" in cm.columns:
        fid = out["apif_fixture_id"].astype("Int64")
        if "apif_fixture_id" in cm.columns:
            m = cm[cm["apif_fixture_id"].notna()].drop_duplicates("apif_fixture_id", keep="last")
            ids = pd.to_numeric(m["apif_fixture_id"], errors="coerce").astype("Int64")
            mid = fid.map(pd.Series(m["match_id"].to_numpy(), index=ids))
        need = out["linked"] & mid.isna() & out["apif_league_id"].isin(list(_LEAGUE_COMPETITION))
        if fixtures is not None and len(fixtures) and need.any():
            from football.player_ratings import link_matches
            fx = fixture_table(fixtures).rename(columns={"kickoff": "kickoff_utc"})
            fx = fx[fx["apif_fixture_id"].isin(fid[need].astype("int64"))]
            near = cm
            if len(fx):                              # the canonical table goes back to 1993: only the days in reach
                lo, hi = fx["kickoff_utc"].min() - pd.Timedelta(days=2), fx["kickoff_utc"].max() + pd.Timedelta(days=2)
                near = cm[cm["match_date"].astype(str).between(lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d"))]
            got = link_matches(fx, near, aliases) if len(fx) and len(near) else pd.DataFrame()
            if len(got):
                extra = pd.Series(got["match_id"].to_numpy(), index=got["apif_fixture_id"].astype("Int64"))
                mid = mid.where(mid.notna(), fid.map(extra))
        comp = mid.map(pd.Series(cm["competition_id"].to_numpy(), index=cm["match_id"]).groupby(level=0).first())
    by_league = out["apif_league_id"].map(_LEAGUE_COMPETITION)
    out["match_id"] = mid.where(mid.notna(), None).astype(object)
    out["competition_id"] = comp.where(comp.notna(), by_league).where(lambda s: s.notna(), None).astype(object)
    return out


def team_table(tm: pd.DataFrame, teams: pd.DataFrame | None = None, links: pd.DataFrame | None = None,
               events: pd.DataFrame | None = None, canonical_matches: pd.DataFrame | None = None) -> pd.DataFrame:
    """gold/team_link's rows: the team map on every Betfair team id of gold/team (``teams``), with API-Football's
    name for the mapped team (from ``links``) and the canonical team id most of the team's canonical matches give
    (``links`` with match_id, ``events``, ``canonical_matches`` with home_team_id and away_team_id)."""
    out = tm.copy()
    if teams is not None and len(teams):
        base = teams[["team_id"] + [c for c in ("team_name",) if c in teams.columns]]
        out = base.merge(out, on="team_id", how="left")
    if "team_name" not in out.columns:
        out["team_name"] = None
    out["apif_team_name"] = None
    out["canonical_team_id"], out["canonical_share"] = None, np.nan
    out["apif_team_id"] = pd.to_numeric(out["apif_team_id"], errors="coerce").astype("Int64")
    if links is not None and len(links):
        lk = links[links["apif_fixture_id"].notna()]
        names = pd.concat([lk[["apif_home_team_id", "apif_home_name"]].set_axis(["id", "name"], axis=1),
                           lk[["apif_away_team_id", "apif_away_name"]].set_axis(["id", "name"], axis=1)])
        top = _vote(names, "id", "name")
        out["apif_team_name"] = out["apif_team_id"].map(pd.Series(top["name"].to_numpy(),
                                                                  index=top["id"].astype("Int64")))
        cm = canonical_matches
        if (events is not None and cm is not None and "match_id" in lk.columns and lk["match_id"].notna().any()
                and {"home_team_id", "away_team_id"} <= set(cm.columns)):
            j = lk[lk["match_id"].notna()][["event_id", "match_id", "swapped"]].merge(
                events[["event_id", "home_id", "away_id"]], on="event_id").merge(
                cm[["match_id", "home_team_id", "away_team_id"]].drop_duplicates("match_id"), on="match_id")
            sw = j["swapped"].fillna(False).to_numpy(dtype=bool)
            pairs = pd.concat([
                pd.DataFrame({"team_id": j["home_id"].to_numpy(),
                              "canon": np.where(sw, j["away_team_id"], j["home_team_id"])}),
                pd.DataFrame({"team_id": j["away_id"].to_numpy(),
                              "canon": np.where(sw, j["home_team_id"], j["away_team_id"])})], ignore_index=True)
            cv = _vote(pairs, "team_id", "canon")
            cv = cv.set_index(cv["team_id"].astype("int64"))
            out["canonical_team_id"] = out["team_id"].map(cv["canon"])
            out["canonical_share"] = out["team_id"].map(cv["share"])
    for c in ("n_linked", "n_agree", "n_apif_ids"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("int64")
    out["low_agreement"] = out["low_agreement"].fillna(False).astype(bool)
    for c in ("team_name", "apif_team_name", "canonical_team_id"):
        out[c] = out[c].where(out[c].notna(), None).astype(object)
    return out[TEAM_LINK.names].sort_values("team_id").reset_index(drop=True)


# --------------------------------------------------------------------------------------------------------------------
# the goal check
# --------------------------------------------------------------------------------------------------------------------

def _minute_err(wall: float, minute: float, slack: float) -> tuple[float, float, int]:
    """(the minute error, wall clock less match clock, the half) of an inferred goal at ``wall`` minutes after
    kick-off against a goal event in match minute ``minute``. A goal in minute m fell in (m - 1, m]: its clock is
    m - 0.5. First half (API-Football folds stoppage into the minute: 45+2 is 47, and the wall clock reads 47
    too): the two clocks agree. Second half: the wall clock runs ahead by the break and the first half's
    stoppage, anything in ``HT_LAG``; only the distance outside that range is error."""
    lag = wall - (minute - 0.5)
    second = minute > 45 and wall > HT_WALL_MIN
    lo, hi = (HT_LAG[0], HT_LAG[1] + slack) if second else (0.0, slack)
    return lag - min(max(lag, lo), hi), lag, 2 if second else 1


def _align(walls, sides, minutes, msides, slack: float) -> list[tuple[int, int, float, float, int]]:
    """Inferred goals paired with goal events, both in time order, order kept: the pairing with the least total
    minute error, a goal left unmatched at ``GAP_COST``. Returns (inferred index, event index, minute error, lag,
    half) for each pair."""
    n, m = len(walls), len(minutes)
    if not n or not m:
        return []
    err = [[_minute_err(walls[i], minutes[j], slack) for j in range(m)] for i in range(n)]
    cost = [[abs(err[i][j][0]) + (SIDE_COST if sides[i] and msides[j] and sides[i] != msides[j] else 0.0)
             for j in range(m)] for i in range(n)]
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        best[i][0] = i * GAP_COST
    for j in range(1, m + 1):
        best[0][j] = j * GAP_COST
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best[i][j] = min(best[i - 1][j - 1] + cost[i - 1][j - 1], best[i - 1][j] + GAP_COST,
                             best[i][j - 1] + GAP_COST)
    pairs, i, j = [], n, m
    while i and j:
        if best[i][j] == best[i - 1][j - 1] + cost[i - 1][j - 1]:
            pairs.append((i - 1, j - 1) + err[i - 1][j - 1])
            i, j = i - 1, j - 1
        elif best[i][j] == best[i - 1][j] + GAP_COST:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def _side_word(s) -> str:
    s = str(s or "").strip().lower()
    return "home" if s.startswith("h") else "away" if s.startswith("a") else ""


def goal_check(goals: pd.DataFrame, events_links: pd.DataFrame, apif_events: pd.DataFrame | None
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(gold/goal_check's rows, one row a matched goal). ``goals``: gold/goal (event_id, goal_no, pt, elapsed_min,
    side). ``events_links``: gold/link rows, with gold/event's ``kickoff_actual`` and ``went_in_play`` beside them
    where held. ``apif_events``: ``api_football/history_events.parquet`` whole (a fixture with any event row has
    its events held, so no goal row means no goal).

    The wall clock of an inferred goal is its ``pt`` less the kick-off: the market's turn in play where the event
    has it, else the fixture's kick-off (then a start up to ``KO_SLACK_MIN`` late is allowed for); ``elapsed_min``
    is used only for a goal without a ``pt``. Counts are checked against the 90-minute score where the link has
    one (extra-time goals are not Betfair's match), else against the goal events. The matched-goal frame has
    event_id, goal_no, apif_minute, wall_min, half, lag_min, minute_err, side_agrees."""
    goal_cols = ["event_id", "goal_no", "apif_minute", "wall_min", "half", "lag_min", "minute_err", "side_agrees"]
    lk = events_links[events_links["apif_fixture_id"].notna()].copy()
    if "went_in_play" in lk.columns:
        lk = lk[lk["went_in_play"].fillna(False).astype(bool)]
    ko_actual = _minutes(lk["kickoff_actual"]) if "kickoff_actual" in lk.columns else np.full(len(lk), np.nan)
    ko_fixture = _minutes(lk["apif_kickoff"]) if "apif_kickoff" in lk.columns else np.full(len(lk), np.nan)
    fth, fta = _num(lk, "apif_ft_home"), _num(lk, "apif_ft_away")
    fid = lk["apif_fixture_id"].astype("int64").to_numpy()

    ae = apif_events if apif_events is not None else pd.DataFrame(columns=["apif_fixture_id", "minute", "type",
                                                                           "detail", "apif_team_id"])
    held = set(pd.to_numeric(ae["apif_fixture_id"], errors="coerce").dropna().astype("int64"))
    g = ae[(ae["type"] == "Goal") & ae["detail"].isin(GOAL_DETAILS)].copy()
    g["minute"] = pd.to_numeric(g["minute"], errors="coerce")
    g = g[(g["minute"] >= 0) & (g["minute"] <= 130) & g["apif_fixture_id"].isin(list(set(fid)))]
    by_fixture = {k: (v["minute"].to_numpy(dtype=float), pd.to_numeric(v["apif_team_id"], errors="coerce").to_numpy())
                  for k, v in g.sort_values(["apif_fixture_id", "minute"], kind="stable").groupby("apif_fixture_id")}

    inf = goals[goals["event_id"].isin(lk["event_id"])].copy() if goals is not None and len(goals) else \
        pd.DataFrame(columns=["event_id", "goal_no", "pt", "elapsed_min", "side"])
    inf["pt_min"] = _minutes(inf["pt"]) if len(inf) else []
    inf = inf.sort_values(["event_id", "pt_min", "goal_no"], kind="stable")
    by_event = {k: v for k, v in inf.groupby("event_id")}

    rows, matched = [], []
    home_ids, away_ids = _num(lk, "apif_home_team_id"), _num(lk, "apif_away_team_id")
    for i, eid in enumerate(lk["event_id"].to_numpy()):
        has_events = int(fid[i]) in held
        n_ft = fth[i] + fta[i]
        if not has_events and n_ft != n_ft:
            continue                                    # nothing to check against
        actual = ko_actual[i] == ko_actual[i]
        ko, slack = (ko_actual[i], 0.0) if actual else (ko_fixture[i], KO_SLACK_MIN)
        mine = by_event.get(eid)
        walls, sides, nos = [], [], []
        if mine is not None:
            w = mine["pt_min"].to_numpy(dtype=float) - ko
            el = pd.to_numeric(mine["elapsed_min"], errors="coerce").to_numpy(dtype=float) \
                if "elapsed_min" in mine.columns else np.full(len(mine), np.nan)
            walls = np.where(np.isnan(w), el, w).tolist()
            sides = [_side_word(s) for s in (mine["side"] if "side" in mine.columns else [""] * len(mine))]
            nos = mine["goal_no"].tolist()
        minutes, teams = by_fixture.get(int(fid[i]), (np.zeros(0), np.zeros(0)))
        if n_ft == n_ft and len(minutes) > n_ft:        # extra time and shoot-outs come after the 90 minutes
            minutes, teams = minutes[:int(n_ft)], teams[:int(n_ft)]
        msides = ["home" if t == home_ids[i] else "away" if t == away_ids[i] else "" for t in teams]
        usable = [k for k, w in enumerate(walls) if w == w]
        pairs = _align([walls[k] for k in usable], [sides[k] for k in usable], minutes.tolist(), msides, slack) \
            if has_events else []
        errs = [abs(p[2]) for p in pairs]
        known = [(sides[usable[p[0]]], msides[p[1]]) for p in pairs if sides[usable[p[0]]] and msides[p[1]]]
        lags2 = [p[3] for p in pairs if p[4] == 2]
        truth = n_ft if n_ft == n_ft else float(len(minutes))
        rows.append({
            "event_id": eid, "apif_fixture_id": int(fid[i]), "clock": "actual" if actual else "fixture",
            "has_events": has_events, "n_inferred": len(walls), "n_apif": len(minutes) if has_events else None,
            "n_ft": n_ft if n_ft == n_ft else None, "count_diff": int(len(walls) - truth),
            "count_agrees": len(walls) == truth, "n_matched": len(pairs),
            "n_within_tol": sum(e <= MINUTE_TOL for e in errs), "n_side_known": len(known),
            "n_side_agree": sum(a == b for a, b in known), "minute_mae": float(np.mean(errs)) if errs else np.nan,
            "minute_err_max": max(errs) if errs else np.nan,
            "ht_lag_min": float(np.median(lags2)) if lags2 else np.nan})
        for a, b, err, lag, half in pairs:
            k = usable[a]
            matched.append({"event_id": eid, "goal_no": nos[k], "apif_minute": minutes[b], "wall_min": walls[k],
                            "half": half, "lag_min": lag, "minute_err": err,
                            "side_agrees": (sides[k] == msides[b]) if sides[k] and msides[b] else None})
    per_event = pd.DataFrame(rows, columns=GOAL_CHECK.names)
    for c in ("n_apif", "n_ft"):
        per_event[c] = pd.to_numeric(per_event[c], errors="coerce").astype("Int64")
    return per_event, pd.DataFrame(matched, columns=goal_cols)


def goal_summary(per_event: pd.DataFrame, per_goal: pd.DataFrame) -> dict:
    """The goal check in a few numbers: how often the count agrees, how many goals pair, how close the minutes."""
    n = int(len(per_event))
    if not n:
        return {"matches_checked": 0}
    err = per_goal["minute_err"].abs() if len(per_goal) else pd.Series(dtype=float)
    apif = int(per_event["n_apif"].fillna(0).sum())
    lag = per_event["ht_lag_min"].dropna()
    known, agree = int(per_event["n_side_known"].sum()), int(per_event["n_side_agree"].sum())
    return {
        "matches_checked": n, "matches_with_events": int(per_event["has_events"].sum()),
        "count_agrees": int(per_event["count_agrees"].sum()),
        "count_agrees_share": round(float(per_event["count_agrees"].mean()), 4),
        "count_diff": {str(k): int(v) for k, v in per_event["count_diff"].clip(-3, 3).value_counts().sort_index()
                       .items()},
        "goals_inferred": int(per_event["n_inferred"].sum()), "goal_events": apif,
        "goals_matched": int(len(per_goal)),
        "goal_events_matched_share": round(len(per_goal) / apif, 4) if apif else None,
        "minute_within_tol_share": round(float((err <= MINUTE_TOL).mean()), 4) if len(err) else None,
        "minute_abs_err_median": round(float(err.median()), 2) if len(err) else None,
        "minute_abs_err_p90": round(float(err.quantile(0.9)), 2) if len(err) else None,
        "side_agrees_share": round(agree / known, 4) if known else None,
        "ht_lag_min_quartiles": [round(float(lag.quantile(q)), 1) for q in (0.25, 0.5, 0.75)] if len(lag) else None,
    }


# --------------------------------------------------------------------------------------------------------------------
# the report, the store wrapper, the command line
# --------------------------------------------------------------------------------------------------------------------

def _share(a: int, b: int):
    return round(a / b, 4) if b else None


def report(events: pd.DataFrame, links: pd.DataFrame, team_links: pd.DataFrame) -> dict:
    """What linked: in all, by method, by month of kick-off, by Betfair country code; where the unlinked events
    are; the team ids mapped and the ones whose matches disagree."""
    j = events[["event_id"]].merge(links[["event_id", "linked", "method", "result_agrees", "kickoff_diff_min",
                                          "match_id"]], on="event_id", how="left")
    j["linked"] = j["linked"].fillna(False).astype(bool)
    j["month"] = pd.to_datetime(events["kickoff"], utc=True).dt.strftime("%Y-%m").fillna("(none)").to_numpy()
    cc = events["country_code"] if "country_code" in events.columns else pd.Series([""] * len(events))
    j["country"] = cc.fillna("").replace("", "(none)").to_numpy()
    n, n_linked = int(len(j)), int(j["linked"].sum())

    def by(col: str) -> dict:
        g = j.groupby(col)["linked"].agg(["size", "sum"])
        return {str(k): {"events": int(r["size"]), "linked": int(r["sum"]), "share": _share(int(r["sum"]),
                                                                                               int(r["size"]))}
                for k, r in g.iterrows()}

    countries = by("country")
    unlinked = j[~j["linked"]]["country"].value_counts().head(20)
    known = j["result_agrees"].notna()
    gap = j.loc[j["linked"], "kickoff_diff_min"].abs()
    tl = team_links
    mapped = tl["apif_team_id"].notna() & ~tl["low_agreement"]
    low = tl[tl["low_agreement"]].sort_values(["n_linked", "team_id"], ascending=[False, True])
    return {
        "events": n, "linked": {"n": n_linked, "share": _share(n_linked, n)},
        "by_method": {m: int((j["method"] == m).sum()) for m in METHODS + ("none",) if (j["method"] == m).any()},
        "by_month": by("month"),
        "by_country": dict(sorted(countries.items(), key=lambda kv: (-kv[1]["events"], kv[0]))),
        "unlinked_top_countries": {str(k): int(v) for k, v in unlinked.items()},
        "result_check": {"known": int(known.sum()),
                         "agree": int(j["result_agrees"].fillna(False).astype(bool).sum())},
        "kickoff_gap_min": {"within_1": int((gap <= 1).sum()), "within_30": int((gap <= TOL_MIN).sum()),
                            "over_30": int((gap > TOL_MIN).sum())},
        "canonical_matches": int(j["match_id"].notna().sum()),
        "teams": {"team_ids": int(len(tl)), "mapped": int(mapped.sum()), "share": _share(int(mapped.sum()), len(tl)),
                  "low_agreement": int(tl["low_agreement"].sum()),
                  "low_agreement_top": [{"team_id": int(r.team_id), "team_name": r.team_name,
                                         "n_linked": int(r.n_linked), "share": round(float(r.share), 3)}
                                        for r in low.head(20).itertuples(index=False)]},
    }


def to_arrow(spec: TableSpec, df: pd.DataFrame) -> pa.Table:
    """A frame as the spec's Arrow table: blanks as nulls, never as NaN cast to a number."""
    arrays = []
    for c in spec.columns:
        s = df[c.name]
        if pa.types.is_integer(c.type):
            arr = pa.array(pd.array(pd.to_numeric(s, errors="coerce"), dtype="Int64"), type=c.type)
        elif pa.types.is_boolean(c.type):
            arr = pa.array(pd.array(s, dtype="boolean"), type=c.type)
        elif pa.types.is_timestamp(c.type):
            arr = pa.array(pd.to_datetime(s, utc=True)).cast(c.type)
        elif pa.types.is_floating(c.type):
            arr = pa.array(pd.to_numeric(s, errors="coerce").to_numpy(dtype=float), type=c.type, from_pandas=True)
        else:
            arr = pa.array([None if v is None or v != v else str(v) for v in s], type=c.type)
        arrays.append(arr)
    return pa.Table.from_arrays(arrays, schema=spec.schema)


def _frame(store, key: str, columns: list[str], since=None, until=None) -> pd.DataFrame | None:
    """A stored parquet's columns of interest (those it has) as a frame, or None when the key is not there.
    ``since`` / ``until``: only rows whose ``kickoff_utc`` is between them (the fixtures table holds ten years)."""
    raw = store.get(key)
    if raw is None:
        return None
    have = pq.read_schema(io.BytesIO(raw)).names
    cols = [c for c in columns if c in have]
    t = pq.read_table(io.BytesIO(raw), columns=cols)
    if since is not None and "kickoff_utc" in cols and pa.types.is_timestamp(t.schema.field("kickoff_utc").type):
        typ = t.schema.field("kickoff_utc").type     # cut in Arrow first: ten years of fixtures need not reach pandas

        def scalar(x):
            x = x.tz_convert("UTC")
            return pa.scalar((x if typ.tz else x.tz_localize(None)).to_pydatetime(), type=typ)
        t = t.filter(pc.and_(pc.greater_equal(t["kickoff_utc"], scalar(since)),
                             pc.less_equal(t["kickoff_utc"], scalar(until))))
    df = t.to_pandas()
    if since is not None and "kickoff_utc" in df.columns:
        k = pd.to_datetime(df["kickoff_utc"], utc=True, errors="coerce")
        df = df[(k >= since) & (k <= until)].reset_index(drop=True)
    return df


def run(store, write: bool = True) -> dict:
    """Read gold/event and gold/team from the warehouse and API-Football's and the canonical tables from the store,
    link, write gold/link, gold/team_link and (when gold/goal exists) gold/goal_check; the report. An input that is
    not there is named in ``missing`` ('no data found: <key>') and the stage does what it can without it."""
    from football.warehouse.store_io import (fingerprint, read_single, read_table, single_key, utc_now,
                                             write_manifest, write_single)
    rep: dict = {"version": VERSION, "missing": [], "written": {}}

    def missing(key: str) -> None:
        rep["missing"].append(f"no data found: {key}")

    ev_cols = ["event_id", "home_name", "away_name", "home_id", "away_id", "country_code", "kickoff",
               "kickoff_actual", "went_in_play", "result"]
    events = read_table(store, "gold/event", ev_cols)
    if events is None:
        events = read_single(store, "gold/event", ev_cols)
    if events is None or not events.num_rows:
        missing(single_key("gold/event").rsplit("/", 1)[0])
        rep["status"] = "no events: nothing linked"
        return rep
    events = events.to_pandas()
    teams = read_single(store, "gold/team", ["team_id", "team_name"])
    if teams is None:
        missing(single_key("gold/team"))
    teams = teams.to_pandas() if teams is not None else None

    k = pd.to_datetime(events["kickoff"], utc=True)
    since, until = k.min() - pd.Timedelta(days=1), k.max() + pd.Timedelta(days=1)
    fx_cols = ["apif_fixture_id", "kickoff_utc", "home", "away", "apif_home_id", "apif_away_id", "apif_league_id",
               "apif_season", "ft_home", "ft_away", "league", "league_country"]
    fixtures = _frame(store, FIXTURES_KEY, fx_cols, since, until)
    history = _frame(store, HISTORY_MATCHES_KEY, fx_cols, since, until)
    for key, df in ((FIXTURES_KEY, fixtures), (HISTORY_MATCHES_KEY, history)):
        if df is None:
            missing(key)
    fx = fixture_table(fixtures, history)
    rep["fixtures"] = {"rows": int(len(fx)), "from_fixtures_table": 0 if fixtures is None else int(len(fixtures)),
                       "from_history": 0 if history is None else int(len(history))}
    if not len(fx):
        rep["status"] = "no fixtures: nothing linked"
        return rep
    canon = _frame(store, CANONICAL_KEY, ["match_id", "competition_id", "match_date", "home_team", "away_team",
                                          "home_team_id", "away_team_id", "apif_fixture_id"], None, None)
    if canon is None:
        missing(CANONICAL_KEY)

    aliases = load_aliases()
    links = canonical(link_events(events, fx, aliases), canon, fx, aliases)
    team_links = team_table(team_map(links, events), teams, links, events, canon)
    rep.update(report(events, links, team_links))

    out = {LINK: links, TEAM_LINK: team_links}
    goals = read_table(store, "gold/goal")
    if goals is None:
        goals = read_single(store, "gold/goal")
    lacks = [] if goals is None else [c for c in ("event_id", "goal_no", "pt") if c not in goals.column_names]
    if goals is None:
        missing(single_key("gold/goal").rsplit("/", 1)[0])
    elif lacks:
        rep["goal_check"] = {"error": f"gold/goal lacks columns {lacks}"}
    else:
        he = _frame(store, HISTORY_EVENTS_KEY, ["apif_fixture_id", "minute", "type", "detail", "apif_team_id"])
        if he is None:
            missing(HISTORY_EVENTS_KEY)
        el = links.merge(events[["event_id", "kickoff_actual", "went_in_play"]], on="event_id")
        per_event, per_goal = goal_check(goals.to_pandas(), el, he)
        rep["goal_check"] = goal_summary(per_event, per_goal)
        out[GOAL_CHECK] = per_event
    if write:
        fp = fingerprint([f"v{VERSION}", f"events:{len(events)}", f"fixtures:{len(fx)}",
                          f"canonical:{0 if canon is None else len(canon)}"])
        for spec, df in out.items():
            wrote = write_single(store, spec, to_arrow(spec, df))
            write_manifest(store, spec.name, {"parts": {"all": {"rows": wrote["rows"], "bytes": wrote["bytes"],
                                                                "fingerprint": fp, "version": VERSION,
                                                                "built_utc": utc_now()}}})
            rep["written"][spec.name] = wrote["rows"]
    rep["status"] = "ok"
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Link Betfair events and teams to API-Football and canonical matches")
    ap.add_argument("--root", default=None, help="a local folder in place of the capture bucket")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="link, write gold/link, gold/team_link and gold/goal_check, print the report")
    r.add_argument("--root", dest="run_root", default=None, help="the same, after the command")
    r.add_argument("--dry-run", action="store_true", help="report only: write nothing")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    from sources.common import Store
    root = a.run_root or a.root
    out = run(Store(root=root) if root else Store(), write=not a.dry_run)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
