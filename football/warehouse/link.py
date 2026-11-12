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
  API-Football's goal events (``api_football/history_events.parquet``), by count, score, minute and side.

How an event finds its fixture (``link_events``), strongest evidence first, a fixture never given to two events
(of a match Betfair listed twice, the listing it settled gets the fixture, not the void one):

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

A link by anything but the names being equal is dropped when a settlement both sources hold disagrees: Betfair's
1X2 result, or its over/under 2.5, against the fixture's score after 90 minutes. The drops are counted by method in
the run's report (``vetoed``): a wrong fixture's score fails one of the two 81 times in 100 (measured: 6,111
linked events, each against 50 other fixtures' scores), so the count says how many wrong links the names and
teams proposed, and about a quarter of it how many got through.

The tolerances are measured (11 Nov 2026, the owner's 2026 Betfair download against API-Football's 22 main
divisions): of 4,246 exact-name pairs within half a day, 3,961 were within 1 minute, 4,241 within 30, the other 5
within 90 (late kick-offs Betfair's market time followed), none between 90 minutes and 12 hours, and every one
whose result both sources hold agreed. So 30 minutes where the names are not proof, 180 where they are.

So is the precision (the same data, the link run blind to the settlements and then scored against them): of
6,115 links every one whose result is known agreed (6,111), and every one whose over/under is (6,112); by the
names alone the fuzzy pass made 1,699 links, none contradicted. With each event's own fixture taken away (the
55,015 Betfair events against one another, any link a wrong one) the fuzzy pass linked 5 pairs, each one match
listed twice, once a name of initials was stopped from resembling everything (``side_score``). The fixtures
table of every competition is about seven times denser than that test's pool: the report of each run lists what
to look at (``vetoed``, ``disagreements``, ``weakest_links``).

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

VERSION = 2
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
        Column("over25_agrees", pa.bool_(), "Whether that score's total (three goals or more) is the over/under "
               "2.5 Betfair settled (blank when either is missing). As result_agrees: False only on exact links."),
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
        Column("usable", pa.bool_(), "gold/timeline's data rule for the match (enough price rows, no long gap, in "
               "play from the kick-off): judge the inference on these. Blank without gold/timeline."),
        Column("has_events", pa.bool_(), "Whether API-Football's events are held for the fixture.", False),
        Column("n_inferred", pa.int16(), "Goals inferred from prices.", False),
        Column("n_untimed", pa.int16(), "Of those, goals the prices cannot time (gold/goal.timed false: the "
               "settlement says they were scored, their moment is a placeholder): in the count and the score, "
               "left out of the pairing with goal events.", False),
        Column("n_apif", pa.int16(), "API-Football's goal events (normal, penalty, own goal; not a missed "
               "penalty) within the 90 minutes; blank without events."),
        Column("n_ft", pa.int16(), "Goals in API-Football's 90-minute score; blank without one."),
        Column("count_diff", pa.int16(), "n_inferred less the true count (n_ft, else n_apif).", False),
        Column("count_agrees", pa.bool_(), "count_diff is 0.", False),
        Column("score_agrees", pa.bool_(), "Whether the inferred goals give the true score, side by side (the "
               "90-minute score, else the goal events'). False when the count differs; blank when the count "
               "agrees and an inferred goal, or a goal event, names no side."),
        Column("n_matched", pa.int16(), "Inferred goals paired with a goal event (in order, within "
               f"{2 * GAP_COST:.0f} minutes).", False),
        Column("n_within_tol", pa.int16(), f"Of those, pairs whose football minute agrees within "
               f"{MINUTE_TOL:.0f}: gold/goal's match_min against the middle of the real goal's minute (the wall "
               "clock's error where a goal has no match_min).", False),
        Column("n_side_known", pa.int16(), "Pairs where both sources name the scoring side (API-Football lists an "
               "own goal under the side it counts for).", False),
        Column("n_side_agree", pa.int16(), "Of those, pairs that name the same side.", False),
        Column("minute_mae", pa.float64(), "Mean absolute minute error of the pairs on the wall clock (in the "
               f"second half only what lies outside a lag of {HT_LAG[0]:.0f} to {HT_LAG[1]:.0f} minutes)."),
        Column("minute_err_max", pa.float64(), "The largest absolute minute error."),
        Column("match_min_mae", pa.float64(), "Mean absolute error of gold/goal's match_min over the pairs, against "
               "the middle of the real goal's minute (capped at 45 and 90 as match_min is)."),
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
    false links). A name of initials alone ('A.E.L.') equals the same letters written together ('AEL') and resembles
    nothing else ('A.E.L. v Aris Limassol' was 'Atromitos v Aris', sides swapped, half an hour apart: measured). A
    score that cannot reach ``floor`` is returned as 0 without the costly comparison."""
    if a == b:
        return 1.0 if a.split("|", 1)[0] else 0.0
    ca, sa = a.split("|", 1)
    cb, sb = b.split("|", 1)
    if sa != sb or not ca or not cb:
        return 0.0
    if ca == cb or ca.replace(" ", "") == cb.replace(" ", ""):
        return 1.0                                   # 'A.E.L.' is 'AEL'
    la, lb = _no_lone(ca), _no_lone(cb)
    if not la or not lb:
        return 0.0                                   # initials alone resemble every name that starts with one of them
    if floor and 2 * min(len(ca), len(cb)) < floor * (len(ca) + len(cb)) and not _related(ca.split(), cb.split()):
        return 0.0                                   # no shared word start, and too unequal for the character ratio
    s = similarity(ca, cb)
    if s and (la != ca or lb != cb):
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


def _text(s) -> pd.Series:
    """A text column with None for every blank (NaN, NA), as plain objects: the same in any pandas."""
    return pd.Series([None if v is None or v is pd.NA or v != v else v for v in s], index=getattr(s, "index", None),
                     dtype=object)


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


class _Settled:
    """What each source says of a finished match: Betfair's settled 1X2 result and over/under 2.5 goals (gold/event's
    ``result`` and ``over25``) and the fixture's score after 90 minutes. Each check is 1.0 where the two agree, 0.0
    where they do not, NaN where either is blank."""

    def __init__(self, ev: pd.DataFrame, fx: pd.DataFrame):
        self.res = pd.Series(ev["result"]).map({"H": 1.0, "D": 0.0, "A": -1.0}).to_numpy(dtype=float) \
            if "result" in ev.columns else np.full(len(ev), np.nan)
        self.ou = _num(ev, "over25")
        self.ft_h, self.ft_a = _num(fx, "ft_home"), _num(fx, "ft_away")

    def result(self, e, f, swapped) -> np.ndarray:
        gd = self.ft_h[f] - self.ft_a[f]
        sign = np.sign(gd) * np.where(swapped, -1.0, 1.0)
        return np.where(np.isnan(self.res[e]) | np.isnan(gd), np.nan, (self.res[e] == sign).astype(float))

    def over25(self, e, f) -> np.ndarray:
        total = self.ft_h[f] + self.ft_a[f]
        return np.where(np.isnan(self.ou[e]) | np.isnan(total), np.nan,
                        ((total >= 3) == (self.ou[e] == 1.0)).astype(float))

    def verdict(self, c: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """For candidate links (e, f, swapped): (whether a settlement both sources hold disagrees, whether the
        result is not confirmed: either source lacks it)."""
        e, f = _ix(c, "e"), _ix(c, "f")
        res = self.result(e, f, c["swapped"].to_numpy(dtype=bool))
        return (res == 0.0) | (self.over25(e, f) == 0.0), res != 1.0


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
        self.vetoed: dict[str, set] = {}             # candidate links (e, f) dropped because a settlement disagreed

    def veto(self, cand: pd.DataFrame, settled: _Settled, method: str) -> pd.DataFrame:
        """``cand`` without the candidates a settlement contradicts (the drops are counted), with ``assign``'s
        ``bad`` (False) and ``unsure``."""
        if not len(cand):
            return cand.assign(bad=False, unsure=False)
        bad, unsure = settled.verdict(cand)
        if bad.any():                                   # a set: the team rounds meet the same candidate again
            e, f = _ix(cand, "e")[bad], _ix(cand, "f")[bad]
            self.vetoed.setdefault(method, set()).update(zip(e.tolist(), f.tolist()))
        return cand[~bad].assign(bad=False, unsure=unsure[~bad])

    def assign(self, cand: pd.DataFrame, method=None) -> int:
        """Give each event of ``cand`` (e, f, swapped, score, gap, bad, unsure, [method]) its best free fixture:
        best score first, then the settlements not disagreeing (``bad``), the result confirmed (``unsure``: of a
        match Betfair listed twice the listing it settled gets the fixture, not the void one), the sides the same
        way round, the nearer kick-off, the lower ids. One event a fixture, one fixture an event."""
        if not len(cand):
            return 0
        c = cand.assign(absgap=cand["gap"].abs()).sort_values(
            ["score", "bad", "unsure", "swapped", "absgap", "e", "f"],
            ascending=[False, True, True, True, True, True, True])
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


def _exact(E: _Side, F: _Side, st: _State, settled: _Settled, tol: float) -> int:
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
    bad, c["unsure"] = settled.verdict(c)
    c["bad"] = bad | alias                           # among equals: settlements that agree first, a plain name first
    return st.assign(c)


def _fuzzy(E: _Side, F: _Side, st: _State, settled: _Settled, tol: float, floor: float, margin: float) -> int:
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
    return st.assign(st.veto(c, settled, "fuzzy"), "fuzzy")


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


def _teams(E: _Side, F: _Side, st: _State, tm: pd.DataFrame, settled: _Settled, tol: float, exact_tol: float,
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

    def finish(c: pd.DataFrame, limit: float, method: str) -> pd.DataFrame:
        c = c.assign(gap=E.t[_ix(c, "e")] - F.t[_ix(c, "f")])
        return st.veto(c[c["gap"].abs() <= limit], settled, method)

    both = ev.dropna(subset=["mh", "ma"])
    c = pd.concat([both.merge(fx, left_on=["mh", "ma"], right_on=["fh", "fa"]).assign(swapped=False),
                   both.merge(fx, left_on=["mh", "ma"], right_on=["fa", "fh"]).assign(swapped=True)],
                  ignore_index=True)
    c["score"] = c[["sh", "sa"]].min(axis=1)
    made = st.assign(finish(c, exact_tol, "team_map"), "team_map")

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
    c = finish(pd.concat(parts, ignore_index=True), tol, "team_anchor")
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
                anchor_min_agree: int = ANCHOR_MIN_AGREE, team_pass: bool = True,
                stats: dict | None = None) -> pd.DataFrame:
    """One row a Betfair event (gold/link's columns but match_id and competition_id): the API-Football fixture it
    is, or method 'none'. ``events``: gold/event rows (event_id, home_name, away_name, kickoff, and where held
    country_code, home_id, away_id, result, over25). ``fixtures``: ``fixture_table``'s rows. ``aliases``:
    ``football.reference.load_aliases()``. ``team_pass`` False stops after the names (exact, alias, fuzzy).
    ``stats``: a dict that is given ``vetoed``: by method, the candidate links dropped because a settlement
    disagreed (a wrong candidate is caught about four times in five, so about a quarter of that count got through)."""
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
    settled = _Settled(ev, fx)
    st = _State(E.n, F.n)

    def team_rounds():
        for _ in range(MAX_ROUNDS):
            tm = _state_map(E, F, st, min_share)
            if not _teams(E, F, st, tm, settled, tol_min, exact_tol_min, anchor_min_agree, fuzzy_min):
                break

    if E.n and F.n:
        _exact(E, F, st, settled, exact_tol_min)
        if team_pass:
            team_rounds()
        if _fuzzy(E, F, st, settled, tol_min, fuzzy_min, fuzzy_margin) and team_pass:
            team_rounds()
    if stats is not None:
        stats["vetoed"] = {m: len(st.vetoed[m]) for m in METHODS if m in st.vetoed}

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
    fth, fta = sided(settled.ft_h, settled.ft_a)
    gap = np.where(on, E.t - (F.t[f] if F.n else np.nan), np.nan)
    every = np.arange(E.n)
    agree = np.where(on, settled.result(every, f, st.swapped), np.nan) if F.n else np.full(E.n, np.nan)
    agree_ou = np.where(on, settled.over25(every, f), np.nan) if F.n else np.full(E.n, np.nan)
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
        "over25_agrees": np.where(np.isnan(agree_ou), None, agree_ou == 1.0),
    })
    out["apif_kickoff"] = pd.to_datetime(out["apif_kickoff"], utc=True)
    for c in ("apif_fixture_id", "apif_league_id", "season", "apif_home_team_id", "apif_away_team_id", "apif_ft_home",
              "apif_ft_away"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    for c in ("apif_league", "apif_league_country", "apif_home_name", "apif_away_name"):
        out[c] = _text(out[c])
    out["score"] = out["score"].astype(float)
    out["kickoff_diff_min"] = out["kickoff_diff_min"].astype(float)
    for c in ("result_agrees", "over25_agrees"):
        out[c] = out[c].astype("boolean")
    return out


# --------------------------------------------------------------------------------------------------------------------
# canonical matches and the team table
# --------------------------------------------------------------------------------------------------------------------

def _canonical_votes(links: pd.DataFrame, events: pd.DataFrame, cm: pd.DataFrame) -> pd.DataFrame:
    """The vote from Betfair team ids to canonical team ids over the events linked to a canonical match (team_id,
    canon, n_linked, n_agree, share). The canonical home side is the fixture's: the event's away side when the
    link is swapped."""
    j = links[links["match_id"].notna()][["event_id", "match_id", "swapped"]].merge(
        events[["event_id", "home_id", "away_id"]], on="event_id").merge(
        cm[["match_id", "home_team_id", "away_team_id"]].drop_duplicates("match_id"), on="match_id")
    sw = j["swapped"].fillna(False).to_numpy(dtype=bool)
    home, away = np.where(sw, j["away_team_id"], j["home_team_id"]), np.where(sw, j["home_team_id"], j["away_team_id"])
    pairs = pd.concat([pd.DataFrame({"team_id": j["home_id"].to_numpy(), "canon": home}),
                       pd.DataFrame({"team_id": j["away_id"].to_numpy(), "canon": away})], ignore_index=True)
    return _vote(pairs, "team_id", "canon")


def _canonical_by_team(out: pd.DataFrame, events: pd.DataFrame, cm: pd.DataFrame) -> pd.Series:
    """match_id for linked events of our competitions that have none yet, through the teams: the canonical match of
    that competition on the fixture's UK day whose sides are the canonical teams the event's Betfair ids vote for.
    Both sides voted for: both must be the match's; one side: it needs ``ANCHOR_MIN_AGREE`` matches behind it (a
    team plays one match a day). One match an event, one event a match, or nothing. This is how a club the names
    never pair ('Paris Saint Germain' against football-data's 'Paris SG') gets its matches."""
    mid = out["match_id"].copy()
    need = out["linked"] & mid.isna() & out["apif_league_id"].isin(list(_LEAGUE_COMPETITION))
    if not need.any() or not mid.notna().any() or not {"home_team_id", "away_team_id", "match_date"} <= set(cm):
        return mid
    votes = _canonical_votes(out, events, cm)
    votes = votes[votes["share"] >= MIN_SHARE]
    canon = pd.Series(votes["canon"].to_numpy(), index=votes["team_id"].astype("int64"))
    strong = pd.Series((votes["n_agree"] >= ANCHOR_MIN_AGREE).to_numpy(), index=votes["team_id"].astype("int64"))
    r = out.loc[need, ["event_id", "apif_kickoff", "swapped", "apif_league_id"]].merge(
        events[["event_id", "home_id", "away_id"]], on="event_id")
    r["match_date"] = r["apif_kickoff"].dt.tz_convert("Europe/London").dt.strftime("%Y-%m-%d")
    r["competition_id"] = r["apif_league_id"].map(_LEAGUE_COMPETITION)
    sw = r["swapped"].fillna(False).to_numpy(dtype=bool)
    for col, first, second in (("fh", "home_id", "away_id"), ("fa", "away_id", "home_id")):
        ids = pd.Series(np.where(sw, r[second], r[first]))       # the Betfair id on the fixture's home (away) side
        r[col], r[col + "_strong"] = ids.map(canon).to_numpy(), ids.map(strong).fillna(False).to_numpy(dtype=bool)
    free = cm[~cm["match_id"].isin(set(mid.dropna()))][["match_id", "competition_id", "match_date", "home_team_id",
                                                        "away_team_id"]]
    c = r.merge(free.assign(match_date=free["match_date"].astype(str)), on=["match_date", "competition_id"])
    ok_h, ok_a = c["fh"].isna() | (c["fh"] == c["home_team_id"]), c["fa"].isna() | (c["fa"] == c["away_team_id"])
    both = c["fh"].notna() & c["fa"].notna()
    one = (c["fh"].notna() & c["fh_strong"]) ^ (c["fa"].notna() & c["fa_strong"])
    c = c[ok_h & ok_a & (both | (~both & one))]
    c = c[~c.duplicated("event_id", keep=False) & ~c.duplicated("match_id", keep=False)]
    found = pd.Series(c["match_id"].to_numpy(), index=c["event_id"].to_numpy())
    return mid.where(mid.notna(), out["event_id"].map(found))


def canonical(links: pd.DataFrame, canonical_matches: pd.DataFrame | None, fixtures: pd.DataFrame | None = None,
              aliases: dict | None = None, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """``links`` with ``match_id`` and ``competition_id``. A fixture is a canonical match where the canonical table
    (``football/gold/matches.parquet``) carries its ``apif_fixture_id`` (``football.data.apply_api_football`` writes
    it on every build). For linked fixtures of the 38 competitions it does not carry, and when ``fixtures`` is
    given, the same rule that writes it is run here (``football.player_ratings.link_matches``: competition, UK
    day, both names); and for what the names still miss, when ``events`` is given, the teams' vote
    (``_canonical_by_team``). ``competition_id`` is the match's, else the one the fixture's league id maps to."""
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
        if fixtures is not None and len(fixtures) and need.any() and "match_date" in cm.columns:
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
        for _ in range(MAX_ROUNDS if events is not None and len(events) else 0):
            more = _canonical_by_team(out.assign(match_id=_text(mid)), events, cm)   # each round's matches add votes
            if more.notna().sum() == mid.notna().sum():
                break
            mid = more
        comp = mid.map(pd.Series(cm["competition_id"].to_numpy(), index=cm["match_id"]).groupby(level=0).first())
    by_league = out["apif_league_id"].map(_LEAGUE_COMPETITION)
    out["match_id"] = _text(mid)
    out["competition_id"] = _text(comp.where(comp.notna(), by_league))
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
            cv = _canonical_votes(lk, events, cm)
            cv = cv.set_index(cv["team_id"].astype("int64"))
            out["canonical_team_id"] = out["team_id"].map(cv["canon"])
            out["canonical_share"] = out["team_id"].map(cv["share"])
    for c in ("n_linked", "n_agree", "n_apif_ids"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("int64")
    out["low_agreement"] = out["low_agreement"].fillna(False).astype(bool)
    for c in ("team_name", "apif_team_name", "canonical_team_id"):
        out[c] = _text(out[c])
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


def goal_check(goals: pd.DataFrame, events_links: pd.DataFrame, apif_events: pd.DataFrame | None,
               timeline: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(gold/goal_check's rows, one row a matched goal). ``goals``: gold/goal (event_id, goal_no, pt, and where
    held elapsed_min, side, match_min). ``events_links``: gold/link rows, with gold/event's ``kickoff_actual`` and
    ``went_in_play`` beside them where held. ``apif_events``: ``api_football/history_events.parquet`` (the rows
    of the linked fixtures are enough: a fixture with any event row has its events held, so no goal row means no
    goal). ``timeline``: gold/timeline (event_id, usable); when given, only its matches are checked (a match it
    does not hold had no goals inferred at all, which is not the same as none found) and ``usable`` is carried.

    Real goals on the wall clock. API-Football gives a goal's minute on the match clock (``minute``: elapsed plus
    added time, so 45+2 is 47) and, in the raw fixture, ``fixture.periods``; those are not the halves' real
    starts (measured on the 9,933 raw fixtures of seasons 2025 and 2026: ``periods.first`` is the scheduled
    kick-off and ``periods.second`` is 3,600 seconds later in all 9,322 that have them). So the wall clock is
    Betfair's: an inferred goal's ``pt`` less the market's turn in play, else less the fixture's kick-off (then a
    start up to ``KO_SLACK_MIN`` late is allowed for); ``elapsed_min`` is used only for a goal without a ``pt``.
    A real goal in minute m of the first half stands at m - 0.5 on it; in the second half at that plus the break
    and the first half's added time, anything in ``HT_LAG`` (``_minute_err``).

    Which goals count: type 'Goal' with detail in ``GOAL_DETAILS`` (a missed penalty is not a goal). An own goal
    is listed under the side it counts for, not the scorer's (measured: the listed sides give the score in 764
    of 764 matches with an own goal; the scorer's side would in 12), so the listed team is the scoring side.
    Counts and the score are checked against the 90-minute score where the link has one (extra-time goals are
    not Betfair's match), else against the goal events. The matched-goal frame has event_id, goal_no,
    apif_minute, wall_min, half, lag_min, minute_err, side_agrees, match_min, match_min_err (gold/goal's
    ``match_min`` less the real minute's middle, the minute capped at 45 and 90 as ``match_min`` is)."""
    goal_cols = ["event_id", "goal_no", "apif_minute", "wall_min", "half", "lag_min", "minute_err", "side_agrees",
                 "match_min", "match_min_err"]
    lk = events_links[events_links["apif_fixture_id"].notna()].copy()
    if "went_in_play" in lk.columns:
        lk = lk[lk["went_in_play"].fillna(False).astype(bool)]
    usable: dict = {}
    if timeline is not None:
        lk = lk[lk["event_id"].isin(timeline["event_id"])]
        if "usable" in timeline.columns:
            usable = dict(zip(timeline["event_id"], timeline["usable"]))
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
        walls, sides, nos, mms = [], [], [], []
        if mine is not None:
            w = mine["pt_min"].to_numpy(dtype=float) - ko
            walls = np.where(np.isnan(w), _num(mine, "elapsed_min"), w).tolist()
            sides = [_side_word(x) for x in (mine["side"] if "side" in mine.columns else [""] * len(mine))]
            nos, mms = mine["goal_no"].tolist(), _num(mine, "match_min").tolist()
        minutes, teams = by_fixture.get(int(fid[i]), (np.zeros(0), np.zeros(0)))
        if n_ft == n_ft and len(minutes) > n_ft:        # extra time and shoot-outs come after the 90 minutes
            minutes, teams = minutes[:int(n_ft)], teams[:int(n_ft)]
        msides = ["home" if t == home_ids[i] else "away" if t == away_ids[i] else "" for t in teams]
        # a goal the prices cannot time has a placeholder moment after the last price: it counts as a goal, and
        # is not paired with a goal event
        clocked = mine["timed"].fillna(True).astype(bool).tolist() if mine is not None and "timed" in mine.columns \
            else [True] * len(walls)
        usable_k = [k for k, w in enumerate(walls) if w == w and clocked[k]]
        pairs = _align([walls[k] for k in usable_k], [sides[k] for k in usable_k], minutes.tolist(), msides, slack) \
            if has_events else []
        errs = [abs(p[2]) for p in pairs]
        known = [(sides[usable_k[p[0]]], msides[p[1]]) for p in pairs if sides[usable_k[p[0]]] and msides[p[1]]]
        lags2 = [p[3] for p in pairs if p[4] == 2]
        truth = n_ft if n_ft == n_ft else float(len(minutes))
        # the score: the 90-minute one, else the goal events' where every one names its side
        real = (fth[i], fta[i]) if n_ft == n_ft else \
            (float(msides.count("home")), float(msides.count("away"))) if all(msides) else None
        if len(walls) != truth:
            score_agrees = False
        elif real is None or not all(sides):
            score_agrees = None                         # the count agrees; a side is not known
        else:
            score_agrees = (sides.count("home"), sides.count("away")) == real
        # a pair is within tolerance by the football minute: the wall clock's error forgives any half-time lag
        # in its band goal by goal, though the lag is one number a match
        mm_errs, tol_errs = [], []
        for a, b, err, lag, half in pairs:
            k = usable_k[a]
            mm = mms[k]
            mm_err = mm - (min(minutes[b], 45.0 if half == 1 else 90.0) - 0.5) if mm == mm else np.nan
            if mm_err == mm_err:
                mm_errs.append(abs(mm_err))
            tol_errs.append(abs(mm_err) if mm_err == mm_err else abs(err))
            matched.append({"event_id": eid, "goal_no": nos[k], "apif_minute": minutes[b], "wall_min": walls[k],
                            "half": half, "lag_min": lag, "minute_err": err,
                            "side_agrees": (sides[k] == msides[b]) if sides[k] and msides[b] else None,
                            "match_min": mm, "match_min_err": mm_err})
        rows.append({
            "event_id": eid, "apif_fixture_id": int(fid[i]), "clock": "actual" if actual else "fixture",
            "usable": usable.get(eid), "has_events": has_events, "n_inferred": len(walls),
            "n_untimed": len(walls) - sum(clocked),
            "n_apif": len(minutes) if has_events else None,
            "n_ft": n_ft if n_ft == n_ft else None, "count_diff": int(len(walls) - truth),
            "count_agrees": len(walls) == truth, "score_agrees": score_agrees, "n_matched": len(pairs),
            "n_within_tol": sum(e <= MINUTE_TOL for e in tol_errs), "n_side_known": len(known),
            "n_side_agree": sum(x == y for x, y in known), "minute_mae": float(np.mean(errs)) if errs else np.nan,
            "minute_err_max": max(errs) if errs else np.nan,
            "match_min_mae": float(np.mean(mm_errs)) if mm_errs else np.nan,
            "ht_lag_min": float(np.median(lags2)) if lags2 else np.nan})
    per_event = pd.DataFrame(rows, columns=GOAL_CHECK.names)
    for c in ("n_apif", "n_ft"):
        per_event[c] = pd.to_numeric(per_event[c], errors="coerce").astype("Int64")
    for c in ("usable", "score_agrees"):
        per_event[c] = per_event[c].astype("boolean")
    return per_event, pd.DataFrame(matched, columns=goal_cols)


def _goal_numbers(per_event: pd.DataFrame, per_goal: pd.DataFrame) -> dict:
    n = int(len(per_event))
    pg = per_goal[per_goal["event_id"].isin(per_event["event_id"])] if len(per_goal) else per_goal
    err = pg["minute_err"].abs() if len(pg) else pd.Series(dtype=float)
    mm = pg["match_min_err"].abs().dropna() if len(pg) else pd.Series(dtype=float)
    ev = per_event[per_event["has_events"]]
    apif, inferred = int(ev["n_apif"].fillna(0).sum()), int(ev["n_inferred"].sum())
    untimed = int(ev["n_untimed"].sum())
    within = int(ev["n_within_tol"].sum())
    lag = per_event["ht_lag_min"].dropna()
    known, agree = int(per_event["n_side_known"].sum()), int(per_event["n_side_agree"].sum())
    sc = per_event["score_agrees"].dropna()

    def q(x: pd.Series, p: float):
        return round(float(x.quantile(p)), 2) if len(x) else None

    return {
        "matches_checked": n, "matches_with_events": int(len(ev)),
        "count_agrees": int(per_event["count_agrees"].sum()),
        "count_agrees_share": _share(int(per_event["count_agrees"].sum()), n),
        "score_known": int(len(sc)), "score_agrees": int(sc.sum()),
        "score_agrees_share": _share(int(sc.sum()), len(sc)),
        "count_diff": {str(k): int(v) for k, v in per_event["count_diff"].clip(-3, 3).value_counts().sort_index()
                       .items()},
        "goal_events": apif, "goals_inferred": inferred, "goals_matched": int(len(pg)),
        "goals_untimed": untimed,
        "goals_within_tol": within, "recall_within_tol": _share(within, apif),
        "precision_within_tol": _share(within, inferred - untimed),
        "minute_abs_err_median": q(err, 0.5), "minute_abs_err_p90": q(err, 0.9),
        "match_min_within_tol_share": round(float((mm <= MINUTE_TOL).mean()), 4) if len(mm) else None,
        "match_min_abs_err_median": q(mm, 0.5), "match_min_abs_err_p90": q(mm, 0.9),
        "side_known": known, "side_agrees_share": _share(agree, known),
        "ht_lag_min_quartiles": [round(float(lag.quantile(p)), 1) for p in (0.25, 0.5, 0.75)] if len(lag) else None,
    }


def goal_summary(per_event: pd.DataFrame, per_goal: pd.DataFrame) -> dict:
    """The goal check in a few numbers. Matches: how many were checked, how often the goal count and the exact
    score agree. Goals, over the matches whose goal events are held: recall (real goals an inferred goal pairs
    with inside ``MINUTE_TOL`` football minutes, of all real goals) and precision (the same pairs, of the
    inferred goals the prices can time); the minute error of the pairs on the wall clock and of gold/goal's
    ``match_min``; how often the scoring side agrees; the half-time lag. ``usable``: the same over
    gold/timeline's usable matches. (The second review round: counted on the wall clock, where any half-time
    lag in its band is forgiven goal by goal, and with the untimed goals paired, recall and precision on usable
    matches read 0.89 and 0.92 against 0.81 and 0.86 here.)"""
    if not len(per_event):
        return {"matches_checked": 0}
    out = _goal_numbers(per_event, per_goal)
    out["tolerance_min"] = MINUTE_TOL
    ok = per_event[per_event["usable"].fillna(False).astype(bool)]
    if per_event["usable"].notna().any():
        out["usable"] = _goal_numbers(ok, per_goal) if len(ok) else {"matches_checked": 0}
    return out


# --------------------------------------------------------------------------------------------------------------------
# the report, the store wrapper, the command line
# --------------------------------------------------------------------------------------------------------------------

def _share(a: int, b: int):
    return round(a / b, 4) if b else None


WEAKEST = 10              # links listed a method in the report's weakest_links
LONG_KEYS = ("weakest_links", "by_month", "by_country")   # the report's long parts: printed last


def _side_by_side(j: pd.DataFrame) -> list[dict]:
    """Links as the report lists them: both sources' names side by side, with what the settlements say."""
    def flag(v):
        return None if v is None or v is pd.NA or v != v else bool(v)

    def num(v, digits: int):
        return None if v is None or v is pd.NA or v != v else round(float(v), digits)

    return [{"event_id": int(r.event_id), "method": r.method, "score": num(r.score, 3),
             "betfair": f"{r.home_name} v {r.away_name}", "api_football": f"{r.apif_home_name} v {r.apif_away_name}",
             "name_similarity": [num(r.sim_home, 2), num(r.sim_away, 2)],
             "kickoff_diff_min": num(r.kickoff_diff_min, 1), "result": r.result if r.result == r.result else None,
             "apif_score": None if pd.isna(r.apif_ft_home) else f"{int(r.apif_ft_home)}-{int(r.apif_ft_away)}",
             "result_agrees": flag(r.result_agrees), "over25_agrees": flag(r.over25_agrees)}
            for r in j.itertuples(index=False)]


def report(events: pd.DataFrame, links: pd.DataFrame, team_links: pd.DataFrame, stats: dict | None = None) -> dict:
    """What linked: in all, by method, by month of kick-off, by Betfair country code; where the unlinked events
    are; the team ids mapped and the ones whose matches disagree. And how far to trust it: by method, the
    settlements that agree (``checks_by_method``); every link whose score is not the result or the over/under
    Betfair settled (``disagreements``: exact links only, the other methods drop them); the candidates the
    settlements vetoed (``vetoed``); and each method's weakest links, the names side by side (``weakest_links``:
    lowest score first, then the least alike names, one link a pair of names; for exact links, the widest
    kick-off gaps)."""
    ev = events.reindex(columns=["event_id", "home_name", "away_name", "kickoff", "country_code", "result"])
    j = ev.merge(links.reindex(columns=[
        "event_id", "linked", "method", "score", "result_agrees", "over25_agrees", "kickoff_diff_min", "match_id",
        "apif_home_name", "apif_away_name", "apif_ft_home", "apif_ft_away"]), on="event_id", how="left")
    j["linked"] = j["linked"].fillna(False).astype(bool)
    j["method"] = j["method"].fillna("none")
    j["month"] = pd.to_datetime(j["kickoff"], utc=True).dt.strftime("%Y-%m").fillna("(none)")
    j["country"] = _text(j["country_code"]).fillna("").replace("", "(none)")
    n, n_linked = int(len(j)), int(j["linked"].sum())

    def by(col: str) -> dict:
        g = j.groupby(col)["linked"].agg(["size", "sum"])
        return {str(k): {"events": int(r["size"]), "linked": int(r["sum"]), "share": _share(int(r["sum"]),
                                                                                               int(r["size"]))}
                for k, r in g.iterrows()}

    def check(col: str, d: pd.DataFrame) -> dict:
        return {"known": int(d[col].notna().sum()), "agree": int(d[col].fillna(False).astype(bool).sum())}

    countries = by("country")
    unlinked = j[~j["linked"]]["country"].value_counts().head(20)
    on = j[j["linked"]].copy()
    gap = on["kickoff_diff_min"].abs()
    on["absgap"] = gap
    # how alike the names are: only for the links the report lists (the comparison is the slow part)
    methods = [m for m in METHODS if (on["method"] == m).any()]
    wrong = on[on["result_agrees"].astype("boolean").eq(False).fillna(False)
               | on["over25_agrees"].astype("boolean").eq(False).fillna(False)]
    weak = {m: (on[on["method"] == m].sort_values(["absgap", "event_id"], ascending=[False, True]).head(WEAKEST)
                if m in ("exact", "alias") else on[on["method"] == m]) for m in methods}
    listed = pd.concat([wrong] + list(weak.values())).drop_duplicates("event_id") if len(on) else on
    sim = {r.event_id: (side_score(name_key(r.home_name), name_key(r.apif_home_name)),
                        side_score(name_key(r.away_name), name_key(r.apif_away_name)))
           for r in listed.itertuples(index=False)}

    def named(d: pd.DataFrame) -> pd.DataFrame:
        return d.assign(sim_home=[sim[e][0] for e in d["event_id"]], sim_away=[sim[e][1] for e in d["event_id"]])

    weakest = {}
    for m in methods:
        d = named(weak[m])
        if m not in ("exact", "alias"):                 # one link a pair of names: the less alike side's
            home = (d["sim_home"] <= d["sim_away"]).to_numpy()
            pair = np.where(home, d["home_name"].astype(str) + "=" + d["apif_home_name"].astype(str),
                            d["away_name"].astype(str) + "=" + d["apif_away_name"].astype(str))
            d = d.assign(sim_min=d[["sim_home", "sim_away"]].min(axis=1), pair=pair).sort_values(
                ["score", "sim_min", "event_id"]).drop_duplicates("pair").head(WEAKEST)
        weakest[m] = _side_by_side(d)
    tl = team_links
    mapped = tl["apif_team_id"].notna() & ~tl["low_agreement"]
    low = tl[tl["low_agreement"]].sort_values(["n_linked", "team_id"], ascending=[False, True])
    return {
        "events": n, "linked": {"n": n_linked, "share": _share(n_linked, n)},
        "by_method": {m: int((j["method"] == m).sum()) for m in METHODS + ("none",) if (j["method"] == m).any()},
        "result_check": check("result_agrees", j), "over25_check": check("over25_agrees", j),
        "checks_by_method": {m: {"linked": int((on["method"] == m).sum()),
                                 "result": check("result_agrees", on[on["method"] == m]),
                                 "over25": check("over25_agrees", on[on["method"] == m]),
                                 "min_score": round(float(on.loc[on["method"] == m, "score"].min()), 3),
                                 "max_kickoff_gap_min": round(float(gap[on["method"] == m].max()), 1)}
                             for m in methods},
        "vetoed": dict((stats or {}).get("vetoed", {})),
        "disagreements": _side_by_side(named(wrong.sort_values("event_id"))),
        "kickoff_gap_min": {"within_1": int((gap <= 1).sum()), "within_30": int((gap <= TOL_MIN).sum()),
                            "over_30": int((gap > TOL_MIN).sum())},
        "canonical_matches": int(j["match_id"].notna().sum()),
        "teams": {"team_ids": int(len(tl)), "mapped": int(mapped.sum()), "share": _share(int(mapped.sum()), len(tl)),
                  "low_agreement": int(tl["low_agreement"].sum()),
                  "several_apif_ids": int((tl["n_apif_ids"] > 1).sum()),
                  "low_agreement_top": [{"team_id": int(r.team_id), "team_name": r.team_name,
                                         "n_linked": int(r.n_linked), "share": round(float(r.share), 3)}
                                        for r in low.head(20).itertuples(index=False)]},
        "unlinked_top_countries": {str(k): int(v) for k, v in unlinked.items()},
        # the long parts (LONG_KEYS) last: a summary that is cut short keeps the checks
        "weakest_links": weakest, "by_month": by("month"),
        "by_country": dict(sorted(countries.items(), key=lambda kv: (-kv[1]["events"], kv[0]))),
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


def _frame(store, key: str, columns: list[str], since=None, until=None, ids=None) -> pd.DataFrame | None:
    """A stored parquet's columns of interest (those it has) as a frame, or None when the key is not there.
    ``since`` / ``until``: only rows whose ``kickoff_utc`` is between them (the fixtures table holds ten years).
    ``ids``: only rows whose ``apif_fixture_id`` is one of them (the events table holds every league's)."""
    raw = store.get(key)
    if raw is None:
        return None
    have = pq.read_schema(io.BytesIO(raw)).names
    cols = [c for c in columns if c in have]
    t = pq.read_table(io.BytesIO(raw), columns=cols)
    if ids is not None and "apif_fixture_id" in cols:
        want = pa.array(sorted({int(x) for x in ids}), type=pa.int64())
        t = t.filter(pc.is_in(pc.cast(t["apif_fixture_id"], pa.int64(), safe=False), value_set=want))
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
    link, write gold/link, gold/team_link and (when gold/goal exists) gold/goal_check; the report. The goal check
    reads API-Football's goal events from ``api_football/history_events.parquet`` (built from the raw fixture files
    by ``sources.api_football_history.build``) and, where it is there, gold/timeline for the matches gold/goal
    covers. An input that is not there is named in ``missing`` ('no data found: <key>') and the stage does what it
    can without it."""
    from football.warehouse.store_io import (fingerprint, read_single, read_table, single_key, utc_now,
                                             write_manifest, write_single)
    rep: dict = {"version": VERSION, "missing": [], "written": {}}

    def missing(key: str) -> None:
        rep["missing"].append(f"no data found: {key}")

    ev_cols = ["event_id", "home_name", "away_name", "home_id", "away_id", "country_code", "kickoff",
               "kickoff_actual", "went_in_play", "result", "over25"]
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
    stats: dict = {}
    links = canonical(link_events(events, fx, aliases, stats=stats), canon, fx, aliases, events)
    team_links = team_table(team_map(links, events), teams, links, events, canon)
    rep.update(report(events, links, team_links, stats))

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
        he = _frame(store, HISTORY_EVENTS_KEY, ["apif_fixture_id", "minute", "type", "detail", "apif_team_id"],
                    ids=links["apif_fixture_id"].dropna())
        if he is None:
            missing(HISTORY_EVENTS_KEY)
        timeline = read_table(store, "gold/timeline", ["event_id", "usable"])
        if timeline is None:
            timeline = read_single(store, "gold/timeline", ["event_id", "usable"])
        if timeline is None:
            missing(single_key("gold/timeline").rsplit("/", 1)[0])
        el = links.merge(events[["event_id", "kickoff_actual", "went_in_play"]], on="event_id")
        keep = [c for c in ("event_id", "goal_no", "pt", "elapsed_min", "side", "match_min", "timed")
                if c in goals.column_names]
        per_event, per_goal = goal_check(goals.select(keep).to_pandas(), el, he,
                                         None if timeline is None else timeline.to_pandas())
        rep["goal_check"] = goal_summary(per_event, per_goal)
        out[GOAL_CHECK] = per_event
    rep["status"] = "ok"
    for k in LONG_KEYS:                                 # after the goal check and the status: the long parts last
        rep[k] = rep.pop(k)
    if write:
        fp = fingerprint([f"v{VERSION}", f"events:{len(events)}", f"fixtures:{len(fx)}",
                          f"canonical:{0 if canon is None else len(canon)}"])
        for spec, df in out.items():
            wrote = write_single(store, spec, to_arrow(spec, df))
            write_manifest(store, spec.name, {"parts": {"all": {"rows": wrote["rows"], "bytes": wrote["bytes"],
                                                                "fingerprint": fp, "version": VERSION,
                                                                "built_utc": utc_now()}}})
            rep["written"][spec.name] = wrote["rows"]
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
