"""Reference data: the competitions in scope, canonical team ids, and team-name matching across sources.

football-data.co.uk's team names are the canonical display names (``team_id`` is ``<country>:<normalised fd name>``).
Every other source (Betfair, openfootball, API-Football) is matched to them in three steps:

1. a stored alias (``football/team_aliases.csv`` in this package, reviewed by hand, plus the aliases learned from
   earlier unique matches, ``sources/football/gold/team_aliases.parquet``);
2. the normalised name (accents, punctuation and club suffixes such as FC/AFC/CF dropped, the common abbreviations
   expanded: Utd, Man, Nottm, Sheff ...);
3. a fuzzy score, only ever among the teams that play on that day in that country (a fixture on a date is the
   constraint that makes fuzzy matching safe), and only when the best pairing is unique.

Names are always keyed on country as well as name: Liverpool (ENG) and Liverpool (URU) are different clubs.
"""

from __future__ import annotations

import csv
import difflib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

ALIASES_PATH = Path(__file__).with_name("team_aliases.csv")


@dataclass(frozen=True)
class Competition:
    competition_id: str
    fd_code: str
    name: str
    country: str          # ISO-ish country name used for pools and team ids
    tier: int
    betfair_country: str  # Betfair's event countryCode
    apif_league_id: int | None = None


_C = Competition
COMPETITIONS: tuple[Competition, ...] = (
    _C("ENG1", "E0", "Premier League", "ENG", 1, "GB", 39), _C("ENG2", "E1", "Championship", "ENG", 2, "GB", 40),
    _C("ENG3", "E2", "League One", "ENG", 3, "GB", 41), _C("ENG4", "E3", "League Two", "ENG", 4, "GB", 42),
    _C("ENG5", "EC", "National League", "ENG", 5, "GB", 43),
    _C("SCO1", "SC0", "Premiership", "SCO", 1, "GB", 179), _C("SCO2", "SC1", "Championship", "SCO", 2, "GB", 180),
    _C("SCO3", "SC2", "League One", "SCO", 3, "GB", 183), _C("SCO4", "SC3", "League Two", "SCO", 4, "GB", 184),
    _C("GER1", "D1", "Bundesliga", "GER", 1, "DE", 78), _C("GER2", "D2", "2. Bundesliga", "GER", 2, "DE", 79),
    _C("ITA1", "I1", "Serie A", "ITA", 1, "IT", 135), _C("ITA2", "I2", "Serie B", "ITA", 2, "IT", 136),
    _C("ESP1", "SP1", "La Liga", "ESP", 1, "ES", 140), _C("ESP2", "SP2", "Segunda", "ESP", 2, "ES", 141),
    _C("FRA1", "F1", "Ligue 1", "FRA", 1, "FR", 61), _C("FRA2", "F2", "Ligue 2", "FRA", 2, "FR", 62),
    _C("NED1", "N1", "Eredivisie", "NED", 1, "NL", 88), _C("BEL1", "B1", "Pro League", "BEL", 1, "BE", 144),
    _C("POR1", "P1", "Primeira Liga", "POR", 1, "PT", 94), _C("TUR1", "T1", "Super Lig", "TUR", 1, "TR", 203),
    _C("GRE1", "G1", "Super League", "GRE", 1, "GR", 197),
    _C("ARG1", "ARG", "Liga Profesional", "ARG", 1, "AR", 128), _C("AUT1", "AUT", "Bundesliga", "AUT", 1, "AT", 218),
    _C("BRA1", "BRA", "Serie A", "BRA", 1, "BR", 71), _C("CHN1", "CHN", "Super League", "CHN", 1, "CN", 169),
    _C("DEN1", "DNK", "Superliga", "DEN", 1, "DK", 119), _C("FIN1", "FIN", "Veikkausliiga", "FIN", 1, "FI", 244),
    _C("IRL1", "IRL", "Premier Division", "IRL", 1, "IE", 357), _C("JPN1", "JPN", "J1 League", "JPN", 1, "JP", 98),
    _C("MEX1", "MEX", "Liga MX", "MEX", 1, "MX", 262), _C("NOR1", "NOR", "Eliteserien", "NOR", 1, "NO", 103),
    _C("POL1", "POL", "Ekstraklasa", "POL", 1, "PL", 106), _C("ROU1", "ROU", "Liga I", "ROU", 1, "RO", 283),
    _C("RUS1", "RUS", "Premier League", "RUS", 1, "RU", 235), _C("SWE1", "SWE", "Allsvenskan", "SWE", 1, "SE", 113),
    _C("SUI1", "SWZ", "Super League", "SUI", 1, "CH", 207), _C("USA1", "USA", "MLS", "USA", 1, "US", 253),
)
BY_FD = {c.fd_code: c for c in COMPETITIONS}
BY_ID = {c.competition_id: c for c in COMPETITIONS}
#: the leagues whose teams meet more than twice a season (a split, or four rounds): no fixed match count
MULTI_ROUND = {"SCO1", "SCO2", "SCO3", "SCO4", "AUT1", "DEN1", "SUI1", "IRL1", "BEL1", "ROU1", "POL1", "FIN1", "USA1",
               "MEX1", "ARG1", "GRE1", "NED1"}

_SUFFIX = {"fc", "afc", "cf", "sc", "ac", "fk", "sk", "if", "bk", "cd", "sd", "ud", "club", "the", "calcio", "ssc",
           "as", "us", "sv", "vfb", "vfl", "tsg", "rcd", "rc", "ogc", "losc", "kv", "krc", "aa", "ca", "cr", "se",
           "ec", "1", "de", "fbc", "bc", "cfc", "jk", "ik", "aik", "sl", "ff", "kas", "kaa", "rsc", "sco", "ss",
           "acf", "asd", "ad", "sa", "sp", "nk", "hnk", "gnk", "afk", "mfk"}
_EXPAND = {"utd": "united", "man": "manchester", "nottm": "nottingham", "sheff": "sheffield", "wolves":
           "wolverhampton", "spurs": "tottenham", "qpr": "queens park rangers", "mk": "milton keynes", "wed":
           "wednesday", "weds": "wednesday", "st": "saint", "ath": "athletic", "atl": "atletico",
           "dep": "deportivo", "inter": "internazionale", "psg": "paris saint germain",
           "munchen": "munich", "monchengladbach": "gladbach", "mgladbach": "gladbach", "ein": "eintracht"}


def strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def normalise(name: str) -> str:
    """'Man United' -> 'manchester united'; 'Nott'm Forest' -> 'nottingham forest'; 'Brighton & Hove Albion FC' ->
    'brighton hove albion'."""
    s = strip_accents(str(name or "")).lower().replace("'", "").replace("&", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    out = []
    for tok in s.split():
        tok = _EXPAND.get(tok, tok)
        for t in tok.split():
            if t and t not in _SUFFIX:
                out.append(t)
    return " ".join(out)


def team_id(country: str, fd_name: str) -> str:
    return f"{country}:{normalise(fd_name)}"


def similarity(a: str, b: str) -> float:
    """Between two normalised names: the better of the character ratio and the token overlap, with a prefix bonus
    (Betfair's 'Wolves' against 'Wolverhampton', 'Sheff Wed' against 'Sheffield Weds')."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    tok = len(ta & tb) / max(1, min(len(ta), len(tb)))
    pre = 0.9 if (a.startswith(b) or b.startswith(a)) else 0.0
    # every token of one is a prefix of a token of the other ('sheffield weds' / 'sheffield wednesday')
    def covered(x, y):
        return all(any(v.startswith(u) or u.startswith(v) for v in y) for u in x)
    cov = 0.85 if covered(ta, tb) or covered(tb, ta) else 0.0
    return max(ratio, 0.95 * tok, pre, cov)


def load_aliases(path: Path = ALIASES_PATH, extra=None) -> dict[tuple[str, str, str], str]:
    """(source, country, normalised source name) -> team_id. ``extra``: rows learned from earlier matches (a
    DataFrame with source, country, source_name, team_id), the reviewed file wins."""
    out: dict[tuple[str, str, str], str] = {}
    if extra is not None and len(extra):
        for r in extra.itertuples(index=False):
            out[(r.source, r.country, normalise(r.source_name))] = r.team_id
    if path.exists():
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                if r.get("source") and r.get("source_name") and r.get("fd_name"):
                    out[(r["source"], r["country"], normalise(r["source_name"]))] = team_id(r["country"], r["fd_name"])
    return out


def resolve(name: str, country: str, candidates: dict[str, str], source: str, aliases=None,
            floor: float = 0.72) -> tuple[str | None, float, str]:
    """One source name -> (team_id, score, method) among ``candidates`` (team_id -> fd name) of that country.
    Alias first, then the exact normalised name, then the fuzzy best when it clears ``floor`` and the runner-up is
    clearly behind; else (None, best score, 'unmatched')."""
    n = normalise(name)
    if aliases:
        tid = aliases.get((source, country, n))
        if tid is not None and (not candidates or tid in candidates):
            return tid, 1.0, "alias"
    for tid, fd in candidates.items():
        if normalise(fd) == n:
            return tid, 1.0, "exact"
    scored = sorted(((similarity(n, normalise(fd)), tid) for tid, fd in candidates.items()), reverse=True)
    if not scored:
        return None, 0.0, "unmatched"
    best = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if best[0] >= floor and best[0] - second >= 0.05:
        return best[1], best[0], "fuzzy"
    return None, best[0], "unmatched"


def pair_event(home: str, away: str, fixtures: list[tuple[str, str, str]], country: str, source: str,
               aliases=None, floor: float = 0.6) -> tuple[int | None, float]:
    """A source's fixture (home, away) -> the index of the one fd fixture it is among ``fixtures`` (match_id,
    home team_id/fd name pairs: (key, home fd name, away fd name)), by the summed similarity of both sides; unique
    or nothing. The fixture list is already narrowed to the day and country, so two sides agreeing is strong."""
    nh, na = normalise(home), normalise(away)
    ah = aliases.get((source, country, nh)) if aliases else None
    aa = aliases.get((source, country, na)) if aliases else None
    scores = []
    for i, (_key, fh, fa) in enumerate(fixtures):
        sh = 1.0 if ah is not None and ah == team_id(country, fh) else similarity(nh, normalise(fh))
        sa = 1.0 if aa is not None and aa == team_id(country, fa) else similarity(na, normalise(fa))
        scores.append((min(sh, sa), sh + sa, i))
    if not scores:
        return None, 0.0
    scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = scores[0]
    second = scores[1][1] if len(scores) > 1 else 0.0
    if best[0] >= floor and best[1] - second >= 0.15:
        return best[2], best[1] / 2
    return None, best[1] / 2
