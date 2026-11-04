"""The modelling frame: Cricsheet tables plus the state before every delivery, with the labels the models need.

Reads the parquet tables cricket/ingest/cricsheet.py writes (a local folder or the S3 store). Scope filters are
explicit so every report states what it covers.
"""

from __future__ import annotations

import pandas as pd

from cricket.state.match_state import states_frame
from sources.common import Store

TABLES = "cricket/tables"
IPL = "Indian Premier League"


def load(store: Store, names=("matches", "innings", "deliveries", "players")) -> dict[str, pd.DataFrame]:
    out = {}
    for n in names:
        df = store.get_parquet(f"{TABLES}/{n}.parquet")
        if df is None:
            raise SystemExit(f"no data found: {store.where(f'{TABLES}/{n}.parquet')} (run cricket.ingest.cricsheet)")
        out[n] = df
    return out


def scope_matches(m: pd.DataFrame, match_types=("T20",), gender="male", event: str | None = None,
                  decided_only=True, no_dls=True) -> pd.DataFrame:
    """Matches in scope. ``decided_only`` drops no-results and ties (a tie's winner comes from a super over, not the
    innings modelled); ``no_dls`` drops revised-target (D/L, VJD) matches."""
    s = m[m.match_type.isin(match_types) & (m.gender == gender)].copy()
    if event:
        s = s[s.event_name == event]
    if decided_only:
        s = s[s.winner.notna() & s.result.isna()]
    if no_dls:
        s = s[s.method.isna()]
    s["year"] = pd.to_datetime(s.start_date).dt.year
    return s


def ball_frame(t: dict[str, pd.DataFrame], matches: pd.DataFrame) -> pd.DataFrame:
    """The state before every delivery of the scoped matches' two main innings, with the outcome labels:
    ``bat_won`` (the batting side won the match) and the delivery's own result columns."""
    ids = set(matches.match_id)
    inn = t["innings"][t["innings"].match_id.isin(ids) & ~t["innings"].super_over.astype(bool)]
    inn = inn[inn.innings <= 2]
    d = t["deliveries"][t["deliveries"].match_id.isin(ids)]
    d = d.merge(inn[["match_id", "innings"]], on=["match_id", "innings"])
    f = states_frame(d, inn, matches)
    f = f.merge(inn[["match_id", "innings", "team"]], on=["match_id", "innings"], how="left")
    f = f.merge(matches[["match_id", "winner", "year", "start_date", "venue", "event_name", "team1", "team2"]],
                on="match_id", how="left")
    f["bat_won"] = (f.team == f.winner).astype(int)
    f["bowl_team"] = f.team1.where(f.team != f.team1, f.team2)
    return f
