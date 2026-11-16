"""gold/team: Betfair's teams, one row a team id.

A side's selection id on the match-odds market is the team's own id on Betfair: the same in every match it plays,
whatever the competition. This is the team dimension every team-level table keys on; the link stage adds the
API-Football and canonical ids beside it (gold/team_link) where the match link finds them.
"""

from __future__ import annotations

import pyarrow as pa

from football.warehouse.schema import TS, Column, TableSpec

VERSION = 2          # 2: a tie for the commonest name or country goes to the first in order
SCOPE = "all"
INPUTS = ("gold/event",)
COLUMNS = {"gold/event": ["event_id", "home_id", "away_id", "home_name", "away_name", "country_code", "kickoff",
                          "went_in_play"]}

SPEC = TableSpec(
    "gold/team",
    "One row a Betfair team id (the selection id of a side on the match-odds market).",
    (
        Column("team_id", pa.int64(), "Betfair's selection id for the team.", False),
        Column("team_name", pa.string(), "The name it is most often given."),
        Column("n_names", pa.int32(), "Distinct names seen for the id.", False),
        Column("n_events", pa.int32(), "Matches held for the team.", False),
        Column("n_home", pa.int32(), "Of those, at home.", False),
        Column("n_played", pa.int32(), "Of those, that went in play.", False),
        Column("country_code", pa.string(), "The country code its home matches most often carry."),
        Column("first_kickoff", TS, "Its first match held."),
        Column("last_kickoff", TS, "Its last match held."),
    ),
    key=("team_id",), sort=("team_id",),
)
SPECS = (SPEC,)

QUERY = """
WITH side AS (
    SELECT home_id AS team_id, home_name AS name, country_code, kickoff, went_in_play, 1 AS home FROM event
    WHERE home_id IS NOT NULL
    UNION ALL
    SELECT away_id, away_name, NULL, kickoff, went_in_play, 0 FROM event WHERE away_id IS NOT NULL)
SELECT team_id, mode(name ORDER BY name) AS team_name, count(DISTINCT name) AS n_names, count(*) AS n_events,
       sum(home) AS n_home, sum(went_in_play::INT) AS n_played,
       mode(country_code ORDER BY country_code) AS country_code,
       min(kickoff) AS first_kickoff, max(kickoff) AS last_kickoff
FROM side GROUP BY team_id
"""


def build(inputs: dict) -> dict:
    from football.warehouse.gold import sql
    return {SPEC.name: sql(QUERY, event=inputs["gold/event"])}
