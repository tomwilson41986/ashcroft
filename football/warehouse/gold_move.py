"""gold/move: the move from each checkpoint to the close: the target a closing-price model trains on.

One row a runner a checkpoint (never the close itself) where the runner's book is complete at the checkpoint and
at the close: the log and logit change of its normalised chance, the raw price ratio, and what says how far to
trust both ends (the books' stalest prices, the activity before and after the checkpoint). ``usable`` is the
default screen: the checkpoint's book no staler than a quarter of the time left to the off (15 minutes at least)
and the closing book no staler than 15 minutes. ``tier`` is the match's liquidity by its pre-off match-odds
price updates (BASIC has no volume): A 300 or more, B 120 to 299, C 40 to 119, D under 40.

Match odds and over/under: every runner. Asian handicap: every line complete at both ends, with ``is_main`` on the
line nearest evens at the checkpoint (a market's main line).

A move is in the chance: positive means the runner shortened (its chance rose) between the checkpoint and the off.
"""

from __future__ import annotations

import pyarrow as pa

from football.warehouse.schema import TS, Column, TableSpec

VERSION = 1
SCOPE = "part"
INPUTS = ("gold/checkpoint", "gold/event")
COLUMNS = {"gold/event": ["event_id", "kickoff", "mo_ticks_pre", "country_code", "result", "over25"]}

TIER_CUTS = (300, 120, 40)            # pre-off match-odds price updates: A, B, C, else D
MIN_AGE_S = 900.0                     # a checkpoint book may always be this stale
AGE_SHARE = 0.25                      # ... or this share of the time left to the off
CLOSE_AGE_S = 900.0

SPEC = TableSpec(
    "gold/move",
    "One row a runner a checkpoint: the move of its normalised chance from the checkpoint to the close, with the "
    "quality of both ends. The training target of a closing-price model.",
    (
        Column("market_id", pa.string(), "Betfair's market id.", False),
        Column("selection_id", pa.int64(), "Betfair's selection id.", False),
        Column("handicap", pa.float64(), "The runner's handicap line; 0.0 where it has none.", False),
        Column("cp_min", pa.int32(), "The checkpoint: minutes before the scheduled kick-off (never 0).", False),
        Column("event_id", pa.int64(), "Betfair's event id.", False),
        Column("market_type", pa.string(), "The market's type.", False),
        Column("role", pa.string(), "home, away, draw, over or under."),
        Column("line", pa.float64(), "The book: 0.0, or the home side's handicap line.", False),
        Column("is_main", pa.bool_(), "Asian handicap: the line nearest evens at the checkpoint. True for match "
               "odds and over/under.", False),
        Column("kickoff", TS, "The scheduled kick-off."),
        Column("cp_time", TS, "The checkpoint's moment.", False),
        Column("ltp_cp", pa.float64(), "The price at the checkpoint.", False),
        Column("ltp_close", pa.float64(), "The closing price.", False),
        Column("p_cp", pa.float64(), "The normalised chance at the checkpoint.", False),
        Column("p_close", pa.float64(), "The normalised chance at the close.", False),
        Column("y_log", pa.float64(), "ln(p_close / p_cp): the move, in log chance.", False),
        Column("y_logit", pa.float64(), "logit(p_close) - logit(p_cp): the move, in log odds.", False),
        Column("price_ratio", pa.float64(), "ltp_cp / ltp_close: above 1 when the price shortened.", False),
        Column("cp_book_sum", pa.float64(), "The book sum at the checkpoint.", False),
        Column("cp_age_s", pa.float64(), "The runner's own price age at the checkpoint, seconds.", False),
        Column("cp_book_age_s", pa.float64(), "The stalest price of its book at the checkpoint, seconds.", False),
        Column("close_book_age_s", pa.float64(), "The stalest price of its book at the close, seconds.", False),
        Column("n_ticks_before", pa.int32(), "The runner's price updates up to the checkpoint.", False),
        Column("n_ticks_after", pa.int32(), "Its price updates between the checkpoint and the close.", False),
        Column("usable", pa.bool_(), "The default screen on staleness at both ends (see the module).", False),
        Column("mo_ticks_pre", pa.int32(), "The match's pre-off match-odds price updates."),
        Column("tier", pa.string(), "The match's liquidity tier: A, B, C or D.", False),
        Column("country_code", pa.string(), "Betfair's country code."),
        Column("won", pa.int8(), "Whether the runner won, where the role says so from the event's result (match "
               "odds and over/under); blank otherwise."),
    ),
    key=("market_id", "selection_id", "handicap", "cp_min"),
    sort=("cp_min", "market_id", "selection_id", "handicap"),
)
SPECS = (SPEC,)

QUERY = """
WITH c AS (SELECT * FROM cp WHERE cp_min > 0 AND book_complete),
z AS (SELECT * FROM cp WHERE cp_min = 0 AND book_complete),
main AS (                                   -- an Asian handicap market's main line at a checkpoint: nearest evens
    SELECT market_id, cp_min, arg_min(line, abs(p_norm - 0.5)) AS main_line
    FROM c WHERE market_type = 'ASIAN_HANDICAP' AND role = 'home' GROUP BY 1, 2)
SELECT c.market_id, c.selection_id, c.handicap, c.cp_min, c.event_id, c.market_type, c.role, c.line,
       CASE WHEN c.market_type = 'ASIAN_HANDICAP' THEN coalesce(c.line = main.main_line, FALSE) ELSE TRUE END
           AS is_main,
       e.kickoff, c.cp_time, c.ltp AS ltp_cp, z.ltp AS ltp_close, c.p_norm AS p_cp, z.p_norm AS p_close,
       ln(z.p_norm / c.p_norm) AS y_log,
       ln(z.p_norm / (1 - z.p_norm)) - ln(c.p_norm / (1 - c.p_norm)) AS y_logit,
       c.ltp / z.ltp AS price_ratio,
       c.book_sum AS cp_book_sum, c.age_s AS cp_age_s, c.book_max_age_s AS cp_book_age_s,
       z.book_max_age_s AS close_book_age_s,
       c.n_ticks_before, z.n_ticks_before - c.n_ticks_before AS n_ticks_after,
       c.book_max_age_s <= greatest({min_age}, {share} * c.cp_min * 60.0)
           AND z.book_max_age_s <= {close_age} AS usable,
       e.mo_ticks_pre,
       CASE WHEN e.mo_ticks_pre >= {a} THEN 'A' WHEN e.mo_ticks_pre >= {b} THEN 'B'
            WHEN e.mo_ticks_pre >= {c} THEN 'C' ELSE 'D' END AS tier,
       e.country_code,
       CAST(CASE c.role WHEN 'home' THEN CASE WHEN c.market_type = 'MATCH_ODDS' THEN (e.result = 'H')::INT END
                        WHEN 'draw' THEN (e.result = 'D')::INT
                        WHEN 'away' THEN CASE WHEN c.market_type = 'MATCH_ODDS' THEN (e.result = 'A')::INT END
                        WHEN 'over' THEN e.over25
                        WHEN 'under' THEN 1 - e.over25 END AS TINYINT) AS won
FROM c
JOIN z USING (market_id, selection_id, handicap)
LEFT JOIN event e ON e.event_id = c.event_id
LEFT JOIN main ON main.market_id = c.market_id AND main.cp_min = c.cp_min
WHERE c.p_norm > 0 AND c.p_norm < 1 AND z.p_norm > 0 AND z.p_norm < 1
"""


def build(inputs: dict) -> dict:
    from football.warehouse.gold import sql
    a, b, c = TIER_CUTS
    out = sql(QUERY.format(min_age=MIN_AGE_S, share=AGE_SHARE, close_age=CLOSE_AGE_S, a=a, b=b, c=c),
              cp=inputs["gold/checkpoint"], event=inputs["gold/event"])
    return {SPEC.name: out}
