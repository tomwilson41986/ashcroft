"""gold/event: the match as Betfair knows it, one row an event.

From a partition's markets and runners: the match's names and its two sides' Betfair ids (the selection ids of the
match-odds runners: a team keeps its id in every market it plays), the scheduled and the actual kick-off, the
markets it has, the result as the market settled it (1X2 from the match-odds winner, over 2.5 from the over/under
winner), how much it traded, and the closing 1X2 chances with their quality (legs priced, book sum, the stalest
leg's age at the off).

Where an event has two markets of a type (Betfair voided and re-made one), the one that traded more is the event's.
"""

from __future__ import annotations

import pyarrow as pa

from football.warehouse.schema import TS, Column, TableSpec

VERSION = 2
SCOPE = "part"
INPUTS = ("silver/market", "silver/runner")

SPEC = TableSpec(
    "gold/event",
    "One row a Betfair event (a football match): sides, kick-off, markets, result, trading, the closing 1X2.",
    (
        Column("event_id", pa.int64(), "Betfair's event id.", False),
        Column("event_name", pa.string(), "'Home v Away' as Betfair names the match."),
        Column("home_name", pa.string(), "The side named first."),
        Column("away_name", pa.string(), "The side named second."),
        Column("home_id", pa.int64(), "The home side's Betfair selection id: the team's id across matches."),
        Column("away_id", pa.int64(), "The away side's Betfair selection id."),
        Column("country_code", pa.string(), "Betfair's country code; blank on many cups and internationals."),
        Column("category", pa.string(), "senior, women, youth or reserves, read from the sides' names ('(W)', "
               "'U21', '(Res)', a name ending ' II' or ' B', 'Jong'): approximate; the kind of football, for "
               "filters and for club work.", False),
        Column("kickoff", TS, "The scheduled kick-off (the match-odds market's market_time; else the earliest of "
               "the event's markets)."),
        Column("kickoff_actual", TS, "When the match-odds market went in play."),
        Column("kickoff_delay_s", pa.int32(), "kickoff_actual less kickoff, seconds."),
        Column("kickoff_shift_s", pa.int32(), "The scheduled kick-off's change over the market's life, seconds "
               "(last less first definition): a reschedule when large."),
        Column("mo_market_id", pa.string(), "The match-odds market."),
        Column("ou_market_id", pa.string(), "The over/under 2.5 goals market."),
        Column("ah_market_id", pa.string(), "The Asian handicap market."),
        Column("n_markets", pa.int16(), "Markets held for the event.", False),
        Column("went_in_play", pa.bool_(), "Whether the match-odds market went in play.", False),
        Column("result", pa.string(), "H, D or A from the match-odds winner; blank if void or unsettled."),
        Column("over25", pa.int8(), "1 if Over 2.5 Goals won, 0 if Under won; blank if void or unsettled."),
        Column("void", pa.bool_(), "The match-odds market closed without a winner (abandoned, postponed).", False),
        Column("settled_time", TS, "When Betfair settled the match-odds market."),
        Column("ft_pt", TS, "When the match-odds market closed: full time, within a few minutes."),
        Column("mo_ticks_pre", pa.int32(), "Match-odds price updates before the off: a liquidity proxy."),
        Column("mo_ticks_inplay", pa.int32(), "Match-odds price updates in play."),
        Column("ou_ticks_pre", pa.int32(), "Over/under price updates before the off."),
        Column("ou_ticks_inplay", pa.int32(), "Over/under price updates in play."),
        Column("ah_ticks_pre", pa.int32(), "Asian handicap price updates before the off."),
        Column("ah_ticks_inplay", pa.int32(), "Asian handicap price updates in play."),
        Column("mo_susp_inplay", pa.int16(), "In-play suspensions of the match-odds market."),
        Column("close_legs", pa.int8(), "Match-odds runners with a price before the off (3: a full closing book)."),
        Column("close_book_sum", pa.float64(), "Sum of 1 / closing price over the match-odds runners priced."),
        Column("close_age_s", pa.float64(), "Age at the off of the stalest closing price, seconds."),
        Column("p_home", pa.float64(), "The closing chance of a home win, normalised to the book (full books only)."),
        Column("p_draw", pa.float64(), "The closing chance of a draw."),
        Column("p_away", pa.float64(), "The closing chance of an away win."),
    ),
    key=("event_id",), sort=("event_id",),
)
SPECS = (SPEC,)

QUERY = r"""
WITH pick AS (                      -- the event's market of each type: the one that traded more
    SELECT * FROM market
    QUALIFY row_number() OVER (PARTITION BY event_id, market_type ORDER BY n_ticks DESC, market_id DESC) = 1),
ev AS (
    SELECT event_id, count(*) AS n_markets, min(market_time) AS first_time,
           arg_max(event_name, n_ticks) AS event_name, arg_max(home_name, n_ticks) AS home_name,
           arg_max(away_name, n_ticks) AS away_name, max(country_code) AS country_code
    FROM market GROUP BY event_id),
mo AS (SELECT * FROM pick WHERE market_type = 'MATCH_ODDS'),
ou AS (SELECT * FROM pick WHERE market_type = 'OVER_UNDER_25'),
ah AS (SELECT * FROM pick WHERE market_type = 'ASIAN_HANDICAP'),
mor AS (
    SELECT market_id,
           max(CASE WHEN role = 'home' THEN selection_id END) AS home_id,
           max(CASE WHEN role = 'away' THEN selection_id END) AS away_id,
           max(CASE WHEN won = 1 THEN role END) AS win_role,
           sum((won = 1)::INT) AS winners,
           count(ltp_last_pre) AS close_legs,
           sum(1.0 / ltp_last_pre) AS close_book_sum,
           min(last_pre_trade_pt) FILTER (WHERE ltp_last_pre IS NOT NULL) AS stalest_pt,
           max(CASE WHEN role = 'home' THEN ltp_last_pre END) AS c_home,
           max(CASE WHEN role = 'draw' THEN ltp_last_pre END) AS c_draw,
           max(CASE WHEN role = 'away' THEN ltp_last_pre END) AS c_away
    FROM runner WHERE market_type = 'MATCH_ODDS' GROUP BY market_id),
our AS (
    SELECT market_id, max(CASE WHEN role = 'over' THEN won END) AS over25
    FROM runner WHERE market_type = 'OVER_UNDER_25' GROUP BY market_id),
ahr AS (
    SELECT market_id, max(CASE WHEN role = 'home' THEN selection_id END) AS home_id,
           max(CASE WHEN role = 'away' THEN selection_id END) AS away_id
    FROM runner WHERE market_type = 'ASIAN_HANDICAP' GROUP BY market_id)
SELECT ev.event_id, ev.event_name, ev.home_name, ev.away_name,
       coalesce(mor.home_id, ahr.home_id) AS home_id, coalesce(mor.away_id, ahr.away_id) AS away_id,
       ev.country_code,
       CASE WHEN regexp_matches(ev.event_name, '\(W\)|\bWomen\b|\bLadies\b|\bFem\b') THEN 'women'
            WHEN regexp_matches(ev.event_name, '\bU-?(1[5-9]|2[0-3])\b') THEN 'youth'
            WHEN regexp_matches(ev.event_name, '\(Res\)|\bReserves?\b|\bJong ')
                 OR (regexp_matches(coalesce(ev.home_name, ''), '( II| B)$') AND ev.home_name <> 'Willem II')
                 OR (regexp_matches(coalesce(ev.away_name, ''), '( II| B)$') AND ev.away_name <> 'Willem II')
                 THEN 'reserves'
            ELSE 'senior' END AS category,
       coalesce(mo.market_time, ev.first_time) AS kickoff,
       mo.in_play_pt AS kickoff_actual,
       mo.kickoff_delay_s,
       CAST(epoch(mo.market_time - mo.market_time_first) AS INTEGER) AS kickoff_shift_s,
       mo.market_id AS mo_market_id, ou.market_id AS ou_market_id, ah.market_id AS ah_market_id,
       ev.n_markets,
       coalesce(mo.in_play_pt IS NOT NULL, FALSE) AS went_in_play,
       CASE WHEN mor.winners = 1 THEN CASE mor.win_role WHEN 'home' THEN 'H' WHEN 'draw' THEN 'D'
                                                        WHEN 'away' THEN 'A' END END AS result,
       our.over25 AS over25,
       coalesce(mo.status_final = 'CLOSED' AND coalesce(mor.winners, 0) = 0, FALSE) AS void,
       mo.settled_time, mo.closed_pt AS ft_pt,
       mo.n_ticks_pre AS mo_ticks_pre, mo.n_ticks_inplay AS mo_ticks_inplay,
       ou.n_ticks_pre AS ou_ticks_pre, ou.n_ticks_inplay AS ou_ticks_inplay,
       ah.n_ticks_pre AS ah_ticks_pre, ah.n_ticks_inplay AS ah_ticks_inplay,
       mo.n_susp_inplay AS mo_susp_inplay,
       mor.close_legs, mor.close_book_sum,
       epoch(coalesce(mo.in_play_pt, mo.closed_pt, mo.last_pt) - mor.stalest_pt) AS close_age_s,
       CASE WHEN mor.close_legs = 3 THEN (1.0 / mor.c_home) / mor.close_book_sum END AS p_home,
       CASE WHEN mor.close_legs = 3 THEN (1.0 / mor.c_draw) / mor.close_book_sum END AS p_draw,
       CASE WHEN mor.close_legs = 3 THEN (1.0 / mor.c_away) / mor.close_book_sum END AS p_away
FROM ev
LEFT JOIN mo ON mo.event_id = ev.event_id
LEFT JOIN ou ON ou.event_id = ev.event_id
LEFT JOIN ah ON ah.event_id = ev.event_id
LEFT JOIN mor ON mor.market_id = mo.market_id
LEFT JOIN our ON our.market_id = ou.market_id
LEFT JOIN ahr ON ahr.market_id = ah.market_id
"""


def build(inputs: dict) -> dict:
    from football.warehouse.gold import sql
    out = sql(QUERY, market=inputs["silver/market"], runner=inputs["silver/runner"])
    return {SPEC.name: out}
