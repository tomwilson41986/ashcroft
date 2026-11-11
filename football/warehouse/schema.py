"""The warehouse's contracts: every bronze and silver table's columns, types, key and meaning.

A table is written through ``conform`` (columns in this order, cast to these types, key columns never null), so a
reader can rely on the schema here whatever built the file. ``docs/football_data_dictionary.md`` is generated from
this module (``python -m football.warehouse dictionary``): the description of a column lives beside its type.

Conventions:

- times are UTC, millisecond timestamps (``TS``); durations and offsets are seconds;
- a price is Betfair's decimal price (1.01 to 1000); a chance is 1 / price before any normalisation;
- ``handicap`` is 0.0 where a runner has none, so (market_id, selection_id, handicap) is always a complete key
  (an Asian handicap runner trades on each line: the same selection id with a different handicap is another runner);
- ``year`` and ``month`` are the partition: the date in the file's own path (Betfair files a market under its event's
  day), carried as columns in the hive-style path ``year=2026/month=01`` and not inside the file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pyarrow as pa

TS = pa.timestamp("ms", tz="UTC")


@dataclass(frozen=True)
class Column:
    name: str
    type: pa.DataType
    doc: str
    nullable: bool = True


@dataclass(frozen=True)
class TableSpec:
    name: str                     # e.g. "silver/tick"
    doc: str
    columns: tuple[Column, ...]
    key: tuple[str, ...]          # unique within the table
    sort: tuple[str, ...] = ()    # the order rows are written in (row-group statistics prune on it)
    row_group: int = 262_144
    uncompressed: tuple[str, ...] = field(default_factory=tuple)   # columns stored without compression

    @property
    def schema(self) -> pa.Schema:
        return pa.schema([pa.field(c.name, c.type, nullable=c.nullable) for c in self.columns])

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]


def _c(name, type_, doc, nullable=True) -> Column:
    return Column(name, type_, doc, nullable)


BRONZE_RAW = TableSpec(
    "bronze/raw",
    "One row a raw Betfair Historic Data file (a market's stream file, bz2 JSON lines), with its bytes. Immutable: "
    "a file enters once; a second copy of the same path is kept only if it is longer (a market read before it "
    "settled).",
    (
        _c("plan_path", pa.string(), "The file's path from Betfair's plan folder on, e.g. "
           "BASIC/2026/Jan/29/35155405/1.252787713.bz2: the same name the Historic Data API gives it.", False),
        _c("event_id", pa.int64(), "Betfair's event id, from the path.", False),
        _c("market_id", pa.string(), "Betfair's market id, from the path.", False),
        _c("day", pa.int16(), "Day of the month in the path.", False),
        _c("size", pa.int32(), "Bytes of the compressed file.", False),
        _c("sha256", pa.string(), "SHA-256 of the compressed bytes.", False),
        _c("source", pa.string(), "Where the file came from: the bundle's name or 'api'.", False),
        _c("data", pa.binary(), "The file itself (bz2).", False),
    ),
    key=("plan_path",), sort=("plan_path",), row_group=4096, uncompressed=("data",),
)

SILVER_MARKET = TableSpec(
    "silver/market",
    "One row a market: what it is, when it ran, how it ended, and how much the file holds on it.",
    (
        _c("market_id", pa.string(), "Betfair's market id.", False),
        _c("event_id", pa.int64(), "Betfair's event id: one football match. A match's markets share it.", False),
        _c("event_type_id", pa.int32(), "Betfair's sport id (1: soccer)."),
        _c("market_type", pa.string(), "MATCH_ODDS, OVER_UNDER_25, ASIAN_HANDICAP, ...", False),
        _c("market_name", pa.string(), "The market's display name."),
        _c("event_name", pa.string(), "The match as Betfair names it: 'Home v Away'."),
        _c("home_name", pa.string(), "The side named first in event_name (blank when the name has no ' v ')."),
        _c("away_name", pa.string(), "The side named second in event_name."),
        _c("country_code", pa.string(), "Betfair's country code for the event; blank on many cups and "
           "internationals."),
        _c("timezone", pa.string(), "The market's display time zone."),
        _c("betting_type", pa.string(), "ODDS, ASIAN_HANDICAP_DOUBLE_LINE, ..."),
        _c("number_of_winners", pa.int16(), "Winners the market pays."),
        _c("market_base_rate", pa.float32(), "Betfair's base commission rate on the market, per cent."),
        _c("turn_in_play_enabled", pa.bool_(), "Whether the market was set to go in play."),
        _c("persistence_enabled", pa.bool_(), "Whether bets could persist in play."),
        _c("bsp_market", pa.bool_(), "Whether Betfair SP was offered (not on football)."),
        _c("regulators", pa.string(), "Regulator codes, comma-joined."),
        _c("market_time", TS, "The scheduled kick-off in the last definition."),
        _c("market_time_first", TS, "The scheduled kick-off in the first definition."),
        _c("n_market_time_changes", pa.int16(), "Times the scheduled kick-off changed in the file (a reschedule)."),
        _c("open_date", TS, "The event's open date in the last definition."),
        _c("suspend_time", TS, "The scheduled suspend time in the last definition."),
        _c("settled_time", TS, "When Betfair settled the market (the last definition's settledTime)."),
        _c("first_pt", TS, "The file's first message.", False),
        _c("last_pt", TS, "The file's last message.", False),
        _c("in_play_pt", TS, "The first message with the market in play: the actual kick-off, to the minute."),
        _c("closed_pt", TS, "The first message with the market CLOSED."),
        _c("first_trade_pt", TS, "The first traded price in the file."),
        _c("last_pre_trade_pt", TS, "The last traded price before the market went in play: the close's time."),
        _c("last_trade_pt", TS, "The last traded price in the file."),
        _c("status_final", pa.string(), "The last definition's status (CLOSED once settled)."),
        _c("in_play_final", pa.bool_(), "Whether the last definition had the market in play."),
        _c("n_messages", pa.int32(), "Messages in the file.", False),
        _c("n_definitions", pa.int32(), "Messages that carried a market definition.", False),
        _c("n_ticks", pa.int32(), "Traded-price updates in the file (rows of silver/tick).", False),
        _c("n_ticks_pre", pa.int32(), "Of those, before the market went in play.", False),
        _c("n_ticks_inplay", pa.int32(), "Of those, in play.", False),
        _c("n_ltp_invalid", pa.int32(), "Price updates outside Betfair's range (1.01 to 1000), left out of "
           "silver/tick: a void market is closed with 1.0 on every runner.", False),
        _c("n_runners", pa.int16(), "Runners in the last definition (an Asian handicap line counts once a side).",
           False),
        _c("n_winners", pa.int16(), "Runners settled WINNER.", False),
        _c("n_removed", pa.int16(), "Runners settled REMOVED.", False),
        _c("n_susp_inplay", pa.int16(), "Suspended spells while in play (a goal, a penalty, a review, half time).",
           False),
        _c("kickoff_delay_s", pa.int32(), "in_play_pt less market_time, seconds (blank if never in play)."),
        _c("plan_path", pa.string(), "The bronze file the row was read from.", False),
        _c("file_sha256", pa.string(), "That file's SHA-256.", False),
        _c("parser_version", pa.int16(), "football.warehouse.parse.PARSER_VERSION that read it.", False),
    ),
    key=("market_id",), sort=("market_id",),
)

SILVER_RUNNER = TableSpec(
    "silver/runner",
    "One row a runner of a market: a (selection, handicap) pair, its role, its result and its first, closing and "
    "last traded prices.",
    (
        _c("market_id", pa.string(), "Betfair's market id.", False),
        _c("event_id", pa.int64(), "Betfair's event id.", False),
        _c("market_type", pa.string(), "The market's type.", False),
        _c("selection_id", pa.int64(), "Betfair's selection id. On match odds and handicaps it is the team's own "
           "id, the same in every market the team plays.", False),
        _c("handicap", pa.float64(), "The runner's handicap line; 0.0 where it has none.", False),
        _c("runner_name", pa.string(), "The selection's name."),
        _c("sort_priority", pa.int16(), "Betfair's display order within the market."),
        _c("role", pa.string(), "home, away, draw, over or under: what the runner is in the match (blank if not "
           "resolved)."),
        _c("status_final", pa.string(), "ACTIVE, WINNER, LOSER, REMOVED ... in the last definition."),
        _c("won", pa.int8(), "1 if settled WINNER, 0 if LOSER, blank otherwise (void, removed, unsettled)."),
        _c("ltp_first", pa.float64(), "The first traded price."),
        _c("ltp_last_pre", pa.float64(), "The last traded price before the market went in play: the close."),
        _c("ltp_last", pa.float64(), "The last traded price in the file."),
        _c("n_ticks", pa.int32(), "Traded-price updates on the runner.", False),
        _c("n_ticks_pre", pa.int32(), "Of those, before the market went in play.", False),
        _c("first_trade_pt", TS, "The runner's first traded price."),
        _c("last_pre_trade_pt", TS, "The runner's last traded price before the market went in play."),
        _c("last_trade_pt", TS, "The runner's last traded price."),
        _c("in_definition", pa.bool_(), "False for a (selection, handicap) that traded but is not in the last "
           "definition's runners.", False),
    ),
    key=("market_id", "selection_id", "handicap"), sort=("market_id", "selection_id", "handicap"),
)

SILVER_DEFINITION = TableSpec(
    "silver/definition",
    "One row each time a market's definition was published: its state at that moment. The history behind "
    "silver/market (reschedules, the turn in play, each suspension, the close).",
    (
        _c("market_id", pa.string(), "Betfair's market id.", False),
        _c("seq", pa.int32(), "The message's position in the file (0 first).", False),
        _c("pt", TS, "When Betfair published the message.", False),
        _c("status", pa.string(), "OPEN, SUSPENDED, CLOSED or INACTIVE."),
        _c("in_play", pa.bool_(), "Whether the market was in play."),
        _c("bet_delay", pa.int16(), "The in-play bet delay, seconds."),
        _c("number_of_active_runners", pa.int16(), "Runners still active."),
        _c("market_time", TS, "The scheduled kick-off as it stood."),
        _c("version", pa.int64(), "Betfair's version of the market."),
        _c("complete", pa.bool_(), "Betfair's 'complete' flag (all runners added)."),
        _c("cross_matching", pa.bool_(), "Whether cross matching was on."),
        _c("settled_time", TS, "settledTime, once the market is settled."),
    ),
    key=("market_id", "seq"), sort=("market_id", "seq"),
)

SILVER_SPELL = TableSpec(
    "silver/spell",
    "One row a spell of a market's life: a run of messages over which (in play, status) did not change. An "
    "in-play SUSPENDED spell is a stoppage Betfair's traders made: a goal, a penalty, a red card, a review.",
    (
        _c("market_id", pa.string(), "Betfair's market id.", False),
        _c("event_id", pa.int64(), "Betfair's event id.", False),
        _c("market_type", pa.string(), "The market's type.", False),
        _c("spell_no", pa.int32(), "The spell's order in the market (0 first).", False),
        _c("start_pt", TS, "The definition that began the spell.", False),
        _c("end_pt", TS, "The definition that ended it (the file's last message for the last spell).", False),
        _c("duration_s", pa.float64(), "end_pt less start_pt, seconds.", False),
        _c("in_play", pa.bool_(), "Whether the market was in play during the spell."),
        _c("status", pa.string(), "The market's status during the spell."),
    ),
    key=("market_id", "spell_no"), sort=("market_id", "spell_no"),
)

SILVER_TICK = TableSpec(
    "silver/tick",
    "One row each time a runner's last traded price was published (BASIC plan: about once a minute a runner while "
    "it trades), with the market's state at that moment. The atomic fact of the warehouse.",
    (
        _c("market_id", pa.string(), "Betfair's market id.", False),
        _c("event_id", pa.int64(), "Betfair's event id.", False),
        _c("market_type", pa.string(), "The market's type.", False),
        _c("selection_id", pa.int64(), "Betfair's selection id.", False),
        _c("handicap", pa.float64(), "The runner's handicap line; 0.0 where it has none.", False),
        _c("pt", TS, "When Betfair published the price.", False),
        _c("seq", pa.int32(), "The message's position in the file: orders prices published at the same instant.",
           False),
        _c("ltp", pa.float64(), "The last traded price.", False),
        _c("in_play", pa.bool_(), "Whether the market was in play when the price was published.", False),
        _c("status", pa.string(), "The market's status then (OPEN or SUSPENDED; a price can be published with "
           "the suspension it caused)."),
        _c("t_off_s", pa.int32(), "Seconds from pt to the scheduled kick-off (market_time in the last "
           "definition); negative after it."),
        _c("t_inplay_s", pa.int32(), "Seconds since the market went in play; blank before."),
    ),
    key=("market_id", "selection_id", "handicap", "pt", "seq"),
    sort=("market_id", "selection_id", "handicap", "pt", "seq"),
)

SILVER = (SILVER_MARKET, SILVER_RUNNER, SILVER_DEFINITION, SILVER_SPELL, SILVER_TICK)
TABLES = {t.name: t for t in (BRONZE_RAW,) + SILVER}


def empty(spec: TableSpec) -> pa.Table:
    return spec.schema.empty_table()


def conform(spec: TableSpec, table: pa.Table) -> pa.Table:
    """``table`` with exactly the spec's columns, in order, cast to its types; a key or required column holding a
    null is an error (the contract a reader relies on)."""
    missing = [n for n in spec.names if n not in table.column_names]
    if missing:
        raise ValueError(f"{spec.name}: missing columns {missing}")
    out = table.select(spec.names).cast(spec.schema, safe=False) if table.num_rows else empty(spec)
    for c in spec.columns:
        if not c.nullable and out.column(c.name).null_count:
            raise ValueError(f"{spec.name}: {out.column(c.name).null_count} nulls in required column {c.name}")
    return out
