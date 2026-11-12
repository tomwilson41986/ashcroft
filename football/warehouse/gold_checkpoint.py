"""gold/checkpoint: each runner's price as it stood at fixed moments before the off, with how far to trust it.

A checkpoint is a number of minutes before the SCHEDULED kick-off (the time everyone knew in advance, so a model
deciding at T-60 can be trained on exactly what it would have seen); checkpoint 0 is the close: the last price
traded before the off. The schedule is the one that stood before the off (silver/market.market_time_pre: Betfair
often rewrites the kick-off time to the actual one as the market turns in play, and a grid measured from that
would put prices from after a checkpoint into it), and ``schedule_known`` says whether that schedule had already
been set at the checkpoint's moment (false: the match was rescheduled later, and nobody could have stood at that
checkpoint). The price at a checkpoint is the last one published at or before it (an as-of join of the pre-off
ticks, pandas.merge_asof), never a later one. A checkpoint that falls after the off (a match that started early)
or after the market closed (a match called off) is left out, and a market that closed before its off has no close.

Only what traded is there: a runner has a row at a checkpoint once it has a price by then (most Asian handicap
lines never trade and never appear). Beside the price: when it traded and how old it is, the ticks so far and
since the previous checkpoint (activity: BASIC has no volume), and its book: for match odds and over/under the
runners priced, whether all are, the sum of 1 / price and the stalest leg's age; for an Asian handicap line the same
over the line's two sides (home at h, away at -h). ``p_norm`` is the chance normalised to a complete book.

The traps earlier research met (reports/football_model_rethink.md) are columns here, not folklore: a stale last
traded price (``age_s``, ``book_max_age_s``), a partial book (``book_complete``), a line market read as one runner
(``line``).
"""

from __future__ import annotations

import pyarrow as pa

from football.warehouse.schema import TS, Column, TableSpec

VERSION = 2          # 2: the grid from the schedule as known before the off; the off of a market never in play
SCOPE = "part"
INPUTS = ("silver/market", "silver/runner", "silver/tick")
COLUMNS = {
    "silver/market": ["market_id", "market_time_pre", "schedule_set_pt", "off_pt", "closed_pt"],
    "silver/runner": ["market_id", "event_id", "market_type", "selection_id", "handicap", "role", "n_ticks_pre"],
    "silver/tick": ["market_id", "selection_id", "handicap", "pt", "seq", "ltp", "pre_off"],
}

#: minutes before the scheduled kick-off; 0 is the close
CHECKPOINTS = (4320, 1440, 720, 360, 180, 120, 90, 60, 45, 30, 15, 5, 1, 0)

SPEC = TableSpec(
    "gold/checkpoint",
    "One row a runner a checkpoint: the price as of a fixed number of minutes before the scheduled kick-off (0: the "
    "close), with its age, the activity behind it and the state of its book.",
    (
        Column("market_id", pa.string(), "Betfair's market id.", False),
        Column("selection_id", pa.int64(), "Betfair's selection id.", False),
        Column("handicap", pa.float64(), "The runner's handicap line; 0.0 where it has none.", False),
        Column("cp_min", pa.int32(), "Minutes before the scheduled kick-off; 0 is the close.", False),
        Column("event_id", pa.int64(), "Betfair's event id.", False),
        Column("market_type", pa.string(), "The market's type.", False),
        Column("role", pa.string(), "home, away, draw, over or under."),
        Column("line", pa.float64(), "The book the runner belongs to: 0.0 for match odds and over/under; for an "
               "Asian handicap the home side's line (home at h and away at -h are one book).", False),
        Column("cp_time", TS, "The checkpoint's moment (for the close: the off, silver/market.off_pt).", False),
        Column("schedule_known", pa.bool_(), "Whether the schedule the checkpoint is measured from had been set "
               "by its moment (always true at the close). False: the kick-off was moved later on, so the "
               "checkpoint could not have been known then; keep such rows out of anything that claims to "
               "decide at the checkpoint.", False),
        Column("ltp", pa.float64(), "The last traded price as of the checkpoint.", False),
        Column("trade_pt", TS, "When that price was published.", False),
        Column("age_s", pa.float64(), "cp_time less trade_pt, seconds: how stale the price is.", False),
        Column("n_ticks_before", pa.int32(), "The runner's price updates up to the checkpoint.", False),
        Column("n_ticks_window", pa.int32(), "Of those, since the previous (earlier) checkpoint.", False),
        Column("book_n", pa.int16(), "Runners of the book with a price as of the checkpoint.", False),
        Column("book_complete", pa.bool_(), "Whether every runner of the book has one (3 for match odds, 2 for "
               "over/under and for a handicap line).", False),
        Column("book_sum", pa.float64(), "Sum of 1 / price over the book's priced runners.", False),
        Column("book_max_age_s", pa.float64(), "The age of the book's stalest price, seconds.", False),
        Column("p_raw", pa.float64(), "1 / price.", False),
        Column("p_norm", pa.float64(), "p_raw / book_sum when the book is complete; blank otherwise."),
    ),
    key=("market_id", "selection_id", "handicap", "cp_min"),
    sort=("market_id", "cp_min", "selection_id", "handicap"),
)
SPECS = (SPEC,)

KEYS = ["market_id", "selection_id", "handicap"]


def _ms(table: pa.Table, cols: list[str]) -> pa.Table:
    """``table`` with the named timestamp columns as integer milliseconds (blank stays blank)."""
    for c in cols:
        i = table.schema.get_field_index(c)
        table = table.set_column(i, c, table.column(c).cast(pa.int64()))
    return table


def build(inputs: dict) -> dict:
    import numpy as np
    import pandas as pd

    m = _ms(inputs["silver/market"].select(COLUMNS["silver/market"]),
            ["market_time_pre", "schedule_set_pt", "off_pt", "closed_pt"]).to_pandas()
    r = inputs["silver/runner"].select(COLUMNS["silver/runner"]).to_pandas()
    tick = inputs["silver/tick"]
    t = _ms(tick.filter(tick.column("pre_off")).select(["market_id", "selection_id", "handicap", "pt", "seq", "ltp"]),
            ["pt"]).to_pandas()

    r = r[r["n_ticks_pre"] > 0].merge(m, on="market_id")
    r = r[r["market_time_pre"].notna() & r["off_pt"].notna()].reset_index(drop=True)
    if r.empty or t.empty:
        return {SPEC.name: SPEC.schema.empty_table()}
    r["gid"] = np.arange(len(r))

    # the runner's ticks in order, numbered: the price as of a moment is the last one at or before it
    t = t.merge(r[KEYS + ["gid"]], on=KEYS, how="inner").sort_values(["gid", "pt", "seq"], kind="stable")
    t["n"] = t.groupby("gid", sort=False).cumcount() + 1
    t = t[["gid", "pt", "ltp", "n"]].sort_values("pt", kind="stable")

    grids = []
    for cp in CHECKPOINTS:
        g = r[["gid"]].copy()
        g["cp_min"] = cp
        g["cp_ms"] = (r["off_pt"] if cp == 0 else r["market_time_pre"] - cp * 60_000.0).to_numpy()
        # a checkpoint after the off never happened, nor one after the market had closed (a match called off)
        open_then = (g["cp_ms"] <= r["off_pt"]) & (r["closed_pt"].isna() | (g["cp_ms"] <= r["closed_pt"]))
        g["schedule_known"] = True if cp == 0 else (g["cp_ms"] >= r["schedule_set_pt"]).to_numpy()
        grids.append(g[open_then.to_numpy()])
    g = pd.concat(grids, ignore_index=True)
    g["cp_ms"] = g["cp_ms"].astype("int64")
    a = pd.merge_asof(g.sort_values("cp_ms", kind="stable"), t, left_on="cp_ms", right_on="pt", by="gid",
                      direction="backward", allow_exact_matches=True)
    a = a[a["ltp"].notna()].merge(r[["gid"] + KEYS + ["event_id", "market_type", "role"]], on="gid")
    if a.empty:
        return {SPEC.name: SPEC.schema.empty_table()}

    ah = a["market_type"].to_numpy() == "ASIAN_HANDICAP"
    a["line"] = np.where(ah, np.where(a["role"].to_numpy() == "away", -a["handicap"], a["handicap"]), 0.0) + 0.0
    a["age_s"] = (a["cp_ms"] - a["pt"]) / 1000.0
    a["p_raw"] = 1.0 / a["ltp"]
    a["mkey"] = pd.factorize(a["market_id"])[0]
    book = a.groupby(["mkey", "cp_min", "line"], sort=False)
    a["book_n"] = book["ltp"].transform("size")
    a["book_sum"] = book["p_raw"].transform("sum")
    a["book_max_age_s"] = book["age_s"].transform("max")
    a["book_complete"] = a["book_n"] == np.where(a["market_type"].to_numpy() == "MATCH_ODDS", 3, 2)
    a["p_norm"] = np.where(a["book_complete"], a["p_raw"] / a["book_sum"], np.nan)
    a = a.sort_values(["gid", "cp_min"], ascending=[True, False], kind="stable")
    a["n_ticks_before"] = a["n"].astype("int64")
    a["n_ticks_window"] = a["n_ticks_before"] - a.groupby("gid", sort=False)["n_ticks_before"].shift(1).fillna(0)

    out = pa.table({
        "market_id": pa.array(a["market_id"], pa.string()),
        "selection_id": pa.array(a["selection_id"], pa.int64()),
        "handicap": pa.array(a["handicap"], pa.float64()),
        "cp_min": pa.array(a["cp_min"], pa.int32()),
        "event_id": pa.array(a["event_id"], pa.int64()),
        "market_type": pa.array(a["market_type"], pa.string()),
        "role": pa.array(a["role"].where(a["role"].notna(), None), pa.string()),
        "line": pa.array(a["line"], pa.float64()),
        "cp_time": pa.array(a["cp_ms"], pa.int64()).cast(TS),
        "schedule_known": pa.array(a["schedule_known"].to_numpy(bool), pa.bool_()),
        "ltp": pa.array(a["ltp"], pa.float64()),
        "trade_pt": pa.array(a["pt"].astype("int64"), pa.int64()).cast(TS),
        "age_s": pa.array(a["age_s"], pa.float64()),
        "n_ticks_before": pa.array(a["n_ticks_before"], pa.int32()),
        "n_ticks_window": pa.array(a["n_ticks_window"].astype("int64"), pa.int32()),
        "book_n": pa.array(a["book_n"], pa.int16()),
        "book_complete": pa.array(a["book_complete"].to_numpy(), pa.bool_()),
        "book_sum": pa.array(a["book_sum"], pa.float64()),
        "book_max_age_s": pa.array(a["book_max_age_s"], pa.float64()),
        "p_raw": pa.array(a["p_raw"], pa.float64()),
        "p_norm": pa.array(a["p_norm"], pa.float64(), from_pandas=True),
    })
    return {SPEC.name: out}
