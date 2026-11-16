"""gold/path: what each runner's price did BEFORE each decision checkpoint: the price-path feature library.

One row a match-odds or over/under runner a decision checkpoint (24 h, 6 h, 2 h, 1 h and 15 min before the
scheduled kick-off), for exactly the rows gold/checkpoint holds there (the runner has a price by then). Everything
is read off the runner's own pre-off ticks published at or before the checkpoint's moment (``pt <= cp_time``),
never a later one: a model deciding at T-60 is trained on what it would have seen.

**The scale.** x = ln(p / (1 - p)) with p = 1 / price: the logit of the raw chance, so a move means the same at
1.5 as at 15. A positive move is a runner shortening (its chance rising), as in gold/move.

**The path.** BASIC publishes a runner's last traded price when it changes, so between two ticks the price stands
still: the path is a step function, known at every instant, and "the price as of a moment" is the last tick at or
before it (however old: ``age_s`` in gold/checkpoint says how old the current one is). A window is the last L
before the checkpoint, (cp_time - L, cp_time]; where the runner first traded inside it, the window is what there is
of it, from the first trade on (``hours_since_first`` says how much that is). A change belongs to the tick that
made it, so the changes in a window add up to the window's return.

**The features.**

- returns: the change in x from the price as of cp_time - L to the price at the checkpoint, L from 15 min to 24 h;
  blank when the runner had no price that far back; the move since its first trade, and the hours since;
- volatility and activity: the root of the sum of squared tick changes over 1 h, 6 h and 24 h, and the ticks in
  each (BASIC has no volume: a tick is a change of the last traded price);
- trend: the least-squares slope of the path on time over 6 h and 24 h, an hour, with its R squared (the regression
  of the step function itself, each price weighted by the time it stood: ticks come in bursts, and a regression of
  the ticks alone would be a regression of the bursts), and the efficiency ratio (Kaufman: the net move over the
  sum of the absolute moves; 1 a straight line, near 0 a round trip);
- range: where the price stands between the 24 h low and high, the fall from the high, the rise from the low;
- jumps: the largest single change of the last 2 h, its sign, the minutes since it;
- mean reversion: the time-weighted average x of the last 1 h and 6 h, less the current x.

**Beyond the brief**, each for a reason in the market-microstructure literature and none needing volume or depth:

- ``last_dx`` and ``run_len``: a last traded price sits at the bid or the offer, whichever the last trade hit, not
  at the mid: its last change is partly bid-ask bounce and tends to be given back (Roll 1984; the short-horizon
  reversal of Jegadeesh 1990 and Lehmann 1990), while a run of changes one way is the footprint of one-sided
  order flow, which persists (Hasbrouck 1991);
- ``tick_imb_*``: up-ticks less down-ticks over all ticks: the tick test (Lee and Ready 1991) as a stand-in for
  signed order flow, robust to the size of any one change;
- ``ac1_*`` and ``roll_24h``: the first-order autocorrelation of the tick changes and Roll's implied spread,
  2 * sqrt(-cov) of consecutive changes: how much of this runner's path is bounce (negative: fade the last move,
  and the spread says by how much) and how much is drift (positive: information arriving in pieces);
- ``book_catchup``: prices of one book trade at different moments (nonsynchronous trading: Scholes and Williams
  1977, Lo and MacKinlay 1990), so a runner that has not traded since the others moved is stale by what they
  moved. The sum of the book's raw chances when the runner last traded, less the sum now, is the change in the
  runner's own chance that would put the book sum back: its move still to come, in logit.

Ratios the columns give without a column of their own: the variance ratio of Lo and MacKinlay (ret_L squared over
rv_L squared), the share of the last 2 h of variance in one jump (jump_abs_2h squared over rv squared), the pace
of trading now against earlier (n_ticks_1h over n_ticks_6h).

A feature that cannot be measured is blank, not zero: a return with no price that far back, a ratio over a window
in which the price never moved, a slope over less than 15 minutes of history.
"""

from __future__ import annotations

import pyarrow as pa

from football.warehouse.gold_checkpoint import CHECKPOINTS
from football.warehouse.schema import TS, Column, TableSpec

VERSION = 2          # 2: pre-off prices are those before the off (silver/tick.pre_off)
SCOPE = "part"
INPUTS = ("silver/tick", "gold/checkpoint")
COLUMNS = {
    "silver/tick": ["market_id", "market_type", "selection_id", "handicap", "pt", "seq", "ltp", "pre_off"],
    "gold/checkpoint": ["market_id", "selection_id", "handicap", "cp_min", "event_id", "market_type", "role",
                        "cp_time"],
}

MARKET_TYPES = ("MATCH_ODDS", "OVER_UNDER_25")
#: the decision checkpoints, minutes before the scheduled kick-off: a subset of gold/checkpoint's grid
PATH_CHECKPOINTS = (1440, 360, 120, 60, 15)
assert set(PATH_CHECKPOINTS) <= set(CHECKPOINTS)
#: the lookbacks, minutes, by the suffix their columns carry
LOOKBACKS = {"15m": 15, "30m": 30, "1h": 60, "2h": 120, "6h": 360, "24h": 1440}
MIN_SPAN_H = 0.25                 # a slope needs this much of the window covered by the runner's history, hours
MIN_PAIRS = 5                     # an autocorrelation needs this many pairs of consecutive changes
FLAT_VAR = 1e-12                  # a window whose time-weighted variance of x is under this never moved
PRICE_MIN, PRICE_MAX = 1.01, 1000.0          # Betfair's price ladder ends
BOOK_RUNNERS = {"MATCH_ODDS": 3, "OVER_UNDER_25": 2}

_F = pa.float64()


def _ret(tag: str, words: str) -> Column:
    return Column(f"ret_{tag}", _F, f"The change in x from the price as of {words} before the checkpoint to the "
                  "price at the checkpoint; 0 if no tick since; blank when the runner had no price that far back.")


SPEC = TableSpec(
    "gold/path",
    "One row a match-odds or over/under runner a decision checkpoint (1440, 360, 120, 60, 15 minutes before the "
    "scheduled kick-off): what its last traded price did up to that moment, in the logit of the raw chance "
    "(x = ln(p / (1 - p)), p = 1 / price; positive is shortening). Every column uses only ticks published at or "
    "before cp_time. A window is the last L before the checkpoint, cut at the runner's first trade where that is "
    "later. The feature library of the closing-price model: joins gold/checkpoint and gold/move on the key.",
    (
        Column("market_id", pa.string(), "Betfair's market id.", False),
        Column("selection_id", pa.int64(), "Betfair's selection id.", False),
        Column("handicap", pa.float64(), "The runner's handicap line: 0.0 on these markets.", False),
        Column("cp_min", pa.int32(), "The checkpoint: minutes before the scheduled kick-off (1440, 360, 120, 60 "
               "or 15).", False),
        Column("event_id", pa.int64(), "Betfair's event id.", False),
        Column("market_type", pa.string(), "MATCH_ODDS or OVER_UNDER_25.", False),
        Column("role", pa.string(), "home, away, draw, over or under."),
        Column("cp_time", TS, "The checkpoint's moment.", False),
        Column("x", _F, "The logit of the raw chance at the checkpoint: ln(p / (1 - p)) with p = 1 / the last "
               "traded price as of cp_time (-ln(price - 1)). Every move below is a difference of this.", False),
        _ret("15m", "15 minutes"), _ret("30m", "30 minutes"), _ret("1h", "1 hour"), _ret("2h", "2 hours"),
        _ret("6h", "6 hours"), _ret("24h", "24 hours"),
        Column("ret_first", _F, "The change in x since the runner's first traded price (0 when that is its only "
               "one).", False),
        Column("hours_since_first", _F, "Hours from the runner's first traded price to the checkpoint: how much "
               "history the windows have to read.", False),
        Column("rv_1h", _F, "Realised volatility of the last hour: the root of the sum of the squared changes in "
               "x made by the ticks in (cp_time - 1 h, cp_time]; 0 when none changed it.", False),
        Column("rv_6h", _F, "The same over the last 6 hours (since the first trade where that is later).", False),
        Column("rv_24h", _F, "The same over the last 24 hours (since the first trade where that is later).",
               False),
        Column("n_ticks_15m", pa.int32(), "Price updates in the last 15 minutes: (cp_time - 15 min, cp_time]. "
               "Activity: BASIC has no volume. The first trade counts as an update.", False),
        Column("n_ticks_1h", pa.int32(), "Price updates in the last hour.", False),
        Column("n_ticks_6h", pa.int32(), "Price updates in the last 6 hours.", False),
        Column("n_ticks_24h", pa.int32(), "Price updates in the last 24 hours.", False),
        Column("slope_6h", _F, "Trend: the least-squares slope of x on time over the last 6 hours, in x an hour; "
               "the price path is a step function and each price weighs by the time it stood. 0 when the price "
               "did not change in the window; blank when the runner's history covers under 15 minutes of it."),
        Column("slope_24h", _F, "The same over the last 24 hours."),
        Column("slope_r2_6h", _F, "The fit of slope_6h: the share of the path's time-weighted variance the line "
               "explains (0 to 1). Blank where slope_6h is, and when the price did not move in the window."),
        Column("slope_r2_24h", _F, "The fit of slope_24h."),
        Column("er_6h", _F, "Efficiency ratio of the last 6 hours: the absolute net change in x over the sum of "
               "the absolute changes (1: every tick the same way; near 0: a round trip). Blank when no tick "
               "changed the price in the window."),
        Column("er_24h", _F, "The same over the last 24 hours."),
        Column("range_pos_24h", _F, "Where x stands between the lowest and the highest x of the last 24 hours "
               "(the price as of the window's start included): 0 at the low, 1 at the high. Blank when the price "
               "did not move in the window."),
        Column("drawdown_24h", _F, "The highest x of the last 24 hours less the current x (0 or more): how far "
               "the chance has fallen from its high.", False),
        Column("runup_24h", _F, "The current x less the lowest x of the last 24 hours (0 or more): how far the "
               "chance has risen from its low.", False),
        Column("jump_abs_2h", _F, "The largest absolute change in x made by one tick in the last 2 hours; 0 when "
               "no tick changed the price.", False),
        Column("jump_sign_2h", pa.int8(), "That change's sign: 1 a shortening, -1 a drift, 0 when there was "
               "none. Of equal changes the latest is taken.", False),
        Column("jump_min_2h", _F, "Minutes from that tick to the checkpoint; blank when there was none."),
        Column("twap_dev_1h", _F, "Mean reversion: the time-weighted average x of the last hour less the current "
               "x (positive: the price now is below its recent average chance). Blank only when the runner's "
               "first trade is the checkpoint's own moment."),
        Column("twap_dev_6h", _F, "The same against the last 6 hours' time-weighted average."),
        Column("last_dx", _F, "The change in x made by the runner's latest tick (the one whose price stands at "
               "the checkpoint): the move a bid-ask bounce would give back. Blank when it has traded once."),
        Column("run_len", pa.int32(), "The run the latest tick ends: how many consecutive ticks moved x the same "
               "way, signed (3: three shortenings in a row; -2: two drifts). 0 when it has traded once, or when "
               "the latest tick repeated the price before it.", False),
        Column("tick_imb_1h", _F, "Tick imbalance of the last hour: (ticks that raised x - ticks that lowered "
               "it) / (both). The tick test as a stand-in for signed order flow. Blank when no tick changed the "
               "price."),
        Column("tick_imb_6h", _F, "The same over the last 6 hours."),
        Column("ac1_6h", _F, "First-order autocorrelation of the tick changes of the last 6 hours: the sum of "
               "products of consecutive changes over the root of the product of their sums of squares (-1 to 1; "
               "negative: bounce, positive: drift). Blank with fewer than 5 pairs of consecutive changes."),
        Column("ac1_24h", _F, "The same over the last 24 hours."),
        Column("roll_24h", _F, "Roll's implied spread over the last 24 hours, in x: 2 * sqrt(-c) where c is the "
               "mean product of consecutive tick changes; 0 when c is positive. Blank with fewer than 5 pairs."),
        Column("book_catchup", _F, "The move the rest of the book has made since this runner last traded, as "
               "the change in this runner's x that would restore the book's sum of raw chances to what it was "
               "at that trade: logit(p + d) - logit(p) with d = (the other runners' 1 / price as of this "
               "runner's last trade, summed) less (the same as of the checkpoint). 0 when no other runner "
               "traded since. Blank unless every other runner of the book (two for match odds, one for "
               "over/under) had a price by this runner's last trade."),
    ),
    key=("market_id", "selection_id", "handicap", "cp_min"),
    sort=("market_id", "cp_min", "selection_id", "handicap"),
)
SPECS = (SPEC,)

KEYS = ["market_id", "selection_id", "handicap"]
_SHIFT = 42                       # a runner's number sits above 42 bits of milliseconds (139 years) in the sort key
_CLIP_H = 49.0                    # no window reaches further back than 48 h before the scheduled kick-off


def _span(ufunc, values, lo, hi):
    """``ufunc`` (minimum, maximum) over ``values[lo..hi]``, both ends included, row by row. ``lo`` and ``hi`` must
    rise down the rows (they do: rows are in runner, then checkpoint, order), which keeps the work linear."""
    import numpy as np
    if not len(lo):
        return values[:0]
    idx = np.empty(2 * len(lo), dtype=np.intp)
    idx[0::2] = lo
    idx[1::2] = hi + 1
    return ufunc.reduceat(np.append(values, values[-1:]), idx)[0::2]


def _div(num, den, ok):
    """num / den where ``ok``, blank elsewhere."""
    import numpy as np
    out = np.full(len(num), np.nan)
    np.divide(num, den, out=out, where=ok)
    return out


def build(inputs: dict) -> dict:
    import numpy as np
    import pandas as pd
    import pyarrow.compute as pc

    cp, tick = inputs["gold/checkpoint"], inputs["silver/tick"]
    types = pa.array(MARKET_TYPES)
    cp = cp.filter(pc.and_(pc.is_in(cp.column("market_type"), value_set=types),
                           pc.is_in(cp.column("cp_min").cast(pa.int32()),
                                    value_set=pa.array(PATH_CHECKPOINTS, pa.int32()))))
    tick = tick.filter(pc.and_(tick.column("pre_off"), pc.is_in(tick.column("market_type"), value_set=types)))
    if not cp.num_rows or not tick.num_rows:
        return {SPEC.name: SPEC.schema.empty_table()}

    # ---- the ticks, in runner then time order; a runner is a number (gid) and a slice [st, en) of the arrays
    t = tick.select(COLUMNS["silver/tick"][:-1])
    t = t.set_column(t.schema.get_field_index("pt"), "pt", t.column("pt").cast(pa.int64())).to_pandas()
    mcode = pd.factorize(t["market_id"])[0]
    order = np.lexsort((t["seq"].to_numpy(), t["pt"].to_numpy(), t["handicap"].to_numpy(),
                        t["selection_id"].to_numpy(), mcode))
    t = t.iloc[order].reset_index(drop=True)
    mcode = mcode[order]
    sel, hc, pt = t["selection_id"].to_numpy(), t["handicap"].to_numpy(), t["pt"].to_numpy()
    n = len(t)
    first = np.ones(n, dtype=bool)
    first[1:] = (mcode[1:] != mcode[:-1]) | (sel[1:] != sel[:-1]) | (hc[1:] != hc[:-1])
    gid = np.cumsum(first) - 1
    st = np.flatnonzero(first)
    en = np.append(st[1:], n)
    runners = t.loc[st, KEYS].reset_index(drop=True)
    runners["gid"] = np.arange(len(st))

    # ---- the rows: gold/checkpoint's, at the decision checkpoints, in runner then checkpoint-time order
    c = cp.select(COLUMNS["gold/checkpoint"])
    c = c.set_column(c.schema.get_field_index("cp_time"), "cp_time", c.column("cp_time").cast(pa.int64()))
    rows = c.to_pandas().merge(runners, on=KEYS, how="inner")
    base = int(pt.min()) - 25 * 3_600_000                 # under every moment asked about, so keys stay positive
    if int(pt.max()) - base >= 1 << _SHIFT or len(st) >= 1 << (62 - _SHIFT):
        raise ValueError("gold/path: the partition's ticks span more than the sort key holds")
    key = (gid.astype(np.int64) << _SHIFT) + (pt - base)

    def count(g, when):
        """How many ticks, counted from the start of the arrays, lie at or before runner ``g``'s moment ``when``:
        one past the tick whose price stands then (``st[g]`` when the runner has no price yet)."""
        return np.searchsorted(key, (g.astype(np.int64) << _SHIFT) + np.maximum(when - base, 0), side="right")

    rows = rows[count(rows["gid"].to_numpy(), rows["cp_time"].to_numpy()) > st[rows["gid"].to_numpy()]]
    rows = rows.sort_values(["gid", "cp_time"], kind="stable").reset_index(drop=True)
    if rows.empty:
        return {SPEC.name: SPEC.schema.empty_table()}
    g, when, cp_min = rows["gid"].to_numpy(), rows["cp_time"].to_numpy(), rows["cp_min"].to_numpy()
    nr = len(rows)
    s0 = st[g]
    i_t = count(g, when) - 1                             # the tick whose price stands at the checkpoint

    # ---- the path: x, its changes, and running sums that restart at each runner (first tick: no change, 0)
    price = np.clip(t["ltp"].to_numpy(), PRICE_MIN, PRICE_MAX)
    x = -np.log(price - 1.0)
    dx = np.zeros(n)
    dx[1:] = x[1:] - x[:-1]
    dx[first] = 0.0
    prev = np.zeros(n)
    prev[1:] = dx[:-1]                                    # the change before (0 at a runner's second tick: dx[st])
    # time in hours from the scheduled kick-off (cp_time + cp_min), floored 49 h out: no window starts before
    # 48 h, and the floor keeps the integrals small; x is centred on the runner's first price for the same reason
    ko = pt[en - 1].copy()
    ko[g] = when + cp_min * 60_000
    u = np.maximum((pt - ko[gid]) / 3_600_000.0, -_CLIP_H)
    xc = x - x[st][gid]
    du = np.zeros(n)
    du[1:] = u[1:] - u[:-1]
    du[first] = 0.0
    du2 = np.zeros(n)
    du2[1:] = (u[1:] ** 2 - u[:-1] ** 2) / 2.0
    du2[first] = 0.0
    xp = np.zeros(n)
    xp[1:] = xc[:-1]                                      # the price that stood until this tick
    cum = pd.DataFrame({
        "abs": np.abs(dx), "sq": dx * dx, "up": (dx > 0).astype(np.int64), "dn": (dx < 0).astype(np.int64),
        "pair": dx * prev, "ia": xp * du, "ib": xp * du2, "ic": xp * xp * du,
    }).groupby(gid, sort=False).cumsum()
    c_abs, c_sq, c_pair = cum["abs"].to_numpy(), cum["sq"].to_numpy(), cum["pair"].to_numpy()
    c_up, c_dn = cum["up"].to_numpy(), cum["dn"].to_numpy()
    c_ia, c_ib, c_ic = cum["ia"].to_numpy(), cum["ib"].to_numpy(), cum["ic"].to_numpy()

    x0 = x[i_t]
    b = -cp_min / 60.0                                    # the checkpoint, hours from the kick-off
    out: dict = {}

    # ---- window by window: j is the tick whose price stood at the window's start (the first trade if later)
    j_of = {}
    for tag, minutes in LOOKBACKS.items():
        cnt = count(g, when - minutes * 60_000)
        j = j_of[tag] = np.maximum(cnt - 1, s0)
        out[f"ret_{tag}"] = np.where(cnt > s0, x0 - x[j], np.nan)
        if tag in ("15m", "1h", "6h", "24h"):
            out[f"n_ticks_{tag}"] = i_t + 1 - np.maximum(cnt, s0)
    out["ret_first"] = x0 - x[s0]
    out["hours_since_first"] = (when - pt[s0]) / 3_600_000.0

    def integrals(tag):
        """Over the window: its length W (hours), and the integrals of x, x * time and x squared along the step
        path (x centred on the runner's first price, time from the kick-off)."""
        j = j_of[tag]
        a = np.maximum(b - LOOKBACKS[tag] / 60.0, u[j])
        head, tail = a - u[j], b - u[i_t]                  # the part of j's spell before the window; the last spell
        ia = c_ia[i_t] - c_ia[j] - xc[j] * head + xc[i_t] * tail
        ib = c_ib[i_t] - c_ib[j] - xc[j] * (a * a - u[j] ** 2) / 2.0 + xc[i_t] * (b * b - u[i_t] ** 2) / 2.0
        ic = c_ic[i_t] - c_ic[j] - xc[j] ** 2 * head + xc[i_t] ** 2 * tail
        return b - a, (a + b) / 2.0, ia, ib, ic

    for tag in ("1h", "6h", "24h"):
        j = j_of[tag]
        out[f"rv_{tag}"] = np.sqrt(np.maximum(c_sq[i_t] - c_sq[j], 0.0))
    for tag in ("6h", "24h"):
        j = j_of[tag]
        moved = i_t > j
        w, mid, ia, ib, ic = integrals(tag)
        ok = w >= MIN_SPAN_H
        mean = _div(ia, w, ok)
        cov = _div(ib - mid * ia, w, ok)
        var_x = _div(ic, w, ok) - mean * mean
        slope = np.where(moved, 12.0 * cov / np.where(ok, w * w, 1.0), 0.0)
        out[f"slope_{tag}"] = np.where(ok, slope, np.nan)
        fit = ok & moved & (var_x >= FLAT_VAR)
        r2 = _div(cov * cov, np.where(fit, w * w / 12.0 * var_x, 1.0), fit)
        out[f"slope_r2_{tag}"] = np.clip(r2, 0.0, 1.0)
        path = c_abs[i_t] - c_abs[j]
        out[f"er_{tag}"] = np.minimum(_div(np.abs(x0 - x[j]), path, path > 0), 1.0)
        pairs = np.maximum(i_t - j - 1, 0)                # pairs of consecutive changes inside the window
        j1 = np.minimum(j + 1, i_t)
        num = c_pair[i_t] - c_pair[j1]
        den = np.sqrt((c_sq[i_t] - c_sq[j1]) * (c_sq[np.maximum(i_t - 1, j)] - c_sq[j]))
        enough = pairs >= MIN_PAIRS
        out[f"ac1_{tag}"] = np.clip(_div(num, den, enough & (den > 0)), -1.0, 1.0)
        if tag == "24h":
            out["roll_24h"] = 2.0 * np.sqrt(np.maximum(-_div(num, pairs, enough), 0.0))
    for tag in ("1h", "6h"):
        j = j_of[tag]
        w, _, ia, _, _ = integrals(tag)
        out[f"twap_dev_{tag}"] = np.where(i_t > j, _div(ia, w, w > 0) - xc[i_t], np.where(w > 0, 0.0, np.nan))
        ups, downs = c_up[i_t] - c_up[j], c_dn[i_t] - c_dn[j]
        out[f"tick_imb_{tag}"] = _div((ups - downs).astype(float), (ups + downs).astype(float), ups + downs > 0)

    # ---- the 24 h range: the price as of the window's start and every tick since
    j = j_of["24h"]
    low, high = _span(np.minimum, x, j, i_t), _span(np.maximum, x, j, i_t)
    out["range_pos_24h"] = _div(x0 - low, high - low, high > low)
    out["drawdown_24h"] = high - x0
    out["runup_24h"] = x0 - low

    # ---- the largest change of the last 2 h: the latest of equals (rank of |dx|, then position, in one number)
    j = j_of["2h"]
    moved = i_t > j
    jump = np.zeros(nr)
    jump_sign = np.zeros(nr, dtype=np.int8)
    jump_min = np.full(nr, np.nan)
    if moved.any():
        rank = np.unique(np.abs(dx), return_inverse=True)[1].astype(np.int64).reshape(-1)
        at = _span(np.maximum, rank * n + np.arange(n, dtype=np.int64), j[moved] + 1, i_t[moved]) % n
        jump[moved] = np.abs(dx[at])
        jump_sign[moved] = np.sign(dx[at]).astype(np.int8)
        jump_min[moved] = (when[moved] - pt[at]) / 60_000.0
    out["jump_abs_2h"], out["jump_sign_2h"], out["jump_min_2h"] = jump, jump_sign, jump_min

    # ---- the latest tick: its change, and the run of changes the same way it ends
    sign = np.sign(dx)
    new_run = np.ones(n, dtype=bool)
    new_run[1:] = (sign[1:] != sign[:-1]) | first[1:]
    run_start = np.maximum.accumulate(np.where(new_run, np.arange(n), 0))
    run = (np.arange(n) - run_start + 1) * sign
    out["last_dx"] = np.where(i_t > s0, dx[i_t], np.nan)
    out["run_len"] = run[i_t].astype(np.int64)

    # ---- the rest of the book since this runner last traded
    types_of = t["market_type"].to_numpy()[st]
    book = pd.DataFrame({"row": np.arange(nr), "mk": mcode[st][g], "gid": g}).merge(
        pd.DataFrame({"mk": mcode[st], "other": np.arange(len(st))}), on="mk")
    book = book[book["gid"] != book["other"]]
    r, o = book["row"].to_numpy(), book["other"].to_numpy()
    then = count(o, pt[i_t][r]) - 1                       # the other runner's price as of this one's last trade
    now = count(o, when[r]) - 1
    priced = then >= st[o]
    moved_p = np.where(priced, 1.0 / price[np.maximum(now, st[o])] - 1.0 / price[np.maximum(then, st[o])], 0.0)
    need = np.where(types_of[g] == "MATCH_ODDS", BOOK_RUNNERS["MATCH_ODDS"], BOOK_RUNNERS["OVER_UNDER_25"]) - 1
    whole = (np.bincount(r, minlength=nr) == need) & (np.bincount(r[priced], minlength=nr) == need)
    p = 1.0 / price[i_t]
    p_new = np.clip(p - np.bincount(r, weights=moved_p, minlength=nr), 1.0 / PRICE_MAX, 1.0 / PRICE_MIN)
    out["book_catchup"] = np.where(whole, np.log(p_new / (1.0 - p_new)) - np.log(p / (1.0 - p)), np.nan)

    table = {
        "market_id": pa.array(rows["market_id"], pa.string()),
        "selection_id": pa.array(rows["selection_id"], pa.int64()),
        "handicap": pa.array(rows["handicap"], pa.float64()),
        "cp_min": pa.array(cp_min, pa.int32()),
        "event_id": pa.array(rows["event_id"], pa.int64()),
        "market_type": pa.array(rows["market_type"], pa.string()),
        "role": pa.array(rows["role"].where(rows["role"].notna(), None), pa.string()),
        "cp_time": pa.array(when, pa.int64()).cast(TS),
        "x": pa.array(x0, _F),
    }
    for col in SPEC.columns[len(table):]:
        v = out[col.name]
        table[col.name] = pa.array(v, col.type, from_pandas=True) if col.type == _F else pa.array(v, col.type)
    return {SPEC.name: pa.table(table)}
