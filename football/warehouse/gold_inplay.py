"""gold/inplay_minute and gold/inplay: the match as the market saw it, minute by minute, and what that says of it.

In play the match-odds market prices the result continuously: each side's chance of winning and the draw's, moved
by goals, cards and the clock. Read once a minute, that is a win-probability chart of every match Betfair held,
with no event data at all. Two tables:

**gold/inplay_minute**, one row a match a minute since the market went in play (wall clock, half time included):
the three chances as of that minute (each runner's last traded price at or before it, the pre-off close standing
in until the first in-play trade; normalised to their sum), how far to trust them (the legs priced, the book's sum,
the stalest leg's age), the over 2.5 chance while that market is open, the phase and the football minute (from the
match's half-time estimate in gold/timeline), and the score as gold/goal infers it.

A minute's reading is ``ok`` when all three legs have a price, the book's sum is inside ``BOOK_SUM`` and the legs
older than ``MAX_AGE_S`` carry no more than ``STALE_MASS`` of it: after a goal the legs re-price one at a time, and
for a minute or two the last traded prices of a thin market belong to two different scores. Staleness is weighed by
the chance a leg carries because a side three goals down stops trading at 200 and its stale price harms nothing; a
rule on age alone would keep only the minutes when the result is open, and bias every average toward the draw.

**gold/inplay**, one row a match: what the path adds to the result.
- the time-averaged chances and the points they are worth (3 for a win, 1 for a draw): ``xpts_inplay``. Every
  minute counts: one without an ok reading takes the last ok reading before it (the closing chances before the
  first), so the average is over the whole match, not over the minutes that happened to trade. Before the
  kick-off the same sum over the closing prices is ``xpts_pre``. A price is a fair forecast of its own future, so
  the average in-play chance has the pre-match chance as its expectation: ``perf`` = xpts_inplay - xpts_pre is
  zero on average and says how the match went for a side beyond what was expected of it, with every minute
  counting, not only the final whistle. It is a performance measure a result cannot give: a side that led for an
  hour and conceded twice late scores well on it and takes no points. (Measured on 48,760 matches of 2026: -0.008
  points for home sides and +0.005 for away sides on average, against a standard deviation of 0.55: last traded
  prices late in a match price the outsider a little too high.)
- the half-time chances, each side's best and worst moment, how far the chances travelled (an eventful match),
  how often the favourite changed, when the result was settled in the market's eyes (``decided_min``: the first
  minute from which the eventual result stayed at or above ``DECIDED``), and how unlikely the result once looked
  (``result_min_p``).

Everything here is the market's view from last traded prices (BASIC files: about one a minute a runner), not a
record of events; thin matches have few ok minutes (``n_ok``) and are best left out of comparisons.
"""

from __future__ import annotations

import pyarrow as pa

from football.warehouse.schema import TS, Column, TableSpec, from_frame

VERSION = 1
SCOPE = "part"
INPUTS = ("gold/event", "gold/goal", "gold/timeline", "silver/market", "silver/runner", "silver/tick")
COLUMNS = {
    "gold/event": ["event_id", "mo_market_id", "ou_market_id", "went_in_play", "kickoff_actual", "ft_pt", "result",
                   "p_home", "p_draw", "p_away"],
    "gold/goal": ["event_id", "goal_no", "pt", "home_score", "away_score"],
    "gold/timeline": ["event_id", "usable", "consistent", "reconciled", "n_unplaced", "ht_start_min", "ht_end_min"],
    "silver/market": ["market_id", "closed_pt"],
    "silver/runner": ["market_id", "market_type", "selection_id", "handicap", "role"],
    "silver/tick": ["market_id", "selection_id", "handicap", "pt", "seq", "ltp"],
}

MAX_MINUTES = 150                 # a minute grid no longer than this (normal time, half time, stoppages)
MAX_AGE_S = 300.0                 # a leg older than this is stale
STALE_MASS = 0.10                 # the share of the book stale legs may carry in an ok reading
BOOK_SUM = (0.90, 1.15)           # an ok reading's 1 / price sum is inside this
HT_START_MIN, HT_END_MIN = 48.0, 64.0   # half time where a match has no timeline row (gold_goal's own fallbacks)
DECIDED = 0.90                    # the chance from which a result is "decided"
MIN_OK = 30                       # ok minutes a match needs for its summary to be filled

MINUTE = TableSpec(
    "gold/inplay_minute",
    "One row a match a minute in play: the market's chances of home win, draw and away win as of that minute, "
    "their quality, the over 2.5 chance while open, the phase and football minute, and the inferred score.",
    (
        Column("event_id", pa.int64(), "Betfair's event id.", False),
        Column("minute", pa.int16(), "Whole minutes since the match-odds market went in play (wall clock: half "
               "time included); 0 is the kick-off.", False),
        Column("pt", TS, "The minute's moment.", False),
        Column("phase", pa.string(), "first_half, half_time or second_half, by the match's half-time estimate "
               "(gold/timeline; the 2026 medians where it has none).", False),
        Column("match_min", pa.float64(), "The football minute: the elapsed minute in the first half (at most 45), "
               "45 at half time, 45 plus the minutes since the restart in the second (at most 90).", False),
        Column("p_home", pa.float64(), "The chance of a home win as of the minute, normalised to the three legs."),
        Column("p_draw", pa.float64(), "The chance of a draw."),
        Column("p_away", pa.float64(), "The chance of an away win."),
        Column("mo_legs", pa.int8(), "Match-odds legs with a price as of the minute.", False),
        Column("mo_book_sum", pa.float64(), "Sum of 1 / price over the legs priced."),
        Column("mo_max_age_s", pa.float64(), "The stalest leg's age at the minute, seconds."),
        Column("mo_stale_mass", pa.float64(), "The share of the book's sum carried by legs older than five "
               "minutes."),
        Column("ok", pa.bool_(), "All three legs priced, the book's sum between 0.90 and 1.15, and no more than a "
               "tenth of it in legs older than five minutes: a reading to trust.", False),
        Column("p_over", pa.float64(), "The chance of over 2.5 goals as of the minute, normalised to its book; "
               "blank once that market has closed (the third goal) or where a leg is missing or stale."),
        Column("ou_open", pa.bool_(), "Whether the over/under 2.5 market was still open at the minute.", False),
        Column("home_score", pa.int16(), "Home goals so far as gold/goal infers them; blank where the match has "
               "no usable timeline. A goal the data cannot time (gold/goal.timed false) counts from the last "
               "price on."),
        Column("away_score", pa.int16(), "Away goals so far, likewise."),
        Column("score_ok", pa.bool_(), "The score columns can be relied on minute by minute: the timeline is "
               "usable, its final score agrees with the settlement (as read, or after reconciliation) and every "
               "goal has a minute. The flag is made with the match's result: a table cut down to score_ok "
               "matches is not a fair sample for testing the prices against results.", False),
    ),
    key=("event_id", "minute"), sort=("event_id", "minute"),
)

MATCH = TableSpec(
    "gold/inplay",
    "One row a match that went in play: the time-averaged in-play chances and the points they are worth against "
    "the pre-match expectation, the half-time chances, the extremes, and when the result was decided.",
    (
        Column("event_id", pa.int64(), "Betfair's event id.", False),
        Column("minutes", pa.int16(), "Minutes on the grid: from the kick-off to the match-odds market's close.",
               False),
        Column("n_ok", pa.int16(), "Minutes with an ok reading.", False),
        Column("filled", pa.bool_(), "At least thirty ok minutes: the summary columns are filled.", False),
        Column("avg_p_home", pa.float64(), "The home win chance averaged over every minute of the match (a "
               "minute without an ok reading takes the last one before it; the close before the first)."),
        Column("avg_p_draw", pa.float64(), "The draw's, likewise."),
        Column("avg_p_away", pa.float64(), "The away win's, likewise."),
        Column("xpts_pre_home", pa.float64(), "The home side's expected points at the close: 3 p_home + p_draw."),
        Column("xpts_pre_away", pa.float64(), "The away side's."),
        Column("xpts_inplay_home", pa.float64(), "The home side's expected points averaged over the match: "
               "3 avg_p_home + avg_p_draw."),
        Column("xpts_inplay_away", pa.float64(), "The away side's."),
        Column("perf_home", pa.float64(), "xpts_inplay_home less xpts_pre_home: how the match went for the home "
               "side against what the market expected of it; close to zero on average."),
        Column("perf_away", pa.float64(), "The away side's."),
        Column("ht_p_home", pa.float64(), "The home win chance at the middle of half time (an ok reading within "
               "three minutes of it)."),
        Column("ht_p_draw", pa.float64(), "The draw's at half time."),
        Column("ht_p_away", pa.float64(), "The away win's at half time."),
        Column("max_p_home", pa.float64(), "The home side's best moment: its highest win chance in play."),
        Column("min_p_home", pa.float64(), "Its lowest."),
        Column("max_p_away", pa.float64(), "The away side's highest win chance in play."),
        Column("min_p_away", pa.float64(), "Its lowest."),
        Column("travel", pa.float64(), "How far the chances moved: the sum over consecutive ok minutes of half the "
               "absolute change of the three chances (an eventful match travels far)."),
        Column("fav_changes", pa.int16(), "Times the most likely result changed between consecutive ok minutes."),
        Column("decided_min", pa.int16(), "The first minute from which the eventual result's chance stayed at or "
               "above 0.90 to the end; blank where it never did or the match has no result."),
        Column("result_min_p", pa.float64(), "The lowest chance the eventual result had in play: how unlikely it "
               "once looked."),
    ),
    key=("event_id",), sort=("event_id",),
)
SPECS = (MINUTE, MATCH)

LEGS = ("home", "draw", "away", "over", "under")


def _ms(table: pa.Table, cols: list[str]) -> pa.Table:
    for c in cols:
        i = table.schema.get_field_index(c)
        table = table.set_column(i, c, table.column(c).cast(pa.int64()))
    return table


def _empty() -> dict:
    return {MINUTE.name: MINUTE.schema.empty_table(), MATCH.name: MATCH.schema.empty_table()}


def minute_frame(inputs: dict):
    """The minute grid as a pandas frame (the columns of gold/inplay_minute), or None where nothing went in play."""
    import numpy as np
    import pandas as pd
    import pyarrow.compute as pc

    ev = _ms(inputs["gold/event"].select(COLUMNS["gold/event"]), ["kickoff_actual", "ft_pt"]).to_pandas()
    ev = ev[ev["went_in_play"] & ev["mo_market_id"].notna() & ev["kickoff_actual"].notna() & ev["ft_pt"].notna()]
    ev = ev[ev["ft_pt"] > ev["kickoff_actual"]].reset_index(drop=True)
    if ev.empty:
        return None
    mk = _ms(inputs["silver/market"].select(COLUMNS["silver/market"]), ["closed_pt"]).to_pandas()
    ev = ev.merge(mk.rename(columns={"market_id": "ou_market_id", "closed_pt": "ou_closed"}), on="ou_market_id",
                  how="left")
    ev["start"] = ev["kickoff_actual"].astype("int64")
    ev["n_min"] = np.minimum((ev["ft_pt"].astype("int64") - ev["start"]) // 60_000, MAX_MINUTES).astype("int64") + 1
    ev["eid"] = np.arange(len(ev))

    # the grid: one row an event a minute
    eid = np.repeat(ev["eid"].to_numpy(), ev["n_min"].to_numpy())
    minute = np.arange(len(eid)) - np.repeat(np.cumsum(ev["n_min"].to_numpy()) - ev["n_min"].to_numpy(),
                                             ev["n_min"].to_numpy())
    g = pd.DataFrame({"eid": eid, "minute": minute})
    g["pt"] = ev["start"].to_numpy()[eid] + minute * 60_000

    # each leg's price as of the minute: the last one at or before it, pre-off prices included
    r = inputs["silver/runner"].select(COLUMNS["silver/runner"]).to_pandas()
    r = r[r["role"].isin(LEGS) & r["market_type"].isin(["MATCH_ODDS", "OVER_UNDER_25"])]
    own = pd.concat([ev[["eid", "mo_market_id"]].rename(columns={"mo_market_id": "market_id"}),
                     ev[["eid", "ou_market_id"]].rename(columns={"ou_market_id": "market_id"}).dropna()])
    r = r.merge(own, on="market_id")
    tick = inputs["silver/tick"]
    tick = tick.filter(pc.is_in(tick.column("market_id"), value_set=pa.array(own["market_id"].unique())))
    t = _ms(tick.select(["market_id", "selection_id", "handicap", "pt", "seq", "ltp"]), ["pt"]).to_pandas()
    t = t.merge(r[["market_id", "selection_id", "handicap", "eid", "role"]], on=["market_id", "selection_id",
                                                                                 "handicap"])
    t = t.sort_values(["pt", "seq"], kind="stable")
    for leg in LEGS:
        tl = t.loc[t["role"] == leg, ["eid", "pt", "ltp"]].rename(columns={"pt": f"pt_{leg}", "ltp": f"ltp_{leg}"})
        g = pd.merge_asof(g.sort_values("pt", kind="stable"), tl, left_on="pt", right_on=f"pt_{leg}", by="eid",
                          direction="backward", allow_exact_matches=True)
    g = g.sort_values(["eid", "minute"], kind="stable").reset_index(drop=True)

    raw = {leg: 1.0 / g[f"ltp_{leg}"].to_numpy(float) for leg in LEGS}
    age = {leg: (g["pt"].to_numpy(float) - g[f"pt_{leg}"].to_numpy(float)) / 1000.0 for leg in LEGS}
    mo = np.c_[raw["home"], raw["draw"], raw["away"]]
    legs = np.isfinite(mo).sum(1)
    book = np.nansum(mo, axis=1)
    book = np.where(legs > 0, book, np.nan)
    full = legs == 3
    max_age = np.where(legs > 0, np.max(np.where(np.isfinite(mo), np.c_[age["home"], age["draw"], age["away"]],
                                                 -np.inf), axis=1), np.nan)
    ages = np.c_[age["home"], age["draw"], age["away"]]
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(full[:, None], mo / book[:, None], np.nan)
        stale = np.where(full, np.where(ages > MAX_AGE_S, mo, 0.0).sum(1) / book, np.nan)
    ok = full & (stale <= STALE_MASS) & (book >= BOOK_SUM[0]) & (book <= BOOK_SUM[1])

    e = ev.iloc[g["eid"].to_numpy()]
    ou_open = (e["ou_market_id"].notna() & (e["ou_closed"].isna() | (g["pt"].to_numpy() < e["ou_closed"].to_numpy()))
               ).to_numpy()
    ou_sum = raw["over"] + raw["under"]
    ou_good = (ou_open & np.isfinite(ou_sum) & (np.maximum(age["over"], age["under"]) <= MAX_AGE_S)
               & (ou_sum >= BOOK_SUM[0]) & (ou_sum <= BOOK_SUM[1]))
    with np.errstate(invalid="ignore", divide="ignore"):
        p_over = np.where(ou_good, raw["over"] / ou_sum, np.nan)

    # the phase and football minute, by the match's half time (the medians where it has none)
    tl = inputs["gold/timeline"].select(COLUMNS["gold/timeline"]).to_pandas()
    e = e[["event_id"]].merge(tl, on="event_id", how="left")
    hs = e["ht_start_min"].fillna(HT_START_MIN).to_numpy(float)
    he = e["ht_end_min"].fillna(HT_END_MIN).to_numpy(float)
    mi = g["minute"].to_numpy(float)
    phase = np.where(mi < hs, "first_half", np.where(mi < he, "half_time", "second_half"))
    match_min = np.where(mi < hs, np.minimum(mi, 45.0), np.where(mi < he, 45.0, np.minimum(45.0 + (mi - he), 90.0)))
    usable = e["usable"].eq(True).to_numpy()                      # a match with no timeline row is not usable

    # the inferred score as of the minute: the last goal at or before it
    out = pd.DataFrame({
        "event_id": e["event_id"].to_numpy(), "minute": g["minute"].to_numpy(), "pt": g["pt"].to_numpy(),
        "phase": phase, "match_min": match_min, "p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2],
        "mo_legs": legs, "mo_book_sum": book, "mo_max_age_s": max_age, "mo_stale_mass": stale, "ok": ok,
        "p_over": p_over,
        "ou_open": ou_open,
        "score_ok": usable & (e["consistent"].eq(True) | e["reconciled"].eq(True)).to_numpy()
        & e["n_unplaced"].eq(0).to_numpy()})
    goals = _ms(inputs["gold/goal"].select(COLUMNS["gold/goal"]), ["pt"]).to_pandas()
    goals = goals.rename(columns={"pt": "goal_pt"}).sort_values(["goal_pt", "goal_no"], kind="stable")
    out["row"] = np.arange(len(out))
    sc = pd.merge_asof(out[["row", "event_id", "pt"]].sort_values("pt", kind="stable"),
                       goals[["event_id", "goal_pt", "home_score", "away_score"]], left_on="pt", right_on="goal_pt",
                       by="event_id", direction="backward", allow_exact_matches=True).sort_values("row")
    for c in ("home_score", "away_score"):
        v = sc[c].to_numpy(float)
        out[c] = np.where(usable, np.where(np.isfinite(v), v, 0.0), np.nan)
    return out.drop(columns="row"), ev


def summary_frame(minute, ev):
    """One row a match from its minute rows (the columns of gold/inplay)."""
    import numpy as np
    import pandas as pd

    m = minute.merge(ev[["event_id", "result", "p_home", "p_draw", "p_away"]].rename(
        columns={"p_home": "c_home", "p_draw": "c_draw", "p_away": "c_away"}), on="event_id", how="left")
    P = m[["p_home", "p_draw", "p_away"]].to_numpy(float)
    ok = m["ok"].to_numpy(bool)
    first = (m["event_id"] != m["event_id"].shift()).to_numpy()
    grp = np.cumsum(first) - 1
    n = int(grp.max()) + 1
    ids = m["event_id"].to_numpy()[first]
    minutes = np.bincount(grp, minlength=n)
    n_ok = np.bincount(grp, weights=ok, minlength=n).astype(int)
    close = m.loc[first, ["c_home", "c_draw", "c_away"]].to_numpy(float)
    # every minute counts: the last ok reading at or before it, the closing chances before the first
    rows = np.arange(len(m))
    last = np.maximum.accumulate(np.where(ok, rows, -1))
    start = np.flatnonzero(first)[grp]
    held = np.where((last >= start)[:, None], P[np.maximum(last, 0)], close[grp])
    have = np.isfinite(held).all(1)
    Pz = np.where(have[:, None], held, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = (np.stack([np.bincount(grp, weights=Pz[:, j], minlength=n) for j in range(3)], axis=1)
               / np.bincount(grp, weights=have, minlength=n)[:, None])

    def ext(col: int, fn, fill: float):
        v = np.where(ok, P[:, col], fill)
        out = np.full(n, fill)
        (np.maximum if fn == "max" else np.minimum).at(out, grp, v)
        return np.where(n_ok > 0, out, np.nan)

    # how far the chances travelled and how often the favourite changed, over consecutive ok minutes of a match
    idx = np.flatnonzero(ok)
    same = grp[idx][1:] == grp[idx][:-1]
    step = 0.5 * np.abs(P[idx][1:] - P[idx][:-1]).sum(1)
    travel = np.bincount(grp[idx][1:][same], weights=step[same], minlength=n)
    fav = P[idx].argmax(1)
    fav_changes = np.bincount(grp[idx][1:][same], weights=(fav[1:] != fav[:-1])[same], minlength=n).astype(int)

    # the eventual result's path: its lowest chance, and the first minute from which it stayed at or above DECIDED
    res = m["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy(float)
    has = ok & np.isfinite(res)
    pr = np.where(has, P[np.arange(len(P)), np.nan_to_num(res, nan=0).astype(int)], np.nan)
    rmin = np.full(n, np.inf)
    np.minimum.at(rmin, grp[has], pr[has])
    below = has & (pr < DECIDED)
    last_below = np.full(n, -1)
    np.maximum.at(last_below, grp[below], m["minute"].to_numpy()[below])
    after = has & (m["minute"].to_numpy() > last_below[grp])
    decided = np.full(n, np.iinfo(np.int64).max)
    np.minimum.at(decided, grp[after], m["minute"].to_numpy()[after])
    decided = np.where(decided == np.iinfo(np.int64).max, np.nan, decided.astype(float))

    # half time: the ok reading nearest the middle of the break, within three minutes
    mids = minute.loc[minute["phase"] == "half_time"].groupby("event_id")["minute"].agg(["min", "max"])
    mid = pd.Series(np.round((mids["min"] + mids["max"]) / 2.0), index=mids.index).reindex(ids).to_numpy(float)
    gap = np.abs(m["minute"].to_numpy(float) - mid[grp])
    cand = ok & np.isfinite(gap) & (gap <= 3)
    order = np.lexsort((gap, grp))
    order = order[cand[order]]
    pick = order[np.r_[True, grp[order][1:] != grp[order][:-1]]] if len(order) else order
    ht = np.full((n, 3), np.nan)
    ht[grp[pick]] = P[pick]

    filled = n_ok >= MIN_OK
    blank = lambda a: np.where(filled, a, np.nan)                 # noqa: E731
    xh_pre, xa_pre = 3 * close[:, 0] + close[:, 1], 3 * close[:, 2] + close[:, 1]
    xh_in, xa_in = 3 * avg[:, 0] + avg[:, 1], 3 * avg[:, 2] + avg[:, 1]
    return pd.DataFrame({
        "event_id": ids, "minutes": minutes, "n_ok": n_ok, "filled": filled,
        "avg_p_home": blank(avg[:, 0]), "avg_p_draw": blank(avg[:, 1]), "avg_p_away": blank(avg[:, 2]),
        "xpts_pre_home": xh_pre, "xpts_pre_away": xa_pre,
        "xpts_inplay_home": blank(xh_in), "xpts_inplay_away": blank(xa_in),
        "perf_home": blank(xh_in - xh_pre), "perf_away": blank(xa_in - xa_pre),
        "ht_p_home": blank(ht[:, 0]), "ht_p_draw": blank(ht[:, 1]), "ht_p_away": blank(ht[:, 2]),
        "max_p_home": blank(ext(0, "max", -np.inf)), "min_p_home": blank(ext(0, "min", np.inf)),
        "max_p_away": blank(ext(2, "max", -np.inf)), "min_p_away": blank(ext(2, "min", np.inf)),
        "travel": blank(travel), "fav_changes": np.where(filled, fav_changes, 0),
        "decided_min": np.where(filled, decided, np.nan),
        "result_min_p": blank(np.where(np.isfinite(rmin), rmin, np.nan)),
    })


def build(inputs: dict) -> dict:
    got = minute_frame(inputs)
    if got is None:
        return _empty()
    minute, ev = got
    return {MINUTE.name: from_frame(MINUTE, minute), MATCH.name: from_frame(MATCH, summary_frame(minute, ev))}
