"""The live-record data set of the 100-strategy harness, b_live (racing², conventions K1-K10, registered 24 Nov 2026).

    load_b_live()    the recorder's books 1 Oct - 23 Nov 2026 as one row a runner a market with the book at every
                     mark, the served 06:00 prices, the live trader's own fills, and what the raw one-minute books add

THE BOOKS LAG: betfair_live_marks and the recorder's raw books come from Betfair's DELAYED key. Every price a
strategy 'takes' here was on the screen some time before the true market showed it; every b_live reading is also
printed at a price 2% worse (K5) and none of them is a fill.

What the loader returns (a dict; every pandas frame in it is passed to every strategy, and none may be changed):

    runners     one row a runner a market (GB/IE, WIN and PLACE): the catalogue (venue, start, names, race type, days
                since the last run, wearing, F = Betfair's forecast price), predicted_bfsp (S3 predictions/<day>.csv,
                WIN markets), bsp and final_status from the final mark (the BSP from betfair_prices where the mark has
                none), places, and for each mark a block of columns <tag>_<field>:
                  tags    c08 c09 c10 c11 c12 (the clock marks, UK), o120 o60 o30 o15 o10 o5 o3 o1 (minutes before the
                          off), last
                  fields  t (the poll, UTC), mto (minutes to the off), mstatus, status, back1..3, back1..3_size,
                          lay1..3, lay1..3_size, last_traded, mtm (the market's matched), n_active
                ``book(runners, tag)`` gives one mark's block under its plain names.
    fills       BASE-B (K7): the live trader's matched backs, one row an order, with the horse's settle row (BSP,
                result, Betfair's settled price) and each fill's cut_factor (K6 CUT)
    settles     the live trader's settle rows, one a horse (S027)
    prices      betfair_prices rows of the window (BSP, in-play high and low, result; win and place)
    removals    every runner the raw books show REMOVED: its reduction factor, removal time and the first poll that
                showed it (K6)
    hist        each card runner's earlier runs in race_results (S047)
    steam       the CLEAN steamer's score for each card runner (S029, S031)
    replay      the bets of every replayed arm (REPLAY-B and its variants, research/arms; one row an order)
    day_<KEY>   what a strategy's own pass over one day's raw books made (strategies_b_live.DAY_FUNCS)
    days        the ISO race days the data set holds (the marks' days)
    meta        a plain dict: days_books, days_replay, days_preds, days_fills, days_steam, problems, day_errors, ...

Sources: horse_racing.db read-only (betfair_live_markets, betfair_live_marks, live_orders, betfair_prices,
race_results) and S3 read-only through research/arms/core (predictions/, racecards/, betfair_live/<day>/ raw books,
Betfair's price files for the replay's settlement). Nothing here reaches Betfair or HRB and nothing places a bet.

K6 CUT as it is read here. K6 cuts an entry price by the factor of EVERY runner removed after the fill
(research/arms/core.reduced), and that is what a BACK gets (CUT_MIN_AF_BACK = 0). Betfair itself makes no reduction
for a factor under 2.5% in a win market (research/queries/done/removal_by_market_1109.py reads it so), so the
literal rule is the harsher one for a back and the kinder one for a lay; a LAY is therefore cut only by factors of
at least 2.5 (CUT_MIN_AF_LAY: the conservative reading on that side). Two exceptions, each said where it is made:
S024 and its benchmark, whose specification is the 2.5% rule itself, cut their backs only by factors of at least
2.5; and BASE-B's real fills are scaled to Betfair's own settled price where the settle row carries it. Both
readings are printed for REPLAY-B and BASE-B at load, and each replayed bet carries both (cut_factor,
cut_factor_bf).

Environment (all optional):
    STRAT100_DB             the database (default horse_racing.db in the working directory)
    STRAT100_WORK           where a day's pass over the raw books is cached (default out/strat100; folder b_live)
    STRAT100_B_DAYS         the days read, 'a..b' or a comma list (default 2026-10-01..2026-11-23)
    STRAT100_B_JOBS         processes for the pass over the raw books (default: the machine's cores, at most 4)
    STRAT100_B_RAW=0        skip the raw books (every strategy that needs CUT or a raw poll: 'no data found')
    STRAT100_B_REPLAY=0     skip the replays (S001-S007, S009, S010, S013, S014, S017-S020, S037: 'no data found')
    STRAT100_B_STEAMER=0    skip the steamer (S031: 'no data found'; S029's steamer vote is 0)
    STRAT100_B_DRAWS        draws of the BSP book a race in the replays (default 20000, K5)
    STRAT100_B_MAX_MINUTES  the raw pass's own guard (default 240): days not done by then are left out and said so
    STRAT100_B_REFRESH=1    ignore the cached days
    STRAT100_B_CUT_MIN_AF_BACK, STRAT100_B_CUT_MIN_AF_LAY   the smallest reduction factor that cuts a back's or a
                            lay's price (defaults 0 and 2.5)
"""
from __future__ import annotations

import gzip
import hashlib
import io
import os
import sqlite3
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.strat100 import core

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

UK = ZoneInfo("Europe/London")
FIRST_DAY, LAST_DAY = core.WINDOWS["b_live"]["select"][0], core.WINDOWS["b_live"]["confirm"][1]
KEY = ["market_id", "selection_id"]
PAIR_B = ["event_id", "market_start_utc", "selection_id"]        # K5: a win row and its place row
BETFAIR_MIN_AF = 2.5                  # Betfair makes no reduction to a win bet for a factor under this
CUT_MIN_AF_BACK = float(os.environ.get("STRAT100_B_CUT_MIN_AF_BACK") or 0.0)             # K6: every factor
CUT_MIN_AF_LAY = float(os.environ.get("STRAT100_B_CUT_MIN_AF_LAY") or BETFAIR_MIN_AF)    # conservative for a lay
STEAM_CUTOFF = "2026-03-27"           # S031: the CLEAN steamer fitted once on 2018-01-01..2026-03-27
DELAYED = ("the recorder's books are on Betfair's DELAYED key: they lag the live market, a price 'taken' here is not "
           "a fill, and every reading is also printed at a price 2% worse (K5)")

#: the marks of betfair_live_marks and the tag each takes in the runners frame
TAGS = {"clock_08:00": "c08", "clock_09:00": "c09", "clock_10:00": "c10", "clock_11:00": "c11", "clock_12:00": "c12",
        "off_120": "o120", "off_60": "o60", "off_30": "o30", "off_15": "o15", "off_10": "o10", "off_5": "o5",
        "off_3": "o3", "off_1": "o1", "last": "last"}
LEVELS = ["back1", "back1_size", "back2", "back2_size", "back3", "back3_size",
          "lay1", "lay1_size", "lay2", "lay2_size", "lay3", "lay3_size"]
#: betfair_live_marks column -> its name in a mark's block
MARK_FIELDS = {"polled_utc": "t", "minutes_to_off": "mto", "market_status": "mstatus", "runner_status": "status",
               **{c: c for c in LEVELS}, "last_traded": "last_traded", "market_total_matched": "mtm",
               "active_runners": "n_active"}
#: the explicit column lists (the data map of 24 Nov 2026); a column a table lacks is read as empty and said so
CAT_COLS = ["race_date", "market_id", "market_type", "market_name", "market_start_utc", "venue", "country",
            "event_id", "race_type", "selection_id", "runner_name", "official_rating", "days_since_last_run",
            "wearing", "forecast_num", "forecast_den", "horse_norm", "race_results_id"]
MARK_COLS = ["race_date", "market_id", "selection_id", "mark", "polled_utc", "minutes_to_off", "market_status",
             "runner_status", *LEVELS, "last_traded", "market_total_matched", "active_runners", "bsp"]
ORDER_COLS = ["race_date", "seq", "ts", "mode", "event", "market_id", "minutes_to_off", "selection_id", "runner",
              "p_model", "p_market", "edge", "move", "best_back", "back_size", "target", "asked", "matched",
              "avg_price", "status", "result", "bsp", "clv", "pnl", "commission"]
ORDER_NUM = ["seq", "minutes_to_off", "selection_id", "p_model", "p_market", "edge", "move", "best_back", "back_size",
             "target", "asked", "matched", "avg_price", "bsp", "clv", "pnl", "commission"]      # TEXT in the table
PRICE_COLS = ["country", "market_type", "race_date", "selection_id", "win_lose", "bsp", "ppwap", "ipmax", "ipmin",
              "pp_vol"]
RR_COLS = ["race_date", "horse_name", "race_type", "race_name", "headgear"]
#: the raw one-minute books (betfair_recorder.BOOK_FIELDS): what is read of them
RAW_COLS = ["polled_utc", "source", "market_id", "status", "inplay", "number_of_active_runners",
            "market_total_matched", "selection_id", "runner_status", "adjustment_factor", "removal_utc",
            "last_traded", *LEVELS, "sp_actual", "number_of_winners"]
RAW_NUM = ["number_of_active_runners", "market_total_matched", "adjustment_factor", "last_traded", *LEVELS,
           "sp_actual", "number_of_winners"]
BETS_COLS = ["race_date", "race_key", "selection", "side", "stake", "entry_price", "exit", "exit_price", "won",
             "void", "cut_factor", "ts"]


class NoData(RuntimeError):
    """'no data found': a source the harness needs is missing or empty (never replaced by invented rows)."""


_QUIET = [False]


def _say(*a) -> None:
    if not _QUIET[0]:
        print(*a, flush=True)


# ---------------------------------------------------------------------------------------------------------------
# small pure helpers (the strategies use them too)
# ---------------------------------------------------------------------------------------------------------------

def _num(df: pd.DataFrame, cols) -> pd.DataFrame:
    """A copy with ``cols`` coerced to numbers (TEXT columns, blanks and 'None' become NaN)."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _utc(s) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def _up(s: pd.Series) -> pd.Series:
    """Upper-case text, a blank where the value is missing."""
    return s.astype(object).where(s.notna(), "").astype(str).str.strip().str.upper()


def uk_time(day: str, hhmm: str) -> pd.Timestamp:
    """'HH:MM' UK on the ISO day, as a UTC timestamp."""
    d = date.fromisoformat(str(day)[:10])
    h, m = (int(x) for x in hhmm.split(":"))
    return pd.Timestamp(datetime(d.year, d.month, d.day, h, m, tzinfo=UK).astimezone(timezone.utc))


def book(runners: pd.DataFrame, tag: str, mtype: str = "WIN", active: bool = True) -> pd.DataFrame:
    """One mark's book, a new frame: the static columns and the mark's block under its plain names (t, mto, mstatus,
    status, back1.., lay1.., last_traded, mtm, n_active), for the runners the mark holds; with ``active`` only the
    runners ACTIVE in an OPEN market at the mark. ``n_rows`` is the count of those rows in the market: a race is
    whole only where it equals n_active (no runner of the book is missing from the frame)."""
    block = {f"{tag}_{v}": v for v in MARK_FIELDS.values()}
    missing = [c for c in block if c not in runners.columns]
    if missing:
        raise NoData(f"no data found: the runners frame has no mark '{tag}'")
    static = [c for c in runners.columns if not any(c.startswith(f"{t}_") for t in TAGS.values())]
    b = runners.loc[runners["market_type"].eq(mtype) & runners[f"{tag}_t"].notna(), static + list(block)]
    b = b.rename(columns=block)
    if active:
        b = b[b["status"].eq("ACTIVE") & b["mstatus"].eq("OPEN")]
    b = b.reset_index(drop=True)
    b["n_rows"] = b.groupby("market_id")["selection_id"].transform("size") if len(b) else 0
    return b


def min_af_for(side: str) -> float:
    """The smallest factor that cuts a bet of this side (see the module's text)."""
    return CUT_MIN_AF_LAY if side == "lay" else CUT_MIN_AF_BACK


def cut_factor(removals: pd.DataFrame | None, market_id, selection_id, ts, min_af: float = 0.0) -> np.ndarray:
    """K6 CUT: for each bet (its market, its runner, the time it was matched) the product of (1 - af/100) over the
    market's OTHER runners removed after that time whose factor is at least ``min_af`` (0: every factor, K6's
    words; 2.5: Betfair's own rule). 1.0 where none; the core applies max(1.01, price x factor)."""
    min_af = float(min_af)
    n = len(market_id)
    out = np.ones(n)
    if removals is None or len(removals) == 0 or n == 0:
        return out
    b = pd.DataFrame({"i": np.arange(n), "market_id": np.asarray(market_id, dtype=object).astype(str),
                      "sel": pd.to_numeric(pd.Series(np.asarray(selection_id)), errors="coerce").to_numpy(),
                      "ts": _utc(pd.Series(np.asarray(ts))).to_numpy()})
    r = removals[["market_id", "selection_id", "af", "rt"]].copy()
    r["market_id"] = r["market_id"].astype(str)
    m = b.merge(r, on="market_id", how="inner")
    m = m[(m["rt"].to_numpy() > m["ts"].to_numpy()) & (m["selection_id"] != m["sel"]) & (m["af"] >= min_af)
          & (m["af"] > 0)]
    if m.empty:
        return out
    f = (1.0 - m["af"].clip(upper=99.0) / 100.0).groupby(m["i"]).prod()
    out[f.index.to_numpy()] = f.to_numpy()
    return out


def parse_days(spec: str | None = None) -> list[str]:
    spec = (spec if spec is not None else os.environ.get("STRAT100_B_DAYS")) or f"{FIRST_DAY}..{LAST_DAY}"
    if ".." in spec:
        a, b = (date.fromisoformat(x.strip()) for x in spec.split(".."))
        out = [(a + timedelta(n)).isoformat() for n in range((b - a).days + 1)]
    else:
        out = sorted({date.fromisoformat(x.strip()).isoformat() for x in spec.split(",") if x.strip()})
    return [d for d in out if FIRST_DAY <= d <= LAST_DAY]


# ---------------------------------------------------------------------------------------------------------------
# S3, read-only, through research/arms/core (boto3; PRED_REGION, PRED_BUCKET)
# ---------------------------------------------------------------------------------------------------------------

def _get(key: str) -> bytes | None:
    """An S3 object's bytes, or None (research/arms/core.get prints the miss). Tests replace this."""
    from research.arms import core as ac
    return ac.get(key)


def _ls(prefix: str) -> list[str]:
    """The keys under a prefix, or [] (no bucket, no credentials)."""
    try:
        from research.arms import core as ac
        keys, kw = [], {"Bucket": ac.bucket(), "Prefix": prefix}
        while True:
            resp = ac.s3().list_objects_v2(**kw)
            keys += [o["Key"] for o in resp.get("Contents", [])]
            if not resp.get("IsTruncated"):
                return sorted(keys)
            kw["ContinuationToken"] = resp["NextContinuationToken"]
    except Exception as exc:  # noqa: BLE001 -- a listing that fails is 'no data found', said by the caller
        _say(f"   {prefix}: not listed ({type(exc).__name__})")
        return []


def _csv(key: str, **kw) -> pd.DataFrame:
    raw = _get(key)
    if raw is None:
        return pd.DataFrame()
    if key.endswith(".gz"):
        raw = gzip.decompress(raw)
    try:
        return pd.read_csv(io.BytesIO(raw), low_memory=False, **kw)
    except Exception as exc:  # noqa: BLE001 -- an empty or broken object is 'no data found'
        _say(f"   {key}: not read ({type(exc).__name__}: {exc})")
        return pd.DataFrame()


@contextmanager
def _memo_s3():
    """Within one day's pass each S3 object is fetched once: research/arms/core.load_day and the raw reader ask for
    the same books. research/arms/core.get is put back after."""
    from research.arms import core as ac
    orig, memo = ac.get, {}

    def get(key):
        if key not in memo:
            memo[key] = orig(key)
        return memo[key]
    ac.get = get
    try:
        yield
    finally:
        ac.get = orig


def read_predictions(days: list[str]) -> pd.DataFrame:
    """S3 predictions/<day>.csv (the served 06:00 price) for each day: race_date, market_id, selection_id,
    predicted_bfsp and the names' keys for the runners a file carries no ids for."""
    import betfair_prices as bp
    text = {c: str for c in ("date", "race_date", "venue", "track", "race_time", "runner_name", "horse_name",
                             "market_id")}
    out = []
    for day in days:
        x = _csv(f"predictions/{day}.csv", dtype=text)
        if x.empty or "predicted_bfsp" not in x.columns:
            continue
        venue = x["venue"] if "venue" in x.columns else x.get("track", pd.Series("", index=x.index))
        name = x["runner_name"] if "runner_name" in x.columns else x.get("horse_name", pd.Series("", index=x.index))
        out.append(pd.DataFrame({
            "race_date": day,
            "market_id": x["market_id"].astype(object) if "market_id" in x.columns else None,
            "selection_id": pd.to_numeric(x["selection_id"], errors="coerce") if "selection_id" in x.columns else np.nan,
            "predicted_bfsp": pd.to_numeric(x["predicted_bfsp"], errors="coerce"),
            "k_track": venue.map(bp.normalise_track), "k_horse": name.map(bp.normalise_horse),
            "k_time": x.get("race_time", pd.Series("", index=x.index)).astype(str).map(bp.race_time_to_24h)}))
    if not out:
        return pd.DataFrame(columns=["race_date", "market_id", "selection_id", "predicted_bfsp", "k_track",
                                     "k_horse", "k_time"])
    p = pd.concat(out, ignore_index=True)
    p = p[p["predicted_bfsp"] > 1.0]
    _say(f"   predictions: {len(p):,} priced runners on {p['race_date'].nunique()} of {len(days)} days "
         f"({int(p['market_id'].notna().sum()):,} with Betfair ids)")
    return p


def read_raw(day: str, names=("books_trader", "books")) -> pd.DataFrame:
    """The day's raw one-minute books (S3 betfair_live/<day>/<name>.csv.gz), typed: t (the poll, UTC), market_id,
    selection_id, status, inplay (bool), runner_status, adjustment_factor, removal_utc (UTC), the three levels each
    side, last_traded, market_total_matched, number_of_active_runners, number_of_winners. Empty when not recorded."""
    parts = []
    for n in names:
        x = _csv(f"betfair_live/{day}/{n}.csv.gz", dtype={"market_id": str}, usecols=lambda c: c in RAW_COLS)
        if len(x) and {"polled_utc", "market_id", "selection_id"} <= set(x.columns):
            parts.append(x)
    if not parts:
        return pd.DataFrame(columns=["t"] + [c for c in RAW_COLS if c != "polled_utc"])
    raw = pd.concat(parts, ignore_index=True)
    return type_raw(raw)


def type_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw book rows typed (a new frame); see read_raw."""
    raw = raw.reindex(columns=list(dict.fromkeys([*raw.columns, *RAW_COLS])))
    raw = _num(raw, RAW_NUM)
    raw["t"] = _utc(raw["polled_utc"])
    raw["selection_id"] = pd.to_numeric(raw["selection_id"], errors="coerce")
    raw = raw[raw["t"].notna() & raw["selection_id"].notna() & raw["market_id"].notna()].copy()
    raw["selection_id"] = raw["selection_id"].astype("int64")
    raw["market_id"] = raw["market_id"].astype(str)
    raw["status"], raw["runner_status"] = _up(raw["status"]), _up(raw["runner_status"])
    raw["inplay"] = raw["inplay"].astype(str).str.strip().str.lower().isin(["1", "true", "1.0"])
    raw["removal_utc"] = _utc(raw["removal_utc"])
    raw["source"] = raw["source"].astype(object).where(raw["source"].notna(), "").astype(str)
    raw = raw.drop(columns=["polled_utc"]).sort_values(["market_id", "t", "selection_id"], kind="stable")
    return raw.drop_duplicates(["market_id", "selection_id", "t"], keep="first").reset_index(drop=True)


def removals_of(raw: pd.DataFrame) -> pd.DataFrame:
    """Every runner the raw books show REMOVED: af (its last reduction factor, 0 where none is given), rt (Betfair's
    removal time, else the first poll that showed it) and first_seen (that first poll): research/arms/core.load_day's
    own table, with the first poll kept beside it."""
    cols = ["market_id", "selection_id", "af", "rt", "first_seen"]
    rem = raw[raw["runner_status"].eq("REMOVED")]
    if rem.empty:
        return pd.DataFrame(columns=cols)
    rem = rem.assign(af=rem["adjustment_factor"].fillna(0.0), rt=rem["removal_utc"].fillna(rem["t"]))
    return rem.groupby(KEY).agg(af=("af", "last"), rt=("rt", "min"), first_seen=("t", "min")).reset_index()[cols]


def places_of(raw: pd.DataFrame) -> pd.DataFrame:
    """K9: the places each market pays, the last number_of_winners its raw books carry (from 7 Oct 2026)."""
    x = raw[raw["number_of_winners"].notna()]
    if x.empty:
        return pd.DataFrame(columns=["market_id", "places_raw"])
    return x.groupby("market_id")["number_of_winners"].last().rename("places_raw").reset_index()


# ---------------------------------------------------------------------------------------------------------------
# the replays (K7: research/arms, the rule as it stands on 24 Nov 2026 and the arms registered beside it)
# ---------------------------------------------------------------------------------------------------------------

#: K7, pinned here so a later change of trading/config_live.json does not move the benchmark
RULE = {"strategy": "closing_clv", "staking": "to_win", "target": 400.0, "clv_bar": 0.03, "closing_volume": False,
        "topups": True, "stake_floor": 15.0, "topup_to": "stake", "min_move": 0.0, "ladder": True, "max_topups": 3,
        "trade_from": "05:15", "trade_until": "10:00", "stop_before_off": 15, "trade_out": True,
        "trade_out_at": "fill", "limits.max_stake": 200.0, "limits.max_price": 200.0,
        "limits.max_daily_turnover": 4000.0, "limits.min_stake": 2.0, "limits.min_bsp_liability": 10.0,
        "limits.liquidity_share": 1.0}


def load_replay_day(day: str):
    """research/arms/core.load_day(day, 'win'): the day's books, markets, prices, real fills and settlement, or None
    (no price file yet, no books). Tests replace this with a toy day."""
    from research.arms import core as ac
    return ac.load_day(date.fromisoformat(day), "win")


def _settled_frame(D) -> pd.DataFrame:
    rows = [{"market_id": str(m), "selection_id": int(s), "bsp": float(b), "result": str(st)}
            for (m, s), (b, st) in D.settled.items()]
    return pd.DataFrame(rows, columns=["market_id", "selection_id", "bsp", "result"])


def _orders_to_bets(rows: pd.DataFrame, D, side: str) -> pd.DataFrame:
    """Matched orders (market_id, selection_id, ts, matched, avg_price) as bets closed at the day's BSP (K3), each
    cut by the removals after it (K6); a runner with no BSP or REMOVED is void."""
    if rows is None or len(rows) == 0:
        return pd.DataFrame(columns=BETS_COLS)
    x = rows.copy()
    x["market_id"] = x["market_id"].astype(str)
    x["selection_id"] = pd.to_numeric(x["selection_id"], errors="coerce").astype("int64")
    for c in ("matched", "avg_price"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.drop(columns=[c for c in ("bsp", "result") if c in x.columns]).merge(
        _settled_frame(D), on=KEY, how="left")
    ts = _utc(x["ts"])
    ok = x["bsp"].gt(1.0) & x["result"].isin(["WINNER", "LOSER"])
    return pd.DataFrame({
        "race_date": str(D.day), "race_key": x["market_id"], "selection": x["selection_id"].astype(str),
        "side": side, "stake": x["matched"], "entry_price": x["avg_price"], "exit": "bsp", "exit_price": x["bsp"],
        "won": np.where(ok, x["result"].eq("WINNER").astype(float), np.nan), "void": ~ok.to_numpy(),
        "cut_factor": cut_factor(D.removed, x["market_id"], x["selection_id"], ts, min_af_for(side)),
        "cut_factor_bf": cut_factor(D.removed, x["market_id"], x["selection_id"], ts, BETFAIR_MIN_AF),
        "ts": ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "edge": pd.to_numeric(x["edge"], errors="coerce") if "edge" in x.columns else np.nan,
        "minutes_to_off": pd.to_numeric(x["minutes_to_off"], errors="coerce") if "minutes_to_off" in x.columns else np.nan,
    })


def replay_day(D, arms: dict, draws: int = 20000, forecast: dict | None = None) -> pd.DataFrame:
    """Every replayed arm on one loaded day: {id: {'kind': 'rule' | 'lay', 'over': config changes, 'policy': the
    session's policy}} (strategies_b_live.REPLAY_ARMS). The plans are made once at a 0% bar and each arm holds its
    own bar when it executes, so the arms share research/arms/core's plan cache; a runner's expected CLV does not
    depend on the bar, so each arm trades exactly as a session planned at its own bar. Returns the arms' bets, one
    row an order, with the arm's id."""
    from research.arms import core as ac
    from research.strat100 import strategies_b_live as S
    rule_cls, lay_cls = S.session_classes()
    ac.set_plan_cfg(ac.make_cfg(**{**RULE, "clv_bar": 0.0, "clv_draws": int(draws)}), reset=True)
    d = D.day
    first = [t for times in D.poll_times.values() for t in times if t >= uk_time(str(d), "05:15").to_pydatetime()]
    out = []
    for arm_id, spec in arms.items():
        over, policy = dict(spec.get("over") or {}), dict(spec.get("policy") or {})
        cfg = ac.make_cfg(**{**RULE, "clv_draws": int(draws), **over})
        if policy.get("first_hour"):
            if not first:
                continue
            until = (min(first) + timedelta(minutes=int(policy["first_hour"]))).astimezone(UK)
            cfg.trade_until = f"{until.hour:02d}:{until.minute:02d}"
        if policy.get("forecast"):
            policy["forecast"] = dict(forecast or {})
        policy["_D"] = D
        h, m = (int(x) for x in cfg.trade_until.split(":"))
        t_end = (datetime(d.year, d.month, d.day, h, m, tzinfo=UK) + timedelta(minutes=3)).astimezone(timezone.utc)
        kind = spec.get("kind", "rule")
        led, s, _ = ac.run_session(D, lay_cls if kind == "lay" else rule_cls, cfg, policy,
                                   ac.session_start(D, "win"), t_end=t_end)
        if kind == "lay":
            pos = s.lay_positions_frame()
            b = _orders_to_bets(pos, D, "lay")
        else:
            b = led[(led["event"] == "back") & (pd.to_numeric(led["matched"], errors="coerce") > 0)] \
                if len(led) and "event" in led.columns else pd.DataFrame()
            b = _orders_to_bets(b, D, "back")
        out.append(b.assign(id=arm_id))
    ac.set_plan_cfg(ac.make_cfg(), reset=True)
    out = [b for b in out if len(b)]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=BETS_COLS + ["id"])


# ---------------------------------------------------------------------------------------------------------------
# one day's pass over the raw books
# ---------------------------------------------------------------------------------------------------------------

def win_raw(raw: pd.DataFrame, runners_day: pd.DataFrame) -> pd.DataFrame:
    """The day's raw rows of its GB/IE WIN markets, with the market's start (off) and minutes to it (mto)."""
    w = runners_day.loc[runners_day["market_type"].eq("WIN"), ["market_id", "market_start_utc"]].drop_duplicates(
        "market_id")
    x = raw.merge(w.assign(off=_utc(w["market_start_utc"]))[["market_id", "off"]], on="market_id", how="inner")
    x["mto"] = (x["off"] - x["t"]).dt.total_seconds() / 60.0
    return x


def make_ctx(day: str, raw: pd.DataFrame, runners_day: pd.DataFrame, fills_day: pd.DataFrame,
             read=None) -> SimpleNamespace:
    """What a strategy's day function is given: the day, its raw WIN books (win_raw), its runners (the loader's
    frame for the day), the live trader's fills of the day, the day's removals, and ``read(name)`` for another raw
    file of the day (typed as read_raw; 'markets_...' files untyped)."""
    return SimpleNamespace(day=day, raw=win_raw(raw, runners_day), runners=runners_day, fills=fills_day,
                           removals=removals_of(raw), read=read or (lambda name: pd.DataFrame()))


def _read_other(day: str):
    def read(name: str) -> pd.DataFrame:
        if name.startswith("markets"):
            return _csv(f"betfair_live/{day}/{name}.csv.gz", dtype={"market_id": str})
        return read_raw(day, names=(name,))
    return read


def day_pass(job: dict) -> dict:
    """One day: the raw books read once, the removals and places kept, every strategy's own day function run on them
    (each inside try/except: an error is recorded against that strategy and day), then the replays. Returns
    {'day', 'frames': {name: frame}, 'errors': {key: message}, 'raw': bool, 'replayed': bool, 'seconds'}."""
    from research.strat100 import strategies_b_live as S
    day, t0 = job["day"], time.time()
    out = {"day": day, "frames": {}, "errors": {}, "raw": False, "replayed": False}
    runners_day, fills_day = job["runners"], job["fills"]
    with _memo_s3():
        try:
            raw = read_raw(day) if job.get("raw", True) else pd.DataFrame()
        except Exception as exc:  # noqa: BLE001
            raw, out["errors"]["raw"] = pd.DataFrame(), f"{type(exc).__name__}: {exc}"
        if len(raw):
            out["raw"] = True
            out["frames"]["removals"] = removals_of(raw).assign(race_date=day)
            out["frames"]["places"] = places_of(raw).assign(race_date=day)
            ctx = make_ctx(day, raw, runners_day, fills_day, read=_read_other(day))
            for key, fn in S.DAY_FUNCS.items():
                try:
                    res = fn(ctx)
                    if res is not None and len(res):
                        out["frames"][f"day_{key}"] = res.assign(race_date=day)
                except Exception as exc:  # noqa: BLE001 -- recorded against the strategy; the day goes on
                    out["errors"][key] = f"{type(exc).__name__}: {exc} | {traceback.format_exc(limit=3)}"
            del ctx
        del raw
        if job.get("replay", True):
            try:
                D = load_replay_day(day)
                if D is not None:
                    fc = runners_day[runners_day["market_type"].eq("WIN") & runners_day["F"].gt(1.0)]
                    forecast = {(str(m), int(s)): float(f) for m, s, f in zip(fc["market_id"], fc["selection_id"], fc["F"])}
                    out["frames"]["replay"] = replay_day(D, S.REPLAY_ARMS, job.get("draws", 20000), forecast)
                    out["replayed"] = True
            except Exception as exc:  # noqa: BLE001
                out["errors"]["replay"] = f"{type(exc).__name__}: {exc} | {traceback.format_exc(limit=4)}"
    out["seconds"] = round(time.time() - t0, 1)
    return out


def _code_hash() -> str:
    h = hashlib.sha1()
    for f in ("data_b.py", "strategies_b_live.py"):
        p = Path(__file__).with_name(f)
        h.update(p.read_bytes() if p.exists() else b"")
    return h.hexdigest()[:10]


def _fingerprint(job: dict) -> str:
    """A hash of what a day's pass is given (its runners and fills): a day whose marks, results or prices have
    changed in the database is made again, not read from the cache."""
    h = hashlib.sha1()
    for name in ("runners", "fills"):
        f = job.get(name)
        if f is not None and len(f):
            h.update(pd.util.hash_pandas_object(f.astype(str), index=False).to_numpy().tobytes())
    return h.hexdigest()[:10]


def run_day_passes(jobs: list[dict], n_jobs: int = 1, work: Path | None = None, max_minutes: float = 240.0,
                   refresh: bool = False) -> list[dict]:
    """Every day's pass, cached a day a file under ``work`` (named by the day, the options, a hash of the day's
    inputs and a hash of this code, so a rerun reads it and a change of the code or the data does not), ``n_jobs``
    days at a time. A day not done within ``max_minutes`` is left out and said so."""
    t0, done, todo, tag = time.time(), {}, [], _code_hash()
    for j in jobs:
        opts = (f"r{int(j.get('raw', True))}p{int(j.get('replay', True))}d{j.get('draws', 20000)}"
                f"c{CUT_MIN_AF_BACK:g}-{CUT_MIN_AF_LAY:g}")
        j["_cache"] = (work / f"day_{j['day']}_{opts}_{_fingerprint(j)}_{tag}.pkl") if work is not None else None
        if j["_cache"] is not None and j["_cache"].exists() and not refresh:
            try:
                done[j["day"]] = pd.read_pickle(j["_cache"])
                continue
            except Exception:  # noqa: BLE001 -- a broken cache file is made again
                pass
        todo.append(j)
    if done:
        _say(f"   the raw pass: {len(done)} days read from {work}")

    def keep(res: dict, cache) -> None:
        done[res["day"]] = res
        _say(f"   {res['day']}: raw {'yes' if res['raw'] else 'NO DATA FOUND'}, replay "
             f"{'yes' if res['replayed'] else 'no'}, {res['seconds']} s"
             + ("; ERRORS " + "; ".join(f"{k}: {v.split(' | ')[0]}" for k, v in res["errors"].items())
                if res["errors"] else ""))
        if cache is not None and not res["errors"]:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                pd.to_pickle(res, cache)
            except Exception as exc:  # noqa: BLE001 -- the cache is a convenience
                _say(f"   {cache}: not written ({type(exc).__name__})")

    def left() -> float:
        return max_minutes * 60 - (time.time() - t0)

    if n_jobs <= 1 or len(todo) <= 1:
        for j in todo:
            if left() <= 0:
                break
            keep(day_pass({k: v for k, v in j.items() if k != "_cache"}), j["_cache"])
    else:
        from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
        pool = ProcessPoolExecutor(max_workers=n_jobs)
        try:
            futs = {pool.submit(day_pass, {k: v for k, v in j.items() if k != "_cache"}): j for j in todo}
            pending = set(futs)
            while pending and left() > 0:
                got, pending = wait(pending, timeout=min(60.0, max(1.0, left())), return_when=FIRST_COMPLETED)
                for f in got:
                    j = futs[f]
                    try:
                        keep(f.result(), j["_cache"])
                    except Exception as exc:  # noqa: BLE001 -- a day that died is a day not read
                        _say(f"   {j['day']}: the pass failed ({type(exc).__name__}: {exc})")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
    missing = [j["day"] for j in jobs if j["day"] not in done]
    if missing:
        _say(f"   THE RAW PASS DID NOT FINISH {len(missing)} DAYS (not read by any strategy that needs them): {missing}")
    return [done[d] for d in sorted(done)]


# ---------------------------------------------------------------------------------------------------------------
# the frames
# ---------------------------------------------------------------------------------------------------------------

def _type_cat(cat: pd.DataFrame) -> pd.DataFrame:
    n0 = len(cat)
    c = cat.reindex(columns=list(dict.fromkeys([*cat.columns, *CAT_COLS]))).copy()
    c["race_date"] = c["race_date"].astype(str).str[:10]
    c["market_id"] = c["market_id"].astype(str)
    c["market_type"], c["country"] = _up(c["market_type"]), _up(c["country"])
    c = _num(c, ["selection_id", "days_since_last_run", "forecast_num", "forecast_den", "race_results_id",
                 "official_rating"])
    for col in ("market_name", "market_start_utc", "venue", "event_id", "race_type", "runner_name", "wearing",
                "horse_norm"):
        c[col] = c[col].astype(object).where(c[col].notna(), "").astype(str)
    c["event_id"] = c["event_id"].str.replace(r"\.0$", "", regex=True)
    keep = c["country"].isin(["GB", "IE"]) & c["market_type"].isin(["WIN", "PLACE"]) & c["selection_id"].notna() \
        & c["race_date"].between(FIRST_DAY, LAST_DAY)
    dropped = (~keep).sum()
    c = c[keep].copy()
    c["selection_id"] = c["selection_id"].astype("int64")
    dup = int(c.duplicated(KEY).sum())
    c = c.drop_duplicates(KEY, keep="last").reset_index(drop=True)
    c["dslr"] = c["days_since_last_run"]
    with np.errstate(divide="ignore", invalid="ignore"):
        c["F"] = np.where((c["forecast_den"] > 0) & (c["forecast_num"] > 0), 1.0 + c["forecast_num"] / c["forecast_den"],
                          np.nan)
    _say(f"   catalogue: {n0:,} rows in; {int(dropped):,} dropped (not GB/IE WIN or PLACE, no selection, or outside "
         f"{FIRST_DAY}..{LAST_DAY}), {dup:,} duplicate runners dropped; {len(c):,} out "
         f"({int(c['market_type'].eq('WIN').sum()):,} WIN, {int(c['market_type'].eq('PLACE').sum()):,} PLACE; "
         f"forecast price on {c['F'].notna().mean():.1%})")
    return c.drop(columns=["days_since_last_run", "forecast_num", "forecast_den"])


def _type_marks(marks: pd.DataFrame) -> pd.DataFrame:
    m = marks.reindex(columns=list(dict.fromkeys([*marks.columns, *MARK_COLS]))).copy()
    m["market_id"] = m["market_id"].astype(str)
    m = _num(m, ["selection_id", "minutes_to_off", *LEVELS, "last_traded", "market_total_matched", "active_runners",
                 "bsp"])
    m = m[m["selection_id"].notna()].copy()
    m["selection_id"] = m["selection_id"].astype("int64")
    m["market_status"], m["runner_status"] = _up(m["market_status"]), _up(m["runner_status"])
    m["mark"] = m["mark"].astype(str)
    m["polled_utc"] = _utc(m["polled_utc"])
    for c in LEVELS[::2] + ["last_traded"]:               # a price at or under 1 is no price
        m[c] = m[c].where(m[c] > 1.0)
    return m


def pivot_marks(cat: pd.DataFrame, marks: pd.DataFrame) -> pd.DataFrame:
    """One row a runner a market with the book at each mark (the blocks of TAGS) and the final mark's bsp and
    final_status. ``cat`` and ``marks`` are typed (_type_cat, _type_marks)."""
    out, n0 = cat.copy(), len(marks)
    marks = marks.merge(cat[KEY], on=KEY, how="inner")
    _say(f"   marks: {n0:,} rows in; {n0 - len(marks):,} dropped (their runner is not in the GB/IE WIN/PLACE "
         f"catalogue); {len(marks):,} kept")
    unknown = sorted(set(marks["mark"].unique()) - set(TAGS) - {"final"})
    if unknown:
        _say(f"   marks not read (no tag): {unknown}")
    counts = {}
    for mark, tag in TAGS.items():
        m = marks[marks["mark"].eq(mark)].sort_values("polled_utc", kind="stable").drop_duplicates(KEY, keep="last")
        counts[tag] = len(m)
        blk = m[KEY + list(MARK_FIELDS)].rename(columns={k: f"{tag}_{v}" for k, v in MARK_FIELDS.items()})
        out = out.merge(blk, on=KEY, how="left", validate="1:1")
    fin = marks[marks["mark"].eq("final")].sort_values("polled_utc", kind="stable").drop_duplicates(KEY, keep="last")
    fin = fin[KEY + ["bsp", "runner_status"]].rename(columns={"runner_status": "final_status"})
    out = out.merge(fin, on=KEY, how="left", validate="1:1")
    out["bsp"] = out["bsp"].where(out["bsp"] > 1.0)
    out["final_status"] = out["final_status"].astype(object).where(out["final_status"].notna(), "").astype(str)
    _say("   marks pivoted, runners a mark: " + ", ".join(f"{t} {n:,}" for t, n in counts.items())
         + f"; final {len(fin):,} (with a BSP {int((fin['bsp'] > 1).sum()):,})")
    return out


def _fill_bsp(runners: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """The BSP from Betfair's price files (betfair_prices) where the final mark has none, on (race_date, selection,
    market type). A horse runs once a day, so the key names one runner; a key that does not is not used."""
    r = runners.copy()
    r["bsp_src"] = np.where(r["bsp"].gt(1.0), "final mark", "")
    if prices is None or prices.empty:
        return r
    p = prices[prices["bsp"].gt(1.0)].assign(market_type=prices["market_type"].astype(str).str.upper())
    p = p.drop_duplicates(["race_date", "selection_id", "market_type"], keep=False)
    got = r[["race_date", "selection_id", "market_type"]].merge(
        p[["race_date", "selection_id", "market_type", "bsp"]].rename(columns={"bsp": "bsp_file"}),
        on=["race_date", "selection_id", "market_type"], how="left")["bsp_file"].to_numpy()
    need = ~r["bsp"].gt(1.0).to_numpy() & np.isfinite(got) & ~r["final_status"].eq("REMOVED").to_numpy()
    r.loc[need, "bsp"] = got[need]
    r.loc[need, "bsp_src"] = "betfair_prices"
    _say(f"   BSP: {int(r['bsp_src'].eq('final mark').sum()):,} from the final mark, {int(need.sum()):,} from "
         f"betfair_prices, {int((~r['bsp'].gt(1.0)).sum()):,} runners with none (void wherever a bet needs it)")
    return r


def attach_predictions(runners: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """predicted_bfsp on the WIN rows: by Betfair's ids where the file carries them, else by day, course, UK off time
    and horse (the keys research/queries/done/small_book_late_1111.py joins on), then by day, time and horse."""
    import betfair_prices as bp
    r = runners.copy()
    r["predicted_bfsp"] = np.nan
    win = r["market_type"].eq("WIN").to_numpy()
    if preds is None or preds.empty:
        _say("   predictions: no data found (every strategy that needs the model's price returns no bet)")
        return r
    p = preds.copy()
    by_id = p[p["market_id"].notna() & p["selection_id"].notna()].copy()
    if len(by_id):
        by_id["market_id"] = by_id["market_id"].astype(str)
        by_id["selection_id"] = by_id["selection_id"].astype("int64")
        by_id = by_id.drop_duplicates(["race_date"] + KEY, keep="last")
        got = r[["race_date"] + KEY].merge(by_id[["race_date"] + KEY + ["predicted_bfsp"]], on=["race_date"] + KEY,
                                           how="left")["predicted_bfsp"].to_numpy()
        r["predicted_bfsp"] = np.where(win, got, np.nan)
    n_id = int(np.isfinite(r["predicted_bfsp"]).sum())
    r["k_track"] = r["venue"].map(bp.normalise_track)
    r["k_time"] = r["market_start_utc"].map(bp.utc_iso_to_uk_hhmm)
    n_names = 0
    for keys in (["race_date", "k_track", "k_time", "k_horse"], ["race_date", "k_time", "k_horse"]):
        need = win & ~np.isfinite(r["predicted_bfsp"].to_numpy(float))
        if not need.any():
            break
        one = p.drop_duplicates(keys, keep=False)
        left = r.loc[need, ["race_date", "k_track", "k_time"]].assign(k_horse=r.loc[need, "horse_norm"])
        got = left.merge(one[keys + ["predicted_bfsp"]], on=keys, how="left")["predicted_bfsp"].to_numpy()
        r.loc[need, "predicted_bfsp"] = got
        n_names += int(np.isfinite(got).sum())
    r = r.drop(columns=["k_track", "k_time"])
    r["predicted_bfsp"] = r["predicted_bfsp"].where(r["predicted_bfsp"] > 1.0)
    n_win = int(win.sum())
    _say(f"   predictions joined: {n_id:,} WIN runners by Betfair's ids, {n_names:,} by names; "
         f"{n_win - n_id - n_names:,} of {n_win:,} WIN runners have no model price")
    return r


def build_fills(orders: pd.DataFrame, removals: pd.DataFrame | None, days_books: list[str]) -> tuple:
    """BASE-B (K7) from live_orders: (fills, settles).

    fills: the 'back' rows with matched > 0, one row an order, in time order; ``first`` marks a horse's first order
    (its back rows within 60 s of its earliest), ``n`` the order's number on the horse; the horse's settle row
    joined (bsp, result, settled_price = Betfair's settled price, after its reductions). ``cut_factor`` is the
    real money's cut: where the horse's settle row carries Betfair's settled price, the raw books' removals after
    each fill (factors of 2.5 or more, Betfair's rule) share the cut among the horse's fills, scaled so their
    stake-weighted cut equals Betfair's own (settled / matched); on a day with no raw books that ratio is every
    fill's cut; with no settled price (a void horse) it is K6 CUT from the raw books (``cut_raw``, every factor).
    settles: one row a settled horse (matched, settled_price, bsp, result)."""
    cols_f = ["race_date", "market_id", "selection_id", "ts", "seq", "n", "first", "matched", "avg_price", "edge",
              "best_back", "minutes_to_off", "p_model", "p_market", "bsp", "result", "settled_price",
              "matched_price", "ratio", "cut_raw", "cut_factor", "cut_src"]
    cols_s = ["race_date", "market_id", "selection_id", "matched", "settled_price", "bsp", "result", "pnl"]
    if orders is None or orders.empty:
        return pd.DataFrame(columns=cols_f), pd.DataFrame(columns=cols_s)
    o = orders.reindex(columns=list(dict.fromkeys([*orders.columns, *ORDER_COLS])))
    o = _num(o, ORDER_NUM)
    n0 = len(o)
    o = o[o["mode"].astype(str).str.lower().eq("live") & o["selection_id"].notna()
          & o["race_date"].astype(str).str[:10].between(FIRST_DAY, LAST_DAY)].copy()
    o["race_date"] = o["race_date"].astype(str).str[:10]
    o["market_id"] = o["market_id"].astype(str)
    o["selection_id"] = o["selection_id"].astype("int64")
    o["event"] = o["event"].astype(str)
    hk = ["race_date"] + KEY
    s = o[o["event"].eq("settle") & o["matched"].gt(0)].sort_values("seq", kind="stable").drop_duplicates(hk, keep="last")
    settles = pd.DataFrame({"race_date": s["race_date"], "market_id": s["market_id"],
                            "selection_id": s["selection_id"], "matched": s["matched"],
                            "settled_price": s["avg_price"], "bsp": s["bsp"].where(s["bsp"] > 1.0),
                            "result": _up(s["result"]), "pnl": s["pnl"]}).reset_index(drop=True)
    b = o[o["event"].eq("back") & o["matched"].gt(0) & o["avg_price"].gt(1.0)].copy()
    b["ts"] = _utc(b["ts"])
    b = b[b["ts"].notna()].sort_values(["ts", "seq"], kind="stable").reset_index(drop=True)
    b["n"] = b.groupby(hk).cumcount()
    b["first"] = (b["ts"] - b.groupby(hk)["ts"].transform("min")).dt.total_seconds() <= 60.0
    cost = (b["matched"] * b["avg_price"]).groupby([b[c] for c in hk]).transform("sum")
    b["matched_price"] = cost / b.groupby(hk)["matched"].transform("sum")
    b = b.drop(columns=["bsp", "result"]).merge(settles[hk + ["bsp", "result", "settled_price"]], on=hk, how="left")
    b["ratio"] = b["settled_price"] / b["matched_price"]
    have = b["race_date"].isin(set(days_books)).to_numpy()
    raw_cut = cut_factor(removals, b["market_id"], b["selection_id"], b["ts"], CUT_MIN_AF_BACK)
    b["cut_raw"] = np.where(have, raw_cut, np.nan)
    # Betfair's own record of the horse's cut, where it is sane (the settle row rounds to 4 places)
    ratio = b["ratio"].where(b["ratio"].between(0.30, 1.0005)).clip(upper=1.0)
    ratio = ratio.mask(ratio >= 0.9995, 1.0)
    base = pd.Series(np.where(have, raw_cut, 1.0), index=b.index)
    # the real cut is shared among a horse's fills as Betfair's own rule spreads it (factors of 2.5 or more, by the
    # time of each fill), then scaled so the horse's stake-weighted cut is exactly Betfair's settled / matched
    share = pd.Series(np.where(have, cut_factor(removals, b["market_id"], b["selection_id"], b["ts"], BETFAIR_MIN_AF),
                               1.0), index=b.index)
    w = b["matched"] * b["avg_price"]
    horse = (w * share).groupby([b[c] for c in hk]).transform("sum") / w.groupby([b[c] for c in hk]).transform("sum")
    scaled = (share * ratio / horse).clip(upper=1.0)
    b["cut_factor"] = np.where(ratio.notna(), scaled, base).clip(1e-6, 1.0)
    b["cut_src"] = np.where(ratio.notna(), np.where(have, "raw books, scaled to Betfair's settled price",
                                                    "Betfair's settled price (no raw books)"),
                            np.where(have, "raw books", "none (no raw books, no settled price)"))
    _say(f"   live_orders: {n0:,} rows in; {len(b):,} matched backs on {b.groupby(hk).ngroups:,} horses over "
         f"{b['race_date'].nunique()} days (first orders {int(b['first'].sum()):,}, top-ups {int((~b['first']).sum()):,}); "
         f"{len(settles):,} settle rows; backs with no settle row {int(b['result'].isna().sum()):,} (void)")
    if len(b):
        ok = b[b["ratio"].notna() & have]
        if len(ok):
            wk = ok["matched"] * ok["avg_price"]
            def raw_at(floor):
                return (wk * cut_factor(removals, ok["market_id"], ok["selection_id"], ok["ts"], floor)).sum() / wk.sum()
            _say(f"   the cut, reconciled to Betfair's settled prices on days with raw books: settled/matched "
                 f"{(wk * ok['ratio'].clip(upper=1.0)).sum() / wk.sum():.5f} (used for BASE-B); raw books, every "
                 f"factor {raw_at(0.0):.5f}; raw books at Betfair's {BETFAIR_MIN_AF:g}% floor {raw_at(BETFAIR_MIN_AF):.5f}")
    return b[cols_f], settles[cols_s]


def build_hist(runners: pd.DataFrame, rr: pd.DataFrame | None) -> pd.DataFrame:
    """S047: for each WIN card runner, from race_results rows dated BEFORE its race day and matched on the horse's
    normalised name: prev_n (earlier runs), prev_type (the last earlier run's race_type), prev_hcap (any earlier
    handicap, by the race name or type), prev_headgear (any earlier run in headgear)."""
    cols = ["race_date", "market_id", "selection_id", "prev_n", "prev_type", "prev_hcap", "prev_headgear"]
    if rr is None or rr.empty:
        return pd.DataFrame(columns=cols)
    import betfair_prices as bp
    w = runners.loc[runners["market_type"].eq("WIN"), ["race_date"] + KEY + ["horse_norm"]]
    h = rr.reindex(columns=RR_COLS).copy()
    names = pd.Series(h["horse_name"].dropna().unique())
    h["horse_norm"] = h["horse_name"].map(dict(zip(names, names.map(bp.normalise_horse))))
    h = h[h["horse_norm"].isin(set(w["horse_norm"]))].copy()
    if h.empty:
        return pd.DataFrame(columns=cols)
    h["d"] = pd.to_datetime(h["race_date"].astype(str).str[:10], errors="coerce")
    h = h[h["d"].notna()].sort_values(["horse_norm", "d"], kind="stable")
    text = (h["race_name"].fillna("").astype(str) + " " + h["race_type"].fillna("").astype(str)).str.lower()
    h["hc"] = text.str.contains(r"handicap|hcap|h'cap|nursery").astype(int)
    h["hg"] = h["headgear"].fillna("").astype(str).str.strip().ne("").astype(int)
    g = h.groupby("horse_norm", sort=False)
    h["prev_n"] = g.cumcount() + 1
    h["prev_hcap"], h["prev_headgear"] = g["hc"].cummax(), g["hg"].cummax()
    h["prev_type"] = h["race_type"].fillna("").astype(str)
    left = w.assign(d=pd.to_datetime(w["race_date"]).astype("datetime64[ns]")).sort_values("d", kind="stable")
    right = h[["horse_norm", "d", "prev_n", "prev_type", "prev_hcap", "prev_headgear"]].assign(
        d=h["d"].astype("datetime64[ns]")).sort_values("d", kind="stable")
    m = pd.merge_asof(left, right, on="d", by="horse_norm", allow_exact_matches=False, direction="backward")
    m["prev_n"] = m["prev_n"].fillna(0).astype(int)
    m["prev_hcap"], m["prev_headgear"] = m["prev_hcap"].fillna(0).astype(int), m["prev_headgear"].fillna(0).astype(int)
    m["prev_type"] = m["prev_type"].fillna("")
    _say(f"   race_results history: {len(h):,} earlier runs of the cards' horses; {int((m['prev_n'] > 0).sum()):,} of "
         f"{len(m):,} WIN runners have an earlier run; the table ends {str(rr['race_date'].max())[:10]}")
    return m[cols].reset_index(drop=True)


def attach_steam(runners: pd.DataFrame, steam: pd.DataFrame | None) -> pd.DataFrame:
    """The steamer's score for each card runner as rows (race_date, market_id, selection_id, steam_score,
    steam_field), joined on the day, course, UK off time and the horse's normalised name."""
    cols = ["race_date", "market_id", "selection_id", "steam_score", "steam_field"]
    if steam is None or steam.empty:
        return pd.DataFrame(columns=cols)
    import betfair_prices as bp
    s = steam.copy()
    s["k_track"], s["k_horse"] = s["track"].map(bp.normalise_track), s["horse_name"].map(bp.normalise_horse)
    s["k_time"] = s["race_time"].astype(str).map(bp.db_time_to_24h)
    s = s.drop_duplicates(["race_date", "k_track", "k_time", "k_horse"], keep=False)
    w = runners.loc[runners["market_type"].eq("WIN"), ["race_date"] + KEY + ["venue", "market_start_utc", "horse_norm"]]
    w = w.assign(k_track=w["venue"].map(bp.normalise_track), k_time=w["market_start_utc"].map(bp.utc_iso_to_uk_hhmm),
                 k_horse=w["horse_norm"])
    m = w.merge(s[["race_date", "k_track", "k_time", "k_horse", "steam_score", "steam_field"]],
                on=["race_date", "k_track", "k_time", "k_horse"], how="inner")
    _say(f"   steamer scores: {len(s):,} card runners scored, {len(m):,} joined to a Betfair WIN runner "
         f"({len(s) - len(m):,} not joined)")
    return m[cols].reset_index(drop=True)


def assemble(cat: pd.DataFrame, marks: pd.DataFrame, orders: pd.DataFrame | None = None,
             prices: pd.DataFrame | None = None, preds: pd.DataFrame | None = None, rr: pd.DataFrame | None = None,
             day_results: list[dict] | None = None, steam: pd.DataFrame | None = None,
             meta: dict | None = None) -> dict:
    """The loader's frames from its sources already read (the tables as the database gives them, the predictions as
    read_predictions gives them, the day passes' results, the steamer's scores). Pure: tests feed it toy tables."""
    meta = dict(meta or {})
    meta.setdefault("problems", [])
    meta["delayed_key"] = DELAYED
    cat_t = _type_cat(cat)
    runners = pivot_marks(cat_t, _type_marks(marks))
    pr = pd.DataFrame(columns=PRICE_COLS) if prices is None else _num(
        prices.reindex(columns=PRICE_COLS), ["selection_id", "win_lose", "bsp", "ppwap", "ipmax", "ipmin", "pp_vol"])
    pr = pr[pr["selection_id"].notna()].copy()
    pr["selection_id"] = pr["selection_id"].astype("int64")
    pr["race_date"] = pr["race_date"].astype(str).str[:10]
    pr["market_type"] = pr["market_type"].astype(str).str.lower()
    runners = _fill_bsp(runners, pr)
    runners = attach_predictions(runners, preds)

    day_results = day_results or []
    frames_by: dict[str, list] = {}
    errors: dict[str, dict] = {}
    for res in day_results:
        for name, f in res["frames"].items():
            frames_by.setdefault(name, []).append(f)
        for key, msg in res["errors"].items():
            errors.setdefault(key, {})[res["day"]] = msg
    cat_days = sorted(runners.loc[runners["market_type"].eq("WIN") & runners[[f"{t}_t" for t in TAGS.values()]]
                                  .notna().any(axis=1), "race_date"].unique())
    days_books = sorted(r["day"] for r in day_results if r.get("raw"))
    days_replay = sorted(r["day"] for r in day_results if r.get("replayed"))

    def cat_frames(name: str, cols: list[str]) -> pd.DataFrame:
        got = [f for f in frames_by.get(name, []) if len(f)]
        return pd.concat(got, ignore_index=True) if got else pd.DataFrame(columns=cols)

    removals = cat_frames("removals", ["market_id", "selection_id", "af", "rt", "first_seen", "race_date"])
    places = cat_frames("places", ["market_id", "places_raw", "race_date"])
    # K9: the places a market pays: the raw book's number_of_winners, else the count of WINNER on its final mark
    n_win = runners["final_status"].eq("WINNER").groupby(runners["market_id"]).transform("sum")
    praw = runners[["market_id"]].merge(places.drop_duplicates("market_id")[["market_id", "places_raw"]],
                                        on="market_id", how="left")["places_raw"].to_numpy(float)
    runners["places"] = np.where(np.isfinite(praw) & (praw > 0), praw, n_win.where(n_win > 0).to_numpy(float))
    runners["places_src"] = np.where(np.isfinite(praw) & (praw > 0), "raw book", "final mark")

    fills, settles = build_fills(orders, removals, days_books)
    frames = {"runners": runners, "fills": fills, "settles": settles, "prices": pr.reset_index(drop=True),
              "removals": removals, "hist": build_hist(runners, rr), "steam": attach_steam(runners, steam),
              "replay": cat_frames("replay", BETS_COLS + ["id"])}
    for name in sorted(frames_by):
        if name.startswith("day_"):
            frames[name] = cat_frames(name, [])
    meta.update({"days_books": days_books, "days_replay": days_replay,
                 "days_preds": sorted(runners.loc[runners["predicted_bfsp"].gt(1.0), "race_date"].unique()),
                 "days_fills": sorted(fills["race_date"].unique()) if len(fills) else [],
                 "days_steam": sorted(frames["steam"]["race_date"].unique()) if len(frames["steam"]) else [],
                 "day_errors": errors, "cut_min_af": {"back": CUT_MIN_AF_BACK, "lay": CUT_MIN_AF_LAY}})
    frames["days"] = cat_days
    frames["meta"] = meta
    _say(f"   b_live: {len(cat_days)} race days {cat_days[0] if cat_days else '-'} .. {cat_days[-1] if cat_days else '-'}; "
         f"raw books on {len(days_books)}, replayed {len(days_replay)}, predictions on {len(meta['days_preds'])}, "
         f"real fills on {len(meta['days_fills'])}; removals {len(removals):,}")
    for key, by_day in errors.items():
        _say(f"   DAY ERRORS {key}: {len(by_day)} days, e.g. {next(iter(by_day.items()))}")
    if len(frames["replay"]):
        rp = frames["replay"]
        base = rp[rp["id"].eq("S001") & ~rp["void"].astype(bool)]
        if len(base):
            def clv(col):
                p = np.maximum(1.01, base["entry_price"] * base[col])
                return float((base["stake"] * (p / base["exit_price"] - 1.0)).sum())
            _say(f"   REPLAY-B (S001) CLV x stake before commission: GBP{clv('cut_factor'):,.2f} as scored (factors "
                 f"of {CUT_MIN_AF_BACK:g} or more: K6, research/arms/core.reduced), GBP{clv('cut_factor_bf'):,.2f} at "
                 f"Betfair's {BETFAIR_MIN_AF:g}% floor, on GBP{base['stake'].sum():,.2f}")
    return frames


# ---------------------------------------------------------------------------------------------------------------
# the steamer (S029's vote, S031)
# ---------------------------------------------------------------------------------------------------------------

def steam_scores(db: str, days: list[str], cutoff: str = STEAM_CUTOFF) -> tuple[pd.DataFrame, dict]:
    """The CLEAN steamer (research/paper/morning_features.train_steamer) fitted once on 2018-01-01..``cutoff`` (no
    row after the cutoff reaches the fit), then each day's first racecard (S3 racecards/<day>_<hhmm>.csv, the
    earliest of the day and never one written after 06:55 UTC) scored on the features built from EARLIER days'
    results (morning/select.history_with_card, as the live morning models build them). Returns the scores
    (race_date, race_time, track, horse_name, steam_score, steam_field) and the fit's facts (the training scores'
    80th and 95th percentiles).

    The history given to each day's build is cut to the rows of the card's horses, trainers and jockeys: every
    feature of a card runner is the horse's own earlier run or its trainer's or jockey's earlier record, so the
    cut changes no feature of the card's rows (it only saves the build of two million others)."""
    from morning import select as MS
    from research.paper import morning_features as MF
    L = MF.L
    hist = L.load(db)
    hist["race_date"] = hist["race_date"].astype(str).str[:10]
    fit = hist[hist["race_date"] <= cutoff]
    booster, thr95, n_fit = MF.train_steamer(fit, cutoff=cutoff)
    x = L.priced(MF.build(fit))
    x = x[(x["race_date"] >= "2018-01-01") & (x["race_date"] <= cutoff) & x["steam"].notna()]
    train = booster.predict(x[MF.CLEAN])
    info = {"steam_cutoff": cutoff, "steam_rows_fit": int(n_fit), "steam_p80": float(np.quantile(train, 0.80)),
            "steam_p95": float(np.quantile(train, 0.95)), "steam_threshold_check": float(thr95), "steam_cards": {}}
    del x, train, fit
    _say(f"   steamer: CLEAN model fitted on 2018-01-01..{cutoff}, {n_fit:,} rows; training scores' 80th percentile "
         f"{info['steam_p80']:.4f}, 95th {info['steam_p95']:.4f}")
    out = []
    for day in days:
        keys = [k for k in _ls(f"racecards/{day}_") if k.endswith(".csv")]
        early = [k for k in keys if k[-8:-4].isdigit() and k[-8:-4] <= "0655"]
        if not early:
            info["steam_cards"][day] = "no data found: no racecard written by 06:55 UTC"
            continue
        card = _csv(early[0], dtype={"race_time": str})
        try:
            jockeys = {MS.split_claim(j)[0] for j in card["jockey_name"]}
            sub = hist[(hist["race_date"] < day) & (hist["horse_name"].isin(set(card["horse_name"]))
                                                    | hist["trainer"].isin(set(card["trainer"]))
                                                    | hist["jockey_name"].isin(jockeys))]
            f = MF.build(MS.history_with_card(sub, card, day))
            f = f[f["race_date"] == day]
            if f.empty:
                info["steam_cards"][day] = "no data found: no runner of the main codes on the card"
                continue
            out.append(pd.DataFrame({"race_date": day, "race_time": f["race_time"].astype(str), "track": f["track"],
                                     "horse_name": f["horse_name"], "steam_score": booster.predict(f[MF.CLEAN]),
                                     "steam_field": pd.to_numeric(f["field"], errors="coerce")}))
            info["steam_cards"][day] = early[0]
        except Exception as exc:  # noqa: BLE001 -- a card that cannot be scored is a day without the steamer
            info["steam_cards"][day] = f"no data found: {type(exc).__name__}: {exc}"
    scored = pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["race_date", "race_time", "track", "horse_name", "steam_score", "steam_field"])
    _say(f"   steamer: {len(scored):,} card runners scored on {scored['race_date'].nunique()} of {len(days)} days")
    return scored, info


# ---------------------------------------------------------------------------------------------------------------
# the loader
# ---------------------------------------------------------------------------------------------------------------

def _db_path() -> str:
    p = Path(os.environ.get("STRAT100_DB") or "horse_racing.db")
    if not p.exists() or p.stat().st_size == 0:
        raise NoData(f"no data found: the database {p} is not here (run scripts/setup_session.sh --db on the runner)")
    return str(p)


def _table(conn: sqlite3.Connection, table: str, cols: list[str], where: str = "", params: tuple = ()) -> pd.DataFrame:
    """The table's listed columns (those it has), read-only; a missing table is 'no data found', said plainly."""
    try:
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    except Exception as exc:  # noqa: BLE001
        raise NoData(f"no data found: {table} could not be read ({type(exc).__name__}: {exc})") from exc
    if not have:
        raise NoData(f"no data found: the database has no table {table}")
    lacking = [c for c in cols if c not in have]
    if lacking:
        _say(f"   {table}: columns not in the table, read as empty: {lacking}")
    sel = ", ".join(c for c in cols if c in have)
    out = pd.read_sql_query(f"SELECT {sel} FROM {table} {where}", conn, params=params)
    _say(f"   {table}: {len(out):,} rows read")
    return out.reindex(columns=cols)


def load_b_live() -> dict:
    """The b_live frames (see the module's text). A source that is missing is said ('no data found') and the
    strategies that need it return no bet or an error; nothing is invented in its place."""
    t0 = time.time()
    days = parse_days()
    if not days:
        raise NoData("no data found: STRAT100_B_DAYS names no day of the b_live window")
    d0, d1 = days[0], days[-1]
    _say(f"== b_live: {d0} .. {d1} ({len(days)} calendar days). {DELAYED}")
    db = _db_path()
    conn = sqlite3.connect(f"file:{Path(db).resolve()}?mode=ro", uri=True)
    try:
        cat = _table(conn, "betfair_live_markets", CAT_COLS, "WHERE race_date >= ? AND race_date <= ?", (d0, d1))
        marks = _table(conn, "betfair_live_marks", MARK_COLS, "WHERE race_date >= ? AND race_date <= ?", (d0, d1))
        if cat.empty or marks.empty:
            raise NoData("no data found: betfair_live_markets or betfair_live_marks holds no row of the window")
        try:
            orders = _table(conn, "live_orders", ORDER_COLS,
                            "WHERE race_date >= ? AND race_date <= ? AND event IN ('back', 'settle')", (d0, d1))
        except NoData as exc:
            _say(f"   {exc}")
            orders = None
        try:
            prices = _table(conn, "betfair_prices", PRICE_COLS, "WHERE race_date >= ? AND race_date <= ?", (d0, d1))
        except NoData as exc:
            _say(f"   {exc}")
            prices = None
        try:
            rr = _table(conn, "race_results", RR_COLS, "WHERE race_date < ?", (d1,))
        except NoData as exc:
            _say(f"   {exc}")
            rr = None
    finally:
        conn.close()
    meta = {"problems": [], "db": db, "days_asked": days}
    preds = read_predictions(days)
    if preds.empty:
        meta["problems"].append("no data found: S3 predictions/<day>.csv (no model price)")

    steam = None
    if os.environ.get("STRAT100_B_STEAMER", "1") == "0":
        meta["problems"].append("no data found: the steamer's scores (STRAT100_B_STEAMER=0)")
    else:
        try:
            steam, info = steam_scores(db, days)
            meta.update(info)
        except Exception as exc:  # noqa: BLE001 -- S031 then reports it; the rest of the data set stands
            meta["problems"].append(f"no data found: the steamer's scores ({type(exc).__name__}: {exc})")
            _say(f"   steamer NOT BUILT: {type(exc).__name__}: {exc}")

    # a first assembly without the raw pass gives each day's runners and fills to its pass
    _QUIET[0] = True
    try:
        base = assemble(cat, marks, orders, prices, preds, None, None, None, {})
    finally:
        _QUIET[0] = False
    runners, fills = base["runners"], base["fills"]
    do_raw = os.environ.get("STRAT100_B_RAW", "1") != "0"
    do_replay = do_raw and os.environ.get("STRAT100_B_REPLAY", "1") != "0"
    results: list[dict] = []
    if not do_raw:
        meta["problems"].append("no data found: the raw books (STRAT100_B_RAW=0): no CUT, no raw poll, no replay")
    else:
        draws = int(os.environ.get("STRAT100_B_DRAWS") or 20000)
        by_day_r = {d: g for d, g in runners.groupby("race_date")}
        by_day_f = {d: g for d, g in fills.groupby("race_date")} if len(fills) else {}
        jobs = [{"day": d, "runners": by_day_r[d].reset_index(drop=True),
                 "fills": by_day_f.get(d, fills.iloc[:0]).reset_index(drop=True), "raw": True, "replay": do_replay,
                 "draws": draws} for d in base["days"] if d in by_day_r]
        n_jobs = int(os.environ.get("STRAT100_B_JOBS") or min(4, os.cpu_count() or 1))
        work = Path(os.environ.get("STRAT100_WORK") or "out/strat100") / "b_live"
        _say(f"   the raw pass: {len(jobs)} days, {n_jobs} at a time, replays {'on' if do_replay else 'OFF'} "
             f"({draws:,} draws a race); cache {work}")
        results = run_day_passes(jobs, n_jobs, work, float(os.environ.get("STRAT100_B_MAX_MINUTES") or 240.0),
                                 os.environ.get("STRAT100_B_REFRESH", "0") == "1")
        if not any(r.get("raw") for r in results):
            meta["problems"].append("no data found: S3 betfair_live/<day>/ raw books (no CUT, no raw poll, no replay)")
    del base
    frames = assemble(cat, marks, orders, prices, preds, rr, results, steam, meta)
    for p in frames["meta"]["problems"]:
        _say(f"   PROBLEM: {p}")
    _say(f"== b_live loaded in {(time.time() - t0) / 60:.1f} min")
    return frames
