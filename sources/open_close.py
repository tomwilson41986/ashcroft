"""One standard open-to-close panel across markets (the data agent, 1 Nov 2026; the owner's ask of 1 Nov: trade the
move from the opening price to the close, in whichever markets carry one, without relying on the result).

One row a selection a market, the same columns whatever the source (``SCHEMA``):

| column | meaning |
|---|---|
| sport, country, market_type | Betfair's names where the source has them (WIN, PLACE, MATCH_ODDS, OVER_UNDER_25, ASIAN_HANDICAP) |
| market_id, event_id, event_name, selection_id, selection_name | the source's identifiers (a football match: event_id = match_id) |
| off_time_utc | the scheduled off / kick-off, UTC, tz-aware |
| n_runners | selections in the market that were not removed |
| open_price, open_time_utc, open_matched | the earliest price the source holds, when it stood (only the recorder has a time), the money matched by then |
| p_t24h, p_t60, p_t15, p_t1 | the price standing 24 h, 60, 15 and 1 minute before the off, where the source has them |
| close_price | the source's closing price: the BSP where the market has one, else the last price traded before in play (Historic Data), else the benchmark close (football) |
| bsp | the Betfair SP where the market has one |
| pre_play_matched | the money matched before the off (a runner's in the price files; the market's in the recorder's books) |
| result | 1 won / placed, 0 lost, None unknown or not settled here (Asian handicap) |
| source | which reader made the row (``betfair_historic``, ``betfair_prices``, ``betfair_live_marks``, ``football_gold:<open book>><close book>``, ``greyhound_prices``) |

Readers (each takes a ``Store`` or a database path and returns a frame in that schema; a source not held gives an
empty frame, never an error):

- ``read_betfair_historic(store, sport)``: Betfair's Historic Data tables (``betfair_historic.py --build``: first
  traded, T-24h/T-60/T-15/T-1, close, BSP, result). The files carry no timestamp of the first trade and no volume
  (the BASIC plan), so open_time_utc, open_matched and pre_play_matched are empty there;
- ``read_horse_prices(db_path)``: ``betfair_prices`` (Betfair's price files): the morning WAP as the open with its
  volume, the BSP as the close; ``read_horse_marks(db_path)``: ``betfair_live_marks`` + ``betfair_live_markets`` (the
  recorder, from 1 Oct 2026): the 08:00 UK mid as an alternative open, with its time and the market's matched money,
  the mids at T-60/T-15/T-1, the last pre-off mid as the close, the settled BSP and result;
- ``read_football_gold(store)``: the gold matches' benchmark open and close a market (1X2, over/under 2.5, Asian
  handicap), one row an outcome, the books that supplied each named in ``source``;
- ``read_greyhound_prices(store)``: the greyhound price-file tables (``sources/greyhound_prices.py``), as the horses';
- ``read_recorded_books(store, sport, tag)``: the recorder's own day files (``betfair_live/<day>/books_<tag>.csv.gz``;
  the GB/IE greyhound record from 6 Oct 2026 is ``greyhound``): the first two-sided pre-off book as the open, with its
  time and the money matched by then (the tradable open the Historic Data files cannot give), the mids at
  T-60/T-15/T-1 and the last book through ``betfair_recorder.marks``, the BSP and the result.

``build(store, sport, db_path=None)`` concatenates the sport's readers and writes ``sources/open_close/panel_<sport>_
<year>.parquet`` (the year of the off), overwriting the same keys: safe to rerun. Sports: ``horse_racing`` (needs the
database), ``greyhound_racing``, ``football``, ``tennis``, ``cricket``, ``darts``, ``snooker``.

    python -m sources.open_close --build --sport greyhound_racing            # the S3 store
    python -m sources.open_close --build --sport horse_racing --db horse_racing.db --root /tmp/x
    python -m sources.open_close --report                                    # what the panel holds

Not in ``source_data.py``'s nightly list: that job builds the Historic Data tables after its source loop (the panel
would read them a night stale) and runs without horse_racing.db.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys

import numpy as np
import pandas as pd

from sources.common import Store

log = logging.getLogger(__name__)

PREFIX = "open_close"
SCHEMA = ["sport", "country", "market_type", "market_id", "event_id", "event_name", "selection_id", "selection_name",
          "off_time_utc", "n_runners", "open_price", "open_time_utc", "open_matched", "p_t24h", "p_t60", "p_t15",
          "p_t1", "close_price", "bsp", "pre_play_matched", "result", "source"]
PRICE_COLS = ["open_price", "open_matched", "p_t24h", "p_t60", "p_t15", "p_t1", "close_price", "bsp",
              "pre_play_matched"]
#: the panel's sport names -> the Historic Data service's (betfair_historic.PLAN_SCOPE) and its table stem
HISTORIC_SPORT = {"greyhound_racing": "greyhound_racing", "football": "soccer", "tennis": "tennis",
                  "cricket": "cricket", "darts": "darts", "snooker": "snooker"}
SPORTS = ("horse_racing",) + tuple(HISTORIC_SPORT)
COUNTRY = {"uk": "GB", "ire": "IE", "aus": "AU", "AUS": "AU", "usa": "US", "USA": "US", "fr": "FR", "rsa": "ZA",
           "RSA": "ZA", "uae": "AE", "nz": "NZ", "NZ": "NZ", "GB": "GB", "IE": "IE", "AU": "AU"}
UK = "Europe/London"


# --------------------------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------------------------

def empty() -> pd.DataFrame:
    return conform(pd.DataFrame(columns=SCHEMA))


def conform(df: pd.DataFrame) -> pd.DataFrame:
    """Every column of the schema, in order, typed: prices float, result and n_runners nullable integers, the
    identifiers text, the times tz-aware UTC."""
    out = pd.DataFrame(index=df.index)
    for c in SCHEMA:
        out[c] = df[c] if c in df.columns else None
    for c in PRICE_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    for c in ("result", "n_runners"):
        out[c] = pd.to_numeric(out[c], errors="coerce").round().astype("Int64")
    for c in ("off_time_utc", "open_time_utc"):
        out[c] = to_utc(out[c])
    for c in ("sport", "country", "market_type", "market_id", "event_id", "event_name", "selection_id",
              "selection_name", "source"):
        s = out[c]
        out[c] = s.where(s.notna(), None).map(lambda v: None if v is None else str(v)).astype(object)
    return out.reset_index(drop=True)


def to_utc(s: pd.Series) -> pd.Series:
    """Any series of times (ISO text with or without a zone, naive or aware timestamps) -> tz-aware UTC; a naive
    time is read as UTC."""
    if len(s) == 0:
        return pd.Series(pd.DatetimeIndex([], tz="UTC"), index=s.index)
    t = pd.to_datetime(s, errors="coerce", utc=True, format="ISO8601") if s.dtype == object else \
        pd.to_datetime(s, errors="coerce", utc=True)
    return t


def uk_local_to_utc(dates: pd.Series, times: pd.Series) -> pd.Series:
    """A UK wall-clock date ('2026-03-12') and time ('14:30') -> UTC, DST-aware; NaT where either is missing or the
    time falls in the clocks' change."""
    d = dates.fillna("").astype(str).str.strip()
    t = times.fillna("").astype(str).str.strip()
    ok = d.str.match(r"^\d{4}-\d{2}-\d{2}$") & t.str.match(r"^\d{1,2}:\d{2}$")
    local = pd.to_datetime((d + " " + t).where(ok), errors="coerce", format="%Y-%m-%d %H:%M")
    return local.dt.tz_localize(UK, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")


def _country(s: pd.Series) -> pd.Series:
    return s.map(lambda v: COUNTRY.get(str(v), str(v).upper()) if v is not None and str(v) not in ("", "nan") else None)


# --------------------------------------------------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------------------------------------------------

def read_betfair_historic(store: Store, sport: str, years: list[int] | None = None) -> pd.DataFrame:
    """Betfair's Historic Data tables of one sport (``betfair_historic/markets_<sport>_<year>.parquet``)."""
    stem = HISTORIC_SPORT.get(sport, sport)
    keys = sorted(k for k in store.listing("betfair_historic/")
                  if re.fullmatch(rf"betfair_historic/markets_{re.escape(stem)}_(\d{{4}})\.parquet", k)
                  and (not years or int(k[-12:-8]) in years))
    frames = []
    for k in keys:
        df = store.get_parquet(k)
        if df is None or df.empty:
            continue
        for c in ("ltp_first", "ltp_t_1440", "ltp_t_60", "ltp_t_15", "ltp_t_1", "ltp_close", "bsp", "won"):
            if c not in df.columns:
                df[c] = np.nan
        active = ~df.get("runner_status", pd.Series("ACTIVE", index=df.index)).astype(str).isin(["REMOVED", "REMOVED_VACANT"])
        n = df[active].groupby("market_id")["selection_id"].transform("size")
        frames.append(pd.DataFrame({
            "sport": sport, "country": _country(df["country"]) if "country" in df else None,
            "market_type": df.get("market_type"), "market_id": df["market_id"], "event_id": df.get("event_id"),
            "event_name": df.get("event_name"), "selection_id": df["selection_id"],
            "selection_name": df.get("runner_name"), "off_time_utc": df.get("market_time"),
            "n_runners": n.reindex(df.index), "open_price": df["ltp_first"], "open_time_utc": None,
            "open_matched": np.nan, "p_t24h": df["ltp_t_1440"], "p_t60": df["ltp_t_60"], "p_t15": df["ltp_t_15"],
            "p_t1": df["ltp_t_1"], "close_price": df["ltp_close"], "bsp": df["bsp"], "pre_play_matched": np.nan,
            "result": df["won"], "source": "betfair_historic"}))
    if not frames:
        return empty()
    return conform(pd.concat(frames, ignore_index=True))


def _connect(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", [name]).fetchone() is not None


def read_horse_prices(db_path: str, date_from: str | None = None, date_to: str | None = None) -> pd.DataFrame:
    """``betfair_prices`` (Betfair's price files, UK/IE win and place): the morning WAP as the open with the morning
    volume, the BSP as the close, the pre-play volume, the result. The files carry no market id: ``market_id`` is the
    file's event id with the market type (``<event_id>:WIN``), ``event_id`` the event id."""
    conn = _connect(db_path)
    try:
        if not _has_table(conn, "betfair_prices"):
            return empty()
        where, params = [], []
        if date_from:
            where.append("race_date >= ?"); params.append(date_from)
        if date_to:
            where.append("race_date <= ?"); params.append(date_to)
        sql = ("SELECT country, market_type, race_date, race_time, event_id, event_name, selection_id, selection_name, "
               "win_lose, bsp, morningwap, morning_vol, ppwap, pp_vol FROM betfair_prices"
               + (" WHERE " + " AND ".join(where) if where else ""))
        d = pd.read_sql_query(sql, conn, params=params or None)
    finally:
        conn.close()
    if d.empty:
        return empty()
    mt = d["market_type"].astype(str).str.upper()
    n = d.groupby(["event_id", "market_type"])["selection_id"].transform("size")
    return conform(pd.DataFrame({
        "sport": "horse_racing", "country": _country(d["country"]), "market_type": mt,
        "market_id": d["event_id"].astype("Int64").astype(str) + ":" + mt, "event_id": d["event_id"],
        "event_name": d["event_name"], "selection_id": d["selection_id"], "selection_name": d["selection_name"],
        "off_time_utc": uk_local_to_utc(d["race_date"], d["race_time"]), "n_runners": n,
        "open_price": d["morningwap"].where(d["morningwap"] > 1), "open_time_utc": None,
        "open_matched": d["morning_vol"], "p_t24h": np.nan, "p_t60": np.nan, "p_t15": np.nan, "p_t1": np.nan,
        "close_price": d["bsp"].where(d["bsp"] > 1), "bsp": d["bsp"].where(d["bsp"] > 1),
        "pre_play_matched": d["pp_vol"], "result": d["win_lose"].where(d["win_lose"].isin([0, 1])),
        "source": "betfair_prices"}))


MARK_OPEN = "clock_08:00"
MARK_PRICE = {"off_60": "p_t60", "off_15": "p_t15", "off_1": "p_t1"}
LIVE_PREFIX = "betfair_live"                       # the recorder's day files, beside sources/ in the capture bucket


def _mid(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where((df["back1"] > 1) & (df["lay1"] > 1), (df["back1"] + df["lay1"]) / 2.0, np.nan),
                     index=df.index)


def _panel_from_marks(marks: pd.DataFrame, cat: pd.DataFrame, sport: str, source: str, open_mark: str | None,
                      first: pd.DataFrame | None = None) -> pd.DataFrame:
    """The recorder's marks (``betfair_recorder.MARK_FIELDS``: the book nearest each mark, the last, the settled one)
    and its catalogue -> the panel. The open is the mid at ``open_mark`` (the 08:00 UK mark for horses), or the
    ``first`` frame (the first two-sided book the record holds, indexed by market and selection, with open_price,
    open_time_utc and open_matched). The mids at T-60/T-15/T-1, the last pre-off mid as the close with the market's
    matched money then, the settled BSP and result. A runner without a two-sided book at a mark has no price there."""
    if marks is None or marks.empty or cat is None or cat.empty:
        return empty()
    marks = marks.copy()
    cat = cat.copy()
    for c in ("back1", "lay1", "market_total_matched", "active_runners", "bsp"):
        marks[c] = pd.to_numeric(marks[c], errors="coerce")
    marks["selection_id"] = pd.to_numeric(marks["selection_id"], errors="coerce").astype("Int64")
    cat["selection_id"] = pd.to_numeric(cat["selection_id"], errors="coerce").astype("Int64")
    cat = cat.drop_duplicates(["market_id", "selection_id"], keep="last")
    key = ["market_id", "selection_id"]
    marks["mid"] = _mid(marks)

    def at(mark: str, cols: dict[str, str]) -> pd.DataFrame:
        m = marks[marks["mark"] == mark].drop_duplicates(key, keep="last").set_index(key)
        return m[list(cols)].rename(columns=cols)

    out = cat.set_index(key)[["market_type", "country", "venue", "event_id", "event_name", "runner_name",
                              "market_start_utc"]].copy()
    if first is not None:
        out = out.join(first[["open_price", "open_time_utc", "open_matched"]])
    else:
        out = out.join(at(open_mark, {"mid": "open_price", "polled_utc": "open_time_utc",
                                      "market_total_matched": "open_matched"}))
    for mark, col in MARK_PRICE.items():
        out = out.join(at(mark, {"mid": col}))
    out = out.join(at("last", {"mid": "close_price", "market_total_matched": "pre_play_matched",
                               "active_runners": "n_runners"}))
    out = out.join(at("final", {"bsp": "bsp", "runner_status": "status_final"}))
    out = out.reset_index()
    out["result"] = out["status_final"].map({"WINNER": 1, "LOSER": 0, "PLACED": 1})
    # a runner the recorder never saw with a book or a settled status: no row
    seen = out[["open_price", "close_price", "bsp"] + list(MARK_PRICE.values())].notna().any(axis=1) | out["result"].notna()
    out = out[seen]
    return conform(pd.DataFrame({
        "sport": sport, "country": _country(out["country"]), "market_type": out["market_type"],
        "market_id": out["market_id"], "event_id": out["event_id"], "event_name": out["event_name"],
        "selection_id": out["selection_id"], "selection_name": out["runner_name"],
        "off_time_utc": out["market_start_utc"], "n_runners": out["n_runners"], "open_price": out["open_price"],
        "open_time_utc": out["open_time_utc"], "open_matched": out["open_matched"], "p_t24h": np.nan,
        "p_t60": out["p_t60"], "p_t15": out["p_t15"], "p_t1": out["p_t1"], "close_price": out["close_price"],
        "bsp": out["bsp"], "pre_play_matched": out["pre_play_matched"], "result": out["result"],
        "source": source}))


def read_horse_marks(db_path: str, open_mark: str = MARK_OPEN, date_from: str | None = None,
                     date_to: str | None = None) -> pd.DataFrame:
    """The recorder's books as the nightly load holds them (``betfair_live_marks`` with ``betfair_live_markets``): the
    mid of the two-sided book at the 08:00 UK mark (``open_mark``) as the open, with its time and the market's
    matched money then; the mids at T-60/T-15/T-1; the last pre-off mid as the close; the settled BSP and result."""
    conn = _connect(db_path)
    try:
        if not (_has_table(conn, "betfair_live_marks") and _has_table(conn, "betfair_live_markets")):
            return empty()
        where, params = [], []
        if date_from:
            where.append("m.race_date >= ?"); params.append(date_from)
        if date_to:
            where.append("m.race_date <= ?"); params.append(date_to)
        marks = pd.read_sql_query(
            "SELECT m.race_date, m.market_id, m.selection_id, m.mark, m.polled_utc, m.back1, m.lay1, "
            "m.market_total_matched, m.active_runners, m.bsp, m.runner_status FROM betfair_live_marks m"
            + (" WHERE " + " AND ".join(where) if where else ""), conn, params=params or None)
        cat = pd.read_sql_query(
            "SELECT market_id, selection_id, market_type, country, venue, event_id, event_name, runner_name, "
            "market_start_utc FROM betfair_live_markets", conn)
    finally:
        conn.close()
    return _panel_from_marks(marks, cat, "horse_racing", "betfair_live_marks", open_mark)


def _live_days(store: Store, tag: str) -> dict[str, tuple[str, str]]:
    """{day: (books file, markets file)} for every day the record holds the tag's books (``books_<tag>.csv.gz``;
    the untagged horse record is ``books.csv.gz``). Local store: ``<root>/../betfair_live/<day>/``."""
    suffix = f"_{tag}" if tag else ""
    books_name, markets_name = f"books{suffix}.csv", f"markets{suffix}.csv"
    out = {}
    if store.root:
        base = store.root.parent / LIVE_PREFIX
        if base.exists():
            for d in sorted(p for p in base.iterdir() if p.is_dir()):
                for ext in ("", ".gz"):
                    b, m = d / (books_name + ext), d / (markets_name + ext)
                    if b.exists():
                        out[d.name] = (str(b), str(m) if m.exists() else None)
                        break
        return out
    keys = set()
    for page in store.s3.get_paginator("list_objects_v2").paginate(Bucket=store.bucket, Prefix=f"{LIVE_PREFIX}/"):
        keys |= {o["Key"] for o in page.get("Contents", [])}
    for k in sorted(keys):
        m = re.fullmatch(rf"{LIVE_PREFIX}/(\d{{4}}-\d{{2}}-\d{{2}})/{re.escape(books_name)}\.gz", k)
        if m:
            mk = f"{LIVE_PREFIX}/{m.group(1)}/{markets_name}.gz"
            out[m.group(1)] = (k, mk if mk in keys else None)
    return out


def _read_live_csv(store: Store, where: str | None) -> pd.DataFrame | None:
    import gzip
    import io
    if not where:
        return None
    raw = open(where, "rb").read() if store.root else store.s3.get_object(Bucket=store.bucket, Key=where)["Body"].read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return pd.read_csv(io.BytesIO(raw), dtype={"market_id": str}, on_bad_lines="skip")


def read_recorded_books(store: Store, sport: str = "greyhound_racing", tag: str = "greyhound",
                        date_from: str | None = None, date_to: str | None = None) -> pd.DataFrame:
    """The recorder's own day files (``betfair_live/<day>/books_<tag>.csv.gz`` with ``markets_<tag>.csv.gz``; the GB/IE
    greyhound record from 6 Oct 2026 is tag ``greyhound``): the FIRST two-sided pre-off book the record holds as the
    open, with its time and the market's matched money then (the Historic Data files have no time or volume for
    the first trade; this is the tradable open), the mids at T-60/T-15/T-1 and the last pre-off book through
    ``betfair_recorder.marks`` (the same picks as the nightly load's), the BSP as the market reconciled it, the
    settled result. Source ``betfair_live_books:<tag>``."""
    from datetime import date as _date
    import betfair_recorder as br
    frames = []
    for day, (bk, mk) in _live_days(store, tag).items():
        if (date_from and day < date_from) or (date_to and day > date_to):
            continue
        books, markets = _read_live_csv(store, bk), _read_live_csv(store, mk)
        if books is None or books.empty or markets is None or markets.empty:
            continue
        for c in ("back1", "lay1", "market_total_matched"):
            books[c] = pd.to_numeric(books[c], errors="coerce")
        books["selection_id"] = pd.to_numeric(books["selection_id"], errors="coerce").astype("Int64")
        markets["selection_id"] = pd.to_numeric(markets["selection_id"], errors="coerce").astype("Int64")
        marks = br.marks(books, markets, _date.fromisoformat(day))
        offs = markets.drop_duplicates("market_id").set_index("market_id")["market_start_utc"]
        b = books.copy()
        b["polled"] = pd.to_datetime(b["polled_utc"], utc=True, errors="coerce")
        b["off"] = pd.to_datetime(b["market_id"].map(offs), utc=True, errors="coerce")
        b["mid"] = _mid(b)
        pre = b[(b["source"] != "final") & (b["status"] == "OPEN") & b["inplay"].astype(str).isin(["0", "False", "false"])
                & (b["polled"] < b["off"]) & b["mid"].notna()]
        first = (pre.sort_values("polled").drop_duplicates(["market_id", "selection_id"], keep="first")
                 .set_index(["market_id", "selection_id"])
                 .rename(columns={"mid": "open_price", "polled_utc": "open_time_utc",
                                  "market_total_matched": "open_matched"}))
        frames.append(_panel_from_marks(marks, markets, sport, f"betfair_live_books:{tag or 'horse'}", None, first))
    frames = [f for f in frames if len(f)]
    return conform(pd.concat(frames, ignore_index=True)) if frames else empty()


FOOTBALL_MARKETS = {
    # market type: (open cols, close cols, source cols, selection names, n_runners)
    "MATCH_ODDS": (("open_h", "open_d", "open_a"), ("close_h", "close_d", "close_a"), ("open_1x2_src", "close_1x2_src"),
                   ("H", "D", "A"), 3),
    "OVER_UNDER_25": (("open_o25", "open_u25"), ("close_o25", "close_u25"), ("open_ou_src", "close_ou_src"),
                      ("Over 2.5", "Under 2.5"), 2),
    "ASIAN_HANDICAP": (("open_ah_h", "open_ah_a"), ("close_ah_h", "close_ah_a"), ("open_ah_src", "close_ah_src"),
                       ("AH home", "AH away"), 2),
}


def _football_result(g: pd.DataFrame, market: str, outcome: str) -> pd.Series:
    fin = g["status"].astype(str).eq("FINISHED") & g["ft_home"].notna() & g["ft_away"].notna()
    h, a = g["ft_home"], g["ft_away"]
    if market == "MATCH_ODDS":
        won = {"H": h > a, "D": h == a, "A": h < a}[outcome]
    elif market == "OVER_UNDER_25":
        won = (h + a > 2.5) if outcome.startswith("Over") else (h + a < 2.5)
    else:                                   # an Asian handicap settles on its line, in halves and quarters: not here
        return pd.Series(np.nan, index=g.index)
    return pd.Series(np.where(fin, won.astype(float), np.nan), index=g.index)


def read_football_gold(store: Store) -> pd.DataFrame:
    """The gold matches' benchmark open and close (``football.data.benchmarks``), one row an outcome a market."""
    g = store.get_parquet("football/gold/matches.parquet")
    if g is None or g.empty:
        return empty()
    frames = []
    for market, (ocols, ccols, (osrc, csrc), names, n) in FOOTBALL_MARKETS.items():
        if not all(c in g.columns for c in ocols + ccols):
            continue
        has = g[list(ocols)].notna().any(axis=1) | g[list(ccols)].notna().any(axis=1)
        m = g[has]
        if m.empty:
            continue
        src = ("football_gold:" + m[osrc].fillna("-").astype(str) + ">" + m[csrc].fillna("-").astype(str)) \
            if osrc in m and csrc in m else "football_gold"
        line = m["open_ah_line"] if market == "ASIAN_HANDICAP" and "open_ah_line" in m else None
        for oc, cc, name in zip(ocols, ccols, names):
            sel = name if line is None else name + " " + line.map(lambda v: "" if pd.isna(v) else f"{v:+g}")
            frames.append(pd.DataFrame({
                "sport": "football", "country": m["country"], "market_type": market, "market_id": None,
                "event_id": m["match_id"],
                "event_name": m["home_team"].astype(str) + " v " + m["away_team"].astype(str),
                "selection_id": None, "selection_name": sel, "off_time_utc": m["kickoff_utc"],
                "n_runners": n, "open_price": m[oc], "open_time_utc": None, "open_matched": np.nan,
                "p_t24h": np.nan, "p_t60": np.nan, "p_t15": np.nan, "p_t1": np.nan, "close_price": m[cc],
                "bsp": np.nan, "pre_play_matched": np.nan, "result": _football_result(m, market, name),
                "source": src}))
    if not frames:
        return empty()
    out = pd.concat(frames, ignore_index=True)
    # a match without a kick-off time (before 2019/20 football-data gives the date only): midday UTC on the day
    if "match_date" in g.columns:
        noon = pd.to_datetime(g.set_index("match_id")["match_date"], errors="coerce").dt.tz_localize("UTC") \
            + pd.Timedelta(hours=12)
        ko = to_utc(out["off_time_utc"])
        out["off_time_utc"] = ko.where(ko.notna(), out["event_id"].map(noon))
    return conform(out)


def read_greyhound_prices(store: Store, years: list[int] | None = None) -> pd.DataFrame:
    """The greyhound price-file tables (``greyhound_prices/prices_<year>.parquet``): as the horses' price files."""
    keys = sorted(k for k in store.listing("greyhound_prices/")
                  if re.fullmatch(r"greyhound_prices/prices_(\d{4})\.parquet", k)
                  and (not years or int(k[-12:-8]) in years))
    frames = []
    for k in keys:
        d = store.get_parquet(k)
        if d is None or d.empty:
            continue
        mt = d["market"].astype(str).str.upper()
        n = d.groupby(["market", "event_id"])["selection_id"].transform("size")
        frames.append(pd.DataFrame({
            "sport": "greyhound_racing", "country": _country(d["country"]), "market_type": mt,
            "market_id": pd.to_numeric(d["event_id"], errors="coerce").astype("Int64").astype(str) + ":" + mt,
            "event_id": pd.to_numeric(d["event_id"], errors="coerce").astype("Int64"),
            "event_name": d["event_name"], "selection_id": pd.to_numeric(d["selection_id"], errors="coerce").astype("Int64"),
            "selection_name": d["selection_name"], "off_time_utc": uk_local_to_utc(d["race_date"], d["race_time"]),
            "n_runners": n, "open_price": d["morningwap"].where(d["morningwap"] > 1), "open_time_utc": None,
            "open_matched": d["morning_vol"], "p_t24h": np.nan, "p_t60": np.nan, "p_t15": np.nan, "p_t1": np.nan,
            "close_price": d["bsp"].where(d["bsp"] > 1), "bsp": d["bsp"].where(d["bsp"] > 1),
            "pre_play_matched": d["pp_vol"], "result": d["win_lose"].where(d["win_lose"].isin([0, 1])),
            "source": "greyhound_prices"}))
    if not frames:
        return empty()
    return conform(pd.concat(frames, ignore_index=True))


# --------------------------------------------------------------------------------------------------------------------
# The build
# --------------------------------------------------------------------------------------------------------------------

def readers(store: Store, sport: str, db_path: str | None = None) -> list[tuple[str, callable]]:
    """The readers of a sport, named, each a no-argument callable."""
    out = []
    if sport == "horse_racing":
        if db_path:
            out += [("betfair_prices", lambda: read_horse_prices(db_path)),
                    ("betfair_live_marks", lambda: read_horse_marks(db_path))]
        else:
            log.warning("horse_racing: no --db given; the price files and the recorder's marks are in horse_racing.db")
    else:
        out.append(("betfair_historic", lambda: read_betfair_historic(store, sport)))
    if sport == "greyhound_racing":
        out.append(("greyhound_prices", lambda: read_greyhound_prices(store)))
        out.append(("betfair_live_books", lambda: read_recorded_books(store, "greyhound_racing", "greyhound")))
    if sport == "football":
        out.append(("football_gold", lambda: read_football_gold(store)))
    return out


def panel_key(sport: str, year) -> str:
    return f"{PREFIX}/panel_{sport}_{year}.parquet"


def build(store: Store, sport: str, db_path: str | None = None) -> dict:
    """The sport's panel, a parquet a year of the off under ``sources/open_close/``, each key overwritten: idempotent.
    One reader failing is logged and left out; the others still write."""
    if sport not in SPORTS:
        raise ValueError(f"unknown sport {sport!r}: one of {SPORTS}")
    parts, out = [], {"sport": sport, "readers": {}, "years": {}}
    for name, fn in readers(store, sport, db_path):
        try:
            df = fn()
            out["readers"][name] = len(df)
            if len(df):
                parts.append(df)
        except Exception as exc:                          # noqa: BLE001 - one source failing must not stop the others
            log.exception("%s/%s failed", sport, name)
            out["readers"][name] = f"error: {str(exc)[:200]}"
    if not parts:
        return out
    df = conform(pd.concat(parts, ignore_index=True))
    year = df["off_time_utc"].dt.year.astype("Int64")
    for y, g in df.groupby(year.fillna(0)):
        label = str(int(y)) if int(y) else "unknown"
        key = panel_key(sport, label)
        store.put_parquet(key, g.reset_index(drop=True))
        out["years"][label] = {"rows": int(len(g)), "markets": int(g["market_id"].fillna(g["event_id"]).nunique()),
                               "with_open_and_close": int((g["open_price"].notna() & g["close_price"].notna()).sum()),
                               "key": store.where(key)}
    return out


def report(store: Store) -> dict:
    out = {}
    for k, size in sorted(store.listing(f"{PREFIX}/").items()):
        m = re.fullmatch(rf"{PREFIX}/panel_([a-z_]+)_(\d{{4}}|unknown)\.parquet", k)
        if m:
            out.setdefault(m.group(1), {})[m.group(2)] = round(size / 1e6, 2)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--report", action="store_true", help="the panel files held (MB)")
    ap.add_argument("--sport", default=None, help=f"one of {','.join(SPORTS)}, or all (every sport but horse_racing "
                                                  "unless --db is given)")
    ap.add_argument("--db", default=None, help="horse_racing.db, for horse_racing")
    ap.add_argument("--root", default=None, help="a local folder in place of S3 (its sources/ layout)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    store = Store(root=a.root)
    out = {}
    if a.build:
        if not a.sport:
            ap.error("--build needs --sport")
        sports = [s for s in SPORTS if s != "horse_racing" or a.db] if a.sport == "all" else [a.sport]
        out["build"] = [build(store, s, a.db) for s in sports]
    if a.report:
        out["report"] = report(store)
    if not out:
        ap.print_help()
        return 1
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
