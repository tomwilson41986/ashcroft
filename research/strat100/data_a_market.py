"""The a_market data set of the 100-strategy harness (racing², conventions K1-K10, registered 24 Nov 2026).

Betfair's own daily price files (the table ``betfair_prices``: UK and IRE, win and place) joined to ``race_results``
by ``race_results_id``, 2018-01-01 to 2026-03-27. No model price. Read-only: nothing here writes to the database,
calls Betfair, HRB or S3, or places a bet.

    read_tables(db)      the two raw tables, the locked holdout left out in the SQL itself (race_date < 2026-03-28)
    build_frames(rr, bp) the frames the strategies read: a pure function of the two raw tables (tests call it on toy
                         frames with the data map's columns)
    load_a_market()      build_frames(*read_tables(db_path()))

THE FRAMES (never changed in place by a strategy):

    win     one row a selection of a WIN market, 2018 on. The file's columns (bsp, ppwap, morningwap, ppmax, ppmin,
            ipmax, ipmin, morning_vol, pp_vol, ip_vol, win_lose; a price at or under 1.0 is no price: NaN) and
              bp_id        betfair_prices.id
              rk           race_date|event_id: the race key of every a_market bet (the win market's id)
              event_ts     the off as a timestamp (event_dt)
              n_sel, n_bsp selections in the market, and those with a BSP
              race_whole   no selection lacks a BSP (K6: an entry before the off is read only in these races)
              q_pp         normalised 1 / win ppwap over the runners with a BSP (NaN in a race where one of them has
                           no ppwap above 1.01); rank_pp its rank, 1 = shortest (ties share the lower rank)
              rr_id        the race_results row it is joined to (NaN where not joined or not before the holdout)
              rr_runners   that race's race_results.number_of_runners (the largest over the market's joined rows)
              placing      race_results.placing_numerical of the joined row
    pairs   PAIR-A: a win row and the place row of the same (country, menu_hint, event_dt, selection_id). The win
            row's columns, the place file's four sound ones (K4) as p_bsp, p_ppwap, p_pp_vol, p_win_lose, and
              n_win_bsp    runners with a win BSP in the win market
              place_whole  every runner with a win BSP has a place BSP
              k            places paid: the count of the place market's rows with win_lose 1
              k_ok         2 <= k <= 4 and k = round(sum of 1 / place bsp) (K9)
              p_place      K9's place chance from q_pp and k (core.place_prob once a race), where the race has 5+
                           runners, is place_whole and k_ok and every runner has a win ppwap; NaN elsewhere
    runs    one row a runner of race_results from 2010 (the history a point-in-time feature needs) with the card
            columns, the strategy studies' code (chase / hurdle / nhflat / aw / flat as research/strat/strat_lib.py),
            hcap, the joined win row's columns under the names above (NaN where there is no price record), and the
            K8 run-style columns from model/race_shape.py on EARLIER runs' comments: p_lead, p_prom, p_mid, p_rear,
            style_n_eff, shape_bin, lead_rank
    days    the sorted ISO race days of the win frame (a day with no bet is then 0, K10)
"""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.strat100 import core

ROOT = Path(__file__).resolve().parents[2]
START = "2018-01-01"                      # K1: the price files' first year
RR_COLS = ["id", "race_date", "race_time", "track", "race_name", "race_class", "race_type", "race_code",
           "surface_type", "dist_furlongs", "going_description", "number_of_runners", "placing_numerical",
           "horse_name", "trainer", "jockey_name", "jockeys_claim", "official_rating", "horse_age", "bfsp",
           "bfsp_place", "bf_plcs_paid", "comment", "stall", "days_since_lr", "career_runs", "headgear"]
RR_REQUIRED = ["id", "race_date", "race_time", "track", "race_name", "race_type", "number_of_runners",
               "placing_numerical", "horse_name", "trainer", "jockey_name", "comment"]
RR_NUM = ["id", "dist_furlongs", "number_of_runners", "placing_numerical", "jockeys_claim", "official_rating",
          "horse_age", "bfsp", "bfsp_place", "bf_plcs_paid", "stall", "days_since_lr", "career_runs"]
RR_TEXT = ["race_time", "track", "race_name", "race_class", "race_type", "race_code", "surface_type",
           "going_description", "horse_name", "trainer", "jockey_name", "comment", "headgear"]
BP_COLS = ["id", "country", "market_type", "race_date", "race_time", "event_id", "menu_hint", "event_name",
           "event_dt", "selection_id", "selection_name", "horse_norm", "win_lose", "bsp", "ppwap", "morningwap",
           "ppmax", "ppmin", "ipmax", "ipmin", "morning_vol", "pp_vol", "ip_vol", "race_results_id"]
PRICES = ["bsp", "ppwap", "morningwap", "ppmax", "ppmin", "ipmax", "ipmin"]
VOLS = ["morning_vol", "pp_vol", "ip_vol"]
PAIR_A = ["country", "menu_hint", "event_dt", "selection_id"]                 # K4
WIN_TO_RUNS = ["bp_id", "rk", "event_id", "selection_id", "country", "menu_hint", "event_name", "event_ts",
               "horse_norm", "win_lose", "n_sel", "n_bsp", "race_whole"] + PRICES + VOLS
STYLE_COLS = ["p_lead", "p_prom", "p_mid", "p_rear", "style_n_eff", "shape_bin", "lead_rank"]
MIN_PLACE_FIELD = 5                       # S069's filter: K9's chance is computed for these races only


class NoDataFound(RuntimeError):
    """The database or one of its tables is not there (the run records it against the data set and goes on)."""


def say(*a) -> None:
    print(*a, flush=True)


def db_path() -> Path:
    """horse_racing.db: $STRAT100_DB, else the working directory, else the repository root."""
    tried = [os.environ.get("STRAT100_DB"), Path.cwd() / "horse_racing.db", ROOT / "horse_racing.db"]
    for p in tried:
        if p and Path(p).is_file():
            return Path(p)
    raise NoDataFound("no data found: horse_racing.db is not at $STRAT100_DB, the working directory or the "
                      "repository root (bash scripts/setup_session.sh --db fetches it on the runner)")


def read_tables(db) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(race_results, betfair_prices) with the explicit columns above. The holdout is left out in the SQL
    (race_date < 2026-03-28), before a row of it is read. Read-only connection."""
    con = sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)
    try:
        tabs = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for t in ("race_results", "betfair_prices"):
            if t not in tabs:
                raise NoDataFound(f"no data found: the table {t} is not in {db}")
        have = {r[1] for r in con.execute("PRAGMA table_info(race_results)")}
        lack = [c for c in RR_REQUIRED if c not in have]
        if lack:
            raise NoDataFound(f"no data found: race_results lacks the columns {lack}")
        cols = [c for c in RR_COLS if c in have]
        if set(RR_COLS) - have:
            say(f"  a_market: race_results has no {sorted(set(RR_COLS) - have)} (read as empty)")
        rr = pd.read_sql_query(f"SELECT {', '.join(cols)} FROM race_results WHERE race_date < ?", con,
                               params=(core.HOLDOUT_FROM,))
        have = {r[1] for r in con.execute("PRAGMA table_info(betfair_prices)")}
        lack = [c for c in BP_COLS if c not in have]
        if lack:
            raise NoDataFound(f"no data found: betfair_prices lacks the columns {lack}")
        bp = pd.read_sql_query(f"SELECT {', '.join(BP_COLS)} FROM betfair_prices WHERE race_date >= ? AND "
                               "race_date < ? AND market_type IN ('win', 'place')", con,
                               params=(START, core.HOLDOUT_FROM))
    finally:
        con.close()
    for c in RR_COLS:
        if c not in rr.columns:
            rr[c] = np.nan
    return rr, bp


def _text(s: pd.Series) -> pd.Series:
    """A text column as object strings with '' for a missing value."""
    return s.astype(object).where(s.notna(), "").astype(str).str.strip()


def strat_code(race_type: pd.Series, race_code: pd.Series) -> pd.Series:
    """chase / hurdle / nhflat / aw / flat exactly as research/strat/strat_lib.load (the strategy studies' code)."""
    rt = _text(race_type).str.lower()
    rc = _text(race_code).str.lower()
    return pd.Series(np.select([rt.str.contains("chase"), rt.str.contains("hurdle"),
                                rt.str.contains("nh flat|bumper"), rc.str.contains("all weather")],
                               ["chase", "hurdle", "nhflat", "aw"], default="flat"), index=race_type.index)


def _clean_results(rr: pd.DataFrame) -> pd.DataFrame:
    rr = core.drop_holdout(rr).copy()                                   # K1: before anything else
    n0 = len(rr)
    for c in RR_COLS:
        if c not in rr.columns:
            rr[c] = np.nan
    rr["race_date"] = rr["race_date"].astype(str).str[:10]
    for c in RR_NUM:
        rr[c] = pd.to_numeric(rr[c], errors="coerce")
    for c in RR_TEXT:
        rr[c] = _text(rr[c])
    rr["date"] = pd.to_datetime(rr["race_date"], errors="coerce")
    bad = rr["date"].isna() | rr["id"].isna() | rr["horse_name"].eq("")
    dup = ~bad & rr["id"].duplicated()
    rr = rr[~bad & ~dup].copy()
    rr["id"] = rr["id"].astype(np.int64)
    rr["code"] = strat_code(rr["race_type"], rr["race_code"])
    rr["hcap"] = rr["race_name"].str.contains("handicap", case=False)
    rr["year"] = rr["date"].dt.year.astype(int)
    rr = rr.sort_values(["race_date", "id"], kind="stable").reset_index(drop=True)
    say(f"  a_market race_results: {n0:,} rows before the holdout in, {len(rr):,} kept ({int(bad.sum()):,} with no "
        f"date, id or horse name dropped, {int(dup.sum()):,} repeated ids dropped); "
        f"{rr['race_date'].min() if len(rr) else '-'} .. {rr['race_date'].max() if len(rr) else '-'}")
    return rr


def _clean_prices(bp: pd.DataFrame) -> pd.DataFrame:
    bp = core.drop_holdout(bp).copy()                                   # K1: before anything else
    n0 = len(bp)
    lack = [c for c in BP_COLS if c not in bp.columns]
    if lack:
        raise NoDataFound(f"no data found: the betfair_prices frame lacks {lack}")
    bp["race_date"] = bp["race_date"].astype(str).str[:10]
    for c in ("country", "market_type", "menu_hint", "event_name", "event_dt", "selection_name", "horse_norm",
              "race_time"):
        bp[c] = _text(bp[c])
    bp["country"], bp["market_type"] = bp["country"].str.lower(), bp["market_type"].str.lower()
    for c in ["id", "event_id", "selection_id", "win_lose", "race_results_id"] + PRICES + VOLS:
        bp[c] = pd.to_numeric(bp[c], errors="coerce")
    for c in PRICES:
        bp[c] = bp[c].where(bp[c] > 1.0)                                # 0 or 1.0 in a file is "no price"
    for c in VOLS:
        bp[c] = bp[c].fillna(0.0).clip(lower=0.0)
    early = bp["race_date"] < START
    other = ~bp["country"].isin(["uk", "ire"]) | ~bp["market_type"].isin(["win", "place"])
    noid = bp["event_id"].isna() | bp["selection_id"].isna() | bp["id"].isna()
    keep = ~early & ~other & ~noid
    dup = keep & bp.duplicated(["market_type", "event_id", "selection_id"], keep="first")
    bp = bp[keep & ~dup].copy()
    for c in ("id", "event_id", "selection_id"):
        bp[c] = bp[c].astype(np.int64)
    ts = pd.to_datetime(bp["event_dt"], format="%d-%m-%Y %H:%M", errors="coerce")
    alt = pd.to_datetime(bp["race_date"] + " " + bp["race_time"], errors="coerce")
    bp["event_ts"] = ts.fillna(alt).fillna(pd.to_datetime(bp["race_date"], errors="coerce"))
    bp["year"] = bp["race_date"].str[:4].astype(int)
    say(f"  a_market betfair_prices: {n0:,} rows before the holdout in, {len(bp):,} kept ({int(early.sum()):,} "
        f"before {START}, {int((~early & other).sum()):,} not UK/IE win or place, "
        f"{int((~early & ~other & noid).sum()):,} with no market or selection id, {int(dup.sum()):,} repeated "
        f"(market, selection) dropped); win {int(bp['market_type'].eq('win').sum()):,}, place "
        f"{int(bp['market_type'].eq('place').sum()):,}")
    return bp.sort_values(["race_date", "event_ts", "event_id", "id"], kind="stable").reset_index(drop=True)


def _win_frame(bp: pd.DataFrame, rr: pd.DataFrame) -> pd.DataFrame:
    w = bp[bp["market_type"].eq("win")].drop(columns=["market_type"]).rename(columns={"id": "bp_id"}).copy()
    w["rk"] = w["race_date"] + "|" + w["event_id"].astype(str)
    has = w["bsp"].notna()
    g = w.groupby("event_id", sort=False)
    w["n_sel"] = g["bp_id"].transform("size").astype(int)
    w["n_bsp"] = has.groupby(w["event_id"], sort=False).transform("sum").astype(int)
    w["race_whole"] = w["n_sel"].eq(w["n_bsp"])
    pp = w["ppwap"].where(has & (w["ppwap"] > 1.01))
    short = (has & pp.isna()).groupby(w["event_id"], sort=False).transform("sum") > 0     # a runner with no ppwap
    inv = (1.0 / pp).where(~short)
    w["q_pp"] = inv / inv.groupby(w["event_id"], sort=False).transform("sum")
    w["rank_pp"] = pp.where(~short).groupby(w["event_id"], sort=False).rank(method="min")
    # the join to race_results: the row must be there (before the holdout) and of the same day
    key = rr[["id", "race_date", "number_of_runners", "placing_numerical"]].rename(
        columns={"id": "rr_id", "race_date": "rr_date", "number_of_runners": "rr_n", "placing_numerical": "placing"})
    w = w.merge(key, left_on="race_results_id", right_on="rr_id", how="left")
    asked, found = int(w["race_results_id"].notna().sum()), int(w["rr_id"].notna().sum())
    wrong = w["rr_id"].notna() & w["rr_date"].ne(w["race_date"])
    w.loc[wrong, ["rr_id", "rr_n", "placing"]] = np.nan
    order = w.assign(_nb=w["bsp"].isna()).sort_values(["_nb", "bp_id"], kind="stable")     # a row with a BSP first
    again = w["rr_id"].notna() & order["rr_id"].duplicated(keep="first").reindex(w.index)
    w.loc[again, ["rr_id", "rr_n", "placing"]] = np.nan
    w["rr_runners"] = w.groupby("event_id", sort=False)["rr_n"].transform("max")
    w = w.drop(columns=["rr_date", "rr_n", "race_results_id"])
    say(f"  a_market win rows: {len(w):,} in {w['event_id'].nunique():,} markets; with a BSP {int(has.sum()):,}; "
        f"markets where no selection lacks a BSP {int(w.loc[w['race_whole'], 'event_id'].nunique()):,}; joined to "
        f"race_results {int(w['rr_id'].notna().sum()):,} ({asked:,} carried an id, {asked - found:,} of them point "
        f"at no row before the holdout, {int(wrong.sum()):,} at a row of another day, {int(again.sum()):,} at a row "
        f"another selection already holds: all left unjoined)")
    return w


def _pairs_frame(w: pd.DataFrame, bp: pd.DataFrame) -> pd.DataFrame:
    p = bp.loc[bp["market_type"].eq("place"), PAIR_A + ["event_id", "bsp", "ppwap", "pp_vol", "win_lose"]].rename(
        columns={"event_id": "p_event_id", "bsp": "p_bsp", "ppwap": "p_ppwap", "pp_vol": "p_pp_vol",
                 "win_lose": "p_win_lose"})                              # K4: the place files' four sound columns
    n_place = len(p)
    pdup = p.duplicated(PAIR_A, keep="first")
    wdup = w.duplicated(PAIR_A, keep="first")
    mk = p.groupby("p_event_id", sort=False).agg(k=("p_win_lose", lambda s: int((s == 1).sum())),
                                                 inv=("p_bsp", lambda s: float((1.0 / s.dropna()).sum())))
    x = w[~wdup].merge(p[~pdup], on=PAIR_A, how="inner")
    if x.empty:
        say(f"  a_market PAIR-A: no data found ({n_place:,} place rows, none pairs with a win row)")
    x["k"] = x["p_event_id"].map(mk["k"]).astype(float)
    inv = x["p_event_id"].map(mk["inv"]).astype(float)
    g = x.groupby("event_id", sort=False)
    one = g["p_event_id"].transform("nunique").eq(1)                     # one place market a win market
    both = (x["bsp"].notna() & x["p_bsp"].notna()).groupby(x["event_id"], sort=False).transform("sum")
    x["n_win_bsp"] = x["n_bsp"]
    x["place_whole"] = one & both.eq(x["n_bsp"]) & (x["n_bsp"] > 0)
    x["k_ok"] = one & x["k"].between(2, 4) & x["k"].eq(np.round(inv))
    x["p_place"] = np.nan
    say(f"  a_market PAIR-A: {len(w):,} win rows and {n_place:,} place rows in, {len(x):,} pairs in "
        f"{x['event_id'].nunique():,} races out ({int(wdup.sum()):,} win and {int(pdup.sum()):,} place rows with a "
        f"repeated key dropped; win rows with no place row {len(w) - int(wdup.sum()) - len(x):,}); races with more "
        f"than one place market {int(x.loc[~one, 'event_id'].nunique()):,} (not read); every runner with a win BSP "
        f"has a place BSP in {int(x.loc[x['place_whole'], 'event_id'].nunique()):,}; k passes K9 in "
        f"{int(x.loc[x['k_ok'], 'event_id'].nunique()):,}")
    return x


def place_ready(pairs: pd.DataFrame) -> pd.Series:
    """The races K9's place chance is computed for (S069's filters): 5+ runners with a win BSP, every one of them
    with a place BSP and a win ppwap, and k passing K9. The rows are the race's runners (a BSP in both markets)."""
    return (pairs["place_whole"] & pairs["k_ok"] & pairs["q_pp"].notna() & pairs["bsp"].notna()
            & pairs["p_bsp"].notna() & (pairs["n_win_bsp"] >= MIN_PLACE_FIELD))


def add_place_chance(pairs: pd.DataFrame, progress: int = 20000) -> pd.DataFrame:
    """K9: p_place = core.place_prob(q, k), once a race, q the normalised 1 / win ppwap. A copy is returned."""
    out = pairs.copy()
    ok = place_ready(out)
    p = np.full(len(out), np.nan)
    pos = np.flatnonzero(ok.to_numpy())
    t0, done = time.time(), 0
    if len(pos):
        sub = out.iloc[pos]
        q_all, k_all = sub["q_pp"].to_numpy(float), sub["k"].to_numpy(float)
        for idx in sub.groupby("event_id", sort=False).indices.values():
            q, k = q_all[idx], int(k_all[idx[0]])
            if abs(q.sum() - 1.0) > 1e-6 or k >= len(q):
                continue
            p[pos[idx]] = np.clip(core.place_prob(q, k), 1e-6, 1.0 - 1e-6)
            done += 1
            if progress and done % progress == 0:
                say(f"    K9 place chance: {done:,} races ({time.time() - t0:.0f}s)")
    out["p_place"] = p
    say(f"  a_market K9 place chance: {done:,} races, {int(np.isfinite(p).sum()):,} runners "
        f"({time.time() - t0:.0f}s)")
    return out


def add_run_styles(runs: pd.DataFrame) -> pd.DataFrame:
    """The K8 run-style columns from model/race_shape.py: each runner's projected early position from the comments
    of its EARLIER runs (point in time: the day's own comments never enter a row), and the race's shape."""
    from model import race_shape as RS
    need = ["race_date", "race_time", "track", "horse_name", "number_of_runners", "race_type", "surface_type",
            "dist_furlongs", "comment"]
    d = runs[need].copy()
    d["comment"] = d["comment"].where(d["comment"].ne(""), np.nan).astype(object)
    d = RS.add_race_shape(RS.add_style_projection(RS.add_run_styles(d)))
    out = runs.copy()
    for c in STYLE_COLS:
        out[c] = d[c].to_numpy(float)
    return out


def _runs_frame(rr: pd.DataFrame, w: pd.DataFrame, styles: bool) -> pd.DataFrame:
    j = w.loc[w["rr_id"].notna(), ["rr_id"] + WIN_TO_RUNS].copy()
    j["rr_id"] = j["rr_id"].astype(np.int64)
    runs = rr.merge(j, left_on="id", right_on="rr_id", how="left").drop(columns=["rr_id"])
    assert len(runs) == len(rr), "the win rows' join repeated a race_results row"
    since = runs["race_date"] >= START
    say(f"  a_market runs: {len(runs):,} race_results rows, {int(since.sum()):,} from {START}; with a win price "
        f"record {int(runs['bp_id'].notna().sum()):,} "
        f"({runs.loc[since, 'bp_id'].notna().mean() if since.any() else float('nan'):.1%} of the rows from {START})")
    for c in STYLE_COLS:
        runs[c] = np.nan
    if styles and len(runs):
        t0 = time.time()
        try:
            runs = add_run_styles(runs)
            say(f"  a_market run styles: p_lead on {int(runs['p_lead'].notna().sum()):,} rows, a parsed earlier "
                f"run behind {int((runs['style_n_eff'] > 0).sum()):,} ({time.time() - t0:.0f}s)")
        except Exception as exc:  # noqa: BLE001 -- the strategies that need the columns then report the error
            say(f"  a_market run styles: NOT BUILT ({type(exc).__name__}: {exc}); S092, S094, S096, S098 will error")
    return runs


def build_frames(rr: pd.DataFrame, bp: pd.DataFrame, place_chance: bool = True, styles: bool = True) -> dict:
    """The a_market frames from the two raw tables (see the module docstring). Rows in and out are printed at each
    step with what was dropped and why."""
    t0 = time.time()
    rr, bp = _clean_results(rr), _clean_prices(bp)
    if bp.empty or not bp["market_type"].eq("win").any():
        raise NoDataFound(f"no data found: betfair_prices holds no UK/IE win row from {START} before the holdout")
    w = _win_frame(bp, rr)
    pairs = _pairs_frame(w, bp)
    if place_chance:
        try:
            pairs = add_place_chance(pairs)
        except Exception as exc:  # noqa: BLE001 -- the place-chance strategies then report the error
            say(f"  a_market K9 place chance: NOT BUILT ({type(exc).__name__}: {exc}); S069-S071, S076 will error")
    runs = _runs_frame(rr, w, styles)
    days = sorted(w["race_date"].unique().tolist())
    frames = {"win": w, "pairs": pairs, "runs": runs, "days": days}
    for name in ("win", "pairs", "runs"):
        core.assert_no_holdout(frames[name], what=f"a_market[{name}]")
    say(f"  a_market: {len(days):,} race days {days[0]} .. {days[-1]}; built in {time.time() - t0:.0f}s")
    return frames


def load_a_market() -> dict:
    """The a_market data set from horse_racing.db (see the module docstring)."""
    db = db_path()
    say(f"  a_market: reading {db}")
    rr, bp = read_tables(db)
    if rr.empty or bp.empty:
        raise NoDataFound(f"no data found: race_results {len(rr):,} rows, betfair_prices {len(bp):,} rows before "
                          "the holdout")
    return build_frames(rr, bp)
