"""The price-file data sets of the 100-strategy harness (racing², conventions K1-K10, registered 24 Nov 2026).

    load_a_model()    the served blend's out-of-sample prices on the development window, joined to Betfair's price
                      files and the results (the a_model builder's; S053-S068)
    load_a_market()   Betfair's price files alone, 2018 on (the a_market builder's; appended at the foot of the file)

Both read horse_racing.db read-only and drop every row dated 2026-03-28 or later before anything else is done with it
(K1: the locked holdout is not read). Nothing here reaches Betfair, HRB or S3, and nothing here places a bet. The only
network call is GitHub's artifact API for the research loop's out-of-sample files (as research/queries/done/
closing_refit_stn_1121.py makes it), and only inside load_a_model().

Environment (all optional):
    STRAT100_DB             the database (default horse_racing.db in the working directory)
    STRAT100_WORK           where the filtered out-of-sample files are written (default out/strat100)
    STRAT100_OOS_DIR        a folder that already holds oos_gate_stn.csv (and oos_gate_st.csv): no download
    STRAT100_OOS_RUN        the research-loop run holding oos_gate_stn.csv (default 44715810929, iteration 131)
    STRAT100_OOS_PREFIX     its artifact's name prefix (default research-iter131)
    STRAT100_OOS_RUN_ST     the run holding oos_gate_st.csv where the first does not (default 44473514515, iter 130)
    STRAT100_OOS_PREFIX_ST  its artifact's name prefix (default research-iter130)
    STRAT100_STEAMER=0      skip the steamer fit (S060, S061 and S053-MAIN then report 'no data found')
"""
from __future__ import annotations

import gc
import io
import os
import sqlite3
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from research.strat100 import core

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KEY = ["race_date", "track", "race_time", "horse_name"]            # a runner in race_results and the model's files
PAIR_A = ["country", "menu_hint", "event_dt", "selection_id"]      # K4: a win row and its place row


class NoData(RuntimeError):
    """'no data found': a source the harness needs is missing or empty (never replaced by invented rows)."""


# ---------------------------------------------------------------------------------------------------------------
# shared private helpers (both loaders)
# ---------------------------------------------------------------------------------------------------------------

def _say(*a) -> None:
    print(*a, flush=True)


def _db_path() -> str:
    """The database's path, or NoData. It is on the GitHub runner, not on a development machine."""
    p = Path(os.environ.get("STRAT100_DB") or "horse_racing.db")
    if not p.exists() or p.stat().st_size == 0:
        raise NoData(f"no data found: the database {p} is not here (run scripts/setup_session.sh --db on the runner)")
    return str(p)


def _connect(db: str) -> sqlite3.Connection:
    """Read-only: the raw inputs are never written."""
    return sqlite3.connect(f"file:{Path(db).resolve()}?mode=ro", uri=True)


def _read_sql(conn: sqlite3.Connection, sql: str, params: tuple = (), what: str = "query") -> pd.DataFrame:
    try:
        out = pd.read_sql_query(sql, conn, params=params)
    except Exception as exc:  # a missing table or column is 'no data found', said plainly
        raise NoData(f"no data found: {what} could not be read ({type(exc).__name__}: {exc})") from exc
    _say(f"   {what}: {len(out):,} rows read")
    return out


def _num(df: pd.DataFrame, cols) -> pd.DataFrame:
    """A copy with ``cols`` coerced to numbers (TEXT columns, blanks and 'None' become NaN)."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _iso(s: pd.Series) -> pd.Series:
    return s.astype(str).str[:10]


# ---------------------------------------------------------------------------------------------------------------
# a_model: the model's out-of-sample prices on the price files
# ---------------------------------------------------------------------------------------------------------------

OOS_RUN = int(os.environ.get("STRAT100_OOS_RUN") or 44715810929)            # iteration 131 (ledger iter131-figure-member)
OOS_PREFIX = os.environ.get("STRAT100_OOS_PREFIX") or "research-iter131"
OOS_RUN_ST = int(os.environ.get("STRAT100_OOS_RUN_ST") or 44473514515)      # iteration 130 (iter130-members-new-blocks)
OOS_PREFIX_ST = os.environ.get("STRAT100_OOS_PREFIX_ST") or "research-iter130"
OOS_COLUMNS = ["race_date", "race_time", "track", "horse_name", "predicted_bfsp", "won"]     # scripts/clv_betfair needs
STEAM_CUTOFF = "2024-11-30"       # S060: the CLEAN steamer fitted once on 2018-01-01..2024-11-30
MIN_TRAIN_RACES = 500             # race_book.expected_clv_walk_forward's own default

WIN_SQL = """
    SELECT b.id AS bp_id, b.country, b.race_date AS bf_race_date, b.event_id, b.menu_hint, b.event_dt,
           b.selection_id, b.win_lose, b.bsp AS bf_bsp, b.ppwap, b.morningwap AS bf_morningwap, b.ppmax, b.ppmin,
           b.ipmax, b.ipmin, b.morning_vol AS bf_morning_vol, b.pp_vol, b.ip_vol, b.race_results_id,
           r.race_date, r.track, r.race_time, r.horse_name, r.race_name, r.race_class AS rr_race_class,
           r.race_code AS rr_race_code, r.surface_type AS rr_surface_type, r.number_of_runners AS rr_runners
    FROM betfair_prices b LEFT JOIN race_results r ON r.id = b.race_results_id
    WHERE b.market_type = 'win' AND b.race_date >= ? AND b.race_date < ?
"""
PLACE_SQL = """
    SELECT country, race_date, event_id AS place_event_id, menu_hint, event_dt, selection_id,
           win_lose AS place_won, bsp AS place_bsp, ppwap AS place_ppwap, pp_vol AS place_pp_vol
    FROM betfair_prices
    WHERE market_type = 'place' AND race_date >= ? AND race_date < ?
"""
HIST_SQL = "SELECT horse_name, race_date FROM race_results WHERE bfsp > 1 AND race_date < ?"
WIN_NUM = ["bp_id", "event_id", "selection_id", "win_lose", "bf_bsp", "ppwap", "bf_morningwap", "ppmax", "ppmin",
           "ipmax", "ipmin", "bf_morning_vol", "pp_vol", "ip_vol", "race_results_id", "rr_runners"]
PLACE_NUM = ["place_event_id", "selection_id", "place_won", "place_bsp", "place_ppwap", "place_pp_vol"]
#: scripts/clv_betfair.load's own derived columns that read the BSP or the result: not carried into the frames
CB_DERIVED = ["pred_move", "net", "hold_morning", "hold_bsp", "mins", "meeting_order", "day_order", "overlay_pct"]


def kind_of(text: pd.Series) -> np.ndarray:
    """research/queries/done/race_types_1120.kind_of, word for word (that file is a query that runs on import, so it
    cannot be imported; tests/test_strat100_a_model.py holds the two patterns to its text)."""
    s = text.fillna("").astype(str).str.lower()
    hcap = s.str.contains(r"handicap|hcap|h'cap|nursery")
    green = s.str.contains(r"novice|nov\b|maiden|mdn|beginner|beg\b|juvenile|juv\b|nh flat|nhf|bumper|inhf")
    return np.where(hcap, "handicap", np.where(green, "novice/maiden/bumper", "other"))


def _artifact_zips(run: int, prefix: str, cache: dict) -> list:
    """[(artifact name, ZipFile)] of a run's artifacts whose name starts with ``prefix`` (GitHub's API, the token
    from GITHUB_TOKEN, as the research queries fetch them). Kept in ``cache`` so one run is downloaded once."""
    if (run, prefix) in cache:
        return cache[(run, prefix)]
    import requests
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY", "tomwilson41986/ashcroft")
    hd = {"Authorization": f"Bearer {token}"} if token else {}
    arts, page = [], 1
    while True:
        resp = requests.get(f"https://api.github.com/repos/{repo}/actions/runs/{run}/artifacts?per_page=100&page={page}",
                            headers=hd, timeout=60)
        if resp.status_code != 200:
            raise NoData(f"no data found: run {run}'s artifacts could not be listed (HTTP {resp.status_code}; "
                         f"GITHUB_TOKEN {'set' if token else 'not set'})")
        got = resp.json().get("artifacts", [])
        arts += got
        if len(got) < 100:
            break
        page += 1
    out = []
    for a in [a for a in arts if str(a.get("name", "")).startswith(prefix)]:
        z = requests.get(a["archive_download_url"], headers=hd, timeout=900)
        if z.status_code != 200:
            raise NoData(f"no data found: artifact {a['name']} of run {run} could not be downloaded "
                         f"(HTTP {z.status_code}; expired artifacts cannot be fetched)")
        out.append((a["name"], zipfile.ZipFile(io.BytesIO(z.content))))
    cache[(run, prefix)] = out
    return out


def _read_oos(buf) -> pd.DataFrame:
    """One out-of-sample file as a frame with the holdout dropped before anything else (K1)."""
    o = pd.read_csv(buf, dtype={"race_time": str})           # "2.30" must not become 2.3
    if "race_date" not in o.columns:
        raise NoData("no data found: the out-of-sample file has no race_date column")
    n = len(o)
    o = core.drop_holdout(o.assign(race_date=_iso(o["race_date"])))
    missing = [c for c in OOS_COLUMNS if c not in o.columns]
    if missing:
        raise NoData(f"no data found: the out-of-sample file lacks {missing}")
    o.attrs["rows_in"], o.attrs["rows_holdout"] = n, n - len(o)
    return o


def _oos_frame(file: str, sources: list, cache: dict) -> tuple[pd.DataFrame, str]:
    """(the file's rows dated before 2026-03-28, where it was read from). ``sources`` are (run, artifact prefix)
    pairs tried in order; STRAT100_OOS_DIR, if it holds the file, is read instead of any download."""
    local = os.environ.get("STRAT100_OOS_DIR")
    if local and (Path(local) / file).exists():
        o, src = _read_oos(Path(local) / file), f"local file {Path(local) / file} (STRAT100_OOS_DIR)"
    else:
        o, src, tried = None, "", []
        for run, prefix in sources:
            for name, zf in _artifact_zips(run, prefix, cache):
                hit = [n for n in zf.namelist() if os.path.basename(n) == file]
                if hit:
                    o, src = _read_oos(io.BytesIO(zf.read(hit[0]))), f"run {run}, artifact {name}, {hit[0]}"
                    break
            tried.append(f"run {run} ({prefix}*)")
            if o is not None:
                break
        if o is None:
            raise NoData(f"no data found: {file} is in none of {', '.join(tried)}")
    if o.empty:
        raise NoData(f"no data found: {file} ({src}) holds no row before {core.HOLDOUT_FROM}")
    _say(f"   {file}: {src}; {o.attrs.get('rows_in', len(o)):,} rows, {o.attrs.get('rows_holdout', 0):,} dated "
         f"{core.HOLDOUT_FROM} or later dropped unread, {len(o):,} kept ({o['race_date'].min()} to "
         f"{o['race_date'].max()})")
    return o, src


def _whole_on_prices(pred_path: str, db: str, tag: str) -> pd.DataFrame:
    """One out-of-sample file joined to Betfair's win prices by scripts/clv_betfair.load(until=the holdout), whole
    races only (race_book.whole_races), in the order the walk-forward closing model reads them."""
    from model import race_book as rb
    from scripts import clv_betfair as cb
    d = core.drop_holdout(cb.load(pred_path, db, None, until=core.HOLDOUT_FROM))
    n_pred = len(pd.read_csv(pred_path, usecols=["race_date"]))
    if d.empty:
        raise NoData(f"no data found: {tag} joined to betfair_prices (win) gives no runner with a morning price, a "
                     f"BSP and a forecast")
    w = rb.whole_races(d, field=d.groupby("race")["field"].first())
    w = w.sort_values(["race_date", "race", "predicted_bfsp"], kind="stable").reset_index(drop=True)
    _say(f"   {tag}: {n_pred:,} forecast runners; {len(d):,} joined to a win row with a morning price, a BSP and a "
         f"forecast above 1 ({n_pred - len(d):,} dropped: no joined win row or a missing price); whole races "
         f"{w['race'].nunique():,} of {d['race'].nunique():,} ({len(d) - len(w):,} runners dropped: a runner of the "
         f"race lacks a price, or under 3 runners), {len(w):,} runners, {w['race_date'].min() if len(w) else '-'} "
         f"to {w['race_date'].max() if len(w) else '-'}")
    return w


def _ev_walk_forward(w: pd.DataFrame, min_train_races: int = MIN_TRAIN_RACES) -> pd.Series:
    """Each runner's expected CLV for a back at the morning WAP, E[morning / BSP] - 1, from the closing model fitted
    on the months BEFORE its own (race_book.expected_clv_walk_forward), the traded volume read as none in its inputs
    as closing_refit_stn_1121.py does (the live feed gives none). NaN in a month that is only fitted on. A runner's
    value uses its race's morning prices and the model's, and earlier months' BSPs: nothing of its own month's close."""
    from model import race_book as rb
    if w.empty:
        return pd.Series(np.nan, index=w.index, dtype=float)
    s = rb.expected_clv_walk_forward(w.assign(morning_vol=0.0), min_train_races=min_train_races)
    return s["ev"].reindex(w.index).astype(float)


def _n_bsp(runners: pd.DataFrame, hist: pd.DataFrame) -> pd.Series:
    """S055: for each runner, the count of its horse's EARLIER race_results rows (same horse_name, race_date strictly
    before) with a Betfair SP above 1. ``hist``: horse_name, race_date of every such row."""
    q = runners[["horse_name", "race_date"]].drop_duplicates()
    h = hist[hist["horse_name"].isin(set(q["horse_name"]))]
    m = q.merge(h.rename(columns={"race_date": "hist_date"}), on="horse_name", how="left")
    m["earlier"] = (m["hist_date"].astype(str) < m["race_date"].astype(str)) & m["hist_date"].notna()
    n = m.groupby(["horse_name", "race_date"], sort=False)["earlier"].sum().rename("n_bsp").reset_index()
    out = runners[["horse_name", "race_date"]].merge(n, on=["horse_name", "race_date"], how="left")["n_bsp"]
    return pd.Series(out.fillna(0).to_numpy(int), index=runners.index)


def _place_frame(place: pd.DataFrame) -> pd.DataFrame:
    """The place rows as S068 reads them (K4: place bsp, ppwap, pp_vol and win_lose only), each with its market's
    places paid k = the count of rows with win_lose 1, and k_ok: k is 2-4 and equals round(sum of 1 / place bsp)
    (K9). Rows that share a PAIR-A key are dropped (the pairing would be a guess) and counted."""
    cols = ["race_date"] + PAIR_A + ["place_event_id", "place_bsp", "place_ppwap", "place_pp_vol", "place_won"]
    if place is None or place.empty:
        return pd.DataFrame(columns=cols + ["k_places", "k_ok"])
    p = core.drop_holdout(_num(place, PLACE_NUM).assign(race_date=_iso(place["race_date"])))
    dup = p.duplicated(PAIR_A, keep=False)
    p = p[~dup].copy()
    mk = ["country", "place_event_id"]
    by = [p[c] for c in mk]
    k = (p["place_won"] == 1).groupby(by).transform("sum")
    book = (1.0 / p["place_bsp"].where(p["place_bsp"] > 1)).groupby(by).transform("sum")
    p["k_places"] = k.astype(float)
    p["k_ok"] = k.between(2, 4) & (book.round() == k)
    _say(f"   place rows: {len(place):,} in, {int(dup.sum()):,} dropped (a PAIR-A key used twice), {len(p):,} kept in "
         f"{p[mk].drop_duplicates().shape[0]:,} markets; places paid 2-4 and equal to the book's round sum in "
         f"{p.loc[p['k_ok'], mk].drop_duplicates().shape[0]:,} markets")
    return p[cols + ["k_places", "k_ok"]].reset_index(drop=True)


def _assemble_a_model(w: pd.DataFrame, win: pd.DataFrame | None = None, w_st: pd.DataFrame | None = None,
                      hist: pd.DataFrame | None = None, place: pd.DataFrame | None = None,
                      steam: pd.DataFrame | None = None, meta: dict | None = None,
                      min_train_races: int = MIN_TRAIN_RACES) -> dict:
    """The a_model frames from what was read (no I/O here, so the tests hold it on toy frames).

    ``w``     gate_stn's whole races on the price files (_whole_on_prices): one row a runner
    ``win``   every Betfair win row of the window with its race_results fields (WIN_SQL)
    ``w_st``  the same as ``w`` for gate_st (S059), or None
    ``hist``  horse_name, race_date of every earlier result with a Betfair SP (S055), or None
    ``place`` the place rows (PLACE_SQL; S068), or None
    ``steam`` KEY + steam_score, steam_field, steam_code (S060, S061), or None

    Returns {'runners', 'place', 'days', 'meta'}. ``runners`` is one row a runner of a whole race:
      known at the morning entry   race_date, race, track, race_time, horse_name, month, n_field, predicted_bfsp,
                                   morningwap, morning_vol, ev, exp_bsp, ev_st, predicted_bfsp_st, n_bsp, code, kind,
                                   race_name, race_code, race_class, cls, surface_type, country, steam_score,
                                   steam_field, steam_code, and the PAIR-A key
      known only later             ppwap, pp_vol, ppmax, ppmin (to the off); bsp, win_lose, ipmax, ipmin, ip_vol,
                                   won, bfsp (the result); k6_ok (K6: no selection of the Betfair market lacks a BSP)
    """
    from research import harness
    meta = dict(meta or {})
    problems = list(meta.get("problems", []))
    r = core.drop_holdout(w).reset_index(drop=True)
    r = r.drop(columns=[c for c in CB_DERIVED if c in r.columns])
    r = _num(r, ["predicted_bfsp", "morningwap", "morning_vol", "bsp"])
    r["month"] = r["race_date"].astype(str).str[:7]
    r["n_field"] = r.groupby("race")["horse_name"].transform("size")
    r["ev"] = _ev_walk_forward(r, min_train_races)
    r["exp_bsp"] = r["morningwap"] / (1.0 + r["ev"])           # 1 / E[1 / BSP]: 1 - morningwap / exp_bsp = -ev
    scored = sorted(r.loc[r["ev"].notna(), "month"].unique())
    _say(f"   closing model walk-forward (volume read as none): months scored "
         f"{scored[0] if scored else '-'} to {scored[-1] if scored else '-'} ({len(scored)}); "
         f"{int(r['ev'].notna().sum()):,} runners scored, {int(r['ev'].isna().sum()):,} in months only fitted on")

    # ---- Betfair's win rows: the later columns, the race's fields, K6
    extra = ["country", "event_id", "menu_hint", "event_dt", "selection_id", "win_lose", "ppwap", "pp_vol", "ppmax",
             "ppmin", "ipmax", "ipmin", "ip_vol", "race_results_id", "race_name", "rr_race_class", "rr_race_code",
             "rr_surface_type", "rr_runners", "bf_morningwap", "bf_bsp", "market_no_bsp"]
    if win is not None and len(win):
        b = _num(win, WIN_NUM)
        b = core.drop_holdout(b.assign(bf_race_date=_iso(b["bf_race_date"])), "bf_race_date")
        mk = [b["country"], b["event_id"]]
        b["market_no_bsp"] = (~(b["bf_bsp"] > 1)).groupby(mk).transform("sum")
        m = b[b["race_results_id"].notna() & b["horse_name"].notna()].copy()
        m = core.drop_holdout(m.assign(race_date=_iso(m["race_date"]), race_time=m["race_time"].astype(str)))
        m = m.sort_values("bp_id", kind="stable")
        dup = m.duplicated(KEY)
        m = m[~dup]
        n0 = len(r)
        r = r.merge(m[KEY + extra], on=KEY, how="left", validate="m:1")
        assert len(r) == n0
        with np.errstate(invalid="ignore"):
            same = np.isclose(r["morningwap"], r["bf_morningwap"]) & np.isclose(r["bsp"], r["bf_bsp"])
        r["k6_ok"] = pd.Series(same & (r["market_no_bsp"] == 0)).groupby(r["race"]).transform("all").to_numpy(bool)
        _say(f"   Betfair win rows: {len(b):,} in the window, {len(m) + int(dup.sum()):,} joined to a result, "
             f"{int(dup.sum()):,} dropped (a second row for one runner); runners with their own row "
             f"{int(same.sum()):,} of {len(r):,}; K6 (no selection of the market lacks a BSP, every runner's row "
             f"found): {r.loc[r['k6_ok'], 'race'].nunique():,} of {r['race'].nunique():,} races kept for bets "
             f"({r.loc[~r['k6_ok'], 'race'].nunique():,} dropped)")
    else:
        problems.append("no data found: Betfair's win rows (the later columns, the race's fields and K6)")
        for c in extra:
            r[c] = np.nan
        r["k6_ok"] = False
    for c, src in (("race_code", "rr_race_code"), ("surface_type", "rr_surface_type"), ("race_class", "rr_race_class")):
        own = r[c] if c in r.columns else pd.Series(np.nan, index=r.index, dtype=object)
        r[c] = r[src].where(r[src].notna(), own)
    r = r.drop(columns=["rr_race_code", "rr_surface_type", "rr_race_class", "bf_morningwap", "bf_bsp"])
    r["code"] = harness.code_of(r)                                                   # 'jumps' | 'aw' | 'flat turf'
    r["kind"] = kind_of(r["race_name"])
    r["cls"] = pd.to_numeric(r["race_class"].astype(str).str.extract(r"(\d)")[0], errors="coerce")
    r["win_lose"] = pd.to_numeric(r["win_lose"], errors="coerce")

    # ---- gate_st's expected CLV (S059)
    r["ev_st"], r["predicted_bfsp_st"] = np.nan, np.nan
    if w_st is not None and len(w_st):
        s = core.drop_holdout(w_st).reset_index(drop=True)
        s = _num(s, ["predicted_bfsp", "morningwap", "morning_vol", "bsp"])
        s["ev_st"] = _ev_walk_forward(s, min_train_races)
        s = s.rename(columns={"predicted_bfsp": "predicted_bfsp_st"}).drop_duplicates(KEY)
        r = r.drop(columns=["ev_st", "predicted_bfsp_st"]).merge(s[KEY + ["ev_st", "predicted_bfsp_st"]], on=KEY,
                                                                 how="left", validate="m:1")
        _say(f"   gate_st: {int(r['ev_st'].notna().sum()):,} of {int(r['ev'].notna().sum()):,} scored runners also "
             f"scored under gate_st")
    elif not any("oos_gate_st" in p for p in problems):
        problems.append("no data found: oos_gate_st.csv (S059)")

    # ---- earlier Betfair SPs (S055)
    if hist is not None and len(hist):
        h = core.drop_holdout(hist.assign(race_date=_iso(hist["race_date"])))
        r["n_bsp"] = _n_bsp(r, h)
        _say(f"   earlier Betfair SPs: {len(h):,} result rows with a BSP; runners with under 3 earlier: "
             f"{int((r['n_bsp'] < 3).sum()):,} of {len(r):,}")
    else:
        problems.append("no data found: race_results rows with a BSP (S055)")
        r["n_bsp"] = np.nan

    # ---- the steamer's score (S060, S061)
    for c in ("steam_score", "steam_field", "steam_code"):
        r[c] = np.nan
    if steam is not None and len(steam):
        s = core.drop_holdout(steam.assign(race_date=_iso(steam["race_date"]), race_time=steam["race_time"].astype(str)))
        s = s.drop_duplicates(KEY)
        r = r.drop(columns=["steam_score", "steam_field", "steam_code"]).merge(
            s[KEY + ["steam_score", "steam_field", "steam_code"]], on=KEY, how="left", validate="m:1")
        _say(f"   steamer score: {int(r['steam_score'].notna().sum()):,} of {len(r):,} runners (main codes only)")
    elif not any("steamer" in p for p in problems):
        problems.append("no data found: the steamer's score (S060, S061, S053-MAIN)")

    pl = _place_frame(place)
    if pl.empty:
        problems.append("no data found: Betfair's place rows (S068)")
    days = sorted(r["race_date"].astype(str).str[:10].unique())
    meta.update({"problems": problems, "scored_months": scored, "holdout_from": core.HOLDOUT_FROM,
                 "runners": len(r), "races": int(r["race"].nunique()), "days": len(days)})
    core.assert_no_holdout(r, what="a_model[runners]")
    core.assert_no_holdout(pl, what="a_model[place]")
    return {"runners": r.reset_index(drop=True), "place": pl, "days": days, "meta": meta}


def _steamer_scores(db: str, first_day: str, cutoff: str = STEAM_CUTOFF) -> tuple[pd.DataFrame, dict]:
    """The CLEAN steamer model (research/paper/morning_features.train_steamer) fitted once on 2018-01-01..``cutoff``
    and its score for every main-code runner from ``first_day`` on, with the 20th, 70th and 95th percentiles of the
    fit's own training scores. The table is strat_lib.load()'s with the holdout dropped before a feature is built;
    every CLEAN feature is a card fact or an earlier day's (morning_features.build, strat_lib.prior_cumsum)."""
    if not first_day > cutoff:
        raise NoData(f"the steamer's fit (to {cutoff}) would overlap the scored days (from {first_day})")
    from research.paper import morning_features as MF
    L = MF.L
    df = core.drop_holdout(L.load(db))
    booster, thr95, n_fit = MF.train_steamer(df, cutoff=cutoff)
    x = MF.build(df)
    del df
    gc.collect()
    tr = L.priced(x)
    tr = tr[(tr["race_date"] >= "2018-01-01") & (tr["race_date"] <= cutoff) & tr["steam"].notna()]
    scores = booster.predict(tr[MF.CLEAN])
    p20, p70, p95 = (float(v) for v in np.quantile(scores, [0.20, 0.70, 0.95]))
    sc = x[x["race_date"] >= first_day]
    out = sc[KEY].assign(steam_score=booster.predict(sc[MF.CLEAN]) if len(sc) else np.nan,
                         steam_field=pd.to_numeric(sc["field"], errors="coerce"), steam_code=sc["code"].astype(str))
    info = {"steam_cutoff": cutoff, "steam_rows_fit": int(n_fit), "steam_p20": p20, "steam_p70": p70,
            "steam_p95": p95, "steam_threshold_check": float(thr95)}
    _say(f"   steamer: CLEAN model fitted on 2018-01-01..{cutoff}, {n_fit:,} rows ({len(tr):,} re-scored); training "
         f"scores' 20th / 70th / 95th percentile {p20:+.4f} / {p70:+.4f} / {p95:+.4f}; {len(out):,} runners scored "
         f"from {first_day}")
    del x, tr
    gc.collect()
    return out.reset_index(drop=True), info


def load_a_model() -> dict:
    """The a_model data set: {'runners', 'place', 'days', 'meta'} (see _assemble_a_model). Every row is dated before
    2026-03-28; the run and artifact read are printed and kept in meta['sources']."""
    _say("a_model: the served blend's out-of-sample prices on Betfair's price files (development window; K1)")
    db = _db_path()
    work = Path(os.environ.get("STRAT100_WORK") or "out/strat100")
    work.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    meta: dict = {"sources": {}, "problems": []}

    stn, src = _oos_frame("oos_gate_stn.csv", [(OOS_RUN, OOS_PREFIX)], cache)
    meta["sources"]["gate_stn"] = src
    p_stn = work / "oos_gate_stn_dev.csv"
    stn.to_csv(p_stn, index=False)
    w = _whole_on_prices(str(p_stn), db, "gate_stn")
    if w.empty:
        raise NoData("no data found: gate_stn has no whole race on the price files before the holdout")
    first_day = str(w["race_date"].min())[:10]

    w_st = None
    try:
        st, src_st = _oos_frame("oos_gate_st.csv", [(OOS_RUN, OOS_PREFIX), (OOS_RUN_ST, OOS_PREFIX_ST)], cache)
        meta["sources"]["gate_st"] = src_st
        p_st = work / "oos_gate_st_dev.csv"
        st.to_csv(p_st, index=False)
        w_st = _whole_on_prices(str(p_st), db, "gate_st")
    except Exception as exc:  # noqa: BLE001 -- only S059 needs it
        meta["problems"].append(f"no data found: oos_gate_st.csv (S059): {type(exc).__name__}: {exc}")
        _say(f"   gate_st NOT LOADED: {type(exc).__name__}: {exc}")
    cache.clear()

    conn = _connect(db)
    try:
        win = _read_sql(conn, WIN_SQL, (first_day, core.HOLDOUT_FROM), "betfair_prices win rows with their results")
        try:
            place = _read_sql(conn, PLACE_SQL, (first_day, core.HOLDOUT_FROM), "betfair_prices place rows")
        except NoData as exc:
            place = None
            meta["problems"].append(str(exc))
        try:
            hist = _read_sql(conn, HIST_SQL, (core.HOLDOUT_FROM,), "race_results rows with a Betfair SP")
        except NoData as exc:
            hist = None
            meta["problems"].append(str(exc))
    finally:
        conn.close()

    steam = None
    if os.environ.get("STRAT100_STEAMER", "1") == "0":
        meta["problems"].append("no data found: the steamer's score (S060, S061, S053-MAIN): STRAT100_STEAMER=0")
    else:
        try:
            steam, info = _steamer_scores(db, first_day)
            meta.update(info)
        except Exception as exc:  # noqa: BLE001 -- only S060, S061 and S053-MAIN need it
            meta["problems"].append(f"no data found: the steamer's score (S060, S061, S053-MAIN): "
                                    f"{type(exc).__name__}: {exc}")
            _say(f"   steamer NOT BUILT: {type(exc).__name__}: {exc}")

    frames = _assemble_a_model(w, win, w_st, hist, place, steam, meta, min_train_races=MIN_TRAIN_RACES)
    m = frames["meta"]
    _say(f"a_model loaded: {m['runners']:,} runners in {m['races']:,} whole races on {m['days']} race days "
         f"{frames['days'][0]} to {frames['days'][-1]}; sources: " + "; ".join(f"{k} = {v}" for k, v in m["sources"].items()))
    for p in m["problems"]:
        _say(f"   PROBLEM {p}")
    return frames


# ---------------------------------------------------------------------------------------------------------------
# a_market (the a_market builder's; S069-S100): the loader's own code is research/strat100/data_a_market.py
# ---------------------------------------------------------------------------------------------------------------

def load_a_market() -> dict:
    """Betfair's price files (win and place) joined to race_results by race_results_id, 2018-01-01 to 2026-03-27,
    no model: {'win', 'pairs', 'runs', 'days'} as research/strat100/data_a_market.py describes them. The holdout is
    left out in the SQL and again by core.drop_holdout on each table before anything else; rows in and out are
    printed at every join. A missing database or table raises data_a_market.NoDataFound ('no data found: ...'),
    which the runner records against the a_market strategies and goes on."""
    from research.strat100 import data_a_market
    return data_a_market.load_a_market()
