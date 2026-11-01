"""The open-to-close data inventory (the data agent, 1 Nov 2026; the owner's ask of 1 Nov: autonomous trading of the
move from the opening price to the close, in whichever markets carry one). Read-only.

What is held, where, for a trade that backs early and lays at the close:

A. horse_racing.db: ``betfair_prices`` (Betfair's price files: morning WAP and volume, pre-play WAP and volume, the
   BSP, the in-play range) by country and market by year; ``betfair_live_marks`` (the recorder's books at 08:00-12:00
   UK and T-120..T-1, the last book, the settled one; from 1 Oct 2026): coverage a day, the marks held, the money
   matched at each, and the 08:00 mid against the BSP by price band (the first market-only drift read);
B. s3://$PRED_BUCKET/sources/: Betfair's Historic Data tables (first traded, T-24h/T-60/T-15/T-1, close, BSP, result),
   the football gold tables (the benchmark open and close a market, by season and by which book supplies each), tennis
   (closing odds only: no open exists there), GBGB, the greyhound price-file tables, openfootball, API-Football;
C. the census of every sport (betfair_live/<day>/census_sports.json, Oct 2026): sport x market type, markets a day,
   matched money on the top markets; and the other recordings under betfair_live/<day>/ (tags, days, sizes);
D. from the Historic Data tables: the first-traded-to-close move by price band, country and market type.

Every section stands on its own (one failing cannot hide the rest). The end prints OPEN_CLOSE_INVENTORY_JSON = {...}
with the key counts for the modelling agent. Column names come from betfair_prices.ensure_table,
betfair_recorder.MARK_FIELDS, betfair_historic._MarketState.rows, football.data.benchmarks, sources/greyhound_prices.py.
Run by .github/workflows/research-query.yml (PRED_BUCKET, read-only AWS keys, horse_racing.db beside it).
"""

from __future__ import annotations

import io
import json
import os
import re
import sqlite3
import sys
import time
import traceback
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)

DB = os.environ.get("OC_DB") or "horse_racing.db"
BUCKET = os.environ.get("PRED_BUCKET") or os.environ.get("CAPTURE_BUCKET") or "ashcroft"
REGION = os.environ.get("PRED_REGION") or os.environ.get("CAPTURE_REGION") or "eu-west-2"
CENSUS_MONTH = os.environ.get("OC_CENSUS_MONTH") or "2026-10"
BANDS = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 51.0, 1001.0]
SUMMARY: dict = {"run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "bucket": BUCKET, "db": DB}
_T0 = time.time()


def section(title: str):
    """Run one section; print its header; never let its failure stop the next."""
    def deco(fn):
        def run(*a, **kw):
            print(f"\n{'=' * 100}\n== {title}\n{'=' * 100}")
            t = time.time()
            try:
                return fn(*a, **kw)
            except Exception as exc:                           # noqa: BLE001 - the point of the wrapper
                tb = traceback.format_exc().strip().splitlines()
                print(f"!! {title}: FAILED ({type(exc).__name__}: {str(exc)[:300]})")
                print("   " + "\n   ".join(tb[-3:]))
                SUMMARY.setdefault("failed_sections", []).append(title)
                return None
            finally:
                print(f"-- {title}: {time.time() - t:.0f} s")
        return run
    return deco


def show(df, index=False, head=None) -> None:
    if df is None or len(df) == 0:
        print("  (empty)")
        return
    d = df.head(head) if head else df
    print("  " + d.to_string(index=index).replace("\n", "\n  "))


def q25(x):
    return float(np.nanpercentile(x, 25)) if len(x) else float("nan")


def q75(x):
    return float(np.nanpercentile(x, 75)) if len(x) else float("nan")


def move_table(d: pd.DataFrame, a: str, b: str, by=None, min_n: int = 30) -> pd.DataFrame:
    """The move from price ``a`` to price ``b`` (both decimal, > 1), by the band of ``a`` (and ``by``): n, median
    log(b/a) (negative: shortened), the share that shortened, the IQR and the std of the log move."""
    x = d[(d[a] > 1) & (d[b] > 1)].copy()
    if x.empty:
        return pd.DataFrame()
    x["band"] = pd.cut(x[a], BANDS, right=False)
    x["lr"] = np.log(x[b] / x[a])
    x["short"] = (x[b] < x[a]).astype(float)
    keys = ([by] if isinstance(by, str) else list(by or [])) + ["band"]
    g = x.groupby(keys, observed=True)
    out = g.agg(n=("lr", "size"), med_a=(a, "median"), med_log_move=("lr", "median"), mean_log_move=("lr", "mean"),
                shortened=("short", "mean"), iqr_lo=("lr", q25), iqr_hi=("lr", q75), sd=("lr", "std"))
    out = out[out["n"] >= min_n].round(3).reset_index()
    out["band"] = out["band"].astype(str)
    return out


# --------------------------------------------------------------------------------------------------------------------
# S3
# --------------------------------------------------------------------------------------------------------------------

_S3 = None


def s3():
    global _S3
    if _S3 is None:
        import boto3
        from botocore.config import Config
        _S3 = boto3.client("s3", region_name=REGION, config=Config(connect_timeout=10, read_timeout=120,
                                                                     retries={"max_attempts": 4}))
    return _S3


def list_keys(prefix: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for page in s3().get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            out[o["Key"]] = o["Size"]
    return out


def get_bytes(key: str) -> bytes:
    return s3().get_object(Bucket=BUCKET, Key=key)["Body"].read()


def get_parquet(key: str, columns=None) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(get_bytes(key)), columns=columns)


def store():
    from sources.common import Store
    return Store(s3=s3(), bucket=BUCKET)


# --------------------------------------------------------------------------------------------------------------------
# A. horse_racing.db
# --------------------------------------------------------------------------------------------------------------------

def conn_ro() -> sqlite3.Connection:
    if not os.path.exists(DB):
        raise FileNotFoundError(f"{DB} not found in {os.getcwd()}")
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


@section("A0. horse_racing.db: the tables and their sizes")
def a0_tables():
    c = conn_ro()
    tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
    want = ["race_results", "betfair_prices", "betfair_prices_files", "betfair_live_marks", "betfair_live_markets",
            "live_orders", "betfair_odds"]
    rows = []
    for t in want:
        if t not in tables:
            rows.append({"table": t, "held": False})
            continue
        n = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        cols = [r[1] for r in c.execute(f"PRAGMA table_info({t})")]
        dcol = "race_date" if "race_date" in cols else None
        rng = c.execute(f"SELECT MIN({dcol}), MAX({dcol}) FROM {t}").fetchone() if dcol else (None, None)
        rows.append({"table": t, "held": True, "rows": n, "first": rng[0], "last": rng[1], "columns": len(cols)})
        SUMMARY.setdefault("db_tables", {})[t] = {"rows": int(n), "first": rng[0], "last": rng[1]}
    show(pd.DataFrame(rows))
    print(f"  other tables: {sorted(set(tables) - set(want))}")
    for t in ("betfair_prices", "betfair_live_marks", "betfair_live_markets"):
        if t in tables:
            print(f"  {t} columns: {[r[1] for r in c.execute(f'PRAGMA table_info({t})')]}")
    c.close()


@section("A1. betfair_prices (Betfair's price files) by country, market and year")
def a1_prices():
    c = conn_ro()
    sql = """
        SELECT country, LOWER(market_type) AS market_type, substr(race_date, 1, 4) AS year,
               COUNT(*) AS runners, COUNT(DISTINCT event_id) AS markets,
               SUM(bsp > 1) AS with_bsp, SUM(morningwap > 1) AS with_morning_wap, SUM(ppwap > 1) AS with_pp_wap,
               SUM(morning_vol > 0) AS with_morning_vol, SUM(ipmin > 0) AS with_inplay,
               SUM(win_lose IN (0, 1)) AS with_result, SUM(race_results_id IS NOT NULL) AS joined_to_results
        FROM betfair_prices GROUP BY 1, 2, 3 ORDER BY 1, 2, 3"""
    g = pd.read_sql_query(sql, c)
    show(g)
    SUMMARY["betfair_prices"] = {"rows": int(g["runners"].sum()), "markets": int(g["markets"].sum()),
                                 "by_country_market": {f"{r.country}_{r.market_type}": int(r.runners)
                                                       for r in g.groupby(["country", "market_type"])["runners"]
                                                       .sum().reset_index().itertuples()},
                                 "years": sorted(g["year"].dropna().unique().tolist())}
    # medians: the morning and pre-play money and the BSP, by market and year (win markets carry the trade)
    med = pd.read_sql_query("""
        SELECT country, LOWER(market_type) AS market_type, substr(race_date, 1, 4) AS year,
               morningwap, morning_vol, ppwap, pp_vol, ip_vol, bsp FROM betfair_prices
        WHERE bsp > 1""", c)
    c.close()
    m = med.groupby(["country", "market_type", "year"]).agg(
        n=("bsp", "size"), med_bsp=("bsp", "median"), med_morning_wap=("morningwap", "median"),
        med_morning_vol=("morning_vol", "median"), p90_morning_vol=("morning_vol", lambda s: s.quantile(0.9)),
        med_pp_vol=("pp_vol", "median"), p90_pp_vol=("pp_vol", lambda s: s.quantile(0.9)),
        med_ip_vol=("ip_vol", "median")).round(2).reset_index()
    print("\n  medians where the BSP is held (GBP matched a runner: morning = before the day's main trading, pp = pre-play):")
    show(m)
    # the morning WAP against the BSP, by morning-price band, win markets, by country (the files' drift read)
    w = med[(med["market_type"] == "win") & (med["morningwap"] > 1)].copy()
    w["period"] = np.where(w["year"] >= "2024", "2024+", "to 2023")
    print("\n  win markets: morning WAP -> BSP by morning-price band and period (med_log_move < 0: shortened):")
    show(move_table(w, "morningwap", "bsp", by=["country", "period"], min_n=200))
    print("\n  win markets: pre-play WAP -> BSP by pre-play band (how much is left to move at the end):")
    show(move_table(w[w["ppwap"] > 1], "ppwap", "bsp", by=["country"], min_n=200))


@section("A2. betfair_live_marks (the recorder's books from 1 Oct 2026): coverage, marks, money matched")
def a2_marks():
    c = conn_ro()
    cov = pd.read_sql_query("""
        SELECT race_date, COUNT(DISTINCT market_id) AS markets, COUNT(DISTINCT market_id || '|' || selection_id) AS runners,
               SUM(mark = 'clock_08:00') AS m0800, SUM(mark = 'clock_10:00') AS m1000, SUM(mark = 'clock_12:00') AS m1200,
               SUM(mark = 'off_120') AS t120, SUM(mark = 'off_60') AS t60, SUM(mark = 'off_15') AS t15,
               SUM(mark = 'off_1') AS t1, SUM(mark = 'last') AS last_, SUM(mark = 'final') AS final_,
               SUM(mark = 'final' AND bsp > 1) AS final_with_bsp
        FROM betfair_live_marks GROUP BY race_date ORDER BY race_date""", c)
    show(cov)
    marks = pd.read_sql_query("""
        SELECT mark, COUNT(*) AS rows_, COUNT(DISTINCT race_date) AS days, AVG(minutes_to_off) AS mean_min_to_off,
               SUM(back1 > 1 AND lay1 > 1) AS with_two_sided_book, SUM(last_traded > 1) AS with_last_traded,
               SUM(bsp > 1) AS with_bsp
        FROM betfair_live_marks GROUP BY mark ORDER BY mean_min_to_off DESC""", c)
    print("\n  the marks held (rows a mark, runner-level):")
    show(marks)
    SUMMARY["betfair_live_marks"] = {"days": int(len(cov)), "first": cov["race_date"].min() if len(cov) else None,
                                     "last": cov["race_date"].max() if len(cov) else None,
                                     "rows_by_mark": {r.mark: int(r.rows_) for r in marks.itertuples()}}
    d = pd.read_sql_query("""
        SELECT race_date, market_id, selection_id, mark, minutes_to_off, market_total_matched, back1, lay1,
               back1_size, lay1_size, last_traded, bsp, active_runners, source
        FROM betfair_live_marks""", c)
    mt = pd.read_sql_query("SELECT market_id, market_type, country, venue FROM betfair_live_markets "
                           "GROUP BY market_id", c)
    c.close()
    d = d.merge(mt, on="market_id", how="left")
    d["mid"] = np.where((d["back1"] > 1) & (d["lay1"] > 1), (d["back1"] + d["lay1"]) / 2.0, np.nan)
    d["spread_ticks_pct"] = np.where(d["mid"].notna(), (d["lay1"] - d["back1"]) / d["mid"], np.nan)
    print("\n  money matched on the MARKET at each mark (GBP, median and 90th percentile a market), by market type:")
    mm = d.drop_duplicates(["race_date", "market_id", "mark"]).groupby(["market_type", "mark"], observed=True).agg(
        markets=("market_id", "size"), med_matched=("market_total_matched", "median"),
        p90_matched=("market_total_matched", lambda s: s.quantile(0.9)),
        med_min_to_off=("minutes_to_off", "median")).round(1).reset_index()
    mm = mm.sort_values(["market_type", "med_min_to_off"], ascending=[True, False])
    show(mm)
    print("\n  the top of the book at each mark (runner-level): share with both sides, median best-back size, median relative spread:")
    bk = d.groupby(["market_type", "mark"], observed=True).agg(
        runners=("selection_id", "size"), two_sided=("mid", lambda s: s.notna().mean()),
        med_back1_size=("back1_size", "median"), med_lay1_size=("lay1_size", "median"),
        med_rel_spread=("spread_ticks_pct", "median"), med_min_to_off=("minutes_to_off", "median")).round(3).reset_index()
    show(bk.sort_values(["market_type", "med_min_to_off"], ascending=[True, False]))
    return d


@section("A3. betfair_live_marks: the 08:00 mid against the BSP by price band (the first market-only drift read)")
def a3_drift(d: pd.DataFrame | None):
    if d is None or d.empty:
        print("  no marks frame (A2 failed or the table is empty)")
        return
    key = ["race_date", "market_id", "selection_id"]
    fin = d[(d["mark"] == "final") & (d["bsp"] > 1)][key + ["bsp"]].drop_duplicates(key)
    print(f"  runners with a BSP on the settled mark: {len(fin):,} over {fin['race_date'].nunique()} days")
    SUMMARY["betfair_live_marks"]["runners_with_bsp"] = int(len(fin))
    for mark in ("clock_08:00", "clock_10:00", "clock_12:00", "off_120", "off_60", "off_15", "off_1", "last"):
        m = d[(d["mark"] == mark) & d["mid"].notna()][key + ["mid", "last_traded", "market_type", "country"]]
        x = m.merge(fin, on=key)
        if x.empty:
            print(f"\n  {mark}: no runner with both a two-sided book and a BSP")
            continue
        print(f"\n  {mark} mid -> BSP, {len(x):,} runners, by market type and mid band (med_log_move < 0: shortened):")
        show(move_table(x, "mid", "bsp", by="market_type", min_n=30))
        if mark == "clock_08:00":
            SUMMARY["betfair_live_marks"]["runners_0800_with_bsp"] = int(len(x))
            print("\n  the same by country (win markets):")
            show(move_table(x[x["market_type"] == "WIN"], "mid", "bsp", by="country", min_n=30))
            # a back at the 08:00 mid laid at the BSP: the greened trade's gross per GBP1 backed, before commission
            w = x[x["market_type"] == "WIN"].copy()
            w["green_per_pound"] = w["mid"] / w["bsp"] - 1.0        # a lay at the BSP for the same liability
            w["band"] = pd.cut(w["mid"], BANDS, right=False)
            g = w.groupby("band", observed=True)["green_per_pound"].agg(["size", "mean", "median"]).round(4)
            g.index = g.index.astype(str)
            print("\n  win: a back at the 08:00 mid laid off at the BSP (mid/BSP - 1 a GBP1 stake, before commission, "
                  "the mid as if takeable):")
            show(g.reset_index())


@section("A4. betfair_live_markets (the recorder's catalogue): markets, types, countries")
def a4_markets():
    c = conn_ro()
    g = pd.read_sql_query("""
        SELECT substr(race_date, 1, 7) AS month, market_type, country, COUNT(DISTINCT market_id) AS markets,
               COUNT(*) AS runners, SUM(race_results_id IS NOT NULL) AS joined_to_results
        FROM betfair_live_markets GROUP BY 1, 2, 3 ORDER BY 1, 2, 3""", c)
    c.close()
    show(g)
    SUMMARY["betfair_live_markets"] = {"markets": int(g["markets"].sum()), "runners": int(g["runners"].sum())}


# --------------------------------------------------------------------------------------------------------------------
# B. sources/ in S3
# --------------------------------------------------------------------------------------------------------------------

HIST_COLS = ["market_id", "event_id", "event_type_id", "market_type", "country", "market_time", "went_in_play_utc",
             "number_of_winners", "bsp_market", "selection_id", "runner_status", "won", "bsp", "ltp_first",
             "ltp_t_1440", "ltp_t_60", "ltp_t_15", "ltp_t_1", "ltp_close"]


@section("B1. sources/betfair_historic/: Betfair's Historic Data (raw files and the built tables)")
def b1_historic():
    from betfair_historic import path_year
    keys = list_keys("sources/betfair_historic/")
    raw = {k: v for k, v in keys.items() if "/raw/" in k}
    by = defaultdict(lambda: {"files": 0, "mb": 0.0})
    for k, v in raw.items():
        m = re.match(r"sources/betfair_historic/raw/([^/]+)/", k)
        sport = m.group(1) if m else "?"
        y = path_year(k)
        by[(sport, y)]["files"] += 1
        by[(sport, y)]["mb"] += v / 1e6
    inv = pd.DataFrame([{"sport": s, "year": y, "raw_files": v["files"], "raw_mb": round(v["mb"], 1)}
                        for (s, y), v in by.items()]).sort_values(["sport", "year"]) if by else pd.DataFrame()
    print(f"  {len(raw):,} raw files, {sum(raw.values()) / 1e9:.2f} GB; other keys: "
          f"{sorted(k.rsplit('/', 1)[-1] for k in keys if '/raw/' not in k)}")
    show(inv)
    tables = sorted(k for k in keys if re.search(r"/markets_[a-z_]+_\d{4}\.parquet$", k))
    SUMMARY["betfair_historic"] = {"raw_files": len(raw), "tables": {}}
    frames = []
    for k in tables:
        df = get_parquet(k)
        name = k.rsplit("/", 1)[-1]
        have = [c for c in HIST_COLS if c in df.columns]
        miss = [c for c in HIST_COLS if c not in df.columns]
        nn = {c: int(df[c].notna().sum()) for c in ("ltp_first", "ltp_t_1440", "ltp_t_60", "ltp_t_15", "ltp_t_1",
                                                      "ltp_close", "bsp", "won") if c in df.columns}
        row = {"table": name, "runners": len(df), "markets": int(df["market_id"].nunique()),
               "events": int(df["event_id"].nunique()) if "event_id" in df else None,
               "first_off": str(df["market_time"].min())[:10] if "market_time" in df else None,
               "last_off": str(df["market_time"].max())[:10] if "market_time" in df else None, **nn}
        print(f"\n  {name}: {len(df):,} runners, {row['markets']:,} markets, offs {row['first_off']} to {row['last_off']}"
              + (f"; columns missing: {miss}" if miss else ""))
        print(f"    non-null: {nn}")
        if "country" in df and "market_type" in df:
            ct = df.drop_duplicates("market_id").groupby(["country", "market_type"], observed=True).size()
            print("    markets by country and type: " + ", ".join(f"{a}/{b}={n:,}" for (a, b), n in ct.items()))
        if "went_in_play_utc" in df:
            print(f"    markets that went in play (a close exists): "
                  f"{df.drop_duplicates('market_id')['went_in_play_utc'].notna().mean():.1%}")
        SUMMARY["betfair_historic"]["tables"][name] = {k2: v for k2, v in row.items() if k2 != "table"}
        frames.append(df[have].assign(table=name))
    if not tables:
        print("  no markets_<sport>_<year>.parquet table held: no data found yet")
    return pd.concat(frames, ignore_index=True) if frames else None


@section("B2. sources/football/gold/: the benchmark open and close by market and season")
def b2_football():
    st = store()
    keys = st.listing("football/")
    print("  files: " + ", ".join(f"{k} ({v / 1e6:.1f} MB)" for k, v in sorted(keys.items()) if k.endswith((".parquet", ".json"))
                                 and "/raw/" not in k))
    g = st.get_parquet("football/gold/matches.parquet")
    if g is None:
        print("  football/gold/matches.parquet: no data found yet")
        return
    print(f"  gold matches: {len(g):,} rows, match_date {g['match_date'].min()} to {g['match_date'].max()}, "
          f"status: {g['status'].value_counts().to_dict()}")
    print(f"  columns ({len(g.columns)}): {list(g.columns)}")
    fin = g[g["status"] == "FINISHED"].copy()
    fin["season"] = fin["season"].astype(int)
    cov = fin.groupby("season").agg(
        matches=("match_id", "size"), competitions=("competition_id", "nunique"),
        open_1x2=("open_h", lambda s: s.notna().mean()), close_1x2=("close_h", lambda s: s.notna().mean()),
        open_ou25=("open_o25", lambda s: s.notna().mean()), close_ou25=("close_o25", lambda s: s.notna().mean()),
        open_ah=("open_ah_h", lambda s: s.notna().mean()), close_ah=("close_ah_h", lambda s: s.notna().mean()),
        bfe_open=("bfe_open_h", lambda s: s.notna().mean()), bfe_close=("bfe_close_h", lambda s: s.notna().mean()),
        kickoff_utc=("kickoff_utc", lambda s: s.notna().mean())).round(3).reset_index()
    print("\n  coverage by season (share of finished matches with the benchmark price):")
    show(cov)
    recent = fin[fin["season"] >= int(fin["season"].max()) - 11]
    for col in ("open_1x2_src", "close_1x2_src", "open_ou_src", "close_ou_src", "open_ah_src", "close_ah_src"):
        if col in fin:
            t = recent.groupby(["season", col], observed=True).size().unstack(fill_value=0)
            print(f"\n  {col}: which bookmaker supplies it, matches by season (the last 12 seasons):")
            show(t.reset_index())
    both = fin[(fin["open_h"] > 1) & (fin["close_h"] > 1)]
    SUMMARY["football_gold"] = {"matches": int(len(g)), "finished": int(len(fin)),
                                "with_open_and_close_1x2": int(len(both)),
                                "seasons": [int(fin["season"].min()), int(fin["season"].max())] if len(fin) else None}
    print(f"\n  finished matches with both an open and a close (1X2): {len(both):,}")
    # the open-to-close move on the home price, by open band (the first football drift read)
    print("\n  1X2 home price: benchmark open -> close by open band (med_log_move < 0: the home side shortened):")
    show(move_table(both, "open_h", "close_h", min_n=200))
    print("\n  over 2.5: benchmark open -> close by open band:")
    show(move_table(fin, "open_o25", "close_o25", min_n=200))
    for k in ("football/gold/odds_long.parquet", "football/gold/betfair_market_map.parquet"):
        df = st.get_parquet(k)
        if df is None:
            print(f"\n  {k}: no data found yet")
            continue
        print(f"\n  {k}: {len(df):,} rows, columns {list(df.columns)[:24]}")
        if "ltp_first" in df:
            nn = {c: int(df[c].notna().sum()) for c in ("ltp_first", "ltp_t_60", "ltp_t_1", "ltp_close", "won")
                  if c in df}
            print(f"    non-null: {nn}; market types: {df['market_type'].value_counts().to_dict() if 'market_type' in df else '?'}")
            SUMMARY["football_gold"]["betfair_market_map_rows"] = int(len(df))
        elif "phase" in df and "bookmaker" in df:
            t = df.groupby(["market", "phase"], observed=True)["bookmaker"].nunique() if "market" in df else None
            print(f"    bookmakers a (market, phase): {t.to_dict() if t is not None else '?'}")
    dq = st.get("football/dq/latest.json")
    if dq:
        try:
            j = json.loads(dq)
            print(f"\n  dq/latest.json: {json.dumps(j, default=str)[:1200]}")
        except ValueError:
            print("\n  dq/latest.json unreadable")


@section("B3. sources/tennis/: Sackmann/TML statistics and tennis-data.co.uk closing odds (is there an open?)")
def b3_tennis():
    st = store()
    keys = st.listing("tennis/")
    tables = {k: v for k, v in keys.items() if k.endswith(".parquet")}
    raw_n = sum(1 for k in keys if "/raw/" in k)
    print(f"  raw files: {raw_n:,}; tables: { {k: f'{v / 1e6:.1f} MB' for k, v in tables.items()} }")
    SUMMARY["tennis"] = {"raw_files": raw_n, "tables": {}}
    m = st.get_parquet("tennis/sackmann_matches.parquet")
    if m is not None:
        dcol = "tourney_date" if "tourney_date" in m else None
        print(f"  sackmann_matches: {len(m):,} rows; {dcol} {m[dcol].min() if dcol else '?'} to {m[dcol].max() if dcol else '?'}; "
              f"columns: {list(m.columns)[:30]} ...")
        for c in ("tour", "level", "source"):
            if c in m:
                print(f"    {c}: {m[c].value_counts().head(8).to_dict()}")
        SUMMARY["tennis"]["tables"]["sackmann_matches"] = int(len(m))
    else:
        print("  sackmann_matches.parquet: no data found yet")
    o = st.get_parquet("tennis/odds_matches.parquet")
    if o is None:
        print("  odds_matches.parquet (tennis-data.co.uk): no data found yet (its Cloudflare refuses GitHub's runners; "
              "the UK server's fetch builds it)")
        SUMMARY["tennis"]["odds_matches"] = 0
        return
    print(f"  odds_matches (tennis-data.co.uk): {len(o):,} rows, {o['match_date'].min()} to {o['match_date'].max()}")
    print(f"    columns: {list(o.columns)}")
    odds_cols = [c for c in o.columns if re.fullmatch(r"(CB|GB|IW|SB|B365|B&W|EX|PS|UB|LB|SJ|Max|Avg)[WL]", str(c))]
    nn = {c: int(o[c].notna().sum()) for c in odds_cols}
    print(f"    odds columns and how many matches carry each: {nn}")
    opening = [c for c in o.columns if re.search(r"open|opening|early", str(c), re.I)]
    print(f"    columns that name an OPENING price: {opening or 'NONE: tennis-data.co.uk publishes closing prices only '}"
          f"{'(Pinnacle PSW/PSL, Bet365, the market maximum and average; Betfair EXW/EXL to 2014)' if not opening else ''}")
    if "PSW" in o and "B365W" in o:
        x = o[(o["PSW"] > 1) & (o["B365W"] > 1)]
        print(f"    Pinnacle close vs Bet365 close on the winner, {len(x):,} matches: median log(B365/PS) "
              f"{float(np.log(x['B365W'] / x['PSW']).median()):+.4f} (two closes, not an open)")
    yr = o.groupby(o["match_date"].astype(str).str[:4]).agg(matches=("match_date", "size"),
                                                           with_pinnacle=("PSW", lambda s: int(s.notna().sum())) if "PSW" in o
                                                           else ("match_date", "size"))
    print("    by year:")
    show(yr.reset_index(), head=40)
    SUMMARY["tennis"]["odds_matches"] = int(len(o))
    SUMMARY["tennis"]["opening_odds_columns"] = opening


@section("B4. sources/gbgb/ and sources/greyhound_prices/: results and the greyhound price-file tables")
def b4_greyhounds():
    st = store()
    keys = st.listing("gbgb/")
    runs = sorted(k for k in keys if re.search(r"gbgb/runs_\d{4}\.parquet$", k))
    raw_n = sum(1 for k in keys if "/raw/" in k)
    print(f"  gbgb: {raw_n:,} raw day files; tables: {[k.rsplit('/', 1)[-1] for k in runs]}")
    rows = []
    for k in runs:
        df = st.get_parquet(k)
        dcol = next((c for c in ("race_date", "date", "meeting_date") if c in df.columns), None)
        rows.append({"table": k.rsplit("/", 1)[-1], "rows": len(df),
                     "races": int(df["race_id"].nunique()) if "race_id" in df else None,
                     "first": str(df[dcol].min())[:10] if dcol else None, "last": str(df[dcol].max())[:10] if dcol else None})
    show(pd.DataFrame(rows))
    SUMMARY["gbgb"] = {"raw_files": raw_n, "runs_rows": {r["table"]: int(r["rows"]) for r in rows}}
    pk = sorted(k for k in st.listing("greyhound_prices/") if re.search(r"prices_\d{4}\.parquet$", k))
    print(f"\n  greyhound_prices tables: {[k.rsplit('/', 1)[-1] for k in pk]}")
    out, frames = [], []
    for k in pk:
        df = st.get_parquet(k)
        if df.empty:
            out.append({"table": k.rsplit("/", 1)[-1], "rows": 0})
            continue
        win = df[df["market"] == "win"]
        out.append({"table": k.rsplit("/", 1)[-1], "rows": len(df), "win_markets": int(win["event_id"].nunique()),
                    "place_markets": int(df[df["market"] == "place"]["event_id"].nunique()),
                    "first": df["race_date"].min(), "last": df["race_date"].max(),
                    "countries": df.drop_duplicates(["market", "event_id"])["country"].value_counts().to_dict(),
                    "with_bsp": int((df["bsp"] > 1).sum()), "with_morning_wap": int((df["morningwap"] > 1).sum()),
                    "med_morning_vol": round(float(win["morning_vol"].median()), 1),
                    "med_pp_vol": round(float(win["pp_vol"].median()), 1),
                    "with_result": int(df["win_lose"].isin([0, 1]).sum())})
        frames.append(win[["race_date", "country", "morningwap", "ppwap", "bsp", "morning_vol", "pp_vol", "win_lose"]])
    show(pd.DataFrame(out))
    SUMMARY["greyhound_prices"] = {r["table"]: int(r["rows"]) for r in out}
    if frames:
        w = pd.concat(frames, ignore_index=True)
        w["year"] = w["race_date"].astype(str).str[:4]
        print("\n  greyhound win: morning WAP -> BSP by morning band and country (the files' drift; the morning WAP is "
              "an average of what traded, not a takeable price):")
        show(move_table(w, "morningwap", "bsp", by="country", min_n=200))
        print("\n  greyhound win: the morning money by country and year (GBP a runner, median / 90th pct; share with any):")
        g = w.groupby(["country", "year"]).agg(runners=("bsp", "size"), med_morning_vol=("morning_vol", "median"),
                                               p90_morning_vol=("morning_vol", lambda s: s.quantile(0.9)),
                                               any_morning=("morning_vol", lambda s: (s > 0).mean()),
                                               med_pp_vol=("pp_vol", "median")).round(2).reset_index()
        show(g)


@section("B5. sources/openfootball/, sources/api_football/, sources/puntingform/: the other tables")
def b5_others():
    st = store()
    for name, key, dcol in (("openfootball", "openfootball/matches.parquet", "match_date"),
                            ("api_football", "api_football/fixtures.parquet", None),
                            ("puntingform", None, None)):
        keys = st.listing(f"{name}/")
        raw_n = sum(1 for k in keys if "/raw/" in k)
        tables = [k for k in keys if k.endswith(".parquet")]
        print(f"\n  {name}: {raw_n:,} raw files, tables {tables}")
        SUMMARY.setdefault("others", {})[name] = {"raw_files": raw_n, "tables": {}}
        if key and key in keys:
            df = st.get_parquet(key)
            d = dcol or next((c for c in df.columns if "date" in c.lower()), None)
            print(f"    {key}: {len(df):,} rows; {d}: {str(df[d].min())[:10] if d else '?'} to {str(df[d].max())[:10] if d else '?'}; "
                  f"columns: {list(df.columns)[:20]}")
            SUMMARY["others"][name]["tables"][key.rsplit("/", 1)[-1]] = int(len(df))
            odds = [c for c in df.columns if re.search(r"odd|price", str(c), re.I)]
            print(f"    odds columns: {odds or 'none (results and status only: no price here)'}")
        elif key:
            print(f"    {key}: no data found yet")


# --------------------------------------------------------------------------------------------------------------------
# C. the census of every sport and the other recordings
# --------------------------------------------------------------------------------------------------------------------

@section("C1. betfair_live/<day>/: every recording held (tags, days, sizes)")
def c1_recordings():
    keys = list_keys("betfair_live/")
    by_name = defaultdict(list)
    days = set()
    for k, v in keys.items():
        m = re.match(r"betfair_live/(\d{4}-\d{2}-\d{2})/(.+)$", k)
        if not m:
            continue
        days.add(m.group(1))
        by_name[m.group(2)].append((m.group(1), v))
    print(f"  {len(keys):,} objects over {len(days)} days ({min(days) if days else '-'} to {max(days) if days else '-'})")
    rows = []
    for name, lst in sorted(by_name.items()):
        sizes = [v for _, v in lst]
        rows.append({"file": name, "days": len(lst), "first": min(d for d, _ in lst), "last": max(d for d, _ in lst),
                     "total_mb": round(sum(sizes) / 1e6, 1), "median_mb": round(float(np.median(sizes)) / 1e6, 2)})
    show(pd.DataFrame(rows))
    SUMMARY["betfair_live_files"] = {r["file"]: {"days": r["days"], "first": r["first"], "last": r["last"],
                                                 "total_mb": r["total_mb"]} for r in rows}
    return keys


@section("C2. the census of every sport (census_sports.json): sport x market type, markets a day, matched money")
def c2_census(keys: dict | None):
    keys = keys or list_keys(f"betfair_live/{CENSUS_MONTH}")
    cen = sorted(k for k in keys if k.endswith("/census_sports.json") and f"/{CENSUS_MONTH}" in k)
    print(f"  census files in {CENSUS_MONTH}: {len(cen)} days")
    if not cen:
        print("  no census found: no data found yet")
        SUMMARY["census"] = {"days": 0}
        return
    rows, types, hours = [], [], []
    for k in cen:
        day = k.split("/")[1]
        try:
            j = json.loads(get_bytes(k))
        except Exception as exc:                                # noqa: BLE001
            print(f"    {k}: unreadable ({exc})")
            continue
        hours.append(j.get("hours"))
        for r in j.get("sports", []):
            rows.append({"day": day, "sport": r.get("sport"), "event_type_id": r.get("event_type_id"),
                         "markets": r.get("markets"), "top_matched_total": r.get("top_matched_total"),
                         "top_matched_median": r.get("top_matched_median"), "error": r.get("error")})
            for mt, n in (r.get("market_types") or {}).items():
                types.append({"day": day, "sport": r.get("sport"), "market_type": mt, "markets": n,
                              "top_matched": (r.get("top_matched_by_type") or {}).get(mt)})
    S = pd.DataFrame(rows)
    T = pd.DataFrame(types)
    print(f"  window hours: {sorted(set(hours))}; days read: {S['day'].nunique()}; sports seen: {S['sport'].nunique()}")
    by_sport = S.groupby("sport").agg(days=("day", "nunique"), med_markets_day=("markets", "median"),
                                      med_top_total=("top_matched_total", "median"),
                                      max_top_total=("top_matched_total", "max"),
                                      med_top_median=("top_matched_median", "median"),
                                      errors=("error", lambda s: int(s.notna().sum()))).round(0)
    by_sport = by_sport.sort_values("med_top_total", ascending=False)
    print("\n  by sport (a day: markets in the next 24 h; GBP matched by the read on its top 200 markets: total, and the "
          "median market of those 200):")
    show(by_sport.reset_index(), head=45)
    SUMMARY["census"] = {"days": int(S["day"].nunique()), "sports": int(S["sport"].nunique()),
                         "top_sports_by_median_top_total": {k: float(v) for k, v in by_sport["med_top_total"].head(15).items()}}
    # the distribution of matched money: how concentrated the top 200 is (total / median gives the shape)
    bs = by_sport.copy()
    bs["top200_mean_over_median"] = (bs["med_top_total"] / 200.0 / bs["med_top_median"].replace(0, np.nan)).round(1)
    print("\n  concentration: the mean matched of the top 200 over its median (1 = even; large = a few markets carry the money):")
    show(bs[["days", "med_top_total", "med_top_median", "top200_mean_over_median"]].head(25).reset_index())
    top_types = T.groupby(["sport", "market_type"]).agg(days=("day", "nunique"), med_markets_day=("markets", "median"),
                                                         med_top_matched=("top_matched", "median"),
                                                         max_top_matched=("top_matched", "max")).round(0).reset_index()
    top_types = top_types.sort_values("med_top_matched", ascending=False)
    print("\n  sport x market type: markets a day (median) and GBP matched on the type within the top 200 (median a day; "
          "blank: the type never reached the top 200). The 60 best-matched types:")
    show(top_types.head(60))
    print("\n  the main sports in full (every type with markets a day >= 5):")
    main = top_types[top_types["sport"].isin(by_sport.index[:8]) & (top_types["med_markets_day"] >= 5)]
    show(main.sort_values(["sport", "med_top_matched"], ascending=[True, False]), head=120)
    SUMMARY["census"]["sport_market_type_rows"] = int(len(top_types))


# --------------------------------------------------------------------------------------------------------------------
# D. the first-traded-to-close move in Betfair's Historic Data tables
# --------------------------------------------------------------------------------------------------------------------

@section("D. Historic Data: first traded -> close (and the marks) by price band, country and market type")
def d_hist_drift(h: pd.DataFrame | None):
    if h is None or h.empty:
        print("  no Historic Data table held: no data found yet")
        return
    h = h.copy()
    h["sport"] = h["table"].str.extract(r"markets_([a-z_]+)_\d{4}")[0]
    for c in ("ltp_first", "ltp_t_1440", "ltp_t_60", "ltp_t_15", "ltp_t_1", "ltp_close", "bsp"):
        if c in h:
            h[c] = pd.to_numeric(h[c], errors="coerce")
    # a market's runners with a close: the market went in play (or closed) after trading
    base = h[(h["ltp_first"] > 1) & (h["ltp_close"] > 1)]
    print(f"  runners with a first and a closing price: {len(base):,} of {len(h):,}; sports: "
          f"{base['sport'].value_counts().to_dict()}")
    SUMMARY["historic_drift"] = {"runners_with_first_and_close": int(len(base))}
    for sport, hs in base.groupby("sport"):
        print(f"\n  --- {sport}: {len(hs):,} runners, {hs['market_id'].nunique():,} markets ---")
        print("  first traded -> close by first-price band and market type (med_log_move < 0: shortened):")
        show(move_table(hs, "ltp_first", "ltp_close", by="market_type", min_n=100))
        win = hs[hs["market_type"] == "WIN"] if (hs["market_type"] == "WIN").any() else hs
        label = "WIN" if (hs["market_type"] == "WIN").any() else "all types"
        print(f"\n  {label}: first traded -> close by country:")
        show(move_table(win, "ltp_first", "ltp_close", by="country", min_n=100))
        for a, name in (("ltp_t_1440", "T-24h"), ("ltp_t_60", "T-60"), ("ltp_t_15", "T-15"), ("ltp_t_1", "T-1")):
            if a in win and (win[a] > 1).sum() >= 100:
                print(f"\n  {label}: {name} -> close by {name} band (how much of the move is left at {name}):")
                show(move_table(win, a, "ltp_close", min_n=100))
        if "bsp" in win and (win["bsp"] > 1).sum() >= 100:
            print(f"\n  {label}: first traded -> BSP (where the market has one):")
            show(move_table(win, "ltp_first", "bsp", by="country", min_n=100))
        # the favourite-longshot shape of the move: does the shortening concentrate on the shortest?
        if "won" in win and win["won"].isin([0, 1]).any():
            x = win[(win["won"].isin([0, 1])) & (win["ltp_close"] > 1)].copy()
            x["band"] = pd.cut(x["ltp_close"], BANDS, right=False).astype(str)
            g = x.groupby("band", observed=True).agg(n=("won", "size"), won=("won", "mean"),
                                                     implied=("ltp_close", lambda s: float((1 / s).mean()))).round(3)
            print(f"\n  {label}: the close against the result by closing band (calibration of the close):")
            show(g.reset_index())
        # when does the first trade happen? the files carry no timestamp of it: the share of runners whose first price
        # equals their T-24h price says how often trading starts a day or more out
        if "ltp_t_1440" in win:
            t24 = win[(win["ltp_t_1440"] > 1)]
            print(f"\n  {label}: runners with a price standing 24 h before the off: {len(t24):,} of {len(win):,} "
                  f"({len(t24) / max(len(win), 1):.1%}); of those, first == T-24h price: "
                  f"{(t24['ltp_first'] == t24['ltp_t_1440']).mean():.1%}")
        SUMMARY["historic_drift"][sport] = {"runners": int(len(hs)), "markets": int(hs["market_id"].nunique())}


# --------------------------------------------------------------------------------------------------------------------

def main() -> int:
    print(f"open_close_inventory_1101: bucket s3://{BUCKET} ({REGION}), db {DB}, census month {CENSUS_MONTH}")
    a0_tables()
    a1_prices()
    marks = a2_marks()
    a3_drift(marks)
    a4_markets()
    hist = b1_historic()
    b2_football()
    b3_tennis()
    b4_greyhounds()
    b5_others()
    keys = c1_recordings()
    c2_census(keys)
    d_hist_drift(hist)
    SUMMARY["seconds"] = round(time.time() - _T0)
    print(f"\n{'=' * 100}\nOPEN_CLOSE_INVENTORY_JSON = {json.dumps(SUMMARY, default=str, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
