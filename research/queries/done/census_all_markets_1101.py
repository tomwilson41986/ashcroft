"""The complete census of every sport and every market type Betfair listed (the owner's ask, 1 Nov 2026: "Evaluate
all markets on Betfair"; the data agent). Read-only.

Every ``betfair_live/<day>/census_sports.json`` held (the recorder's evening census, ``betfair_recorder.census_sports``:
once a day, every sport with markets starting in the next 24 hours, its market types with counts, the money matched by
then on its 200 busiest markets, by type, and the five busiest). Printed for the research agent to evaluate every row:

1. every sport ever seen: event_type_id, days seen, median and max markets a day, median top-200 matched total and
   median market, and its five busiest events over the month (one row an event, the largest matched kept);
2. every sport x market type row: days seen, median and max markets a day, median and max matched on the type within
   the top 200 (0 where the type never reached the top 200 that day), and the share of the sport's top-200 money the
   type carries over the month;
3. for horse racing and greyhounds the on-the-day profile the evening read misses: the market's matched money at
   08:00 UK, T-60, T-15, T-1 and the last book (``betfair_live_marks`` for horses; the recorded greyhound books,
   ``betfair_live/<day>/books_greyhound.csv.gz``, through ``betfair_recorder.marks``), so the evening figure's bias
   can be stated;
4. the event types seen, with their ids; the runner cannot call listEventTypes, so the rest of Betfair's list is
   for the research agent to take from Betfair's published event types.

Under 2,000 lines; ends with a JSON block between "=== JSON ===" lines holding (1) and (2). Each section stands on
its own. Run by .github/workflows/research-query.yml (PRED_BUCKET, read-only AWS keys, horse_racing.db beside it).
"""

from __future__ import annotations

import gzip
import io
import json
import os
import re
import sqlite3
import sys
import time
import traceback
from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_colwidth", 70)

DB = os.environ.get("OC_DB") or "horse_racing.db"
BUCKET = os.environ.get("PRED_BUCKET") or os.environ.get("CAPTURE_BUCKET") or "ashcroft"
REGION = os.environ.get("PRED_REGION") or os.environ.get("CAPTURE_REGION") or "eu-west-2"
GREYHOUND_DAYS = int(os.environ.get("CENSUS_GREYHOUND_DAYS") or "40")     # the most recent days of the record read
OUT: dict = {"run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "bucket": BUCKET}
_T0 = time.time()
BUSIEST_RE = re.compile(r"^(?P<event>.*?): (?P<market>.*) \((?P<matched>[\d,]+)\)$")


def section(title: str):
    def deco(fn):
        def run(*a, **kw):
            print(f"\n{'=' * 100}\n== {title}\n{'=' * 100}")
            t = time.time()
            try:
                return fn(*a, **kw)
            except Exception as exc:                           # noqa: BLE001
                tb = traceback.format_exc().strip().splitlines()
                print(f"!! {title}: FAILED ({type(exc).__name__}: {str(exc)[:300]})")
                print("   " + "\n   ".join(tb[-3:]))
                OUT.setdefault("failed_sections", []).append(title)
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


# --------------------------------------------------------------------------------------------------------------------
# The census files
# --------------------------------------------------------------------------------------------------------------------

@section("0. The census files held")
def load_census(keys: dict | None = None):
    keys = keys if keys is not None else list_keys("betfair_live/")
    cen = sorted(k for k in keys if k.endswith("/census_sports.json"))
    days = [k.split("/")[1] for k in cen]
    print(f"  {len(cen)} census files: {days[0] if days else '-'} to {days[-1] if days else '-'}")
    sports, types, busiest = [], [], []
    read_utc, hours = [], set()
    for k in cen:
        day = k.split("/")[1]
        try:
            j = json.loads(get_bytes(k))
        except Exception as exc:                                # noqa: BLE001
            print(f"    {k}: unreadable ({exc})")
            continue
        read_utc.append(j.get("read_utc"))
        hours.add(j.get("hours"))
        for r in j.get("sports", []):
            total = float(r.get("top_matched_total") or 0.0)
            sports.append({"day": day, "event_type_id": str(r.get("event_type_id")), "sport": r.get("sport"),
                           "markets": r.get("markets"), "top_total": total,
                           "top_median": float(r.get("top_matched_median") or 0.0),
                           "n_types": len(r.get("market_types") or {}), "error": r.get("error")})
            by_type = r.get("top_matched_by_type") or {}
            for mt, n in (r.get("market_types") or {}).items():
                types.append({"day": day, "sport": r.get("sport"), "market_type": mt, "markets": n,
                              "top_matched": float(by_type.get(mt) or 0.0), "sport_total": total})
            for mt in by_type:                                   # a type matched in the top 200 but missing from the
                if mt not in (r.get("market_types") or {}):      # type list (the two calls read moments apart)
                    types.append({"day": day, "sport": r.get("sport"), "market_type": mt, "markets": np.nan,
                                  "top_matched": float(by_type.get(mt) or 0.0), "sport_total": total})
            for i, s in enumerate(r.get("busiest") or []):
                m = BUSIEST_RE.match(str(s))
                if m:
                    busiest.append({"day": day, "sport": r.get("sport"), "rank": i + 1, "event": m.group("event"),
                                    "market": m.group("market"), "matched": float(m.group("matched").replace(",", ""))})
                else:
                    busiest.append({"day": day, "sport": r.get("sport"), "rank": i + 1, "event": str(s), "market": "",
                                    "matched": np.nan})
    S, T, B = pd.DataFrame(sports), pd.DataFrame(types), pd.DataFrame(busiest)
    times = sorted({str(t)[11:16] for t in read_utc if t})
    print(f"  read at (UTC, distinct): {times}; window hours: {sorted(hours)}; sport-days {len(S):,}; "
          f"type-days {len(T):,}; busiest rows {len(B):,}")
    OUT["census"] = {"files": len(cen), "first_day": days[0] if days else None, "last_day": days[-1] if days else None,
                     "read_utc_times": times, "sport_days": int(len(S)), "type_days": int(len(T))}
    return S, T, B, len(cen)


@section("1. Every sport seen: ids, days, markets a day, the money a day ahead, the five busiest events")
def sports_table(S: pd.DataFrame, B: pd.DataFrame, n_days: int):
    g = S.groupby(["event_type_id", "sport"]).agg(
        days=("day", "nunique"), med_markets_day=("markets", "median"), max_markets_day=("markets", "max"),
        med_top200_total=("top_total", "median"), max_top200_total=("top_total", "max"),
        med_top200_median_market=("top_median", "median"), med_types=("n_types", "median"),
        errors=("error", lambda s: int(s.notna().sum()))).reset_index()
    g = g.sort_values("med_top200_total", ascending=False)
    g["share_of_days"] = (g["days"] / n_days).round(2)
    print(f"  {len(g)} sports over {n_days} census days (GBP matched on the sport's 200 busiest next-day markets by the "
          f"evening read):")
    show(g.round(0))
    OUT["sports"] = [{"event_type_id": r.event_type_id, "sport": r.sport, "days": int(r.days),
                      "med_markets_day": float(r.med_markets_day), "max_markets_day": float(r.max_markets_day),
                      "med_top200_total": float(r.med_top200_total), "max_top200_total": float(r.max_top200_total),
                      "med_top200_median_market": float(r.med_top200_median_market)} for r in g.itertuples()]
    print("\n  the five busiest events a sport over the month (one row an event, its largest matched reading; GBP):")
    if B.empty:
        print("  (no busiest rows)")
        return g
    b = B.dropna(subset=["matched"]).sort_values("matched", ascending=False)
    b = b.drop_duplicates(["sport", "event"], keep="first")
    top5 = {}
    for sport in g["sport"]:
        rows = b[b["sport"] == sport].head(5)
        if rows.empty:
            continue
        print(f"  {sport}:")
        for r in rows.itertuples():
            print(f"      {r.matched:>14,.0f}  {r.day}  {r.event[:60]}  [{r.market[:40]}]")
        top5[sport] = [{"event": r.event, "market": r.market, "matched": float(r.matched), "day": r.day}
                       for r in rows.itertuples()]
    for s in OUT["sports"]:
        s["busiest"] = top5.get(s["sport"], [])
    return g


@section("2. Every sport x market type row")
def types_table(T: pd.DataFrame, S: pd.DataFrame, order):
    # the share of the sport's top-200 money the type carries over the month: the sums over days where the sport
    # was read (a day the type was absent counts 0 for the type, the sport's total still counts)
    sport_total = S.groupby("sport")["top_total"].sum()
    g = T.groupby(["sport", "market_type"]).agg(
        days=("day", "nunique"), med_markets_day=("markets", "median"), max_markets_day=("markets", "max"),
        med_top_matched=("top_matched", "median"), max_top_matched=("top_matched", "max"),
        sum_top_matched=("top_matched", "sum")).reset_index()
    # a type absent on a day the sport was read: its median should count that day as 0
    sport_days = S.groupby("sport")["day"].nunique()

    def med_with_zero_days(row):
        n_sport = int(sport_days.get(row["sport"], row["days"]))
        vals = T[(T["sport"] == row["sport"]) & (T["market_type"] == row["market_type"])]["top_matched"].tolist()
        vals += [0.0] * max(0, n_sport - len(vals))
        return float(np.median(vals)) if vals else 0.0
    g["med_top_matched_sport_days"] = g.apply(med_with_zero_days, axis=1)
    g["share_of_sport_money"] = (g["sum_top_matched"] / g["sport"].map(sport_total).replace(0, np.nan)).fillna(0.0)
    g["ever_in_top200"] = g["max_top_matched"] > 0
    rank = {s: i for i, s in enumerate(order)}
    g["_o"] = g["sport"].map(rank).fillna(len(rank))
    g = g.sort_values(["_o", "med_top_matched", "max_top_matched", "med_markets_day"],
                      ascending=[True, False, False, False]).drop(columns="_o")
    print(f"  {len(g)} rows ({int(g['ever_in_top200'].sum())} types ever in a sport's top 200). Columns: days the "
          "type was listed; markets a day (median, max); GBP matched on the type within the sport's top 200 (median "
          "over the days the type was listed; median over all the sport's days, absent days as 0; max); the type's "
          "share of the sport's top-200 money over the month.")
    out = g[["sport", "market_type", "days", "med_markets_day", "max_markets_day", "med_top_matched",
             "med_top_matched_sport_days", "max_top_matched", "share_of_sport_money"]].copy()
    out["share_of_sport_money"] = out["share_of_sport_money"].round(3)
    for c in ("med_top_matched", "med_top_matched_sport_days", "max_top_matched", "med_markets_day", "max_markets_day"):
        out[c] = out[c].round(0)
    show(out)
    OUT["sport_market_types"] = [
        {"sport": r.sport, "market_type": r.market_type, "days": int(r.days),
         "med_markets_day": None if pd.isna(r.med_markets_day) else float(r.med_markets_day),
         "max_markets_day": None if pd.isna(r.max_markets_day) else float(r.max_markets_day),
         "med_top_matched": float(r.med_top_matched), "max_top_matched": float(r.max_top_matched),
         "share_of_sport_money": round(float(r.share_of_sport_money), 4)} for r in out.itertuples()]
    return g


# --------------------------------------------------------------------------------------------------------------------
# 3. The on-the-day profile the evening read misses
# --------------------------------------------------------------------------------------------------------------------

PROFILE_MARKS = ["clock_08:00", "clock_10:00", "clock_12:00", "off_120", "off_60", "off_15", "off_1", "last"]


@section("3a. Horse racing on the day: the market's matched money at the marks (betfair_live_marks)")
def horse_profile():
    if not os.path.exists(DB):
        print(f"  {DB} not found: no data")
        return
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        d = pd.read_sql_query("""
            SELECT m.race_date, m.market_id, m.mark, m.market_total_matched, k.market_type, k.country
            FROM betfair_live_marks m
            LEFT JOIN (SELECT market_id, MAX(market_type) AS market_type, MAX(country) AS country
                       FROM betfair_live_markets GROUP BY market_id) k ON k.market_id = m.market_id
            WHERE m.mark IN ({})""".format(",".join("?" * len(PROFILE_MARKS))), conn, params=PROFILE_MARKS)
    finally:
        conn.close()
    d = d.drop_duplicates(["race_date", "market_id", "mark"])
    g = d.groupby(["market_type", "country", "mark"]).agg(
        markets=("market_id", "size"), med_matched=("market_total_matched", "median"),
        p90_matched=("market_total_matched", lambda s: s.quantile(0.9))).reset_index()
    g["_o"] = g["mark"].map({m: i for i, m in enumerate(PROFILE_MARKS)})
    g = g.sort_values(["market_type", "country", "_o"]).drop(columns="_o")
    print(f"  {d['race_date'].nunique()} days, {d['market_id'].nunique():,} markets; GBP matched a market (median, 90th pct):")
    show(g.round(0))
    # the day's total against the evening census's figure: the sum over a day's win markets at the last book
    last = d[(d["mark"] == "last") & (d["market_type"] == "WIN")].groupby("race_date")["market_total_matched"].sum()
    print(f"\n  the day's GB+IE WIN markets' matched at the last book, summed a day: median GBP{last.median():,.0f}, "
          f"min GBP{last.min():,.0f}, max GBP{last.max():,.0f} over {len(last)} days (the evening census read "
          f"GBP23,155 at the median on the next day's top 200 Horse Racing markets: run 41291139483)")
    OUT["horse_profile"] = {f"{r.market_type}_{r.country}_{r.mark}": float(r.med_matched) for r in g.itertuples()}
    OUT["horse_profile"]["win_day_total_last_median"] = float(last.median()) if len(last) else None


@section("3b. Greyhounds on the day: the recorded GB/IE books (books_greyhound.csv.gz) at the marks")
def greyhound_profile(keys: dict | None = None):
    import betfair_recorder as br
    keys = keys if keys is not None else list_keys("betfair_live/")
    days = sorted({k.split("/")[1] for k in keys if k.endswith("/books_greyhound.csv.gz")})
    days = days[-GREYHOUND_DAYS:]
    print(f"  {len(days)} days of the greyhound record read ({days[0] if days else '-'} to {days[-1] if days else '-'})")
    rows, firsts, totals = [], [], []
    for day in days:
        try:
            b = pd.read_csv(io.BytesIO(gzip.decompress(get_bytes(f"betfair_live/{day}/books_greyhound.csv.gz"))),
                            dtype={"market_id": str}, on_bad_lines="skip")
            m = pd.read_csv(io.BytesIO(gzip.decompress(get_bytes(f"betfair_live/{day}/markets_greyhound.csv.gz"))),
                            dtype={"market_id": str}, on_bad_lines="skip")
        except Exception as exc:                                # noqa: BLE001
            print(f"    {day}: not read ({exc})")
            continue
        if b.empty or m.empty:
            continue
        marks = br.marks(b, m, date.fromisoformat(day))
        if len(marks):
            mk = marks.drop_duplicates(["market_id", "mark"])
            mk = mk.merge(m.drop_duplicates("market_id")[["market_id", "market_type", "country"]], on="market_id", how="left")
            rows.append(mk[["market_id", "mark", "market_total_matched", "market_type", "country"]].assign(day=day))
            win_last = mk[(mk["mark"] == "last") & (mk["market_type"] == "WIN")]
            totals.append({"day": day, "win_markets": int(win_last["market_id"].nunique()),
                           "win_matched_last_sum": float(pd.to_numeric(win_last["market_total_matched"], errors="coerce").sum())})
        # the first book the record holds a market: how long before the off, and the money then
        offs = m.drop_duplicates("market_id").set_index("market_id")["market_start_utc"]
        bb = b.copy()
        bb["polled"] = pd.to_datetime(bb["polled_utc"], utc=True, errors="coerce")
        bb["off"] = pd.to_datetime(bb["market_id"].map(offs), utc=True, errors="coerce")
        bb = bb[(bb["source"] != "final") & (bb["status"] == "OPEN") & (bb["polled"] < bb["off"])]
        f = bb.sort_values("polled").drop_duplicates("market_id", keep="first")
        f = f.merge(m.drop_duplicates("market_id")[["market_id", "market_type"]], on="market_id", how="left")
        f["min_to_off"] = (f["off"] - f["polled"]).dt.total_seconds() / 60.0
        firsts.append(f[["market_id", "market_type", "min_to_off", "market_total_matched"]].assign(day=day))
    if not rows:
        print("  no greyhound books read: no data")
        return
    R = pd.concat(rows, ignore_index=True)
    R["market_total_matched"] = pd.to_numeric(R["market_total_matched"], errors="coerce")
    g = R.groupby(["market_type", "mark"]).agg(market_days=("market_id", "size"),
                                               med_matched=("market_total_matched", "median"),
                                               p90_matched=("market_total_matched", lambda s: s.quantile(0.9))).reset_index()
    mark_order = [f"clock_{c}" for c in br.CLOCK_MARKS] + [f"off_{o}" for o in br.OFF_MARKS] + ["last", "final"]
    g["_o"] = g["mark"].map({m: i for i, m in enumerate(mark_order)})
    g = g.sort_values(["market_type", "_o"]).drop(columns="_o")
    print(f"  {R['day'].nunique()} days, {R['market_id'].nunique():,} markets; GBP matched a market at each mark:")
    show(g.round(0))
    F = pd.concat(firsts, ignore_index=True)
    F["market_total_matched"] = pd.to_numeric(F["market_total_matched"], errors="coerce")
    fg = F.groupby("market_type").agg(markets=("market_id", "size"), med_min_to_off_first_book=("min_to_off", "median"),
                                      p90_min_to_off_first_book=("min_to_off", lambda s: s.quantile(0.9)),
                                      med_matched_at_first_book=("market_total_matched", "median"),
                                      share_first_book_over_gbp100=("market_total_matched", lambda s: float((s > 100).mean()))
                                      ).reset_index()
    print("\n  the first book the record holds a market (minutes before the off; GBP matched then):")
    show(fg.round(2))
    tot = pd.DataFrame(totals)
    if len(tot):
        print(f"\n  the day's GB/IE greyhound WIN markets' matched at the last book, summed a day: median "
              f"GBP{tot['win_matched_last_sum'].median():,.0f} over {len(tot)} days, {tot['win_markets'].median():.0f} "
              f"markets a day (the evening census read GBP109 at the median on the next day's top 200: run 41291139483)")
    OUT["greyhound_profile"] = {f"{r.market_type}_{r.mark}": float(r.med_matched) for r in g.itertuples()}
    OUT["greyhound_profile"]["days"] = int(R["day"].nunique())
    if len(tot):
        OUT["greyhound_profile"]["win_day_total_last_median"] = float(tot["win_matched_last_sum"].median())


@section("4. The event types seen, and what the census cannot say")
def event_types(S: pd.DataFrame):
    seen = S.groupby(["event_type_id", "sport"])["day"].nunique().reset_index().rename(columns={"day": "days"})
    seen["event_type_id_num"] = pd.to_numeric(seen["event_type_id"], errors="coerce")
    seen = seen.sort_values("event_type_id_num").drop(columns="event_type_id_num")
    print(f"  {len(seen)} event types appeared in at least one census day:")
    show(seen)
    print("\n  The census lists only the sports with a market starting in the next 24 hours at the evening read "
          "(listEventTypes with that time filter: betfair_recorder.census_sports). An event type with no market in "
          "that window on any of the days (a sport out of season, or one Betfair lists only around its events) is "
          "absent here, not absent from Betfair. The runner cannot call listEventTypes; the full list of Betfair's "
          "event types, with ids, is Betfair's published one (the Betting API's listEventTypes without a filter, or the "
          "developer documentation's event type table), and the research agent should take the difference from there.")
    OUT["event_types_seen"] = [{"event_type_id": r.event_type_id, "sport": r.sport, "days": int(r.days)}
                               for r in seen.itertuples()]


def main() -> int:
    print(f"census_all_markets_1101: bucket s3://{BUCKET} ({REGION}), db {DB}")
    keys = None
    try:
        keys = list_keys("betfair_live/")
    except Exception as exc:                                    # noqa: BLE001
        print(f"!! betfair_live/ listing failed ({exc}); the census and greyhound sections will try their own")
    got = load_census(keys)
    if got is not None:
        S, T, B, n_days = got
        g = sports_table(S, B, n_days)
        types_table(T, S, list(g["sport"]) if g is not None else [])
        event_types(S)
    horse_profile()
    greyhound_profile(keys)
    OUT["seconds"] = round(time.time() - _T0)
    print("\n=== JSON ===")
    print(json.dumps(OUT, default=str, sort_keys=True, separators=(",", ":")))
    print("=== JSON ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
