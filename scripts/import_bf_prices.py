#!/usr/bin/env python3
"""Load Betfair historic price files into horse_racing.db from a file set you supply.

    python scripts/import_bf_prices.py <file-or-folder>... --db horse_racing.db --out bf_out

betfair_prices.py fetches these files from promo.betfair.com itself, which
refuses GitHub's US runners. This loads what someone with a UK connection has
already downloaded: Betfair's own daily files (dwbfprices{uk,ire}{win,place}
DDMMYYYY.csv), a notebook's combined file with a SOURCE_FILE column, or a
combined file without one (split by a REGION/COUNTRY column when there is
one). Column case does not matter.

Then it links the rows to race_results, reports coverage, and writes a slim
extract -- Betfair's BSP, morning and pre-play prices beside the result row
each one matched -- for the closing-line-value test, plus an agreement check
of Betfair's BSP against the BSP horseracebase recorded for the same runner.
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import betfair_prices as bp  # noqa: E402


def parse_any(path: str) -> pd.DataFrame:
    """One file, whatever its shape, as betfair_prices' typed frame."""
    name = os.path.basename(path)
    if bp.parse_file_name(name):
        return bp.parse_file(path)
    head = pd.read_csv(path, dtype=str, keep_default_na=False, nrows=2)
    cols = {c.strip().upper(): c for c in head.columns}
    if "SOURCE_FILE" in cols:
        return bp.parse_combined(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    region = next((cols[k] for k in ("REGION", "COUNTRY") if k in cols), None)
    parts = df.groupby(df[region].str.strip().str.lower()) if region else [("uk", df)]
    frames = []
    for reg, part in parts:
        reg = reg if reg in ("uk", "ire") else "uk"
        body = part.drop(columns=[region]) if region else part
        # The name only labels country and market; dates come from EVENT_DT per row.
        frames.append(bp.parse_file_text(body.to_csv(index=False), f"dwbfprices{reg}win01012000.csv"))
    out = pd.concat(frames, ignore_index=True)
    out["source_file"] = name
    return out


def collect(inputs: list[str]) -> list[str]:
    paths = []
    for p in inputs:
        paths += sorted(glob.glob(os.path.join(p, "**", "*.csv"), recursive=True)) if os.path.isdir(p) else [p]
    return [p for p in paths if not os.path.basename(p).lower().startswith("download_log")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--db", default=bp.DEFAULT_DB)
    ap.add_argument("--out", default="bf_out")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    paths = collect(a.inputs)
    frames = [parse_any(p) for p in paths]
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if frame.empty:
        print("no rows parsed from", paths)
        return 1
    frame = frame.drop_duplicates(["event_id", "selection_id", "market_type"], keep="last")
    dmin, dmax = frame["race_date"].dropna().min(), frame["race_date"].dropna().max()
    print(f"parsed {len(frame):,} rows from {len(paths)} file(s), {dmin} to {dmax}; "
          f"markets {frame['market_type'].value_counts().to_dict()}; "
          f"morningwap on {frame['morningwap'].gt(1).mean():.1%}")

    conn = sqlite3.connect(a.db)
    bp.ensure_table(conn)
    n = bp._upsert(conn, frame)
    conn.commit()
    conn.close()
    print(f"upserted {n:,} rows into betfair_prices")

    stats = bp.match_to_results(a.db, dmin, dmax)
    print(f"matched {stats['matched']:,}/{stats['total']:,} ({100 * stats['rate']:.1f}%) by {stats.get('by_method')}")
    if stats.get("unmatched_hints"):
        print("unmatched course hints:", stats["unmatched_hints"])
    cov = bp.coverage_report(a.db)
    cov.to_csv(os.path.join(a.out, "coverage.csv"), index=False)
    print(cov.to_string(index=False))

    conn = sqlite3.connect(a.db)
    ex = pd.read_sql_query("""
        SELECT b.race_date, b.event_dt, b.menu_hint, b.selection_name, b.horse_norm, b.market_type,
               b.win_lose, b.bsp, b.morningwap, b.ppwap, b.ppmax, b.ppmin, b.ipmax, b.ipmin,
               b.morning_vol, b.pp_vol, b.race_results_id,
               r.track, r.race_time, r.horse_name, r.bfsp AS hrb_bfsp, r.placing_numerical
        FROM betfair_prices b LEFT JOIN race_results r ON r.id = b.race_results_id
        WHERE b.race_date BETWEEN ? AND ?""", conn, params=(dmin, dmax))
    conn.close()
    ex.to_csv(os.path.join(a.out, "extract.csv.gz"), index=False, compression="gzip")
    m = ex[ex["race_results_id"].notna() & (ex["bsp"] > 1) & (ex["hrb_bfsp"] > 1)]
    if len(m):
        rel = np.abs(np.log(m["bsp"] / m["hrb_bfsp"]))
        print(f"BSP agreement on {len(m):,} matched runners: median |log ratio| {rel.median():.4f}, "
              f"within 1%: {(rel < 0.01).mean():.1%}")
    with open(os.path.join(a.out, "summary.txt"), "w") as f:
        f.write(f"rows {len(frame)} dates {dmin}..{dmax} matched {stats['matched']}/{stats['total']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
