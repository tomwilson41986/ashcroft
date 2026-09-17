#!/usr/bin/env python3
"""Print database info (row count, date range), and optionally audit `odds`.

    python scripts/db_info.py [db] [--odds-check]

``--odds-check`` answers one question the reporting depends on: is
``race_results.odds`` an early price or a closing one? Closing-line value is
the price you got against the price the market closed at, so it can only be
measured if the database holds a price struck *before* the off. If ``odds`` is
the returned industry SP it is itself a closing price, every "CLV" computed
from it is really a comparison of two closing prices, and historic CLV is not
available at all -- it has to accrue forward from a racecard price captured at
prediction time.

The signature is unambiguous either way. Two closing prices on the same race
agree closely: the industry SP and the Betfair SP differ mainly by the
exchange's lower overround, so expect a correlation above 0.97 on the logs, a
median |log ratio| under 0.08, and the two to name the same favourite in more
than 95% of races. A morning price would sit far wider -- a median |log ratio|
around 0.25 to 0.35 -- because hours of money move it.
"""
import argparse
import sqlite3
import sys


def _basic(conn) -> None:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM race_results")
    rows = cur.fetchone()[0]
    cur.execute("SELECT MIN(race_date), MAX(race_date) FROM race_results")
    d = cur.fetchone()
    print(f"Rows: {rows:,}, Date range: {d[0]} to {d[1]}")


def _fav_agreement(d):
    """Share of races where `odds` and the BSP name the same favourite.

    Races with a single priced runner are excluded: they agree by
    construction and would flatter the number."""
    import numpy as np
    by = d.groupby("raceid")
    multi = by.size() > 1
    if not multi.any():
        return float("nan")
    same = (by["odds"].idxmin() == by["bfsp"].idxmin())[multi]
    return float(np.mean(same))


def odds_check(conn) -> None:
    """Compare `odds` with the Betfair SP, by year."""
    import numpy as np
    import pandas as pd

    df = pd.read_sql_query(
        "SELECT race_date, race_time, track, horse_name, odds, bfsp "
        "FROM race_results WHERE race_date IS NOT NULL", conn)
    if df.empty:
        print("odds-check: no rows")
        return

    df["year"] = pd.to_datetime(df["race_date"], errors="coerce").dt.year
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
    df["bfsp"] = pd.to_numeric(df["bfsp"], errors="coerce")
    df["raceid"] = (df["race_date"].astype(str) + "_" + df["track"].astype(str)
                    + "_" + df["race_time"].astype(str))

    print(f"\n=== odds vs Betfair SP, by year ({len(df):,} rows) ===")
    print(f"{'year':>6} {'rows':>9} {'odds>1':>8} {'both':>9} {'corr':>7} "
          f"{'med|log|':>9} {'within10%':>10} {'fav agree':>10}")
    for year, g in df.groupby("year", dropna=True):
        have = float((g["odds"] > 1).mean())
        both = g[(g["odds"] > 1) & (g["bfsp"] > 1)]
        if len(both) < 50:
            print(f"{int(year):>6} {len(g):>9,} {have:>8.3f} {len(both):>9,}"
                  f" {'-':>7} {'-':>9} {'-':>10} {'-':>10}")
            continue
        lr = np.log(both["odds"] / both["bfsp"])
        corr = float(np.corrcoef(np.log(both["odds"]), np.log(both["bfsp"]))[0, 1])
        fav = _fav_agreement(both)
        print(f"{int(year):>6} {len(g):>9,} {have:>8.3f} {len(both):>9,} "
              f"{corr:>7.3f} {float(lr.abs().median()):>9.3f} "
              f"{float((lr.abs() < np.log(1.1)).mean()):>10.3f} "
              f"{fav:>10.3f}")

    both = df[(df["odds"] > 1) & (df["bfsp"] > 1)]
    if len(both) < 50:
        return
    lr = np.log(both["odds"] / both["bfsp"])
    corr = float(np.corrcoef(np.log(both["odds"]), np.log(both["bfsp"]))[0, 1])
    med = float(lr.abs().median())
    print(f"\npooled: corr {corr:.3f}, median |log ratio| {med:.3f}, "
          f"mean {float(lr.mean()):+.3f} (odds longer than BSP if positive)")

    # A wide spread alone does not say WHAT the column is, and the difference
    # matters: a timing gap means historic closing-line value is measurable, a
    # units or scale difference means the column is the same price in another
    # form and there is still no early price in the database.
    #
    # The two separate cleanly by price band. A constant overround gap (industry
    # SP against Betfair SP) is roughly constant in log space across bands. A
    # column holding decimal-minus-one -- fractional odds as a number, so 4/1
    # stored as 4.0 rather than 5.0 -- has a gap that shrinks as the price
    # lengthens, because (p-1)/p tends to 1.
    print("\n=== by BSP band: is the gap constant (overround) or shrinking (units)? ===")
    print(f"{'BSP band':>14} {'n':>9} {'med log(odds/bsp)':>18} "
          f"{'med log((bsp-1)/bsp)':>21} {'gap explained':>14}")
    bands = [(1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 8.0),
             (8.0, 15.0), (15.0, 40.0), (40.0, 1e9)]
    for lo, hi in bands:
        g = both[(both["bfsp"] >= lo) & (both["bfsp"] < hi)]
        if len(g) < 50:
            continue
        obs = float(np.log(g["odds"] / g["bfsp"]).median())
        # what decimal-minus-one would predict for this band
        pred = float(np.log((g["bfsp"] - 1.0) / g["bfsp"]).median())
        share = f"{100 * obs / pred:6.0f}%" if abs(pred) > 1e-9 else "     -"
        label = f"{lo:g}-{hi:g}" if hi < 1e9 else f"{lo:g}+"
        print(f"{label:>14} {len(g):>9,} {obs:>18.3f} {pred:>21.3f} {share:>14}")

    # Direct test of the same hypothesis, runner by runner.
    implied = both["bfsp"] - 1.0
    close = (both["odds"] - implied).abs() / implied.clip(lower=1e-9)
    print(f"\nshare of runners with odds within 5% of (BSP - 1): "
          f"{float((close < 0.05).mean()):.3f}")
    print(f"share of runners with odds within 5% of BSP:       "
          f"{float((np.abs(both['odds'] - both['bfsp']) / both['bfsp'] < 0.05).mean()):.3f}")

    if corr > 0.97 and med < 0.08:
        verdict = ("a closing price (industry SP): historic CLV is not measurable "
                   "from this column")
    else:
        verdict = ("NOT the same quantity as BSP. Read the band table before "
                   "concluding it is an early price: a gap that tracks "
                   "log((BSP-1)/BSP) is a units difference, not a timing one")
    print(f"\nverdict: odds looks like {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db", nargs="?", default="horse_racing.db")
    ap.add_argument("--odds-check", action="store_true",
                    help="audit race_results.odds against the Betfair SP")
    a = ap.parse_args()

    conn = sqlite3.connect(a.db)
    try:
        _basic(conn)
        if a.odds_check:
            odds_check(conn)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
