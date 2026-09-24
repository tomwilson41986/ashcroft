#!/usr/bin/env python3
"""Closing-line value of the walk-forward BSP forecast against Betfair's morning price.

    python scripts/clv_betfair.py --predictions data/oos_predictions.csv --db horse_racing.db
    python scripts/clv_betfair.py --predictions <oos.csv> --extract bf_out/extract.csv.gz

The rule was fixed before the first look (reports/clv_betfair_2026q1.md): back where the
model's price is at least 22% shorter than the morning volume-weighted price, with at least
£100 matched in the morning; take the morning price, close at BSP. Net CLV per unit is
morningwap / bsp - 1, less commission only when positive. It passes if the 90%
race-bootstrap interval is above zero.

Also prints the checks that decide what a pass means: the edge by how many races that day
had already been run (the same-day-leak test), by morning price band against every runner
in the band, month by month, and settled three ways.
"""

from __future__ import annotations

import argparse
import sqlite3

import numpy as np
import pandas as pd

COMMISSION = 0.05
KEY = ["race_date", "track", "race_time", "horse_name"]

EXTRACT_SQL = """
    SELECT r.race_date, r.track, r.race_time, r.horse_name,
           b.morningwap, b.morning_vol, b.bsp
    FROM betfair_prices b JOIN race_results r ON r.id = b.race_results_id
    WHERE b.market_type = 'win'
"""


def load(predictions: str, db: str | None, extract: str | None, until: str | None = None) -> pd.DataFrame:
    """Predictions joined to Betfair's win prices. `until` drops every price on or after
    that date before anything else is done with it (the locked holdout)."""
    o = pd.read_csv(predictions, dtype={"race_time": str})    # "2.30" must not become 2.3
    if extract:
        ex = pd.read_csv(extract, dtype={"race_time": str})
        ex = ex[ex["race_results_id"].notna()]
    else:
        conn = sqlite3.connect(db)
        sql, params = EXTRACT_SQL, ()
        if until:
            sql, params = sql + " AND r.race_date < ?", (until,)
        ex = pd.read_sql_query(sql, conn, params=params)
        conn.close()
    if until:
        o = o[o["race_date"].astype(str) < until]
        ex = ex[ex["race_date"].astype(str) < until]
    d = o.merge(ex[KEY + ["morningwap", "morning_vol", "bsp"]], on=KEY)
    d["race"] = d["race_date"] + "|" + d["track"] + "|" + d["race_time"]
    d = d[(d["morningwap"] > 1) & (d["bsp"] > 1) & (d["predicted_bfsp"] > 1)].copy()
    d["won"] = d["won"].astype(bool)
    d["pred_move"] = np.log(d["morningwap"] / d["predicted_bfsp"])
    clv = d["morningwap"] / d["bsp"] - 1
    d["net"] = np.where(clv > 0, clv * (1 - COMMISSION), clv)
    d["hold_morning"] = np.where(d["won"], (d["morningwap"] - 1) * (1 - COMMISSION), -1.0)
    d["hold_bsp"] = np.where(d["won"], (d["bsp"] - 1) * (1 - COMMISSION), -1.0)
    d["mins"] = d["race_time"].map(_minutes)
    races = d.drop_duplicates("race")[["race", "race_date", "track", "mins"]].copy()
    races["meeting_order"] = races.groupby(["race_date", "track"])["mins"].rank(method="min")
    races["day_order"] = races.groupby("race_date")["mins"].rank(method="min")
    return d.merge(races[["race", "meeting_order", "day_order"]], on="race")


def _minutes(t) -> float:
    """'1.50.' or '1:50' -> minutes after midnight; card times 1-9 are afternoon."""
    parts = str(t).replace(":", ".").strip(".").split(".")
    try:
        h, m = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return np.nan
    return (h + 12 if h < 10 else h) * 60 + m


def interval(frame: pd.DataFrame, col: str, n: int = 2000, seed: int = 0) -> str:
    if len(frame) < 30:
        return f"too few ({len(frame)})"
    g = frame.groupby("race")[col].agg(["sum", "size"])
    s, c = g["sum"].to_numpy(), g["size"].to_numpy()
    rng = np.random.default_rng(seed)
    b = [s[i].sum() / c[i].sum() for i in (rng.integers(0, len(g), len(g)) for _ in range(n))]
    return (f"{100 * frame[col].mean():+6.2f}% ({100 * np.percentile(b, 5):+.2f} to "
            f"{100 * np.percentile(b, 95):+.2f}), n={len(frame):,}")


def information(d: pd.DataFrame) -> None:
    """Does the forecast add to the morning price about where BSP closes?

    ln q_BSP on a quadratic in ln q_morning (the morning price's own bias) and ln q_model,
    all race-normalised. The weight on the model gets a race-bootstrap interval; then each
    month is scored with coefficients fitted on the months before it.
    """
    d = d.copy()
    for c, q in (("morningwap", "q_m"), ("predicted_bfsp", "q_f"), ("bsp", "q_b")):
        inv = 1 / d[c]
        d[q] = inv / inv.groupby(d["race"]).transform("sum")
    lm, lf, y = np.log(d["q_m"]).to_numpy(), np.log(d["q_f"]).to_numpy(), np.log(d["q_b"]).to_numpy()
    X = np.column_stack([np.ones(len(d)), lm, lm ** 2, lf])
    A = X[:, :3]
    w = np.linalg.lstsq(X, y, rcond=None)[0][3]
    races = d["race"].to_numpy()
    uniq = np.unique(races)
    rows = {r: np.where(races == r)[0] for r in uniq}
    rng = np.random.default_rng(0)
    boot = [np.linalg.lstsq(X[p], y[p], rcond=None)[0][3]
            for p in (np.concatenate([rows[r] for r in rng.choice(uniq, len(uniq))]) for _ in range(300))]
    print(f"\nWeight on the model beside the morning price: {w:.3f} "
          f"(95% CI {np.percentile(boot, 2.5):.3f} to {np.percentile(boot, 97.5):.3f})")
    month = d["race_date"].str[:7].to_numpy()
    for m in sorted(set(month))[1:]:
        tr, te = month < m, month == m
        if tr.sum() < 1000 or te.sum() < 1000:
            continue
        ca = np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
        cb = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
        print(f"  {m}: RMSE of the BSP forecast, morning price alone {np.sqrt(np.mean((y[te] - A[te] @ ca) ** 2)):.4f}"
              f" -> with the model {np.sqrt(np.mean((y[te] - X[te] @ cb) ** 2)):.4f}  (n={te.sum():,})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--predictions", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--db")
    src.add_argument("--extract")
    ap.add_argument("--threshold", type=float, default=0.2, help="ln(morning/forecast) to back at")
    ap.add_argument("--min-vol", type=float, default=100.0)
    ap.add_argument("--until", default=None, help="ignore every price on or after this date (YYYY-MM-DD)")
    a = ap.parse_args(argv)

    d = load(a.predictions, a.db, a.extract, a.until)
    print(f"{len(d):,} runners in {d['race'].nunique():,} races, {d['race_date'].min()} to {d['race_date'].max()}")
    base = d[d["morning_vol"] >= a.min_vol]
    rule = base[base["pred_move"] >= a.threshold]
    print("\nNet CLV per unit (90% race-bootstrap interval)")
    print(f"  every runner                  {interval(base, 'net')}")
    for t in (0.1, a.threshold, 0.3):
        print(f"  model >= {100 * (np.exp(t) - 1):3.0f}% shorter         {interval(base[base['pred_move'] >= t], 'net')}")
    print(f"  model >= {100 * (np.exp(a.threshold) - 1):3.0f}% LONGER          {interval(base[base['pred_move'] <= -a.threshold], 'net')}")
    information(d)
    print("\nThe rule, by how many races that day had already been run (same-day-leak test)")
    for lab, m in (("first race at its meeting", rule["meeting_order"] == 1),
                   ("first 3 races of the day", rule["day_order"] <= 3),
                   ("race 10+ of the day", rule["day_order"] >= 10)):
        print(f"  {lab:28s}  {interval(rule[m], 'net')}")
    print("\nBy morning price band: the rule against every runner in the band")
    for lo_p, hi_p in ((1, 4), (4, 8), (8, 16), (16, 1001)):
        band = base[base["morningwap"].between(lo_p, hi_p, inclusive="left")]
        pick = band[band["pred_move"] >= a.threshold]
        print(f"  {lo_p:>3}-{hi_p:<4} rule {interval(pick, 'net')}   all {interval(band, 'net')}")
    band = base[base["morningwap"] >= 16]
    print(f"  16+ where the model says >= {100 * (np.exp(a.threshold) - 1):.0f}% LONGER  "
          f"{interval(band[band['pred_move'] <= -a.threshold], 'net')}")
    print("\nThe rule month by month")
    for mth, g in rule.groupby(rule["race_date"].str[:7]):
        print(f"  {mth}  {interval(g, 'net')}")
    print("\nThe rule settled three ways")
    print(f"  traded out at BSP     {interval(rule, 'net')}")
    print(f"  held, morning price   {interval(rule, 'hold_morning')}")
    print(f"  held, BSP             {interval(rule, 'hold_bsp')}   every runner at BSP {100 * base['hold_bsp'].mean():+.2f}%")
    print(f"\nMorning volume on the rule's bets: median £{rule['morning_vol'].median():,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
