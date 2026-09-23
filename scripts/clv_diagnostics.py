#!/usr/bin/env python3
"""Where does the early-price edge come from? Clock, meeting order, going changes, withdrawals.

    python scripts/clv_diagnostics.py --predictions <oos.csv> --extract <betfair extract.csv.gz> \
        [--from 2026-04-01] [--to 2026-09-06]

The rule is #73's (scripts/clv_betfair.py): back at the morning WAP where
ln(morning / forecast) >= 0.2 with >= GBP100 matched in the morning, close at BSP, commission
on positive CLV. Its out-of-sample edge sits in later races, which two readings fit:

  1. the morning price is staler before later races (tradeable): the edge should follow the
     clock, the hours between the morning window and the off;
  2. the backtest's forecast knows race-day facts the result rows carry (not tradeable): the
     going as it was at the off, the riders who rode, the field after withdrawals. Facts that
     accumulate through a meeting follow the race's order at it, not the clock.

The first race of an evening meeting is late on the clock and first in its meeting, so it
separates them. Two channels of reading 2 are measured directly:
  - going changes: the meeting's going differs from its first race's;
  - late withdrawals: the extract holds only horses that ran, so a horse withdrawn after the
    morning leaves the runners' morning book short of a full field's, and Betfair cuts the odds
    of morning bets on the rest by its reduction factor, which the CLV formula ignores. The
    factor is estimated as 1 - book / (median book), floored at 0.

These are breakdowns of an already-scored window: descriptive, not a test.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

COMMISSION = 0.05
KEY = ["race_date", "track", "race_time", "horse_name"]


def minutes(t) -> float:
    """'1.50' or '1:50' -> minutes after midnight; card times 1-9 are afternoon."""
    parts = str(t).replace(":", ".").strip(".").split(".")
    try:
        h, m = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return np.nan
    return (h + 12 if h < 10 else h) * 60 + m


def net_clv(morning: pd.Series, bsp: pd.Series) -> np.ndarray:
    clv = morning / bsp - 1
    return np.where(clv > 0, clv * (1 - COMMISSION), clv)


def load(predictions: str, extract: str, start: str, end: str) -> pd.DataFrame:
    o = pd.read_csv(predictions)
    ex = pd.read_csv(extract)
    ex = ex[ex["race_results_id"].notna()][KEY + ["morningwap", "morning_vol", "bsp"]]
    d = o.merge(ex, on=KEY)
    d = d[(d["race_date"] >= start) & (d["race_date"] <= end)]
    d = d[(d["morningwap"] > 1) & (d["bsp"] > 1) & (d["predicted_bfsp"] > 1)].copy()
    d["race"] = d["race_date"] + "|" + d["track"] + "|" + d["race_time"].astype(str)
    d["mins"] = d["race_time"].map(minutes)
    r = d.drop_duplicates("race")[["race", "race_date", "track", "mins", "going_description"]].copy()
    r["meeting_order"] = r.groupby(["race_date", "track"])["mins"].rank(method="min").astype(int)
    first = (r.sort_values("mins").groupby(["race_date", "track"])
             .agg(first_mins=("mins", "first"), first_going=("going_description", "first"),
                  goings=("going_description", "nunique")))
    r = r.merge(first, left_on=["race_date", "track"], right_index=True)
    r["evening_meeting"] = r["first_mins"] >= 17 * 60
    r["going_changed_meeting"] = r["goings"] > 1
    r["after_change"] = r["going_description"] != r["first_going"]
    d = d.merge(r[["race", "meeting_order", "evening_meeting", "going_changed_meeting", "after_change"]], on="race")
    d["pred_move"] = np.log(d["morningwap"] / d["predicted_bfsp"])
    d["net"] = net_clv(d["morningwap"], d["bsp"])
    book = d.groupby("race")["morningwap"].apply(lambda s: (1 / s).sum())
    d["rf"] = d["race"].map((1 - book / book.median()).clip(lower=0))
    d["net_rf"] = net_clv(1 + (d["morningwap"] - 1) * (1 - d["rf"]), d["bsp"])
    return d


def interval(frame: pd.DataFrame, col: str = "net", n: int = 2000) -> str:
    if len(frame) < 30:
        return f"too few ({len(frame)})"
    g = frame.groupby("race")[col].agg(["sum", "size"])
    s, c = g["sum"].to_numpy(), g["size"].to_numpy()
    rng = np.random.default_rng(2)
    b = [s[i].sum() / c[i].sum() for i in (rng.integers(0, len(g), len(g)) for _ in range(n))]
    return (f"{100 * frame[col].mean():+6.2f}% ({100 * np.percentile(b, 5):+.2f} to "
            f"{100 * np.percentile(b, 95):+.2f}), n={len(frame):,}")


def clock_and_order(frame: pd.DataFrame, seed: int = 3) -> list[tuple[str, float, float, float]]:
    """Race-level weighted regression of net CLV on hours after 11:00 and order at the meeting."""
    g = frame.groupby("race").agg(net=("net", "mean"), n=("net", "size"), mins=("mins", "first"),
                                  order=("meeting_order", "first")).reset_index()
    X = np.column_stack([np.ones(len(g)), (g["mins"] - 11 * 60) / 60, g["order"] - 1])
    y, w = g["net"].to_numpy(), np.sqrt(g["n"].to_numpy())
    beta = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
    rng = np.random.default_rng(seed)
    bs = np.array([np.linalg.lstsq(X[i] * w[i][:, None], y[i] * w[i], rcond=None)[0]
                   for i in (rng.integers(0, len(g), len(g)) for _ in range(1000))])
    names = ["intercept (at 11:00)", "per hour after 11:00", "per race of meeting order"]
    return [(nm, beta[j], np.percentile(bs[:, j], 5), np.percentile(bs[:, j], 95)) for j, nm in enumerate(names)]


def report(d: pd.DataFrame, threshold: float = 0.2, min_vol: float = 100.0) -> None:
    base = d[d["morning_vol"] >= min_vol]
    rule = base[base["pred_move"] >= threshold].copy()
    print(f"{len(d):,} runners in {d['race'].nunique():,} races, {d['race_date'].min()} to {d['race_date'].max()}")
    print(f"\nThe rule            {interval(rule)}")
    print(f"Every runner        {interval(base)}")

    rule["clock"] = pd.cut(rule["mins"], [0, 15 * 60, 17 * 60, 24 * 60], right=False,
                           labels=["before 15:00", "15:00-16:59", "17:00 on"])
    rule["order"] = pd.cut(rule["meeting_order"], [0, 1, 3, 99], labels=["1st at meeting", "2nd-3rd", "4th on"])
    print("\nBy clock (rows) and order at the meeting (columns)")
    for clock, g in rule.groupby("clock", observed=True):
        print(f"  {clock:13s} " + " | ".join(f"{o}: {interval(h)}" for o, h in g.groupby("order", observed=True)))
    f1 = rule[rule["meeting_order"] == 1]
    print(f"\nFirst race, afternoon meeting   {interval(f1[~f1['evening_meeting']])}")
    print(f"First race, evening meeting     {interval(f1[f1['evening_meeting']])}")
    print("\nRegression of the rule's net CLV (race level, weighted by bets):")
    for nm, b, lo, hi in clock_and_order(rule):
        print(f"  {nm:28s} {100 * b:+.2f}%  (90% {100 * lo:+.2f} to {100 * hi:+.2f})")

    print("\nGoing changes during the meeting (read from the result rows)")
    print(f"  going never changed            {interval(rule[~rule['going_changed_meeting']])}")
    print(f"  changed meeting, before change {interval(rule[rule['going_changed_meeting'] & ~rule['after_change']])}")
    print(f"  changed meeting, after change  {interval(rule[rule['going_changed_meeting'] & rule['after_change']])}")

    print("\nLate withdrawals (reduction factor estimated from the runners' morning book)")
    print(f"  the rule, reduction factors applied   {interval(rule, 'net_rf')}")
    print(f"  every runner, reduction factors       {interval(base, 'net_rf')}")
    clean = rule[(rule["rf"] <= 0.025) & ~rule["going_changed_meeting"]]
    print(f"  no sign of a withdrawal (RF <= 2.5%) and no going change: {interval(clean)}")
    for nm, b, lo, hi in clock_and_order(clean):
        print(f"    {nm:28s} {100 * b:+.2f}%  (90% {100 * lo:+.2f} to {100 * hi:+.2f})")
    c1 = clean[clean["meeting_order"] == 1]
    print(f"    first race, evening meeting   {interval(c1[c1['evening_meeting']])}")
    print(f"    first race, afternoon meeting {interval(c1[~c1['evening_meeting']])}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--predictions", required=True, help="walk-forward forecasts (evaluate_oos output)")
    ap.add_argument("--extract", required=True, help="Betfair price extract matched to race_results")
    ap.add_argument("--from", dest="start", default="2026-04-01")
    ap.add_argument("--to", dest="end", default="2026-09-06")
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--min-vol", type=float, default=100.0)
    a = ap.parse_args(argv)
    report(load(a.predictions, a.extract, a.start, a.end), a.threshold, a.min_vol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
