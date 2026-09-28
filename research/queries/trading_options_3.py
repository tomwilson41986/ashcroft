"""Trading options, third pass: a threshold that slides with the price, chosen on one half and scored on the other.

trading_options_2.py (research query run 36411192738) found that how far the morning price must sit
above the forecast before a back pays depends on the price: under 8.0 a 5% overlay pays, from 8.0
the slices under 22% lose, and above 30 only the big overlays pay. That table was read on the whole
window, so a threshold built from it would be judged on the data that chose it. Here each half of
the window (to 13 Feb, from 14 Feb) chooses, for each morning-price band, the smallest threshold
(0, 5%, 11%, 22%, 42% above the forecast) from which every slice above it paid at £1 level, and the
other half scores it, against the fixed 22% rule and against backing any price above the forecast.

Scored at £1 level, at £100 at short prices (a quarter of the morning volume as the ceiling) and at
the owner's £500 (all of it), flat and with the square-root taper past 4.0. Traded out at BSP, 5%
commission on winnings; the best model's forecasts, the locked holdout excluded. Read-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "research/queries/done")
import staking_backtest as sb  # noqa: E402
import trading_options as to  # noqa: E402
from research_loop import average_arms  # noqa: E402

OUT = Path("out/trading_options_3")
sb.OUT = OUT
SPLIT = "2026-02-14"
BANDS = [1.0, 4.0, 8.0, 16.0, 30.0, np.inf]
BAND_LABELS = ["1-4", "4-8", "8-16", "16-30", "30+"]
CUTS = [0.0, 0.05, 0.10, 0.20, 0.35]


def choose(d: pd.DataFrame) -> dict[str, float]:
    """For each price band, the smallest cut from which every slice above it paid at £1 level."""
    b = d[(d["morning_vol"] >= to.MIN_VOL) & (d["pred_move"] > 0)]
    out = {}
    for band, g in b.groupby("band", observed=True):
        edges = CUTS + [np.inf]
        roi = [g.loc[(g["pred_move"] >= lo) & (g["pred_move"] < hi), "net"].mean() for lo, hi in zip(edges, edges[1:])]
        pick = CUTS[-1]
        for i in range(len(CUTS)):
            if all(r > 0 for r in roi[i:] if not np.isnan(r)):
                pick = CUTS[i]
                break
        out[str(band)] = pick
    return out


def backed(d: pd.DataFrame, rule) -> np.ndarray:
    move = d["pred_move"].to_numpy(float)
    ok = d["morning_vol"].to_numpy(float) >= to.MIN_VOL
    if isinstance(rule, dict):
        thr = d["band"].astype(str).map(rule).to_numpy(float)
        return ok & (move >= thr) & (move > 0)
    return ok & ((move > 0) if rule == 0 else (move >= rule))


def stakes(d: pd.DataFrame, on: np.ndarray, scale: str) -> np.ndarray:
    price, vol = d["morningwap"].to_numpy(float), d["morning_vol"].to_numpy(float)
    if scale == "£1 level":
        s = np.ones(len(d))
    elif scale == "£100 flat":
        to.C = 100.0
        s = np.minimum(0.25 * vol, to.price_cap(price, "flat"))
    elif scale == "£500 flat":
        to.C = 500.0
        s = np.minimum(vol, to.price_cap(price, "flat"))
    else:
        to.C = 500.0
        s = np.minimum(vol, to.price_cap(price, "sqrt"))
    s = np.where(on, s, 0.0)
    if scale != "£1 level":
        s = np.floor(s * 100) / 100
        s = np.where(s >= to.MIN_STAKE, s, 0.0)
    return s


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cache: dict = {}
    files = {n: sb.fetch(n, cache) for n in ("main_h18", "dml_h18", "rx16_h18")}
    average_arms([str(files["main_h18"]), str(files["dml_h18"])], str(OUT / "oos_pair_h18.csv"))
    sb.gate(OUT / "oos_pair_h18.csv", files["rx16_h18"], to.STEP, OUT / "oos_gate_step_h18.csv")
    d = sb.frame(OUT / "oos_gate_step_h18.csv")
    d["band"] = pd.cut(d["morningwap"], BANDS, right=False, labels=BAND_LABELS)
    halves = {"to 13 Feb": d[d["race_date"].astype(str) < SPLIT].reset_index(drop=True),
              "from 14 Feb": d[d["race_date"].astype(str) >= SPLIT].reset_index(drop=True)}
    rng = np.random.default_rng(0)
    rows = []
    for fit_name, score_name in (("to 13 Feb", "from 14 Feb"), ("from 14 Feb", "to 13 Feb")):
        rule = choose(halves[fit_name])
        print(f"chosen on the half {fit_name}: " + ", ".join(f"{k}: {'>0' if v == 0 else f'>={np.expm1(v):.0%}'}"
                                                                for k, v in rule.items()))
        h = halves[score_name]
        for rname, r in (("any price above the forecast", 0), ("fixed 22%", 0.2), ("sliding, chosen on the other half", rule)):
            on = backed(h, r)
            for scale in ("£1 level", "£100 flat", "£500 flat", "£500 sqrt"):
                res = to.score(h, stakes(h, on, scale), "trade", rng)
                rows.append({"scored_on": score_name, "rule": rname, "stakes": scale,
                             **{k: res.get(k) for k in ("bets", "turnover", "profit", "roi_%", "roi_lo", "roi_hi",
                                                        "profit_per_day", "sd_day", "worst_day", "max_drawdown")}})
    t = pd.DataFrame(rows)
    for scale, g in t.groupby("stakes", sort=False):
        print(f"\n-- {scale}, each half scored with the thresholds the other half chose --")
        print(g.drop(columns="stakes").to_string(index=False))
    both = t.groupby(["stakes", "rule"], sort=False)[["bets", "turnover", "profit"]].sum()
    both["roi_%"] = (100 * both["profit"] / both["turnover"]).round(2)
    print("\n-- both halves together (each scored out of sample) --")
    print(both.to_string())


if __name__ == "__main__":
    main()
