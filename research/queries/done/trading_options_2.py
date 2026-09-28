"""Trading options, second pass: the owner's ideas without hindsight in the stake.

trading_options.py (research query run 36410545194) sized stakes by the whole morning's traded volume
on the runner. That flatters any option that takes more: a runner that attracts a lot of money in the
morning is disproportionately one whose price shortens, and at 08:00 nobody knows how much will trade
by 11:00. This pass keeps volume only as a ceiling on the stake (the money that could have been
matched), not as its size, and reads the edge at £1 level stakes where it can:

A. what each slice above the forecast pays at £1 level, by morning price (is the edge in the
   long shots real without volume weighting?);
B. the same by the race's morning liquidity (all its runners' morning volume, which says how big
   the market is rather than how one runner was backed): do small overlays pay in big markets?
C. the threshold (any price above the forecast, 5%, 11%, 22%, 35% above) against the stake's shape
   by price, with a £100 cap at short prices and a quarter of the morning volume as the ceiling;
D. the same with the owner's scale: £500 at short prices and all the morning's volume as the ceiling.

Traded out at BSP, 5% commission on winnings. The best model's forecasts (the complete-careers pair
gated with race_xent), 31 Dec 2025 to 31 Mar 2026, the locked holdout excluded. Read-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "research/queries")
sys.path.insert(0, "research/queries/done")
import staking_backtest as sb  # noqa: E402
import trading_options as to  # noqa: E402
from research_loop import average_arms  # noqa: E402

OUT = Path("out/trading_options_2")
sb.OUT = OUT
MOVE_EDGES = [0.0, 0.05, 0.10, 0.20, 0.35, np.inf]
MOVE_LABELS = ["0-5%", "5-10%", "10-22%", "22-42%", "42%+"]
THRESHOLDS = [0.0, 0.05, 0.10, 0.20, 0.35]


def level_roi(g: pd.DataFrame, rng) -> tuple[int, float, float, float]:
    races = g.groupby("race")["net"].agg(["sum", "count"])
    s, n = races["sum"].to_numpy(), races["count"].to_numpy()
    idx = rng.integers(0, len(races), (to.N_BOOT, len(races)))
    boot = s[idx].sum(1) / n[idx].sum(1)
    return len(g), 100 * s.sum() / n.sum(), 100 * np.percentile(boot, 5), 100 * np.percentile(boot, 95)


def grid(d: pd.DataFrame, by: str, labels: list, rng) -> pd.DataFrame:
    b = d[(d["morning_vol"] >= to.MIN_VOL) & (d["pred_move"] > 0)].copy()
    b["slice"] = pd.cut(b["pred_move"], MOVE_EDGES, right=False, labels=MOVE_LABELS)
    rows = []
    for (grp, sl), g in b.groupby([by, "slice"], observed=True):
        n, roi, lo, hi = level_roi(g, rng)
        rows.append({by: grp, "above_forecast": sl, "bets": n, "roi_%": round(roi, 2),
                     "ci": f"{lo:+.1f} to {hi:+.1f}"})
    t = pd.DataFrame(rows)
    order = [c for c in d[by].cat.categories if c in set(t[by])]
    return t.pivot(index="above_forecast", columns=by, values="roi_%").reindex(index=MOVE_LABELS, columns=order), t


def capped(d: pd.DataFrame, cap: float, share: float, rng) -> pd.DataFrame:
    to.C = cap
    rows = []
    for thr in THRESHOLDS:
        for shape in ("flat", "sqrt", "to_win", "bands"):
            r = to.score(d, to.stakes(d, thr, share, shape), "trade", rng)
            r = {"above_forecast_by": ">0%" if thr == 0 else f">={np.expm1(thr):.0%}", "stake_by_price": shape, **r}
            r["profit_per_sd"] = round(r["profit_per_day"] / r["sd_day"], 2) if r.get("sd_day") else None
            rows.append(r)
    keep = ["above_forecast_by", "stake_by_price", "bets_per_day", "turnover", "profit", "roi_%", "roi_lo", "roi_hi",
            "profit_per_day", "sd_day", "profit_per_sd", "worst_day", "max_drawdown", "losing_days_%", "median_stake"]
    return pd.DataFrame(rows)[keep]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cache: dict = {}
    files = {n: sb.fetch(n, cache) for n in ("main_h18", "dml_h18", "rx16_h18")}
    average_arms([str(files["main_h18"]), str(files["dml_h18"])], str(OUT / "oos_pair_h18.csv"))
    sb.gate(OUT / "oos_pair_h18.csv", files["rx16_h18"], to.STEP, OUT / "oos_gate_step_h18.csv")
    d = sb.frame(OUT / "oos_gate_step_h18.csv")
    rng = np.random.default_rng(0)
    print(f"the best model: {d['race'].nunique():,} races, {len(d):,} runners, {d['race_date'].nunique()} days")

    d["price_band"] = pd.cut(d["morningwap"], [1, 4, 8, 16, 30, np.inf], right=False,
                             labels=["1-4", "4-8", "8-16", "16-30", "30+"])
    piv, long = grid(d, "price_band", None, rng)
    print("\n-- A. £1 level, traded out: return % by how far the morning price is above the forecast (rows) "
          "and the morning price (columns) --")
    print(piv.to_string())
    print(long.to_string(index=False))

    race_vol = d.groupby("race")["morning_vol"].transform("sum")
    d["race_liquidity"] = pd.qcut(race_vol, 3, labels=["small market", "medium market", "big market"])
    q = race_vol.quantile([1 / 3, 2 / 3]).round(-2).to_list()
    piv, long = grid(d, "race_liquidity", None, rng)
    print(f"\n-- B. £1 level, traded out: by the race's morning volume (thirds: under £{q[0]:,.0f}, "
          f"£{q[0]:,.0f}-{q[1]:,.0f}, over £{q[1]:,.0f}) --")
    print(piv.to_string())
    print(long.to_string(index=False))

    print("\n-- C. Threshold x stake shape: £100 at short prices, a quarter of the morning volume as the ceiling --")
    print(capped(d, 100.0, 0.25, rng).to_string(index=False))
    print("\n-- D. The owner's scale: £500 at short prices, all the morning volume as the ceiling --")
    print(capped(d, 500.0, 1.0, rng).to_string(index=False))


if __name__ == "__main__":
    main()
