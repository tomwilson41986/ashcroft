"""Trading options the owner asked about (28 Sep): back at any price above our forecast, take all the
liquidity, and stake less on long shots.

The owner's idea, taken apart into its three choices and each scored on real prices:

1. Where to back. The tested rule backs where the morning price is at least 22% above the forecast
   (ln(morning / forecast) >= 0.2). "Anything above our forecast" is a threshold of 0. The table of
   move bins shows what each slice adds: the return per £1 on runners whose morning price is 0-5%,
   5-10%, 10-22%, 22-42% and 42%+ above the forecast. A bet just above the forecast is only worth
   having if its slice pays after commission.
2. How much. "All available liquidity": the historic files do not record the order book, only what
   traded in the morning (MORNINGTRADEDVOL at its volume-weighted price, MORNINGWAP). So the stake is
   a share of the morning's traded volume on the runner, from a tenth to all of it, filled at the
   morning price. The larger shares are optimistic: money that size would move the price.
3. Long shots. Variable stakes cap the stake by price: flat, falling with the square root of the
   price past 4.0, staking to win a set amount, and bands (full to 8.0, half to 16, a quarter to 30,
   nothing beyond).

Every bet is traded out at BSP, the way the edge is taken (TRADING.md): the net closing-line value,
5% commission on winnings. The owner's configurations are also shown held to the result.

Forecasts: the out-of-sample forecasts of the best model (the complete-careers pair gated with
race_xent) and of the complete-careers pair being switched to, 31 Dec 2025 to 31 Mar 2026, the
locked holdout excluded. The caveat of staking_backtest.py stands: these forecasts are built from
result rows and the edge is unconfirmed at bet time. This compares options; it does not prove the
edge. Read-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "research/queries/done")
import staking_backtest as sb  # noqa: E402  (its fetch, gate and price join)
from research_loop import average_arms  # noqa: E402

OUT = Path("out/trading_options")
sb.OUT = OUT
STEP = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0]
MIN_VOL = 100.0
MIN_STAKE = 2.0
C = 500.0                        # the stake cap at short prices for the variable shapes, £
MOVE_BINS = [0.0, 0.05, 0.10, 0.20, 0.35, np.inf]
PRICE_BANDS = [1.0, 4.0, 8.0, 16.0, 30.0, np.inf]
N_BOOT = 2000


def price_cap(price: np.ndarray, shape: str) -> np.ndarray:
    """The most we stake on a runner at this morning price."""
    if shape == "none":
        return np.full(len(price), np.inf)
    if shape == "flat":
        return np.full(len(price), C)
    if shape == "sqrt":                                   # full to 4.0, then ~ 1/sqrt(price)
        return C * np.minimum(1.0, np.sqrt(4.0 / price))
    if shape == "to_win":                                 # win 3C: C at 4.0, £100 at 16, £52 at 30
        return np.minimum(C, 3 * C / (price - 1.0))
    if shape == "bands":                                  # full to 8, half to 16, a quarter to 30
        return np.select([price <= 8, price <= 16, price <= 30], [C, C / 2, C / 4], 0.0)
    raise ValueError(shape)


def stakes(d: pd.DataFrame, threshold: float, share: float, shape: str) -> np.ndarray:
    move = d["pred_move"].to_numpy(float)
    ok = (d["morning_vol"].to_numpy(float) >= MIN_VOL) & (move > threshold if threshold == 0 else move >= threshold)
    s = np.minimum(share * d["morning_vol"].to_numpy(float), price_cap(d["morningwap"].to_numpy(float), shape))
    s = np.floor(np.where(ok, s, 0.0) * 100) / 100
    return np.where(s >= MIN_STAKE, s, 0.0)


def score(d: pd.DataFrame, s: np.ndarray, settle: str, rng) -> dict:
    pay = d["net" if settle == "trade" else "hold_morning"].to_numpy(float)
    b = d.assign(stake=s, profit=s * pay)[lambda x: x["stake"] > 0]
    if b.empty:
        return {"bets": 0}
    races = b.groupby("race").agg(t=("stake", "sum"), p=("profit", "sum"))
    t, p = races["t"].to_numpy(), races["p"].to_numpy()
    idx = rng.integers(0, len(races), (N_BOOT, len(races)))
    roi = p[idx].sum(1) / t[idx].sum(1)
    daily = b.groupby("race_date")["profit"].sum().sort_index()
    cum = daily.cumsum().to_numpy()
    dd = float(np.max(np.maximum.accumulate(np.r_[0.0, cum])[1:] - cum)) if len(cum) else 0.0
    return {"bets": len(b), "bets_per_day": round(len(b) / d["race_date"].nunique(), 1),
            "turnover": round(t.sum()), "profit": round(p.sum()), "roi_%": round(100 * p.sum() / t.sum(), 2),
            "roi_lo": round(100 * np.percentile(roi, 5), 2), "roi_hi": round(100 * np.percentile(roi, 95), 2),
            "profit_per_day": round(daily.mean()), "sd_day": round(daily.std()), "worst_day": round(daily.min()),
            "max_drawdown": round(dd), "losing_days_%": round(100 * (daily < 0).mean(), 1),
            "median_stake": round(float(np.median(b["stake"])), 2)}


def by_bins(d: pd.DataFrame, col: str, edges: list, labels: list, s: np.ndarray, rng) -> pd.DataFrame:
    """Return per £ within each bin, with a race-bootstrap interval (races resampled together)."""
    b = d.assign(stake=s, profit=s * d["net"].to_numpy(float))[lambda x: x["stake"] > 0].copy()
    b["bin"] = pd.cut(b[col], edges, right=False, labels=labels)
    rows = []
    for lab, g in b.groupby("bin", observed=True):
        races = g.groupby("race").agg(t=("stake", "sum"), p=("profit", "sum"))
        t, p = races["t"].to_numpy(), races["p"].to_numpy()
        idx = rng.integers(0, len(races), (N_BOOT, len(races)))
        roi = p[idx].sum(1) / t[idx].sum(1)
        rows.append({col: lab, "bets": len(g), "turnover": round(t.sum()), "profit": round(p.sum()),
                     "roi_%": round(100 * p.sum() / t.sum(), 2), "roi_lo": round(100 * np.percentile(roi, 5), 2),
                     "roi_hi": round(100 * np.percentile(roi, 95), 2),
                     "win_%": round(100 * g["won"].mean(), 1)})
    return pd.DataFrame(rows)


def report(name: str, d: pd.DataFrame) -> None:
    rng = np.random.default_rng(0)
    days = d["race_date"].nunique()
    print(f"\n===================== {name}: {d['race'].nunique():,} races, {len(d):,} runners, {days} days, "
          f"{d['race_date'].min()} to {d['race_date'].max()} =====================")
    bettable = d["morning_vol"] >= MIN_VOL
    print(f"runners with >= £{MIN_VOL:.0f} matched in the morning: {int(bettable.sum()):,}; "
          f"morning volume median £{d.loc[bettable, 'morning_vol'].median():,.0f}, "
          f"mean £{d.loc[bettable, 'morning_vol'].mean():,.0f}")

    unit = np.where(bettable & (d["pred_move"] > 0), 1.0, 0.0)
    print("\n-- 1. What each slice above the forecast pays: £1 level, traded out at BSP, by how far the morning "
          "price is above the forecast --")
    labels = ["0-5%", "5-10%", "10-22%", "22-42%", "42%+"]
    print(by_bins(d.assign(move_pct=d["pred_move"]), "move_pct", MOVE_BINS, labels, unit, rng).to_string(index=False))
    below = np.where(bettable & (d["pred_move"] <= 0), 1.0, 0.0)
    r = score(d, below, "trade", rng)
    print(f"   (for contrast, every runner priced at or below the forecast: {r['bets']:,} bets, "
          f"{r['roi_%']:+.2f}% ({r['roi_lo']:+.2f} to {r['roi_hi']:+.2f}))")

    print("\n-- 2. The threshold and the liquidity taken, no cap by price: traded out at BSP --")
    rows = []
    for thr in (0.0, 0.05, 0.10, 0.20):
        for share in (0.10, 0.25, 0.50, 1.00):
            rows.append({"above_forecast_by": f">{thr:.0%}" if thr == 0 else f">={np.expm1(thr):.0%}",
                         "share_of_morning_volume": f"{share:.0%}",
                         **score(d, stakes(d, thr, share, "none"), "trade", rng)})
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n-- 3. Variable stakes on long shots: all the morning's liquidity, capped by price (£{C:.0f} at short "
          "prices), traded out at BSP --")
    rows = []
    for thr in (0.0, 0.20):
        for shape in ("none", "flat", "sqrt", "to_win", "bands"):
            rows.append({"above_forecast_by": ">0%" if thr == 0 else ">=22%", "stake_by_price": shape,
                         **score(d, stakes(d, thr, 1.0, shape), "trade", rng)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n-- 4. Where the money is: by morning price, above the forecast at all, all the liquidity --")
    for shape in ("none", "sqrt", "bands"):
        print(f"   stake_by_price = {shape}")
        s = stakes(d, 0.0, 1.0, shape)
        print(by_bins(d, "morningwap", PRICE_BANDS, ["1-4", "4-8", "8-16", "16-30", "30+"], s, rng)
              .to_string(index=False))

    print("\n-- 5. The same held to the result at the morning price (not traded out) --")
    rows = []
    for thr in (0.0, 0.20):
        for shape in ("none", "sqrt", "bands"):
            rows.append({"above_forecast_by": ">0%" if thr == 0 else ">=22%", "stake_by_price": shape,
                         **score(d, stakes(d, thr, 1.0, shape), "hold", rng)})
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cache: dict = {}
    files = {n: sb.fetch(n, cache) for n in ("main_h18", "dml_h18", "rx16_h18")}
    average_arms([str(files["main_h18"]), str(files["dml_h18"])], str(OUT / "oos_pair_h18.csv"))
    sb.gate(OUT / "oos_pair_h18.csv", files["rx16_h18"], STEP, OUT / "oos_gate_step_h18.csv")
    for model, label in (("gate_step_h18", "the best model: the complete-careers pair gated with race_xent"),
                         ("pair_h18", "the complete-careers pair (being switched to)")):
        d = sb.frame(OUT / f"oos_{model}.csv")
        report(label, d)


if __name__ == "__main__":
    main()
