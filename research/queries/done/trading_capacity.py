"""How much money the trade can take: liquidity on the runners the rule backs, and the edge later in the day.

The owner doubts the liquidity is there to get volume on, rightly: the earlier passes assumed fills of
up to £500 a runner at the morning's average price. This reads what the market actually traded on the
runners the rule backs, and what realistic stakes make:

A. morning and pre-play traded volume on the rule's runners (the whole morning to 11:00, and the whole
   pre-play market to the off), by morning price;
B. capacity in the morning: the stake capped both in pounds (£25 to £250) and at a share of what
   traded on the runner all morning (5% to 20%: beyond a tenth or so our own money would move the
   price), with the rule's profit per day at each;
C. the edge later in the day, where the money is: backing at the pre-play average price (PPWAP, most
   of whose volume trades in the last hour) wherever it sits at least a threshold above the forecast,
   closed at BSP; its return at £1 level, and its capacity at shares of the pre-play volume.

Traded out at BSP, 5% commission on winnings; the best model's out-of-sample forecasts (the
complete-careers pair gated with race_xent), 31 Dec 2025 to 31 Mar 2026, the locked holdout excluded.
Fills at the average price are an approximation either way; the paper trader measures the book.
Read-only.
"""

from __future__ import annotations

import sqlite3
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

OUT = Path("out/trading_capacity")
sb.OUT = OUT
KEY = ["race_date", "track", "race_time", "horse_name"]
SQL = """
    SELECT r.race_date, r.track, r.race_time, r.horse_name,
           b.morningwap, b.morning_vol, b.ppwap, b.pp_vol, b.bsp
    FROM betfair_prices b JOIN race_results r ON r.id = b.race_results_id
    WHERE b.market_type = 'win' AND r.race_date < '2026-04-01'
"""
COMMISSION = 0.05
N_BOOT = 1000


def load(pred: Path) -> pd.DataFrame:
    o = pd.read_csv(pred, dtype={"race_time": str})
    o = o[o["race_date"].astype(str) < "2026-04-01"]
    conn = sqlite3.connect("horse_racing.db")
    ex = pd.read_sql_query(SQL, conn)
    conn.close()
    d = o.merge(ex, on=KEY)
    d = d[(d["morningwap"] > 1) & (d["bsp"] > 1) & (d["predicted_bfsp"] > 1)].copy()
    d["race"] = d["race_date"].astype(str) + "|" + d["track"] + "|" + d["race_time"]
    d["won"] = d["won"].astype(bool)
    for col in ("morning_vol", "pp_vol"):
        d[col] = d[col].fillna(0.0)
    d["move_m"] = np.log(d["morningwap"] / d["predicted_bfsp"])
    clv = d["morningwap"] / d["bsp"] - 1
    d["net_m"] = np.where(clv > 0, clv * (1 - COMMISSION), clv)
    ok = d["ppwap"] > 1
    d["move_p"] = np.where(ok, np.log(d["ppwap"].where(ok, 1.0) / d["predicted_bfsp"]), np.nan)
    clvp = d["ppwap"] / d["bsp"] - 1
    d["net_p"] = np.where(clvp > 0, clvp * (1 - COMMISSION), clvp)
    return d


def run(d: pd.DataFrame, on: np.ndarray, vol: str, net: str, share: float, cap: float, rng) -> dict:
    s = np.minimum(cap, share * d[vol].to_numpy(float))
    s = np.floor(np.where(on, s, 0.0) * 100) / 100
    s = np.where(s >= 2.0, s, 0.0)
    b = d.assign(stake=s, profit=s * d[net].to_numpy(float))[lambda x: x["stake"] > 0]
    days = d["race_date"].nunique()
    if b.empty:
        return {"bets_per_day": 0}
    races = b.groupby("race").agg(t=("stake", "sum"), p=("profit", "sum"))
    t, p = races["t"].to_numpy(), races["p"].to_numpy()
    idx = rng.integers(0, len(races), (N_BOOT, len(races)))
    roi = p[idx].sum(1) / t[idx].sum(1)
    capped = (np.minimum(cap, share * b[vol]) >= cap - 1e-9).mean()
    return {"bets_per_day": round(len(b) / days, 1), "median_stake": round(float(b["stake"].median()), 2),
            "stake_at_cap_%": round(100 * capped, 0), "turnover_per_day": round(t.sum() / days),
            "profit_per_day": round(p.sum() / days), "roi_%": round(100 * p.sum() / t.sum(), 2),
            "roi_lo": round(100 * np.percentile(roi, 5), 2), "roi_hi": round(100 * np.percentile(roi, 95), 2)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cache: dict = {}
    files = {n: sb.fetch(n, cache) for n in ("main_h18", "dml_h18", "rx16_h18")}
    average_arms([str(files["main_h18"]), str(files["dml_h18"])], str(OUT / "oos_pair_h18.csv"))
    sb.gate(OUT / "oos_pair_h18.csv", files["rx16_h18"], to.STEP, OUT / "oos_gate_step_h18.csv")
    d = load(OUT / "oos_gate_step_h18.csv")
    rng = np.random.default_rng(0)
    days = d["race_date"].nunique()
    print(f"the best model: {d['race'].nunique():,} races, {len(d):,} runners, {days} days")

    rule = (d["morning_vol"] >= 100) & (d["move_m"] >= 0.2)
    r = d[rule].copy()
    r["band"] = pd.cut(r["morningwap"], [1, 4, 8, 16, 30, np.inf], right=False,
                       labels=["1-4", "4-8", "8-16", "16-30", "30+"])
    q = [0.1, 0.25, 0.5, 0.75, 0.9]
    print(f"\n-- A. What traded on the rule's runners ({len(r):,} bets, {len(r) / days:.0f} a day) --")
    tab = pd.DataFrame({
        "morning volume": r["morning_vol"].quantile(q).round(0).to_numpy(),
        "pre-play volume": r["pp_vol"].quantile(q).round(0).to_numpy(),
        "morning share of pre-play %": (100 * r["morning_vol"] / r["pp_vol"].where(r["pp_vol"] > 0)).quantile(q)
        .round(1).to_numpy()}, index=[f"{int(x * 100)}th percentile" for x in q])
    print(tab.to_string())
    print("\nmedian morning / pre-play volume on the rule's runners by morning price:")
    print(r.groupby("band", observed=True).agg(bets=("band", "size"), morning_median=("morning_vol", "median"),
                                              preplay_median=("pp_vol", "median")).round(0).to_string())
    print(f"\nall runners, whole card: median morning volume £{d['morning_vol'].median():,.0f}, "
          f"pre-play £{d['pp_vol'].median():,.0f}")

    print("\n-- B. Morning capacity: stake = the smaller of the cap and a share of what traded on the runner all "
          "morning; the rule (>=22% above the forecast), traded out at BSP --")
    rows = []
    for share in (0.05, 0.10, 0.20):
        for cap in (25.0, 50.0, 100.0, 250.0):
            rows.append({"share_of_morning_volume": f"{share:.0%}", "cap_£": cap,
                         **run(d, rule.to_numpy(), "morning_vol", "net_m", share, cap, rng)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n-- C. Later in the day: backing at the pre-play average price, closed at BSP --")
    print("   £1 level, by how far the pre-play average price sat above the forecast:")
    rows = []
    for thr in (0.0, 0.10, 0.20, 0.35):
        on = (d["pp_vol"] >= 100) & (d["move_p"] > thr if thr == 0 else d["move_p"] >= thr)
        g = d[on]
        races = g.groupby("race")["net_p"].agg(["sum", "count"])
        s, n = races["sum"].to_numpy(), races["count"].to_numpy()
        idx = rng.integers(0, len(races), (N_BOOT, len(races)))
        boot = s[idx].sum(1) / n[idx].sum(1)
        rows.append({"above_forecast_by": ">0%" if thr == 0 else f">={np.expm1(thr):.0%}",
                     "bets_per_day": round(len(g) / days, 1), "roi_%": round(100 * s.sum() / n.sum(), 2),
                     "roi_lo": round(100 * np.percentile(boot, 5), 2), "roi_hi": round(100 * np.percentile(boot, 95), 2)})
    print(pd.DataFrame(rows).to_string(index=False))
    print("   capacity at >=22% above the forecast, stake = the smaller of the cap and a share of the pre-play volume:")
    on = ((d["pp_vol"] >= 100) & (d["move_p"] >= 0.2)).to_numpy()
    rows = []
    for share in (0.02, 0.05, 0.10):
        for cap in (50.0, 100.0, 250.0, 500.0):
            rows.append({"share_of_preplay_volume": f"{share:.0%}", "cap_£": cap,
                         **run(d, on, "pp_vol", "net_p", share, cap, rng)})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
