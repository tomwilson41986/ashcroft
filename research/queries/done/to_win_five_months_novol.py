"""The five-month backtest, read the way the live trader reads the market today: without matched volume (read-only).

research/queries/done/to_win_five_months.py (run 37115884197) fitted the closing model with each runner's morning
matched money, which only Betfair's live application key gives. The trader is on the delayed key: its feed carries no
matched money (every back on 1-2 Oct logged "closing CLV, feed without volume"), so it reads the closing model fitted
without volume and holds no runner to the GBP100 floor (trading/strategy.py _plan_closing_clv). Here are the same five
months, November 2025 to March 2026, read three ways on the same races:
  with volume, GBP100 floor       the backtest of run 37115884197 (what the live key would give)
  without volume, no floor        the trader on the delayed key, as it runs now
  without volume, GBP100 floor    the same model held to the floor, to split the model's loss from the floor's
Each: the walk-forward closing model (fitted on the months before each month with at least 500 races, every runner's
matched money set to nothing for the two without), the owner's rule (expected CLV >= 3%, GBP250 to win at the morning
price), 2% commission on each race's net, unconstrained and with a day's money capped, best expected CLV first (the
session's order). Nothing on or after 1 April 2026 is read.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research", "queries", "done"))
import race_book_backtest as rbb  # noqa: E402
from model import race_book as rb  # noqa: E402

pd.set_option("display.width", 240)
rbb.START = "2025-09-27"
COMM, TO_WIN, MIN_VOL = 0.02, 250.0, 100.0
BUDGETS = (1500.0, 3000.0, 5000.0)

d = rbb.load()
vol = rb.expected_clv_walk_forward(d, n_draws=300)
blind = rb.expected_clv_walk_forward(d.assign(morning_vol=0.0), n_draws=300)
blind["morning_vol"] = d.loc[blind.index, "morning_vol"].to_numpy()          # the floor still reads the real money
ARMS = {"with volume, GBP100 floor": vol[(vol.ev >= 0.03) & (vol.morning_vol.fillna(0) >= MIN_VOL)],
        "without volume, no floor (the trader now)": blind[blind.ev >= 0.03],
        "without volume, GBP100 floor": blind[(blind.ev >= 0.03) & (blind.morning_vol.fillna(0) >= MIN_VOL)]}
days = vol.race_date.nunique()
print(f"scored {vol.month.min()} to {vol.month.max()}: {vol.race.nunique():,} races, {days} days")


def by_race(f: pd.DataFrame) -> pd.DataFrame:
    m = f.morningwap.to_numpy(float)
    st = TO_WIN / (m - 1.0)
    g = f.assign(stake=st, clv_v=st * (m / f.bsp.to_numpy(float) - 1.0),
                 res_v=np.where(f.won.astype(bool), st * (m - 1.0), -st)).groupby("race").agg(
        race_date=("race_date", "first"), month=("month", "first"), bets=("stake", "size"), staked=("stake", "sum"),
        clv=("clv_v", "sum"), result=("res_v", "sum"))
    g["clv"] -= COMM * g.clv.clip(lower=0)
    g["result"] -= COMM * g.result.clip(lower=0)
    return g


rows, months = [], []
for name, f in ARMS.items():
    g = by_race(f)
    lo, hi = rb.ratio_interval(g.clv.to_numpy(), g.staked.to_numpy())
    rows.append({"arm": name, "bets_a_day": round(g.bets.sum() / days, 1), "staked_a_day": round(g.staked.sum() / days),
                 "clv_pct": round(100 * g.clv.sum() / g.staked.sum(), 2), "lo90": round(100 * lo, 2),
                 "hi90": round(100 * hi, 2), "clv_a_day": round(g.clv.sum() / days, 1),
                 "result_pct": round(100 * g.result.sum() / g.staked.sum(), 2)})
    for mth, t in g.groupby("month"):
        months.append({"arm": name, "month": mth, "clv_pct": round(100 * t.clv.sum() / t.staked.sum(), 2),
                       "clv_a_day": round(t.clv.sum() / t.race_date.nunique(), 1)})
    for budget in BUDGETS:
        take = []
        for _, t in f.assign(stake=TO_WIN / (f.morningwap - 1.0)).groupby("race_date"):
            left = budget
            for ix, st in zip(t.sort_values("ev", ascending=False).index, t.sort_values("ev", ascending=False).stake):
                if st <= left:
                    take.append(ix)
                    left -= st
        b = by_race(f.loc[take])
        rows.append({"arm": f"  {name}, GBP{budget:,.0f} a day, best first", "bets_a_day": round(b.bets.sum() / days, 1),
                     "staked_a_day": round(b.staked.sum() / days), "clv_pct": round(100 * b.clv.sum() / b.staked.sum(), 2),
                     "clv_a_day": round(b.clv.sum() / days, 1),
                     "result_pct": round(100 * b.result.sum() / b.staked.sum(), 2)})
print("\n== the owner's rule three ways (GBP; CLV and result net of 2% on each race's net)")
print(pd.DataFrame(rows).to_string(index=False))
print("\n== by month")
print(pd.DataFrame(months).pivot(index="month", columns="arm", values=["clv_pct", "clv_a_day"]).to_string())
