"""The owner's rule backtested on five months instead of two: November 2025 to March 2026 (read-only).

The to-win backtest of 30 Sep (research/queries/done/to_win_staking.py, run 36677259421) scored February-March 2026,
all the morning prices the database then held. It now holds Betfair's win rows from 27 Sep 2025 (the nightly load;
research/queries/done/place_model_stage3.py, run 37115359245), so the same test reads five months: iteration 105's
walk-forward forecasts against Betfair's morning WAP (the entry) and BSP (the close), the closing model fitted on the
months before each month it scores (at least 500 races), every horse with an expected CLV of at least 3% and GBP100
matched in the morning backed to win GBP250 at the morning price, 2% commission on each race's net (the owner's rate).
Nothing on or after 1 April 2026 is read.

1. The rule and its neighbours (ev >= 0/3/5/10%), and month by month: bets, turnover, expected CLV, CLV at the BSP,
   the result held to the off, with 90% race-bootstrap intervals.
2. Calibration by morning price band: the closing model's expected CLV against the CLV the backs took (stake-weighted)
   and each band's share of the turnover. Live, on 1-2 Oct, horses under 4.0 took 30% of the stake at -0.3% against
   the BSP.
3. The short-price variants the replay tests (research/queries/done/topup_policy_replay.py): a horse under 4.0 backed
   only at 6% or 10% expected CLV, or not at all.
4. A day's money capped (GBP1,500, about the account's balance; GBP3,000; GBP5,000): the day's backs taken in race
   order (roughly the live trader's: it sweeps the day's markets as their edges appear) or best expected CLV first (the
   order that gets the most value from a fixed sum), a back that does not fit held and the next tried. Stakes are not
   recycled within the day, so this is the tight case (by mid-morning the trader has committed the balance before any
   race has run).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research", "queries", "done"))
import betfair_prices as bp  # noqa: E402
import race_book_backtest as rbb  # noqa: E402
from model import race_book as rb  # noqa: E402

pd.set_option("display.width", 240)
pd.set_option("display.max_rows", 300)
rbb.START = "2025-09-27"                 # the forecasts' first day; the load reads the window at call time
COMM = 0.02
TO_WIN = 250.0
MIN_VOL = 100.0
BUDGETS = (1500.0, 3000.0, 5000.0)
BANDS = [1, 2, 3, 4, 6, 10, 20, 1e9]
LABELS = ["<2", "2-3", "3-4", "4-6", "6-10", "10-20", "20+"]

d = rbb.load()
s = rb.expected_clv_walk_forward(d, n_draws=300)
s = s.copy()
s["m"] = s.morningwap.astype(float)
s["stake"] = TO_WIN / (s.m - 1.0)
s["clv_r"] = s.m / s.bsp - 1.0
s["res_r"] = np.where(s.won, s.m - 1.0, -1.0)
s["vol_ok"] = s.morning_vol.fillna(0) >= MIN_VOL
s["band"] = pd.cut(s.m, BANDS, right=False, labels=LABELS)
s["off"] = s.race_time.map(bp.db_time_to_24h).fillna("23:59")
days = s.race_date.nunique()
print(f"scored: {s.month.min()} to {s.month.max()}, {s.race.nunique():,} races, {len(s):,} runners, {days} days")

live = (s.ev >= 0.03) & s.vol_ok
SEL = {
    "ev>=0": (s.ev >= 0.0) & s.vol_ok,
    "ev>=3% (the live rule)": live,
    "ev>=5%": (s.ev >= 0.05) & s.vol_ok,
    "ev>=10%": (s.ev >= 0.10) & s.vol_ok,
    "live, under 4.0 at 6%": live & ((s.m >= 4.0) | (s.ev >= 0.06)),
    "live, under 4.0 at 10%": live & ((s.m >= 4.0) | (s.ev >= 0.10)),
    "live, none under 4.0": live & (s.m >= 4.0),
}


def by_race(f: pd.DataFrame) -> pd.DataFrame:
    """One row a race: stakes, expected CLV, CLV at the BSP and the result, net of commission on the race's net."""
    g = f.assign(exp_v=f.stake * f.ev, clv_v=f.stake * f.clv_r, res_v=f.stake * f.res_r).groupby("race").agg(
        race_date=("race_date", "first"), month=("month", "first"), bets=("stake", "size"), staked=("stake", "sum"),
        expected=("exp_v", "sum"), clv=("clv_v", "sum"), result=("res_v", "sum"))
    g["clv"] -= COMM * g.clv.clip(lower=0)
    g["result"] -= COMM * g.result.clip(lower=0)
    return g


def line(name: str, g: pd.DataFrame, n_days: int | None = None) -> dict:
    if g.empty:
        return {"selection": name, "bets": 0}
    n_days = n_days or days
    clo, chi = rb.ratio_interval(g.clv.to_numpy(), g.staked.to_numpy())
    rlo, rhi = rb.ratio_interval(g.result.to_numpy(), g.staked.to_numpy())
    per_day = g.groupby("race_date").clv.sum()
    return {"selection": name, "bets": int(g.bets.sum()), "a_day": round(g.bets.sum() / n_days, 1),
            "turnover": round(g.staked.sum()), "a_day_staked": round(g.staked.sum() / n_days),
            "expected": round(g.expected.sum()), "clv": round(g.clv.sum()), "clv_pct": round(100 * g.clv.sum() / g.staked.sum(), 2),
            "clv_lo": round(100 * clo, 2), "clv_hi": round(100 * chi, 2), "clv_a_day": round(g.clv.sum() / n_days, 1),
            "days_clv_up": round(float((per_day > 0).mean()), 2), "result": round(g.result.sum()),
            "result_pct": round(100 * g.result.sum() / g.staked.sum(), 2), "res_lo": round(100 * rlo, 2),
            "res_hi": round(100 * rhi, 2)}


print("\n== 1. the selections, unconstrained (GBP; CLV and result net of 2% on each race's net)")
print(pd.DataFrame([line(n, by_race(s[m])) for n, m in SEL.items()]).to_string(index=False))
print("\n   the live rule by month")
g = by_race(s[live])
print(pd.DataFrame([line(mth, t, t.race_date.nunique()) for mth, t in g.groupby("month")])
      .rename(columns={"selection": "month"}).set_index("month").to_string())

print("\n== 2. the live rule by morning price band: the expected CLV against the CLV taken (stake-weighted, before commission)")
f = s[live]
rows = []
for band, t in f.groupby("band", observed=True):
    r = t.assign(num=t.stake * t.clv_r).groupby("race").agg(num=("num", "sum"), den=("stake", "sum"))
    lo, hi = rb.ratio_interval(r.num.to_numpy(), r.den.to_numpy())
    rows.append({"band": band, "bets": len(t), "share_of_turnover": round(t.stake.sum() / f.stake.sum(), 3),
                 "expected_clv": round(float((t.stake * t.ev).sum() / t.stake.sum()), 4),
                 "clv_taken": round(float((t.stake * t.clv_r).sum() / t.stake.sum()), 4), "lo90": round(lo, 4),
                 "hi90": round(hi, 4), "result": round(float((t.stake * t.res_r).sum() / t.stake.sum()), 4),
                 "clv_gbp": round(float((t.stake * t.clv_r).sum()))})
print(pd.DataFrame(rows).to_string(index=False))
print("\n   the same, horses under 4.0 by expected CLV (stake-weighted CLV taken)")
u = f[f.m < 4.0]
for lo_e, hi_e in ((0.03, 0.06), (0.06, 0.10), (0.10, 9.0)):
    t = u[(u.ev >= lo_e) & (u.ev < hi_e)]
    if len(t):
        print(f"   expected {lo_e:.0%}-{min(hi_e, 1):.0%}: {len(t):,} bets, GBP{t.stake.sum():,.0f} staked, expected "
              f"{(t.stake * t.ev).sum() / t.stake.sum():+.2%}, taken {(t.stake * t.clv_r).sum() / t.stake.sum():+.2%}")

print("\n== 4. a day's money capped: the backs that fit, in race order or best expected CLV first (no recycling)")
rows = []
for name in ("ev>=3% (the live rule)", "live, under 4.0 at 6%", "live, under 4.0 at 10%", "live, none under 4.0", "ev>=5%"):
    cand = s[SEL[name]]
    for budget in BUDGETS:
        for order in ("race order", "best first"):
            take = []
            for _, t in cand.groupby("race_date"):
                t = t.sort_values(["off", "race", "ev"], ascending=[True, True, False]) if order == "race order" \
                    else t.sort_values("ev", ascending=False)
                left = budget
                for ix, st in zip(t.index, t.stake.to_numpy()):
                    if st <= left:
                        take.append(ix)
                        left -= st
            g = by_race(s.loc[take])
            rows.append({"selection": name, "budget": budget, "order": order, "bets": int(g.bets.sum()),
                         "staked_a_day": round(g.staked.sum() / days), "clv_a_day": round(g.clv.sum() / days, 1),
                         "clv_pct": round(100 * g.clv.sum() / g.staked.sum(), 2),
                         "result_a_day": round(g.result.sum() / days, 1)})
print(pd.DataFrame(rows).to_string(index=False))
