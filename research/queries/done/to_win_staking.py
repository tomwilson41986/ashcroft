"""The owner's staking (30 Sep): back every selected horse to win the same amount, GBP250, and judge the portfolio.

The owner: "I back each horse to win the same amount. GBP250. It doesn't matter to me which individual horse
wins. Only that we have positive expectation vs the market and profitability over time across all runners."
So the stake on a horse at price m is 250 / (m - 1): each winner pays GBP250 before commission.

Same data and closing model as research/queries/done/race_book_backtest.py: January-March 2026 only (nothing on
or after 1 April), the served blend's walk-forward forecasts against Betfair's morning volume-weighted price
(the entry) and the BSP (the close); the closing model fitted on the months before each test month; February
and March scored. Horses only where at least GBP100 was matched in the morning.

Selections, each staked to win GBP250:
  ev>=0, ev>=3%, ev>=5%, ev>=10%   every horse whose expected CLV under the closing model clears the bar
  rule22                            the pre-registered rule (model at least 22% shorter than the morning price)
  book                              the owner's book: horses in order of expected CLV, each at least -3%, kept
                                    while the race's expected value at these stakes is at least +3% of them
Settled two ways: closed out at the BSP (CLV, what the prices did) and held to the result at the morning price.
Commission 5% on each race's net winnings. Reported in pounds, per GBP staked, by day (the curve, the worst
drawdown, the longest run of losing days) and by price band; level stakes on the same horses beside them.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "research/queries/done")
from model import race_book as rb  # noqa: E402
from race_book_backtest import MIN_VOL, load  # noqa: E402

TO_WIN = 250.0
LEVEL = 50.0
COMMISSION = rb.COMMISSION


def selections(e: np.ndarray, m: np.ndarray, f: np.ndarray, allow: np.ndarray) -> dict[str, np.ndarray]:
    """Which horses each selection backs (booleans), from the expected CLV e and the prices."""
    out = {f"ev>={int(t * 100)}%" if t else "ev>=0": (e >= t) & allow for t in (0.0, 0.03, 0.05, 0.10)}
    out["rule22"] = (np.log(m / f) >= 0.2) & allow
    # the owner's book at to-win stakes: by expected CLV, each >= -3%, while the race keeps >= +3% of its stakes
    stake = TO_WIN / (m - 1)
    chosen = np.zeros(len(m), bool)
    for i in np.argsort(-e):
        if e[i] < -0.03:
            break
        if not allow[i]:
            continue
        trial = chosen.copy()
        trial[i] = True
        if (stake[trial] @ e[trial]) < 0.03 * stake[trial].sum():
            break
        chosen = trial
    out["book"] = chosen
    return out


def settle(sel: np.ndarray, stake: np.ndarray, m: np.ndarray, bsp: np.ndarray, won: np.ndarray) -> tuple[float, float]:
    """(CLV closed out at the BSP, profit held to the result) for one race, net of commission on the race."""
    s = np.where(sel, stake, 0.0)
    clv = float(s @ (m / bsp - 1.0))
    res = float(-s.sum() + (s[won] * m[won]).sum())
    return clv - COMMISSION * max(clv, 0.0), res - COMMISSION * max(res, 0.0)


def boot_roi(num: np.ndarray, den: np.ndarray, n: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(num), (n, len(num)))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-9)
    return float(np.percentile(r, 5)), float(np.percentile(r, 95))


def day_stats(days: pd.Series) -> dict:
    """The daily profit curve: share of days up, worst drawdown, longest run of losing days, worst day."""
    cum = days.cumsum()
    dd = float((cum - cum.cummax()).min())
    run = best = 0
    for v in days.to_numpy():
        run = run + 1 if v < 0 else 0
        best = max(best, run)
    return {"days": len(days), "up": float((days > 0).mean()), "drawdown": dd, "losing_run": best,
            "worst_day": float(days.min()), "best_day": float(days.max())}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecasts", help="local copy of the forecasts (default: the iteration 105 artifact)")
    ap.add_argument("--extract", help="local Betfair export (default: the database)")
    args = ap.parse_args()
    d = load(args.forecasts, args.extract)
    months = sorted(d["month"].unique())
    rows = []
    for test_month in months[1:]:
        train, test = d[d["month"] < test_month], d[d["month"] == test_month]
        model = rb.fit_closing_model(rb.closing_inputs(train), train["bsp"], rb.race_codes(train["race"]))
        rng = np.random.default_rng(int(test_month.replace("-", "")))
        for race, r in test.groupby("race", sort=False):
            r = r.reset_index(drop=True)
            inputs = rb.closing_inputs(r)
            g = np.zeros(len(r), int)
            qhat, sig = model.predict(inputs, g), model.sigma(inputs)
            m, bsp = r["morningwap"].to_numpy(float), r["bsp"].to_numpy(float)
            f, won = r["predicted_bfsp"].to_numpy(float), r["won"].to_numpy(bool)
            vol = r["morning_vol"].fillna(0).to_numpy(float)
            allow = vol >= MIN_VOL
            draws = rb.closing_draws(qhat, sig, book=model.book, n_draws=300, rng=rng)
            e = (m * draws).mean(axis=0) - 1.0
            to_win = TO_WIN / (m - 1.0)
            level = np.full(len(m), LEVEL)
            for name, sel in selections(e, m, f, allow).items():
                if not sel.any():
                    continue
                clv_w, res_w = settle(sel, to_win, m, bsp, won)
                clv_l, res_l = settle(sel, level, m, bsp, won)
                rows.append({"selection": name, "race": race, "date": r["race_date"].iat[0], "month": test_month,
                             "bets": int(sel.sum()), "staked": float(to_win[sel].sum()), "expected": float(to_win[sel] @ e[sel]),
                             "clv": clv_w, "result": res_w, "staked_level": float(level[sel].sum()),
                             "clv_level": clv_l, "result_level": res_l,
                             "thin": int((sel & (to_win > 0.5 * vol)).sum()),
                             "bands": [(float(m[i]), float(to_win[i]), float(m[i] / bsp[i] - 1), float(won[i] * m[i] * to_win[i] - to_win[i]))
                                       for i in np.flatnonzero(sel)]})
    res = pd.DataFrame(rows)
    print(f"February-March, stakes to win GBP{TO_WIN:.0f} on each horse (level GBP{LEVEL:.0f} beside them)")
    head = (f"  {'selection':10s} {'bets':>6s} {'races':>6s} {'turnover':>10s} {'exp. CLV':>9s} {'CLV':>9s} {'CLV %':>7s} "
            f"{'result':>9s} {'result %':>9s} {'result % 90%':>18s} {'days up':>7s} {'drawdown':>9s} {'losing run':>10s} "
            f"{'worst day':>9s} {'level CLV %':>11s} {'level res %':>11s} {'stake>half vol':>14s}")
    print(head)
    for name, s in res.groupby("selection", sort=False):
        lo, hi = boot_roi(s["result"].to_numpy(), s["staked"].to_numpy())
        days = s.groupby("date")["result"].sum()
        st = day_stats(days)
        print(f"  {name:10s} {s['bets'].sum():6,d} {len(s):6,d} {s['staked'].sum():10,.0f} {s['expected'].sum():+9,.0f} "
              f"{s['clv'].sum():+9,.0f} {100 * s['clv'].sum() / s['staked'].sum():+6.2f}% {s['result'].sum():+9,.0f} "
              f"{100 * s['result'].sum() / s['staked'].sum():+8.2f}% ({100 * lo:+6.2f} to {100 * hi:+6.2f}) "
              f"{st['up']:7.0%} {st['drawdown']:+9,.0f} {st['losing_run']:10d} {st['worst_day']:+9,.0f} "
              f"{100 * s['clv_level'].sum() / s['staked_level'].sum():+10.2f}% {100 * s['result_level'].sum() / s['staked_level'].sum():+10.2f}% "
              f"{s['thin'].sum() / s['bets'].sum():13.1%}")
    for name in ("ev>=0", "ev>=5%", "book"):
        s = res[res["selection"] == name]
        print(f"\n{name}: by month")
        for mth, t in s.groupby("month"):
            print(f"  {mth}  bets {t['bets'].sum():5,d}  turnover {t['staked'].sum():9,.0f}  CLV {t['clv'].sum():+8,.0f} "
                  f"({100 * t['clv'].sum() / t['staked'].sum():+.2f}%)  result {t['result'].sum():+8,.0f} "
                  f"({100 * t['result'].sum() / t['staked'].sum():+.2f}%)")
        b = pd.DataFrame([x for lst in s["bands"] for x in lst], columns=["price", "stake", "clv", "result"])
        b["band"] = pd.cut(b["price"], [1, 3, 5, 8, 13, 21, 1001], labels=["<3", "3-5", "5-8", "8-13", "13-21", "21+"])
        print(f"{name}: by morning price band (before commission)")
        for band, t in b.groupby("band", observed=True):
            print(f"  {str(band):6s} bets {len(t):5,d}  turnover {t['stake'].sum():9,.0f}  "
                  f"CLV {(t['stake'] * t['clv']).sum():+8,.0f} ({100 * (t['stake'] * t['clv']).sum() / t['stake'].sum():+6.2f}%)  "
                  f"result {t['result'].sum():+8,.0f} ({100 * t['result'].sum() / t['stake'].sum():+6.2f}%)")
        days = s.groupby("date")[["clv", "result"]].sum().cumsum()
        print(f"{name}: cumulative by week (CLV, result)")
        wk = days.groupby(pd.to_datetime(days.index).to_period("W")).last()
        print("  " + "  ".join(f"{str(p.start_time.date())[5:]} {v.clv:+,.0f}/{v.result:+,.0f}" for p, v in wk.iterrows()))


if __name__ == "__main__":
    main()
