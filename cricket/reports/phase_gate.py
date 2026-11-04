"""The Phase 1 gate (build spec 8), read off the paper ledger, the feed record and the books.

All must hold: CLV positive over at least 500 paper bets against the price at delay-plus-30-seconds and against the
pre-match close; paper P&L positive after commission; the feed's lag measured and under the bet delay;
transactions under 5,000 in every clock hour. Anything that cannot be measured yet is reported as no data found,
never as a pass.

Feed lag, per delivery: its receipt time minus the time of Betfair's first price move after the previous delivery's
receipt (a move: any runner's best back or lay changing between consecutive open books). Positive: the market moved
before our feed told us.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

MIN_BETS = 500


def _t_stat(x: pd.Series) -> float | None:
    x = x.dropna()
    if len(x) < 2 or x.std(ddof=1) == 0:
        return None
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def feed_lag(receipts: list[dt.datetime], books) -> pd.DataFrame:
    """Lag per delivery (seconds) from the receipt times and one market's books in time order."""
    moves = []
    prev = None
    for b in books:
        if not b.open:
            prev = None
            continue
        key = tuple((s, r.back[0][0] if r.back else None, r.lay[0][0] if r.lay else None)
                    for s, r in sorted(b.runners.items()))
        if prev is not None and key != prev:
            moves.append(b.pt)
        prev = key
    rows = []
    for i, t in enumerate(receipts):
        since = receipts[i - 1] if i else None
        first = next((m for m in moves if (since is None or m > since) and m <= t + dt.timedelta(seconds=60)), None)
        rows.append({"receipt": t, "first_move": first,
                     "lag_s": (t - first).total_seconds() if first is not None else np.nan})
    return pd.DataFrame(rows)


def gate(ledger: pd.DataFrame, max_tx_hour: int, lag: pd.DataFrame | None, bet_delays: list[int],
         commission_paid: float) -> dict:
    m = ledger[ledger.matched > 0] if len(ledger) else ledger
    n = int(len(m))
    res = {"bets_matched": n, "orders": int(len(ledger))}
    res["clv_30_mean"] = float(m.clv_30.mean()) if n and m.clv_30.notna().any() else None
    res["clv_30_t"] = _t_stat(m.clv_30) if n else None
    res["clv_close_mean"] = float(m.clv_close.mean()) if n and m.clv_close.notna().any() else None
    res["clv_close_t"] = _t_stat(m.clv_close) if n else None
    res["pnl_before_commission"] = float(m.pnl.sum()) if n else 0.0
    res["commission"] = float(commission_paid)
    res["pnl_after_commission"] = res["pnl_before_commission"] - res["commission"]
    res["max_transactions_hour"] = int(max_tx_hour)
    if lag is not None and lag.lag_s.notna().any():
        res["feed_lag_median_s"] = float(lag.lag_s.median())
        res["feed_lag_p90_s"] = float(lag.lag_s.quantile(0.9))
    else:
        res["feed_lag_median_s"] = None
    res["bet_delay_min_s"] = min(bet_delays) if bet_delays else None
    checks = {
        f"at least {MIN_BETS} paper bets": n >= MIN_BETS,
        "CLV positive at delay + 30 s": res["clv_30_mean"] is not None and res["clv_30_mean"] > 0,
        "CLV positive against the pre-match close": res["clv_close_mean"] is not None and res["clv_close_mean"] > 0,
        "paper P&L positive after commission": n > 0 and res["pnl_after_commission"] > 0,
        "feed lag measured and under the bet delay": (res["feed_lag_median_s"] is not None
                                                      and res["bet_delay_min_s"] is not None
                                                      and res["feed_lag_p90_s"] < res["bet_delay_min_s"]),
        "transactions under 5,000 in every hour": max_tx_hour < 5000,
    }
    res["checks"] = checks
    res["pass"] = all(checks.values())
    return res


def report_md(res: dict, source: str) -> str:
    def f(x, fmt="{:.4f}"):
        return "no data found" if x is None else fmt.format(x)
    L = ["# Cricket trader - Phase 1 (paper) gate", "", f"Source: {source}.", "",
         f"Gate: **{'PASS' if res['pass'] else 'NOT MET'}**", "", "| check | holds |", "|---|---|"]
    L += [f"| {k} | {'yes' if v else 'no'} |" for k, v in res["checks"].items()]
    L += ["", "| measure | value |", "|---|---:|",
          f"| orders / matched bets | {res['orders']:,} / {res['bets_matched']:,} |",
          f"| CLV at delay + 30 s, mean (t) | {f(res['clv_30_mean'])} ({f(res['clv_30_t'], '{:.2f}')}) |",
          f"| CLV against the close, mean (t) | {f(res['clv_close_mean'])} ({f(res['clv_close_t'], '{:.2f}')}) |",
          f"| P&L before commission, GBP | {res['pnl_before_commission']:.2f} |",
          f"| commission, GBP | {res['commission']:.2f} |",
          f"| P&L after commission, GBP | {res['pnl_after_commission']:.2f} |",
          f"| feed lag median, s | {f(res['feed_lag_median_s'], '{:.2f}')} |",
          f"| smallest bet delay, s | {f(res['bet_delay_min_s'], '{}')} |",
          f"| most transactions in a clock hour | {res['max_transactions_hour']:,} |", ""]
    return "\n".join(L)
