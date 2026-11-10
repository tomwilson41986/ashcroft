"""Betfair's Expert Fee as a cost of the whole account (the consultant review, 9 Nov 2026: it was in no code).

What is modelled is the repository's own reading of Betfair's terms (reports/open_close/markets.md 5.1, read via a
search snippet of Betfair's FAQ on 1 Nov 2026; the support pages refused the fetch, so the account's own Expert Fee
page settles any point below):

- an account is charged only when its gross profit over its last 52 active weeks is over GBP25,000 and it is in
  lifetime profit;
- the rate is 20% while that 52-week gross profit is GBP25,000-100,000 and 40% over GBP100,000;
- "Expert Fee Due = (Weekly Gross P&L - Buffer) x Current Expert Fee Rate - Weekly Commission Generated", never
  below zero;
- losing weeks build the buffer (a third-party reading, not confirmed on Betfair's page).

One point is open in the sources: whether the rate applies to all of a charged week's gross profit (the FAQ's
formula, ``band="week"``, the default) or only to the part of the 52-week profit over each threshold (the guides'
reading, ``band="excess"``). Both are here; a figure quoted from this module should say which.

A fee on profit takes the same share of every winning week, so it does not change which bet has a positive expected
value or the order of the bets under a day limit: it is a cost of the programme (what the account keeps, and what a
losing strategy on the same account costs or saves), not an input to the trader's bar.

    fee_rate(gross_52w)                      the rate at a 52-week gross profit
    weekly_fees(weeks)                       the fee week by week from each week's gross P&L and commission
    project(weeks, n_weeks)                  the record carried forward at its mean week: when the fee starts, what
                                             it takes

Read-only arithmetic: nothing here touches a trader.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BANDS = ((25_000.0, 0.20), (100_000.0, 0.40))      # 52-week gross profit over this -> the rate
WINDOW = 52                                         # active weeks in the rolling profit


def fee_rate(gross_52w: float, bands=BANDS) -> float:
    """The Expert Fee rate at a gross profit over the last 52 active weeks (0 under the first threshold)."""
    rate = 0.0
    for over, r in bands:
        if gross_52w > over:
            rate = r
    return rate


def _excess(before: float, after: float, bands=BANDS) -> float:
    """The fee on the part of a week's profit lying over each threshold, when the 52-week profit moves from
    ``before`` to ``after`` (the guides' reading): each slice of profit at the rate of the band it falls in."""
    if after <= before:
        return 0.0
    edges = [b for b, _ in bands] + [np.inf]
    fee = 0.0
    for (lo, r), hi in zip(bands, edges[1:]):
        fee += r * max(0.0, min(after, hi) - max(before, lo))
    return fee


def weekly_fees(weeks: pd.DataFrame, band: str = "week", lifetime_before: float = 0.0, bands=BANDS,
                window: int = WINDOW) -> pd.DataFrame:
    """The Expert Fee week by week.

    ``weeks``: one row an active week in time order, ``gross`` (the week's P&L before commission) and ``commission``
    (the commission the week generated). ``lifetime_before`` is the account's gross P&L before the first row.
    Returns the rows with ``gross_52w`` (this week and the 51 active weeks before it), ``rate``, ``buffer`` (the
    losses still to be set against profit, after this week), ``fee`` and ``kept`` (gross less commission less fee).
    """
    if band not in ("week", "excess"):
        raise ValueError("band is 'week' or 'excess'")
    g = pd.to_numeric(weeks["gross"], errors="coerce").fillna(0.0).to_numpy(float)
    c = pd.to_numeric(weeks["commission"], errors="coerce").fillna(0.0).to_numpy(float)
    out = weeks.copy()
    buffer, lifetime = 0.0, float(lifetime_before)
    g52, rates, buffers, fees = [], [], [], []
    for i in range(len(g)):
        roll = float(g[max(0, i - window + 1):i + 1].sum())
        lifetime += g[i]
        rate = fee_rate(roll, bands) if lifetime > 0 else 0.0
        taxable = max(g[i] - buffer, 0.0)
        buffer = max(buffer - g[i], 0.0)             # a losing week adds its loss; a winning week uses the buffer up
        if band == "week":
            fee = max(taxable * rate - c[i], 0.0)
        else:
            fee = max(_excess(roll - taxable, roll, bands) - c[i], 0.0) if rate > 0 else 0.0
        g52.append(roll), rates.append(rate), buffers.append(buffer), fees.append(fee)
    out["gross_52w"], out["rate"], out["buffer"], out["fee"] = g52, rates, buffers, fees
    out["kept"] = g - c - out["fee"].to_numpy(float)
    return out


def project(weeks: pd.DataFrame, n_weeks: int = 52, band: str = "week", lifetime_before: float = 0.0,
            bands=BANDS) -> dict:
    """The record carried forward ``n_weeks`` at its own mean week (arithmetic, not a forecast: it assumes the
    record's mean week repeats). Returns the record's mean week, the first future week a fee is due (None if none
    within ``n_weeks``), and over the projected weeks the gross, commission, fee and what is kept."""
    rec = weeks[["gross", "commission"]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if rec.empty:
        return {"weeks_on_record": 0, "first_fee_week": None}
    mg, mc = float(rec.gross.mean()), float(rec.commission.mean())
    fut = pd.DataFrame({"gross": [mg] * n_weeks, "commission": [mc] * n_weeks})
    allw = weekly_fees(pd.concat([rec, fut], ignore_index=True), band=band, lifetime_before=lifetime_before,
                       bands=bands)
    f = allw.iloc[len(rec):].reset_index(drop=True)
    first = next((i + 1 for i, x in enumerate(f.fee) if x > 0), None)
    return {"weeks_on_record": int(len(rec)), "mean_week_gross": round(mg, 2), "mean_week_commission": round(mc, 2),
            "record_gross": round(float(rec.gross.sum()), 2), "band": band, "n_weeks": int(n_weeks),
            "first_fee_week": first,
            "projected_gross": round(float(f.gross.sum()), 2),
            "projected_commission": round(float(f.commission.sum()), 2),
            "projected_fee": round(float(f.fee.sum()), 2), "projected_kept": round(float(f.kept.sum()), 2),
            "fee_share_of_net": (round(float(f.fee.sum() / (f.gross.sum() - f.commission.sum())), 4)
                                 if f.gross.sum() - f.commission.sum() > 0 else None)}
