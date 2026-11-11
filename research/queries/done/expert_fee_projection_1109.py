"""Betfair's Expert Fee on the account's own record (read-only; ledger `expert-fee-projection-1109`).

The fee was in no code (the consultant review, 9 Nov). research/expert_fee.py models the repository's reading of
Betfair's terms (reports/open_close/markets.md 5.1); this reads every live trader's ledger in S3 as the dashboard
does, makes the account's weeks (Monday to Sunday) of gross P&L and commission, and carries the record forward at
its own mean week: the week the fee would start and what it would take in the next 52, on both readings of the open
point (the rate on all of a charged week's profit, or only on the profit over each threshold).

Arithmetic on the record, not a forecast: it assumes the mean week repeats, and the terms themselves were read from a
search snippet (the account's own Expert Fee page settles them). The horse trader's gross is its ledger's settle rows
and its commission rows (else 2% of a winning market's net); the other live traders' ledgers hold only a settled
P&L, taken here as gross with no commission (they are small). Nothing here touches a trader.
"""
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from dashboard import traders as dt  # noqa: E402
from research import expert_fee as ef  # noqa: E402

pd.set_option("display.width", 200)
src = dt.S3Source()
today = date.today()
rows = []
for t in dt.TRADERS:
    if t.mode != "live":
        continue
    n0 = len(rows)
    for day in src.days(t.kind, t.base + "/"):
        if day >= today.isoformat():
            continue
        if t.reader is dt.horse_markets:
            led = dt._csv(src.get(t.kind, f"{t.base}/{day}/ledger.csv"))
            if led is None or "event" not in led.columns:
                continue
            for c in ("pnl", "commission", "selection_id"):
                led[c] = dt._num(led[c]) if c in led.columns else np.nan
            st = led[led.event == "settle"].drop_duplicates(dt.K, keep="last")
            gross = st.groupby("market_id").pnl.sum()
            com = led[led.event == "commission"].groupby("market_id").commission.sum().reindex(gross.index).fillna(0.0)
            com = com.where(com > 0, dt.RATE * gross.clip(lower=0))
            rows.append({"trader": t.id, "day": day, "gross": float(gross.sum()), "commission": float(com.sum()),
                         "markets": int(len(gross))})
        else:
            m, _ = t.reader(src, t.kind, t.base, day)
            if m is None or not len(m):
                continue
            s = m[m.settled.astype(bool)]
            rows.append({"trader": t.id, "day": day, "gross": float(pd.to_numeric(s.pnl, errors="coerce").sum()),
                         "commission": 0.0, "markets": int(len(s))})
    print(f"{t.id}: {len(rows) - n0} days read from {src.where(t.kind, t.base)}")
if not rows:
    raise SystemExit("no data found: no live ledger read")
D = pd.DataFrame(rows)
D["week"] = pd.to_datetime(D.day).dt.to_period("W-SUN").astype(str)
print("\n== the record by trader")
print(D.groupby("trader").agg(days=("day", "nunique"), first=("day", "min"), last=("day", "max"),
                              markets=("markets", "sum"), gross=("gross", "sum"),
                              commission=("commission", "sum")).round(2).to_string())
W = D.groupby("week", as_index=False).agg(days=("day", "nunique"), gross=("gross", "sum"),
                                          commission=("commission", "sum"))
full = W[W.days >= 5] if (W.days >= 5).sum() >= 3 else W        # part weeks at either end would bend the mean week
print(f"\n== the account's weeks (Monday-Sunday): {len(W)} with a settled live market; the mean week is taken over "
      f"the {len(full)} with 5 or more days")
print(ef.weekly_fees(W).round(2).to_string(index=False))
gross_now = float(W.gross.sum())
mean_w = float(full.gross.mean())
print(f"\nrecord: gross GBP{gross_now:,.2f}, commission GBP{W.commission.sum():,.2f} over {D.day.nunique()} days; "
      f"mean full week gross GBP{mean_w:,.2f}")
if mean_w > 0 and gross_now < 25_000:
    print(f"GBP25,000 of gross profit is GBP{25_000 - gross_now:,.2f} away: {(25_000 - gross_now) / mean_w:.1f} "
          f"weeks at the mean week")
for band in ("week", "excess"):
    hist = W[["gross", "commission"]].copy()
    fut = pd.DataFrame({"gross": [mean_w] * 52, "commission": [float(full.commission.mean())] * 52})
    allw = ef.weekly_fees(pd.concat([hist, fut], ignore_index=True), band=band)
    f = allw.iloc[len(hist):].reset_index(drop=True)
    first = next((i + 1 for i, x in enumerate(f.fee) if x > 0), None)
    net = float(f.gross.sum() - f.commission.sum())
    print(f"\n== the next 52 weeks at the mean week, the rate on "
          f"{'all of a charged week' if band == 'week' else 'the profit over each threshold'}: first fee in week "
          f"{first}; gross GBP{f.gross.sum():,.0f}, commission GBP{f.commission.sum():,.0f}, Expert Fee "
          f"GBP{f.fee.sum():,.0f} ({f.fee.sum() / net if net > 0 else float('nan'):.1%} of the profit after "
          f"commission), kept GBP{f.kept.sum():,.0f}; 52-week gross at the end GBP{f.gross_52w.iloc[-1]:,.0f} "
          f"(rate {f.rate.iloc[-1]:.0%})")
