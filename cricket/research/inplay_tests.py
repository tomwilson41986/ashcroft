"""In-play market-only edge tests on the BASIC panel (reports/cricket/research_log.md, H1 and H2).

    python -m cricket.research.inplay_tests --root data/cricket/store            # discovery markets only
    python -m cricket.research.inplay_tests --root data/cricket/store --holdout   # a frozen strategy, once

Bets are priced one tick worse than the last traded price and pay 5% commission on each market's net winnings.
Intervals resample whole markets (matches).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from cricket.market.inplay_panel import tick_down, tick_up
from sources.common import Store

COMMISSION = 0.05


# ---------------------------------------------------------------------------------------------------------------
# bets and their settlement
# ---------------------------------------------------------------------------------------------------------------

def settle(bets: pd.DataFrame, c: float = COMMISSION) -> pd.DataFrame:
    """Each bet's gross P&L, then commission on each market's net winnings. Columns needed: market_id, side
    (BACK/LAY), price, stake (the backer's stake), won (the selection won)."""
    b = bets.copy()
    back = b.side == "BACK"
    b["gross"] = np.where(back, np.where(b.won == 1, b.stake * (b.price - 1), -b.stake),
                          np.where(b.won == 1, -b.stake * (b.price - 1), b.stake))
    b["risk"] = np.where(back, b.stake, b.stake * (b.price - 1))
    net = b.groupby("market_id").gross.transform("sum")
    share = np.where(net > 0, b.gross.clip(lower=0) / b.groupby("market_id").gross.transform(
        lambda s: s.clip(lower=0).sum()).replace(0, np.nan), 0)
    b["commission"] = np.nan_to_num(share) * np.where(net > 0, net * c, 0)
    b["pnl"] = b.gross - b.commission
    return b


def summary(b: pd.DataFrame, n_boot: int = 4000, seed: int = 0) -> dict:
    if b.empty:
        return {"bets": 0}
    m = b.groupby("market_id").agg(pnl=("pnl", "sum"), risk=("risk", "sum"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(m), size=(n_boot, len(m)))
    roi = m.pnl.to_numpy()[idx].sum(1) / m.risk.to_numpy()[idx].sum(1)
    lo, hi = np.quantile(roi, [0.025, 0.975])
    return {"bets": int(len(b)), "markets": int(len(m)), "pnl": float(b.pnl.sum()), "risk": float(b.risk.sum()),
            "roi": float(b.pnl.sum() / b.risk.sum()), "roi_lo": float(lo), "roi_hi": float(hi),
            "win_rate": float(b.won.mean()), "mean_price": float(b.price.mean())}


# ---------------------------------------------------------------------------------------------------------------
# H1: in-play favourite-longshot bias
# ---------------------------------------------------------------------------------------------------------------

def usable(p: pd.DataFrame, max_age: float) -> pd.DataFrame:
    return p[~p.suspended & (p.age_a <= max_age) & (p.age_b <= max_age)]


def long_view(p: pd.DataFrame) -> pd.DataFrame:
    """Each usable minute as two rows, one a runner: its implied probability, price and result."""
    a = p.assign(sel="a", prob=p.p_a, ltp=p.ltp_a, won=p.a_won)
    b = p.assign(sel="b", prob=1 - p.p_a, ltp=p.ltp_b, won=1 - p.a_won)
    return pd.concat([a, b], ignore_index=True)[["market_id", "minute", "sel", "prob", "ltp", "won"]]


def h1_bets(p: pd.DataFrame, below: float = 0.20, side: str = "LAY", liability: float = 10.0) -> pd.DataFrame:
    lv = long_view(usable(p, 5.0))
    first = lv[lv.prob < below].sort_values(["market_id", "minute"]).groupby("market_id").head(1)
    price = first.ltp.map(tick_up if side == "LAY" else tick_down)
    stake = liability / (price - 1) if side == "LAY" else pd.Series(liability, index=first.index)
    return first.assign(side=side, price=price, stake=stake)


def calibration(p: pd.DataFrame, every: int = 10) -> pd.DataFrame:
    """Implied probability against the result, at checkpoints every ``every`` minutes (usable minutes)."""
    lv = long_view(usable(p, 5.0))
    lv = lv[lv.minute % every == 0]
    bins = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001]
    lv["band"] = pd.cut(lv.prob, bins, right=False)
    g = lv.groupby("band", observed=True).agg(n=("won", "size"), markets=("market_id", "nunique"),
                                              implied=("prob", "mean"), won=("won", "mean"))
    return g.reset_index()


# ---------------------------------------------------------------------------------------------------------------
# H2: overreaction to big moves
# ---------------------------------------------------------------------------------------------------------------

def h2_bets(p: pd.DataFrame, jump: float = 0.10, contrarian: bool = True, stake: float = 10.0) -> pd.DataFrame:
    p = p.sort_values(["market_id", "minute"])
    fresh = (~p.suspended) & (p.age_a <= 1) & (p.age_b <= 1)
    q = p.assign(fresh=fresh)
    g = q.groupby("market_id", sort=False)
    q["d"] = g.p_a.diff()
    q["prev_minute"] = g.minute.shift(1)
    q["prev_fresh"] = g.fresh.shift(1)
    jumps = q[(q.d.abs() >= jump) & q.fresh & q.prev_fresh.fillna(False).astype(bool)
              & (q.minute - q.prev_minute == 1)]
    nxt = q.set_index(["market_id", "minute"])
    rows = []
    for j in jumps.itertuples():
        key = (j.market_id, j.minute + 1)
        if key not in nxt.index:
            continue
        r = nxt.loc[key]
        if not r.fresh:
            continue
        a_fell = j.d < 0
        back_a = a_fell if contrarian else not a_fell
        ltp = r.ltp_a if back_a else r.ltp_b
        rows.append({"market_id": j.market_id, "minute": j.minute + 1, "jump": j.d, "side": "BACK",
                     "price": tick_down(ltp), "stake": stake, "won": r.a_won if back_a else 1 - r.a_won,
                     "prob_then": r.p_a if back_a else 1 - r.p_a})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------

def run(panel: pd.DataFrame) -> dict:
    out = {"markets": int(panel.market_id.nunique())}
    out["H1_primary_lay_under_0.20"] = summary(settle(h1_bets(panel)))
    out["H1_exploratory_back_under_0.20"] = summary(settle(h1_bets(panel, side="BACK")))
    out["H1_calibration"] = calibration(panel).to_dict("records")
    out["H2_primary_contrarian_0.10"] = summary(settle(h2_bets(panel)))
    for j in (0.05, 0.20):
        out[f"H2_exploratory_contrarian_{j}"] = summary(settle(h2_bets(panel, j)))
    for j in (0.05, 0.10, 0.20):
        out[f"H2_exploratory_momentum_{j}"] = summary(settle(h2_bets(panel, j, contrarian=False)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root")
    ap.add_argument("--holdout", action="store_true", help="the holdout markets: only for a frozen strategy")
    a = ap.parse_args(argv)
    panel = Store(root=a.root).get_parquet("cricket/eval/inplay_panel.parquet")
    panel = panel[panel.holdout] if a.holdout else panel[~panel.holdout]
    print(json.dumps(run(panel), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
