"""Scoring the drift forecasts: accuracy against the no-change baseline, and the hedged trade's P&L.

Accuracy (``accuracy``): RMSE and R² of log(close/open) (R² against the no-change forecast, 1 - SSE / SSE_0, and
the plain R² about the month's mean), the direction hit rate (the share of selections whose close moved the way
the forecast said), and the RMSE of the share change ``y_share`` from the forecast prices made into shares.

The hedged trade (``hedged_trades``): for each selection whose forecast move clears a threshold, GBP1 at the open
and the position closed at the close, greened so the result does not matter (model/race_book.close_payoffs):

- forecast shorter (y_hat < -t): back 1 at the open o, lay o / c at the close c: o / c - 1 whatever wins;
- forecast longer (y_hat > t): lay 1 at o, back at c: 1 - o / c.

Commission as the exchange charges it, on each market's net profit (research/queries/done/staking_backtest.py and
race_book.settle_close: a book pays once, on its net): 2% (the owner's rate) and 5% as the sensitivity. The spread:
the same trades a tick worse at each end on Betfair's ladder (trading/exchange.py: 0.01 to 2, 0.02 to 3, 0.05 to 4,
0.1 to 6, 0.2 to 10, 0.5 to 20, 1 to 30, 2 to 50, 5 to 100, 10 to 1000): a back takes the price a tick under, a lay
the price a tick over, at the open and again at the close.

``summarise`` gives ROI (profit over GBP staked), a 90% bootstrap interval by day (race_book.ratio_interval on the
days' sums), bets a day, the worst drawdown of the cumulative daily P&L, and the CLV-style share of trades whose
close moved the forecast way; ``sweep`` runs the thresholds; ``by_group`` breaks a threshold down by price band
and by market.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.race_book import ratio_interval
from trading.exchange import tick_down, tick_up

from .features import price_band

THRESHOLDS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
COMMISSION = 0.02
COMMISSION_ALT = 0.05


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------

def accuracy(d: pd.DataFrame, pred_col: str) -> dict:
    """RMSE, R² against no-change and about the mean, direction hit rate, share-change RMSE, for one forecast."""
    y, yhat = d["y"].to_numpy(float), d[pred_col].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    if len(y) == 0:
        return {"n": 0}
    sse, sse0, sst = float(((y - yhat) ** 2).sum()), float((y ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    moved = yhat != 0
    hit = float((np.sign(y[moved]) == np.sign(yhat[moved])).mean()) if moved.any() else np.nan
    # the forecast as shares: p_hat = exp(-yhat) / open, normalised in the market
    p_hat = np.exp(-yhat) / d.loc[ok, "open_price"].to_numpy(float)
    share_hat = p_hat / pd.Series(p_hat).groupby(d.loc[ok, "market_id"].to_numpy()).transform("sum").to_numpy()
    share_change = share_hat - d.loc[ok, "p_share"].to_numpy(float)
    ys = d.loc[ok, "y_share"].to_numpy(float)
    return {"n": int(len(y)), "rmse": float(np.sqrt(sse / len(y))), "rmse_no_change": float(np.sqrt(sse0 / len(y))),
            "r2_vs_no_change": float(1.0 - sse / sse0) if sse0 > 0 else np.nan,
            "r2": float(1.0 - sse / sst) if sst > 0 else np.nan,
            "direction_hit": hit, "forecast_moved_share": float(moved.mean()),
            "rmse_share": float(np.sqrt(np.mean((ys - share_change) ** 2))),
            "rmse_share_no_change": float(np.sqrt(np.mean(ys ** 2)))}


def accuracy_table(scored: pd.DataFrame, pred_cols: list[str]) -> pd.DataFrame:
    rows = {c.removeprefix("pred_"): accuracy(scored, c) for c in pred_cols}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# the hedged trade
# ---------------------------------------------------------------------------

def tick_under(price: float) -> float:
    """The ladder price one full tick under ``price`` (a price on the ladder moves a tick; one between two ticks
    moves to the tick below the one under it: the conservative reading of a volume-weighted average)."""
    return tick_down(tick_down(price) - 1e-6)


def tick_over(price: float) -> float:
    return tick_up(tick_up(price) + 1e-6)


def _with_spread(open_price: np.ndarray, close_price: np.ndarray, side: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The prices a tick worse at each end: backing takes the tick under, laying the tick over."""
    o = np.where(side > 0, [tick_under(p) for p in open_price], [tick_over(p) for p in open_price])
    c = np.where(side > 0, [tick_over(p) for p in close_price], [tick_under(p) for p in close_price])
    return np.asarray(o, float), np.asarray(c, float)


def hedged_trades(scored: pd.DataFrame, pred_col: str, threshold: float, spread: bool = False) -> pd.DataFrame:
    """One row a trade: GBP1 at the open on every selection whose forecast clears ``threshold`` in size, closed at
    the close and greened. ``side`` +1 is back-then-lay (forecast shorter), -1 lay-then-back (forecast longer).
    ``profit`` is before commission; ``stake`` is 1 (the money at risk is 1 backing, o - 1 laying: the ROI is on
    the GBP1 unit, the owner's stake per selection)."""
    yhat = scored[pred_col].to_numpy(float)
    side = np.where(yhat < -threshold, 1, np.where(yhat > threshold, -1, 0))
    if threshold <= 0:
        side = np.where(yhat < 0, 1, np.where(yhat > 0, -1, 0))
    keep = side != 0
    t = scored.loc[keep, ["market_id", "day", "month", "sport", "country", "market_type", "open_price",
                          "close_price", "y"]].copy()
    t["side"] = side[keep]
    t["yhat"] = yhat[keep]
    o, c = t["open_price"].to_numpy(float), t["close_price"].to_numpy(float)
    if spread:
        o, c = _with_spread(o, c, t["side"].to_numpy())
    t["fill_open"], t["fill_close"] = o, c
    t["profit"] = np.where(t["side"] > 0, o / c - 1.0, 1.0 - o / c)
    t["stake"] = 1.0
    t["band"] = price_band(t["open_price"].to_numpy()).astype(str).to_numpy()
    t["right_way"] = np.sign(t["y"]) == t["side"] * -1           # back (+1) wants y < 0
    t["row"] = np.flatnonzero(keep)                              # the scored row, for other breakdown keys
    return t.reset_index(drop=True)


def net_by_market(trades: pd.DataFrame, commission: float) -> pd.DataFrame:
    """Each market's net profit after commission on its positive net (the exchange's charge)."""
    g = trades.groupby(["day", "market_id"], sort=False).agg(profit=("profit", "sum"), stake=("stake", "sum"),
                                                              bets=("stake", "size")).reset_index()
    g["net"] = g["profit"] - commission * g["profit"].clip(lower=0)
    return g


def drawdown(daily_net: np.ndarray) -> float:
    """The worst peak-to-trough fall of the cumulative daily P&L (GBP on GBP1 stakes)."""
    if len(daily_net) == 0:
        return 0.0
    cum = np.cumsum(daily_net)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    return float(np.max(peak - cum))


def summarise(trades: pd.DataFrame, commission: float = COMMISSION, n_boot: int = 2000, seed: int = 0) -> dict:
    """ROI after commission, its 90% bootstrap interval by day, bets a day, worst drawdown, right-way share."""
    if trades.empty:
        return {"bets": 0}
    m = net_by_market(trades, commission)
    day = m.groupby("day").agg(net=("net", "sum"), stake=("stake", "sum"), bets=("bets", "sum"))
    lo, hi = (ratio_interval(day["net"].to_numpy(), day["stake"].to_numpy(), n=n_boot, seed=seed) if n_boot > 0
              else (np.nan, np.nan))
    staked = float(day["stake"].sum())
    return {"bets": int(len(trades)), "days": int(len(day)), "bets_per_day": float(len(trades) / len(day)),
            "staked": staked, "profit": float(day["net"].sum()), "roi": float(day["net"].sum() / staked),
            "roi_lo90": lo, "roi_hi90": hi, "worst_drawdown": drawdown(day["net"].to_numpy()),
            "worst_drawdown_per_staked": drawdown(day["net"].to_numpy()) / staked if staked else np.nan,
            "right_way": float(trades["right_way"].mean()), "backs_share": float((trades["side"] > 0).mean()),
            "losing_days": float((day["net"] < 0).mean())}


def sweep(scored: pd.DataFrame, pred_col: str, thresholds=THRESHOLDS, commission: float = COMMISSION,
          commission_alt: float = COMMISSION_ALT, n_boot: int = 2000) -> pd.DataFrame:
    """One row a threshold: the hedged trade at the mid, at the alternative commission, and a tick worse each end."""
    rows = []
    for t in thresholds:
        tr = hedged_trades(scored, pred_col, t)
        base = summarise(tr, commission, n_boot=n_boot)
        row = {"threshold": t, **base}
        if base.get("bets", 0):
            row["roi_alt_commission"] = _roi(tr, commission_alt)
            sp = hedged_trades(scored, pred_col, t, spread=True)
            s = summarise(sp, commission, n_boot=n_boot)
            row.update({"roi_spread": s["roi"], "roi_spread_lo90": s["roi_lo90"], "roi_spread_hi90": s["roi_hi90"],
                        "worst_drawdown_spread": s["worst_drawdown"], "roi_spread_alt_commission": _roi(sp, commission_alt)})
        rows.append(row)
    return pd.DataFrame(rows)


def _roi(trades: pd.DataFrame, commission: float) -> float:
    m = net_by_market(trades, commission)
    return float(m["net"].sum() / m["stake"].sum()) if len(m) else np.nan


def by_group(scored: pd.DataFrame, pred_col: str, threshold: float, key: str, commission: float = COMMISSION,
             spread: bool = False, n_boot: int = 500) -> pd.DataFrame:
    """The hedged trade at one threshold broken down by ``key`` (band, market_type, country, sport, month)."""
    tr = hedged_trades(scored, pred_col, threshold, spread=spread)
    if tr.empty:
        return pd.DataFrame()
    if key not in tr.columns:
        tr[key] = scored[key].to_numpy()[tr["row"].to_numpy()]
    rows = {}
    for k, part in tr.groupby(key, sort=True):
        rows[str(k)] = summarise(part, commission, n_boot=n_boot)
    out = pd.DataFrame(rows).T
    return out[[c for c in ("bets", "days", "bets_per_day", "roi", "roi_lo90", "roi_hi90", "worst_drawdown",
                            "right_way", "backs_share") if c in out.columns]]
