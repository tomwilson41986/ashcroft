"""The high-frequency feature set and the next-move targets, from ``events.book_rows``.

Every feature at a row uses only that row and earlier rows of the same runner (rolling windows look back in time);
every target uses only later rows. In probability space (see the package docstring): bid = the lay side (price
``bl``, size ``bls``), ask = the back side (``bb``, ``bbs``).

Features:
- ``spread_ticks``  ticks between the best back and best lay
- ``qi``            queue imbalance at the touch, (bid size - ask size) / (bid size + ask size)
- ``qi3``           the same over the best three levels
- ``micro_dev``     the microprice minus the mid, as a share of the spread (-0.5 to 0.5)
- ``ofi_1s/5s``     Cont's order-flow imbalance at the touch over the last 1 and 5 seconds, over the touch's money
- ``flow_1s/5s/30s`` money taken by backers minus by layers, over the touch's money
- ``trades_5s``     how many rows saw money traded in the last 5 seconds
- ``ltp_gap``       log of the last traded probability over the mid probability
- ``spn_gap``       log of the projected SP's probability over the mid probability
- ``book_back``     the market's back overround (sum of 1 / best back)
- ``to_off``, ``field``

Targets (classes -1 down, 0 none, +1 up, in probability):
- ``y_next``        the direction of the runner's next change of mid (rows with no later change dropped)
- ``y_1s/5s/10s``   the direction of the mid h seconds on (the last row at or before t + h); rows whose t + h is past
                    the runner's last row are dropped (the market went off before the horizon)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.micro.events import tick_index

MAX_SPREAD_TICKS = 10          # a wider book has no meaningful mid
HORIZONS = (1, 5, 10)
FEATURES = ("spread_ticks", "qi", "qi3", "micro_dev", "ofi_1s", "ofi_5s", "flow_1s", "flow_5s", "flow_30s",
            "trades_5s", "ltp_gap", "spn_gap", "book_back", "to_off", "field")


def _ofi(bid_p, bid_s, ask_p, ask_s) -> np.ndarray:
    """Cont, Kukanov and Stoikov's order-flow imbalance, one value per row against the row before (0 for the first)."""
    e = np.zeros(len(bid_p))
    if len(bid_p) < 2:
        return e
    bp0, bs0, ap0, as0 = bid_p[:-1], bid_s[:-1], ask_p[:-1], ask_s[:-1]
    bp1, bs1, ap1, as1 = bid_p[1:], bid_s[1:], ask_p[1:], ask_s[1:]
    e[1:] = ((bp1 >= bp0) * bs1 - (bp1 <= bp0) * bs0 - (ap1 <= ap0) * as1 + (ap1 >= ap0) * as0)
    return np.nan_to_num(e)


def _rolling_sum(t_s: np.ndarray, x: np.ndarray, window: float) -> np.ndarray:
    """For each row, the sum of x over rows in (t - window, t]."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    lo = np.searchsorted(t_s, t_s - window, side="right")
    return c[np.arange(1, len(x) + 1)] - c[lo]


def _direction(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    return np.where(np.abs(d) < 1e-12, 0, np.sign(d)).astype(float)


def build(rows: pd.DataFrame) -> pd.DataFrame:
    """Features and targets for every row; rows with no two-sided book within MAX_SPREAD_TICKS are dropped."""
    if rows.empty:
        return rows
    df = rows.copy()
    for c in ("bb", "bbs", "bl", "bls", "back_depth", "lay_depth", "buy", "sell", "flat", "tv", "ltp", "spn",
              "book_back", "to_off", "field"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    ok = df.bb.notna() & df.bl.notna() & (df.bb > 1) & (df.bl > 1)
    df = df[ok].copy()
    df["spread_ticks"] = [tick_index(l) - tick_index(b) for b, l in zip(df.bb, df.bl)]
    df = df[(df.spread_ticks >= 1) & (df.spread_ticks <= MAX_SPREAD_TICKS)].copy()
    bid_p, ask_p = 1.0 / df.bl.to_numpy(float), 1.0 / df.bb.to_numpy(float)
    bid_s, ask_s = df.bls.to_numpy(float), df.bbs.to_numpy(float)
    df["pm"] = (bid_p + ask_p) / 2
    tot = bid_s + ask_s
    df["qi"] = np.where(tot > 0, (bid_s - ask_s) / np.where(tot > 0, tot, 1), 0.0)
    d3 = df.lay_depth.to_numpy(float) + df.back_depth.to_numpy(float)
    df["qi3"] = np.where(d3 > 0, (df.lay_depth - df.back_depth) / np.where(d3 > 0, d3, 1), 0.0)
    micro = np.where(tot > 0, (bid_p * ask_s + ask_p * bid_s) / np.where(tot > 0, tot, 1), df.pm)
    df["micro_dev"] = (micro - df.pm) / (ask_p - bid_p)
    df["ltp_gap"] = np.log((1.0 / df.ltp) / df.pm).where(df.ltp > 1)
    df["spn_gap"] = np.log((1.0 / df.spn) / df.pm).where(df.spn > 1)
    df["traded"] = (df.buy + df.sell + df.flat > 0).astype(float)
    df["net_flow"] = df.buy - df.sell
    out = []
    for _, g in df.groupby(["market", "sid"], sort=False):
        g = g.copy()
        t_s = (g.t - g.t.iloc[0]).dt.total_seconds().to_numpy()
        touch = (g.bls + g.bbs).to_numpy(float)
        touch = np.where(touch > 0, touch, 1.0)
        ofi = _ofi(1.0 / g.bl.to_numpy(float), g.bls.to_numpy(float), 1.0 / g.bb.to_numpy(float),
                   g.bbs.to_numpy(float))
        for w in (1, 5):
            g[f"ofi_{w}s"] = _rolling_sum(t_s, ofi, w) / touch
        for w in (1, 5, 30):
            g[f"flow_{w}s"] = _rolling_sum(t_s, g.net_flow.to_numpy(float), w) / touch
        g["trades_5s"] = _rolling_sum(t_s, g.traded.to_numpy(float), 5)
        pm = g.pm.to_numpy(float)
        # the next change: the mid at the first later row where the mid moved (walked back from the end)
        nxt = np.full(len(pm), np.nan)
        later = np.nan
        for i in range(len(pm) - 1, -1, -1):
            nxt[i] = later
            if i > 0 and abs(pm[i] - pm[i - 1]) > 1e-12:
                later = pm[i]
        g["y_next"] = np.where(np.isnan(nxt), np.nan, _direction(pm, np.nan_to_num(nxt, nan=0.0)))
        for h in HORIZONS:
            j = np.searchsorted(t_s, t_s + h, side="right") - 1
            valid = (t_s + h) <= t_s[-1]
            y = _direction(pm, pm[j])
            g[f"y_{h}s"] = np.where(valid, y, np.nan)
        out.append(g)
    res = pd.concat(out, ignore_index=True)
    res["day"] = res.t.dt.strftime("%Y-%m-%d")
    return res
