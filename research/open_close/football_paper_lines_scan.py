"""Every football strategy the paper book can score at Betfair's real books: crossing and resting, back and lay,
each checkpoint, market type and price band; and the cross-market lag rule fixed on 6 Nov.

racing² football research, 8 Nov 2026 (the owner: "Evaluate all possible strategies. Any profitable so far? Any
edge"). The football feed's paper book (``football/paper_book.py``) gives every selection of every recorded market
an executable line at each checkpoint before the off (1440, 360, 120, 60, 15, 2 minutes): crossing (the offered
prices walked for a stake) and, at 1440-60 minutes, a resting GBP10 order one tick inside the spread, filled from
the recorded traded volume and queue. Each line carries its value greened at the close (crossing the spread there:
``green_close``), against the close mid (``clv_mid``), and held to Betfair's result (``pnl``).

This scan reads every ``paper_lines.csv`` the feed has written and reports, for each strategy cell:

- lines; for resting, the share filled;
- greened at the close: the mean return a stake unit after 2% commission on each market's net win, with its 95%
  bootstrap interval by market;
- held to the result: return per GBP staked (the GBP10 line), with its interval;
- the mean of each match day (is it the same each day?).

A cell is shown when it holds at least ``MIN_LINES`` lines. Nothing is chosen here: with a few match days a cell that
looks good is a lead for a pre-registered trial, never a result.

The cross-market rule (reports/football_model_rethink.md, 6 Nov, fixed before these books existed): at 60 and 15
minutes, the gap between match odds' mid chances and double chance's / draw no bet's mid chances; over 1 or 2
points, take the follower's best offered price towards match odds, closed at the close crossing.

    python -m research.open_close.football_paper_lines_scan --days 14 --out out/paper_lines_scan.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

COMMISSION = 0.02
MIN_LINES = 100
BANDS = [1.01, 1.5, 2.0, 3.0, 5.0, 10.0, 1000.0]


def load(store, days: int, today: date | None = None) -> pd.DataFrame:
    from football.exchange_feed import read_day
    today = today or date.today()
    frames = []
    for i in range(days):
        d = (today - timedelta(days=i)).isoformat()
        f = read_day(store, d, "paper_lines")
        if f is not None and len(f):
            frames.append(f)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for c in ["checkpoint_min", "best_back", "best_lay", "green_close", "clv_mid", "pnl", "stake", "filled",
              "won", "px_10", "post_price", "close_mid", "close_back", "close_lay", "spread_ticks", "depth_back_gbp",
              "depth_lay_gbp"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _boot(values: np.ndarray, groups: np.ndarray, weights: np.ndarray | None = None, n: int = 1000,
          seed: int = 0) -> tuple[float, float, float]:
    """Mean (weighted), and its 95% interval resampling groups (markets)."""
    w = np.ones(len(values)) if weights is None else weights
    s = pd.DataFrame({"g": groups, "v": values * w, "w": w}).groupby("g")[["v", "w"]].sum()
    v, ww = s["v"].to_numpy(), s["w"].to_numpy()
    if ww.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (n, len(v)))
    b = v[idx].sum(1) / np.maximum(ww[idx].sum(1), 1e-12)
    return float(v.sum() / ww.sum()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def _net(x: pd.Series, market: pd.Series) -> pd.Series:
    """Commission on each market's net win, spread back over its lines."""
    tot = x.groupby(market).transform("sum")
    n = x.groupby(market).transform("size")
    return np.where(tot > 0, x * (1 - COMMISSION), x) if len(x) else x


def cell(g: pd.DataFrame) -> dict:
    out = {"lines": int(len(g)), "markets": int(g["market_id"].nunique()), "days": int(g["day"].nunique())}
    if (g["kind"] == "rest").all():
        out["filled_share"] = round(float((g["filled"] > 0).mean()), 4)
        g = g[g["filled"] > 0]
        if g.empty:
            return out
    gc = g.dropna(subset=["green_close"])
    if len(gc):
        net = _net(gc["green_close"], gc["market_id"])
        m, lo, hi = _boot(np.asarray(net, float), gc["market_id"].to_numpy())
        out["greened_at_close"] = {"n": int(len(gc)), "mean": round(m, 4), "lo95": round(lo, 4), "hi95": round(hi, 4),
                                   "by_day": {d: round(float(np.mean(np.asarray(net)[gc["day"].to_numpy() == d])), 4)
                                              for d in sorted(gc["day"].unique())}}
    hp = g.dropna(subset=["pnl"])
    hp = hp[hp["stake"] > 0]
    if len(hp):
        m, lo, hi = _boot((hp["pnl"] / hp["stake"]).to_numpy(float), hp["market_id"].to_numpy(),
                          hp["stake"].to_numpy(float))
        out["held_to_result"] = {"n": int(len(hp)), "per_gbp": round(m, 4), "lo95": round(lo, 4), "hi95": round(hi, 4)}
    return out


def grid(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    df["band"] = pd.cut(df["best_back"], BANDS, right=False,
                        labels=[f"{a}-{b}" for a, b in zip(BANDS, BANDS[1:])]).astype(str)
    rows = []
    for keys, g in df.groupby(["kind", "side", "checkpoint_min", "market_type"]):
        if len(g) >= MIN_LINES:
            rows.append(dict(zip(["kind", "side", "checkpoint", "market_type"], keys), band="all") | cell(g))
        for b, gb in g.groupby("band"):
            if len(gb) >= MIN_LINES:
                rows.append(dict(zip(["kind", "side", "checkpoint", "market_type"], keys), band=b) | cell(gb))
    return rows


def _label(market_type: str, runner: str, event: str) -> str | None:
    home, _, away = str(event).partition(" v ")
    r = str(runner)
    if market_type == "MATCH_ODDS":
        return "D" if r == "The Draw" else ("H" if r == home.strip() else ("A" if r == away.strip() else None))
    if market_type == "DRAW_NO_BET":
        return "H" if r == home.strip() else ("A" if r == away.strip() else None)
    if market_type == "DOUBLE_CHANCE":
        return {"Home or Draw": "HD", "Draw or Away": "DA", "Home or Away": "HA"}.get(r)
    return None


def cross_market(df: pd.DataFrame) -> dict:
    """The 6 Nov lag rule at the recorded books (crossing lines at 60 and 15 minutes)."""
    c = df[(df["kind"] == "cross") & df["checkpoint_min"].isin([60.0, 15.0])
           & df["market_type"].isin(["MATCH_ODDS", "DOUBLE_CHANCE", "DRAW_NO_BET"])].copy()
    if c.empty:
        return {"status": "no data found"}
    c["lab"] = [_label(m, r, e) for m, r, e in zip(c["market_type"], c["runner"], c["event"])]
    c = c.dropna(subset=["lab", "best_back", "best_lay"])
    c["mid"] = (c["best_back"] + c["best_lay"]) / 2
    one = c[c["side"] == "back"].drop_duplicates(["market_id", "selection_id", "checkpoint_min"])
    one = one.assign(inv=1 / one["mid"])
    one["q"] = one["inv"] / one.groupby(["market_id", "checkpoint_min"])["inv"].transform("sum")
    one.loc[one["market_type"] == "DOUBLE_CHANCE", "q"] *= 2
    ev_key = ["event", "kickoff_utc", "checkpoint_min"]
    mo = one[one["market_type"] == "MATCH_ODDS"].pivot_table(index=ev_key, columns="lab", values="q", aggfunc="first")
    if not {"H", "D", "A"} <= set(mo.columns):
        return {"status": "no match odds books found"}
    der = pd.DataFrame({"HD": mo["H"] + mo["D"], "DA": mo["D"] + mo["A"], "HA": mo["H"] + mo["A"],
                        "DNB_H": mo["H"] / (mo["H"] + mo["A"]), "DNB_A": mo["A"] / (mo["H"] + mo["A"])})
    fol = one[one["market_type"] != "MATCH_ODDS"].copy()
    fol["dkey"] = np.where(fol["market_type"] == "DRAW_NO_BET", "DNB_" + fol["lab"], fol["lab"])
    fol = fol.join(der.stack().rename("derived"), on=ev_key + ["dkey"])
    fol["gap"] = fol["derived"] - fol["q"]
    out = {}
    for t in (0.01, 0.02):
        pick = fol[fol["gap"].abs() > t][["market_id", "selection_id", "checkpoint_min", "gap"]]
        pick["side"] = np.where(pick["gap"] > 0, "back", "lay")
        lines = c.merge(pick, on=["market_id", "selection_id", "checkpoint_min", "side"])
        for cp in (60.0, 15.0):
            g = lines[lines["checkpoint_min"] == cp]
            out[f"gap_over_{int(t * 100)}pt_at_{int(cp)}min"] = cell(g) if len(g) else {"lines": 0}
    out["events_with_both_books"] = int(fol.dropna(subset=["derived"])[["event", "kickoff_utc"]].drop_duplicates()
                                        .shape[0])
    return out


MAX_SPREAD_TICKS = 3
MAX_CLOSE_RATIO = 1.05
MAX_PRICE = 20.0


def tradable(df: pd.DataFrame) -> pd.Series:
    """A line a trader could act on and close: entry spread at most MAX_SPREAD_TICKS, a two-sided close whose lay is
    within MAX_CLOSE_RATIO of its back, and a price at most MAX_PRICE. Without it, empty books at the close (a 1.01
    back against a 1000 lay) turn a GBP1 lay into thousands of per cent."""
    ok = (df["spread_ticks"] <= MAX_SPREAD_TICKS) & (df["best_back"] <= MAX_PRICE)
    ok &= df["close_back"].notna() & df["close_lay"].notna() & (df["close_lay"] / df["close_back"] <= MAX_CLOSE_RATIO)
    return ok


def liquidity(df: pd.DataFrame) -> dict:
    """How much of the record a trader can use, by market type."""
    one = df[(df["kind"] == "cross") & (df["side"] == "back")]
    out = {}
    for mt, g in one.groupby("market_type"):
        out[mt] = {"lines": int(len(g)), "tradable_share": round(float(tradable(g).mean()), 4),
                   "median_spread_ticks": float(g["spread_ticks"].median()),
                   "median_depth_back_gbp": float(g["depth_back_gbp"].median()) if "depth_back_gbp" in g else None}
    return out


def run(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"status": "no paper lines found"}
    liq = liquidity(df)
    df = df[tradable(df)].copy()
    if df.empty:
        return {"status": "no tradable paper lines", "liquidity": liq}
    g = grid(df)
    pos = [r for r in g if r.get("greened_at_close", {}).get("lo95", -1) > 0
           or r.get("held_to_result", {}).get("lo95", -1) > 0]
    return {"filter": {"max_spread_ticks": MAX_SPREAD_TICKS, "max_close_lay_over_back": MAX_CLOSE_RATIO,
                       "max_price": MAX_PRICE},
            "liquidity_before_filter": liq,
            "lines": int(len(df)), "markets": int(df["market_id"].nunique()), "days": sorted(df["day"].unique()),
            "by_market_type": df.drop_duplicates("market_id")["market_type"].value_counts().to_dict(),
            "overall": {f"{k}_{s}": cell(x) for (k, s), x in df.groupby(["kind", "side"])},
            "cells_scored": len(g),
            "cells_with_interval_above_zero": pos,
            "cross_market_rule": cross_market(df),
            "grid": g}


def main(argv=None) -> int:
    from sources.common import Store
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = run(load(Store(root=a.root), a.days))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "grid"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
