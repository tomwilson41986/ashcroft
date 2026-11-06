"""Price discovery in Betfair football markets, and a scan for simple trading rules that pay after costs.

racing² football research, 6 Nov 2026 (the owner: "keep an open mind, we are looking at price discovery and discovery
of profitable trading strategies"). Model-free: Betfair's own prices only. Every market type in the owner's Historic
Data bundle (``betfair_historic/markets_soccer_<year>.parquet``), every runner with a result, priced at the marks the
tables hold: the first trade, 24 h, 60, 15 and 1 minute before the scheduled off, and the close (the last trade before
in play).

1. **Where the price is found.** For each window between marks: the mean absolute move in log price, and the result's
   Brier score at each mark (lower = more of the truth in the price). Per market type.
2. **Momentum or reversal.** Does a window's move carry on into the next window, to the close? Correlation, and the
   trade: enter at the start of the later window in the direction (or against) the earlier move, close at the close.
3. **Held to the result, by price band.** Backing (and laying) at each mark and holding to the result, per market
   type and price band: the favourite-longshot bias and its kin, at the close and earlier.

**Honesty rules (fixed before the run):**
- Rules are found on the first half of the period (by kick-off) and checked on the second, never chosen on it.
- A rule is a candidate when, on the first half, it has at least ``MIN_BETS`` bets and its 95% interval is above zero
  after costs. Every candidate is then reported on the second half, whatever it shows; the number of rules tried is
  reported beside them.
- Costs: every entry and exit a tick worse than the last traded price, and 2% commission on each winning bet (or on a
  market's net win for a trade closed at the close).
- The last traded price is a trade that happened, not a quote that could be hit: a rule that pays only at it is noted
  as such and not taken further without the recorded books.
- Markets with several handicap lines a runner (Asian handicap, alternative totals) are left out: the tables give
  every line of a runner one line's prices (``drop_line_markets``).
- The last traded price bounces between the back and the lay side, so a window's move and the next one's are
  negatively correlated by construction: momentum and fade rules here are leads for the recorded books, never
  evidence on their own.

    python -m research.open_close.football_price_discovery --out out/price_discovery.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MARKS = ["ltp_first", "ltp_t_1440", "ltp_t_60", "ltp_t_15", "ltp_t_1", "ltp_close"]
BANDS = [1.01, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 1000.0]
MIN_BETS = 300
COMMISSION = 0.02
TOP_TYPES = 12
MOMENTUM = [("ltp_t_1440", "ltp_t_60"), ("ltp_t_60", "ltp_t_15"), ("ltp_t_15", "ltp_t_1"), ("ltp_first", "ltp_t_1440")]
THRESH = (0.0, 0.03, 0.08)           # the earlier window's move (log price) a trade needs


def load(store, local: str | None = None) -> pd.DataFrame:
    if local:
        df = pd.read_parquet(local)
    else:
        keys = sorted(k for k in store.listing("betfair_historic/") if "markets_soccer" in k and k.endswith(".parquet"))
        frames = [store.get_parquet(k) for k in keys]
        frames = [f for f in frames if f is not None and len(f)]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
    marks = [m for m in MARKS if m in df.columns]
    keep = ["market_id", "market_type", "country", "market_time", "won", "handicap"] + marks
    df = df[[c for c in keep if c in df.columns]].copy()
    df = drop_line_markets(df)
    for m in marks:
        df[m] = pd.to_numeric(df[m], errors="coerce")
        df.loc[(df[m] < 1.01) | (df[m] > 1000), m] = np.nan
    df["won"] = pd.to_numeric(df["won"], errors="coerce")
    df = df[df["won"].isin([0, 1]) & df["ltp_close"].notna()]
    df["kickoff"] = pd.to_datetime(df["market_time"], utc=True, errors="coerce")
    return df.dropna(subset=["kickoff"])


LINE_TYPES = ("ASIAN_HANDICAP", "ALT_TOTAL_GOALS")


def drop_line_markets(df: pd.DataFrame) -> pd.DataFrame:
    """Leave out markets with several handicap lines a runner (Asian handicap, alternative totals): the tables key a
    runner's prices by its selection id alone, so every line of it carries one line's prices against its own result
    (found 6 Nov 2026: returns of +29,000% in the first scan). Any market with a non-zero handicap goes too."""
    bad = df["market_type"].isin(LINE_TYPES)
    if "handicap" in df.columns:
        hc = pd.to_numeric(df["handicap"], errors="coerce").fillna(0)
        bad |= df["market_id"].isin(df.loc[hc != 0, "market_id"])
    return df[~bad].drop(columns=[c for c in ("handicap",) if c in df.columns])


def _ticks(p: np.ndarray, up: bool) -> np.ndarray:
    from research.open_close.backtest import tick_over, tick_under
    f = tick_over if up else tick_under
    return np.array([f(x) for x in p])


def _summ(profit: np.ndarray, market: np.ndarray | None = None, n_boot: int = 1000, seed: int = 0) -> dict:
    """Bets, return a GBP1 stake (commission on positive profit, per market when ``market`` is given), 95% interval."""
    if len(profit) == 0:
        return {"bets": 0}
    if market is not None:
        s = pd.Series(profit).groupby(market).agg(["sum", "size"])
        pnl, n = s["sum"].to_numpy(float), s["size"].to_numpy(float)
    else:
        pnl, n = profit.astype(float), np.ones(len(profit))
    pnl = np.where(pnl > 0, pnl * (1 - COMMISSION), pnl)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pnl), (n_boot, len(pnl)))
    roi = pnl[idx].sum(1) / n[idx].sum(1)
    return {"bets": int(n.sum()), "roi": round(float(pnl.sum() / n.sum()), 4),
            "lo95": round(float(np.quantile(roi, 0.025)), 4), "hi95": round(float(np.quantile(roi, 0.975)), 4)}


def discovery(df: pd.DataFrame) -> dict:
    """Mean |move| per window and the Brier score of the raw implied chance at each mark."""
    marks = [m for m in MARKS if m in df.columns]
    out = {}
    for mt, g in [("ALL", df)] + [(t, df[df["market_type"] == t]) for t in df["market_type"].value_counts()
                                   .head(TOP_TYPES).index]:
        r = {"runners": int(len(g)), "markets": int(g["market_id"].nunique())}
        r["brier"] = {m: round(float(np.nanmean((1 / g[m] - g["won"]) ** 2)), 5) for m in marks if g[m].notna().any()}
        r["mean_abs_move"] = {f"{a}->{b}": round(float(np.nanmean(np.abs(np.log(g[b] / g[a])))), 4)
                              for a, b in zip(marks, marks[1:]) if g[a].notna().any() and g[b].notna().any()}
        out[mt] = r
    return out


def momentum_rules(df: pd.DataFrame) -> list[dict]:
    """Rules: after the window a->b moved (log price) by more than t, at b back (follow a shortening / fade a drift)
    or lay, closed at the close a tick worse each end. One rule per (market type, window, threshold, follow|fade)."""
    rules = []
    types = ["ALL"] + list(df["market_type"].value_counts().head(TOP_TYPES).index)
    for mt in types:
        g = df if mt == "ALL" else df[df["market_type"] == mt]
        for a, b in MOMENTUM:
            if a not in g or b not in g:
                continue
            h = g.dropna(subset=[a, b, "ltp_close"])
            if len(h) < MIN_BETS:
                continue
            m1 = np.log(h[b] / h[a]).to_numpy()          # negative = shortened (backed)
            pb, pc = h[b].to_numpy(float), h["ltp_close"].to_numpy(float)
            for t in THRESH:
                for style in ("follow", "fade"):
                    # follow: back what shortened, lay what drifted; fade: the reverse
                    side = np.where(m1 < -t, 1, np.where(m1 > t, -1, 0)) * (1 if style == "follow" else -1)
                    rules.append({"kind": "momentum", "market_type": mt, "window": f"{a}->{b}", "threshold": t,
                                  "style": style, "_h": h, "_side": side, "_pb": pb, "_pc": pc})
    return rules


def _momentum_profit(r: dict, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    side, pb, pc = r["_side"][mask], r["_pb"][mask], r["_pc"][mask]
    keep = side != 0
    side, pb, pc = side[keep], pb[keep], pc[keep]
    back = _ticks(pb, False) / _ticks(pc, True) - 1                 # back at b, lay out at the close
    lay = 1 - _ticks(pb, True) / _ticks(pc, False)                  # lay at b, back out at the close (per GBP1 back)
    prof = np.where(side > 0, back, lay)
    return prof, r["_h"]["market_id"].to_numpy()[mask][keep]


def held_rules(df: pd.DataFrame) -> list[dict]:
    """Rules: back (or lay) every runner priced in a band at a mark, held to the result, a tick worse at entry."""
    rules = []
    types = ["ALL"] + list(df["market_type"].value_counts().head(TOP_TYPES).index)
    for mt in types:
        g = df if mt == "ALL" else df[df["market_type"] == mt]
        for mark in [m for m in ("ltp_t_1440", "ltp_t_60", "ltp_close") if m in g]:
            h = g.dropna(subset=[mark])
            band = pd.cut(h[mark], BANDS, right=False, labels=[f"{lo}-{hi}" for lo, hi in zip(BANDS, BANDS[1:])])
            for b in band.cat.categories:
                sub = h[band == b]
                if len(sub) < MIN_BETS:
                    continue
                for side in ("back", "lay"):
                    rules.append({"kind": "held", "market_type": mt, "mark": mark, "band": str(b), "side": side,
                                  "_h": sub})
    return rules


def _held_profit(r: dict, mask: np.ndarray) -> np.ndarray:
    h = r["_h"][mask]
    p, won = h[r["mark"]].to_numpy(float), h["won"].to_numpy(float)
    if r["side"] == "back":
        pb = _ticks(p, False)
        return np.where(won == 1, pb - 1, -1.0)
    pl = _ticks(p, True)                                           # lay a GBP1 back stake: liability pl - 1
    return np.where(won == 1, -(pl - 1), 1.0)


def scan(df: pd.DataFrame) -> dict:
    split = df["kickoff"].quantile(0.5)
    out = {"split_kickoff": str(split), "rules_tried": 0, "candidates": []}
    for r in momentum_rules(df) + held_rules(df):
        early = (r["_h"]["kickoff"] < split).to_numpy()
        out["rules_tried"] += 1
        if r["kind"] == "momentum":
            p1, m1 = _momentum_profit(r, early)
            s1 = _summ(p1, m1)
        else:
            s1 = _summ(_held_profit(r, early))
        if s1["bets"] < MIN_BETS or s1.get("lo95", -1) <= 0:
            continue
        if r["kind"] == "momentum":
            p2, m2 = _momentum_profit(r, ~early)
            s2 = _summ(p2, m2)
        else:
            s2 = _summ(_held_profit(r, ~early))
        out["candidates"].append({k: v for k, v in r.items() if not k.startswith("_")}
                                 | {"first_half": s1, "second_half": s2})
    out["candidates"].sort(key=lambda c: -c["second_half"].get("roi", -9))
    out["confirmed"] = [c for c in out["candidates"] if c["second_half"].get("lo95", -1) > 0]
    return out


def momentum_corr(df: pd.DataFrame) -> dict:
    """Corr of each window's log move with the move from its end to the close (negative = reversal)."""
    out = {}
    for a, b in MOMENTUM:
        if a in df and b in df:
            h = df.dropna(subset=[a, b, "ltp_close"])
            x, y = np.log(h[b] / h[a]), np.log(h["ltp_close"] / h[b])
            ok = (x != 0) | (y != 0)
            out[f"{a}->{b} then ->close"] = {"runners": int(ok.sum()),
                                             "corr": round(float(np.corrcoef(x[ok], y[ok])[0, 1]), 4) if ok.sum() > 2
                                             else None}
    return out


def run(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"status": "no data found (betfair_historic soccer tables)"}
    return {"runners": int(len(df)), "markets": int(df["market_id"].nunique()),
            "first_kickoff": str(df["kickoff"].min()), "last_kickoff": str(df["kickoff"].max()),
            "market_types": df.drop_duplicates("market_id")["market_type"].value_counts().head(25).to_dict(),
            "marks_held": {m: round(float(df[m].notna().mean()), 3) for m in MARKS if m in df},
            "where_the_price_is_found": discovery(df), "momentum_correlation": momentum_corr(df),
            "rule_scan": scan(df)}


def main(argv=None) -> int:
    from sources.common import Store
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None)
    ap.add_argument("--local", default=None, help="a local runner table instead of the S3 tables")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = run(load(Store(root=a.root), a.local))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res, indent=1, default=str)[:60000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
