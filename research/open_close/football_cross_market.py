"""Cross-market consistency in Betfair football: do related markets agree, and which one moves when they do not?

racing² football research, 6 Nov 2026 (the owner: "keep an open mind ... price discovery and discovery of profitable
trading strategies"). Model-free, Betfair's own last traded prices from the Historic Data soccer tables.

Pairs: each market's chances (implied by its last traded prices, normalised to one) against the same chances
derived from a related market of the same event, at each mark (60, 15, 1 minute before the off):

- **match odds -> double chance:** P(home or draw) = P(H) + P(D), and so on;
- **match odds -> draw no bet:** P(home | no draw) = P(H) / (P(H) + P(A));
- **correct score -> over/under 2.5:** P(over 2.5) = the scores with three or more goals, and every 'Any Other'.

1. **Agreement:** the mean absolute gap (derived less own, in chance), per pair and mark.
2. **Who moves (price discovery):** regress each market's own move from the mark to the close on the gap. The
   market whose move closes the gap is the follower; the other leads.
3. **Rules (fixed before the run):** at a mark, when the gap is over a threshold (1, 2, 4 points of chance), back the
   runner the follower prices too long (or lay it when it prices it too short), in the follower market, and close at
   the close (a tick worse each end, 2% commission on the market's net), or hold it to the result (a tick worse,
   2% commission on a winning bet). Found on the first half by kick-off, a candidate only with ``MIN_BETS`` bets and
   a 95% interval above zero, reported on the second half whatever it shows; the count of rules tried beside them.

    python -m research.open_close.football_cross_market --out out/cross_market.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MARKS = ["ltp_t_60", "ltp_t_15", "ltp_t_1"]
TYPES = ("MATCH_ODDS", "DOUBLE_CHANCE", "DRAW_NO_BET", "CORRECT_SCORE", "OVER_UNDER_25")
THRESH = (0.01, 0.02, 0.04)
MIN_BETS = 300
COMMISSION = 0.02


def load(store, local: str | None = None) -> pd.DataFrame:
    if local:
        df = pd.read_parquet(local)
    else:
        keys = sorted(k for k in store.listing("betfair_historic/") if "markets_soccer" in k and k.endswith(".parquet"))
        frames = [store.get_parquet(k) for k in keys]
        frames = [f[f["market_type"].isin(TYPES)] for f in frames if f is not None and len(f)]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
    df = df[df["market_type"].isin(TYPES)].copy()
    for c in ["won", "sort_priority", "ltp_close"] + MARKS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _label(r) -> str | None:
    t, n, sp = r.market_type, str(r.runner_name), r.sort_priority
    if t == "MATCH_ODDS":
        return "D" if n == "The Draw" else {1: "H", 2: "A"}.get(sp)
    if t == "DRAW_NO_BET":
        return {1: "H", 2: "A"}.get(sp)
    if t == "DOUBLE_CHANCE":
        return {"Home or Draw": "HD", "Draw or Away": "DA", "Home or Away": "HA"}.get(n)
    if t == "OVER_UNDER_25":
        return {"Under 2.5 Goals": "U", "Over 2.5 Goals": "O"}.get(n)
    if t == "CORRECT_SCORE":
        if n.startswith("Any Other"):
            return "O"
        try:
            a, b = (int(x) for x in n.split(" - "))
        except ValueError:
            return None
        return "O" if a + b >= 3 else "U"
    return None


def events(df: pd.DataFrame) -> pd.DataFrame:
    """One row an event: each market's normalised chances (by label) and its prices at each mark and the close,
    and the results."""
    df = df.copy()
    df["lab"] = [_label(r) for r in df[["market_type", "runner_name", "sort_priority"]].itertuples(index=False)]
    df = df[df["lab"].notna() & df["won"].isin([0, 1])]
    out = []
    cols = [c for c in MARKS + ["ltp_close"] if c in df.columns]
    for c in cols:
        df[f"inv_{c}"] = 1 / df[c]
        # double chance has two winners: its chances sum to two
        k = np.where(df["market_type"] == "DOUBLE_CHANCE", 2.0, 1.0)
        df[f"q_{c}"] = k * df[f"inv_{c}"] / df.groupby("market_id")[f"inv_{c}"].transform("sum")
    # correct score: sum the chances into over/under (its own 'market' of two labels)
    agg = {f"q_{c}": "sum" for c in cols} | {"won": "max"}
    cs = df[df["market_type"] == "CORRECT_SCORE"].groupby(["event_id", "market_id", "lab"]).agg(agg).reset_index()
    cs["market_type"] = "CS_OU"
    rest = df[df["market_type"] != "CORRECT_SCORE"][["event_id", "market_id", "market_type", "lab", "won"]
                                                    + [f"q_{c}" for c in cols] + cols]
    long = pd.concat([rest, cs], ignore_index=True)
    long["key"] = long["market_type"] + ":" + long["lab"]
    wide_q = long.pivot_table(index="event_id", columns="key", values=[f"q_{c}" for c in cols], aggfunc="first")
    wide_q.columns = [f"{a}|{b}" for a, b in wide_q.columns]
    wide_p = long[long["market_type"] != "CS_OU"].pivot_table(index="event_id", columns="key", values=cols,
                                                               aggfunc="first")
    wide_p.columns = [f"p_{a}|{b}" for a, b in wide_p.columns]
    won = long.pivot_table(index="event_id", columns="key", values="won", aggfunc="first")
    won.columns = [f"won|{b}" for b in won.columns]
    mid = long.pivot_table(index="event_id", columns="market_type", values="market_id", aggfunc="first")
    mid.columns = [f"mid|{b}" for b in mid.columns]
    kick = df.groupby("event_id")["market_time"].first().rename("market_time")
    w = wide_q.join(wide_p).join(won).join(mid).join(kick).reset_index()
    w["kickoff"] = pd.to_datetime(w["market_time"], utc=True, errors="coerce")
    return w


def derived(w: pd.DataFrame, c: str) -> dict[str, tuple[pd.Series, pd.Series]]:
    """(follower runner key) -> (derived chance from the reference market, the follower's own chance) at mark c."""
    q = lambda k: w.get(f"q_{c}|{k}", pd.Series(np.nan, index=w.index))           # noqa: E731
    h, d, a = q("MATCH_ODDS:H"), q("MATCH_ODDS:D"), q("MATCH_ODDS:A")
    return {"DOUBLE_CHANCE:HD": (h + d, q("DOUBLE_CHANCE:HD")), "DOUBLE_CHANCE:DA": (d + a, q("DOUBLE_CHANCE:DA")),
            "DOUBLE_CHANCE:HA": (h + a, q("DOUBLE_CHANCE:HA")),
            "DRAW_NO_BET:H": (h / (h + a), q("DRAW_NO_BET:H")), "DRAW_NO_BET:A": (a / (h + a), q("DRAW_NO_BET:A")),
            "OVER_UNDER_25:O": (q("CS_OU:O"), q("OVER_UNDER_25:O")),
            "OVER_UNDER_25:U": (q("CS_OU:U"), q("OVER_UNDER_25:U"))}


REF = {"DOUBLE_CHANCE": "MATCH_ODDS", "DRAW_NO_BET": "MATCH_ODDS", "OVER_UNDER_25": "CS_OU"}


def agreement(w: pd.DataFrame) -> dict:
    """Mean |gap| at each mark, and who moves: the slope of each side's move to the close on the gap."""
    out = {}
    for c in [m for m in MARKS if f"q_{m}|MATCH_ODDS:H" in w.columns or f"q_{m}|OVER_UNDER_25:O" in w.columns]:
        for k, (der, own) in derived(w, c).items():
            der_c, own_c = derived(w, "ltp_close")[k]
            ok = der.notna() & own.notna() & der_c.notna() & own_c.notna()
            if ok.sum() < 50:
                continue
            gap = (der - own)[ok]
            mv_own, mv_ref = (own_c - own)[ok], (der_c - der)[ok]
            var = float(np.var(gap))
            out[f"{k} @ {c}"] = {"events": int(ok.sum()), "mean_abs_gap": round(float(gap.abs().mean()), 4),
                                 "own_move_slope": round(float(np.cov(gap, mv_own)[0, 1] / var), 3) if var else None,
                                 "reference_move_slope": round(float(np.cov(gap, mv_ref)[0, 1] / var), 3) if var
                                 else None}
    return out


def _tick(p, up):
    from research.open_close.backtest import tick_over, tick_under
    f = tick_over if up else tick_under
    return np.array([f(x) for x in p])


def _summ(profit, market=None, n_boot=1000, seed=0) -> dict:
    if len(profit) == 0:
        return {"bets": 0}
    if market is not None:
        s = pd.Series(profit).groupby(np.asarray(market)).agg(["sum", "size"])
        pnl, n = s["sum"].to_numpy(float), s["size"].to_numpy(float)
    else:
        pnl, n = np.asarray(profit, float), np.ones(len(profit))
    pnl = np.where(pnl > 0, pnl * (1 - COMMISSION), pnl)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pnl), (n_boot, len(pnl)))
    roi = pnl[idx].sum(1) / n[idx].sum(1)
    return {"bets": int(n.sum()), "roi": round(float(pnl.sum() / n.sum()), 4),
            "lo95": round(float(np.quantile(roi, 0.025)), 4), "hi95": round(float(np.quantile(roi, 0.975)), 4)}


def trades(w: pd.DataFrame, k: str, c: str, t: float, exit_: str, mask: np.ndarray) -> dict:
    """The follower's runner k at mark c: back when the reference says it is likelier than its own price says
    (gap > t), lay when less likely (gap < -t); closed at the close or held to the result."""
    der, own = derived(w, c)[k]
    p = w.get(f"p_{c}|{k}")
    pc = w.get(f"p_ltp_close|{k}")
    won = w.get(f"won|{k}")
    mt = k.split(":")[0]
    if p is None or pc is None or won is None:
        return {"bets": 0}
    gap = (der - own).to_numpy(float)
    side = np.where(gap > t, 1, np.where(gap < -t, -1, 0))
    ok = mask & (side != 0) & np.isfinite(p.to_numpy(float)) & np.isfinite(pc.to_numpy(float))
    side, pe, pcl, wn = side[ok], p.to_numpy(float)[ok], pc.to_numpy(float)[ok], won.to_numpy(float)[ok]
    if exit_ == "close":
        back = _tick(pe, False) / _tick(pcl, True) - 1
        lay = 1 - _tick(pe, True) / _tick(pcl, False)
        return _summ(np.where(side > 0, back, lay), w[f"mid|{mt}"].to_numpy()[ok])
    pb, pl = _tick(pe, False), _tick(pe, True)
    back = np.where(wn == 1, pb - 1, -1.0)
    lay = np.where(wn == 1, -(pl - 1), 1.0)
    return _summ(np.where(side > 0, back, lay))


def scan(w: pd.DataFrame) -> dict:
    split = w["kickoff"].quantile(0.5)
    early = (w["kickoff"] < split).to_numpy()
    out = {"split_kickoff": str(split), "rules_tried": 0, "candidates": []}
    for c in MARKS:
        if not any(col.startswith(f"q_{c}|") for col in w.columns):
            continue
        for k in derived(w, c):
            for t in THRESH:
                for exit_ in ("close", "result"):
                    out["rules_tried"] += 1
                    s1 = trades(w, k, c, t, exit_, early)
                    if s1["bets"] < MIN_BETS or s1.get("lo95", -1) <= 0:
                        continue
                    out["candidates"].append({"runner": k, "mark": c, "gap_over": t, "exit": exit_,
                                              "first_half": s1, "second_half": trades(w, k, c, t, exit_, ~early)})
    out["confirmed"] = [x for x in out["candidates"] if x["second_half"].get("lo95", -1) > 0]
    return out


def run(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"status": "no data found (betfair_historic soccer tables)"}
    w = events(df)
    return {"events": int(len(w)), "first_kickoff": str(w["kickoff"].min()), "last_kickoff": str(w["kickoff"].max()),
            "events_with": {mt: int(w[f"mid|{mt}"].notna().sum()) for mt in ("MATCH_ODDS",) + tuple(REF) + ("CS_OU",)
                            if f"mid|{mt}" in w},
            "agreement_and_who_moves": agreement(w), "rule_scan": scan(w)}


def main(argv=None) -> int:
    from sources.common import Store
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None)
    ap.add_argument("--local", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = run(load(Store(root=a.root), a.local))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
