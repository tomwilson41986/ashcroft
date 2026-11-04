"""Order-book features from the football feed's record, for execution and high-volume strategies on Betfair.

racing² football research, 4 Nov 2026 (the owner: "work on the improvements"; "high volume quant trading"). The live
feed (football/exchange_feed.py) writes one row a selection a read: the depth a side, the traded money since the
previous read, the matched totals. This module turns those reads into what a trading model or a simulator needs at
any decision moment, using only reads at or before it:

- **the book now:** best back and lay, spread in ticks, mid, the book's overround, money within the best levels on
  each side, the imbalance between them, the depth-weighted mid;
- **the flow:** money traded in the last 15 and 60 minutes (from the per-read traded deltas), the share of it at or
  above the mid, the matched total, its growth;
- **the path:** the mid's log change over the last 15, 60 and 240 minutes, its realised volatility over 60 minutes,
  minutes since the market was first read, minutes to the off;
- **the label** (for fitting, never a feature): the log change from the mid now to the last pre-play mid, and whether
  a back at the best back now, closed by a lay at the last pre-play best lay, made money.

Nothing here reads the future when building features: each moment's features come from reads at or before it (tested).

    python -m football.feed_features --day 2026-11-07 --every 15 --out out/feed_features_2026-11-07.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

WINDOWS_MIN = (15, 60, 240)
DEPTH_LEVELS = 3


def _ticks_between(back: float, lay: float, cap: int = 100) -> float:
    from trading.exchange import tick_up
    if not (np.isfinite(back) and np.isfinite(lay)) or lay <= back:
        return np.nan
    n, p = 0, float(back)
    while p < lay - 1e-9 and n < cap:
        p, n = tick_up(p + 1e-6), n + 1
    return float(n)


def prepare_books(books: pd.DataFrame) -> pd.DataFrame:
    """The feed's book rows, typed and sorted, pre-play and open only, with the mid and depth sums."""
    b = books.copy()
    b["polled"] = pd.to_datetime(b["polled_utc"], utc=True, errors="coerce")
    b["market_id"] = b["market_id"].astype(str)
    b["selection_id"] = pd.to_numeric(b["selection_id"], errors="coerce").astype("Int64")
    num = [c for c in b.columns if c.startswith(("back", "lay")) or c in ("traded_total", "runner_total_matched",
                                                                           "market_total_matched", "last_traded")]
    for c in num:
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b = b[b["polled"].notna() & (b["status"].astype(str) == "OPEN") & (pd.to_numeric(b["inplay"], errors="coerce") == 0)]
    b = b.sort_values(["market_id", "selection_id", "polled"]).reset_index(drop=True)
    b["mid"] = np.where((b["back1"] > 1) & (b["lay1"] > 1), (b["back1"] + b["lay1"]) / 2, np.nan)
    bs = [f"back{i}_size" for i in range(1, DEPTH_LEVELS + 1) if f"back{i}_size" in b]
    ls = [f"lay{i}_size" for i in range(1, DEPTH_LEVELS + 1) if f"lay{i}_size" in b]
    b["depth_back"] = b[bs].fillna(0).sum(axis=1)
    b["depth_lay"] = b[ls].fillna(0).sum(axis=1)
    b["traded_new"] = b["traded_delta"].map(_delta_total) if "traded_delta" in b else 0.0
    b["traded_new_above_mid"] = [_delta_above(t, m) for t, m in zip(b.get("traded_delta", [""] * len(b)), b["mid"])]
    return b


def _delta_total(text) -> float:
    from football.exchange_feed import parse_ladder
    return float(sum(parse_ladder(text).values()))


def _delta_above(text, mid) -> float:
    from football.exchange_feed import parse_ladder
    if not np.isfinite(mid):
        return 0.0
    return float(sum(v for p, v in parse_ladder(text).items() if p >= mid))


def features_at(b: pd.DataFrame, moment: pd.Timestamp, start: pd.Timestamp | None = None) -> pd.DataFrame:
    """Every selection's features at ``moment`` from the reads at or before it (one row a market and selection)."""
    past = b[b["polled"] <= moment]
    if past.empty:
        return pd.DataFrame()
    rows = []
    for (mid_, sid), g in past.groupby(["market_id", "selection_id"], sort=False):
        last = g.iloc[-1]
        f = {"market_id": mid_, "selection_id": int(sid), "moment_utc": moment.isoformat(),
             "read_age_min": (moment - last["polled"]).total_seconds() / 60,
             "back1": last["back1"], "lay1": last["lay1"], "mid": last["mid"],
             "spread_ticks": _ticks_between(last["back1"], last["lay1"]),
             "depth_back": last["depth_back"], "depth_lay": last["depth_lay"],
             "imbalance": ((last["depth_back"] - last["depth_lay"]) / (last["depth_back"] + last["depth_lay"])
                           if last["depth_back"] + last["depth_lay"] > 0 else np.nan),
             "matched": last.get("runner_total_matched", np.nan),
             "minutes_since_first_read": (moment - g["polled"].iloc[0]).total_seconds() / 60}
        size_b, size_l = last.get("back1_size", np.nan), last.get("lay1_size", np.nan)
        f["microprice"] = ((last["back1"] * size_l + last["lay1"] * size_b) / (size_b + size_l)
                           if np.isfinite(size_b) and np.isfinite(size_l) and size_b + size_l > 0 else last["mid"])
        for w in WINDOWS_MIN:
            win = g[g["polled"] >= moment - pd.Timedelta(minutes=w)]
            before = g[g["polled"] < moment - pd.Timedelta(minutes=w)]
            ref = before["mid"].iloc[-1] if len(before) and np.isfinite(before["mid"].iloc[-1]) else np.nan
            f[f"mid_logret_{w}m"] = (np.log(ref / last["mid"]) if np.isfinite(ref) and np.isfinite(last["mid"])
                                     else np.nan)                    # positive: price shortened (chance up)
            if w in (15, 60):
                f[f"traded_{w}m"] = float(win["traded_new"].sum())
                f[f"traded_above_mid_share_{w}m"] = (float(win["traded_new_above_mid"].sum()) / f[f"traded_{w}m"]
                                                     if f[f"traded_{w}m"] > 0 else np.nan)
        hour = g[g["polled"] >= moment - pd.Timedelta(minutes=60)]["mid"].dropna()
        f["vol_60m"] = float(np.std(np.diff(np.log(hour.to_numpy())))) if len(hour) >= 3 else np.nan
        if start is not None:
            f["minutes_to_off"] = (start - moment).total_seconds() / 60
        rows.append(f)
    out = pd.DataFrame(rows)
    books_over = out.groupby("market_id")["back1"].transform(lambda s: (1 / s).sum())
    out["book_over"] = books_over
    return out


def labels(b: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """For fitting only: from each moment's mid and best back to the last pre-play read of the selection."""
    last = b.groupby(["market_id", "selection_id"]).tail(1).set_index(["market_id", "selection_id"])
    f = feats.copy()
    key = list(zip(f["market_id"], f["selection_id"]))
    close_mid = np.array([last["mid"].get(k, np.nan) for k in key], float)
    close_lay = np.array([last["lay1"].get(k, np.nan) for k in key], float)
    f["y_mid_logret_to_close"] = np.log(f["mid"].to_numpy(float) / close_mid)       # positive: shortened by close
    f["y_back_now_lay_close"] = f["back1"].to_numpy(float) / close_lay - 1
    return f


def build_day(store, day: str, every_min: float = 15.0, root=None) -> pd.DataFrame:
    """The features of every market of a match day at every ``every_min`` minutes, with their labels."""
    from football.exchange_feed import read_day
    books = read_day(store, day, "books", root)
    markets = read_day(store, day, "markets", root)
    if books is None or markets is None:
        return pd.DataFrame()
    b = prepare_books(books)
    starts = markets.drop_duplicates("market_id").set_index("market_id")["market_start_utc"]
    out = []
    for mid_, g in b.groupby("market_id"):
        st = pd.to_datetime(starts.get(mid_), utc=True, errors="coerce")
        t0, t1 = g["polled"].min().ceil(f"{int(every_min)}min"), g["polled"].max()
        for t in pd.date_range(t0, t1, freq=f"{int(every_min)}min"):
            fx = features_at(g, t, st if pd.notna(st) else None)
            if len(fx):
                out.append(labels(g, fx))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main(argv=None) -> int:
    from sources.common import Store
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--day", required=True)
    ap.add_argument("--every", type=float, default=15.0)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    f = build_day(Store(root=a.root), a.day, a.every)
    if f.empty:
        print("no data found")
        return 0
    f.to_parquet(a.out, index=False)
    print(f"{len(f)} rows, {f['market_id'].nunique()} markets -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
