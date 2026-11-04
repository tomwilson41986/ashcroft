"""The stacked move rule's resting-order variant, scored at the recorded Betfair books
(reports/preregistration_football_stack_resting.md).

The signals are the stacked rule's (reports/preregistration_football_stack.md): the same frozen models
(models/football/stack_v1, hashes checked), the same threshold, leagues and book widths. Only the entry differs.

The crossing rule takes the best price at the early book and pays the spread. This variant posts a resting order
one tick better than the best price on its side, and waits:

1. **Post.** At the early book (the first recorded snapshot within 60 min after football-data's collection moment,
   the crossing rule's), a back is posted one tick above the best back if that is still below the best lay, else
   at the best lay (joining the backers queued there). A lay is posted one tick below the best lay if that is still
   above the best back, else at the best back.
2. **Fill: only when the recorded book proves it.** A resting back at P is filled at the first later snapshot,
   before the deadline, whose best back is at P or longer: a lay offer at or above P can rest only once no back at
   P or below is left, ours included. A resting lay at P is filled when the best lay is at P or shorter. A move
   through P between two snapshots that reverses before the next one is missed, so the rule undercounts fills; it
   never counts a fill that could not have happened. Size: the notional GBP10, small beside the books.
3. **Deadline.** Unfilled orders are cancelled 90 minutes before the market's start, before the team sheets
   (about 60 minutes before) make a resting order a free option for the best-informed.
4. **Exit.** As the crossing rule: crossed at the last two-sided book within 30 minutes of the off, after the fill.
   Greened; 2% commission on each market's net.

The fill is conditional on the price moving against the forecast for a moment (a back fills when the price drifts
to P). That adverse selection is the thing this trial measures: the result reports the right-way share of the filled
signals against the unfilled ones.

    python -m research.open_close.football_stack_resting score --cache <lab cache> --models models/football/stack_v1 \\
        --root <sources> --out-json out/football_stack_resting.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PREREG = "reports/preregistration_football_stack_resting.md"
CANCEL_BEFORE_MIN = 90.0
GATE = {"min_bets": 300, "min_days": 90, "roi_lo95_above": 0, "clv_mean_above": 0}


def _next_up(p: float) -> float:
    from trading.exchange import tick_up
    return tick_up(float(p) + 1e-6)


def _next_down(p: float) -> float:
    from trading.exchange import tick_down
    return tick_down(float(p) - 1e-6)


def post_price(side: int, back1: float, lay1: float) -> float | None:
    """The resting order's price: one tick better than the best on its side while that stays inside the spread,
    else the best price on the other side (joining that queue). None without a best price on its own side."""
    has_back, has_lay = back1 > 1.0, lay1 > 1.0
    if side > 0:
        if not has_back:
            return None
        p = _next_up(back1)
        return p if not has_lay or p < lay1 - 1e-9 else float(lay1)
    if not has_lay:
        return None
    p = _next_down(lay1)
    return p if not has_back or p > back1 + 1e-9 else float(back1)


def rest_fill(side: int, price: float, rows: pd.DataFrame, posted, deadline) -> pd.Timestamp | None:
    """The first snapshot after ``posted`` and at or before ``deadline`` (open, pre-play) that proves a resting order
    at ``price`` filled: for a back, a best back at ``price`` or longer; for a lay, a best lay at ``price`` or
    shorter. ``rows`` is one selection's recorded books (clean_books columns). None if unfilled."""
    w = rows[(rows["polled"] > pd.Timestamp(posted)) & (rows["polled"] <= pd.Timestamp(deadline))
             & (rows["status"] == "OPEN") & ~rows["inplay"]].sort_values("polled")
    hit = w[w["back1"] >= price - 1e-9] if side > 0 else w[(w["lay1"] > 1.0) & (w["lay1"] <= price + 1e-9)]
    return None if hit.empty else pd.Timestamp(hit["polled"].iloc[0])


def resting_trade(side: int, early: pd.Series, rows: pd.DataFrame, b_market: pd.DataFrame, posted, start,
                  sid: str) -> dict:
    """One signal as a resting order -> the trade, or ``skip`` naming why it did not trade."""
    from research.open_close import football_paper_trial as pt
    price = post_price(side, float(early["back1"]), float(early["lay1"]))
    if price is None:
        return {"skip": "no early price on the side"}
    st = pd.to_datetime(start, utc=True, errors="coerce")
    if pd.isna(st):
        return {"skip": "no market start", "post_price": price}
    deadline = st - pd.Timedelta(minutes=CANCEL_BEFORE_MIN)
    mid_o = (early["back1"] + early["lay1"]) / 2.0 if early["lay1"] > 1.0 else float(early["back1"])
    filled = rest_fill(side, price, rows, posted, deadline)
    close = pt.close_book(b_market, after=filled if filled is not None else posted, start=start)
    c_row = close.loc[sid] if sid in close.index else None
    base = {"post_price": price, "early_mid": float(mid_o), "deadline_utc": deadline.isoformat()}
    if c_row is not None:
        mid_c = (c_row["back1"] + c_row["lay1"]) / 2.0
        base["close_mid"] = float(mid_c)
        base["right_way"] = bool(mid_c < mid_o) if side > 0 else bool(mid_c > mid_o)
    if filled is None:
        return {"skip": "unfilled by the deadline", **base}
    if c_row is None:
        return {"skip": "no two-sided close", **base, "fill_utc": filled.isoformat()}
    c = c_row["lay1"] if side > 0 else c_row["back1"]
    c_size = c_row["lay1_size"] if side > 0 else c_row["back1_size"]
    profit = price / c - 1.0 if side > 0 else 1.0 - price / c
    clv = price / base["close_mid"] - 1.0 if side > 0 else 1.0 - price / base["close_mid"]
    return {"skip": None, **base, "fill_utc": filled.isoformat(), "fill_open": float(price), "fill_close": float(c),
            "profit": float(profit), "clv": float(clv),
            "minutes_to_fill": float((filled - pd.Timestamp(posted)).total_seconds() / 60.0),
            "close_size_short": bool(not (c_size >= 10.0 * price / c))}


def trade_resting(t, sig: pd.DataFrame, widths: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The signals as resting orders at the recorded books -> (trades, not traded), each not traded with its
    reason (the unfilled ones keep their close mid, for the adverse-selection read)."""
    from research.open_close import football_paper_trial as pt
    trades, skipped = [], []
    meta = t.gold.drop_duplicates("match_id").set_index("match_id")
    for day, part in sig.groupby("match_day"):
        dd = date.fromisoformat(day)
        markets = pt.recorded_day(t.read_day, dd, "markets")
        books = pt.recorded_day(t.read_day, dd, "books")
        fix = pd.DataFrame({"match_id": part["event_id"].unique()})
        for c, k in (("HomeTeam", "home_team"), ("AwayTeam", "away_team"), ("country", "country")):
            fix[c] = fix["match_id"].map(meta[k]).astype(str)
        fix["kickoff_utc"] = fix["match_id"].map(meta["kickoff_utc"])
        mp = pt.map_markets(markets, fix, t.aliases) if markets is not None else pt.map_markets(None, fix)
        bk = pt.clean_books(books) if books is not None and len(books) else None
        for (mid, mt), sel in part.groupby(["event_id", "market_type"]):
            todo = sel[sel["side"] != 0]
            if todo.empty:
                continue
            legs = mp[(mp["match_id"] == mid) & (mp["market_type"] == mt)]
            base = {"day": day, "match_id": mid, "market_type": mt, "fd_code": sel["fd_code"].iloc[0],
                    "tier": sel["tier"].iloc[0], "collection": sel["collection"].iloc[0],
                    "weekday": pd.Timestamp(day).day_name()}
            if legs.empty or bk is None or set(legs["leg"]) != set(pt.SELECTIONS[mt]):
                skipped += [{**base, "selection": s, "side": int(sd), "reason": "no recorded market"}
                            for s, sd in zip(todo["selection_id"], todo["side"])]
                continue
            bf_mid = legs["market_id"].iloc[0]
            leg_to_sel = dict(zip(legs["leg"], legs["selection_id"]))
            b = bk[bk["market_id"] == bf_mid]
            t_early, early = pt.early_book(b, [leg_to_sel[x] for x in pt.SELECTIONS[mt]],
                                           pd.Timestamp(sel["moment_utc"].iloc[0]), t.tolerance_min)
            if early is None:
                skipped += [{**base, "selection": s, "side": int(sd), "reason": "no recorded book at the early moment"}
                            for s, sd in zip(todo["selection_id"], todo["side"])]
                continue
            over = float((1.0 / early["back1"]).sum())
            wl = pt.width_label(over)
            start = legs["market_start_utc"].iloc[0]
            for r in todo.itertuples(index=False):
                sid = leg_to_sel[r.selection_id]
                row = {**base, "selection": r.selection_id, "side": int(r.side), "yhat": float(r.yhat),
                       "market_id": bf_mid, "early_utc": t_early.isoformat(), "book_over": over, "book_width": wl}
                if wl not in widths:
                    skipped.append({**row, "reason": "recorded book wide"})
                    continue
                res = resting_trade(int(r.side), early.loc[sid], b[b["selection_id"] == sid], b, t_early, start, sid)
                skip = res.pop("skip")
                (skipped.append({**row, **res, "reason": skip}) if skip is not None
                 else trades.append({**row, **res}))
    tr = pd.DataFrame(trades)
    if len(tr):
        tr["stake"] = 1.0
        tr["band"] = pt.price_band(tr["fill_open"].to_numpy()).astype(str).to_numpy()
        tr["month"] = tr["day"].str[:7]
    return tr, pd.DataFrame(skipped)


def interval95(trades: pd.DataFrame, commission: float, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """The 95% bootstrap interval, by day, of the return after commission on each market's net."""
    from research.open_close.backtest import net_by_market
    if trades.empty or n_boot <= 0:
        return np.nan, np.nan
    day = net_by_market(trades, commission).groupby("day").agg(net=("net", "sum"), stake=("stake", "sum"))
    if len(day) < 2:
        return np.nan, np.nan
    num, den = day["net"].to_numpy(float), day["stake"].to_numpy(float)
    idx = np.random.default_rng(seed).integers(0, len(num), (n_boot, len(num)))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def adverse_selection(tr: pd.DataFrame, sk: pd.DataFrame) -> dict:
    """The right-way share (close mid against early mid, in the signal's direction) of the filled signals and of the
    unfilled ones, and the fill rate among the signals that reached a takeable recorded book."""
    unf = sk[sk["reason"] == "unfilled by the deadline"] if len(sk) else sk
    nf, nu = len(tr), len(unf)
    rw = lambda x: float(x["right_way"].dropna().astype(bool).mean()) if len(x) and x["right_way"].notna().any() else None  # noqa: E731
    return {"filled": nf, "unfilled": nu, "fill_rate": nf / (nf + nu) if nf + nu else None,
            "right_way_filled": rw(tr) if nf and "right_way" in tr else None,
            "right_way_unfilled": rw(unf) if nu and "right_way" in unf else None,
            "median_minutes_to_fill": float(tr["minutes_to_fill"].median()) if nf else None}


def gate(res: dict, gt: dict = GATE) -> dict:
    s, w, h = res["headline"], res["window"], res["halves"]
    lo = s.get("roi_lo95")
    checks = {"bets": s.get("bets", 0) >= gt["min_bets"], "days": w["calendar_days"] >= gt["min_days"],
              "roi_lo95_above_0": lo is not None and bool(np.isfinite(lo)) and lo > gt["roi_lo95_above"],
              "clv_above_0": (s.get("clv_mean") or -1) > gt["clv_mean_above"],
              "both_halves_above_0": (h["first"].get("roi") or -1) > 0 and (h["second"].get("roi") or -1) > 0}
    if w["status"] in ("running", "not started"):
        return {"checks": checks, "verdict": "running", "passed": None}
    return {"checks": checks, "verdict": "pass" if all(checks.values()) else "fail", "passed": bool(all(checks.values()))}


def score(cache: str, models: str, root: str | None, live_root: str | None, start: str | None,
          as_of: str | None, prereg: str = PREREG, n_boot: int = 2000) -> dict:
    from football.data import load_gold
    from football.reference import load_aliases
    from research.open_close import football_paper_trial as pt
    from research.open_close import football_stack_trial as st
    from research.open_close.football_exchange import read_prereg
    from sources.common import Store
    rule, digest = read_prereg(prereg)
    st.verify_models(models, rule["model_sha256"])
    read_day = pt.recorder_reader(live_root)
    as_of_d = date.fromisoformat(as_of) if as_of else date.today() - timedelta(days=1)
    if start:
        start_d = date.fromisoformat(start)
    else:
        first = pt.first_recorded_day(read_day, max(date.fromisoformat(pt.RECORDER_FROM),
                                                    date.fromisoformat(rule["written_utc"][:10])), as_of_d)
        if first is None:
            return {"status": "no football book record found", "prereg_sha256": digest}
        start_d = pt.trial_start(first)
    if start_d > as_of_d:                  # the record has begun, the trial's first collection has not
        return {"status": "not started", "start": start_d.isoformat(), "as_of": as_of_d.isoformat(),
                "prereg_sha256": digest, "rule": rule}
    d, _cols = st._matrix(cache)
    d = d[(d["match_date"] >= start_d.isoformat()) & (d["match_date"] <= as_of_d.isoformat())
          & d["fd_code"].isin(rule["leagues_fd"])]
    if d.empty:
        return {"status": "no matches in the window yet", "start": start_d.isoformat(), "as_of": as_of_d.isoformat(),
                "prereg_sha256": digest, "rule": rule}
    sig = st.signals(d, st.predict(models, d), rule["threshold"])
    store = Store(root=root)
    t = SimpleNamespace(gold=load_gold(store), read_day=read_day, tolerance_min=pt.TOLERANCE_MIN,
                        aliases=load_aliases(extra=store.get_parquet("football/gold/team_aliases.parquet")))
    tr, sk = trade_resting(t, sig, rule["book_width"]) if (sig["side"] != 0).any() else (pd.DataFrame(), pd.DataFrame())
    w = pt.window(tr, start_d, as_of_d, GATE["min_bets"], GATE["min_days"])
    inw = tr[pd.to_datetime(tr["day"]).dt.date <= date.fromisoformat(w["end"])] if len(tr) else tr
    head = pt.scored(inw, rule["commission"], n_boot)
    if len(inw):
        head["roi_lo95"], head["roi_hi95"] = interval95(inw, rule["commission"], n_boot)
    res = {"prereg": prereg, "prereg_sha256": digest, "rule": rule, "start": start_d.isoformat(),
           "as_of": as_of_d.isoformat(), "forecasts": int(len(sig)), "signals": int((sig["side"] != 0).sum()),
           "window": w, "headline": head, "adverse_selection": adverse_selection(tr, sk),
           "skipped": sk["reason"].value_counts().to_dict() if len(sk) else {}}
    res["halves"] = (pt.halves(inw, start_d, date.fromisoformat(w["end"]), rule["commission"]) if len(inw)
                     else {"first": {"bets": 0}, "second": {"bets": 0}})
    for key in ("book_width", "band", "tier", "fd_code", "month"):
        res[f"by_{key}"] = pt.by_key(inw, key, rule["commission"])
    res["gate"] = gate(res)
    return res


def main(argv=None) -> int:
    from research.open_close.football_paper_trial import _json
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("score")
    c.add_argument("--cache", required=True)
    c.add_argument("--models", default="models/football/stack_v1")
    c.add_argument("--root", default=None)
    c.add_argument("--live-root", default=None)
    c.add_argument("--start", default=None)
    c.add_argument("--as-of", default=None)
    c.add_argument("--prereg", default=PREREG)
    c.add_argument("--out-json", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    res = score(a.cache, a.models, a.root, a.live_root, a.start, a.as_of, a.prereg)
    text = _json(res)
    if a.out_json:
        from pathlib import Path
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
