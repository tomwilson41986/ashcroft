"""Arm 3: the place markets (read-only research).

The model prices the win; each runner's chance of a place is the Harville chance of finishing in the places the
market pays (number_of_winners in the recorder's books; else 2 for fields of 5-7, 3 for 8-15, 4 for 16 and more),
from the model's normalised win chances over the runners the market lists. A closing model of the same form as the
live one (model/race_book.py: a conditional logit on the market's price now and the model's) is fitted walk-forward
to the place markets' 08:00 UK marks against their BSP (betfair_live_marks / betfair_live_markets, from 1 Oct), one
fit a replayed day on the days before it. The live rule then runs on the place books with that model and its place
prices, exactly as it runs on the win books.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from model import race_book as rb
from research.arms import core
from trading.exchange import Market
from trading.matching import attach_ids

FIT_FROM = "2026-10-01"
MIN_FIT_RUNNERS = 400


def harville_place(p, k: int) -> np.ndarray:
    """Each runner's chance of finishing in the first k under Harville (the next finisher drawn in proportion to
    the win chances of those left): exact, by recursion over the sets of runners already home."""
    p = np.asarray(p, float)
    p = p / p.sum()
    n = len(p)
    k = int(max(0, min(k, n)))
    if k >= n:
        return np.ones(n)
    memo: dict = {}

    def f(mask: int, left: int) -> np.ndarray:
        """P(each runner still in ``mask`` finishes within the next ``left`` places)."""
        if left == 0:
            return np.zeros(n)
        key = (mask, left)
        if key in memo:
            return memo[key]
        idx = [i for i in range(n) if mask >> i & 1]
        tot = p[idx].sum()
        out = np.zeros(n)
        for j in idx:
            pj = p[j] / tot if tot > 0 else 1.0 / len(idx)
            out[j] += pj
            if left > 1:
                out += pj * f(mask & ~(1 << j), left - 1)
        memo[key] = out
        return out

    return np.clip(f((1 << n) - 1, k), 0.0, 1.0)


def places_for(n_runners: int, known: int | None) -> int:
    if known and known > 0:
        return int(known)
    if n_runners <= 4:
        return 1
    if n_runners <= 7:
        return 2
    if n_runners <= 15:
        return 3
    return 4


def place_prices(preds: pd.DataFrame, markets: list[Market], places: dict) -> pd.DataFrame:
    """The model's place price for every runner of every place market: market_id, selection_id, predicted_bfsp
    (the place price), p_win, p_place, places."""
    p = attach_ids(preds, markets)
    p = p[p["market_id"].notna()].copy()
    if p.empty:
        return pd.DataFrame(columns=["market_id", "selection_id", "predicted_bfsp"])
    if "predicted_win_prob_norm" in p.columns and p["predicted_win_prob_norm"].notna().any():
        p["p_win"] = pd.to_numeric(p["predicted_win_prob_norm"], errors="coerce")
    else:
        p["p_win"] = 1.0 / pd.to_numeric(p["predicted_bfsp"], errors="coerce")
    p = p[p["p_win"] > 0]
    out = []
    for mid, g in p.groupby("market_id"):
        k = places_for(len(g), places.get(str(mid)))
        pp = harville_place(g["p_win"].to_numpy(), k)
        for (_, r), q in zip(g.iterrows(), pp):
            out.append({"market_id": str(mid), "selection_id": int(r["selection_id"]), "p_win": float(r["p_win"]),
                        "p_place": float(q), "places": k,
                        "predicted_bfsp": float(1.0 / q) if q > 0 else np.nan})
    f = pd.DataFrame(out)
    return f[f["predicted_bfsp"] > 1]


@lru_cache(maxsize=None)
def _db_markets(db: str, from_day: str, to_day: str) -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(
            f"""SELECT race_date, market_id, selection_id, venue, country, market_start_utc, market_name, runner_name
                FROM betfair_live_markets WHERE UPPER(COALESCE(market_type, '')) = 'PLACE'
                AND race_date >= '{from_day}' AND race_date < '{to_day}'""", conn)
    finally:
        conn.close()


def _db_marks(db: str, from_day: str, to_day: str) -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(
            f"""SELECT k.race_date, k.market_id, k.selection_id, k.mark, k.back1, k.lay1, k.bsp, k.runner_status
                FROM betfair_live_marks k JOIN (SELECT DISTINCT market_id FROM betfair_live_markets
                     WHERE UPPER(COALESCE(market_type, '')) = 'PLACE' AND race_date >= '{from_day}'
                     AND race_date < '{to_day}') m ON m.market_id = k.market_id
                WHERE k.mark IN ('clock_08:00', 'final') AND k.race_date >= '{from_day}' AND k.race_date < '{to_day}'""",
            conn)
    finally:
        conn.close()


def fit_place_closing(db: str, upto: date, out_dir: Path, places_known: dict | None = None) -> str | None:
    """A closing model for the place markets fitted on the days before ``upto``: the 08:00 UK mark's price (the
    geometric mid when the spread is tight, else the back) and the model's place price against the place BSP.
    Returns the path of the saved model, or None when the panel is too thin."""
    to_day = upto.isoformat()
    mk = _db_markets(db, FIT_FROM, to_day)
    if mk.empty:
        print(f"  place closing: no place markets in the database before {to_day}")
        return None
    marks = _db_marks(db, FIT_FROM, to_day)
    if marks.empty:
        print(f"  place closing: no marks for the place markets before {to_day}")
        return None
    for c in ("back1", "lay1", "bsp", "selection_id"):
        marks[c] = pd.to_numeric(marks[c], errors="coerce")
    m8 = marks[marks["mark"].eq("clock_08:00")].drop_duplicates(["market_id", "selection_id"])
    fin = marks[marks["mark"].eq("final")].drop_duplicates(["market_id", "selection_id"])[
        ["market_id", "selection_id", "bsp", "runner_status"]].rename(columns={"runner_status": "final_status"})
    m8 = m8.merge(fin, on=["market_id", "selection_id"], suffixes=("", "_final"))
    m8["bsp"] = m8["bsp_final"].where(m8["bsp_final"] > 1, m8["bsp"])
    with np.errstate(invalid="ignore", divide="ignore"):
        tight = (m8["lay1"] > 1) & (m8["lay1"] / m8["back1"] <= rb.TIGHT)
    m8["market_now"] = np.where(tight, np.sqrt(m8["back1"] * m8["lay1"].where(tight, m8["back1"])), m8["back1"])
    m8 = m8[(m8["market_now"] > 1) & (m8["bsp"] > 1) & ~m8["final_status"].astype(str).eq("REMOVED")]
    rows = []
    for day, g in mk.groupby("race_date"):
        preds = core.predictions(date.fromisoformat(day))
        if preds.empty:
            continue
        markets = []
        for mid, gg in g.groupby("market_id"):
            start = pd.Timestamp(gg["market_start_utc"].iloc[0]).tz_convert("UTC").to_pydatetime()
            markets.append(Market(str(mid), str(gg["venue"].iloc[0]), str(gg["country"].iloc[0]), start,
                                  name=str(gg["market_name"].iloc[0]),
                                  runners={int(s): str(n) for s, n in zip(gg["selection_id"], gg["runner_name"])}))
        pp = place_prices(preds, markets, places_known or {})
        if len(pp):
            pp["race_date"] = day
            rows.append(pp)
    if not rows:
        print(f"  place closing: no model place prices before {to_day}")
        return None
    pp = pd.concat(rows, ignore_index=True)
    pp["market_id"] = pp["market_id"].astype(str)
    m8["market_id"] = m8["market_id"].astype(str)
    m8["selection_id"] = m8["selection_id"].astype(int)
    panel = m8.merge(pp[["market_id", "selection_id", "predicted_bfsp"]], on=["market_id", "selection_id"])
    whole = panel.groupby("market_id").size()
    field = mk.groupby("market_id").selection_id.nunique()
    keep = whole.index[whole >= 0.9 * field.reindex(whole.index).fillna(999)]
    panel = panel[panel.market_id.isin(keep)]
    if len(panel) < MIN_FIT_RUNNERS:
        print(f"  place closing: {len(panel)} runners in whole place races before {to_day}: too few to fit")
        return None
    d = pd.DataFrame({"race": panel.market_id.to_numpy(), "morningwap": panel.market_now.to_numpy(),
                      "predicted_bfsp": panel.predicted_bfsp.to_numpy(), "morning_vol": 0.0})
    inputs = rb.closing_inputs(d)
    g = rb.race_codes(d["race"])
    model = rb.fit_closing_model(inputs, panel.bsp.to_numpy(), g)
    out_dir = core.ensure_dir(out_dir)
    path = out_dir / f"place_closing_{to_day}.json"
    rb.save_closing_model(model, path, fitted_on=f"{FIT_FROM}..{to_day}", runners=int(len(panel)),
                          races=int(g.max() + 1))
    print(f"  place closing before {to_day}: {len(panel)} runners in {g.max() + 1} races, beta "
          f"{np.round(model.beta[:2], 3).tolist()} (market, model), sigma {np.round(model.band_sigma, 3).tolist()}")
    return str(path)
