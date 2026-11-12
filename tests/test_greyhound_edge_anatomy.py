"""research/greyhound/edge_anatomy.py: every entry priced at what the book offers; a planted edge (the projected SP
knows where the SP ends) found on the holdout; the whole book taken loses the spread; read-only."""
from __future__ import annotations

import ast
import inspect
from datetime import date, timedelta

import numpy as np
import pandas as pd

import betfair_recorder
from greyhound import fav_back as fb
from research.greyhound import edge_anatomy as ea
from sources.common import Store

D0 = date(2026, 11, 3)


def _days(n_days=6, seed=0):
    rng = np.random.default_rng(seed)
    books, markets, close = {}, {}, {}
    for k in range(n_days):
        day = D0 + timedelta(days=k)
        B, M, C = [], [], []
        for m in range(30):
            mid_ = f"1.{k}{m:03d}"
            start = pd.Timestamp(f"{day}T12:00:00Z") + pd.Timedelta(minutes=10 * m)
            sp = np.exp(rng.normal(1.7, 0.5, 6))
            mid = sp * np.exp(rng.normal(0, 0.12, 6))              # the book is off the SP; the projection is not
            for s in range(6):
                sid = 100 * m + s
                M.append({"market_id": mid_, "market_type": "WIN", "market_start_utc": start.isoformat(),
                          "venue": "Hove", "selection_id": sid})
                C.append({"market_id": mid_, "selection_id": sid, "bsp": sp[s], "won": int(s == 0)})
                for secs in (605, 305, 185, 125, 65, 35, 20):
                    B.append({"polled_utc": (start - pd.Timedelta(seconds=secs)).isoformat(), "source": "live",
                              "market_id": mid_, "status": "OPEN", "inplay": False, "selection_id": sid,
                              "back1": mid[s] * 0.98, "back1_size": 30.0, "lay1": mid[s] * 1.02, "lay1_size": 30.0,
                              "last_traded": mid[s], "runner_total_matched": 100.0, "market_total_matched": 900.0,
                              "sp_near": sp[s] * np.exp(rng.normal(0, 0.01)), "sp_far": sp[s]})
            B.append({"polled_utc": (start + pd.Timedelta(seconds=12)).isoformat(), "source": "live",
                      "market_id": mid_, "status": "OPEN", "inplay": True, "selection_id": 100 * m})
        books[day], markets[day], close[day] = pd.DataFrame(B), pd.DataFrame(M), pd.DataFrame(C)
    return books, markets, close


def _patch(monkeypatch, books, markets, close):
    def read(day, name, s3=None, root=None):
        return (books if name.startswith("books") else markets).get(day)
    monkeypatch.setattr(betfair_recorder, "_read_day_csv", read)
    monkeypatch.setattr(fb, "closing", lambda d, s3=None: close[d])


def test_the_planted_edge_is_found_and_the_blind_taker_pays_the_spread(tmp_path, monkeypatch):
    books, markets, close = _days()
    _patch(monkeypatch, books, markets, close)
    res = ea.run(Store(root=tmp_path / "s3"), D0, D0 + timedelta(days=5), tmp_path / "out" / "edge")
    assert len(res["development_days"]) == 3 and len(res["holdout_days"]) == 3
    assert all(d["late_off_median_s"] == 12.0 for d in res["days"])
    g = pd.read_csv(tmp_path / "out" / "edge_grid_all.csv")
    blind = g[(g.signal == "none") & (g.band == "all")]
    assert (blind["ret%"] < 0).all()                             # 2% each side of the mid, and commission on the wins
    near = g[(g.signal == "near") & (g.band == "all") & (g.thr == 0.05) & (g.mark == 60)]
    assert (near["ret%"] > 3).all() and (near["lo%"] > 0).all() and (near.bets >= 100).all()
    assert res["candidates_with_holdout_interval_above_0"] >= 5
    md = (tmp_path / "out" / "edge.md").read_text()
    assert "Where the SP ends against the book" in md and "read on the holdout" in md


def test_a_lay_is_priced_at_the_offer_and_staked_at_the_money_shown():
    d = pd.DataFrame({"day": ["d"], "market_id": ["1.1"], "selection_id": [1.0], "mark": [60], "track": ["Hove"],
                      "trap": [1], "spread": [0.04], "bsp": [5.0], "won": [0], "back1": [3.9], "lay1": [4.0],
                      "back1_size": [4.0], "lay1_size": [50.0], "pred_bsp": [6.0], "sp_near": [5.0], "sp_far": [0.0]})
    e = ea.entries(d).set_index("side")
    assert np.isclose(e.loc["lay", "g"], 1 - 4.0 / 5.0) and np.isclose(e.loc["lay", "net"], 0.2 * 0.98)
    assert np.isclose(e.loc["back", "g"], 3.9 / 5.0 - 1) and e.loc["back", "net"] < 0
    assert e.loc["lay", "stake"] == ea.STAKE and e.loc["back", "stake"] == 4.0
    assert np.isclose(e.loc["lay", "s_model"], np.log(6 / 4)) and np.isnan(e.loc["lay", "s_far"])


def test_the_false_discovery_rate_and_no_order_call():
    assert list(ea.bh(pd.Series([0.001, 0.002, 0.5, 0.9]))) == [True, True, False, False]
    src = inspect.getsource(ea)
    names = {n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute)}
    assert not names & {"_place", "place", "back_at_sp", "cancel", "_call", "put", "put_parquet"}
    assert "placeOrders" not in src and "betfair_client" not in src
