"""The greyhound limit-order study (research/greyhound/limit_study.py): the ladder, the later books, the fills, the
settlement, the behaviour tables, and the study end to end on a synthetic panel."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from greyhound import tradeout as to
from research.greyhound import limit_study as ls


def test_ticks_on_betfairs_ladder():
    assert ls.ticks_from([2.0, 3.0, 9.8, 19.5], 1, up=True).tolist() == [2.02, 3.05, 10.0, 20.0]
    assert ls.ticks_from([2.0, 3.0, 10.0], 3, up=False).tolist() == [1.97, 2.94, 9.4]
    assert ls.ladder([2.013, 5.04], up=True).tolist() == [2.02, 5.1]
    assert ls.ladder([2.013, 5.04], up=False).tolist() == [2.0, 5.0]


def _books():
    t0 = pd.Timestamp("2026-10-20 12:00", tz="UTC")
    rows = []
    for i, (b, l, lt) in enumerate([(4.0, 4.4, 4.2), (4.6, 5.0, 4.8), (4.2, 4.5, 4.3), (3.9, 4.1, 4.0)]):
        rows.append({"market_id": "1.1", "selection_id": 7.0, "polled": t0 + pd.Timedelta(minutes=i),
                     "back1": b, "lay1": l, "last_traded": lt})
    return pd.DataFrame(rows), t0


def test_the_later_books_are_strictly_after_each_row():
    pre, t0 = _books()
    fut = ls.future_extremes(pre)
    panel = pd.DataFrame({"market_id": ["1.1", "1.1"], "selection_id": [7.0, 7.0],
                          "polled": [t0, t0 + pd.Timedelta(minutes=2)]})
    a = ls.attach_future(panel, fut)
    # after the first book: back 4.6, 4.2, 3.9 (max 4.6); lay 5.0, 4.5, 4.1 (min 4.1)
    assert a.fut_back_max.tolist() == [4.6, 3.9] and a.fut_lay_min.tolist() == [4.1, 4.1]
    assert a.fut_lt_max.tolist() == [4.8, 4.0] and a.fut_lt_min.tolist() == [4.0, 4.0]
    assert a.last_back1.tolist() == [3.9, 3.9] and a.last_lay1.tolist() == [4.1, 4.1]      # the last book before the off
    last = ls.attach_future(panel.assign(polled=t0 + pd.Timedelta(minutes=3)), fut)
    assert last.fut_back_max.isna().all()                     # no later book: nothing can be matched


def test_fills_take_now_rest_or_lapse():
    d = pd.DataFrame({"back1": [4.0, 4.0, 4.0], "lay1": [4.4, 4.4, 4.4], "fut_back_max": [4.6, 4.6, np.nan],
                      "fut_lay_min": [4.1, 4.1, np.nan], "fut_lt_max": [4.8, 4.8, np.nan],
                      "fut_lt_min": [4.0, 4.0, np.nan]})
    f, at, taken = ls.fills(d, "back", np.array([3.8, 4.6, 4.6]), "through")
    assert f.tolist() == [True, True, False] and at.tolist() == [4.0, 4.6, 4.6] and taken.tolist() == [True, False, False]
    f, _, _ = ls.fills(d, "back", np.array([4.8, 4.8, 4.8]), "through")
    assert f.tolist() == [False, False, False]
    f, _, _ = ls.fills(d, "back", np.array([4.8, 4.8, 4.8]), "traded")
    assert f.tolist() == [True, True, False]
    f, at, _ = ls.fills(d, "lay", np.array([4.5, 4.1, 4.1]), "through")
    assert f.tolist() == [True, True, False] and at.tolist() == [4.4, 4.1, 4.1]
    f, _, _ = ls.fills(d, "back", np.array([25.0, np.nan, 1.0]), "traded")
    assert not f.any()                                        # over the cap, no price, no price on the ladder


def test_orders_settle_as_the_horse_trader_closes():
    d = pd.DataFrame({"race_date": ["2026-10-20"] * 2, "market_id": ["1.1"] * 2, "mark": ["off_5"] * 2,
                      "won": [0, 1], "bsp": [4.0, 4.0], "back1": [4.0, 4.0], "lay1": [4.4, 4.4], "mid": [4.2, 4.2],
                      "f_bsp": [4.0, 4.0], "rank": [1.0, 2.0], "fut_back_max": [5.0, 5.0], "fut_lay_min": [3.5, 3.5],
                      "fut_lt_max": [5.0, 5.0], "fut_lt_min": [3.5, 3.5], "last_back1": [5.0, 5.0],
                      "last_lay1": [5.2, 5.2]})
    o = ls.orders(d)
    b = o[(o.order == "ask") & (o.param == 0.2) & (o.fill == "through") & (o["exit"] == "sp")]
    # a back at 4.8 (4.0 x 1.2) matched as the price drifts to 5.0, closed at an SP of 4.0
    assert b.filled.all() and np.allclose(b.price, 4.8) and np.allclose(b.clv, 0.2)
    lost, won = b.iloc[0], b.iloc[1]
    s = to.target_stake(4.8)
    assert np.isclose(lost.pnl, -s + s * 3.8 / 3.0) and won.pnl == 0.0
    lay = o[(o.order == "bid") & (o.param == 0.1) & (o.fill == "through") & (o["exit"] == "sp")]
    assert np.allclose(lay.price, 3.6) and np.isclose(lay.iloc[1].pnl, to.target_stake(3.6) * (4.0 - 3.6))
    assert set(o.fill) == set(ls.FILLS) and set(o["exit"]) == {"sp", "last"}     # no path: no take-profit exits
    assert o.g_fav.tolist().count(True) == o.g_fav.size // 2      # the favourite: the first runner only
    # lay to back: laid at 3.6, backed at the last book's best back price 5.0, greened up whatever the result
    lb = o[(o.order == "bid") & (o.param == 0.1) & (o.fill == "through") & (o["exit"] == "last")]
    s = to.target_stake(3.6)
    assert np.allclose(lb.pnl, s * (1 - 3.6 / 5.0)) and lb.closed_at_market.all()
    # back to lay: backed at 4.8, laid at the last best lay price 5.2: a loss either way
    bl = o[(o.order == "ask") & (o.param == 0.2) & (o.fill == "through") & (o["exit"] == "last")]
    assert np.allclose(bl.pnl, to.target_stake(4.8) * (4.8 / 5.2 - 1))
    tk = o[(o.order == "take") & (o.side == "back") & (o.fill == "through") & (o["exit"] == "sp")]
    assert tk.taken.all() and np.allclose(tk.price, 4.0)


def test_the_take_profit_reads_the_books_after_the_fill():
    pre, t0 = _books()          # back/lay/traded: 4.0/4.4/4.2, 4.6/5.0/4.8, 4.2/4.5/4.3, 3.9/4.1/4.0 a minute apart
    e = pd.DataFrame({"race_date": ["2026-10-20"] * 5, "market_id": ["1.1"] * 5, "selection_id": [7.0] * 5,
                      "polled": [t0] * 5, "side": ["lay", "lay", "lay", "back", "back"],
                      "fill": ["through", "through", "through", "through", "traded"],
                      "limit": [4.4, 4.5, 4.1, 4.6, 4.8], "taken": [True, False, False, False, False],
                      "filled": [True, True, True, True, True]})
    best = ls.exit_paths(e, pre)
    # taken at the mark: the highest back after it 4.6; a lay at 4.5 matched at 2 min, then backs of 3.9 only;
    # a lay at 4.1 matched in the last book: nothing after; a back at 4.6 matched at 1 min, then the lowest lay 4.1;
    # a back at 4.8 by the traded rule matched at 1 min, then the lowest traded 4.0
    assert best[0] == 4.6 and best[1] == 3.9 and np.isnan(best[2]) and best[3] == 4.1 and best[4] == 4.0
    assert np.isnan(ls.exit_paths(e.assign(filled=False), pre)).all()


def test_take_profit_locks_the_move_else_greens_up_at_the_last_book():
    o = pd.DataFrame({"side": ["lay", "lay", "back"], "price": [4.4, 4.4, 4.6], "bsp": [4.0, 4.0, 4.0],
                      "won": [1, 0, 0], "exit": ["tp2", "tp5", "tp5"], "exit_best": [4.6, 4.6, 4.1],
                      "last_back1": [4.2, 4.2, 4.2], "last_lay1": [4.3, 4.3, 4.3]})
    r = ls.settle_orders(o)
    s = to.target_stake(np.array([4.4, 4.4, 4.6]))
    # a lay at 4.4: two ticks up is 4.6, reached: backed there, profit locked though the dog won
    assert r.took_profit.tolist() == [True, False, True] and np.isclose(r.pnl[0], s[0] * (1 - 4.4 / 4.6))
    # five ticks up (4.9) never came: greened up at the last back price, 4.2, a loss
    assert np.isclose(r.pnl[1], s[1] * (1 - 4.4 / 4.2))
    # a back at 4.6: five ticks down is 4.1, reached: laid there
    assert np.isclose(r.pnl[2], s[2] * (4.6 / 4.1 - 1)) and r.closed_at_market.all()


def test_the_orders_carry_the_take_profit_exits_with_the_path():
    pre, t0 = _books()
    d = pd.DataFrame({"race_date": ["2026-10-20"], "market_id": ["1.1"], "selection_id": [7.0], "polled": [t0],
                      "mark": ["off_5"], "won": [0], "bsp": [4.0], "back1": [4.0], "lay1": [4.4], "mid": [4.2],
                      "f_bsp": [4.0], "rank": [1.0], "fut_back_max": [4.6], "fut_lay_min": [4.1],
                      "fut_lt_max": [4.8], "fut_lt_min": [4.0], "last_back1": [3.9], "last_lay1": [4.1]})
    o = ls.orders(d, pre)
    assert set(o["exit"]) == set(ls.EXITS)
    lay = o[(o.order == "take") & (o.side == "lay") & (o.fill == "through") & (o["exit"] == "tp2")]
    assert lay.took_profit.all() and np.isclose(lay.pnl.iloc[0], to.target_stake(4.4) * (1 - 4.4 / 4.6))


def synthetic_panel(n_days=16, races=30, seed=0):
    """Races whose SP is the price now moved by a signal our price carries, with books after each mark."""
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp("2026-10-06 12:00", tz="UTC")
    for d in range(n_days):
        day = (t0 + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        for r in range(races):
            k = 6
            p = rng.dirichlet(np.ones(k) * 3)
            sig = rng.normal(0, 0.25, k)
            close = p * np.exp(sig)
            close /= close.sum()
            bsp = 1.0 / close
            won = np.zeros(k, int)
            won[rng.choice(k, p=close)] = 1
            for i, mark in enumerate(ls.ORDER_MARKS):
                mid = 1.0 / (p * (1.06 - 0.02 * i))                 # the book tightens towards the off
                back1 = mid / 1.04
                lay1 = mid * 1.04
                for j in range(k):
                    rows.append({"race_id": f"{day}-{r}", "race_date": day, "market_id": f"1.{d}{r:03d}",
                                 "selection_id": float(j), "mark": mark, "field": k, "won": won[j],
                                 "p_model": float(close[j] * 0.7 + p[j] * 0.3), "bsp": float(bsp[j]),
                                 "back1": float(back1[j]), "lay1": float(lay1[j]), "back1_prev": float(back1[j]),
                                 "lay1_prev": float(lay1[j]), "back1_size": 20.0, "lay1_size": 20.0,
                                 "last_traded": float(mid[j]), "runner_total_matched": 100.0,
                                 "market_total_matched": 600.0, "polled": t0 + pd.Timedelta(days=d),
                                 "fut_back_max": float(max(back1[j], bsp[j] * rng.uniform(0.9, 1.1))),
                                 "fut_lay_min": float(min(lay1[j], bsp[j] * rng.uniform(0.9, 1.1))),
                                 "fut_lt_max": float(max(mid[j], bsp[j])), "fut_lt_min": float(min(mid[j], bsp[j])),
                                 "last_back1": float(bsp[j] / 1.02), "last_lay1": float(bsp[j] * 1.02)})
    return to.add_features(pd.DataFrame(rows))


def test_behaviour_tables():
    b = ls.behaviour(synthetic_panel(n_days=2))
    assert b["band"] and b["order_now"]
    r = b["order_now"][0]
    assert {"mark", "order_now", "runners", "drift%", "mean_log_sp_over_back", "back_reaches_sp%"} <= set(r)


def test_the_study_end_to_end(tmp_path, monkeypatch):
    panel = synthetic_panel(n_days=20)
    books = panel[["market_id", "selection_id", "polled"]].drop_duplicates(["market_id", "selection_id"]).assign(
        polled=lambda x: x.polled + pd.Timedelta(minutes=1), back1=panel.last_back1, lay1=panel.last_lay1,
        last_traded=panel.bsp)                                   # one book after the marks: the path

    def fake_load(*a, books_out=None, **k):
        if books_out is not None:
            books_out.append(books)
        return panel
    monkeypatch.setattr(ls, "load", fake_load)
    out = tmp_path / "limit"
    assert ls.main(["--store", str(tmp_path), "--from", "2026-10-06", "--to", "2026-10-25", "--out", str(out)]) == 0
    res = json.loads(out.with_suffix(".json").read_text())
    assert len(res["dev_days"]) == 20 - ls.MIN_TRAIN_DAYS - ls.HOLDOUT_DAYS and len(res["holdout_days"]) == 7
    assert res["dev_top"] and res["dev_chosen"] and res["holdout"]
    fam = {(r["mark"], r["order"], r["side"], r["fill"], r["exit"]) for r in res["holdout"]}
    assert len(fam) == len(res["holdout"])                    # one rule a family, fixed on the development days
    assert {r["exit"] for r in res["dev_rules"]} == set(ls.EXITS)  # the take-profit exits, read off the path
    hc = res["holdout_count"]
    assert hc["rules"] == len(res["holdout"]) and hc["expected_by_chance"] == round(0.05 * hc["rules"], 1)
    md = out.with_suffix(".md").read_text()
    assert "## 1. The market's behaviour" in md and "## 2. Holdout" in md


def test_the_forecast_is_walk_forward():
    panel = synthetic_panel(n_days=10)
    fc = ls.forecast(panel)
    days = sorted(panel.race_date.unique())
    assert sorted(fc.race_date.unique()) == days[ls.MIN_TRAIN_DAYS:] and (fc.f_bsp > 1).all()


def test_the_exits_compared_on_the_same_entries():
    base = {"mark": "first_traded", "order": "bid", "side": "lay", "param": 0.2, "gate": "drift", "fill": "traded",
            "per_day": 5.0}
    dev = pd.DataFrame([dict(base, exit="last", **{"roi%": 1.0}), dict(base, exit="sp", **{"roi%": 2.0}),
                        dict(base, exit="tp2", **{"roi%": 0.5}), dict(base, gate="all", exit="last", **{"roi%": 0.0},
                                                                     per_day=0.5)])
    hold = dev.assign(**{"roi%": [3.0, 1.0, 4.0, 9.0]})
    xc = ls.exit_compare(dev, hold)
    assert xc["rules"] == 1                                   # the thin entry rule (half a trade a day) left out
    s = {r["exit"]: r for r in xc["summary"]}
    assert s["sp"]["dev_median_vs_last"] == 1.0 and s["sp"]["holdout_median_vs_last"] == -2.0
    assert s["tp2"]["dev_better%"] == 0.0 and s["tp2"]["holdout_better%"] == 100.0
    assert xc["rows"][0]["hold_tp2"] == 4.0
