"""research/greyhound/back_study.py: every back rule read on both periods, the two endings the owner asked for (trade
out at the SP, back to win), the controls, the false discovery rate."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.greyhound import back_study as bs, limit_study as ls
from tests.test_greyhound_limit_study import synthetic_panel


def _run(tmp_path, monkeypatch, n_days=20):
    panel = synthetic_panel(n_days=n_days)
    books = panel[["market_id", "selection_id", "polled"]].drop_duplicates(["market_id", "selection_id"]).assign(
        polled=lambda x: x.polled + pd.Timedelta(minutes=1), back1=panel.last_back1, lay1=panel.last_lay1,
        last_traded=panel.bsp)

    def fake_load(*a, books_out=None, **k):
        if books_out is not None:
            books_out.append(books)
        return panel
    monkeypatch.setattr(ls, "load", fake_load)
    out = tmp_path / "back"
    res = bs.run(None, pd.Timestamp("2026-10-06").date(), pd.Timestamp("2026-10-25").date(), None, out)
    return res, json.loads(out.with_suffix(".json").read_text()), out.with_suffix(".md").read_text()


def test_every_back_rule_is_read_on_both_periods_with_both_endings(tmp_path, monkeypatch):
    res, full, md = _run(tmp_path, monkeypatch)
    rules = pd.DataFrame(full["rules"])
    assert set(rules["exit"]) == set(bs.EXITS) | {"hold"}
    assert set(rules.gate) <= set(bs.GATES) and {"model_fav", "value10", "shorten20"} & set(rules.gate)
    assert rules.trades_dev.notna().any() and rules.trades_hold.notna().any()
    assert res["rules_tried"] == int((~rules.control).sum())
    assert res["expected_by_chance"] == round(0.05 * res["rules_tried"], 1)
    # the held ending is the same entries as the close at the SP
    k = list(bs.KEYS)
    sp = rules[rules["exit"] == "sp"].set_index(k).trades_dev
    hold = rules[rules["exit"] == "hold"].set_index(k).trades_dev
    assert sp.reindex(hold.index).equals(hold)
    assert "Back to win" in md and "trade out at the SP" in md and "The market on its own" in md


def test_the_false_discovery_rate():
    p = pd.Series([0.001, 0.002, 0.04, 0.3, 0.6, 0.9])
    d = bs.bh(p, q=0.10)
    assert list(d) == [True, True, True, False, False, False]
    assert not bs.bh(pd.Series([0.5, 0.6, 0.7])).any()


def test_the_interval_and_p_value():
    lo, hi, p = bs._interval(np.array([1.0, 2.0, 1.5, 0.5]), np.array([10.0, 10.0, 10.0, 10.0]))
    assert 0 < lo <= hi and p == 0.0
    lo, hi, p = bs._interval(np.array([-1.0, -2.0, 1.0, -1.5]), np.array([10.0, 10.0, 10.0, 10.0]))
    assert lo < 0 and p > 0.5
