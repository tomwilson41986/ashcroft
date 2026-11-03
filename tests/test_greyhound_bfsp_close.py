"""The closing-price study (research/greyhound/bfsp_close.py) end to end on a synthetic panel whose SP moves from the
price now by a signal the book carries: the boosted model must beat the mid, and nothing reads the day it forecasts."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.greyhound import bfsp_close as bc, limit_study as ls
from tests.test_greyhound_limit_study import synthetic_panel


def panel_with_marks(n_days=12):
    p = synthetic_panel(n_days=n_days)
    # the weight of money carries the move to the SP: the side with more money is where the price goes
    rng = np.random.default_rng(1)
    move = np.log(p.bsp / p["mid"])
    p["back1_size"] = 20.0 * np.exp(-2.0 * move + rng.normal(0, 0.1, len(p)))
    p["lay1_size"] = 20.0 * np.exp(2.0 * move + rng.normal(0, 0.1, len(p)))
    return p


def test_the_study_end_to_end(tmp_path, monkeypatch):
    panel = panel_with_marks()
    monkeypatch.setattr(ls, "load", lambda *a, **k: panel)
    out = tmp_path / "bf"
    assert bc.main(["--store", str(tmp_path), "--from", "2026-10-06", "--to", "2026-10-17", "--out", str(out),
                    "--min-train-days", "7"]) == 0
    res = json.loads(out.with_suffix(".json").read_text())
    s = pd.DataFrame(res["scores"])
    assert set(s.model) == set(bc.MODELS) and len(s)
    for mark, g in s.groupby("mark"):
        r = g.set_index("model")
        assert r.loc["gbm", "rmse"] < r.loc["mid", "rmse"]           # the book's signal is learnt
    assert "| RMSE log(BSP) |" in out.with_suffix(".md").read_text()


def test_a_forecast_never_reads_its_own_day():
    panel = bc.book_features(panel_with_marks(n_days=9))
    g = panel[panel["mark"] == "first_traded"]
    days = sorted(g.race_date.unique())
    tr, te = g[g.race_date < days[-1]], g[g.race_date == days[-1]]
    a = bc.forecast_day(tr, te)
    te2 = te.assign(bsp=te.bsp * 3.0, won=1 - te.won)                # the day's own results rewritten
    b = bc.forecast_day(tr, te2)
    for m in bc.MODELS:
        assert np.allclose(a[m], b[m])


def test_rescale_keeps_the_order_and_sets_the_book():
    d = pd.DataFrame({"race_id": ["r"] * 3, "mark": ["off_1"] * 3})
    lp = np.log([2.0, 4.0, 8.0])
    out = bc.rescale(lp, d, book=1.10)
    assert np.all(np.diff(out) > 0) and np.isclose(np.exp(-out).sum(), 1.10)
