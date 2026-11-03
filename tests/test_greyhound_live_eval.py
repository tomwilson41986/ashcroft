"""Every strategy on the live books (research/greyhound/live_eval.py), end to end on a synthetic panel."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd

from research.greyhound import limit_study as ls, live_eval as le, tradeout_study as ts
from tests.test_greyhound_limit_study import synthetic_panel


def test_the_live_evaluation_end_to_end(tmp_path, monkeypatch):
    hist = synthetic_panel(n_days=10)
    live = synthetic_panel(n_days=1, seed=3).assign(race_date="2026-11-03")
    live["race_id"] = "L" + live.race_id.astype(str)
    live["market_id"] = "9" + live.market_id.astype(str)
    books = live[["market_id", "selection_id", "polled"]].drop_duplicates(["market_id", "selection_id"]).assign(
        polled=lambda x: x.polled + pd.Timedelta(minutes=1), back1=live.last_back1, lay1=live.last_lay1,
        last_traded=live.bsp)
    monkeypatch.setattr(le, "live_panel", lambda store, d: (live, books) if d == date(2026, 11, 3) else
                        (pd.DataFrame(), None))
    monkeypatch.setattr(ls, "load", lambda *a, **k: hist)
    out = tmp_path / "ev"
    assert le.main(["--store", str(tmp_path), "--from", "2026-11-03", "--to", "2026-11-03", "--out", str(out)]) == 0
    res = json.loads(out.with_suffix(".json").read_text())
    assert res["days"] == 1 and res["races"] == live.race_id.nunique()
    assert res["early_lay"] and {r["fill"] for r in res["early_lay"]} == set(ls.FILLS)
    assert {r["exit"] for r in res["early_lay"]} == set(ls.EXITS)
    assert res["families"] and res["top"]
    md = out.with_suffix(".md").read_text()
    assert "## 1. The early lay" in md and "## 3. By family" in md


def test_the_forecast_is_fitted_on_the_days_before():
    hist = synthetic_panel(n_days=9)
    import research.greyhound.live_eval as m
    f = {}
    for mark in ls.ORDER_MARKS:
        g = hist[hist["mark"] == mark]
        f[mark] = ts.fit_cl(g, ts.CL_SETS["micro"])
    fc = m.forecast(synthetic_panel(n_days=1, seed=5), f)
    assert len(fc) and (fc.f_bsp > 1).all() and set(fc["mark"]) == set(ls.ORDER_MARKS)
