"""research/greyhound/day_diagnostic.py: the day read end to end from the store; read-only (no order call)."""
from __future__ import annotations

import ast
import inspect
from datetime import date

import numpy as np
import pandas as pd

from greyhound import early_lay as el, fav_back as fb
from research.greyhound import day_diagnostic as dd
from sources.common import Store

DAY = date(2026, 11, 11)


def _put(store, key, df):
    store.put(key, df.to_csv(index=False).encode())


def _day(tmp_path):
    store = Store(root=tmp_path / "s3")
    rng = np.random.default_rng(0)
    preds, close, led = [], [], []
    for m in range(12):
        mid = f"1.{100 + m}"
        sp = np.exp(rng.normal(1.6, 0.5, 6))
        for k in range(6):
            preds.append({"market_id": mid, "selection_id": 10 * m + k, "race_date": f"{DAY}", "race_time":
                          f"{12 + m // 4}:{15 * (m % 4):02d}", "track": "Romford" if m % 2 else "Hove", "trap": k + 1,
                          "p_model": 1 / sp[k], "pred_bsp": sp[k] * np.exp(rng.normal(0, 0.2)),
                          "pred_bsp_raw": sp[k], "model_objective": "stack_share"})
            if m < 10:                                           # two races not settled yet
                close.append({"market_id": mid, "selection_id": 10 * m + k, "bsp": sp[k], "won": int(k == 0)})
        base = {"mode": "live", "market_id": mid, "selection_id": 10 * m + 5, "track": "Hove", "race_time": "12:00",
                "trap": 6, "dog": "D"}
        led.append({**base, "time_utc": f"{DAY}T11:{m:02d}:00Z", "action": "lay", "price": 5.0, "size": 25.0,
                    "f_bsp": 7.0, "mid": 5.2, "seconds_to_off": 600})
        if m % 3 == 0:
            led.append({**base, "time_utc": f"{DAY}T11:{m:02d}:50Z", "action": "green", "lay_matched": 25.0,
                        "locked_pnl": 1.5, "seconds_to_off": 5})
        led.append({**base, "selection_id": 10 * m + 1, "time_utc": f"{DAY}T11:{m:02d}:01Z", "action": "no_lay",
                    "error": "risk gate: over the market limit 812.50"})
    preds, close, led = pd.DataFrame(preds), pd.DataFrame(close), pd.DataFrame(led)
    _put(store, f"greyhound/live/predictions/{DAY}.csv", preds)
    _put(store, f"{el.PREFIX}/live/{DAY}/ledger.csv", led)
    _put(store, f"{el.PREFIX}/paper/{DAY}/ledger.csv", led.assign(mode="paper"))
    return store, close


def test_the_day_is_read_end_to_end(tmp_path, monkeypatch):
    store, close = _day(tmp_path)
    monkeypatch.setattr(fb, "closing", lambda d, s3=None: close)
    import betfair_recorder
    monkeypatch.setattr(betfair_recorder, "_read_day_csv", lambda *a, **k: None)
    res = dd.run(store, DAY, tmp_path / "out" / "day")
    assert res["coverage"]["settled_markets"] == 10 and res["coverage"]["settled_not_priced"] == 0
    assert res["price"]["runners"] == 60 and 0.1 < res["price"]["rmse_log_bsp"] < 0.3
    t = res["early_lay_live"]["totals"]
    assert t["dogs_laid"] == 12 and t["dogs_matched"] == 4 and np.isclose(t["locked_pnl"], 6.0)
    assert np.isclose(t["liability_matched"], 4 * 25.0 * 4.0)
    assert any("risk gate" in k for k in res["early_lay_live"]["reasons"])
    assert res["live_vs_paper"]["in_both"] == 12
    md = (tmp_path / "out" / "day.md").read_text()
    assert "By lay price" in md and "The 20 worst misses" in md and "no data found" in md


def test_missing_files_say_so(tmp_path, monkeypatch):
    monkeypatch.setattr(fb, "closing", lambda d, s3=None: pd.DataFrame(columns=["market_id", "selection_id", "bsp",
                                                                                 "won"]))
    import betfair_recorder
    monkeypatch.setattr(betfair_recorder, "_read_day_csv", lambda *a, **k: None)
    dd.run(Store(root=tmp_path / "s3"), DAY, tmp_path / "out" / "day")
    md = (tmp_path / "out" / "day.md").read_text()
    assert "Predictions file: no data found" in md and "no ledger found" in md


def test_the_module_cannot_send_an_order():
    src = inspect.getsource(dd)
    names = {n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute)}
    assert not names & {"_place", "place", "back_at_sp", "lay", "back", "cancel", "_call", "put", "put_parquet"}
    assert "placeOrders" not in src and "betfair_client" not in src
