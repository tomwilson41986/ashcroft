"""Close model on the books: features are known at the checkpoint, the target is the move to the close."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.open_close import football_close_model_books as cm


def _rows(day, mid_id, cp, bb, bl, close_mid, runner="A"):
    return {"day": day, "market_id": mid_id, "market_type": "MATCH_ODDS", "event": "A v B", "runner": runner,
            "selection_id": 1, "checkpoint_min": cp, "minutes_to_off": cp, "kind": "cross", "side": "back",
            "best_back": bb, "best_lay": bl, "spread_ticks": 1, "book_over": 1.01, "depth_back_gbp": 100.0,
            "depth_lay_gbp": 50.0, "close_back": close_mid - 0.01, "close_lay": close_mid + 0.01,
            "close_mid": close_mid, "green_close": 0.01}


def test_build_drift_and_target():
    df = pd.DataFrame([_rows("d1", "1", 1440.0, 2.0, 2.02, 1.9), _rows("d1", "1", 60.0, 1.96, 1.98, 1.9)])
    r = cm.build(df).sort_values("checkpoint_min", ascending=False)
    assert abs(r["drift_since_first"].iloc[0]) < 1e-12
    assert r["drift_since_first"].iloc[1] > 0                     # the price shortened from 24 h to 60 min
    assert abs(r["y"].iloc[1] - np.log((1 / 1.9) / (1 / 1.97))) < 1e-9
    assert r["leg_code"].iloc[0] == 1


def test_run_reports_too_few_days():
    df = pd.DataFrame([_rows("d1", "1", 60.0, 2.0, 2.02, 1.9)])
    assert "too few days" in cm.run(df)["status"]


def test_gate_merges_numeric_and_text_selection_ids():
    p = pd.DataFrame({"market_id": ["1"], "selection_id": [5], "checkpoint_min": [60.0], "day": ["d3"],
                      "y_hat": [0.2], "spread_rel": [0.01], "close_half_spread": [0.005]})
    lines = pd.DataFrame({"market_id": ["1"], "selection_id": ["5"], "checkpoint_min": [60.0], "side": ["back"],
                          "green_close": [0.03], "day": ["d3"]})
    g = cm.gate(p, lines, "own")
    assert g["trades"] == 1 and g["backs"] == 1
