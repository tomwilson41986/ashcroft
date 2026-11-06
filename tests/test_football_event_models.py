"""Event models: geometry, shot features, the xT grid's shape and sense."""

from __future__ import annotations

import numpy as np
import pandas as pd

from football import event_models as em
from sources.wyscout_public import _unescape


def test_geometry_penalty_spot_and_angle():
    d, a = em.geometry([100 - 11 / 1.05], [50])
    assert abs(d[0] - 11) < 1e-9 and 0.6 < a[0] < 0.7          # ~36.8 degrees from the spot


def _ev():
    rows = [
        # cross then a header goal; through ball then a shot; a penalty
        (1, "1H", 10.0, 7, 85, 8, "1801", 60, 90, 95, 50),
        (1, "1H", 12.0, 7, 80, 8, "1801", 80, 95, 95, 50),
        (1, "1H", 13.0, 7, 100, 10, "101 403", 94, 50, None, None),
        (1, "1H", 50.0, 8, 86, 8, "901 1801", 50, 50, 88, 45),
        (1, "1H", 52.0, 8, 100, 10, "402", 88, 45, None, None),
        (1, "2H", 5.0, 7, 35, 3, "101", 89, 50, None, None),
    ]
    df = pd.DataFrame(rows, columns=["match_id", "period", "sec", "team_id", "sub_event_id", "event_id", "tags",
                                     "x", "y", "x_end", "y_end"])
    df["id"] = range(len(df))
    df["competition"] = "X"
    return df


def test_shot_features():
    s = em.shots(_ev())
    assert len(s) == 3
    head = s.iloc[0]
    assert head["goal"] == 1 and head["header"] == 1 and head["from_cross"] == 1
    assert s.iloc[1]["from_through"] == 1 and s.iloc[1]["goal"] == 0
    assert s.iloc[2]["penalty"] == 1


def test_xt_grid_rises_towards_goal():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(4000):
        x = rng.uniform(0, 100)
        rows.append((1, "1H", i, 7, 85, 8, "1801", x, 50, min(100, x + 10), 50))
        if x > 80 and rng.random() < 0.2:
            rows.append((1, "1H", i + 0.5, 7, 100, 10, "101" if rng.random() < 0.2 else "", x, 50, None, None))
    df = pd.DataFrame(rows, columns=["match_id", "period", "sec", "team_id", "sub_event_id", "event_id", "tags",
                                     "x", "y", "x_end", "y_end"])
    xt, info = em.fit_xt(df)
    assert xt.shape == (em.GRID_Y, em.GRID_X)
    row = xt[em.GRID_Y // 2]
    assert row[-1] > row[0] and info["shots"] > 0


def test_unescape_names():
    assert _unescape("K. Mbapp\\u00e9") == "K. Mbappé" and _unescape("Salah") == "Salah"
