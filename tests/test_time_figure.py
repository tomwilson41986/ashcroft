"""The horse's time figure (model/blocks/time_figure.py), worked by hand.

A course whose 6f standard is 12.0 s a furlong (six earlier days at 72.0 s),
then a meeting of two races, one 0.1 s/f fast and one 0.1 s/f slow, so the
day's allowance is nil. Mechanics only; nothing is trained on this.
"""
import numpy as np
import pandas as pd
import pytest

from model.blocks import time_figure as tf
from model.perf_figures import lbs_per_length


def _race(day, time, secs, runners, track="york", dist=6.0):
    return [dict(race_date=pd.Timestamp(day), race_time=time, track=track, horse_name=h, race_type="Handicap",
                 surface_type="Turf", going_description="Good", dist_furlongs=dist, number_of_runners=len(runners),
                 comptime_numeric=secs, placing_numerical=float(p), LB=float(lb), pounds=float(w))
            for h, p, lb, w in runners]


def _frame():
    rows = []
    for d in range(6):
        rows += _race(f"2024-01-{d + 1:02d}", "2.00", 72.0, [(f"a{d}", 1, 0, 130), (f"b{d}", 2, 1, 130)])
    rows += _race("2024-02-01", "2.00", 71.4, [("x", 2, 2.0, 135), ("w", 1, 0, 130)])
    rows += _race("2024-02-01", "3.00", 72.6, [("y", 1, 0, 130), ("z", 2, 1.0, 130)])
    rows += _race("2024-03-01", "2.00", np.nan, [("x", np.nan, np.nan, 130), ("y", np.nan, np.nan, 130)])
    return pd.DataFrame(rows)


def test_the_figure_by_hand():
    df = _frame()
    f = tf.figures(df)
    lpl = lbs_per_length(6.0)
    speed = 6 * tf.FURLONG_METRES / 71.4
    race_a = 0.6 * speed / tf.LENGTH_METRES * lpl                 # 0.6 s faster than standard, no allowance
    x = df.index[(df["horse_name"] == "x") & (df["race_date"] == "2024-02-01")][0]
    assert f["ga"][x] == pytest.approx(0.0, abs=1e-12)
    assert f["race_fig"][x] == pytest.approx(race_a)
    assert f["fig"][x] == pytest.approx(race_a - 2.0 * lpl + 5.0)  # beaten 2 lengths, carried 135
    y = df.index[(df["horse_name"] == "y") & (df["race_date"] == "2024-02-01")][0]
    speed_b = 6 * tf.FURLONG_METRES / 72.6
    assert f["fig"][y] == pytest.approx(-0.6 * speed_b / tf.LENGTH_METRES * lpl)
    # the first days have no standard yet: no figure
    assert np.isnan(f["fig"][0])


def test_the_windows_read_earlier_runs():
    out = tf.build(_frame())
    card = out[out["race_date"] == "2024-03-01"].set_index("horse_name")
    f = tf.figures(_frame())
    x_fig = f["fig"][(_frame()["horse_name"] == "x").to_numpy() & (_frame()["race_date"] == "2024-02-01").to_numpy()][0]
    y_fig = f["fig"][(_frame()["horse_name"] == "y").to_numpy() & (_frame()["race_date"] == "2024-02-01").to_numpy()][0]
    assert card.loc["x", "tf_l1"] == pytest.approx(x_fig) and card.loc["x", "tf_best"] == pytest.approx(x_fig)
    assert card.loc["x", "tf_n"] == 1
    best = max(x_fig, y_fig)
    assert card.loc["x", "tf_best5_rel"] == pytest.approx(x_fig - best)
    assert card.loc["y", "tf_best5_rel"] == pytest.approx(y_fig - best)
