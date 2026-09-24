"""Form read against the pace and the draw each past run met (model/shape_form.py).

Synthetic frames test mechanics only: the actual shape from the comments, the
sign and bookkeeping of the pace-and-draw credit, the speed drawn near each
runner, and the lag rule. Nothing is trained or evaluated on them.
"""
import numpy as np
import pandas as pd
import pytest

from model.bfsp_features import assert_no_post_race_features
from model.shape_form import (
    SHAPE_FORM_FEATURES, SHAPE_FORM_POST_RACE, actual_shape, add_run_bias, add_shape_form, add_speed_near,
)


def _race(raceid, classes, lbs_c, day="2024-05-01", track="york", stalls=None, draw=None, p_lead=None):
    n = len(classes)
    return pd.DataFrame({
        "raceid": raceid, "race_date": pd.Timestamp(day), "race_time": "2.30", "track": track,
        "horse_name": [f"{raceid}-{i}" for i in range(n)], "race_type": "Handicap", "surface_type": "Turf",
        "dist_furlongs": 8.0, "number_of_runners": n, "rs_class": classes, "rs_lbs_c": lbs_c,
        "stall": stalls if stalls is not None else np.arange(1, n + 1),
        "dc_edge_rel_lbs": draw if draw is not None else np.zeros(n),
        "p_lead": p_lead if p_lead is not None else np.full(n, 0.2),
    })


def test_the_actual_shape_comes_from_the_runners_comments():
    df = pd.concat([
        _race("a", [1, 2, 3, 3], [0, 0, 0, 0]),                   # nobody led
        _race("b", [0, 2, 3, 3], [0, 0, 0, 0]),                   # one leader
        _race("c", [0, 0, 0, 3], [0, 0, 0, 0]),                   # three disputed it
        _race("d", [0, np.nan, np.nan, np.nan], [0, 0, 0, 0]),    # the comments place one of four
    ], ignore_index=True)
    shape = pd.Series(actual_shape(df)).groupby(df["raceid"]).first()
    assert shape["a"] == 0 and shape["b"] == 1 and shape["c"] == 2 and np.isnan(shape["d"])


def _history(n_days=400):
    """Contested races at one course where the hold-up horses always beat the leaders."""
    days = pd.date_range("2023-01-01", periods=n_days, freq="D")
    races = [_race(f"r{i}", [0, 0, 1, 2, 3, 3], [-6.0, -6.0, -1.0, 1.0, 6.0, 6.0], day=d)
             for i, d in enumerate(days)]
    return pd.concat(races, ignore_index=True)


def test_a_run_is_credited_for_the_pace_it_met():
    df = add_run_bias(_history())
    first = df[df["raceid"] == "r0"]
    last = df[df["raceid"] == df["raceid"].iloc[-1]].reset_index(drop=True)
    # the first race has no earlier day to learn from: no credit either way
    assert first["sf_pos_lbs"].abs().max() == 0
    # by the end, a leader's run was hindered by the pace it met and a hold-up run helped
    assert last.loc[0, "sf_pos_lbs"] < -3 and last.loc[5, "sf_pos_lbs"] > 3
    assert last["sf_pos_lbs"].sum() == pytest.approx(0.0, abs=1e-9)        # centred within the race
    # the adjusted figure is the run less the credit, so the leaders' runs read closer to the field's
    np.testing.assert_allclose(last["sf_adj"], last["rs_lbs_c"] - last["sf_bias"])
    assert abs(last.loc[0, "sf_adj"]) < 3 and abs(last.loc[5, "sf_adj"]) < 3


def test_the_draw_is_part_of_the_credit_and_an_unknown_position_takes_none():
    df = pd.concat([_history(30), _race("x", [0, np.nan, 3, 3, 3, 3], [0.0] * 6, day="2023-03-01",
                                         draw=[2.0, 1.0, 0.0, 0.0, -1.0, -2.0])], ignore_index=True)
    out = add_run_bias(df)
    x = out[out["raceid"] == "x"].reset_index(drop=True)
    assert np.isnan(x.loc[1, "sf_pos_lbs"]) and x.loc[1, "sf_bias"] == pytest.approx(1.0)   # the draw alone
    np.testing.assert_allclose(x["sf_bias"], np.nan_to_num(x["sf_pos_lbs"]) + x["sf_draw_lbs"])


def test_speed_drawn_inside_outside_and_near():
    # stalls 1, 2, 3, 5, 6, 8 (4 and 7 empty); two likely leaders, in stalls 2 and 6
    p = [0.1, 0.6, 0.0, 0.1, 0.5, 0.0]
    df = _race("s", [np.nan] * 6, [np.nan] * 6, stalls=[1, 2, 3, 5, 6, 8], p_lead=p)
    df = pd.concat([df, _race("nh", [np.nan] * 3, [np.nan] * 3, p_lead=[0.5, 0.5, 0.5])], ignore_index=True)
    df.loc[df["raceid"] == "nh", "race_type"] = "Handicap Hurdle"
    out = add_speed_near(df).set_index("horse_name")
    assert out.loc["s-0", "sf_speed_inside"] == 0 and out.loc["s-0", "sf_speed_outside"] == pytest.approx(1.2)
    assert out.loc["s-3", "sf_speed_inside"] == pytest.approx(0.7)             # stall 5: stalls 1-3 inside
    assert out.loc["s-3", "sf_speed_outside"] == pytest.approx(0.5)
    assert out.loc["s-3", "sf_speed_near"] == pytest.approx(0.0 + 0.5)       # stalls 3 and 6 (4, 7 empty)
    assert out.loc["s-5", "sf_speed_near"] == pytest.approx(0.5)             # stall 8: stall 6
    assert out.loc["s-1", "sf_speed_near"] == pytest.approx(0.1 + 0.0)       # stall 2: stalls 1, 3 (0, 4 empty)
    assert out.filter(like="nh-", axis=0)[["sf_speed_inside", "sf_speed_near"]].isna().all().all()


def test_the_per_run_readings_are_refused_as_inputs():
    for col in sorted(SHAPE_FORM_POST_RACE):
        with pytest.raises(ValueError, match="describe the race being predicted"):
            assert_no_post_race_features(["rNFP", col])
    assert not set(SHAPE_FORM_FEATURES) & SHAPE_FORM_POST_RACE


def test_the_block_needs_the_shape_block_first():
    with pytest.raises(ValueError, match="shape block"):
        add_shape_form(pd.DataFrame({"horse_name": ["x"], "race_date": [pd.Timestamp("2024-01-01")]}))


def test_the_market_miss_by_draw_reads_earlier_days_only():
    # a course where the lowest stalls win far more often than their price says
    rows = []
    for d, day in enumerate(pd.date_range("2023-01-01", periods=300, freq="D")):
        winner = 0 if d % 2 == 0 else 1                          # stall 1 or 2 wins every race
        for i in range(10):
            rows.append(dict(raceid=f"r{d}", race_date=day, race_time="2.30", track="chester",
                             horse_name=f"h{d}-{i}", race_type="Handicap", surface_type="Turf",
                             dist_furlongs=5.0, number_of_runners=10, stall=i + 1,
                             placing_numerical=1 if i == winner else 2 + i, bfsp=10.0,
                             p_lead=0.1, p_prom=0.2, p_mid=0.3, p_rear=0.4, shape_bin=1.0))
    df = pd.DataFrame(rows)
    from model.shape_form import add_market_miss_cells
    out = add_market_miss_cells(df.copy())
    last = out[out["raceid"] == "r299"].set_index("stall")
    assert last.loc[1, "sf_draw_ae"] > 0.05 and last.loc[10, "sf_draw_ae"] < -0.02
    assert out[out["raceid"] == "r0"]["sf_draw_ae"].abs().max() == 0       # nothing earlier to learn from
    # the day's own result never enters its cells
    alt = df.copy()
    alt.loc[alt["raceid"] == "r299", "placing_numerical"] = alt.loc[alt["raceid"] == "r299", "placing_numerical"][::-1].to_numpy()
    new = add_market_miss_cells(alt).set_index(["raceid", "stall"]).loc["r299"]
    np.testing.assert_array_equal(new["sf_draw_ae"].to_numpy(), last["sf_draw_ae"].to_numpy())
    np.testing.assert_array_equal(new["sf_pos_ae"].to_numpy(), last["sf_pos_ae"].to_numpy())
