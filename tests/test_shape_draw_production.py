"""Run style, race shape, position value and draw curves as production features.

The blocks themselves (model/race_shape.py, model/draw_curve.py) are tested in
tests/test_race_shape.py. These tests pin their promotion: the metrics engine
builds them on every frame, the served feature list and the win-probability
model's list include them, their per-run columns are refused as inputs, and a
06:00 card priced through the whole engine gets exactly the values the same day
gets once its results are in. The frames are synthetic and test mechanics only.
"""
import numpy as np
import pandas as pd
import pytest

from model.bfsp_features import (
    ALL_FEATURE_COLS, SHAPE_DRAW_FEATURES, assert_no_post_race_features, needs_race_shape,
)
from model.custom_metrics import CustomMetricsEngine
from model.draw_curve import DRAW_CURVE_FEATURES, DRAW_STYLE_FEATURES
from model.race_shape import RACE_SHAPE_FEATURES, RACE_SHAPE_POST_RACE
from tests.test_leakage import _full_card

KEY = ["raceid", "horse_name"]


def test_the_served_feature_list_carries_the_block():
    assert SHAPE_DRAW_FEATURES == RACE_SHAPE_FEATURES + DRAW_CURVE_FEATURES + DRAW_STYLE_FEATURES
    assert set(SHAPE_DRAW_FEATURES) <= set(ALL_FEATURE_COLS)
    assert len(ALL_FEATURE_COLS) == len(set(ALL_FEATURE_COLS))
    assert not set(ALL_FEATURE_COLS) & RACE_SHAPE_POST_RACE


@pytest.mark.parametrize("col", sorted(RACE_SHAPE_POST_RACE))
def test_a_runs_own_style_and_margins_are_refused_as_inputs(col):
    with pytest.raises(ValueError, match="describe the race being predicted"):
        assert_no_post_race_features(list(ALL_FEATURE_COLS) + [col])


def test_the_win_probability_model_reads_the_block_too():
    from model.prerace_builder import PreRaceBuilder
    cols = PreRaceBuilder.get_feature_columns(PreRaceBuilder.__new__(PreRaceBuilder))
    assert set(SHAPE_DRAW_FEATURES) <= set(cols)


def test_only_a_model_that_reads_the_block_needs_it_built():
    assert needs_race_shape(ALL_FEATURE_COLS)
    assert needs_race_shape(["rNFP", "dc_edge_lbs"])
    assert not needs_race_shape([c for c in ALL_FEATURE_COLS if c not in set(SHAPE_DRAW_FEATURES)])


def test_the_engine_builds_the_block_and_can_leave_it_out():
    built = CustomMetricsEngine().calculate_all(_full_card())
    assert set(SHAPE_DRAW_FEATURES) <= set(built.columns)
    last = built[built["race_date"] == built["race_date"].max()]
    # every runner has earlier runs to project a style from, and every race a stall draw
    assert last[SHAPE_DRAW_FEATURES].notna().all().all()
    np.testing.assert_allclose(last[["p_lead", "p_prom", "p_mid", "p_rear"]].sum(axis=1), 1.0)

    plain = CustomMetricsEngine(race_shape=False).calculate_all(_full_card())
    assert not set(SHAPE_DRAW_FEATURES) & set(plain.columns)
    assert not RACE_SHAPE_POST_RACE & set(plain.columns)
    # and nothing else the engine builds depends on it
    shared = [c for c in plain.columns if c in built.columns]
    assert len(shared) == len(plain.columns)
    pd.testing.assert_frame_equal(plain[shared], built[shared])


def test_the_0600_card_gets_the_values_its_day_gets_with_results_in():
    """The live path: history plus a card with no results, no margins, no
    comments and no prices, through the whole engine. The card day's block
    must be bit-identical to the same day built with its results known."""
    full = _full_card()
    day = full["race_date"].max()
    blank = ["placing_numerical", "place", "total_dst_bt", "distbt", "comment", "comptime",
             "comptime_numeric", "bfsp", "bfsp_place"]            # `won` is rebuilt from the placing
    card = full.copy()
    card.loc[card["race_date"] == day, blank] = np.nan
    control = full.copy()                  # the same blanking a day earlier
    control.loc[control["race_date"] == sorted(full["race_date"].unique())[-2], blank] = np.nan

    def block(frame):
        out = CustomMetricsEngine().calculate_all(frame)
        return out[out["race_date"] == day].sort_values(KEY).set_index(KEY)[SHAPE_DRAW_FEATURES]

    a, b, c = block(full), block(card), block(control)
    assert len(a) == 16
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    # the comparison can see a change: yesterday's results do reach today
    moved = [f for f in SHAPE_DRAW_FEATURES if not np.array_equal(a[f].to_numpy(float), c[f].to_numpy(float),
                                                                  equal_nan=True)]
    assert len(moved) > 10, moved
