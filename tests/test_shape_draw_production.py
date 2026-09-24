"""The blocks the metrics engine builds: shape and draw, intent, freshness.

The blocks themselves are tested in tests/test_race_shape.py,
tests/test_intent_features.py and tests/test_freshness_features.py. These tests
pin their place in production:
- the engine builds all three on every frame, and each can be left out;
- the served feature list carries card-safe intent and freshness, not shape and
  draw (built and measured, but not served: iteration 25);
- a run's own style and margins, and the four intent features a 06:00 card
  cannot know, are refused as inputs;
- the live path builds only what the model it serves reads;
- a 06:00 card priced through the whole engine gets exactly the values the same
  day gets once its results are in.

The frames are synthetic and test mechanics only.
"""
import numpy as np
import pandas as pd
import pytest

from model.bfsp_features import (
    ALL_FEATURE_COLS, INTENT_CARD_UNSAFE, INTENT_SERVED_FEATURES, PRODUCTION_BLOCKS,
    SERVED_FRESHNESS_FEATURES, SHAPE_DRAW_FEATURES, assert_no_post_race_features, blocks_needed,
    needs_race_shape,
)
from model.custom_metrics import CustomMetricsEngine
from model.draw_curve import DRAW_CURVE_FEATURES, DRAW_STYLE_FEATURES
from model.freshness_features import FRESHNESS_FEATURES
from model.intent_features import INTENT_FEATURES
from model.race_shape import RACE_SHAPE_FEATURES, RACE_SHAPE_POST_RACE
from tests.test_leakage import _full_card

KEY = ["raceid", "horse_name"]
ENGINE_BLOCKS = SHAPE_DRAW_FEATURES + INTENT_FEATURES + FRESHNESS_FEATURES


def test_the_served_feature_list():
    assert SHAPE_DRAW_FEATURES == RACE_SHAPE_FEATURES + DRAW_CURVE_FEATURES + DRAW_STYLE_FEATURES
    assert set(INTENT_SERVED_FEATURES) | set(INTENT_CARD_UNSAFE) == set(INTENT_FEATURES)
    assert set(INTENT_SERVED_FEATURES) <= set(ALL_FEATURE_COLS)
    assert set(SERVED_FRESHNESS_FEATURES) == set(FRESHNESS_FEATURES) <= set(ALL_FEATURE_COLS)
    assert not set(SHAPE_DRAW_FEATURES) & set(ALL_FEATURE_COLS)      # built, not served
    assert not set(INTENT_CARD_UNSAFE) & set(ALL_FEATURE_COLS)
    assert len(ALL_FEATURE_COLS) == len(set(ALL_FEATURE_COLS)) == 501 + 20 + 14
    assert not set(ALL_FEATURE_COLS) & RACE_SHAPE_POST_RACE
    assert set(PRODUCTION_BLOCKS) == {"shape_draw", "intent", "freshness"}


@pytest.mark.parametrize("col", sorted(RACE_SHAPE_POST_RACE))
def test_a_runs_own_style_and_margins_are_refused_as_inputs(col):
    with pytest.raises(ValueError, match="describe the race being predicted"):
        assert_no_post_race_features(list(ALL_FEATURE_COLS) + [col])


@pytest.mark.parametrize("col", INTENT_CARD_UNSAFE)
def test_what_the_card_cannot_know_is_refused_as_an_input(col):
    with pytest.raises(ValueError, match="06:00 card"):
        assert_no_post_race_features(list(ALL_FEATURE_COLS) + [col])


def test_the_win_probability_model_reads_the_served_blocks_too():
    from model.prerace_builder import PreRaceBuilder
    cols = PreRaceBuilder.get_feature_columns(PreRaceBuilder.__new__(PreRaceBuilder))
    assert set(INTENT_SERVED_FEATURES) | set(SERVED_FRESHNESS_FEATURES) <= set(cols)
    assert not set(INTENT_CARD_UNSAFE) & set(cols)


def test_the_live_path_builds_only_what_the_model_reads():
    assert blocks_needed(ALL_FEATURE_COLS) == {"race_shape": False, "intent": True, "freshness": True}
    served_before = [c for c in ALL_FEATURE_COLS if c not in set(INTENT_FEATURES) | set(FRESHNESS_FEATURES)]
    assert blocks_needed(served_before) == {"race_shape": False, "intent": False, "freshness": False}
    assert needs_race_shape(["rNFP", "dc_edge_lbs"])
    assert blocks_needed(["fr_runs_30d"]) == {"race_shape": False, "intent": False, "freshness": True}


def test_the_engine_builds_the_blocks_and_can_leave_them_out():
    built = CustomMetricsEngine().calculate_all(_full_card())
    assert set(ENGINE_BLOCKS) <= set(built.columns)
    last = built[built["race_date"] == built["race_date"].max()]
    # every runner has earlier runs to project a style from, and every race a stall draw
    assert last[SHAPE_DRAW_FEATURES].notna().all().all()
    np.testing.assert_allclose(last[["p_lead", "p_prom", "p_mid", "p_rear"]].sum(axis=1), 1.0)
    assert last[["it_seen_runs", "it_runs_for_yard", "fr_runs_30d", "fr_usual_gap"]].notna().all().all()

    plain = CustomMetricsEngine(race_shape=False, intent=False, freshness=False).calculate_all(_full_card())
    assert not set(ENGINE_BLOCKS) & set(plain.columns)
    assert not RACE_SHAPE_POST_RACE & set(plain.columns)
    # and nothing else the engine builds depends on them
    shared = [c for c in plain.columns if c in built.columns]
    assert len(shared) == len(plain.columns)
    pd.testing.assert_frame_equal(plain[shared], built[shared])


def test_the_0600_card_gets_the_values_its_day_gets_with_results_in():
    """The live path: history plus a card with no results, no margins, no
    comments and no prices, through the whole engine. The card day's blocks
    must be bit-identical to the same day built with its results known."""
    full = _full_card()
    day = full["race_date"].max()
    blank = ["placing_numerical", "place", "total_dst_bt", "distbt", "comment", "comptime",
             "comptime_numeric", "bfsp", "bfsp_place"]            # `won` is rebuilt from the placing
    card = full.copy()
    card.loc[card["race_date"] == day, blank] = np.nan
    control = full.copy()                  # the same blanking a day earlier
    control.loc[control["race_date"] == sorted(full["race_date"].unique())[-2], blank] = np.nan

    def blocks(frame):
        out = CustomMetricsEngine().calculate_all(frame)
        return out[out["race_date"] == day].sort_values(KEY).set_index(KEY)[ENGINE_BLOCKS]

    a, b, c = blocks(full), blocks(card), blocks(control)
    assert len(a) == 16
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    # the comparison can see a change: yesterday's results do reach today
    moved = [f for f in ENGINE_BLOCKS if not np.array_equal(a[f].to_numpy(float), c[f].to_numpy(float),
                                                            equal_nan=True)]
    assert len(moved) > 10, moved
    # freshness reads yesterday's places; intent's results-driven features are its trainer x
    # angle records, which this fixture never triggers (tests/test_intent_features.py has the
    # control for those)
    assert any(f.startswith("fr_") for f in moved), moved
