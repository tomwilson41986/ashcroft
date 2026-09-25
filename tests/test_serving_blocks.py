"""Serving a drop-in block: the computation it was evaluated and trained on, not a port of it.

A model names its blocks by its features; the live path switches on the engine
blocks they read and builds them on the rows the training matrix holds (runs
with a usable price) plus the card's. The card must get exactly what training
gave the same runners, and building on every run instead must not (else the
row rule would be untested). Synthetic frames test the mechanics only; nothing
is trained for use on them.
"""
import numpy as np
import pandas as pd
import pytest

import predict_bfsp_today
import train_bfsp
from model import blocks
from model.custom_metrics import CustomMetricsEngine
from tests.test_feature_blocks import FLIP_DAY, KEY, _blanked, _history

SERVED = ["form_variants", "race_relative"]


def test_a_model_names_its_blocks_by_its_features():
    cols = ["LR_ORR2", "rr_LR_ORR2_z", "fv_nfp_e3", "fw_nfp_w5"]
    assert blocks.used_by(cols) == ["form_variants", "race_relative"]
    assert blocks.used_by(["LR_ORR2", "fw_nfp_w5"]) == []
    assert blocks.engine_flags(["race_relative"]) == {"form_windows": True, "freshness": True}
    assert blocks.engine_flags(["form_variants"]) == {}
    assert blocks.build_order(["race_relative", "form_variants", "race_relative"]) == ["race_relative", "form_variants"]


@pytest.fixture(scope="module")
def frames():
    hist = _history()
    rng = np.random.default_rng(3)
    no_price = ((hist["race_date"] < FLIP_DAY) & (rng.random(len(hist)) < 0.1)).to_numpy()
    hist.loc[no_price, "bfsp"] = np.nan                   # runs the matrix does not hold
    trained = CustomMetricsEngine().calculate_all(hist.copy())
    live = CustomMetricsEngine().calculate_all(_blanked(hist))     # the 06:00 card: no results, no prices
    return hist, trained, live


def _day(frame, cols):
    return frame[frame["race_date"] == FLIP_DAY].set_index(KEY)[cols].sort_index()


def test_the_card_gets_what_training_gave_the_same_runners(frames):
    _, trained, live = frames
    matrix = trained[blocks.matrix_rows(trained)].reset_index(drop=True)      # evaluate_oos.build_feature_frame
    matrix, cols = blocks.attach(matrix, SERVED)
    card = (live["race_date"] == FLIP_DAY).to_numpy()
    served, cols2 = blocks.attach_as_trained(live.copy(), SERVED, card=card)
    assert cols2 == cols
    want, got = _day(matrix, cols), _day(served, cols)
    assert len(got) == len(want) > 10
    pd.testing.assert_frame_equal(got, want, check_dtype=False)
    # every other row the matrix lacks gets nothing
    lacking = ~blocks.matrix_rows(live) & ~card
    assert lacking.sum() > 0 and served.loc[lacking, cols].isna().all().all()

    # and the row rule matters: the same blocks on every run price some card rows differently
    naive, _ = blocks.attach(live.copy(), SERVED)
    diff = (_day(naive, cols) - want).abs().to_numpy()
    assert np.nanmax(diff) > 1e-9


def test_a_block_that_reads_a_column_the_frame_lacks_is_refused(frames):
    _, trained, _ = frames
    with pytest.raises(ValueError, match="reads .*LB"):
        blocks.attach_as_trained(trained.drop(columns=["LB"]), ["form_variants"])


def test_the_live_path_builds_what_the_model_reads(frames, monkeypatch):
    """prepare_and_predict on a day in the database: the engine gets the flags the
    blocks need, and the prices are asked of the features training gave that day."""
    hist, trained, _ = frames
    feature_cols = ["LR_ORR2", "fv_nfp_e3", "fv_lb_w5", "rr_LR_ORR2_z", "rr_fw_nfp_w5_gap"]
    flags_seen, seen = [], {}
    real_init = CustomMetricsEngine.__init__

    def spy_init(self, *a, **kw):
        flags_seen.append(kw)
        real_init(self, *a, **kw)

    def capture(model, df, cols, **kw):
        seen["df"] = df.copy()
        df["predicted_bfsp"] = 5.0
        return df

    monkeypatch.setattr(CustomMetricsEngine, "__init__", spy_init)
    monkeypatch.setattr(predict_bfsp_today, "predict_prices", capture)
    hist = hist.copy()
    predict_bfsp_today.prepare_and_predict(hist, hist.iloc[:0], model=None, feature_cols=feature_cols,
                                           target_date=FLIP_DAY.date())
    assert flags_seen and flags_seen[-1].get("form_windows") and flags_seen[-1].get("freshness")
    matrix = trained[blocks.matrix_rows(trained)].reset_index(drop=True)
    matrix, _ = blocks.attach(matrix, SERVED)
    block_cols = [c for c in feature_cols if c in blocks.features("form_variants") + blocks.features("race_relative")]
    want = _day(matrix, block_cols)
    got = seen["df"].set_index(KEY)[block_cols].sort_index()
    pd.testing.assert_frame_equal(got, want, check_dtype=False)


def test_training_takes_engine_and_drop_in_blocks_from_the_matrix(frames):
    _, trained, _ = frames
    matrix = trained[blocks.matrix_rows(trained)].reset_index(drop=True)
    out, cols = train_bfsp.attach_training_blocks(matrix.copy(), "shape_form,form_variants,race_relative")
    from model.bfsp_features import SHAPE_FORM_FEATURES
    assert cols[:len(SHAPE_FORM_FEATURES)] == list(SHAPE_FORM_FEATURES)
    assert cols[len(SHAPE_FORM_FEATURES):] == blocks.features("form_variants") + blocks.features("race_relative")
    ref, _ = blocks.attach(matrix.copy(), SERVED)
    pd.testing.assert_frame_equal(out[cols].reset_index(drop=True), ref[cols].reset_index(drop=True),
                                  check_dtype=False)
    with pytest.raises(SystemExit, match="unknown block"):
        train_bfsp.attach_training_blocks(matrix.copy(), "no_such_block")
    with pytest.raises(SystemExit, match="is served"):
        train_bfsp.attach_training_blocks(matrix.copy(), "form_windows")
