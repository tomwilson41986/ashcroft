"""The committed model must be one we are willing to serve.

The artefact this guards against was real and was being served every morning.
`data/models/bfsp_model.lgb` carried 419 features whose top three by gain were
`NFP_residual`, `FSALB` and `win_surprise` -- the finishing position, the
beaten lengths, and `won` multiplied by the log of the price. All three are
post-race columns that the leak guard now forbids, and for a race that has not
been run they are NaN, so the live job was making predictions with its three
most important splits routed down a missing-value branch.

Nothing detected it. The metadata recorded a params dict and no recipe, its
schema was not even this trainer's (it came from `build_datasets.py`'s
single-split path), and `predict_bfsp_today.load_bfsp_model` loaded whatever
was on disk. These tests read the committed file and fail the build instead.
"""

import json
import os

import pytest

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "models")
META_PATH = os.path.join(MODEL_DIR, "bfsp_model_meta.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(META_PATH),
    reason="no committed BFSP model metadata to check",
)


def _meta_paths() -> list[str]:
    """The served model's metadata: the directory's own, and every member's where a
    manifest (bfsp_ensemble.json) serves several boosters as one."""
    paths = [META_PATH]
    manifest = os.path.join(MODEL_DIR, "bfsp_ensemble.json")
    if os.path.exists(manifest):
        with open(manifest) as f:
            for m in json.load(f).get("members", []):
                path = os.path.normpath(os.path.join(MODEL_DIR, m.get("dir", "."), "bfsp_model_meta.json"))
                if path not in paths:
                    paths.append(path)
    return paths


@pytest.fixture(scope="module", params=_meta_paths(), ids=lambda p: os.path.relpath(p, MODEL_DIR))
def meta(request):
    with open(request.param) as f:
        return json.load(f)


def test_every_member_the_manifest_names_is_committed():
    """A manifest naming a member that is not there fails at 06:00, not here, without this."""
    from predict_bfsp_today import booster_path

    for path in _meta_paths():
        assert os.path.exists(path), f"no metadata at {path}"
        assert os.path.exists(booster_path(os.path.dirname(path))), f"no booster beside {path}"


def test_the_committed_model_is_servable(meta):
    """The same check `load_bfsp_model` runs, at build time rather than 06:00."""
    from model.bfsp_model import assert_meta_is_servable

    assert_meta_is_servable(meta)


def test_no_post_race_column_is_a_feature(meta):
    """Named separately from the guard above so a failure says which columns."""
    from model import blocks
    from model.custom_metrics import POST_RACE_ONLY
    from model.draw_metrics import DRAW_POST_RACE_ONLY
    from model.pace_metrics import PACE_POST_RACE_ONLY
    from model.primitives import POST_RACE_PRIMITIVES

    banned = (set(POST_RACE_ONLY) | set(POST_RACE_PRIMITIVES)
              | set(PACE_POST_RACE_ONLY) | set(DRAW_POST_RACE_ONLY))
    for name in blocks.names():
        banned |= blocks.post_race(name)
    bad = sorted(set(meta.get("feature_cols", [])) & banned)
    assert not bad, f"the committed model was trained on the race's own result: {bad}"


def test_the_recipe_is_recorded(meta):
    """An artefact that does not say what it is cannot be told from the other one."""
    assert meta.get("objective") in ("l2", "profit_weighted")
    from model.bfsp_model import SERVABLE_TARGETS

    assert meta.get("target") in SERVABLE_TARGETS
    assert isinstance(meta.get("sample_weighting"), dict)
    assert meta["sample_weighting"].get("type") in ("none", "exponential_decay")
    assert meta.get("feature_code_hash"), "no feature_code_hash: provenance unknown"


def test_it_was_trained_by_this_trainer_on_the_current_feature_list(meta):
    """Catches an artefact left behind by a different script or an older list. A
    drop-in block's features count as known: the live path builds the blocks a model
    reads (`blocks.attach_as_trained`)."""
    from model import blocks
    from model.bfsp_features import ALL_FEATURE_COLS

    known = set(ALL_FEATURE_COLS)
    for name in blocks.names():
        known |= set(blocks.features(name))
    cols = meta.get("feature_cols", [])
    assert len(cols) == meta.get("n_features"), "n_features disagrees with feature_cols"
    unknown = sorted(set(cols) - known)
    assert not unknown, f"trained on columns the feature list no longer has: {unknown[:10]}"


def test_the_top_features_are_not_the_result(meta):
    """The specific failure: the result was the model's strongest signal."""
    top = [name for name, _ in meta.get("top_gain", [])]
    for banned in ("NFP_residual", "FSALB", "win_surprise"):
        assert banned not in top, f"{banned} is among the model's most important features"


def test_a_categorical_vocabulary_travels_with_the_model(meta):
    """Without it, track_cat means one thing in training and another on a card."""
    vocab = meta.get("categorical_vocab") or {}
    assert vocab, "no categorical_vocab: live encoding will not match training"
    assert "track" in vocab and len(vocab["track"]) > 1
