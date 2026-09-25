"""The research loop split into one job per fold (evaluate_oos --fold, scripts/research_loop.py).

A fold fitted in its own job must be the fold the serial walk fits: the same
rows trained, the same rows scored, the same index, the same predictions. And a
base run reused from the cache is only the base run if its key moves whenever
anything that decides it moves. Synthetic frames test the mechanics only;
nothing is trained or evaluated on them.
"""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import evaluate_oos
from model.bfsp_model import DEFAULT_PARAMS, QUICK_PARAMS, RECIPES, TrainConfig, parse_param_overrides
from scripts import research_loop as rl


def _frame(days=("2023-01-01", "2025-12-31"), per_day=4, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.date_range(*days, freq="D")
    n = len(d) * per_day
    x = rng.normal(size=(n, 3))
    return pd.DataFrame({
        "race_date": np.repeat(d, per_day), "race_time": "2.30", "track": "york",
        "horse_name": [f"h{i}" for i in range(n)],
        "raceid": np.repeat(np.arange(len(d)), per_day),
        "a": x[:, 0], "b": x[:, 1], "c": x[:, 2],
        "bfsp": np.exp(1.5 + 0.6 * x[:, 0] + 0.2 * rng.normal(size=n)).clip(1.1, 500),
    })


SCHEDULE = dict(min_train_days=365, val_window_days=91, step_days=91,
                eval_from="2024-06-01", eval_until="2025-10-01")


def _stub(monkeypatch):
    def fit(train_df, feature_cols, cfg):
        return SimpleNamespace(booster=None, best_iteration=1, early_stopped=True, n_holdout=0,
                               holdout_start=train_df["race_date"].max(), holdout_metrics={"mae": 0.0},
                               init_offset=0.0, n_train=len(train_df))

    def predict(model, df, feature_cols, **kw):
        df["predicted_bfsp"] = 5.0
        return df

    monkeypatch.setattr(evaluate_oos, "fit_bfsp", fit)
    monkeypatch.setattr(evaluate_oos, "predict_prices", predict)


def test_the_plan_is_the_folds_the_serial_walk_scores(monkeypatch):
    _stub(monkeypatch)
    df = _frame()
    serial = evaluate_oos.walk_forward_predict(df, ["a"], **SCHEDULE)
    plan = evaluate_oos.fold_plan(df["race_date"], cfg=TrainConfig(), **SCHEDULE)
    assert [p["fold"] for p in plan] == sorted(serial["fold_idx"].unique())
    for p in plan:
        rows = serial[serial["fold_idx"] == p["fold"]]
        assert len(rows) == p["n_val"]
        assert rows["race_date"].min() >= pd.Timestamp(p["val_start"])
        assert rows["race_date"].max() < min(pd.Timestamp(p["val_end"]), pd.Timestamp(SCHEDULE["eval_until"]))


def test_one_fold_on_its_own_scores_exactly_that_fold(monkeypatch):
    _stub(monkeypatch)
    df = _frame()
    serial = evaluate_oos.walk_forward_predict(df, ["a"], **SCHEDULE)
    for k in sorted(serial["fold_idx"].unique()):
        alone = evaluate_oos.walk_forward_predict(df, ["a"], only_fold=int(k), **SCHEDULE)
        assert set(alone["fold_idx"]) == {k}
        pd.testing.assert_frame_equal(
            alone.reset_index(drop=True),
            serial[serial["fold_idx"] == k].reset_index(drop=True))


def test_folds_fitted_apart_give_the_serial_runs_predictions():
    """Real boosters, tiny: the fold split and the determinism together."""
    df = _frame(per_day=6)
    cfg = TrainConfig(num_boost_round=25, holdout_days=30,
                      params={**DEFAULT_PARAMS, "num_leaves": 7, "min_child_samples": 10})
    serial = evaluate_oos.walk_forward_predict(df, ["a", "b", "c"], cfg=cfg, **SCHEDULE)
    parts = [evaluate_oos.walk_forward_predict(df, ["a", "b", "c"], cfg=cfg, only_fold=int(k), **SCHEDULE)
             for k in sorted(serial["fold_idx"].unique())]
    key = ["race_date", "horse_name"]
    a = serial.sort_values(key).reset_index(drop=True)
    b = pd.concat(parts).sort_values(key).reset_index(drop=True)
    np.testing.assert_array_equal(a["predicted_bfsp"].to_numpy(), b["predicted_bfsp"].to_numpy())


def test_the_recipes():
    assert RECIPES["served"] == ({}, TrainConfig.num_boost_round)
    params, rounds = RECIPES["quick"]
    assert params is QUICK_PARAMS and rounds < TrainConfig.num_boost_round
    assert params["learning_rate"] > DEFAULT_PARAMS["learning_rate"]
    # reproducible fits: the base can be reused from the cache
    assert DEFAULT_PARAMS["deterministic"] and DEFAULT_PARAMS["force_col_wise"]


CFG = {"tag": "t", "bfsp": True, "bfsp_from": "2025-07-01", "blocks": "form_windows",
       "bfsp_base_withhold": "", "bfsp_variant_withhold": "", "bfsp_variant_drop": "",
       "bfsp_variant_args": "--num-boost-round 6000"}


def test_the_arms_arguments():
    assert rl.arm_args(CFG, "base") == []
    assert rl.arm_args({**CFG, "bfsp_base_withhold": "intent"}, "base") == ["--withhold", "intent"]
    assert rl.arm_args(CFG, "base_rep") == rl.arm_args(CFG, "base")
    assert rl.arm_args(CFG, "blocks") == ["--blocks", "form_windows", "--num-boost-round", "6000"]
    v = rl.arm_args({**CFG, "bfsp_variant_drop": "fw_ae", "bfsp_variant_withhold": "intent"}, "blocks")
    assert v[:6] == ["--blocks", "form_windows", "--drop-features", "fw_ae", "--withhold", "intent"]
    assert "--recipe" in rl.common_args(CFG) and rl.common_args(CFG)[-1] == "served"
    assert rl.common_args({**CFG, "recipe": "quick"})[-1] == "quick"
    with pytest.raises(SystemExit):
        rl.recipe({"recipe": "fast"})


def test_the_base_key_moves_with_what_decides_the_base_and_not_with_the_variant():
    folds = [{"fold": 0, "val_start": "2025-09-27", "val_end": "2025-12-27", "n_train": 1, "n_val": 1}]
    k = rl.base_key(CFG, "fkey", folds)
    assert k == rl.base_key(dict(CFG), "fkey", json.loads(json.dumps(folds)))       # stable
    assert k == rl.base_key({**CFG, "blocks": "shape_form", "bfsp_variant_args": ""}, "fkey", folds)
    assert k != rl.base_key(CFG, "fkey2", folds)                                     # the matrix
    assert k != rl.base_key({**CFG, "bfsp_base_withhold": "intent"}, "fkey", folds)  # the base's features
    assert k != rl.base_key({**CFG, "recipe": "quick"}, "fkey", folds)               # the recipe
    assert k != rl.base_key({**CFG, "bfsp_from": "2025-01-01"}, "fkey", folds)      # the window
    assert k != rl.base_key(CFG, "fkey", [dict(folds[0], n_val=2)])                  # the fold plan


def test_every_arm_or_only_the_base_can_take_further_arguments():
    cfg = {**CFG, "bfsp_args": "--num-boost-round 6000"}
    assert rl.arm_args(cfg, "base") == ["--num-boost-round", "6000"]
    assert rl.arm_args({**cfg, "bfsp_base_args": "--learning-rate 0.02"}, "base")[-2:] == ["--learning-rate", "0.02"]
    assert rl.arm_args({**cfg, "bfsp_base_args": "--learning-rate 0.02"}, "blocks").count("--learning-rate") == 0
    v = rl.arm_args({**cfg, "variants": [{"name": "fw", "blocks": "form_windows"}]}, "fw")
    assert v == ["--blocks", "form_windows", "--num-boost-round", "6000"]
    folds = [{"fold": 0}]
    assert rl.base_key(cfg, "fkey", folds) != rl.base_key(CFG, "fkey", folds)


def test_the_base_key_follows_the_source_of_a_drop_in_block_the_base_carries(tmp_path, monkeypatch):
    from model import feature_cache
    assert rl.block_code_hash("") == rl.block_code_hash("form_windows,shape_form") == ""   # engine blocks: in the matrix key
    assert rl.block_code_hash("form_variants") != ""
    assert rl.block_code_hash("form_variants") != rl.block_code_hash("race_relative")
    # a block's imports are walked: pace_v2 reads form_variants' ladder and the shrinkage module
    files = {p.name for p in feature_cache.feature_source_files(["model/blocks/__init__.py", "model/blocks/pace_v2.py"])}
    assert {"form_variants.py", "shrinkage.py", "__init__.py", "pace_v2.py"} <= files

    src = tmp_path / "block.py"
    src.write_text("FEATURES = ['x_a']\n")
    monkeypatch.setattr(feature_cache, "feature_source_files", lambda entry_points: [src])
    folds = [{"fold": 0}]
    cfg = {**CFG, "bfsp_base_blocks": "form_variants"}
    before = rl.base_key(cfg, "fkey", folds)
    src.write_text("FEATURES = ['x_a']  # edited\n")
    assert rl.base_key(cfg, "fkey", folds) != before          # the block changed: the cached base is stale
    assert rl.base_key(CFG, "fkey", folds) == rl.base_key(CFG, "fkey", folds)


def _fold_output(root, arm, k, n):
    d = root / f"fit-{arm}-{k}" / "reports" / "fit"
    d.mkdir(parents=True)
    pd.DataFrame({"race_date": ["2025-10-01"] * n, "race_time": ["2.30"] * n, "track": "york",
                  "horse_name": [f"{arm}{k}-{i}" for i in range(n)], "predicted_bfsp": 4.0,
                  "fold_idx": k}).to_csv(d / f"oos_{arm}_f{k}.csv", index=False)
    (d / f"summary_{arm}_f{k}.json").write_text(json.dumps(
        {"fold": k, "fold_fits": [{"best_iteration": 10 + k}], "recipe": "quick",
         "elapsed_seconds": 60.0 + k, "cpu": "cpu-x", "n_features_used": 535}))


def test_gather_joins_an_arms_folds_in_order_and_checks_the_plan(tmp_path, monkeypatch):
    src, out = tmp_path / "fits", tmp_path / "out"
    for k in (2, 0, 1):
        _fold_output(src, "base", k, 3)
        _fold_output(src, "base_rep", k, 3)                      # must not be read as "base"
    assert rl.gather(str(src), str(out), "base", expect=[0, 1, 2])
    df = pd.read_csv(out / "oos_base.csv")
    assert len(df) == 9 and list(df["fold_idx"]) == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    s = json.loads((out / "summary_base.json").read_text())
    assert s["folds"] == [0, 1, 2] and [f["best_iteration"] for f in s["fold_fits"]] == [10, 11, 12]
    assert s["cpu"] == ["cpu-x"] and s["n_predictions"] == 9
    assert not rl.gather(str(src), str(out), "blocks")           # nothing here: served from the cache
    with pytest.raises(SystemExit, match="planned"):
        rl.gather(str(src), str(out), "base", expect=[0, 1, 2, 3])


def test_the_replicate_report_reads_identical_as_identical(tmp_path):
    a = pd.DataFrame({"race_date": ["2025-10-01"] * 4, "race_time": "2.30", "track": "york",
                      "horse_name": list("wxyz"), "predicted_bfsp": [2.0, 4.0, 8.0, 16.0]})
    a.to_csv(tmp_path / "a.csv", index=False)
    a.to_csv(tmp_path / "b.csv", index=False)
    assert "identical predictions on 100.0%" in rl.replicate_report(str(tmp_path / "a.csv"), str(tmp_path / "b.csv"))
    a.assign(predicted_bfsp=a["predicted_bfsp"] * 1.01).to_csv(tmp_path / "c.csv", index=False)
    assert "identical predictions on 0.0%" in rl.replicate_report(str(tmp_path / "a.csv"), str(tmp_path / "c.csv"))


def test_several_variants_against_one_base():
    cfg = {**CFG, "bfsp_base_blocks": "form_windows", "bfsp_base_withhold": "shape_draw", "replicate_base": True,
           "variants": [{"name": "fv", "blocks": "form_variants"},
                        {"name": "fv_nolb", "blocks": "form_variants", "drop": "fv_lb"},
                        {"name": "r6000", "args": "--num-boost-round 6000", "withhold": "intent"}]}
    assert rl.all_arms(cfg) == ["base", "fv", "fv_nolb", "r6000", "base_rep"]
    assert rl.arm_args(cfg, "base") == ["--blocks", "form_windows", "--withhold", "shape_draw"]
    # a variant is the base plus its own blocks, less its own withheld features
    assert rl.arm_args(cfg, "fv") == ["--blocks", "form_windows,form_variants", "--withhold", "shape_draw"]
    assert rl.arm_args(cfg, "fv_nolb")[:4] == ["--blocks", "form_windows,form_variants", "--drop-features", "fv_lb"]
    assert rl.arm_args(cfg, "r6000") == ["--blocks", "form_windows", "--withhold", "shape_draw,intent",
                                         "--num-boost-round", "6000"]
    # the base key sees the base's blocks, not the variants
    folds = [{"fold": 0}]
    k = rl.base_key(cfg, "f", folds)
    assert k == rl.base_key({**cfg, "variants": [{"name": "x", "blocks": "shape_form"}]}, "f", folds)
    assert k != rl.base_key({**cfg, "bfsp_base_blocks": ""}, "f", folds)
    for bad in ("base", "Base", "a b", "", "base_rep"):
        with pytest.raises(SystemExit):
            rl.variants({**CFG, "variants": [{"name": bad}]})
    with pytest.raises(SystemExit):
        rl.variants({**CFG, "variants": [{"name": "a"}, {"name": "a"}]})


def test_the_summary_table_reads_the_decision():
    row = {"delta": -0.0066, "delta_ci": [-0.0082, -0.0050], "rank1_delta": -0.0104, "brier_skill": 0.00222,
           "brier_skill_ci": [0.00063, 0.0039], "concordance": 0.00146, "replaces": True}
    t = rl.summary_table([("fv", row), ("x", {**row, "replaces": False, "rank1_delta": None})])
    assert "| fv | -0.0066 (-0.0082 to -0.0050) | -0.0104 |" in t and "**replaces the base**" in t
    assert "| x |" in t and "base stands" in t


def test_a_recipe_iteration_can_move_any_setting_and_only_real_ones():
    assert parse_param_overrides(None) == {}
    got = parse_param_overrides(["num_leaves=255", "min_child_samples=20", "feature_fraction=0.7",
                                 "extra_trees=true"])
    assert got == {"num_leaves": 255, "min_child_samples": 20, "feature_fraction": 0.7, "extra_trees": True}
    assert isinstance(got["num_leaves"], int) and isinstance(got["feature_fraction"], float)
    for bad in (["num_leaves"], ["=3"], ["num_leaves="], ["num_leafs=255"]):
        with pytest.raises(ValueError):
            parse_param_overrides(bad)
    # the override reaches the fit's parameters, on top of the recipe
    cfg = TrainConfig(params={**DEFAULT_PARAMS, **QUICK_PARAMS, **parse_param_overrides(["num_leaves=31"])})
    assert cfg.params["num_leaves"] == 31 and cfg.params["learning_rate"] == QUICK_PARAMS["learning_rate"]
