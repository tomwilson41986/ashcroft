"""Several boosters served as one (predict_bfsp_today.AveragedBooster, a manifest in the model directory).

The research loop scores an average of fitted arms as the geometric mean of their
normalised prices, renormalised per race (scripts/research_loop.average_arms). The
serving path must give the same prices from the members' boosters, or the numbers
that justified serving the average would describe something else. Synthetic frames
test mechanics only; nothing is trained or evaluated on them.
"""
import json
import lzma
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

import predict_bfsp_today as pbt
from model.bfsp_model import predict_prices

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import research_loop as rl  # noqa: E402

KEY = ["race_date", "race_time", "track", "horse_name"]
VOCAB = {"track_cat": ["Ascot", "York"]}


def _frame(n_races=40, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        for h in range(int(rng.integers(5, 10))):
            rows.append({"race_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=r // 4),
                         "race_time": f"{1 + r % 4}:15", "track": "York", "horse_name": f"h{r}_{h}",
                         "f_a": rng.normal(), "f_b": rng.normal(), "f_c": rng.normal(),
                         "bfsp": float(np.exp(rng.normal(2.0, 0.8))) + 1.0})
    d = pd.DataFrame(rows)
    d["raceid"] = d["race_date"].astype(str) + "|" + d["track"] + "|" + d["race_time"]
    return d


def _member(root: Path, name: str, cols, objective: str, seed: int, offset=0.0, xz=False, vocab=VOCAB,
            target="log_bfsp") -> Path:
    d = _frame(seed=seed)
    y = np.log(d["bfsp"]).to_numpy() + 0.3 * d[cols[0]].to_numpy()
    booster = lgb.train({"objective": objective, "num_leaves": 7, "learning_rate": 0.1, "verbose": -1,
                         "seed": seed, "deterministic": True},
                        lgb.Dataset(d[cols].astype(float), y - offset), num_boost_round=30)
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    text = booster.model_to_string()
    if xz:
        with lzma.open(out / "bfsp_model.lgb.xz", "wt") as f:
            f.write(text)
    else:
        (out / "bfsp_model.lgb").write_text(text)
    (out / "bfsp_model_meta.json").write_text(json.dumps({
        "feature_cols": list(cols), "objective": objective, "target": target, "init_offset": offset,
        "categorical_vocab": vocab, "trained_through": "2026-09-22"}))
    return out


def _manifest(root: Path, members) -> None:
    (root / pbt.ENSEMBLE_MANIFEST).write_text(json.dumps({"members": [{"name": n, "dir": d} for n, d in members]}))


@pytest.fixture
def averaged(tmp_path):
    _member(tmp_path, ".", ["f_a", "f_b"], "regression", seed=3, offset=0.25)
    _member(tmp_path, "members/huber", ["f_b", "f_c", "f_a"], "huber", seed=4, xz=True)
    _manifest(tmp_path, [("l2", "."), ("huber", "members/huber")])
    return tmp_path


def test_the_served_average_is_the_loops_average_of_the_members(averaged, tmp_path):
    model, cols, vocab = pbt.load_bfsp_model(str(averaged))
    assert isinstance(model, pbt.AveragedBooster) and vocab == VOCAB
    assert cols == ["f_a", "f_b", "f_c"]                      # every member's features, first seen first
    frame = _frame(seed=9)
    served = predict_prices(model, frame, cols, race_col="raceid")
    # each member priced on its own, then averaged as the research loop averages arms
    paths = []
    for name, booster, mcols in model.members:
        p = predict_prices(booster, frame, mcols, race_col="raceid")
        path = tmp_path / f"oos_{name}.csv"
        p[KEY + ["predicted_bfsp", "bfsp", "predicted_win_prob_norm"]].to_csv(path, index=False)
        paths.append(str(path))
    rl.average_arms(paths, str(tmp_path / "oos_avg.csv"))
    avg = pd.read_csv(tmp_path / "oos_avg.csv", dtype={"race_time": str})
    got = served.assign(race_date=served["race_date"].astype(str))[KEY + ["predicted_bfsp"]]
    want = avg.assign(race_date=pd.to_datetime(avg["race_date"]).astype(str))[KEY + ["predicted_bfsp"]]
    m = got.merge(want, on=KEY, suffixes=("_served", "_loop"))
    assert len(m) == len(frame)
    np.testing.assert_allclose(m["predicted_bfsp_served"], m["predicted_bfsp_loop"], rtol=1e-9)
    book = served.groupby("raceid")["predicted_bfsp"].apply(lambda s: (1 / s).sum())
    np.testing.assert_allclose(book, 1.0, rtol=1e-12)


def test_each_members_offset_counts_and_the_output_is_the_mean_log_price(averaged):
    model, cols, _ = pbt.load_bfsp_model(str(averaged))
    X = _frame(seed=5)[cols].astype(float)
    # both fixture members are fitted on log prices: the mean of their offset outputs
    means = np.mean([b.predict(X[c]) + getattr(b, "serving_offset", 0.0) for _, b, c in model.members], axis=0)
    np.testing.assert_allclose(model.predict(X), means, rtol=0, atol=1e-12)
    assert model.members[0][1].serving_offset == 0.25 and model.serving_offset == 0.0
    assert model.serving_target == "log_bfsp"


def test_without_a_manifest_one_booster_serves_as_before(tmp_path):
    _member(tmp_path, ".", ["f_a", "f_b"], "regression", seed=3)
    model, cols, vocab = pbt.load_bfsp_model(str(tmp_path))
    assert isinstance(model, lgb.Booster) and cols == ["f_a", "f_b"] and vocab == VOCAB


def test_members_that_number_categories_differently_are_refused(tmp_path):
    _member(tmp_path, ".", ["f_a", "f_b"], "regression", seed=3)
    _member(tmp_path, "b", ["f_a", "f_b"], "huber", seed=4, vocab={"track_cat": ["York", "Ascot"]})
    _manifest(tmp_path, [("a", "."), ("b", "b")])
    with pytest.raises(ValueError, match="vocabulary"):
        pbt.load_bfsp_model(str(tmp_path))


@pytest.mark.parametrize("targets", [("logit_norm_prob", "logit_norm_prob"), ("logit_norm_prob", "log_bfsp"),
                                     ("logit_norm_prob", "demeaned_log"), ("logit_norm_prob", "race_xent")])
def test_members_on_the_served_target_average_as_the_loop_averages(tmp_path, targets):
    """The served models are fitted on the logit of the race-normalised probability: each
    member's output is priced by its own target's rule before the prices are averaged."""
    _member(tmp_path, ".", ["f_a", "f_b"], "regression", seed=3, target=targets[0])
    _member(tmp_path, "b", ["f_c", "f_a"], "huber", seed=4, target=targets[1])
    _manifest(tmp_path, [("a", "."), ("b", "b")])
    model, cols, _ = pbt.load_bfsp_model(str(tmp_path))
    frame = _frame(seed=9)
    served = predict_prices(model, frame, cols, race_col="raceid")
    logs = [np.log(predict_prices(b, frame, c, race_col="raceid")["predicted_bfsp"].to_numpy())
            for _, b, c in model.members]
    p = np.exp(-np.mean(logs, axis=0))
    want = p / pd.Series(p).groupby(frame["raceid"].to_numpy()).transform("sum").to_numpy()
    np.testing.assert_allclose(served["predicted_win_prob_norm"].to_numpy(), want, rtol=1e-9)


def test_a_member_whose_booster_disagrees_with_its_metadata_is_refused(tmp_path):
    _member(tmp_path, ".", ["f_a", "f_b"], "regression", seed=3)
    bad = _member(tmp_path, "b", ["f_a", "f_b"], "huber", seed=4)
    meta = json.loads((bad / "bfsp_model_meta.json").read_text())
    (bad / "bfsp_model_meta.json").write_text(json.dumps({**meta, "feature_cols": ["f_a", "f_b", "f_c"]}))
    _manifest(tmp_path, [("a", "."), ("b", "b")])
    with pytest.raises(ValueError, match="booster reads"):
        pbt.load_bfsp_model(str(tmp_path))


def test_verify_reads_an_averaged_model_and_its_checks_pass(averaged):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import verify_model as vm
    model, meta = vm._load(str(averaged))
    assert isinstance(model, pbt.AveragedBooster)
    assert meta["members"] == ["l2", "huber"] and meta["feature_cols"] == ["f_a", "f_b", "f_c"]
    assert vm.check_servable(model, meta) == []
    problems, notes = vm.check_prices(model, meta, None, None, _frame(seed=6))
    assert problems == [], problems


STEP = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0]


def _gated(root: Path, weights=STEP) -> Path:
    _member(root, ".", ["f_a", "f_b"], "huber", seed=3, target="logit_norm_prob")
    _member(root, "members/twin", ["f_b", "f_c"], "regression", seed=4, target="demeaned_log")
    _member(root, "members/rx", ["f_c", "f_a"], "regression", seed=5, target="race_xent")
    (root / pbt.ENSEMBLE_MANIFEST).write_text(json.dumps({
        "members": [{"name": "main", "dir": "."}, {"name": "twin", "dir": "members/twin"},
                    {"name": "rx", "dir": "members/rx"}],
        "gate": {"member": "rx", "weights": weights}}))
    return root


def test_a_gated_average_is_the_blend_the_research_queries_scored(tmp_path):
    """research/queries/done/gate_clv.py: the other members' average (the pair) ranks each race;
    the gated member's normalised log price takes weight w by that rank, the pair's 1 - w, and
    the race is renormalised."""
    model, cols, _ = pbt.load_bfsp_model(str(_gated(tmp_path)))
    assert model.reads_races and model.gate == {"member": "rx", "weights": STEP}
    frame = _frame(seed=9)
    served = predict_prices(model, frame, cols, race_col="raceid")
    race = frame["raceid"].to_numpy()
    own = {n: np.log(predict_prices(b, frame, c, race_col="raceid")["predicted_bfsp"].to_numpy())
           for n, b, c in model.members}
    p = np.exp(-(own["main"] + own["twin"]) / 2)
    pair = -np.log(p / pd.Series(p).groupby(race).transform("sum").to_numpy())
    rank = pd.Series(pair).groupby(race).rank(method="first").to_numpy()
    w = np.where(rank <= 3, 1.0, np.where(rank <= 7, 0.5, 0.0))
    q = np.exp(-(w * own["rx"] + (1 - w) * pair))
    want = q / pd.Series(q).groupby(race).transform("sum").to_numpy()
    np.testing.assert_allclose(served["predicted_win_prob_norm"].to_numpy(), want, rtol=1e-9)
    assert (w == 1).any() and (w == 0.5).any() and (w == 0).any()     # the frame reaches every step
    book = served.groupby("raceid")["predicted_bfsp"].apply(lambda s: (1 / s).sum())
    np.testing.assert_allclose(book, 1.0, rtol=1e-12)


def test_weights_of_nothing_and_everything_are_the_pair_and_the_gated_member(tmp_path):
    frame = _frame(seed=11)
    a = tmp_path / "zero"
    b = tmp_path / "one"
    zero, cols, _ = pbt.load_bfsp_model(str(_gated(a, weights=[0.0])))
    one, _, _ = pbt.load_bfsp_model(str(_gated(b, weights=[1.0])))
    got0 = predict_prices(zero, frame, cols, race_col="raceid")["predicted_bfsp"].to_numpy()
    got1 = predict_prices(one, frame, cols, race_col="raceid")["predicted_bfsp"].to_numpy()
    main, twin, rx = zero.members
    logs = [np.log(predict_prices(bst, frame, c, race_col="raceid")["predicted_bfsp"].to_numpy())
            for _, bst, c in (main, twin)]
    p = np.exp(-np.mean(logs, axis=0))
    pair = 1 / (p / pd.Series(p).groupby(frame["raceid"].to_numpy()).transform("sum").to_numpy())
    np.testing.assert_allclose(got0, pair, rtol=1e-9)
    alone = predict_prices(rx[1], frame, rx[2], race_col="raceid")["predicted_bfsp"].to_numpy()
    np.testing.assert_allclose(got1, alone, rtol=1e-9)


def test_a_gate_without_the_races_is_refused(tmp_path):
    model, cols, _ = pbt.load_bfsp_model(str(_gated(tmp_path)))
    with pytest.raises(ValueError, match="races"):
        model.predict(_frame(seed=5)[cols].astype(float))


@pytest.mark.parametrize("gate, match", [({"member": "nobody", "weights": [1.0]}, "not a member"),
                                         ({"member": "rx", "weights": []}, "weights by rank"),
                                         ({"member": "rx", "weights": [1.2, 0.0]}, "between 0 and 1"),
                                         ({"member": "rx", "weights": [float("nan")]}, "between 0 and 1")])
def test_a_gate_that_cannot_be_served_is_refused(tmp_path, gate, match):
    _gated(tmp_path)
    spec = json.loads((tmp_path / pbt.ENSEMBLE_MANIFEST).read_text())
    (tmp_path / pbt.ENSEMBLE_MANIFEST).write_text(json.dumps({**spec, "gate": gate}))
    with pytest.raises(ValueError, match=match):
        pbt.load_bfsp_model(str(tmp_path))


def test_verify_reads_a_gated_model_and_its_checks_pass(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import verify_model as vm
    model, meta = vm._load(str(_gated(tmp_path)))
    assert meta["members"] == ["main", "twin", "rx"] and meta["gate"]["member"] == "rx"
    assert vm.check_servable(model, meta) == []
    problems, notes = vm.check_prices(model, meta, None, None, _frame(seed=6))
    assert problems == [], problems


def test_each_members_log_line_names_the_loss_it_was_fitted_with(averaged, caplog):
    """The metadata's "objective" is the training config's label ("l2" for a loss set through
    --param objective=huber); the log reads the booster's own parameters instead."""
    import logging
    for sub in (".", "members/huber"):                 # as train_bfsp.py writes it for either loss
        path = averaged / sub / "bfsp_model_meta.json"
        path.write_text(json.dumps({**json.loads(path.read_text()), "objective": "l2"}))
    with caplog.at_level(logging.INFO, logger=pbt.log.name):
        pbt.load_bfsp_model(str(averaged))
    lines = [r.getMessage() for r in caplog.records if "Averaged model member" in r.getMessage()]
    assert len(lines) == 2
    assert "member l2:" in lines[0] and "objective=regression" in lines[0]
    assert "member huber:" in lines[1] and "objective=huber" in lines[1]


def test_a_lone_compressed_booster_serves_as_the_plain_file_would(tmp_path):
    """A booster past GitHub's 100 MB file limit is committed as bfsp_model.lgb.xz alone."""
    plain, packed = tmp_path / "plain", tmp_path / "packed"
    _member(plain, ".", ["f_a", "f_b"], "regression", seed=3)
    _member(packed, ".", ["f_a", "f_b"], "regression", seed=3, xz=True)
    assert not (packed / "bfsp_model.lgb").exists()
    a, cols, vocab = pbt.load_bfsp_model(str(plain))
    b, cols_b, vocab_b = pbt.load_bfsp_model(str(packed))
    assert isinstance(b, lgb.Booster) and cols_b == cols and vocab_b == vocab
    X = _frame(seed=5)[cols].astype(float)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    assert b.serving_target == a.serving_target


def test_verify_reads_a_lone_compressed_booster(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import verify_model as vm
    _member(tmp_path, ".", ["f_a", "f_b"], "regression", seed=3, xz=True)
    model, meta = vm._load(str(tmp_path))
    assert isinstance(model, lgb.Booster) and meta["feature_cols"] == ["f_a", "f_b"]
    assert vm.check_servable(model, meta) == []


def test_with_both_files_the_plain_one_is_read_and_the_stale_pair_is_named(tmp_path, caplog):
    import logging
    _member(tmp_path, ".", ["f_a", "f_b"], "regression", seed=3)
    (tmp_path / "bfsp_model.lgb.xz").write_bytes(b"stale")
    with caplog.at_level(logging.WARNING, logger=pbt.log.name):
        assert pbt.booster_path(str(tmp_path)).endswith("bfsp_model.lgb")
    assert any("both bfsp_model.lgb and bfsp_model.lgb.xz" in r.getMessage() for r in caplog.records)


def test_the_repo_never_holds_a_plain_and_a_compressed_booster_side_by_side():
    """The plain file would be read and the compressed one ignored: a deploy that adds a
    model as .lgb.xz must delete the bfsp_model.lgb it replaces."""
    root = Path(__file__).resolve().parent.parent / "data" / "models"
    both = [str(d.relative_to(root.parent.parent)) for d in [root, *root.rglob("*")]
            if d.is_dir() and (d / "bfsp_model.lgb").exists() and (d / "bfsp_model.lgb.xz").exists()]
    assert both == [], f"stale booster beside its compressed replacement in {both}"
