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


@pytest.mark.parametrize("targets", [("logit_norm_prob", "logit_norm_prob"), ("logit_norm_prob", "log_bfsp")])
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
