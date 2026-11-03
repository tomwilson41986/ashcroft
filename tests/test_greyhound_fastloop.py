"""The greyhound fast loop and its feature blocks (greyhound/fastloop.py, greyhound/blocks/): mechanics on a synthetic
cache, never a figure of record."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from greyhound import blocks, fastloop as fl


def synthetic_cache(n_days=120, races_a_day=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    rid = 0
    for d in range(n_days):
        day = pd.Timestamp("2026-03-01") + pd.Timedelta(days=d)
        for r in range(races_a_day):
            rid += 1
            ability = rng.normal(0, 1, 6)
            p = np.exp(ability) / np.exp(ability).sum()
            order = rng.choice(6, size=6, replace=False, p=p)
            pos = np.empty(6, int)
            pos[order] = np.arange(1, 7)
            bsp_p = np.exp(0.8 * ability + rng.normal(0, 0.3, 6))
            bsp_p /= bsp_p.sum()
            for i in range(6):
                rows.append({"race_id": rid, "t": day + pd.Timedelta(minutes=10 * r), "race_date": f"{day:%Y-%m-%d}",
                             "race_time": "12:00", "track": "Romford" if r % 2 else "Hove", "trap": i + 1,
                             "dog_id": rid * 10 + i, "dog_name": "d", "field": 6, "won": int(pos[i] == 1),
                             "position": int(pos[i]), "sp_decimal": 1 / bsp_p[i], "sp_p_norm": bsp_p[i],
                             "bsp": 1 / bsp_p[i], "bsp_p_norm": bsp_p[i], "distance_m": 400.0,
                             "esr_mean3": ability[i] + rng.normal(0, 0.5), "esr_max3": ability[i] + 1,
                             "brk_mean6": rng.normal(), "led_mean6": rng.random(), "path_mean6": rng.uniform(1, 3),
                             "lb_kr": ability[i] + rng.normal(0, 0.7), "gp_nfpz_w5": ability[i] + rng.normal(0, 1),
                             "weight_change": rng.normal(), "is_trial": False})
    df = pd.DataFrame(rows)
    meta = {"hash": "test", "last_day": str(df.race_date.max()), "rows": len(df),
            "features": {"base": ["esr_mean3", "esr_max3", "brk_mean6", "led_mean6", "path_mean6", "trap",
                                  "weight_change", "distance_m"],
                         "lib": ["lb_kr"], "parity": ["gp_nfpz_w5"]},
            "card_unsafe": ["weight_change"]}
    return df, meta


QUICK = {"train_from": "2026-03-01", "learning_rate": 0.2, "early_stop": 10, "max_bin": 31, "rounds": 60}


@pytest.mark.parametrize("name", blocks.NAMES)
def test_blocks_read_no_result_of_their_own_day(name):
    df, _ = synthetic_cache(n_days=30, races_a_day=6)
    a = df.copy()
    feats = blocks.attach(a, name)
    assert feats and all(f.startswith(__import__(f"greyhound.blocks.{name}", fromlist=["x"]).PREFIX) for f in feats)
    b = df.copy()
    last = b.race_date == b.race_date.max()
    b.loc[last, "won"] = 1 - b.loc[last, "won"]                  # the last day's results rewritten
    b.loc[last, "position"] = 7 - b.loc[last, "position"]
    blocks.attach(b, name)
    pd.testing.assert_frame_equal(a.loc[last, feats].reset_index(drop=True), b.loc[last, feats].reset_index(drop=True))
    assert blocks.attach(a, name) == feats                        # attached once


def test_style_draw_counts_earlier_days_only():
    df, _ = synthetic_cache(n_days=5, races_a_day=4)
    blocks.attach(df, "style_draw")
    first = df.race_date == df.race_date.min()
    assert (df.loc[first, "sd_n"] == 0).all()                     # nothing before the first day
    assert (df.loc[~first, "sd_n"] >= 0).all() and df.sd_n.max() > 0


def test_first_bend_lead_chances_are_a_book():
    df, _ = synthetic_cache(n_days=3, races_a_day=3)
    blocks.attach(df, "first_bend")
    assert np.allclose(df.groupby("race_id").fb2_lead_p.sum(), 1.0, atol=1e-5)


def test_feature_sets():
    _, meta = synthetic_cache(n_days=1, races_a_day=1)
    assert "weight_change" in fl.feature_set(meta, "all")
    assert "weight_change" not in fl.feature_set(meta, "card_safe")
    assert fl.feature_set(meta, "base-card") == [f for f in meta["features"]["base"] if f != "weight_change"]
    with pytest.raises(ValueError):
        fl.feature_set(meta, "nonsense")


@pytest.mark.parametrize("objective", fl.OBJECTIVES)
def test_each_objective_fits_and_scores_out_of_sample(objective):
    df, meta = synthetic_cache()
    exp = {"name": f"t_{objective}", "features": "card_safe", "objective": objective, "add_blocks": ["race_rel2"]}
    res, pred = fl.run_experiment(df, meta, exp, QUICK, test_from="2026-06-01")
    assert pred.race_date.min() >= "2026-06-01"                   # every scored race after the fit
    assert np.allclose(pred.groupby("race_id").p_model.sum(), 1.0)
    uniform = -(np.log(1 / 6) / 6 + 5 / 6 * np.log(5 / 6))          # a runner's log-loss knowing nothing
    assert res["logloss"] < uniform
    assert res["on_bsp_races"]["runners"] == len(pred) and "logloss_second_half" in res["benter_vs_bsp"]
    if objective == "logbsp":
        assert res["log_bsp_rmse"] > 0


def test_an_informative_feature_beats_none():
    df, meta = synthetic_cache()
    good, _ = fl.run_experiment(df, meta, {"name": "g", "features": "card_safe"}, QUICK, test_from="2026-06-01")
    bad, _ = fl.run_experiment(df, meta, {"name": "b", "features": "card_safe", "drop": ["lb_", "gp_", "esr"]},
                               QUICK, test_from="2026-06-01")
    assert good["logloss"] < bad["logloss"]


def test_spec_checks_and_the_cli(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        fl.check_spec({"experiments": [{"name": "Bad Name"}]})
    with pytest.raises(ValueError):
        fl.check_spec({"experiments": [{"name": "a", "objective": "nope"}]})
    spec = {"recipe": "quick", "baseline": "a", "experiments": [{"name": "a"}, {"name": "b", "add_blocks": ["first_bend"]}]}
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    assert fl.main(["--store", str(tmp_path), "--spec", str(p), "--list"]) == 0
    df, meta = synthetic_cache()
    monkeypatch.setattr(fl, "load_cache", lambda store: (df, meta))
    monkeypatch.setitem(fl.RECIPES, "quick", QUICK)
    spec["test_from"] = "2026-06-01"
    p.write_text(json.dumps(spec))
    out = tmp_path / "out"
    assert fl.main(["--store", str(tmp_path), "--spec", str(p), "--out", str(out)]) == 0
    assert (out / "result_a.json").exists() and (out / "result_b.json").exists()
    assert fl.main(["--store", str(tmp_path), "--spec", str(p), "--collect", str(out)]) == 0
    t = (out / "table.md").read_text()
    assert "| a |" in t and "| b |" in t
    paired = fl.paired_vs_baseline(out, "a")
    assert paired["a"]["diff"] == 0.0 and paired["b"]["ci90"][0] <= paired["b"]["diff"] <= paired["b"]["ci90"][1]


def test_code_hash_is_stable():
    assert fl.code_hash() == fl.code_hash() and len(fl.code_hash()) == 12


def test_lanes_split_the_quick_from_the_slow(tmp_path, capsys):
    spec = {"recipe": "quick", "baseline": "a", "experiments": [
        {"name": "a"}, {"name": "b", "recipe": "confirm"}, {"name": "c", "study": True}, {"name": "d"}]}
    assert [fl.lane(e, spec) for e in spec["experiments"]] == ["quick", "slow", "slow", "quick"]
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    for lane, want in (("quick", ["a", "d"]), ("slow", ["b", "c"]), (None, ["a", "b", "c", "d"])):
        capsys.readouterr()
        assert fl.main(["--store", str(tmp_path), "--spec", str(p), "--list"] + (["--lane", lane] if lane else [])) == 0
        assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == want


def test_the_time_budget_stops_the_fit_at_its_best_round():
    df, meta = synthetic_cache()
    res, _ = fl.run_experiment(df, meta, {"name": "t", "features": "card_safe", "objective": "logbsp"},
                               dict(QUICK, rounds=100000, early_stop=100000, max_seconds=1e-9), test_from="2026-06-01")
    assert res["stopped_by"] == "time" and res["rounds"] >= 1
    res, _ = fl.run_experiment(df, meta, {"name": "c", "features": "card_safe", "objective": "logbsp"},
                               dict(QUICK, rounds=5, early_stop=100), test_from="2026-06-01")
    assert res["stopped_by"] == "cap"
