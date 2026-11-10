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
    if "weight_kg" in b:
        b.loc[last, "weight_kg"] += 3.0                           # and the last day's weigh-in
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
    assert fl.lane({"name": "e", "limit": True}, spec) == "slow"
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


def _acc(seed, better=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range("2026-10-14", periods=10).strftime("%Y-%m-%d"):
        for mark in fl.SHORT_MARKS:
            base = 1.5 + rng.normal(0, 0.01)
            for model, x in (("market_micro", base), ("micro", base - 0.01 - better)):
                rows.append({"day": d, "mark": mark, "model": model, "races": 50, "rows": 300, "xent": x,
                             "xent_market": base + 0.005, "sse": 10.0, "sse_nochange": 20.0, "right_way": 0.6})
    return pd.DataFrame(rows)


def test_the_shortening_score_and_its_paired_comparison(tmp_path):
    s = fl.shortening_summary(_acc(0))
    assert set(s) == set(fl.SHORT_MARKS)
    assert abs(s["off_5"]["model_adds"] - 0.01) < 1e-9 and s["off_5"]["r2"] == 0.5 and s["off_5"]["days"] == 10
    _acc(0).to_parquet(tmp_path / "short_a.parquet")
    _acc(0, better=0.002).to_parquet(tmp_path / "short_b.parquet")
    pr = fl.paired_shortening(tmp_path, "a")
    assert pr["a"]["diff"] == 0.0 and abs(pr["b"]["diff"] + 0.002) < 1e-6 and pr["b"]["ci90"][1] < 0
    r = {"name": "b", "features": 3, "spec": {"objective": "logbsp"}, "rounds": 9, "logloss": 0.44, "seconds": 1,
         "test_from": "2026-06-01", "test_to": "2026-11-02", "shortening": s, "paired_short": pr["b"]}
    t = fl.table([dict(r, name="a", paired_short=pr["a"]), r], "a")
    assert "0.01 / 0.01 / 0.01" in t and "-0.002" in t


def test_the_bsp_rmse_is_paired_with_the_baseline(tmp_path):
    import numpy as np
    import pandas as pd
    from greyhound import fastloop as fl
    rng = np.random.default_rng(0)
    n = 400
    base = pd.DataFrame({"race_id": np.repeat(np.arange(n // 4), 4), "dog_id": np.arange(n),
                         "race_date": np.repeat([f"2026-10-{d:02d}" for d in range(1, 11)], n // 10),
                         "bsp": np.exp(rng.normal(1.5, 0.6, n))})
    good = base.assign(pred_bsp=base.bsp * np.exp(rng.normal(0, 0.1, n)))
    bad = base.assign(pred_bsp=base.bsp * np.exp(rng.normal(0, 0.4, n)))
    bad.to_parquet(tmp_path / "pred_base.parquet")
    good.to_parquet(tmp_path / "pred_better.parquet")
    out = fl.paired_bsp_rmse(tmp_path, "base")
    assert out["base"]["diff"] == 0.0
    b = out["better"]
    assert b["diff"] < 0 and b["ci90"][1] < 0 and b["rmse"] < out["base"]["rmse"]
    r = {"name": "better", "spec": {"objective": "logbsp"}, "features": 1, "rounds": 1, "logloss": 0.4, "seconds": 1,
         "test_from": "2026-10-01", "test_to": "2026-10-10", "paired_bsp": b}
    assert "RMSE log(BSP)" in fl.table([r], "better") and str(b["rmse"]) in fl.table([r], "better")


def test_the_bsp_forecast_rescaled_to_the_sp_book():
    df, meta = synthetic_cache()
    exp = {"name": "r", "features": "card_safe", "objective": "logbsp", "rescale_book": True}
    res, pred = fl.run_experiment(df, meta, exp, QUICK, test_from="2026-06-01")
    totals = (1.0 / pred.pred_bsp).groupby(pred.race_id).sum()
    fit = df[(df.t < "2026-06-01") & (df.bsp > 1)]
    book = float(np.median((1.0 / fit.bsp).groupby(fit.race_id).sum()))
    assert np.allclose(totals, totals.iloc[0]) and abs(totals.iloc[0] - book) < 0.05   # every race at the SP book
    assert res["log_bsp_rmse"] > 0


def _compressed_races(n_races=400, b_true=1.6, seed=1):
    """Races whose log(BSP) is b_true times the model's race-centred log price, and winners drawn from the BSP."""
    rng = np.random.default_rng(seed)
    race = np.repeat(np.arange(n_races), 6)
    log_pred = rng.normal(1.6, 0.4, len(race))
    c = fl._centre(log_pred, race)
    p = pd.Series(np.exp(-b_true * c)).groupby(race).transform(lambda s: s / s.sum()).to_numpy()
    log_bsp = -np.log(p) + rng.normal(0, 0.05, len(race))          # a book of 1.0 with a little noise
    won = np.zeros(len(race))
    for r in range(n_races):
        i = np.flatnonzero(race == r)
        won[rng.choice(i, p=p[i] / p[i].sum())] = 1
    track = np.where(race % 2, "Romford", "Hove")
    return log_pred, log_bsp, won, race, track


def test_the_stretch_recovers_a_compressed_price():
    log_pred, log_bsp, won, race, track = _compressed_races()
    s = fl.fit_stretch(log_pred, log_bsp, won, race, track, "stretch")
    assert abs(s["b"] - 1.6) < 0.05
    t = fl.fit_stretch(log_pred, log_bsp, won, race, track, "temper")
    assert t["b"] > 1.0                                            # the winners also say: spread the prices
    st = fl.fit_stretch(log_pred, log_bsp, won, race, track, "stretch_track", shrink=0.0)
    assert set(st["by_track"]) == {"Hove", "Romford"} and all(abs(v - 1.6) < 0.1 for v in st["by_track"].values())
    with pytest.raises(ValueError):
        fl.fit_stretch(log_pred, log_bsp, won, race, track, "isotonic")


def test_the_stretched_prices_beat_the_compressed_ones_and_stay_a_book():
    log_pred, log_bsp, won, race, track = _compressed_races(seed=2)
    pred = pd.DataFrame({"race_id": race, "race_date": [f"2026-10-{1 + r % 20:02d}" for r in race], "dog_id":
                         np.arange(len(race)), "track": track, "won": won, "bsp": np.exp(log_bsp)})
    book = 1.0
    p0 = pd.Series(np.exp(-log_pred)).groupby(race).transform(lambda s: s / s.sum())
    pred["p_model"], pred["pred_bsp"] = p0.to_numpy(), 1.0 / (book * p0.to_numpy())
    fit = fl.fit_stretch(log_pred, log_bsp, won, race, track, "stretch")
    cal = fl.apply_stretch(pred, log_pred, fit, book)
    assert np.allclose(cal.groupby("race_id").p_model.sum(), 1.0)
    assert np.allclose((1.0 / cal.pred_bsp).groupby(cal.race_id).sum(), book)
    sc = fl.score_pair(pred, cal)
    assert sc["rmse_vs_uncalibrated"] < 0 and sc["rmse_ci90"][1] < 0
    assert sc["logloss_vs_uncalibrated"] < 0


def test_calibration_in_an_experiment_and_its_spec_check():
    df, meta = synthetic_cache()
    exp = {"name": "c", "features": "card_safe", "objective": "logbsp", "rescale_book": True,
           "calibrate": ["stretch", "temper_track"]}
    res, _ = fl.run_experiment(df, meta, exp, QUICK, test_from="2026-06-01")
    assert set(res["calibrated"]) == {"stretch", "temper_track"}
    assert "rmse_ci90" in res["calibrated"]["stretch"] and res["calibrated"]["stretch"]["fit"]["b"] > 0
    assert "+ stretch" in fl.table([res], "c")
    with pytest.raises(ValueError):
        fl.check_spec({"experiments": [{"name": "x", "objective": "softmax", "calibrate": ["stretch"]}]})
    with pytest.raises(ValueError):
        fl.check_spec({"experiments": [{"name": "x", "objective": "logbsp", "calibrate": ["isotonic"]}]})


def test_track_groups_are_scored_apart():
    df, meta = synthetic_cache()
    exp = {"name": "g", "features": "card_safe", "objective": "logbsp", "rescale_book": True,
           "groups": {"weighin": ["hove"]}}
    res, pred = fl.run_experiment(df, meta, exp, QUICK, test_from="2026-06-01")
    bg = res["by_group"]
    assert set(bg) == {"weighin", "rest"}
    assert bg["weighin"]["runners"] + bg["rest"]["runners"] == len(pred)
    assert bg["weighin"]["tracks"] == 1 and bg["rest"]["tracks"] == 1          # "hove" matches "Hove"
    assert "gap" in bg["weighin"]["on_bsp_races"] and bg["rest"]["log_bsp_rmse"] > 0
    assert fl.split_groups(pred, {"weighin": ["Hove"]})["rest"] == ["Romford"]
    assert "[weighin: 1 tracks" in fl.table([res], "g")
    assert set(fl.split_groups(pred, "each_track")) == {"Hove", "Romford"}


def _weighed_dogs(n_dogs=300, runs=20, seed=3):
    """Dogs whose weight reverts to their own norm, a run a week from Oct 2024 into 2025, with a finishing position."""
    rng = np.random.default_rng(seed)
    rows = []
    for dog in range(n_dogs):
        mu = 30 + rng.normal(0, 2)
        w, hist = mu + rng.normal(0, 0.5), []
        for k in range(runs):
            day = pd.Timestamp("2024-10-01") + pd.Timedelta(days=7 * k + dog % 7)
            w = mu + 0.4 * (w - mu) + rng.normal(0, 0.5) if hist else w
            rows.append({"dog_id": dog, "t": day, "race_date": f"{day:%Y-%m-%d}", "weight_kg": w,
                         "weight_change": w - hist[-1] if hist else np.nan,
                         "weight_vs_mean": w - np.mean(hist) if hist else np.nan,
                         "position": int(rng.integers(1, 7)), "won": 0})
            hist.append(w)
    return pd.DataFrame(rows)


def test_the_predicted_weigh_in_from_weight_history_recency_and_form(monkeypatch):
    from greyhound.blocks import wpred
    monkeypatch.setattr(wpred, "MIN_FIT", 1000)
    df = _weighed_dogs()
    df.loc[df.index[5], "weight_kg"] = 2855.0                  # an impossible weight is read as missing
    names = blocks.attach(df, "wpred")
    assert df.wp_prev_weight.max() < 50 and df.wp_pred_weight.max() < 50
    assert {"wp_pred_weight", "wp_pred_change", "wp_pred_vs_mean", "wp_prev_weight", "wp_gap_days"} <= set(names)
    late = df[df.t >= "2025-01-01"].dropna(subset=["wp_pred_change"])
    assert len(late) > 1000 and df[df.t < "2025-01-01"].wp_pred_change.isna().all()   # 2025 from 2024 only
    actual = late.weight_kg - late.wp_prev_weight
    assert np.corrcoef(late.wp_pred_change, actual)[0, 1] > 0.3                 # the predictable part is found
    assert np.abs(late.wp_pred_weight - late.weight_kg).mean() < np.abs(late.wp_prev_weight - late.weight_kg).mean()
    last = df.t.dt.normalize() == df.t.dt.normalize().max()                     # nothing of the day's own weigh-in
    df2 = _weighed_dogs()
    df2.loc[df2.index[5], "weight_kg"] = 2855.0
    df2.loc[last, "weight_kg"] += 5.0
    blocks.attach(df2, "wpred")
    assert np.allclose(df.loc[last, "wp_pred_weight"], df2.loc[last, "wp_pred_weight"], equal_nan=True)


def test_the_predicted_weigh_in_without_raw_weights_falls_back_to_the_relative_model():
    df = _weighed_dogs().drop(columns="weight_kg")
    names = blocks.attach(df, "wpred")
    assert set(names) == {"wp_pred_change", "wp_pred_vs_mean", "wp_prev_vs_mean", "wp_gap_days"}
    late = df[df.t >= "2024-12-01"].dropna(subset=["wp_pred_change", "weight_change"])
    assert len(late) > 1000 and np.corrcoef(late.wp_pred_change, late.weight_change)[0, 1] > 0.3


def test_add_features_names_cached_columns():
    df, meta = synthetic_cache()
    exp = {"name": "a", "features": "card_safe", "objective": "logbsp", "add_features": ["weight_change"]}
    res, _ = fl.run_experiment(df, meta, exp, QUICK, test_from="2026-06-01")
    assert res["features"] == len(fl.feature_set(meta, "card_safe")) + 1


# ---- the plan of 10 Nov 2026 (reports/greyhound_bfsp_plan.md): steps 0 and 1

def test_the_soft_race_objective_learns_the_book_shares():
    from greyhound.model import _race_softmax
    race = np.repeat(np.arange(3), 4)
    target = np.array([.5, .3, .1, .1, .25, .25, .25, .25, .7, .1, .1, np.nan])     # the last race is not wholly priced
    fobj, feval = fl.race_soft_objective(race, target)
    score = np.zeros(12)
    for _ in range(300):                                          # Newton steps on the objective alone
        g, h = fobj(score, None)
        score -= g / h
    p = _race_softmax(score, fl._starts(race))
    assert np.allclose(p[:8], target[:8], atol=1e-3)              # the shares recovered where the race is priced
    assert np.allclose(p[8:], 0.25)                               # no gradient where it is not
    assert feval(score, None)[1] == pytest.approx(-(target[:8] * np.log(target[:8])).sum() / 2, abs=1e-3)
    mix = fl.soft_target(pd.DataFrame({"race_id": [1, 1], "bsp_p_norm": [.6, .4], "won": [0, 1]}), "mix", 0.5)
    assert np.allclose(mix, [.3, .7])


@pytest.mark.parametrize("objective", ["share", "mix"])
def test_the_race_objectives_price_every_runner_as_a_coherent_book(objective):
    df, meta = synthetic_cache()
    res, pred = fl.run_experiment(df, meta, {"name": "s", "features": "card_safe", "objective": objective}, QUICK,
                                  test_from="2026-06-01")
    assert np.allclose(pred.groupby("race_id").p_model.sum(), 1.0)
    book = (1 / pred.pred_bsp).groupby(pred.race_id).sum()
    assert np.allclose(book, book.iloc[0])                        # every race's book at the SP book's usual total
    blind, _ = fl.run_experiment(df, meta, {"name": "b", "features": "card_safe", "objective": objective,
                                            "keep": ["trap"]}, QUICK, test_from="2026-06-01")
    assert res["log_bsp_rmse"] < blind["log_bsp_rmse"]            # the informative features price better than none
    assert res["breakdown"]["r2_within_race"] > blind["breakdown"]["r2_within_race"]


def test_keep_seeds_halflife_and_huber_fit():
    df, meta = synthetic_cache()
    res, _ = fl.run_experiment(df, meta, {"name": "k", "features": "card_safe", "objective": "logbsp",
                                          "keep": ["lb_kr", "trap"]}, QUICK, test_from="2026-06-01")
    assert res["features"] == 2
    res2, pred = fl.run_experiment(df, meta, {"name": "w", "features": "card_safe", "objective": "logbsp", "seeds": 2,
                                              "halflife_days": 30, "huber": 0.5}, QUICK, test_from="2026-06-01")
    assert res2["seeds"] == 2 and np.isfinite(pred.pred_bsp).all()
    with pytest.raises(ValueError):
        fl.check_spec({"experiments": [{"name": "x", "seeds": 0}]})


def test_the_importance_table_finds_the_informative_family():
    df, meta = synthetic_cache()
    res, _ = fl.run_experiment(df, meta, {"name": "i", "features": "card_safe", "objective": "logbsp",
                                          "importance": True, "importance_races": 300, "shap_rows": 500}, QUICK,
                               test_from="2026-06-01")
    imp = res["importance"]
    # the lib family (lb_kr: ability plus noise) and the base family (esr: ability plus noise) carry the price
    assert imp["perm"]["lib"]["rmse_rise"] > 0 and imp["perm"]["base"]["rmse_rise"] > 0
    assert set(imp["shap_family"]) >= {"lib", "base", "parity"} and imp["shap_top"]
    # the within-race shuffle keeps each race's set of values: shuffled rows are a permutation within each race
    race = np.repeat(np.arange(50), 6)
    base, shuf = fl._within_race_perm(race, np.random.default_rng(0))
    assert (race[base] == race[shuf]).all() and sorted(shuf) == list(range(300))


def test_the_walk_refits_through_the_window_and_scores_it_as_one(tmp_path):
    df, meta = synthetic_cache()
    res, pred = fl.run_walk(df, meta, {"name": "wk", "features": "card_safe", "objective": "logbsp"}, QUICK,
                            ["2026-05-15", "2026-06-05"])
    assert [w["test_from"] for w in res["walk"]] == ["2026-05-15", "2026-06-05"]
    assert res["walk"][0]["test_to"] == "2026-06-04" and pred.race_date.min() >= "2026-05-15"
    assert res["test_runners"] == len(pred) == pred[["race_id", "dog_id"]].drop_duplicates().shape[0]
    assert res["breakdown"]["runners"] == int((pred.bsp > 1).sum())
    md = fl.details([res])
    assert "segment by segment" in md and "Where wk's error sits" in md


def test_the_paired_comparison_carries_a_z_score(tmp_path):
    df, meta = synthetic_cache()
    for name, keep in (("good", None), ("blind", ["trap"])):
        e = {"name": name, "features": "card_safe", "objective": "logbsp", **({"keep": keep} if keep else {})}
        _, pred = fl.run_experiment(df, meta, e, QUICK, test_from="2026-06-01")
        pred[["race_id", "race_date", "dog_id", "won", "p_model", "bsp", "pred_bsp"]].to_parquet(
            tmp_path / f"pred_{name}.parquet")
    pb = fl.paired_bsp_rmse(tmp_path, "blind")
    assert pb["good"]["diff"] < 0 and pb["good"]["z"] < -3 and pb["blind"]["z"] is None
    assert "z" in fl.paired_vs_baseline(tmp_path, "blind")["good"]


def test_the_stretch_calibrates_the_race_objective_too():
    df, meta = synthetic_cache()
    res, pred = fl.run_experiment(df, meta, {"name": "sc", "features": "card_safe", "objective": "share",
                                             "calibrate": ["stretch", "stretch_track"]}, QUICK, test_from="2026-06-01")
    c = res["calibrated"]["stretch"]
    assert c["fit"]["b"] > 0 and np.isfinite(c["log_bsp_rmse"]) and "rmse_vs_uncalibrated" in c



@pytest.mark.parametrize("objective", ["nn_share", "stack_share"])
def test_the_neural_set_model_and_the_stack_price_a_coherent_book(objective, monkeypatch):
    pytest.importorskip("torch")
    from greyhound import nn
    monkeypatch.setattr(nn, "MAX_EPOCHS", 6)
    df, meta = synthetic_cache()
    res, pred = fl.run_experiment(df, meta, {"name": "n", "features": "card_safe", "objective": objective,
                                             "nn_seconds": 120}, QUICK, test_from="2026-06-01")
    assert np.allclose(pred.groupby("race_id").p_model.sum(), 1.0)
    blind, _ = fl.run_experiment(df, meta, {"name": "b", "features": "card_safe", "objective": "share",
                                            "keep": ["trap"]}, QUICK, test_from="2026-06-01")
    assert res["log_bsp_rmse"] < blind["log_bsp_rmse"] and res["nn"]["epochs"] >= 1
    if objective == "stack_share":
        assert set(res["stack"]) == {"a_trees", "b_network"}


def test_several_networks_are_averaged(monkeypatch):
    pytest.importorskip("torch")
    from greyhound import nn
    monkeypatch.setattr(nn, "MAX_EPOCHS", 4)
    df, meta = synthetic_cache()
    res, pred = fl.run_experiment(df, meta, {"name": "n2", "features": "card_safe", "objective": "stack_share",
                                             "nn_seeds": 2, "nn_seconds": 120}, QUICK, test_from="2026-06-01")
    assert res["nn"]["seeds"] == 2 and len(res["nn"]["best_by_seed"]) == 2
    assert np.allclose(pred.groupby("race_id").p_model.sum(), 1.0)
    with pytest.raises(ValueError):
        fl.check_spec({"experiments": [{"name": "x", "objective": "nn_share", "nn_seeds": 11}]})


def test_saved_prices_are_reused_and_the_served_price_is_the_stretched_one(tmp_path, monkeypatch):
    df, meta = synthetic_cache()
    monkeypatch.setattr(fl, "load_cache", lambda store: (df, meta))
    monkeypatch.setitem(fl.RECIPES, "quick", QUICK)
    spec = {"recipe": "quick", "test_from": "2026-06-01", "experiments": [
        {"name": "s", "objective": "share", "calibrate": ["stretch_track"], "serve_calibrated": "stretch_track",
         "save_pred": "greyhound/research/pred", "test_from": "2026-06-10"}]}
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    out = tmp_path / "out"
    assert fl.main(["--store", str(tmp_path), "--spec", str(p), "--out", str(out)]) == 0
    from sources.common import Store
    saved = Store(root=tmp_path).get_parquet("greyhound/research/pred/s.parquet")
    assert str(saved.race_date.min()) >= "2026-06-10"                    # the experiment's own test start
    assert {"pred_bsp", "pred_bsp_raw", "p_model"} <= set(saved.columns)
    assert not np.allclose(saved.pred_bsp, saved.pred_bsp_raw)          # the stretch moved the price
    r = json.loads((out / "result_s.json").read_text())
    spec2 = {"experiments": [{"name": "s", "pred_from": "greyhound/research/pred"}]}
    p.write_text(json.dumps(spec2))
    out2 = tmp_path / "out2"
    assert fl.main(["--store", str(tmp_path), "--spec", str(p), "--out", str(out2)]) == 0
    r2 = json.loads((out2 / "result_s.json").read_text())
    assert r2["stopped_by"] == "saved" and r2["logloss"] == r["logloss"] and r2["test_runners"] == r["test_runners"]
    with pytest.raises(ValueError):
        fl.check_spec({"experiments": [{"name": "x", "serve_calibrated": "stretch"}]})
    with pytest.raises(ValueError):
        fl.check_spec({"experiments": [{"name": "x", "save_pred": "../etc"}]})
