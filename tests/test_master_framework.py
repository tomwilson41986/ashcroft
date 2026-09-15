"""Tests for the racing² v3 master-framework modules: stage_f, state_space,
primitives, ordering, feature_registry (synthetic data exercises code paths only)."""

import numpy as np
import pandas as pd
import pytest

from model.feature_registry import classify, leakage_audit, stage_f_columns, stage_of
from model.ordering import fit_ordering_params, place_probabilities, top3_probabilities
from model.primitives import (BeatenLengthsPar, add_run_primitives, aggregate_history, fss_cred, lto_contradiction_features,
                              n_eff_from_ratings, nmfp, prb2_par, strength_of_schedule, within_race_transforms)
from model.stage_f import (ConditionalLogit, StageC, TemperatureScaler, conditional_calibration, delta_r2, lgb_race_softmax,
                           mcfadden_r2, race_softmax, walk_forward_stage_f)
from model.state_space import add_kalman_features, ewm_baseline, fit_kalman_params


def _pl_races(n_races=1500, beta=(1.0, -0.5, 0.3), seed=0):
    rng = np.random.default_rng(seed); rows = []
    for r in range(n_races):
        n = int(rng.integers(5, 13)); X = rng.normal(0, 1, (n, 3)); V = X @ np.asarray(beta)
        p = np.exp(V) / np.exp(V).sum(); w = rng.choice(n, p=p)
        for i in range(n):
            rows.append({"race": r, "race_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=r // 10), "x1": X[i, 0], "x2": X[i, 1],
                         "x3": X[i, 2], "won": float(i == w), "pi": p[i]})
    return pd.DataFrame(rows)


def test_conditional_logit_recovers_coefficients():
    d = _pl_races()
    m = ConditionalLogit(l2=0.1).fit(d[["x1", "x2", "x3"]].values, d["won"].values, d["race"].values)
    assert np.allclose(m.coef_, [1.0, -0.5, 0.3], atol=0.12)
    p = m.predict_proba(d[["x1", "x2", "x3"]].values, d["race"].values)
    assert np.allclose(pd.Series(p).groupby(d["race"]).sum(), 1.0)
    assert mcfadden_r2(p, d["won"].values, d["race"].values) > 0.15


def test_delta_r2_and_stage_c():
    d = _pl_races()
    m = ConditionalLogit(l2=0.1).fit(d[["x1", "x2", "x3"]].values, d["won"].values, d["race"].values)
    f = m.predict_proba(d[["x1", "x2", "x3"]].values, d["race"].values)
    noisy = race_softmax(np.log(d["pi"].values) + np.random.default_rng(1).normal(0, 0.5, len(d)), d["race"].values)
    dr = delta_r2(f, noisy, d["won"].values, d["race"].values, n_boot=100)
    assert dr["delta_r2"] > 0 and dr["ci"][0] > 0
    c = StageC().fit(f, noisy, d["won"].values, d["race"].values)
    assert c.alpha_ > c.gamma_
    exact = delta_r2(d["pi"].values, d["pi"].values, d["won"].values, d["race"].values, n_boot=50)
    assert abs(exact["delta_r2"]) < 1e-12


def test_temperature_scaler_recovers_temperature():
    d = _pl_races(800)
    p = d["pi"].values ** 2.0; p = p / pd.Series(p).groupby(d["race"]).transform("sum").values
    ts = TemperatureScaler().fit(p, d["won"].values, d["race"].values)
    assert abs(ts.T_ - 2.0) < 0.3
    assert np.allclose(pd.Series(ts.transform(p, d["race"].values)).groupby(d["race"]).sum(), 1.0)


def test_walk_forward_and_conditional_calibration():
    d = _pl_races(1200)
    wf = walk_forward_stage_f(d, ["x1", "x2", "x3"], "won", "race", "race_date", market_col="pi", n_folds=4, min_train_races=100)
    assert wf["f_oof"].notna().sum() > 0 and wf["c_oof"].notna().sum() > 0
    ok = wf["c_oof"].notna()
    t = conditional_calibration(wf.loc[ok, "f_oof"], wf.loc[ok, "pi"], wf.loc[ok, "won"], wf.loc[ok, "race"])
    assert {"model > market", "model <= market"} == set(t["side"])


def test_lgb_objective_shapes():
    d = _pl_races(50).sort_values("race")
    sizes = d.groupby("race").size().values
    fobj, feval = lgb_race_softmax(sizes)

    class DS:
        def get_label(self):
            return d["won"].values

    g, h = fobj(np.zeros(len(d)), DS())
    assert g.shape == (len(d),) and (h > 0).all()
    assert abs(pd.Series(g).groupby(d["race"].values).sum()).max() < 1e-9
    name, val, hib = feval(np.zeros(len(d)), DS())
    assert name == "race_logloss" and val > 0 and hib is False


def _drifting_horses(seed=0):
    rng = np.random.default_rng(seed); rows = []
    for hname in range(300):
        theta = 80 + rng.normal(0, 8); day = 0
        for k in range(int(rng.integers(3, 14))):
            gap = int(rng.integers(14, 120)); day += gap; theta += rng.normal(0, np.sqrt(20 * gap / 365))
            rows.append({"horse_name": f"h{hname}", "race_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day), "race_time": "1.00",
                         "raceid": f"r{k}_{hname % 31}", "perf": theta + rng.normal(0, np.sqrt(40))})
    return pd.DataFrame(rows)


def test_kalman_fit_and_features_are_lag_safe():
    h = _drifting_horses()
    fit = fit_kalman_params(h, "perf")
    assert 8 < fit["q"] < 50 and 20 < fit["r"] < 80
    hk = add_kalman_features(h, "perf", q=fit["q"], r=fit["r"], init_var=fit["init_var"])
    first = hk.sort_values(["horse_name", "race_date"]).groupby("horse_name").head(1)
    assert (first["kf_n"] == 0).all() and first["kf_sd"].nunique() == 1        # nothing known before the first run
    ok = hk["kf_n"] >= 1
    mse_kf = np.mean((hk["perf"] - hk["kf_rating"])[ok] ** 2); mse_ewm = np.mean((hk["perf"] - ewm_baseline(h, "perf"))[ok] ** 2)
    assert mse_kf < mse_ewm
    assert {"kf_z", "kf_rank", "kf_vs_max"} <= set(hk.columns)


def test_primitives_match_document_values():
    assert abs(float(nmfp(16, 1)) - 0.5423) < 1e-3 and abs(float(nmfp(5, 1)) - 0.4714) < 1e-3
    assert abs(float(prb2_par(8)) - 0.3571) < 1e-3 and abs(float(fss_cred(2)) - 0.5774) < 1e-3
    assert abs(float(nmfp(8, 3)) + float(nmfp(8, 6))) < 1e-12   # symmetric


def test_run_primitives_and_par():
    r = pd.DataFrame({"placing_numerical": [1, 2, 3, 4, 5, 6], "number_of_runners": 6, "total_dst_bt": ["0", "1", "2.5", "4", "8", "25"],
                      "dist_furlongs": 8.0, "comptime_numeric": 100.0, "comment": ["", "", "", "", "", "tailed off"], "race_type": "Handicap",
                      "horse_name": list("abcdef"), "race_date": "2025-01-01", "race_time": "1.00", "raceid": "R"})
    pr = add_run_primitives(r)
    assert pr["prb2_fsa"].iloc[0] > 0 > pr["prb2_fsa"].iloc[-1] and pr["bl_censored"].iloc[-1] == 1 and np.isnan(pr["dabl"].iloc[-1])
    assert pr["bl_trunc"].iloc[-1] < 25 and pr["plc_fsa"].iloc[0] > 0
    par = BeatenLengthsPar().fit(pr)
    t = par.transform(pr)
    assert np.allclose(t["bl_vs_par"].dropna(), 0.0)


def test_aggregation_lag_safe_and_context_features():
    rows = []
    for h in range(20):
        for k in range(6):
            rows.append({"horse_name": f"h{h}", "race_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=30 * k), "race_time": "1.00",
                         "raceid": f"r{k}", "nmfp": 0.1 * (h % 5) + 0.01 * k, "perf_lbs": 70 + h + k, "timefigure": 60 + h, "fss_cred": 0.8,
                         "official_rating": 75 + h})
    d = pd.DataFrame(rows)
    a = aggregate_history(d, ["nmfp", "perf_lbs", "timefigure"], windows=(3, 5))
    first = a.sort_values(["horse_name", "race_date"]).groupby("horse_name").head(1)
    assert first["nmfp_mean_L3"].isna().all() and (first["runs_count"] == 0).all()
    a = lto_contradiction_features(a, tfig_col="timefigure", perf_col="perf_lbs")
    assert "lto_vs_career" in a.columns and "lto_tfig_above_ability" in a.columns
    s = strength_of_schedule(d, "official_rating")
    assert s["fss_qual"].notna().all() and "sos_vs_today" in s.columns
    ne = n_eff_from_ratings(d, "official_rating")
    assert (ne > 1).all() and (ne <= 20 + 1e-9).all()
    w = within_race_transforms(d, ["official_rating"])
    assert abs(w.groupby("raceid")["official_rating_z"].mean()).max() < 1e-9 and w["official_rating_vs_max"].max() <= 0


def test_ordering_probabilities_and_fit():
    p1, p2, p3 = top3_probabilities(np.array([0.5, 0.3, 0.2]), 0.8, 0.65)
    assert abs(p1.sum() - 1) < 1e-9 and abs(p2.sum() - 1) < 1e-9 and abs(p3.sum() - 1) < 1e-9
    h1, h2, _ = top3_probabilities(np.array([0.5, 0.3, 0.2]))            # Harville
    assert h2[2] < p2[2]                                                 # flattening lifts the outsider's 2nd-place chance
    assert np.allclose(place_probabilities(np.array([0.5, 0.3, 0.2]), 2, 0.8, 0.65), p1 + p2)
    rng = np.random.default_rng(3); races = []
    for _ in range(800):
        n = int(rng.integers(6, 11)); p = rng.dirichlet(np.ones(n) * 0.7)
        s = p ** 0.8 / (p ** 0.8).sum(); a = rng.choice(n, p=p); s2 = s.copy(); s2[a] = 0; b = rng.choice(n, p=s2 / s2.sum())
        t = p ** 0.65 / (p ** 0.65).sum(); t3 = t.copy(); t3[[a, b]] = 0; c = rng.choice(n, p=t3 / t3.sum())
        races.append((p, (a, b, c)))
    fit = fit_ordering_params(races)
    assert abs(fit["gamma"] - 0.8) < 0.12 and abs(fit["delta"] - 0.65) < 0.15 and fit["nll"] < fit["nll_harville"]


def test_feature_registry_enforces_p5():
    assert stage_of("ORR2") == "C" and stage_of("rPFD5") == "C" and stage_of("LR_mkt_steam") == "C" and stage_of("win_surprise") == "C"
    assert stage_of("preracehorsecareerNFP") == "F" and stage_of("kf_z") == "F" and stage_of("number_of_runners") == "S"
    cols = ["ORR2", "NFP", "OFS3", "dist_furlongs", "tf_master_pre"]
    assert stage_f_columns(cols) == ["NFP", "tf_master_pre"] and classify(cols)["stage"].tolist() == ["C", "F", "C", "S", "F"]
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"raceid": np.repeat(np.arange(100), 8), "bfsp": rng.uniform(2, 30, 800)})
    d["taut"] = -np.log(d["bfsp"]) + rng.normal(0, 0.01, 800); d["clean"] = rng.normal(0, 1, 800)
    au = leakage_audit(d, ["taut", "clean"]).set_index("feature")
    assert bool(au.loc["taut", "flag"]) and not bool(au.loc["clean", "flag"])


def test_phase1_prepare_handles_feed_sentinels():
    from model.phase1 import prepare_blandford_frame
    rows = []
    for r in range(2):
        for i in range(5):
            rows.append({"meeting_date": "2025-03-01", "course_bf": "KEMPTON PARK", "race_number": r + 1, "race_time": "14:30",
                         "race_type": "Flat", "n_runners": 5, "position": i + 1, "horse_name": f"H{r}{i}", "betfair_win_sp": 2.0 + i,
                         "performance_rating": 0 if i == 4 else 60 + i, "pre_race_master_rating": 999 if i == 0 else 70 + i,
                         "pre_race_adjusted_rating": 75, "timefigure": 50, "draw": i + 1, "age": 4})
    d = prepare_blandford_frame(pd.DataFrame(rows))
    assert len(d) == 10 and d["won"].sum() == 2
    assert d["pre_race_master_rating"].max() < 900 and d["performance_rating"].isna().sum() == 2
    assert np.allclose(d.groupby("raceid")["pi_market"].sum(), 1.0)
