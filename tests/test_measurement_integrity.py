"""Tests for the measurement-integrity layer of the racing² framework:

    purge and embargo around every walk-forward fold boundary (VI.2.3)
    the unratable-runner rule and race skip (IV.1)
    deflated Sharpe ratio and probability of backtest overfitting (VI.2.6)
    the Part 9 feature dictionary and its validator

Synthetic frames throughout — these test the guards, not the edge.
"""

import numpy as np
import pandas as pd
import pytest

import train_stage_f as tsf
from model import uncertainty as unc
from model.feature_registry import AS_OF, REQUIRED_FIELDS, describe, feature_dictionary, validate_dictionary
from model.phase1 import STAGE_F_FEATURES
from model.stage_f import apply_unratable_rule, purge_embargo_masks

DAY = pd.Timedelta(days=1)


def _daily_races(n_days: int = 150, per_day: int = 6, seed: int = 0, thin_every: int = 0) -> pd.DataFrame:
    """A frame shaped like the Stage F input: one race id per race, a winner,
    a market vector, three informative features and a runs_count."""
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(n_days):
        for r in range(per_day):
            n = int(rng.integers(6, 11))
            X = rng.normal(0, 1, (n, 3))
            V = X @ np.array([1.0, -0.5, 0.3])
            p = np.exp(V) / np.exp(V).sum()
            w = rng.choice(n, p=p)
            pi = p * rng.uniform(0.7, 1.3, n)
            pi = pi / pi.sum()
            for i in range(n):
                rows.append({"raceid": f"d{day}_r{r}", "race_date": pd.Timestamp("2025-01-01") + day * DAY,
                             "race_time": "13:00", "horse_name": f"d{day}_r{r}_h{i}", "x1": X[i, 0], "x2": X[i, 1],
                             "x3": X[i, 2], "won": float(i == w), "pi_market": pi[i],
                             "runs_count": 0.0 if (thin_every and i % thin_every == 0) else 12.0})
    return pd.DataFrame(rows)


FEATS = ["x1", "x2", "x3"]


# ---------------------------------------------------------------------------
# O6 / O7 — purge and embargo
# ---------------------------------------------------------------------------

def test_purge_and_embargo_cut_at_the_exact_boundary():
    dates = pd.Series(pd.date_range("2025-01-01", periods=120))
    start = pd.Timestamp("2025-03-01")
    test_dates = pd.date_range(start, periods=30).values
    tr, te = purge_embargo_masks(dates, test_dates, gap_days=30, embargo_days=7)

    assert tr[(dates == start - 31 * DAY).values][0]            # just outside the purge window: kept
    assert not tr[(dates == start - 30 * DAY).values][0]        # first purged day
    assert not tr[(dates == start - 1 * DAY).values][0]         # last purged day
    assert not tr[((dates >= start - 30 * DAY) & (dates < start)).values].any()
    assert tr[(dates < start - 30 * DAY).values].all()

    assert not te[((dates >= start) & (dates < start + 7 * DAY)).values].any()
    assert te[(dates == start + 7 * DAY).values][0]
    assert not te[(dates < start).values].any()                 # never trains and tests on the same row


def test_purge_severs_the_trailing_window_link():
    """A 30-day trailing feature on a test row is a function of the labels of
    every row in the 30 days before it. With gap_days=30 not one of those rows
    survives in the training set; with gap_days=0 they all do."""
    dates = pd.Series(pd.date_range("2025-01-01", periods=120))
    test_dates = pd.date_range("2025-03-01", periods=30).values
    window = 30 * DAY
    leaked = {}
    for gap in (0, 30):
        tr, te = purge_embargo_masks(dates, test_dates, gap_days=gap, embargo_days=0)
        first_test = dates[te].min()
        feeds_the_window = ((dates >= first_test - window) & (dates < first_test)).values
        leaked[gap] = int((tr & feeds_the_window).sum())
    assert leaked[0] == 30 and leaked[30] == 0


def test_walk_forward_never_fits_on_purged_rows(monkeypatch):
    """Spy on the estimator: whatever the fold layout, no training race is
    within gap_days of the earliest test race actually scored."""
    d = _daily_races(n_days=150)
    real = tsf.fit_predict_clogit
    seen = []

    def spy(Xtr, ytr, gtr, Xte, gte, **kw):
        seen.append((np.unique(gtr), np.unique(gte)))
        return real(Xtr, ytr, gtr, Xte, gte, **kw)

    monkeypatch.setattr(tsf, "fit_predict_clogit", spy)
    when = d.drop_duplicates("raceid").set_index("raceid")["race_date"]

    def gaps(**kw):
        seen.clear()
        tsf.walk_forward(d, FEATS, "clogit", test_start="2025-03-01", n_folds=3, l2=0.5, rounds=10,
                         min_train_races=50, gate=None, **kw)
        return [when[te].min() - when[tr].max() for tr, te in seen]

    assert len(seen) == 0
    tight = gaps(gap_days=0, embargo_days=0)
    assert tight and all(g == DAY for g in tight)               # contiguous without the guards
    wide = gaps(gap_days=30, embargo_days=7)
    assert wide and all(g >= 38 * DAY for g in wide)            # purge + embargo + the boundary day


def test_embargoed_test_rows_get_no_prediction():
    d = _daily_races(n_days=150)
    out, _ = tsf.walk_forward(d, FEATS, "clogit", test_start="2025-03-01", n_folds=3, l2=0.5, rounds=10,
                              min_train_races=50, gap_days=30, embargo_days=7, gate=None)
    scored = out.loc[out["f_oof"].notna(), "race_date"]
    assert len(scored) > 0
    # the first 7 days of the test period are the embargo of fold 0
    assert not ((scored >= pd.Timestamp("2025-03-01")) & (scored < pd.Timestamp("2025-03-08"))).any()
    assert (scored >= pd.Timestamp("2025-03-08")).any()


def test_walk_forward_records_what_it_dropped():
    d = _daily_races(n_days=150)
    out, _ = tsf.walk_forward(d, FEATS, "clogit", test_start="2025-03-01", n_folds=3, l2=0.5, rounds=10,
                              min_train_races=50, gap_days=30, embargo_days=7, gate=None)
    v = out.attrs["validation"]
    assert v["gap_days"] == 30 and v["embargo_days"] == 7
    assert v["train_rows_purged"] > 0 and v["test_rows_embargoed"] > 0


# ---------------------------------------------------------------------------
# M6 — the unratable-runner rule
# ---------------------------------------------------------------------------

def test_unratable_runner_takes_the_market_price_and_the_rest_renormalise():
    p = np.array([0.5, 0.3, 0.2])
    pi = np.array([0.40, 0.35, 0.25])
    g = np.array(["r1"] * 3)
    out, ratable, ok = apply_unratable_rule(p, pi, g, runs_count=np.array([10.0, 10.0, 0.0]), min_runs=3)

    assert list(ratable) == [True, True, False]
    assert ok.all()
    assert out[2] == pytest.approx(0.25)                        # the market's own number, not a guess
    assert out[0] == pytest.approx(0.5 * 0.75 / 0.8)            # (1 - pi_unratable) / sum p_ratable
    assert out.sum() == pytest.approx(1.0)
    assert out[0] / out[1] == pytest.approx(p[0] / p[1])        # ordering among the rated is untouched


def test_race_skipped_when_too_few_runners_are_ratable():
    p = np.array([0.5, 0.3, 0.2])
    pi = np.array([0.40, 0.35, 0.25])
    g = np.array(["r1"] * 3)
    thin = np.array([10.0, 0.0, 0.0])
    kept, _, ok = apply_unratable_rule(p, pi, g, runs_count=thin, min_runs=3, min_ratable=1)
    assert ok.all() and np.isfinite(kept).all()
    skipped, _, ok2 = apply_unratable_rule(p, pi, g, runs_count=thin, min_runs=3, min_ratable=2)
    assert not ok2.any() and np.isnan(skipped).all()            # one rated runner is just the market again
    nothing, _, ok3 = apply_unratable_rule(p, pi, g, runs_count=np.zeros(3), min_runs=3, min_ratable=1)
    assert not ok3.any() and np.isnan(nothing).all()            # Benter's literal rule


def test_kalman_observation_count_gates_independently():
    p = np.array([0.4, 0.35, 0.25])
    pi = np.array([0.30, 0.45, 0.25])
    g = np.array(["r1"] * 3)
    _, ratable, _ = apply_unratable_rule(p, pi, g, runs_count=np.full(3, 20.0), kf_n=np.array([5.0, 0.0, 3.0]), min_kf_n=1)
    assert list(ratable) == [True, False, True]                 # plenty of runs, but the filter never read one


def test_race_without_a_price_for_an_unratable_runner_is_skipped():
    p = np.array([0.5, 0.3, 0.2])
    pi = np.array([0.40, 0.35, np.nan])
    g = np.array(["r1"] * 3)
    out, _, ok = apply_unratable_rule(p, pi, g, runs_count=np.array([10.0, 10.0, 0.0]), min_runs=3)
    assert not ok.any() and np.isnan(out).all()


def test_gate_leaves_a_fully_ratable_race_alone():
    p = np.array([0.5, 0.3, 0.2, 0.6, 0.4])
    pi = np.array([0.4, 0.35, 0.25, 0.5, 0.5])
    g = np.array(["r1", "r1", "r1", "r2", "r2"])
    out, ratable, ok = apply_unratable_rule(p, pi, g, runs_count=np.full(5, 20.0), min_runs=3)
    assert ratable.all() and ok.all()
    assert out == pytest.approx(p)


def test_walk_forward_gate_substitutes_the_market_for_thin_runners():
    d = _daily_races(n_days=150, thin_every=3)                  # every third runner is a debutant
    out, _ = tsf.walk_forward(d, FEATS, "clogit", test_start="2025-03-01", n_folds=3, l2=0.5, rounds=10,
                              min_train_races=50, gap_days=30, embargo_days=7,
                              gate={"min_runs": 3, "min_kf_n": 0, "min_ratable": 2})
    scored = out[out["f_oof"].notna()]
    assert len(scored) > 0
    thin = scored[scored["runs_count"] < 3]
    assert len(thin) > 0
    assert np.allclose(thin["f_oof"], thin["pi_market"])
    assert np.allclose(scored.groupby("raceid")["f_oof"].sum(), 1.0)
    assert out.attrs["validation"]["runners_unratable"] == len(thin)


def test_walk_forward_without_the_gate_rates_everyone():
    d = _daily_races(n_days=150, thin_every=3)
    out, _ = tsf.walk_forward(d, FEATS, "clogit", test_start="2025-03-01", n_folds=3, l2=0.5, rounds=10,
                              min_train_races=50, gap_days=30, embargo_days=7, gate=None)
    scored = out[out["f_oof"].notna()]
    thin = scored[scored["runs_count"] < 3]
    assert len(thin) > 0 and not np.allclose(thin["f_oof"], thin["pi_market"])


# ---------------------------------------------------------------------------
# O9 — deflated Sharpe ratio
# ---------------------------------------------------------------------------

def test_probabilistic_sharpe_ratio_matches_hand_computation():
    # Phi(0.1 * sqrt(999) / sqrt(1 - 0*0.1 + (3-1)/4 * 0.1^2)) = Phi(3.15282...)
    assert unc.probabilistic_sharpe_ratio(sr=0.1, n=1000, skew=0.0, kurtosis=3.0) == pytest.approx(0.9991915033756518, abs=1e-12)


def test_expected_max_sharpe_matches_hand_computation():
    # sqrt(0.01) * [(1 - 0.5772156649) Z^-1(0.99) + 0.5772156649 Z^-1(1 - 1/(100e))]
    assert unc.expected_max_sharpe(100, 0.01) == pytest.approx(0.2530602893201685, abs=1e-12)
    assert unc.expected_max_sharpe(1, 0.01) == 0.0              # a single trial deflates by nothing
    assert unc.expected_max_sharpe(100, 0.0) == 0.0


def test_deflated_sharpe_matches_hand_computation():
    res = unc.deflated_sharpe_ratio(n_trials=100, sr_variance=0.01, sr=0.1, n=1000, skew=0.0, kurtosis=3.0)
    assert res["sr_star"] == pytest.approx(0.2530602893201685, abs=1e-12)
    assert res["dsr"] == pytest.approx(6.974870890894661e-07, abs=1e-15)
    assert res["psr_vs_zero"] == pytest.approx(0.9991915033756518, abs=1e-12)


def test_deflation_bites_harder_the_more_configurations_were_tried():
    kw = dict(sr=0.25, n=1500, skew=0.0, kurtosis=3.0, sr_variance=0.01)
    dsrs = [unc.deflated_sharpe_ratio(n_trials=n, **kw)["dsr"] for n in (2, 10, 100, 1000)]
    assert dsrs == sorted(dsrs, reverse=True)
    assert dsrs[0] > 0.9 and dsrs[-1] < 0.1


def test_negative_skew_and_fat_tails_discount_the_same_sharpe():
    base = unc.probabilistic_sharpe_ratio(sr=0.1, n=1000, skew=0.0, kurtosis=3.0)
    assert unc.probabilistic_sharpe_ratio(sr=0.1, n=1000, skew=-1.0, kurtosis=3.0) < base
    assert unc.probabilistic_sharpe_ratio(sr=0.1, n=1000, skew=0.0, kurtosis=9.0) < base


def test_deflated_sharpe_from_a_return_series():
    rng = np.random.default_rng(3)
    r = rng.normal(0.02, 1.0, 2000)
    res = unc.deflated_sharpe_ratio(r, trial_sharpes=rng.normal(0, 0.05, 40))
    assert res["n_obs"] == 2000 and res["n_trials"] == 40
    assert res["sharpe"] == pytest.approx(unc.sharpe_ratio(r))
    assert res["sr_star"] > 0 and 0.0 <= res["dsr"] <= 1.0


def test_configuration_ledger_counts_and_persists(tmp_path):
    rng = np.random.default_rng(4)
    path = tmp_path / "trials.json"
    led = unc.ConfigurationLedger(path)
    for i in range(12):
        led.record({"l2": 2.0 ** i}, returns=rng.normal(0.01 * i, 1.0, 1500))
    assert led.n_trials == 12
    assert np.isfinite(led.sharpes()).all()
    assert led.best() is led.trials[int(np.argmax(led.sharpes()))]

    reloaded = unc.ConfigurationLedger(path)
    assert reloaded.n_trials == 12                              # the count survives the session
    assert reloaded.best()["config"] == led.best()["config"]
    res = reloaded.deflated_sharpe()
    assert res["n_trials"] == 12 and res["sr_star"] > 0
    assert len(reloaded.to_frame()) == 12

    # deflation is what the count is for: pretending one thing was tried flatters it
    honest = res["dsr"]
    flattering = unc.deflated_sharpe_ratio(sr=res["sharpe"], n=res["n_obs"], skew=res["skew"],
                                           kurtosis=res["kurtosis"], n_trials=1, sr_variance=res["sr_variance"])["dsr"]
    assert flattering > honest and honest < 0.999


# ---------------------------------------------------------------------------
# O10 — probability of backtest overfitting
# ---------------------------------------------------------------------------

def _seesaw(n_per_block: int = 200) -> np.ndarray:
    """Two configurations, each good in one half of the sample and bad in the
    other: whichever wins in sample must lose out of sample."""
    a = np.r_[np.full(n_per_block, 1.0), np.full(n_per_block, -1.0)]
    return np.column_stack([a, -a]) + 1e-9


def test_pbo_is_one_when_in_sample_selection_is_anti_predictive():
    res = unc.probability_of_backtest_overfitting(_seesaw(), n_splits=2)
    assert res["pbo"] == 1.0
    assert res["n_combinations"] == 2 and res["n_configs"] == 2
    assert (res["logits"] <= 0).all()


def test_pbo_near_a_half_for_pure_noise():
    rng = np.random.default_rng(5)
    res = unc.probability_of_backtest_overfitting(rng.normal(0, 1, (2000, 30)), n_splits=10)
    assert 0.3 < res["pbo"] < 0.7                               # selection is a coin flip
    assert res["degradation_slope"] < 0.5                       # and does not carry out of sample


def test_pbo_low_when_one_configuration_is_genuinely_better():
    rng = np.random.default_rng(6)
    M = rng.normal(0, 1, (2000, 30))
    M[:, 11] += 0.25
    res = unc.probability_of_backtest_overfitting(M, n_splits=10)
    assert res["pbo"] < 0.1
    assert res["prob_oos_loss"] < 0.1


def test_pbo_accepts_a_dataframe_and_rejects_bad_shapes():
    rng = np.random.default_rng(7)
    df = pd.DataFrame(rng.normal(0, 1, (600, 6)), columns=[f"cfg{i}" for i in range(6)])
    assert 0.0 <= unc.probability_of_backtest_overfitting(df, n_splits=6)["pbo"] <= 1.0
    with pytest.raises(ValueError):
        unc.probability_of_backtest_overfitting(df.iloc[:, :1], n_splits=6)
    with pytest.raises(ValueError):
        unc.probability_of_backtest_overfitting(df, n_splits=5)
    with pytest.raises(ValueError):
        unc.probability_of_backtest_overfitting(df.iloc[:10], n_splits=6)


# ---------------------------------------------------------------------------
# L1 — the Part 9 feature dictionary
# ---------------------------------------------------------------------------

def test_dictionary_entry_carries_every_part_9_field():
    s = describe("kf_sd")
    for field in REQUIRED_FIELDS:
        assert getattr(s, field), f"{field} is empty"
    assert s.stage == "F" and s.origin == "entry"
    assert "Kalman" in s.shrinkage and s.version


def test_transform_suffix_inherits_the_base_entry():
    """trainer_sr_shrunk_z is a within-race z *of a shrunk strike rate*: the
    shrinkage constant must survive the suffix."""
    base, z = describe("trainer_sr_shrunk"), describe("trainer_sr_shrunk_z")
    assert z.transform == "z" and base.transform == "raw"
    assert z.shrinkage == base.shrinkage and "k = 30" in z.shrinkage
    assert z.as_of_rule == base.as_of_rule
    assert describe("kf_sd_z").definition == describe("kf_sd").definition


def test_validator_flags_a_feature_nobody_documented():
    bad = validate_dictionary(["kf_sd", "some_new_idea_v2"])
    assert list(bad["feature"]) == ["some_new_idea_v2"]
    missing = bad["missing_fields"].iloc[0].split(",")
    assert {"definition", "as_of_rule", "missing_policy", "version"} <= set(missing)
    assert bad["origin"].iloc[0] == "unregistered"


def test_validator_flags_post_race_and_wrong_stage():
    bad = validate_dictionary(["EPF", "placing_numerical", "bfsp", "kf_sd"], stage="F").set_index("feature")
    assert bad.loc["EPF", "post_race"] and bad.loc["placing_numerical", "post_race"]
    assert describe("EPF").as_of_rule == AS_OF["POST_RACE"]
    assert bad.loc["bfsp", "stage_mismatch"] and not bad.loc["bfsp", "post_race"]
    assert "kf_sd" not in bad.index


def test_the_stage_f_block_actually_shipped_is_fully_documented():
    """The Phase 0 exit test: the matrix train_stage_f.py fits on has a
    dictionary entry for every column."""
    connections = [f"{e}_{m}" for e in ("trainer", "jockey") for m in
                   ("runs", "sr_shrunk", "nmfp_shrunk", "form_14d", "form_30d", "form_vs_career")]
    extras = ["tj_runs", "tj_sr_shrunk", "jockey_booking_upgrade", "dist_apt", "surf_apt", "is_female",
              "draw_x_lnN", "draw_x_sprint", "kf_sd_x_runs", "trainer_sr_shrunk_z", "jockey_sr_shrunk_z",
              "tj_sr_shrunk_z", "trainer_form_30d_z", "kf_sd_z"]
    cols = list(STAGE_F_FEATURES) + connections + extras
    bad = validate_dictionary(cols, stage="F")
    assert bad.empty, f"undocumented Stage F features: {list(bad['feature'])}"
    table = feature_dictionary(cols)
    assert len(table) == len(cols)
    assert set(REQUIRED_FIELDS) <= set(table.columns)
    assert (table["version"] != "").all()
