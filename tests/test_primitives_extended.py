"""Tests for the Section III Part 1-3 primitives added to model.primitives:
beaten-lengths cluster geometry, the field-size par surfaces, the within-race
margin normalisations, PvS, and the Part 3 aggregation machinery.

Every test here is meant to fail if the implementation is wrong rather than
merely absent: the geometry is checked against a hand-computed finish, PvS
against a planted confound that makes the naive career mean rank horses
backwards, and the connection features against a same-race stablemate leak.
"""

import numpy as np
import pandas as pd
import pytest

from model.lagsafe import race_lagged_expanding_mean
from model.perf_figures import MARGIN_WORDS, parse_beaten_lengths, winning_margin
from model.primitives import (MARGIN_AGGREGATE_FEATURES, MARGIN_NORMALISATIONS, POST_RACE_PRIMITIVES,
                              BeatenLengthsPar, PerformanceVsSchedule, add_dslr_features, add_lto_context_features,
                              add_margin_aggregates, add_margin_geometry, add_margin_normalisations,
                              add_pvs_features, add_run_primitives, aggregate_history, connection_plc_fsa,
                              dist_adjust_bl, harmonic_bl_par, lps_scale, n_eff_from_probs,
                              proportion_beaten_figure, race_strength, skew_aware_transform,
                              strength_of_schedule, within_race_transforms)


# ---------------------------------------------------------------------------
# §1.5.a cluster geometry
# ---------------------------------------------------------------------------

def _one_race(margins, positions=None, n=None, **cols):
    n = n or len(margins)
    d = pd.DataFrame({
        "raceid": "R1", "race_date": pd.Timestamp("2025-04-01"), "race_time": "14.30", "track": "Ascot",
        "horse_name": [f"h{i}" for i in range(len(margins))],
        "placing_numerical": positions if positions is not None else list(range(1, len(margins) + 1)),
        "number_of_runners": n, "total_dst_bt": margins, "dist_furlongs": 8.0, "comptime_numeric": 100.0,
        "going_description": "Good", "race_class": 4, "race_type": "Handicap", "race_code": "Flat", "comment": "",
    })
    for k, v in cols.items():
        d[k] = v
    return d


def test_cluster_geometry_hand_computed():
    """Five finishers, margins published cumulatively as 0, hd, 2, 2¼, 9.

    The horse in fourth is the point of the feature: beaten 2¼ in total but
    only a quarter of a length off the third and six and three-quarters clear
    of the fifth, so it finished with the principals rather than in the ruck.
    """
    d = _one_race(["0", "hd", "2", "2¼", "9"])
    g = add_margin_geometry(d)

    hd = MARGIN_WORDS["hd"]
    assert g["bl_winner"].tolist() == pytest.approx([0.0, hd, 2.0, 2.25, 9.0])
    assert g["bl_won_by"].tolist() == pytest.approx([hd] * 5)          # race-level
    assert g["bl_signed"].tolist() == pytest.approx([hd, -hd, -2.0, -2.25, -9.0])
    assert g["bl_next"].tolist() == pytest.approx([0.0, hd, 2.0 - hd, 0.25, 6.75])
    assert g["bl_behind"].tolist() == pytest.approx([hd, 2.0 - hd, 0.25, 6.75, 0.0])
    assert g["bl_isolation"].tolist() == pytest.approx([hd, 2.0 - 2 * hd, 0.25 - (2.0 - hd), 6.5, -6.75])
    assert g["bl_to_last"].tolist() == pytest.approx([9.0, 9.0 - hd, 7.0, 6.75, 0.0])
    assert g["bl_spread"].tolist() == pytest.approx([9.0] * 5)

    # the shape of the finish, not the margin: fourth is the least isolated
    # runner behind the winner even though two horses finished closer
    assert g["bl_isolation"].iloc[3] == max(g["bl_isolation"].iloc[1:])
    assert g["bl_next"].iloc[3] < g["bl_next"].iloc[2]                # closer to the horse in front than third was
    assert g["bl_gap_ahead"].equals(g["bl_next"]) and g["bl_gap_behind"].equals(g["bl_behind"])


def test_bl_signed_is_continuous_through_the_winner():
    """Without bl_signed the winner is a structural zero and an eight-length
    win is indistinguishable from a nose."""
    narrow = add_margin_geometry(_one_race(["0", "nse", "3"]))
    wide = add_margin_geometry(_one_race(["0", "8", "11"]))
    assert narrow["bl_signed"].iloc[0] == pytest.approx(MARGIN_WORDS["nse"])
    assert wide["bl_signed"].iloc[0] == pytest.approx(8.0)
    assert wide["bl_signed"].iloc[0] > narrow["bl_signed"].iloc[0] > 0 > narrow["bl_signed"].iloc[1]


def test_dead_heat_and_non_finisher_handling():
    d = _one_race(["0", "0", "2", ""], positions=[1, 1, 3, np.nan], n=4)
    g = add_margin_geometry(d)
    assert (g["bl_won_by"] == 0).all()                                # a dead-heated win is a margin of zero
    assert g["is_dead_heat"].tolist() == [1, 1, 0, 0]
    assert g.loc[:1, "bl_next"].tolist() == [0.0, 0.0]                # both share the leading cluster
    assert g.loc[:1, "bl_behind"].tolist() == pytest.approx([2.0, 2.0])
    assert g["finished"].tolist() == [1, 1, 1, 0] and (g["n_finishers"] == 3).all()
    assert g.loc[3, ["bl_next", "bl_behind", "bl_isolation", "bl_signed"]].isna().all()
    assert winning_margin(d).tolist() == pytest.approx([0.0] * 4)


def test_geometry_is_registered_as_post_race():
    """Within-race geometry describes the race being predicted. It is only ever
    an input to a lagged feature, so it has to be in the post-race set."""
    for name in ("bl_next", "bl_behind", "bl_gap_ahead", "bl_isolation", "bl_signed", "bl_won_by",
                 "bl_vs_par", "bl_par_ratio", "won_by_vs_par", "pvs_run", *MARGIN_NORMALISATIONS):
        assert name in POST_RACE_PRIMITIVES
    # ... and the lagged features built from it are not
    assert not set(MARGIN_AGGREGATE_FEATURES) & POST_RACE_PRIMITIVES


# ---------------------------------------------------------------------------
# §1.5.b-c margin conversion
# ---------------------------------------------------------------------------

def test_lps_scale_and_proportion_figure():
    assert lps_scale("Good") == 6.0 and lps_scale("Heavy") == 5.0 and lps_scale("Good To Soft") == 5.5
    assert lps_scale("Standard") == 6.0                                # all-weather
    assert lps_scale("Heavy", "Handicap Chase") == 4.0                 # jumps column
    assert lps_scale("Good", "Novices' Hurdle") == 5.0
    assert lps_scale(["Good", "Heavy"]).tolist() == [6.0, 5.0]
    assert proportion_beaten_figure(100.0, 100.0, 2000.0) == pytest.approx(0.0)
    assert proportion_beaten_figure(100.0, 101.0, 1000.0) == pytest.approx((1 - 100 / 101) * 1000)
    assert np.isnan(proportion_beaten_figure(0.0, 100.0, 1600.0))


def test_run_primitives_carry_margin_time_and_flags():
    d = _one_race(["0", "2", "10"], comment=["", "hampered 2f out", "eased when held"])
    p = add_run_primitives(d, censor_policy="raw")
    assert p["lps"].tolist() == [6.0] * 3                              # good ground, flat
    assert p["bl_seconds"].tolist() == pytest.approx([0.0, 2 / 6, 10 / 6])
    assert p["bl_pct_wintime"].tolist() == pytest.approx([0.0, 2 / 6, 10 / 6])   # winning time 100s -> % == seconds
    assert p["hampered_flag"].tolist() == [0, 1, 0] and p["eased_flag"].tolist() == [0, 0, 1]
    assert p["bl_censored"].tolist() == [0, 1, 1]
    assert p["bl_prop_figure"].iloc[1] > 0 and p["bl_prop_figure"].iloc[2] > p["bl_prop_figure"].iloc[1]


def test_censoring_policies():
    d = _one_race(["0", "2", "40"], comment=["", "", "tailed off"])
    dist_m = 8.0 * 201.168
    missing = add_run_primitives(d, censor_policy="missing")
    ceiling = add_run_primitives(d, censor_policy="ceiling")
    raw = add_run_primitives(d, censor_policy="raw")
    assert np.isnan(missing["dabl"].iloc[2])                                        # option 3
    assert ceiling["dabl"].iloc[2] == pytest.approx(dist_adjust_bl(15.0, dist_m))    # option 2: truncation ceiling
    assert ceiling["bl_censored"].iloc[2] == 1                                       # ... carried as a feature
    assert raw["dabl"].iloc[2] == pytest.approx(dist_adjust_bl(15 * np.tanh(40 / 15), dist_m))
    with pytest.raises(ValueError, match="censor_policy"):
        add_run_primitives(d, censor_policy="nonsense")


# ---------------------------------------------------------------------------
# §1.5.d within-race normalisation
# ---------------------------------------------------------------------------

def test_within_race_margin_normalisations_hand_computed():
    d = _one_race(["0", "1", "2", "5"])
    d["dabl"] = [0.0, 1.0, 2.0, 5.0]                                   # bypass the distance adjustment
    out = add_margin_normalisations(d)
    v = np.array([0.0, 1.0, 2.0, 5.0])

    assert out["bl_vs_median"].tolist() == pytest.approx(v - 1.5)
    assert out["bl_share"].tolist() == pytest.approx(v / 8.0)
    assert out["bl_norm_spread"].tolist() == pytest.approx(v / 5.0)    # 0 = winner, 1 = last
    assert out["bl_vs_2nd"].tolist() == pytest.approx(v - 1.0)
    assert out["bl_z"].tolist() == pytest.approx((v - v.mean()) / v.std(ddof=1))
    assert out["bl_rankpct"].tolist() == pytest.approx([1.0, 2 / 3, 1 / 3, 0.0])
    assert set(MARGIN_NORMALISATIONS) <= set(out.columns)


def test_vs_2nd_falls_back_to_the_second_smallest_without_positions():
    d = pd.DataFrame({"raceid": ["R"] * 3, "x": [4.0, 1.0, 7.0]})
    out = within_race_transforms(d, ["x"], suffixes=("vs_2nd",))
    assert out["x_vs_2nd"].tolist() == pytest.approx([0.0, -3.0, 3.0])


# ---------------------------------------------------------------------------
# §1.5.e field-size par
# ---------------------------------------------------------------------------

def test_harmonic_par_matches_the_spec_table():
    cum = harmonic_bl_par([12] * 12, list(range(1, 13)))
    gaps = np.diff(np.concatenate([[0.0], cum]))
    assert gaps[1:].tolist() == pytest.approx(
        [0.091, 0.100, 0.111, 0.125, 0.143, 0.167, 0.200, 0.250, 0.333, 0.500, 1.000], abs=5e-4)
    assert harmonic_bl_par(12, 12, normalise=True) == pytest.approx(1.0)
    assert np.isnan(harmonic_bl_par(1, 1))                              # walkovers guard


def _par_frame(seed=0, n_races=400):
    rng = np.random.default_rng(seed); rows = []
    for r in range(n_races):
        n = int(rng.integers(5, 17))
        # margins grow with position, and the winning margin shrinks with field size
        gaps = rng.gamma(1.4, 1.0, n - 1) * (1.0 + np.arange(n - 1) / n)
        gaps[0] = rng.gamma(2.0, 3.0 / n)
        bl = np.concatenate([[0.0], np.cumsum(gaps)])
        for i in range(n):
            rows.append({"raceid": f"r{r}", "race_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=r),
                         "race_time": "14.00", "track": "Ascot", "horse_name": f"h{r}_{i}",
                         "placing_numerical": i + 1, "number_of_runners": n, "total_dst_bt": f"{bl[i]:.3f}",
                         "dist_furlongs": float(rng.choice([6.0, 8.0, 12.0])), "comptime_numeric": 100.0,
                         "going_description": str(rng.choice(["Good", "Soft"])), "race_class": int(rng.integers(2, 7)),
                         "race_type": "Handicap", "race_code": "Flat", "comment": ""})
    return add_run_primitives(pd.DataFrame(rows))


def test_par_surface_is_monotone_and_yields_the_three_new_ratios():
    p = _par_frame()
    par = BeatenLengthsPar().fit(p)
    t = par.transform(p)

    med = t.groupby("placing_numerical")["bl_par"].median().dropna()
    assert (med.diff().dropna() >= -1e-9).all()                        # monotone in finishing position
    assert t.loc[t["placing_numerical"] == 1, "bl_par"].abs().max() < 1e-9

    ok = t["bl_par"].notna() & t["dabl"].notna() & (t["placing_numerical"] > 1)
    s = t[ok]
    assert (s["bl_par_ratio"] * s["bl_par"]).tolist() == pytest.approx(s["dabl"].tolist())
    assert (s["bl_per_position"] * (s["placing_numerical"] - 1)).tolist() == pytest.approx(s["dabl"].tolist())
    assert t.loc[t["placing_numerical"] == 1, "bl_per_position"].isna().all()
    assert s["bl_vs_par"].tolist() == pytest.approx((s["bl_par"] - s["dabl"]).tolist())


def test_won_by_par_shrinks_with_field_size():
    p = _par_frame()
    par = BeatenLengthsPar().fit(p)
    t = par.transform(p)
    assert par.won_by_par([6])[0] > par.won_by_par([16])[0]            # bigger field, tighter finish
    w = t[t["placing_numerical"] == 1]
    assert w["won_by_vs_par"].tolist() == pytest.approx((w["bl_won_by"] - w["won_by_par"]).tolist())
    assert t.loc[t["placing_numerical"] > 1, "won_by_vs_par_win"].isna().all()

    # a three-length win in a big field beats a three-length win in a small one
    small = par.won_by_par([5])[0]; big = par.won_by_par([16])[0]
    assert (3.0 - big) > (3.0 - small)


def test_par_surface_shrinks_thin_context_cells_toward_the_pooled_base():
    p = _par_frame()
    thin = p[(p["going_description"] == "Soft") & (p["race_class"] == 6)]
    par = BeatenLengthsPar(k=1e9).fit(p)                               # infinite shrinkage == the pooled base
    heavy = par.transform(p)
    par_none = BeatenLengthsPar(k=0.0).fit(p)                          # no shrinkage == the raw cell median
    raw = par_none.transform(p)
    assert len(thin) > 0
    assert not np.allclose(heavy["bl_par"].dropna().values[:50], raw["bl_par"].dropna().values[:50])
    base = BeatenLengthsPar(context=False).fit(p).transform(p)
    assert heavy["bl_par"].dropna().tolist() == pytest.approx(base["bl_par"].dropna().tolist())


# ---------------------------------------------------------------------------
# §1.5.g aggregation and §3.2 aggregators
# ---------------------------------------------------------------------------

def test_margin_aggregates_exist_and_are_lagged():
    p = _par_frame(seed=1, n_races=120)
    # give the horses histories: reuse names across races
    p["horse_name"] = "h" + (p.groupby("raceid").cumcount()).astype(str)
    p = add_margin_normalisations(p)
    p = BeatenLengthsPar().fit(p).transform(p)
    out = add_margin_aggregates(p)
    assert set(MARGIN_AGGREGATE_FEATURES) <= set(out.columns)
    first = out.sort_values(["horse_name", "race_date"]).groupby("horse_name").head(1)
    for c in MARGIN_AGGREGATE_FEATURES:
        assert first[c].isna().all(), f"{c} is not lagged: it has a value on the horse's first run"


def test_aggregator_set_hand_computed():
    d = pd.DataFrame({"horse_name": ["h"] * 5, "race_time": "14.00",
                      "race_date": pd.date_range("2025-01-01", periods=5, freq="100D"),
                      "x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    a = aggregate_history(d, ["x"], windows=(3,), weight_col=None, iqm_min_window=3,
                          aggs=("mean", "max", "min", "iqm", "slope", "slope_date", "first", "above_par"), par=2.5)
    assert a["x_last"].tolist() == pytest.approx([np.nan, 1, 2, 3, 4], nan_ok=True)
    assert a["x_mean_L3"].tolist() == pytest.approx([np.nan, 1.0, 1.5, 2.0, 3.0], nan_ok=True)
    assert a["x_min_L3"].tolist() == pytest.approx([np.nan, 1, 1, 1, 2], nan_ok=True)
    assert a["x_first"].tolist() == pytest.approx([np.nan, 1, 1, 1, 1], nan_ok=True)
    assert a["x_slope_L3"].iloc[4] == pytest.approx(1.0)               # +1 per run
    assert a["x_slopedate_L3"].iloc[4] == pytest.approx(1.0 / 100)     # +1 per 100 days
    assert a["x_pct_above_par_L3"].tolist() == pytest.approx([np.nan, 0.0, 0.0, 1 / 3, 2 / 3], nan_ok=True)
    assert a["x_n_above_par_L3"].iloc[4] == pytest.approx(2.0)         # runs 3 and 4 beat par 2.5


def test_iqm_window_floor_is_respected():
    d = pd.DataFrame({"horse_name": ["h"] * 6, "race_time": "14.00",
                      "race_date": pd.date_range("2025-01-01", periods=6, freq="30D"), "x": np.arange(6.0)})
    default = aggregate_history(d, ["x"], windows=(3, 5), weight_col=None)
    opened = aggregate_history(d, ["x"], windows=(3, 5), weight_col=None, iqm_min_window=3)
    assert "x_iqm_L3" not in default.columns and "x_iqm_L3" in opened.columns


def test_dslr_and_season_counters():
    dates = ["2024-03-01", "2024-04-01", "2024-07-01", "2025-05-01"]
    d = pd.DataFrame({"horse_name": "h", "race_time": "14.00", "race_date": pd.to_datetime(dates)})
    out = add_dslr_features(d)
    assert out["dslr"].tolist() == pytest.approx([np.nan, 31, 91, 304], nan_ok=True)
    assert out["dslr_mean_career"].tolist() == pytest.approx([np.nan, np.nan, 31, 61], nan_ok=True)
    assert out["dslr_vs_horse_norm"].iloc[2] == pytest.approx(91 / 31)
    assert out["runs_this_season"].tolist() == [0, 1, 2, 0]            # new calendar year resets
    assert out["days_since_first_run"].tolist() == [0, 31, 122, 426]


def test_skew_aware_transform_picks_a_transform_and_rezscores():
    rng = np.random.default_rng(0)
    heavy = rng.lognormal(0, 1.4, 2000)
    v, kind = skew_aware_transform(heavy)
    assert kind in ("sqrt", "log1p")
    assert abs(np.nanmean(v)) < 1e-9 and np.nanstd(v, ddof=1) == pytest.approx(1.0)
    assert abs(pd.Series(v).skew()) < abs(pd.Series(heavy).skew())
    _, kind2 = skew_aware_transform(rng.normal(0, 1, 2000))
    assert kind2 == "none"


# ---------------------------------------------------------------------------
# §2.1-2.4 context, strength and PvS
# ---------------------------------------------------------------------------

def test_fss_qual_aggregators_are_leave_one_out():
    d = pd.DataFrame({"raceid": ["R"] * 5, "race_date": pd.Timestamp("2025-01-01"), "race_time": "14.00",
                      "horse_name": list("abcde"), "official_rating": [60.0, 70.0, 80.0, 90.0, 100.0]})
    s = strength_of_schedule(d, "official_rating", qual_aggs=("mean", "max", "iqm"))
    assert s["fss_qual_mean"].tolist() == pytest.approx([85.0, 82.5, 80.0, 77.5, 75.0])
    assert s["fss_qual_max"].tolist() == pytest.approx([100, 100, 100, 100, 90])   # the top horse sees the second
    # IQM of each runner's four opponents: the middle two of the four
    assert s["fss_qual_iqm"].tolist() == pytest.approx([85.0, 85.0, 80.0, 75.0, 75.0])
    assert s["race_rating_sd"].iloc[0] == pytest.approx(np.std([60, 70, 80, 90, 100], ddof=1))
    assert s["race_rating_iqr"].iloc[0] == pytest.approx(20.0)


def test_sos_delta_and_vs_today_are_lagged():
    rows = []
    for k, q in enumerate([60.0, 60.0, 60.0, 90.0]):                   # three weak races then a class rise
        rows += [{"raceid": f"r{k}", "race_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=30 * k),
                  "race_time": "14.00", "horse_name": h, "official_rating": q}
                 for h in ("target", "x", "y")]
    d = pd.DataFrame(rows)
    s = strength_of_schedule(d, "official_rating", window=3)
    t = s[s["horse_name"] == "target"].sort_values("race_date")
    assert np.isnan(t["sos_vs_today"].iloc[0])                         # nothing known before the first run
    assert t["sos_L3"].iloc[3] == pytest.approx(60.0)                  # the three prior runs only
    assert t["sos_vs_today"].iloc[3] == pytest.approx(30.0)            # stepping up 30lb
    assert t["sos_delta"].iloc[3] == pytest.approx(0.0)


def _pvs_frame(seed=7):
    """nmfp = horse effect − 0.1·(race strength − 70), with good horses sent to
    strong races. The confound is strong enough that the naive career mean of
    nmfp ranks the good horses *worst*; only the residual recovers them."""
    rng = np.random.default_rng(seed); rows = []
    effects = {f"h{i}": float(rng.choice([-0.3, 0.0, 0.3])) for i in range(150)}
    for i, (h, e) in enumerate(effects.items()):
        for k in range(10):
            q = 70.0 + 20.0 * e + rng.uniform(-10, 10)
            rows.append({"raceid": f"r{i}_{k}", "race_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=30 * k),
                         "race_time": "14.00", "horse_name": h, "fss_qual": q,
                         "nmfp": e - 0.1 * (q - 70.0), "effect": e})
    return pd.DataFrame(rows)


def test_pvs_recovers_ability_that_the_naive_career_mean_gets_backwards():
    d = _pvs_frame()
    out, model = add_pvs_features(d)
    out = aggregate_history(out, ["nmfp"], windows=(5,), weight_col=None, aggs=("mean",))
    last = out.sort_values(["horse_name", "race_date"]).groupby("horse_name").tail(1)
    eff = last["effect"].values

    raw = np.corrcoef(last["nmfp_mean_career"], eff)[0, 1]
    pvs = np.corrcoef(last["pvs_mean_career"], eff)[0, 1]
    assert raw < -0.7, "the planted confound should make the naive aggregate rank horses backwards"
    assert pvs > 0.8, "PvS should recover the horse effect from the residual"
    assert model.coef_[1] < 0 and model.n_ == len(d)
    assert out["pvs_run"].tolist() == pytest.approx((d["nmfp"] - out["nmfp_expected"]).tolist())


def test_pvs_aggregates_are_lag_safe_and_fit_window_is_respected():
    d = _pvs_frame(seed=8)
    out, _ = add_pvs_features(d)
    first = out.sort_values(["horse_name", "race_date"]).groupby("horse_name").head(1)
    assert first["pvs_mean_career"].isna().all() and first["lto_was_flattered"].isna().all()

    fitted = PerformanceVsSchedule().fit(d, mask=pd.to_datetime(d["race_date"]) < pd.Timestamp("2024-03-01"))
    _, m2 = add_pvs_features(d, fit_end="2024-03-01")
    assert m2.coef_.tolist() == pytest.approx(fitted.coef_.tolist())
    assert m2.n_ < len(d)


def test_race_strength_uses_matched_windows_and_pads_debutants():
    rng = np.random.default_rng(2); rows = []
    for r in range(60):
        for i in range(8):
            rows.append({"raceid": f"r{r}", "race_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=r),
                         "race_time": "14.00", "horse_name": f"h{i}", "number_of_runners": 8,
                         "placing_numerical": int(rng.integers(1, 9))})
    d = pd.DataFrame(rows)
    d["prb2"] = ((8 - d["placing_numerical"]) / 7) ** 2
    out = race_strength(d)
    debut = out.sort_values(["horse_name", "race_date"]).groupby("horse_name").head(1)
    assert debut["race_strength_runner"].tolist() == pytest.approx([0.3050] * 8)    # the research's debutant par
    assert out["race_strength"].groupby(out["raceid"]).nunique().max() == 1         # race-level, constant within race
    assert (out["race_strength_max"] >= out["race_strength"]).all()


def test_n_eff_from_probs_counts_contenders_not_bodies():
    p = pd.Series([0.75] + [0.25 / 15] * 15)
    key = ["R"] * 16
    assert n_eff_from_probs(p, key).iloc[0] < 4.0                       # 16 bodies, a handful of contenders
    assert n_eff_from_probs(pd.Series([1 / 8] * 8), ["R"] * 8).iloc[0] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# §2.6 connection place flags - the lag-safety rule that bites
# ---------------------------------------------------------------------------

def test_connection_plc_fsa_never_sees_a_same_race_stablemate():
    """Trainer repeats inside a race, so shift(1).expanding() would step back to
    a stablemate in the same race. The feature must be constant across the race
    and contain only earlier races."""
    rows = []
    for k, (a, b) in enumerate([(0.5, -0.5), (0.2, -0.2)]):
        rows += [{"raceid": f"r{k}", "race_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=k),
                  "race_time": "14.00", "trainer": "T", "jockey_name": f"J{i}", "horse_name": f"h{k}{i}",
                  "plc_fsa": v} for i, v in enumerate((a, b))]
    rows += [{"raceid": "r2", "race_date": pd.Timestamp("2025-01-03"), "race_time": "14.00", "trainer": "T",
              "jockey_name": "J9", "horse_name": "h20", "plc_fsa": 0.9}]
    d = pd.DataFrame(rows)
    out = connection_plc_fsa(d, min_races=1)

    r0 = out[out["raceid"] == "r0"]["trainer_lto_plc_fsa"]
    assert r0.isna().all(), "the first race has no earlier races to average"
    r1 = out[out["raceid"] == "r1"]["trainer_lto_plc_fsa"]
    assert r1.nunique() == 1, "a same-race stablemate leaked into a runner's own feature"
    assert r1.iloc[0] == pytest.approx(0.0)                             # mean of race r0 only
    assert out[out["raceid"] == "r2"]["trainer_lto_plc_fsa"].iloc[0] == pytest.approx(0.0)  # r0 and r1
    assert out["trainer_lto_plc_fsa"].tolist() == pytest.approx(
        race_lagged_expanding_mean(d, "trainer", "plc_fsa", min_races=1).tolist(), nan_ok=True)


def test_lto_context_features_are_the_previous_run():
    d = pd.DataFrame({"horse_name": "h", "race_time": "14.00",
                      "race_date": pd.date_range("2025-01-01", periods=3, freq="30D"),
                      "number_of_runners": [8, 12, 20], "dist_furlongs": [6.0, 8.0, 10.0],
                      "wide_flag": [0, 1, 0], "slow_start_flag": [1, 0, 0], "fss_qual": [70.0, 80.0, 90.0],
                      "official_rating": [75.0, 78.0, 80.0]})
    out = add_lto_context_features(d)
    assert out["lto_field_size"].tolist() == pytest.approx([np.nan, 8, 12], nan_ok=True)
    assert out["lto_FSS_qual"].tolist() == pytest.approx([np.nan, 70, 80], nan_ok=True)
    assert out["lto_wide_flag"].tolist() == pytest.approx([np.nan, 0, 1], nan_ok=True)
    assert out["lto_slow_start_flag"].tolist() == pytest.approx([np.nan, 1, 0], nan_ok=True)
    assert out["lto_dist_delta"].tolist() == pytest.approx([np.nan, 2.0, 2.0], nan_ok=True)


def test_margin_word_table_is_the_only_one():
    """One table for the repo: parse_beaten_lengths must agree with it, and the
    number+word forms the feed uses must not silently become NaN."""
    assert parse_beaten_lengths("nk") == MARGIN_WORDS["nk"]
    assert parse_beaten_lengths("2hd") == 2 + MARGIN_WORDS["hd"]
    assert parse_beaten_lengths("1nk") == 1 + MARGIN_WORDS["nk"]
    assert parse_beaten_lengths("dist") == MARGIN_WORDS["dist"]
    assert parse_beaten_lengths("dh") == 0.0
    assert np.isnan(parse_beaten_lengths("not a margin"))


def test_no_post_race_primitive_reaches_the_production_feature_list():
    """model.custom_metrics.POST_RACE_ONLY is what train_bfsp asserts against;
    POST_RACE_PRIMITIVES is the same guarantee for this module's columns. If a
    wiring change ever puts one of them in a feature list, fail here."""
    from train_bfsp import ALL_FEATURE_COLS

    leaked = sorted(set(ALL_FEATURE_COLS) & POST_RACE_PRIMITIVES)
    assert not leaked, f"post-race primitives in the feature list: {leaked}"
