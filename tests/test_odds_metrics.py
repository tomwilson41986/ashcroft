"""Part 4 odds-derived metrics, Stage C plumbing and the exotics.

Everything here is checked against a number worked out independently of the
implementation: an entropy computed by hand, a book generated from known true
probabilities by Shin's forward map, an analytic ordering distribution the
simulator has to reproduce, and the race-lag rule that a trainer's stablemate
must not leak into its own race.
"""

import numpy as np
import pandas as pd
import pytest

from model.feature_registry import stage_of
from model.market_features import (MARKET_FEATURES, MORNING_CLOCK_HOURS, PARKINSON_K, add_market_features,
                                   compute_run_descriptors, off_hours)
from model.odds_metrics import (ODDS_FEATURES, add_odds_metrics, benter_reliability, effective_field_size,
                                implied_probs, n_eff_by_race, race_odds_summary, reliability_comparison,
                                shin_book, shin_probabilities, shin_z, unratable_runner_rule)
from model.ordering import (DEFAULT_FIELD_BANDS, exacta_matrix, exacta_probability, exacta_table, expected_return,
                            fair_price, fit_ordering_params_by_band, ordering_params_for, place_probabilities,
                            placepot_probability, position_probabilities, quinella_probability,
                            simulate_finishing_orders, top3_probabilities, trifecta_probability)


# ---------------------------------------------------------------------------
# §2.2 / §4.3  N_eff, OFSR_eff, OFSR_conc
# ---------------------------------------------------------------------------

def test_n_eff_hand_computed():
    # H = -(0.5 ln0.5 + 2*0.25 ln0.25) = 1.5 ln2, so N_eff = 2^1.5 = 2*sqrt(2)
    assert effective_field_size([0.5, 0.25, 0.25]) == pytest.approx(2 * np.sqrt(2))
    assert effective_field_size([0.25] * 4) == pytest.approx(4.0)          # uniform: bodies = contenders
    assert effective_field_size([0.9, 0.05, 0.05]) < 1.6                   # one short-priced favourite
    # unnormalised input is normalised first
    assert effective_field_size([2.0, 1.0, 1.0]) == pytest.approx(2 * np.sqrt(2))
    # a 20-runner handicap with a 1/3 shot is not a twenty-horse puzzle
    big = np.r_[0.75, np.full(19, 0.25 / 19)]
    assert effective_field_size(big) < 5.0


def test_ofsr_variables_on_a_known_book():
    df = pd.DataFrame({"raceid": ["r1"] * 3, "bfsp": [2.0, 4.0, 4.0]})   # pi = .5/.25/.25, booksum exactly 1
    o = add_odds_metrics(df)
    assert o["pi_market"].tolist() == pytest.approx([0.5, 0.25, 0.25])
    assert o["overround"].iloc[0] == pytest.approx(1.0)
    assert o["n_eff_market"].iloc[0] == pytest.approx(2 * np.sqrt(2))
    assert o["OFSR_eff"].iloc[0] == pytest.approx(2 * np.sqrt(2) / 3)     # N_eff / N
    assert o["OFSR_conc"].iloc[0] == pytest.approx(1.5)                   # pi_fav * N = 0.5 * 3
    summary = race_odds_summary(df)
    assert len(summary) == 1 and summary["OFSR_eff"].iloc[0] == pytest.approx(2 * np.sqrt(2) / 3)


def test_n_eff_by_race_works_on_fundamental_probabilities():
    d = pd.DataFrame({"raceid": ["a"] * 3 + ["b"] * 4, "f": [0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]})
    ne = n_eff_by_race(d, "f")
    assert ne.iloc[0] == pytest.approx(2 * np.sqrt(2)) and ne.iloc[3] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# §4.2  OTRR and its variants
# ---------------------------------------------------------------------------

def test_otrr_variants():
    df = pd.DataFrame({"raceid": ["r"] * 4, "bfsp": [2.0, 4.0, 8.0, 8.0]})  # pi = .5/.25/.125/.125
    o = add_odds_metrics(df)
    assert o["OTRR"].tolist() == pytest.approx([2.0, 1.0, 0.5, 0.5])        # pi * N, N = 4
    assert o["ln_OTRR"].tolist() == pytest.approx(np.log([2.0, 1.0, 0.5, 0.5]))
    assert o["OTRR_rank"].tolist() == [1.0, 2.0, 3.0, 3.0]
    assert o["OTRR_norm"].tolist() == pytest.approx([1.0, 0.5, 0.25, 0.25])  # relative to the favourite
    assert o["OTRR_gap_fav"].tolist() == pytest.approx([0.0, np.log(2), np.log(4), np.log(4)])
    # gap to the next shorter-priced runner: NaN for the favourite, -ln2 for the second favourite
    assert np.isnan(o["OTRR_gap_next"].iloc[0])
    assert o["OTRR_gap_next"].iloc[1] == pytest.approx(-np.log(2))
    assert o["is_fav"].tolist() == [1.0, 0.0, 0.0, 0.0] and o["is_jt_fav"].sum() == 0
    assert o["fav_rank"].tolist() == o["OTRR_rank"].tolist()


def test_joint_favourites_and_a_second_race():
    df = pd.DataFrame({"raceid": ["r1", "r1", "r2", "r2", "r2"], "bfsp": [2.0, 2.0, 3.0, 4.0, 5.0]})
    o = add_odds_metrics(df)
    r1 = o[o["raceid"] == "r1"]
    assert r1["is_fav"].tolist() == [1.0, 1.0] and r1["is_jt_fav"].tolist() == [1.0, 1.0]
    r2 = o[o["raceid"] == "r2"]
    assert r2["is_jt_fav"].sum() == 0 and r2["OTRR_rank"].tolist() == [1.0, 2.0, 3.0]
    # each race normalises to its own book
    assert o.groupby("raceid")["pi_market"].sum().tolist() == pytest.approx([1.0, 1.0])


def test_missing_prices_do_not_contaminate_the_vector():
    df = pd.DataFrame({"raceid": ["r"] * 4, "bfsp": [2.0, 4.0, 4.0, np.nan]})
    o = add_odds_metrics(df)
    assert o["n_priced"].iloc[0] == 3                         # N counts priced runners only
    assert np.isnan(o["pi_market"].iloc[3]) and np.isnan(o["OTRR"].iloc[3])
    assert o["pi_market"].sum() == pytest.approx(1.0)
    assert o["n_eff_market"].iloc[0] == pytest.approx(2 * np.sqrt(2))


# ---------------------------------------------------------------------------
# §4.1  Shin (1993)
# ---------------------------------------------------------------------------

def test_shin_recovers_a_book_with_a_known_overround():
    p_true = np.array([0.5, 0.3, 0.2])
    z = 0.1
    book = shin_book(p_true, z)
    # the booksum follows in closed form: B = (sum_i sqrt(p_i (z + (1-z) p_i)))^2
    expected_book = (np.sqrt(0.5 * 0.55) + np.sqrt(0.3 * 0.37) + np.sqrt(0.2 * 0.28)) ** 2
    assert book.sum() == pytest.approx(expected_book) and book.sum() == pytest.approx(1.1973048, abs=1e-6)
    assert shin_z(book) == pytest.approx(z, abs=1e-6)
    assert shin_probabilities(book) == pytest.approx(p_true, abs=1e-9)
    # and the point of the method: the margin sits on the longshot
    prop = book / book.sum()
    assert prop[0] < p_true[0] and prop[-1] > p_true[-1]


def test_shin_is_the_identity_on_a_fair_book_and_on_an_underround_one():
    fair = np.array([0.5, 0.3, 0.2])
    assert shin_z(fair) == 0.0
    assert shin_probabilities(fair) == pytest.approx(fair)
    under = fair * 0.98                                    # a Betfair book can sum below 1
    assert shin_z(under) == 0.0
    assert shin_probabilities(under) == pytest.approx(fair)


def test_shin_frame_path_and_implied_probs():
    assert implied_probs([2.0, 4.0, 1.0, 0.0]) == pytest.approx([0.5, 0.25, np.nan, np.nan], nan_ok=True)
    odds = 1.0 / shin_book(np.array([0.5, 0.3, 0.2]), 0.1)
    df = pd.DataFrame({"raceid": ["r"] * 3, "bfsp": odds})
    o = add_odds_metrics(df, method="shin")
    assert o["pi_market"].tolist() == pytest.approx([0.5, 0.3, 0.2], abs=1e-8)
    assert o["overround"].iloc[0] == pytest.approx(1.1973048, abs=1e-6)
    prop = add_odds_metrics(df, method="proportional")["pi_market"]
    assert o["pi_market"].iloc[0] > prop.iloc[0] and o["pi_market"].iloc[2] < prop.iloc[2]
    with pytest.raises(ValueError):
        add_odds_metrics(df, method="nonsense")


# ---------------------------------------------------------------------------
# §IV.3  Monte-Carlo Plackett-Luce and the exotics
# ---------------------------------------------------------------------------

def test_plackett_luce_simulation_recovers_the_input_strengths():
    p = np.array([0.4, 0.25, 0.2, 0.1, 0.05])
    rng = np.random.default_rng(7)
    orders = simulate_finishing_orders(p, n_sims=200_000, rng=rng)          # gamma = delta = 1: pure PL
    pos = position_probabilities(orders, len(p))
    assert pos[:, 0] == pytest.approx(p, abs=0.004)                          # win frequencies == strengths
    h1, h2, h3 = top3_probabilities(p)
    assert pos[:, 1] == pytest.approx(h2, abs=0.005)                         # and Harville for 2nd / 3rd
    assert pos[:, 2] == pytest.approx(h3, abs=0.005)
    assert pos.sum(axis=1) == pytest.approx(np.ones(len(p)))                 # everybody finishes somewhere


def test_simulation_matches_the_benter_correction_when_flattened():
    p = np.array([0.45, 0.25, 0.15, 0.1, 0.05])
    orders = simulate_finishing_orders(p, n_sims=200_000, gamma=0.8, delta=0.65, rng=np.random.default_rng(11))
    pos = position_probabilities(orders, len(p))
    b1, b2, b3 = top3_probabilities(p, 0.8, 0.65)
    assert pos[:, 0] == pytest.approx(b1, abs=0.005)
    assert pos[:, 1] == pytest.approx(b2, abs=0.006)
    assert pos[:, 2] == pytest.approx(b3, abs=0.006)
    # flattening lifts the outsider's second-place chance above Harville's
    _, h2, _ = top3_probabilities(p)
    assert b2[-1] > h2[-1]


def test_exacta_quinella_trifecta_are_consistent():
    p = np.array([0.4, 0.3, 0.2, 0.1])
    m = exacta_matrix(p, 0.8)
    assert m.sum() == pytest.approx(1.0) and np.allclose(np.diag(m), 0.0)
    assert m.sum(axis=1) == pytest.approx(p)                                 # row total = P(win)
    _, p2, _ = top3_probabilities(p, 0.8, 0.65)
    assert m.sum(axis=0) == pytest.approx(p2)                                # column total = P(2nd)
    assert quinella_probability(p, 0, 1, 0.8) == pytest.approx(m[0, 1] + m[1, 0])
    assert exacta_probability(p, 0, 1, 0.8) == pytest.approx(m[0, 1])
    tri = sum(trifecta_probability(p, i, j, k, 0.8, 0.65)
              for i in range(4) for j in range(4) for k in range(4) if len({i, j, k}) == 3)
    assert tri == pytest.approx(1.0, abs=1e-9)


def test_exotic_pricing_and_the_commercial_case_for_exotics():
    p = np.array([0.4, 0.3, 0.2, 0.1])
    # fair price is break-even by construction, net of commission
    assert expected_return(0.25, fair_price(0.25), 0.0) == pytest.approx(1.0)
    assert expected_return(0.25, fair_price(0.25, 0.02), 0.02) == pytest.approx(1.0)
    t = exacta_table(p, 0.8, runners=list("ABCD"), top_k=3)
    assert list(t.columns) == ["first", "second", "prob", "fair_price"] and len(t) == 3
    assert t["prob"].is_monotonic_decreasing and t["first"].iloc[0] == "A"
    # Benter's point: two losing win bets can make a winning forecast, because
    # the probability difference compounds while the price multiplies out
    prob = exacta_probability(p, 0, 1, 0.8)
    assert expected_return(prob, fair_price(prob) * 1.2) == pytest.approx(1.2, abs=1e-9)
    mkt = {(0, 1): fair_price(prob) * 1.2}
    t2 = exacta_table(p, 0.8, top_k=1, market_prices=mkt)
    assert t2["expected_return"].iloc[0] > 1.0


def test_place_probabilities_beyond_third_and_placepots():
    p = np.array([0.4, 0.25, 0.2, 0.1, 0.05])
    p1, p2, p3 = top3_probabilities(p, 0.8, 0.65)
    assert place_probabilities(p, 1, 0.8, 0.65) == pytest.approx(p1)
    assert place_probabilities(p, 3, 0.8, 0.65) == pytest.approx(p1 + p2 + p3)
    four = place_probabilities(p, 4, 0.8, 0.65, n_sims=40_000, rng=np.random.default_rng(3))
    assert four.sum() == pytest.approx(4.0)                                  # four places, five runners
    assert np.all(four >= p1 + p2 + p3 - 1e-9) and np.all(four <= 1.0)
    assert np.all(np.diff(four) <= 1e-9)                                     # ordered like the win vector
    assert placepot_probability([[0.5, 0.2], [0.4], [0.9]]) == pytest.approx(0.7 * 0.4 * 0.9)
    assert placepot_probability([[0.8, 0.5]]) == pytest.approx(1.0)          # a leg cannot exceed certainty


def test_ordering_params_are_fitted_per_field_size_band():
    rng = np.random.default_rng(5)
    races = []
    for _ in range(1400):
        small = rng.random() < 0.5
        n = int(rng.integers(5, 8)) if small else int(rng.integers(12, 16))
        gamma = 0.9 if small else 0.55                       # different flattening in the two bands
        p = rng.dirichlet(np.ones(n) * 0.7)
        s = p ** gamma / (p ** gamma).sum()
        a = rng.choice(n, p=p)
        s2 = s.copy(); s2[a] = 0
        b = rng.choice(n, p=s2 / s2.sum())
        t = p ** 0.65 / (p ** 0.65).sum(); t3 = t.copy(); t3[[a, b]] = 0
        c = rng.choice(n, p=t3 / t3.sum())
        races.append((p, (a, b, c)))
    fit = fit_ordering_params_by_band(races, min_races=200)
    assert set(fit["bands"]) == {f"{lo}-{hi}" for lo, hi in DEFAULT_FIELD_BANDS}
    small_fit, big_fit = fit["bands"]["2-7"], fit["bands"]["12-15"]
    assert not small_fit["fallback"] and not big_fit["fallback"]
    assert small_fit["gamma"] > big_fit["gamma"] + 0.15      # the bands are genuinely different
    assert abs(small_fit["gamma"] - 0.9) < 0.15 and abs(big_fit["gamma"] - 0.55) < 0.15
    assert fit["bands"]["8-11"]["fallback"]                  # no races in that band -> global pair
    assert ordering_params_for(fit, 6)[0] == pytest.approx(small_fit["gamma"])
    assert ordering_params_for(fit, 13)[0] == pytest.approx(big_fit["gamma"])
    assert ordering_params_for({"global": {"gamma": 0.8, "delta": 0.6}}, 9) == (0.8, 0.6)


# ---------------------------------------------------------------------------
# §IV.1 / §IV.2  Stage C plumbing
# ---------------------------------------------------------------------------

def test_unratable_runners_take_the_market_price():
    pi = np.array([0.5, 0.3, 0.2, 0.6, 0.4])
    f = np.array([0.6, 0.4, 0.9, 0.5, 0.5])
    ratable = np.array([True, True, False, False, False])
    groups = ["r1", "r1", "r1", "r2", "r2"]
    out = unratable_runner_rule(f, pi, ratable, groups, min_ratable=2)
    # the unratable runner keeps its 0.2; the other two share 0.8 in the ratio 0.6 : 0.4
    assert out["p"][:3] == pytest.approx([0.48, 0.32, 0.2])
    assert out["p"][3:] == pytest.approx([0.6, 0.4])         # no ratable runner -> the market vector
    assert out["skip"].tolist() == [False, False, False, True, True]
    assert out["n_ratable"].tolist() == [2, 2, 2, 0, 0]
    for g in ("r1", "r2"):
        m = np.array(groups) == g
        assert out["p"][m].sum() == pytest.approx(1.0)


def test_benter_reliability_flags_a_biased_forecast():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.02, 0.45, 20_000)
    y = (rng.random(20_000) < p).astype(float)
    good = benter_reliability(y, p)
    assert set(good.columns) >= {"band_lo", "band_hi", "n", "expected", "actual", "z"}
    assert good["n"].sum() == 20_000
    assert good["expected"].sum() == pytest.approx(p.sum())
    assert good["z"].abs().max() < 4.0                        # calibrated: no band is out of line
    bad = benter_reliability(y, np.clip(p * 1.5, 0, 0.999))
    assert bad["z"].abs().max() > 4.0                         # inflated probabilities show up as Z
    table = reliability_comparison(y, {"market": p, "fundamental": p * 1.5, "combined": p * 1.1})
    assert set(table["forecast"]) == {"market", "fundamental", "combined"}


# ---------------------------------------------------------------------------
# P5: none of this may reach Stage F
# ---------------------------------------------------------------------------

def test_every_odds_column_is_registered_as_stage_c():
    for name in ODDS_FEATURES:
        assert stage_of(name) == "C", f"{name} would reach the fundamental model"
    for name in MARKET_FEATURES:
        assert stage_of(name) == "C", f"{name} would reach the fundamental model"
    for name in ("am_OTRR", "am_is_fav", "pi_shin", "trainer_mkt_ae", "sire_mkt_p_mean"):
        assert stage_of(name) == "C"
    # and the Stage F names are untouched by the new patterns
    for name in ("kf_z", "sos_vs_today", "nmfp_mean3_z", "first_run", "draw_pct"):
        assert stage_of(name) == "F"


def test_phase1_keeps_the_odds_block_out_of_stage_f():
    from model.phase1 import STAGE_C_ODDS_FEATURES, STAGE_F_FEATURES
    assert not set(STAGE_C_ODDS_FEATURES) & set(STAGE_F_FEATURES)
    assert "OTRR" in STAGE_C_ODDS_FEATURES and "OFSR_eff" in STAGE_C_ODDS_FEATURES


# ---------------------------------------------------------------------------
# §4.4 / §4.5  market_features: drift rate, volatility, connections, lag safety
# ---------------------------------------------------------------------------

def _rr_and_market():
    """Two meetings. Trainer T saddles BOTH runners in every race — the case
    `shift(1).expanding()` gets wrong."""
    rr = pd.DataFrame({
        "id": [1, 2, 3, 4], "race_date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-02-01", "2026-02-01"]),
        "race_time": ["2.30", "2.30", "3.00", "3.00"], "track": ["Kempton"] * 4,
        "horse_name": ["A", "B", "A", "B"], "placing_numerical": [1, 2, 2, 1], "number_of_runners": [2, 2, 2, 2],
        "trainer": ["T", "T", "T", "T"], "jockey_name": ["J", "K", "J", "K"],
    })
    mk = pd.DataFrame({"race_results_id": [1, 2, 3, 4], "bfp_bsp": [2.0, 3.0, 2.5, 2.2],
                       "bfp_ppwap": [2.1, 2.9, 2.5, 2.3], "bfp_morningwap": [3.0, 2.5, 2.4, 2.6],
                       "bfp_ppmax": [3.2, 3.1, 2.7, 2.8], "bfp_ppmin": [1.9, 2.4, 2.3, 2.1],
                       "bfp_ipmax": [10, 12, 8, 9], "bfp_ipmin": [1.01, 3.0, 1.2, 1.01],
                       "bfp_morning_vol": [100, 50, 80, 90], "bfp_pp_vol": [1000, 500, 800, 900],
                       "bfp_ip_vol": [300, 200, 100, 400], "bfp_place_bsp": [1.3, 1.5, 1.4, 1.3]})
    return rr, mk


def test_trainer_aggregates_do_not_see_a_stablemate_in_the_same_race():
    rr, mk = _rr_and_market()
    out = add_market_features(rr, market=mk).sort_values(["horse_name", "race_date"]).reset_index(drop=True)
    first_meeting = out[out["race_date"] == pd.Timestamp("2026-01-01")]
    # both runners are trained by T; with a row-wise shift the second would have
    # read the first's result out of the race being predicted
    for col in ("trainer_steam_rate", "trainer_outperform", "trainer_mkt_p_mean", "trainer_mkt_ae"):
        assert first_meeting[col].isna().all(), col
    second = out[out["race_date"] == pd.Timestamp("2026-02-01")]
    # after one prior race the yard's mean implied probability is that race's mean
    assert second["trainer_mkt_p_mean"].tolist() == pytest.approx([0.5, 0.5])
    assert second["trainer_mkt_ae"].tolist() == pytest.approx([1.0, 1.0])   # one winner in two runners / 0.5 expected
    assert second["trainer_steam_rate"].tolist() == pytest.approx([0.5, 0.5])
    assert "_mkt_won" not in out.columns and "_steamer" not in out.columns
    assert set(MARKET_FEATURES).issubset(out.columns)


def test_drift_rate_and_pre_off_volatility():
    assert off_hours(pd.Series(["2.30", "12.15.", "1.00", "14:30", "6.00"])).tolist() == [14.5, 12.25, 13.0, 14.5, 18.0]
    rr, mk = _rr_and_market()
    d = compute_run_descriptors(rr.merge(mk, left_on="id", right_on="race_results_id"))
    elapsed = 14.5 - MORNING_CLOCK_HOURS
    assert d["mkt_drift_rate"].iloc[0] == pytest.approx(np.log(3.0 / 2.0) / elapsed)
    # a later race on the same card has longer for the same move, so a lower rate
    assert abs(d["mkt_drift_rate"].iloc[2]) < abs(d["mkt_steam"].iloc[2])
    assert d["mkt_pp_vol"].iloc[0] == pytest.approx(np.log(3.2 / 1.9) * PARKINSON_K)
    assert (d["mkt_pp_vol_gk"].dropna() >= 0).all()
    gk = np.sqrt(max(0.5 * np.log(3.2 / 1.9) ** 2 - (2 * np.log(2) - 1) * np.log(2.0 / 3.0) ** 2, 0.0))
    assert d["mkt_pp_vol_gk"].iloc[0] == pytest.approx(gk)
    assert d["mkt_race_vol_ln"].iloc[0] == pytest.approx(np.log1p(1100 + 550))


def test_delta_r2_splits_by_market_shape():
    from model.phase1 import delta_r2_by_band
    rng = np.random.default_rng(4)
    rows = []
    for r in range(90):
        n = 8
        pi = rng.dirichlet(np.ones(n) * (0.4 if r % 3 == 0 else 2.0))     # some books concentrated, some flat
        w = rng.choice(n, p=pi)
        for i in range(n):
            rows.append({"raceid": f"r{r}", "won": float(i == w), "pi_market": pi[i], "c_oof": pi[i]})
    wf = pd.DataFrame(rows)
    wf["OFSR_eff"] = n_eff_by_race(wf, "pi_market") / 8.0
    same = delta_r2_by_band(wf, n_bands=3)
    assert len(same) == 3 and same["n_races"].sum() == 90
    assert same["delta_r2"].abs().max() < 1e-12          # combined == market -> no edge anywhere
    assert same["mean_OFSR_eff"].is_monotonic_increasing
    # a combined forecast that leans toward the winner beats the market in every band
    better = wf.copy()
    better["c_oof"] = np.where(better["won"] > 0, better["pi_market"] * 1.5, better["pi_market"])
    better["c_oof"] /= better.groupby("raceid")["c_oof"].transform("sum")
    out = delta_r2_by_band(better, n_bands=3)
    assert (out["delta_r2"] > 0).all() and (out["r2_combined"] > out["r2_market"]).all()
    assert delta_r2_by_band(wf.iloc[0:0], n_bands=3).empty
