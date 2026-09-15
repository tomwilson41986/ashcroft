"""
Tests for portfolio staking, execution and bet-performance attribution.

The point of a portfolio test is that it must fail when the maths is wrong,
not merely when the code raises. So the joint Kelly solution is checked
against a two-runner case solved by hand, the within-race covariance against
the multinomial covariance it is supposed to be (and against a simulation of
it), the slippage against a book that is walked by hand, and the two
diagnostics -- effective breadth and alpha decay -- against synthetics whose
answers are known by construction.
"""

import numpy as np
import pandas as pd
import pytest

from model.portfolio import (alpha_decay, apply_depth_fills, back_return_moments, block_covariance,
                             effective_breadth, estimate_block_correlations, ev_stake_curve,
                             expected_log_growth, factor_attribution, grinold_report, ladder_fill,
                             ladder_price_fn, ledoit_wolf_shrinkage, liquidity_stress, max_ev_stake,
                             net_odds, pool_price_fn, portfolio_kelly, race_kelly_exact,
                             race_kelly_numeric, race_return_covariance, realised_ic, required_bets,
                             split_order)
from model.staking import (PHI_DEFAULT, PHI_LADDER, add_kelly, baker_mchale_shrinkage, check_phi,
                           confidence_stakes, phi_from_confidence)


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def _card(n_days=6, meetings=3, races=4, runners=8, edge=0.0, day_shock=0.0,
          meeting_shock=0.0, seed=17):
    """A card of races with controllable shared error.

    ``edge`` is a constant proportional overlay in the price, so the model has
    a real and known advantage; the shocks add a common component to every
    return at that level, which is the correlation the block covariance and the
    breadth diagnostic exist to find."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        d_eps = rng.normal(0.0, day_shock)
        day = (pd.Timestamp("2025-01-06") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        for m in range(meetings):
            m_eps = rng.normal(0.0, meeting_shock)
            for r in range(races):
                s = rng.normal(0, 0.9, runners)
                p = np.exp(s) / np.exp(s).sum()
                w = int(rng.choice(runners, p=p))
                rest = [i for i in range(runners) if i != w]
                rng.shuffle(rest)
                pos = {w: 1, **{h: k + 2 for k, h in enumerate(rest)}}
                price = (1.0 + edge) / p
                for i in range(runners):
                    ret = (price[i] - 1.0) if i == w else -1.0
                    rows.append({"race_date": day, "race_time": "%02d:00" % (12 + r),
                                 "track": f"T{m}", "raceid": f"{day}-{m}-{r}",
                                 "horse_name": f"h{i}", "field": runners,
                                 "p_model": p[i], "model_price": 1.0 / p[i], "bsp": price[i],
                                 "won": i == w, "placing_numerical": pos[i],
                                 "ret_back": ret + d_eps + m_eps})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Portfolio Kelly (V.3, N5 / N6)
# ---------------------------------------------------------------------------

def test_two_runner_portfolio_kelly_matches_the_hand_solution():
    """p = (0.5, 0.3) at 2.5 and 4.0, no commission.

    p_S = 0.8, r_S = 1/2.5 + 1/4 = 0.65, so (1 - p_S)/(1 - r_S) = 0.2/0.35 and
        f_1 = 0.5 - (0.2/0.35)/2.5 = 0.2714285714
        f_2 = 0.3 - (0.2/0.35)/4.0 = 0.1571428571
    """
    p, o = [0.5, 0.3], [2.5, 4.0]
    f = race_kelly_exact(p, o, commission=0.0)
    assert np.allclose(f, [0.2714285714, 0.1571428571], atol=1e-9)
    assert np.allclose(race_kelly_numeric(p, o, commission=0.0), f, atol=1e-6)


def test_backing_only_the_top_advantage_runner_leaves_growth_on_the_table():
    """The framework's V.3 warning, made arithmetic."""
    p, o = [0.5, 0.3], [2.5, 4.0]
    joint = race_kelly_exact(p, o, commission=0.0)
    solo = np.array([(0.5 * 1.5 - 0.5) / 1.5, 0.0])       # textbook Kelly on the favourite alone
    assert solo[0] == pytest.approx(1 / 6)
    assert joint[0] > solo[0]                              # the hedge lets the top pick take MORE
    g_joint = expected_log_growth(joint, p, o, ["r", "r"], commission=0.0, n_sims=400_000)
    g_solo = expected_log_growth(solo, p, o, ["r", "r"], commission=0.0, n_sims=400_000)
    assert g_joint > g_solo
    assert g_joint == pytest.approx(0.0543450851, abs=2e-3)   # the exact objective at the optimum
    assert g_solo == pytest.approx(0.0204109973, abs=2e-3)


def test_single_runner_reduces_to_textbook_kelly():
    assert race_kelly_exact([0.5], [3.0], commission=0.0)[0] == pytest.approx((0.5 * 2 - 0.5) / 2)
    assert race_kelly_exact([0.5], [3.0], commission=0.5)[0] == pytest.approx(0.0)   # commission eats it


def test_a_fairly_priced_race_is_not_backed():
    p = np.array([0.4, 0.3, 0.2, 0.1])
    assert np.allclose(race_kelly_exact(p, 1.0 / p, commission=0.0), 0.0, atol=1e-9)
    assert np.allclose(race_kelly_numeric(p, 1.0 / p, commission=0.0), 0.0, atol=1e-6)


def test_caps_bind_on_the_numeric_solution():
    p, o = [0.5, 0.3], [2.5, 4.0]
    assert race_kelly_numeric(p, o, commission=0.0, cap_race=0.20).sum() == pytest.approx(0.20, abs=1e-6)
    capped = race_kelly_numeric(p, o, commission=0.0, cap_selection=0.05)
    assert capped.max() <= 0.05 + 1e-9
    assert np.all(race_kelly_numeric(p, o, commission=0.0, bounds_hi=[0.0, 1.0])[:1] == 0.0)


def test_portfolio_kelly_respects_every_exposure_cap():
    d = _card(n_days=1, edge=0.35, seed=5)
    for method in ("quadratic", "exact"):
        out = portfolio_kelly(d, phi=0.25, cap_total=0.20, cap_race=0.05, cap_selection=0.02,
                              method=method)
        a = out.attrs["portfolio"]
        assert a["exposure"] <= 0.20 + 1e-8
        assert a["max_race_exposure"] <= 0.05 + 1e-8
        assert a["max_selection"] <= 0.02 + 1e-8
        assert (out["stake_fraction"] >= -1e-12).all()
        assert out["stake"].sum() == pytest.approx(a["exposure"])


def test_portfolio_kelly_scales_with_phi_and_stakes_nothing_on_a_fair_book():
    d = _card(n_days=1, edge=0.35, seed=5)
    quarter = portfolio_kelly(d, phi=0.25, cap_total=1.0, cap_race=1.0, cap_selection=1.0)
    half = portfolio_kelly(d, phi=0.5, cap_total=1.0, cap_race=1.0, cap_selection=1.0)
    assert half.attrs["portfolio"]["exposure"] > quarter.attrs["portfolio"]["exposure"]
    # with the caps slack, quadratic Kelly is linear in phi
    assert np.allclose(half["stake_fraction"], 2 * quarter["stake_fraction"], rtol=1e-3, atol=1e-6)
    fair = portfolio_kelly(_card(n_days=1, edge=0.0, seed=5), phi=0.25, commission=0.0)
    assert fair["stake_fraction"].sum() == pytest.approx(0.0, abs=1e-6)


def test_per_selection_cap_can_come_from_exchange_volume():
    d = _card(n_days=1, races=1, meetings=1, edge=0.5, seed=8).copy()
    d["depth"] = 20.0                                       # 20 currency units available
    out = portfolio_kelly(d, phi=0.5, bank=1000.0, volume_col="depth", volume_fraction=0.10,
                          cap_selection=1.0, cap_race=1.0, cap_total=1.0)
    assert out["stake"].max() <= 0.10 * 20.0 + 1e-6         # never more than a tenth of the book


def test_quadratic_and_exact_agree_when_the_stakes_are_small():
    """The quadratic objective is the second-order expansion of the log one, so
    the two must agree wherever the stakes are small enough for the third-order
    term not to matter. Two runners carry an overlay, the rest are underlays,
    and the book stays above 100% -- no arbitrage for the exact solver to find
    and no regime the expansion cannot follow."""
    p = np.array([0.30, 0.22, 0.18, 0.15, 0.15])
    price = (1.0 / p) * np.array([1.10, 1.08, 0.85, 0.85, 0.85])
    assert (1.0 / price).sum() > 1.0                          # an over-round book, not an arb
    d = pd.DataFrame({"raceid": "r1", "race_date": "2025-05-01", "track": "T",
                      "p_model": p, "bsp": price})
    quad = portfolio_kelly(d, phi=1.0, commission=0.0, cap_total=1.0, cap_race=1.0,
                           cap_selection=1.0, method="quadratic",
                           rho_meeting=0.0, rho_day=0.0)["stake_fraction"].to_numpy()
    exact = race_kelly_exact(p, price, commission=0.0)
    assert np.count_nonzero(exact) == 2 and exact.sum() < 0.15
    assert np.allclose(quad, exact, atol=0.01)
    g_q = expected_log_growth(quad, p, price, ["r1"] * 5, commission=0.0, n_sims=200_000)
    g_e = expected_log_growth(exact, p, price, ["r1"] * 5, commission=0.0, n_sims=200_000)
    assert g_q == pytest.approx(g_e, abs=5e-4)


def test_the_exact_solver_dutches_a_whole_race_when_the_book_is_an_arbitrage():
    """A book under 100% is risk-free money and the exact solution takes all of
    it. The quadratic expansion sees only a mean and a covariance, so it cannot
    recognise the certainty -- a real difference between the two methods, worth
    pinning down rather than papering over."""
    p = np.array([0.5, 0.3, 0.2])
    price = 1.05 / p                                          # book = 1/1.05 < 1
    f = race_kelly_exact(p, price, commission=0.0)
    assert np.allclose(f, p)                                  # stake the whole bank, split as p
    assert f.sum() == pytest.approx(1.0)


def test_empty_frame_is_handled():
    out = portfolio_kelly(pd.DataFrame(columns=["p_model", "bsp", "raceid", "race_date"]))
    assert out.empty and out.attrs["portfolio"]["bets"] == 0


# ---------------------------------------------------------------------------
# Covariance (IIA.5, B18 / N7)
# ---------------------------------------------------------------------------

def test_race_covariance_is_the_multinomial_one_and_survives_simulation():
    p = np.array([0.5, 0.3, 0.15, 0.05])
    price = 1.0 / p
    C = race_return_covariance(p, price, commission=0.0)
    o = net_odds(price, 0.0)
    assert np.allclose(np.diag(C), o ** 2 * p * (1 - p))
    assert C[0, 1] == pytest.approx(-o[0] * o[1] * p[0] * p[1])
    assert np.all(C[~np.eye(4, dtype=bool)] < 0)             # only one can win
    rng = np.random.default_rng(0)
    draw = rng.choice(4, size=400_000, p=p)
    R = np.where(np.eye(4, dtype=bool)[draw], o, 0.0) - 1.0
    assert np.allclose(np.cov(R, rowvar=False), C, rtol=0.02, atol=0.02)


def test_block_covariance_within_race_block_is_exactly_the_multinomial_one():
    """The one part of the matrix the framework says is *known*, not estimated."""
    d = _card(n_days=1, meetings=1, races=1, runners=6, edge=0.1, seed=3)
    S, info = block_covariance(d, rho_meeting=0.0, rho_day=0.0, commission=0.04)
    assert np.allclose(S - np.diag(np.diag(S)) * 0,
                       race_return_covariance(d["p_model"], d["bsp"], commission=0.04), atol=1e-9)
    assert info["psd_repair"] == 0.0


def test_block_covariance_has_three_levels_and_stays_psd():
    d = _card(n_days=2, meetings=2, races=3, runners=5, edge=0.1, seed=11)
    rho_m, rho_d = 0.30, 0.08
    S, info = block_covariance(d, rho_meeting=rho_m, rho_day=rho_d, commission=0.0)
    _, sd = back_return_moments(d["p_model"], d["bsp"], 0.0)
    race = d["raceid"].to_numpy()
    meet = (d["track"] + d["race_date"]).to_numpy()
    day = d["race_date"].to_numpy()
    assert np.allclose(np.diag(S), sd ** 2)                                   # variance preserved
    i, j = np.flatnonzero((meet == meet[0]) & (race != race[0]))[0], 0
    assert S[i, j] == pytest.approx(rho_m * sd[i] * sd[j])                    # same meeting
    k = np.flatnonzero((day == day[0]) & (meet != meet[0]))[0]
    assert S[k, j] == pytest.approx(rho_d * sd[k] * sd[j])                    # same day
    m = np.flatnonzero(day != day[0])[0]
    assert S[m, j] == pytest.approx(0.0)                                      # different day
    assert np.linalg.eigvalsh((S + S.T) / 2).min() > -1e-8                    # PSD by construction
    assert info["psd_repair"] == 0.0
    # the naive alternative -- equicorrelation laid straight on the exact blocks -- is not PSD
    naive = np.where(race[:, None] == race[None, :],
                     race_return_covariance(d["p_model"], d["bsp"], 0.0),
                     rho_m * np.outer(sd, sd))
    assert np.linalg.eigvalsh((naive + naive.T) / 2).min() < -1e-6


def test_block_covariance_requires_monotone_correlations():
    d = _card(n_days=1, meetings=1, races=1, seed=1)
    with pytest.raises(ValueError):
        block_covariance(d, rho_meeting=0.05, rho_day=0.30)


def test_ledoit_wolf_shrinks_toward_the_target_and_beats_the_raw_sample():
    rng = np.random.default_rng(4)
    n, T = 12, 40                                            # few observations, many bets
    A = rng.normal(size=(n, n))
    truth = A @ A.T / n + np.eye(n)
    panel = rng.multivariate_normal(np.zeros(n), truth, size=T)
    S = np.cov(panel, rowvar=False, bias=True)
    shrunk, delta = ledoit_wolf_shrinkage(panel, truth)
    assert 0.0 < delta < 1.0
    assert np.linalg.norm(shrunk - truth) < np.linalg.norm(S - truth)
    # a perfectly estimated covariance needs no shrinkage toward a wrong target
    big = rng.multivariate_normal(np.zeros(n), truth, size=20_000)
    _, small_delta = ledoit_wolf_shrinkage(big, np.eye(n))
    assert small_delta < delta


def test_ledoit_wolf_falls_back_to_the_target_without_a_panel():
    t = np.eye(3)
    out, delta = ledoit_wolf_shrinkage(np.zeros((1, 3)), t)
    assert delta == 1.0 and np.allclose(out, t)


def test_estimate_block_correlations_finds_a_shared_card_error():
    quiet = estimate_block_correlations(_card(edge=0.05, seed=21))
    noisy = estimate_block_correlations(_card(edge=0.05, meeting_shock=0.8, seed=21))
    assert quiet["rho_meeting"] < 0.02                        # independent bets: nothing to find
    assert noisy["rho_meeting"] > 5 * max(quiet["rho_meeting"], 1e-3)
    assert noisy["rho_day"] <= noisy["rho_meeting"] + 1e-12    # monotone, as the target requires


# ---------------------------------------------------------------------------
# Execution: depth, impact and splitting (V.4, N8 / N9 / N11)
# ---------------------------------------------------------------------------

LADDER_P = [6.0, 5.8, 5.6, 5.4]
LADDER_S = [100.0, 100.0, 100.0, 100.0]


def test_a_larger_order_fills_at_a_worse_average_price():
    small = ladder_fill(LADDER_P, LADDER_S, 50)
    big = ladder_fill(LADDER_P, LADDER_S, 350)
    assert small["avg_price"] == 6.0 and small["slippage_pct"] == 0.0
    assert big["avg_price"] == pytest.approx((100 * 6.0 + 100 * 5.8 + 100 * 5.6 + 50 * 5.4) / 350)
    assert big["avg_price"] < small["avg_price"]
    assert big["slippage_pct"] > small["slippage_pct"] == 0.0
    prices = [ladder_fill(LADDER_P, LADDER_S, s)["avg_price"] for s in (50, 150, 250, 350, 400)]
    assert all(a >= b for a, b in zip(prices, prices[1:]))     # monotone, never improves with size


def test_a_ladder_walk_conserves_money_and_reports_what_it_could_not_match():
    f = ladder_fill(LADDER_P, LADDER_S, 600)
    assert f["matched"] == 400.0 and f["unmatched"] == 200.0
    assert f["levels"] == 4 and f["depth"] == 400.0
    assert ladder_fill(LADDER_P, LADDER_S, 0)["matched"] == 0.0
    assert ladder_fill([], [], 100)["matched"] == 0.0


def test_ev_is_not_linear_in_stake_and_two_thirds_captures_eight_ninths():
    deep = ladder_price_fn(np.linspace(6.0, 3.0, 60), np.full(60, 50.0))
    curve = ev_stake_curve(0.20, deep, commission=0.0, n_grid=400)
    assert curve["avg_price"].is_monotonic_decreasing
    m = max_ev_stake(0.20, deep, commission=0.0, n_grid=2000)
    assert not m["at_capacity"]
    assert m["two_thirds_stake"] == pytest.approx(2 * m["max_ev_stake"] / 3)
    assert m["two_thirds_ev"] < m["max_ev"]
    assert m["ev_captured_pct"] == pytest.approx(100 * 8 / 9, abs=2.0)
    assert m["recommended_stake"] == m["two_thirds_stake"]      # the framework's default
    assert (max_ev_stake(0.20, deep, commission=0.0, n_grid=2000, default="max_ev")
            ["recommended_stake"] == m["max_ev_stake"])
    # a book too shallow to price the impact says so instead of inventing a maximum
    assert max_ev_stake(0.20, ladder_price_fn(LADDER_P, LADDER_S), commission=0.0)["at_capacity"]


def test_pari_mutuel_curve_reproduces_the_frameworks_illustration():
    """§V.4: p = 0.06, dividend 20, pool 100,000 -> max-EV bet 416 for 39.60,
    and two-thirds of it, 277, captures 35.50, i.e. 90% of the profit."""
    m = max_ev_stake(0.06, pool_price_fn(100_000, 4_141, takeout=0.172), commission=0.0,
                     n_grid=4000, max_stake=2000)
    assert m["max_ev_stake"] == pytest.approx(416, abs=8)
    assert m["max_ev"] == pytest.approx(39.6, abs=0.5)
    assert m["two_thirds_stake"] == pytest.approx(277, abs=6)
    assert m["two_thirds_ev"] == pytest.approx(35.5, abs=0.5)
    assert m["ev_captured_pct"] == pytest.approx(90, abs=1.5)


def test_an_unbackable_price_has_no_max_ev_stake():
    assert max_ev_stake(0.05, ladder_price_fn(LADDER_P, LADDER_S))["recommended_stake"] == 0.0


def test_splitting_an_order_helps_exactly_when_the_book_replenishes():
    one = ladder_fill(LADDER_P, LADDER_S, 300)["avg_price"]
    split = split_order(300, LADDER_P, LADDER_S, n_slices=3, replenish=0.8)
    assert split.attrs["split"]["blended_price"] > one
    flat = split_order(300, LADDER_P, LADDER_S, n_slices=3, replenish=0.0)
    assert flat.attrs["split"]["blended_price"] == pytest.approx(one)   # no refill, no gain
    assert flat.attrs["split"]["matched"] == pytest.approx(300.0)


def test_splitting_can_send_part_of_the_order_to_bsp():
    out = split_order(400, LADDER_P, LADDER_S, n_slices=2, replenish=0.5,
                      bsp_share=0.25, bsp_price=5.9)
    assert set(out["venue"]) == {"exchange", "bsp"}
    assert out.loc[out["venue"] == "bsp", "stake"].sum() == pytest.approx(100.0)
    assert out.attrs["split"]["bsp_share"] == 0.25
    assert out.attrs["split"]["blended_net_price"] < out.attrs["split"]["blended_price"]  # commission


def test_depth_fills_cap_the_stake_and_degrade_the_price():
    d = pd.DataFrame({"stake": [10.0, 100.0], "depth": [1000.0, 100.0], "bsp": [5.0, 5.0],
                      "won": [True, True]})
    out = apply_depth_fills(d, impact=0.5, commission=0.0, max_depth_share=0.5)
    assert out["stake_matched"].tolist() == [10.0, 50.0]         # capped at half the book
    assert out["unmatched"].tolist() == [0.0, 50.0]
    assert out["avg_price"].iloc[0] > out["avg_price"].iloc[1]   # the bigger share pays more impact
    assert out["ret_back_filled"].iloc[1] == pytest.approx(out["avg_price"].iloc[1] - 1.0)


def test_liquidity_stress_reports_what_could_not_be_matched():
    d = add_kelly(_card(n_days=2, edge=0.4, seed=9), commission=0.05)
    d["depth"] = 0.002                                          # a very thin book
    out = liquidity_stress(d, fraction=0.25, max_depth_share=0.5, bank=1.0)
    assert list(out["scenario"]) == ["quoted price, unlimited depth", "depth-capped with slippage"]
    info = out.attrs["liquidity"]
    assert info["unmatched_pct"] > 50.0
    assert out["turnover_bank_multiples"].iloc[1] < out["turnover_bank_multiples"].iloc[0]


# ---------------------------------------------------------------------------
# Grinold: IC and breadth (IIA.2, B11 / B12 / O21)
# ---------------------------------------------------------------------------

def test_realised_ic_is_positive_when_the_model_names_the_winner():
    good = pd.DataFrame({"raceid": ["r"] * 4, "p_model": [0.7, 0.1, 0.1, 0.1],
                         "won": [True, False, False, False]})
    bad = good.assign(won=[False, False, False, True])
    assert realised_ic(good)["ic"].iloc[0] > 0.9
    assert realised_ic(bad)["ic"].iloc[0] < 0
    flat = pd.DataFrame({"raceid": ["r"] * 3, "p_model": [1 / 3] * 3, "won": [True, False, False]})
    assert realised_ic(flat).empty                              # no separation, no IC


def test_effective_breadth_only_equals_the_race_count_under_independence():
    d = _card(n_days=4, meetings=3, races=5, seed=2)
    indep = effective_breadth(d, rho_meeting=0.0, rho_day=0.0)
    assert indep["breadth_effective"] == pytest.approx(indep["races"])
    for rho in (0.1, 0.3, 0.6):
        b = effective_breadth(d, rho_meeting=rho, rho_day=rho / 4)
        assert b["breadth_effective"] < indep["races"]
    assert (effective_breadth(d, rho_meeting=0.6, rho_day=0.15)["breadth_effective"]
            < effective_breadth(d, rho_meeting=0.2, rho_day=0.05)["breadth_effective"])


def test_a_full_card_of_correlated_races_is_worth_far_fewer_independent_bets():
    """Eight races on one card at rho = 0.2 -> 8 / (1 + 7 * 0.2) = 3.33."""
    d = _card(n_days=1, meetings=1, races=8, runners=4, seed=6)
    b = effective_breadth(d, rho_meeting=0.2, rho_day=0.2)
    assert b["races"] == 8
    assert b["breadth_effective"] == pytest.approx(8 / (1 + 7 * 0.2), rel=1e-6)


def _top_pick(d):
    """One bet per race -- the model's own top pick -- so that breadth counts
    forecasts. Backing a whole field is a hedge, not extra breadth, and the
    diagnostic would read the negative within-race correlation as a bonus."""
    return d[d["p_model"] == d.groupby("raceid")["p_model"].transform("max")]


def test_grinold_shortfall_detects_days_whose_bets_move_together():
    quiet_card = _card(n_days=90, meetings=2, races=4, edge=0.05, seed=31)
    loud_card = _card(n_days=90, meetings=2, races=4, edge=0.05, day_shock=0.9, seed=31)
    indep = grinold_report(_top_pick(quiet_card), commission=0.0, field=quiet_card)
    shocked = grinold_report(_top_pick(loud_card), commission=0.0, field=loud_card)
    assert np.isfinite(indep["ic_mean"]) and indep["ic_mean"] > 0
    assert np.isnan(grinold_report(_top_pick(quiet_card), commission=0.0)["ic_mean"])
    assert indep["ir_shortfall_ratio"] == pytest.approx(1.0, abs=0.25)
    assert indep["breadth_implied"] == pytest.approx(indep["bets_per_day"], rel=0.6)
    assert shocked["ir_shortfall_ratio"] < indep["ir_shortfall_ratio"]
    assert shocked["breadth_implied"] < 0.6 * shocked["bets_per_day"]
    assert shocked["breadth_lost_pct"] > indep["breadth_lost_pct"]
    assert shocked["ic_mean"] == pytest.approx(indep["ic_mean"], abs=1e-9)   # same forecasts
    assert indep["ic_gain_equivalent_to_doubling_breadth_pct"] == pytest.approx(41.4)


def test_backing_the_whole_field_is_a_hedge_not_extra_breadth():
    """The mirror image: within a race the returns are negatively correlated,
    so a day of full-field bets is *less* volatile than independence implies
    and the shortfall ratio goes above 1. Reading that as skill would be an
    error, which is why the diagnostic belongs on selections."""
    full = grinold_report(_card(n_days=90, meetings=2, races=4, edge=0.05, seed=31), commission=0.0)
    assert full["ir_shortfall_ratio"] > 1.1


def test_required_bets_scales_as_variance_over_squared_edge():
    a = required_bets(0.02, sd=2.0)
    b = required_bets(0.01, sd=2.0)
    c = required_bets(0.02, sd=4.0)
    assert b["required_bets"] == pytest.approx(4 * a["required_bets"])
    assert c["required_bets"] == pytest.approx(4 * a["required_bets"])
    assert a["required_bets"] > 1000                            # far past the framework's "1,000+"
    assert required_bets(0.02, bets=pd.DataFrame({"ret_back": [1.0, -1.0, 2.0, -1.0]}))["sd"] > 0


# ---------------------------------------------------------------------------
# Attribution and decay (IIA.7, VI.4, B20 / B21 / O20)
# ---------------------------------------------------------------------------

def _pnl_frame(seed=13, n=6000, concentrate=True):
    rng = np.random.default_rng(seed)
    field = rng.integers(5, 18, n)
    price = np.exp(rng.normal(1.4, 0.6, n)) + 1.0
    p = 1.0 / price
    p = np.where((field <= 8) & concentrate, p * 1.35, p * 0.98)
    won = rng.random(n) < np.clip(p, 0, 1)
    return pd.DataFrame({"raceid": [f"r{i // 3}" for i in range(n)],
                         "model_price": price, "field": field,
                         "going": rng.choice(["good", "soft", "firm"], n),
                         "track": rng.choice(["A", "B", "C"], n), "won": won,
                         "ret_back": np.where(won, (price - 1) * 0.95, -1.0)})


def test_factor_attribution_names_the_factor_the_pnl_actually_came_from():
    res = factor_attribution(_pnl_frame())
    info = res.attrs["attribution"]
    assert info["largest_single_level"].startswith("field_band=")
    assert info["largest_level_pnl_share_pct"] > 60.0           # an accidental bet on small fields
    top = res.iloc[0]
    assert top["factor"] == "field_band" and top["t"] > 2.0
    assert set(res["factor"]) >= {"field_band", "price_band", "going", "course"}
    assert info["clusters"] == 2000                             # SEs clustered on the race


def test_factor_attribution_finds_nothing_when_there_is_nothing_to_find():
    res = factor_attribution(_pnl_frame(seed=99, concentrate=False))
    assert (res["t"].abs() < 3.0).all()
    assert res.attrs["attribution"]["largest_level_pnl_share_pct"] < 200.0
    assert factor_attribution(pd.DataFrame(columns=["ret_back"])).empty


def _decay_frame(slope=0.011, days=540, seed=4):
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(days):
        since = day % 90
        strength = max(1.0 - slope * since, 0.0)
        for r in range(8):
            s = rng.normal(0, 1, 9)
            truth = np.exp(s) / np.exp(s).sum()
            w = int(rng.choice(9, p=truth))
            pm = np.exp(strength * s) / np.exp(strength * s).sum()
            when = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
            for i in range(9):
                rows.append({"race_date": when, "raceid": f"d{day}r{r}", "p_model": pm[i],
                             "won": i == w})
    return pd.DataFrame(rows)


REFITS = [pd.Timestamp("2024-01-01") + pd.Timedelta(days=90 * k) for k in range(7)]


def test_alpha_decay_measures_the_decay_and_sets_a_refit_horizon():
    """Skill is built to run out at day 1/0.011 = 91; the fitted horizon should
    land near it, and the slope should be unambiguously negative."""
    out = alpha_decay(_decay_frame(), refit_dates=REFITS, market_col=None, ret_col=None)
    info = out.attrs["alpha_decay"]
    assert info["reference"] == "uniform 1/n"
    assert out["delta_r2"].iloc[0] > out["delta_r2"].iloc[-1]
    assert info["slope_per_30d"] < 0 and info["slope_t"] < -3
    assert 60 < info["days_to_zero"] < 160
    assert out["races"].sum() == 540 * 8


def test_alpha_decay_finds_no_slope_in_a_model_that_does_not_decay():
    info = alpha_decay(_decay_frame(slope=0.0), refit_dates=REFITS, market_col=None,
                       ret_col=None).attrs["alpha_decay"]
    assert abs(info["slope_t"]) < 3.0
    assert info["days_to_zero"] > 500


def test_alpha_decay_uses_the_market_as_the_reference_when_it_is_there():
    d = _decay_frame(days=90)
    d["p_market"] = d["p_model"]                                # model == market -> no edge
    out = alpha_decay(d, refit_dates=REFITS, ret_col=None)
    assert out.attrs["alpha_decay"]["reference"] == "market"
    assert np.allclose(out["delta_r2"], 0.0, atol=1e-9)
    assert out.attrs["alpha_decay"]["days_to_zero"] == 0.0


# ---------------------------------------------------------------------------
# The phi ladder and Baker-McHale shrinkage (V.1 / V.2, N2 / N3 / N4)
# ---------------------------------------------------------------------------

def test_full_kelly_is_refused_and_the_ladder_tops_out_at_a_half():
    assert check_phi(0.25) == 0.25 and check_phi(0.5) == 0.5
    for bad in (1.0, 0.75, 0.0, -0.1):
        with pytest.raises(ValueError):
            check_phi(bad)
    assert max(PHI_LADDER) == 0.5 and PHI_DEFAULT == 0.25


def test_phi_rises_with_confidence_and_never_passes_its_ceiling():
    conf = np.array([0.0, 0.2, 0.5, 0.9, 1.0])
    phi = phi_from_confidence(conf)
    assert np.all(np.diff(phi) > 0) and phi.max() == pytest.approx(PHI_DEFAULT)
    assert phi[0] == 0.0
    rungs = phi_from_confidence(conf, ladder=PHI_LADDER)
    assert set(np.unique(rungs[1:])) <= set(PHI_LADDER)
    assert rungs.max() == pytest.approx(PHI_DEFAULT)
    assert phi_from_confidence(conf, phi_max=0.5).max() == pytest.approx(0.5)
    with pytest.raises(ValueError):
        phi_from_confidence(conf, phi_max=0.01, ladder=PHI_LADDER)


def test_baker_mchale_pulls_the_uncertain_runner_toward_the_field():
    d = pd.DataFrame({"raceid": ["r1"] * 4, "p_model": [0.5, 0.25, 0.15, 0.10],
                      "bsp": [2.5, 5.0, 8.0, 12.0], "kf_sd": [0.05, 0.05, 0.8, 2.0]})
    out = baker_mchale_shrinkage(d)
    assert out["p_shrunk"].sum() == pytest.approx(1.0)          # still a probability vector
    assert out["confidence"].is_monotonic_decreasing
    # the defining identity: within the race, every log-deviation from the mean
    # is multiplied by that runner's credibility weight (renormalising adds the
    # same constant to all four, so it cancels in the differences)
    s, w = np.log(out["p_model"].to_numpy()), out["confidence"].to_numpy()
    dev = s - s.mean()
    s2 = np.log(out["p_shrunk"].to_numpy())
    assert np.allclose(s2 - s2[0], w * dev - w[0] * dev[0], atol=1e-9)
    # the wide posterior is the one that actually moves, and it moves up toward the field
    moved = np.abs(s2 - s)
    assert moved[3] > moved[0] and moved[3] > moved[2]
    assert out["p_shrunk"].iloc[3] > out["p_model"].iloc[3]
    assert out["p_shrunk"].iloc[0] < out["p_model"].iloc[0]


def test_zero_posterior_variance_changes_nothing():
    d = pd.DataFrame({"raceid": ["r1"] * 3, "p_model": [0.6, 0.3, 0.1], "kf_sd": [0.0, 0.0, 0.0]})
    out = baker_mchale_shrinkage(d)
    assert np.allclose(out["p_shrunk"], out["p_model"])
    assert np.allclose(out["confidence"], 1.0)
    assert np.allclose(baker_mchale_shrinkage(d.drop(columns=["kf_sd"]))["p_shrunk"], d["p_model"])


def test_confidence_stakes_are_smaller_where_the_model_is_less_sure():
    d = pd.DataFrame({"raceid": ["r1"] * 3, "p_model": [0.5, 0.3, 0.2],
                      "bsp": [4.0, 6.0, 9.0], "kf_sd": [0.02, 0.02, 1.5]})
    out = confidence_stakes(d, commission=0.0)
    assert (out["stake_fraction"] <= PHI_DEFAULT * out["kelly_shrunk"] + 1e-12).all()
    assert out["phi"].iloc[2] < out["phi"].iloc[0]
    assert out["stake_fraction"].max() < 1.0
    with pytest.raises(ValueError):
        confidence_stakes(d, phi_max=1.0)


def test_shrinkage_accepts_a_probability_scale_posterior():
    d = pd.DataFrame({"raceid": ["r1"] * 3, "p_model": [0.5, 0.3, 0.2], "sd_p": [0.01, 0.01, 0.18]})
    out = baker_mchale_shrinkage(d, sd_col="sd_p", sd_is_probability=True)
    assert out["confidence"].iloc[2] < out["confidence"].iloc[0]
    assert out["p_shrunk"].sum() == pytest.approx(1.0)
