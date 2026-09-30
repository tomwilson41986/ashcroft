"""The race book (model/race_book.py): the closing-price fit, the closing draws, the book under the owner's
constraints, the dutched subset and settlement.

Mechanics only, on small stand-in races with known answers; the model is scored on real prices by the
research query, not here."""

import numpy as np
import pandas as pd
import pytest

from model import race_book as rb
from model.portfolio import race_kelly_exact


def _race(prices):
    return np.asarray(prices, float)


def test_group_log_softmax_is_a_book_of_one_per_race():
    g = np.array([0, 0, 0, 1, 1])
    q = np.exp(rb.group_log_softmax(np.array([1.0, 2.0, 3.0, -5.0, 5.0]), g))
    assert np.allclose(np.bincount(g, weights=q), 1.0)


def test_the_closing_fit_learns_how_far_to_trust_the_market():
    # stand-in races whose closing book is exactly 0.7 x market + 0.3 x model in log probability
    rng = np.random.default_rng(3)
    rows = []
    for r in range(400):
        n = 8
        lm, lf = rng.normal(0, 1, n), rng.normal(0, 1, n)
        u = 0.7 * lm + 0.3 * lf
        q = np.exp(u - u.max()); q /= q.sum()
        pm, pf = np.exp(lm) / np.exp(lm).sum(), np.exp(lf) / np.exp(lf).sum()
        for i in range(n):
            rows.append({"race": r, "morningwap": 1 / pm[i], "predicted_bfsp": 1 / pf[i],
                         "morning_vol": 500.0, "bsp": 1 / q[i]})
    d = pd.DataFrame(rows)
    g = rb.race_codes(d["race"])
    inputs = rb.closing_inputs(d)
    m = rb.fit_closing_model(inputs, d["bsp"], g)
    assert m.beta[0] == pytest.approx(0.7, abs=0.02) and m.beta[1] == pytest.approx(0.3, abs=0.02)
    assert np.allclose(np.bincount(g, weights=m.predict(inputs, g)), 1.0)
    assert m.book == pytest.approx(1.0)


def test_closing_draws_sum_to_the_book_and_centre_on_the_forecast():
    q = np.array([0.5, 0.3, 0.2])
    draws = rb.closing_draws(q, np.full(3, 0.1), book=1.02, n_draws=20000, rng=np.random.default_rng(0))
    assert np.allclose(draws.sum(axis=1), 1.02)
    assert np.allclose(draws.mean(axis=0) / 1.02, q, atol=0.005)


def test_no_book_when_nothing_clears_the_race_floor():
    back = _race([2.0, 3.0, 6.0])
    A = rb.close_payoffs(np.tile(1 / back, (50, 1)), back)          # every price closes where it is now
    b = rb.build_book(A, np.ones(50), back, theta=0.02)
    assert b.exposure == 0 and b.n_positions == 0


def test_the_book_keeps_the_owner_constraints():
    back = _race([2.0, 4.5, 6.0, 8.0, 12.0, 21.0])
    q = np.array([0.36, 0.22, 0.165, 0.12, 0.08, 0.055]); q /= q.sum()
    draws = rb.closing_draws(q, np.full(6, 0.2), n_draws=2000, rng=np.random.default_rng(1))
    A = rb.close_payoffs(draws, back)
    b = rb.build_book(A, np.ones(len(A)), back, delta=0.03, theta=0.02, exposure_cap=1.0, stake_cap=0.25)
    risk = np.concatenate([np.ones(6), back - 1])
    x = np.concatenate([b.back, b.lay])
    assert risk @ x <= 1.0 + 1e-6                                   # the race's exposure cap
    assert b.edges @ x >= 0.02 * (risk @ x) - 1e-9                  # positive price expectation on the race
    assert np.all(b.edges[x > 1e-9] >= -0.03 - 1e-12)               # small underlays at most
    assert np.all(x <= 0.25 + 1e-9)
    assert b.lay[0] > 0                                             # the poor-value favourite is opposed


def test_a_lone_overlay_is_backed_and_the_poor_value_favourite_is_not():
    back = _race([2.0, 4.5, 6.0, 8.0, 12.0, 21.0])
    q = np.array([0.36, 0.22, 0.165, 0.12, 0.08, 0.055]); q /= q.sum()
    draws = rb.closing_draws(q, np.full(6, 0.2), n_draws=2000, rng=np.random.default_rng(1))
    A = rb.close_payoffs(draws, back)
    b = rb.build_book(A, np.ones(len(A)), back, allow_lay=np.zeros(6, bool), exposure_cap=1.0)
    assert b.back[5] > 0 and b.back[0] == 0 and b.lay.sum() == 0


def test_hold_mode_matches_the_closed_form_joint_kelly_without_constraints():
    back = _race([2.0, 3.6, 6.5, 11.0])                             # a 102% book
    p = np.array([0.40, 0.29, 0.18, 0.13])
    exact = race_kelly_exact(p, back, commission=0.0)
    assert exact[0] == 0 and np.all(exact[1:] > 0)                  # three of the four, the favourite left out
    b = rb.build_book(rb.result_payoffs(back), p, back, delta=1.0, theta=0.0, exposure_cap=0.99,
                      allow_lay=np.zeros(4, bool), commission=0.0)
    assert np.allclose(b.back, exact, atol=2e-3)


def test_the_dutch_book_takes_level_or_small_underlays_while_the_race_stays_positive():
    back = _race([3.0, 4.0, 5.0, 8.0])
    q = np.array([0.40, 0.25, 0.19, 0.16])
    e = back * q - 1                                                # +20%, 0%, -5%, +28%
    s = rb.dutch_book(back, e, q, delta=0.03, theta=0.02)
    assert s[3] > 0 and s[0] > 0 and s[1] > 0 and s[2] == 0          # level horse in, the 5% underlay out
    assert s.sum() == pytest.approx(1.0)
    ret = s[s > 0] * back[s > 0]
    assert np.allclose(ret, ret[0])                                 # the same return whichever wins
    chosen = s > 0
    assert q[chosen].sum() / (1 / back[chosen]).sum() - 1 >= 0.02


def test_settlement_charges_commission_once_on_the_race_net():
    back = _race([2.0, 5.0])
    # backed both at their prices; they close at 2.5 and 4.0: +(2/2.5-1) + (5/4-1) = -0.2 + 0.25
    v = rb.settle_close([1.0, 1.0], [0.0, 0.0], back, back, [2.5, 4.0], commission=0.05)
    assert v == pytest.approx(0.05 * 0.95)
    # held: the 5.0 shot wins; +4 - 1 = +3, less 5%
    assert rb.settle_result([1.0, 1.0], [0.0, 0.0], back, back, 1, commission=0.05) == pytest.approx(3 * 0.95)
    # a lay of the favourite at 2.0 when the other wins: +1, less 5%; when it wins: -1
    assert rb.settle_result([0, 0], [1.0, 0], back, back, 1) == pytest.approx(0.95)
    assert rb.settle_result([0, 0], [1.0, 0], back, back, 0) == pytest.approx(-1.0)


def _priced(n_months=3, races_per_month=6, seed=4):
    """Stand-in races, eight runners each, over consecutive months: morning prices, forecasts, BSPs, a winner."""
    rng = np.random.default_rng(seed)
    rows = []
    for mth in range(n_months):
        for k in range(races_per_month):
            race = f"2026-0{mth + 1}-10|York|{k}"
            q = rng.dirichlet(np.ones(8) * 2)
            w = rng.choice(8, p=q)
            for i in range(8):
                rows.append({"race": race, "race_date": f"2026-0{mth + 1}-10", "horse_name": f"h{k}{i}",
                             "morningwap": 1 / q[i] * 1.02, "predicted_bfsp": 1 / q[i] * rng.uniform(0.8, 1.25),
                             "bsp": 1 / q[i] * rng.uniform(0.85, 1.15), "morning_vol": 500.0, "won": i == w})
    return pd.DataFrame(rows)


def test_whole_races_keeps_only_races_priced_for_every_runner():
    d = _priced(1, 3)
    d.loc[d.index[0], "morningwap"] = np.nan                        # one runner of the first race unpriced
    field = d.groupby("race").size()
    w = rb.whole_races(d, field=field)
    assert set(w["race"]) == set(d["race"].unique()[1:])


def test_the_walk_forward_scores_only_months_with_enough_behind_them():
    d = _priced(3, 6)
    s = rb.expected_clv_walk_forward(d, n_draws=50, min_train_races=6)
    assert sorted(s["month"].unique()) == ["2026-02", "2026-03"]    # January is only fitted on
    again = rb.expected_clv_walk_forward(d, n_draws=50, min_train_races=6)
    assert np.allclose(s["ev"], again["ev"])                        # seeded: a run repeats exactly
    assert rb.expected_clv_walk_forward(d, n_draws=50, min_train_races=12)["month"].unique().tolist() == ["2026-03"]


def test_the_to_win_selection_stakes_and_settles_as_the_owner_does():
    s = pd.DataFrame({"race": ["a", "a", "a", "b"], "race_date": ["2026-02-01"] * 4,
                      "morningwap": [3.0, 6.0, 11.0, 5.0], "bsp": [2.5, 6.0, 12.0, 4.0],
                      "ev": [0.10, 0.03, 0.02, 0.2], "morning_vol": [500.0, 500.0, 500.0, 50.0],
                      "won": [False, True, False, True]})
    g = rb.to_win_selection(s, bar=0.03)
    assert list(g["race"]) == ["a"]                                 # race b's horse had too little matched
    stakes = np.array([250 / 2, 250 / 5])                           # to win 250 at 3.0 and 6.0
    assert g["bets"].iat[0] == 2 and g["staked"].iat[0] == pytest.approx(stakes.sum())
    clv = stakes @ (np.array([3.0, 6.0]) / np.array([2.5, 6.0]) - 1)
    assert g["clv"].iat[0] == pytest.approx(clv * 0.95)             # commission on the race's net
    held = 250.0 - stakes[0]                                        # the 6.0 winner pays 250, the other stake lost
    assert g["result"].iat[0] == pytest.approx(held * 0.95)
    assert g["expected"].iat[0] == pytest.approx(stakes @ np.array([0.10, 0.03]))
