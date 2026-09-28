"""Staking and pricing for automated trading (trading/staking.py, trading/pricing.py).

Mechanics only, on hand-made races: nothing is trained or evaluated here.
"""
import numpy as np
import pytest

from trading import pricing, staking


def test_effective_odds_and_edge_take_commission_off_the_winnings():
    assert staking.effective_odds(3.0, 0.05) == pytest.approx(1 + 2 * 0.95)
    assert staking.edge(0.5, 3.0, 0.0) == pytest.approx(0.5)
    assert staking.edge(1 / 3, 3.0, 0.0) == pytest.approx(0.0)              # a fair price, no commission
    assert staking.edge(1 / 3, 3.0, 0.05) < 0                                # the commission makes it a loser


def test_fixed_win_stakes_win_the_same_amount_on_every_runner():
    prices = np.array([1.5, 3.0, 11.0, 51.0])
    stakes = staking.fixed_win_stakes(prices, target=10.0, commission=0.05)
    np.testing.assert_allclose(stakes * (prices - 1) * 0.95, 10.0)
    assert stakes[0] > stakes[1] > stakes[2] > stakes[3]                     # most on the favourite


def test_single_kelly_backs_only_an_edge_and_scales_with_the_fraction():
    f = staking.single_kelly([0.5, 0.2], [3.0, 4.0], commission=0.0)
    assert f[0] == pytest.approx((0.5 * 2 - 0.5) / 2) and f[1] == 0.0      # 0.2 at 4.0 has no edge
    np.testing.assert_allclose(staking.single_kelly([0.5], [3.0], 0.0, fraction=0.25), f[:1] * 0.25)


def test_race_kelly_matches_the_single_runner_formula_when_one_runner_is_backed():
    p = np.array([0.5, 0.3, 0.2])
    price = np.array([3.0, 2.4, 3.5])            # a book over 100%; only the first has an edge
    f = staking.race_kelly(p, price, commission=0.0)
    # one runner in: R = (1 - 0.5) / (1 - 1/3) = 0.75, the next returns 0.72 < R; f = 0.5 - 0.75 / 3 = 0.25
    assert f[0] == pytest.approx(0.25) and f[1] == 0.0 and f[2] == 0.0
    assert f[0] == pytest.approx(staking.single_kelly([0.5], [3.0], 0.0)[0])


def test_race_kelly_backs_every_runner_in_proportion_when_the_book_is_under_100_percent():
    """Prices of 3, 3 and 4 make a book of 91.7%: an arbitrage, so Kelly stakes the whole bank
    in proportion to the chances (f = p). Betfair's back books sit at or above 100%."""
    p = np.array([0.5, 0.3, 0.2])
    np.testing.assert_allclose(staking.race_kelly(p, [3.0, 3.0, 4.0], 0.0), p)


def test_race_kelly_takes_a_slight_underlay_when_it_raises_the_growth_of_the_bank():
    """The second runner is priced 2% short of fair on its own (p * O = 0.98), but beside the
    first it pays when the first loses: race Kelly backs it, single-runner Kelly does not."""
    p = np.array([0.30, 0.35, 0.35])
    price = np.array([5.0, 0.98 / 0.35, 2.0])
    f = staking.race_kelly(p, price, commission=0.0)
    assert f[0] > 0 and f[1] > 0 and f[2] == 0.0
    assert staking.edge(0.35, price[1], 0.0) < 0                             # an underlay on its own
    assert staking.single_kelly(p, price, 0.0)[1] == 0.0
    grown = staking.race_log_growth(p, price, f, 0.0)
    alone = staking.race_log_growth(p, price, np.array([staking.race_kelly(p[:1] / 1, price[:1], 0.0)[0], 0, 0]), 0.0)
    assert grown > alone                                                    # the hedge is worth having
    # and the race's expectation stays positive with it
    assert staking.race_expectation(p, price, f, 0.0) > 0


def test_race_kelly_is_the_maximum_of_the_expected_log_growth():
    rng = np.random.default_rng(3)
    for _ in range(20):
        p = rng.dirichlet(np.ones(6))
        price = 1 / (p * rng.uniform(0.8, 1.25, 6))                          # some over, some under fair
        f = staking.race_kelly(p, price, commission=0.02)
        best = staking.race_log_growth(p, price, f, 0.02)
        for _ in range(50):                                                  # no nearby staking does better
            g = np.clip(f + rng.normal(0, 0.01, 6) * (rng.random(6) < 0.5), 0, None)
            if g.sum() < 1:
                assert staking.race_log_growth(p, price, g, 0.02) <= best + 1e-9


def test_race_kelly_backs_nothing_when_no_runner_has_an_edge():
    p = np.array([0.5, 0.3, 0.2])
    np.testing.assert_array_equal(staking.race_kelly(p, 0.95 / p, 0.0), 0.0)
    np.testing.assert_array_equal(staking.race_kelly([], [], 0.0), [])


def test_market_probabilities_take_out_the_overround_and_the_pool_sits_between():
    race = np.array([1, 1, 1, 2, 2])
    price = np.array([2.0, 3.0, 5.0, 1.5, 3.0])
    pk = pricing.market_probabilities(price, race)
    assert pk[:3].sum() == pytest.approx(1) and pk[3:].sum() == pytest.approx(1)
    pm = np.array([0.6, 0.25, 0.15, 0.7, 0.3])
    for w, want in ((0.0, pk), (1.0, pm)):
        np.testing.assert_allclose(pricing.pooled_probabilities(pm, pk, race, w), want)
    half = pricing.pooled_probabilities(pm, pk, race, 0.5)
    assert half[:3].sum() == pytest.approx(1)
    assert min(pm[0], pk[0]) < half[0] < max(pm[0], pk[0])
    with pytest.raises(ValueError):
        pricing.pooled_probabilities(pm, pk, race, 1.5)
