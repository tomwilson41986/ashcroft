"""The Tote-against-Betfair arithmetic by hand (made-up prices; nothing here is scored)."""

import numpy as np
import pytest

import tote_compare as tc
from model.ordering import top3_probabilities, trifecta_probability


def test_the_hedge_pays_only_on_winners_above_the_break_even():
    # laid at 10 with 2% commission, the Tote must return 1 + 9/0.98 = 10.1837 on a winner to break even
    assert tc.breakeven(10.0) == pytest.approx(1 + 9 / 0.98)
    assert tc.hedged_return(12.0, 10.0, won=False) == 0.0
    assert tc.hedged_return(12.0, 10.0, won=True) == pytest.approx(12.0 - (1 + 9 / 0.98))
    assert tc.hedged_return(9.0, 10.0, won=True) < 0
    assert tc.edge(tc.breakeven(10.0), 10.0) == pytest.approx(0.0)
    # S on the Tote and S/(1 - c) laid: a loser costs S on the Tote and the lay keeps S/(1 - c) less commission
    s, lay, c = 10.0, 6.0, 0.02
    lay_stake = s / (1 - c)
    assert -s + lay_stake * (1 - c) == pytest.approx(0.0)
    assert s * (8.0 - 1) - lay_stake * (lay - 1) == pytest.approx(s * float(tc.hedged_return(8.0, lay, True)))


def test_a_direct_bet_carries_the_guarantee_and_tote_plus():
    assert tc.direct_return(5.0, "win", sp=6.0) == pytest.approx(6.0)          # paid the SP, the bigger
    assert tc.direct_return(5.0, "win", sp=4.0) == pytest.approx(5.5)          # Tote+ on the dividend
    assert tc.direct_return(1.15, "win") == pytest.approx(1.15)                # no Tote+ at 1.20 or under
    assert tc.direct_return(20.0, "trifecta") == pytest.approx(21.0)           # 5% on the exotics
    assert tc.direct_return(5.0, "win", sp=6.0, plus=False, guarantee=False) == pytest.approx(5.0)
    np.testing.assert_allclose(tc.direct_return([2.0, 30.0], "place", plus=False), [2.0, 30.0])


@pytest.mark.parametrize("n,handicap,places", [(4, False, 0), (5, False, 2), (7, True, 2), (8, False, 3),
                                                (15, True, 3), (16, False, 3), (16, True, 4)])
def test_place_terms(n, handicap, places):
    assert tc.place_terms(n, handicap) == places


def test_implied_shares():
    assert tc.implied_share(4.0, "win") == pytest.approx(0.8075 / 4)
    assert tc.implied_share(20.0, "exacta") == pytest.approx(0.75 / 20)
    assert tc.implied_share(7.0, "swinger") == pytest.approx(0.70 / 7 / 3)


P = tc.market_probabilities([2.5, 4.0, 6.0, 9.0, 15.0, 26.0])


@pytest.mark.parametrize("gamma,delta", [(1.0, 1.0), (0.81, 0.65)])
def test_the_trifecta_tensor_is_the_three_way_form(gamma, delta):
    t = tc.trifecta_tensor(P, gamma, delta)
    assert t.sum() == pytest.approx(1.0)
    assert t[0, 1, 2] == pytest.approx(trifecta_probability(P, 0, 1, 2, gamma, delta))
    assert t[3, 0, 5] == pytest.approx(trifecta_probability(P, 3, 0, 5, gamma, delta))


@pytest.mark.parametrize("gamma,delta", [(1.0, 1.0), (0.81, 0.65)])
def test_the_swinger_matrix(gamma, delta):
    m = tc.swinger_matrix(P, gamma, delta)
    np.testing.assert_allclose(m, m.T)
    assert np.all(np.diag(m) == 0)
    assert m[np.triu_indices(len(P), 1)].sum() == pytest.approx(3.0)       # three winning pairs a race
    p1, p2, p3 = top3_probabilities(P, gamma, delta)
    np.testing.assert_allclose(m.sum(axis=1), 2 * (p1 + p2 + p3))


def test_fair_dividends():
    ex = tc.fair_dividends(P, "exacta")
    assert np.isinf(ex[0, 0]) and ex[0, 1] == pytest.approx(1 / (P[0] * P[1] / (1 - P[0])))
    sw = tc.fair_dividends(P, "swinger")
    assert sw[0, 1] == pytest.approx(1 / tc.swinger_matrix(P)[0, 1])
    with pytest.raises(ValueError):
        tc.fair_dividends(P, "win")
