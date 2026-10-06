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


# ---------------------------------------------------------------------------------------------------------------
# The record read: declared dividends and the join to Betfair (made-up answers in the shape the API returns)
# ---------------------------------------------------------------------------------------------------------------

def _result(race_id, track="BRIGHTON", post="2026-10-06T12:38:00Z", win=(5, 4.2), abandoned=False):
    placed = [("3", "Launceston", 1, 500, win, (1.9, 1.9)), ("2", "Jenson Benson", 2, 600, (6, 5), (1.7, 1.7)),
              ("7", "Private Project", 3, 500, (5, 5), (1.9, 1.9))]
    return {"raceId": str(race_id), "trackName": track, "postTime": post, "runners": 9, "nonRunners": [10],
            "abandoned": abandoned,
            "placeRunners": [{"programNumber": c, "name": n, "finishingPosition": f, "decimalShowPrice": sp,
                              "winDividend": {"dividend": w[0], "baseDividend": w[1], "paid": True},
                              "placeDividend": {"dividend": p[0], "baseDividend": p[1], "paid": True}}
                             for c, n, f, sp, w, p in placed],
            "pools": [{"name": "WIN", "total": 5792.08, "dividends": [5], "baseDividends": [5]},
                      {"name": "PLACE", "total": 1704.6, "dividends": [1.9, 1.7, 1.9], "baseDividends": [1.9, 1.7, 1.9]},
                      {"name": "EXACTA", "total": 3579.42, "dividends": [19.2], "baseDividends": [19.2]},
                      {"name": "SWINGER", "total": 725.66, "dividends": [3.3, 4.1, 3.9]},
                      {"name": "PLACEPOT", "total": 58893.61, "estimatedDividends": 89.2}]}


def test_the_declared_dividends_are_read_from_each_races_latest_answer():
    early = {"polled_utc": "2026-10-06T12:48:00+00:00", "answer": {"results": [_result(1, win=(5, 4.0))]}}
    late = {"polled_utc": "2026-10-06T13:08:00+00:00",
            "answer": {"results": [_result(1), _result(2, abandoned=True)]}}
    broken = {"polled_utc": "2026-10-06T13:09:00+00:00", "answer": None}
    runners, pools = tc.declared([late, broken, early])
    assert set(runners.race_id) == {"1"}                                       # the abandoned race is left out
    w = runners[runners.finish == 1].iloc[0]
    assert (w.cloth, w["name"], w.sp, w.win, w.win_base) == ("3", "Launceston", 5.0, 5.0, 4.2)   # the later answer
    assert runners.place.tolist() == [1.9, 1.7, 1.9] and runners.non_runners.iloc[0] == 1
    sw = pools[pools.pool == "SWINGER"].iloc[0]
    assert sw.dividends == [3.3, 4.1, 3.9] and sw.base_dividends == [] and sw.total == pytest.approx(725.66)
    assert pools[pools.pool == "PLACEPOT"].iloc[0].dividends == []


def test_tote_races_and_runners_find_their_betfair_markets():
    import pandas as pd
    tote = pd.DataFrame({"race_id": ["1", "2", "3"], "track": ["BRIGHTON", "CHELMSFORD", "BRIGHTON"],
                         "post_utc": ["2026-10-06T12:38:00Z", "2026-10-06T18:00:00Z", "2026-10-06T15:00:00Z"]})
    cat = pd.DataFrame({
        "market_id": ["1.10", "1.10", "1.11", "1.20", "1.20"], "market_type": ["WIN", "WIN", "PLACE", "WIN", "WIN"],
        "venue": ["Brighton", "Brighton", "Brighton", "Chelmsford City", "Chelmsford City"],
        "market_start_utc": ["2026-10-06T12:38:00.000Z"] * 3 + ["2026-10-06T18:01:00.000Z"] * 2,
        "selection_id": [11, 12, 11, 21, 22], "cloth_number": ["3", None, "3", "1", "2"],
        "runner_name": ["Launceston", "Jenson Benson (IRE)", "Launceston", "Alpha", "Beta"]})
    m = tc.match_races(tote, cat)
    assert m == {"1": "1.10", "2": "1.20"}                     # race 3 has no market at its off
    assert tc.match_races(tote, cat, market_type="PLACE") == {"1": "1.11"}
    rows = pd.DataFrame({"race_id": ["1", "1", "2", "3"], "cloth": ["3", "2", "9", "3"],
                         "name": ["Launceston", "Jenson Benson", "Beta", "Launceston"]})
    assert tc.selection_ids(rows, cat, m).tolist() == [11, 12, 22, None]   # by cloth, by name, by name, no market
