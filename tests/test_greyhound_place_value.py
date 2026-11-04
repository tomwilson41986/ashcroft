"""research/greyhound/place_value.py: Harville's place chances, the race filter and the rules' arithmetic."""
import itertools

import numpy as np
import pandas as pd

from research.greyhound import place_value as pv


def _brute(q, k):
    """P(i in the first k) by enumerating every finishing order under Harville."""
    n = len(q)
    out = np.zeros(n)
    for order in itertools.permutations(range(n)):
        pr, left = 1.0, 1.0
        for i in order:
            pr *= q[i] / left
            left -= q[i]
        for i in order[:k]:
            out[i] += pr
    return out


def test_harville_matches_enumeration_and_sums_to_the_places():
    rng = np.random.default_rng(1)
    for n, k in ((6, 2), (6, 3), (5, 2), (8, 3)):
        q = rng.dirichlet(np.ones(n))
        Q = np.zeros((1, pv.MAXN))
        Q[0, :n] = q
        h = pv.harville(Q, np.array([k]))[0, :n]
        assert np.allclose(h, _brute(q, k), atol=1e-9)
        assert np.isclose(h.sum(), k)


def test_only_whole_races_are_kept():
    rows = []
    for race, (winners, placed) in {"a": (1, 2), "b": (2, 2), "c": (1, 1)}.items():
        for trap in range(1, 7):
            rows.append({"race_date": "2025-01-01", "race_time": f"1{len(race)}:{ord(race)}", "track": race,
                         "trap": trap, "ppwap_w": 4.0, "bsp_p": 2.0, "bsp_w": 4.0,
                         "win_lose_w": int(trap <= winners), "win_lose_p": int(trap <= placed)})
    j, why = pv.clean(pd.DataFrame(rows))
    assert set(j.track) == {"a"} and why["kept races"] == 1


def test_the_rules_pay_as_betfair_would():
    t = pd.DataFrame({"race_date": ["2025-01-01"] * 40 + ["2025-01-02"] * 40, "bsp_p": 3.0,
                      "win_lose_p": [1, 0] * 40, "p": 0.6})
    r = [x for x in pv.score(t, "p") if x["comm"] == 0.02 and x["side"] == "back" and x["edge>"] == 0.0][0]
    assert r["bets"] == 80 and np.isclose(r["roi_stake%"], 100 * (0.5 * 2 * 0.98 - 0.5))
    assert not [x for x in pv.score(t, "p") if x["side"] == "lay" and x.get("roi_stake%") is not None]
