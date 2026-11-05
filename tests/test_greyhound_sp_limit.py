"""research/greyhound/sp_limit.py: the temper, Harville's place chance and the limit-on-close arithmetic."""
import numpy as np
import pandas as pd

from research.greyhound import sp_limit as sl


def test_temper_normalises_and_identity_at_one():
    p = np.array([0.5, 0.3, 0.2, 0.6, 0.4])
    race = np.array(["a", "a", "a", "b", "b"])
    t = sl.temper(p, race, 1.0)
    assert np.allclose(t, p)
    t2 = sl.temper(p, race, 2.0)
    assert np.isclose(t2[:3].sum(), 1) and t2[0] > p[0]


def test_place_chance_sums_to_the_places():
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.dirichlet(np.ones(6)), rng.dirichlet(np.ones(5))])
    race = np.array(["a"] * 6 + ["b"] * 5)
    k = np.array([2] * 6 + [3] * 5)
    pc = sl.place_chance(p, race, k)
    assert np.isclose(pc[:6].sum(), 2) and np.isclose(pc[6:].sum(), 3)


def test_a_limit_on_close_back_is_matched_only_at_or_above_its_limit():
    n = 120
    d = pd.DataFrame({"race_date": ["2025-01-01"] * 60 + ["2025-01-02"] * 60, "year": 2025})
    p = np.full(n, 0.25)                                  # fair 4.0; limit at m=0.2 is 4.8
    sp = np.array([5.0, 4.0] * 60)
    hit = np.array([1, 0, 0, 0] * 30)
    rows = sl._rules(d, p, sp, hit, "win", "p")
    r = [x for x in rows if x["side"] == "back" and x["margin"] == 0.2 and x["comm"] == 0.02][0]
    assert r["holdout"]["bets"] == 60 and np.isclose(r["holdout"]["mean_sp"], 5.0)
    # the 5.0 runners are rows 0, 2, 4...: hit on rows 0, 4, 8... = every other one of them
    assert np.isclose(r["holdout"]["roi%"], 100 * (0.5 * 4 * 0.98 - 0.5))
