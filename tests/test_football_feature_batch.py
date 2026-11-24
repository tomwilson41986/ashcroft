"""The 24 Nov feature blocks' helpers: the time decay, the lag to the collection day, the split record, the simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football.blocks import _util as u
from football.blocks import sim


def side_rows(dates, values, team="T"):
    return pd.DataFrame({"team": team, "date": pd.to_datetime(dates), "match_id": range(len(dates)), "v": values,
                         "venue": "H"})


def test_the_decayed_mean_weighs_by_days_and_never_sees_the_match_itself():
    # Saturdays, a week apart: each its own collection window
    s = side_rows(["2026-01-03", "2026-01-10", "2026-01-17"], [1.0, 3.0, 100.0])
    v, w = u.decayed(s, ["v"], 7.0)
    assert np.isnan(v["v"].iloc[0]) and w[0] == 0
    assert v["v"].iloc[1] == pytest.approx(1.0) and w[1] == pytest.approx(0.5)
    # at the third match: the first is two half-lives old (0.25), the second one (0.5)
    assert v["v"].iloc[2] == pytest.approx((0.25 * 1.0 + 0.5 * 3.0) / 0.75) and w[2] == pytest.approx(0.75)


def test_a_second_match_in_the_same_collection_window_takes_the_first_ones_values():
    # Saturday and Monday share Friday's collection; the Monday match must not see Saturday's result
    s = side_rows(["2026-01-03", "2026-01-10", "2026-01-12"], [1.0, 50.0, 2.0])
    v, _ = u.decayed(s, ["v"], 30.0)
    assert v["v"].iloc[2] == pytest.approx(v["v"].iloc[1]) == pytest.approx(1.0)
    first = side_rows(["2026-01-03", "2026-01-05"], [9.0, 1.0])            # a side's very first window: both blank
    assert u.decayed(first, ["v"], 30.0)[0]["v"].isna().all()


def test_the_split_record_counts_only_earlier_matches_of_its_kind():
    s = side_rows(["2026-01-03", "2026-01-10", "2026-01-17", "2026-01-24"], [1.0, 2.0, 4.0, 8.0])
    kind = np.array(["a", "b", "a", "b"])
    rec = u.split_record(s, "v", kind, ("a", "b"), 10)
    assert rec["a"][0].tolist() == [0.0, 1.0, 1.0, 5.0] and rec["a"][1].tolist() == [0, 1, 1, 2]
    assert rec["b"][0].tolist() == [0.0, 0.0, 2.0, 2.0]


def test_the_simulated_chances_are_the_poisson_score_matrix():
    h, d, a = sim.chances(np.array([1.5, 1.0]), np.array([1.5, 2.0]))
    assert h[0] == pytest.approx(a[0]) and h[0] + d[0] + a[0] == pytest.approx(1.0)
    assert a[1] > h[1]
    from scipy.stats import poisson
    g = np.arange(11)
    m = np.outer(poisson.pmf(g, 1.0), poisson.pmf(g, 2.0))
    assert h[1] == pytest.approx(np.tril(m, -1).sum() / m.sum()) and d[1] == pytest.approx(np.trace(m) / m.sum())
