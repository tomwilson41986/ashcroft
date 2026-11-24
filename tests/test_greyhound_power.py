"""research/greyhound/power.py: the t bar reproduces dsr_check, and the day-cluster interval."""

import numpy as np
import pytest

from research.greyhound import power
from research.overfit import dsr_check


@pytest.mark.parametrize("n_trials", [1, 436, 37_931])
def test_the_bar_reproduces_dsr_check_on_either_side(n_trials):
    """Normal returns scaled to sit just under and just over the bar: passes() and dsr_check agree, and the large-
    sample t bar (required_t) is within a few percent of the exact boundary."""
    rng = np.random.default_rng(7)
    n = 4000
    z = rng.standard_normal(n)
    z = (z - z.mean()) / z.std(ddof=1)                       # exact mean 0, sd 1: only the mean is moved below
    bar = power.required_t(n_trials)
    for mult, want in ((0.90, False), (1.12, True)):
        sr = mult * bar / np.sqrt(n)
        r = z + sr
        got = dsr_check(r, n_trials)["passed"]
        assert got is want and power.passes(sr, n, n_trials) is want


def test_fewer_than_thirty_periods_never_pass():
    assert power.passes(5.0, 29, 1) is False


def test_cluster_t_reads_days_that_move_together():
    """Races of a day sharing a shock: the clustered standard error is wider than the independent one (design effect
    above 1); independent races give a design effect near 1."""
    rng = np.random.default_rng(3)
    days = np.repeat(np.arange(60), 40)
    shared = cluster = power.cluster_t(rng.standard_normal(60)[days] + rng.standard_normal(len(days)), days)
    alone = power.cluster_t(rng.standard_normal(len(days)), days)
    assert shared["design_effect"] > 5 and 0.5 < alone["design_effect"] < 2.0
    assert cluster["days"] == 60 and power.cluster_t([1.0], [1])["t"] is None


def test_days_needed_and_the_chance_at_a_read():
    """t grows with the square root of the days: a rule at half the bar after 10 days needs more than 40; a read
    before 30 daily periods is under-powered whatever its t."""
    bar = power.required_t(37_931)
    assert 5.5 < bar < 6.1 and abs(power.required_t(1) - 1.645) < 1e-3
    need = power.days_needed(bar / 2, 10, 37_931)
    assert need > 40 and power.days_needed(-1.0, 10, 37_931) is None
    res = power.table([{"name": "x", "mean": 1.26, "lo": 0.59, "hi": 1.77, "days": 8, "read_days": 20}], 37_931)
    row = res["rows"][0]
    assert row["chance_at_read"] == 0.0 and row["under_powered"] is True and 3.0 < row["t"] < 4.0
    assert power.t_from_interval(1.0, 0.0, 2.0) == pytest.approx(1.6449, abs=1e-3)
