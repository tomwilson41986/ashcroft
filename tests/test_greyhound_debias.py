"""core.debiased_probs (the feature review of 17 Nov 2026, B5): Shin and power chances sum to 1 a race, take more
from the outsiders than the proportional method does, and leave a two-runner book with no margin proportional."""

import numpy as np

from greyhound.pit.core import debiased_probs


def test_sums_and_favourite_longshot():
    odds = [1.5, 4, 8, 15, 30, 50, 2.2, 3, 5, 9, 12, 20]
    race = [1] * 6 + [2] * 6
    q = 1 / np.array(odds)
    for m in ("shin", "power"):
        p = debiased_probs(odds, race, m)
        for r in (1, 2):
            sel = np.array(race) == r
            assert abs(p[sel].sum() - 1) < 1e-9
            prop = q[sel] / q[sel].sum()
            assert p[sel][np.argmax(prop)] > prop.max()          # the favourite gains
            assert p[sel][np.argmin(prop)] < prop.min()          # the outsider loses


def test_missing_and_fair_book():
    p = debiased_probs([2.0, 2.0, np.nan], [1, 1, 1])
    assert np.isnan(p[2]) and np.allclose(p[:2], 0.5)
