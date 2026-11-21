"""research/greyhound/cut_check.py: a feature that reads a later day is caught, runner by runner."""
import numpy as np
import pandas as pd

from research.greyhound.cut_check import compare


def test_a_feature_that_differs_between_the_cache_and_the_cut_build_is_named():
    k = pd.DataFrame({"race_id": [1, 1, 2, 2], "dog_id": [10, 11, 12, 13]})
    cache = k.assign(clean=[1.0, np.nan, 3.0, 4.0], leaky=[0.5, 0.6, 0.7, 0.8], big=[1e6, 2e6, 3e6, 4e6])
    cut = k.assign(clean=[1.0, np.nan, 3.0, 4.0], leaky=[0.5, 0.6, 0.9, 0.8], big=[1e6 + 0.5, 2e6, 3e6, 4e6])
    r = compare(cache, cut.iloc[::-1], ["clean", "leaky", "big"])
    assert r["runners_compared"] == 4 and r["agree_all"] == 2 and list(r["differ"]) == ["leaky"]
    assert r["differ"]["leaky"] == {"agree": 0.75, "differ": 1, "median_abs_gap": 0.2}
