"""model/blocks/rank_fix.py by hand: a runner with no figure ranks behind the worst, not beside the best."""
import numpy as np
import pandas as pd

from model.blocks import rank_fix


def _race(values, rid="2025-03-01|York|2.30"):
    n = len(values)
    return pd.DataFrame({
        "raceid": [rid] * n, "race_date": [pd.Timestamp("2025-03-01")] * n, "race_time": ["2.30"] * n,
        "track": ["York"] * n, "horse_name": [f"h{i}" for i in range(n)],
        "preracehorsecareerLB": values, "career_nfp_std": values,
        "dist_from_preferred": values, "going_from_preferred": values,
    })


def test_lower_is_better_and_missing_goes_last():
    # beaten 2 lengths on average, no runs yet, beaten 10, beaten 2 (a tie)
    out = rank_fix.build(_race([2.0, np.nan, 10.0, 2.0]))
    assert out["rfix_LB"].tolist() == [1.0, 4.0, 3.0, 1.0]
    # the engine's way (highest first, missing last) put the debutant beside the best
    engine = _race([2.0, np.nan, 10.0, 2.0]).groupby("raceid")["preracehorsecareerLB"].rank(
        ascending=False, method="min", na_option="bottom")
    assert engine.tolist() == [2.0, 4.0, 1.0, 2.0]
    for c in rank_fix.FEATURES:
        assert out[c].tolist() == [1.0, 4.0, 3.0, 1.0]


def test_each_race_is_ranked_on_its_own():
    df = pd.concat([_race([5.0, 1.0]), _race([np.nan, 3.0, 0.5], rid="2025-03-01|York|3.00")], ignore_index=True)
    df.loc[2:, "race_time"] = "3.00"
    out = rank_fix.build(df)
    assert out["rfix_DistApt"].tolist() == [2.0, 1.0, 3.0, 2.0, 1.0]
