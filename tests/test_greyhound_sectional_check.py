"""research/greyhound/sectional_check.py: the GB runner match and the sectional tallies (planted rows only)."""
import pandas as pd

from research.greyhound import sectional_check as sc


def test_tallies_both_one_side_and_agreement():
    g = pd.DataFrame({"race_date": ["2026-01-02"] * 3, "race_time": ["19:01", "19:01", "19:01"], "track": ["Romford"] * 3,
                      "trap": [1, 2, 3], "sectional": [4.10, None, 4.30], "run_time": [24.0, 24.2, 24.5]})
    a = pd.DataFrame({"start_local": ["2026-01-02T19:01:00"] * 3, "track": ["Romford"] * 3, "trap": [1, 2, 3],
                      "sectional_s": [4.10, 4.20, None], "run_time_s": [24.0, 24.2, 24.5]})
    r = sc.compare(sc.keyed_api(a), sc.keyed_gbgb(g))[0]
    assert r["matched"] == 3 and r["sec_both"] == 1 and r["sec_api_only"] == 1 and r["sec_gbgb_only"] == 1
    assert r["sec_equal_share"] == 1.0 and r["time_equal_share"] == 1.0
    assert "API only" in sc.report_md([r])
