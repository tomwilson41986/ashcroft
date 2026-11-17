"""greyhound/pit/irish_style.py: the Irish run style from the in-running comments, read from earlier days only
(planted rows only)."""
import numpy as np
import pandas as pd

from greyhound.pit import irish_style as st


def _comments():
    return pd.DataFrame({
        "race_id": [1, 1, 2, 2], "dog_id": [100, 101, 100, 101],
        "dog_name": ["Fast Dog", "Other One", "Fast Dog", "Other One"],
        "start_utc": ["2026-01-05T20:00:00Z", "2026-01-05T20:00:00Z", "2026-02-05T20:00:00Z", "2026-02-05T20:00:00Z"],
        "events": ["fast_start|led|line_rails", "slow_away|baulked", "fast_start|led", "ran_on"],
        "places": ["|bend_1|", "|bend_1", "|run_in", ""]})


def _runs():
    return pd.DataFrame({"race_id": [9, 9, 10], "dog_id": [7, 8, 7], "dog_name": ["Fast Dog", "Gb Only", "Fast Dog"],
                         "race_date": ["2026-02-05", "2026-02-05", "2026-03-01"], "race_time": ["19:00"] * 3,
                         "track": ["Romford"] * 3, "trap": [1, 2, 1]})


def test_groups_read_the_codes():
    f = st.run_flags(_comments()).reset_index(drop=True)
    assert list(f.early) == [1, 0, 1, 0] and list(f.slow) == [0, 1, 0, 0] and list(f.trouble) == [0, 1, 0, 0]
    assert list(f.led_early) == [1, 0, 0, 0] and list(f.rails) == [1, 0, 0, 0] and list(f.ranon) == [0, 0, 0, 1]
    assert st.coverage(_comments()) == 1.0


def test_reads_earlier_irish_days_only():
    tab, notes = st.build(_runs(), ctx={"api_ie_comments": _comments(), "api_ie": None, "api_gb": None})
    a, b, c = tab.iloc[0], tab.iloc[1], tab.iloc[2]
    assert a.is_n == 1 and a.is_last_early == 1 and a.is_last_led == 1          # 5 Feb sees 5 Jan only
    assert b.is_n == 0 and np.isnan(b.is_early_rate)
    assert c.is_n == 2 and c.is_last_early == 1 and c.is_last_trouble == 0
    p0 = notes["prior_shares"]["early"]
    assert np.isclose(c.is_early_rate, (2 + st.K * p0) / (2 + st.K))


def test_no_comments_is_said():
    tab, notes = st.build(_runs(), ctx={})
    assert "no data found" in notes["note"] and (tab.is_n == 0).all()
