"""Comment features: the classes read what they should, and only earlier runs."""

import numpy as np
import pandas as pd
import pytest

from model.comment_features import COMMENT_POST_RACE, add_comment_features, comment_flags


@pytest.mark.parametrize("comment, on, off", [
    ("held up, hampered 2f out, ran on", {"trouble", "finished_well", "excuse"}, {"weakened"}),
    ("slowly away, keen, raced wide, weakened", {"slow_start", "keen", "wide", "weakened", "excuse"}, {"trouble"}),
    ("tracked leaders, not clear run over 1f out, kept on", {"trouble", "finished_well"}, {"slow_start"}),
    ("led, drew clear 2f out, won easily", {"easy_win"}, {"trouble", "weakened"}),
    ("in touch, ran green, not knocked about", {"tender"}, {"trouble"}),
    ("chased leaders, lost action, pulled up", {"problem", "no_finish", "excuse"}, set()),
    ("prominent, switched left 2f out, one pace", {"switched", "weakened"}, {"trouble"}),
])
def test_classes(comment, on, off):
    f = comment_flags(pd.Series([comment])).iloc[0]
    for k in on:
        assert f[f"_c_{k}"] == 1.0, k
    for k in off:
        assert f[f"_c_{k}"] == 0.0, k


def test_no_comment_is_unknown_not_clean():
    f = comment_flags(pd.Series([None, ""]))
    assert f.isna().all().all()


def _history():
    rows = [
        ("A", "2025-01-01", "held up, badly hampered 2f out, ran on", 3, 10, "1 1/2"),
        ("A", "2025-02-01", "made all, won easily", 1, 8, ""),
        ("A", "2025-03-01", "prominent, weakened", 7, 9, "12"),
        ("B", "2025-01-01", "slowly away, never nearer", 2, 10, "nk"),
        ("A", "2025-04-01", "tracked leaders, kept on", 2, 10, "hd"),
    ]
    return pd.DataFrame([{"horse_name": h, "race_date": pd.Timestamp(dt), "race_time": "2.00", "comment": c,
                          "placing_numerical": p, "number_of_runners": n, "total_dst_bt": lb}
                         for h, dt, c, p, n, lb in rows])


def test_features_read_only_earlier_runs():
    d = _history()
    out, names = add_comment_features(d)
    a = out[out["horse_name"] == "A"].sort_values("race_date").reset_index(drop=True)
    assert np.isnan(a.loc[0, "LR_c_trouble"])                 # a debut has no last run
    assert a.loc[1, "LR_c_trouble"] == 1.0                    # the January run
    assert a.loc[1, "LR_excuse_close"] == 1.0                 # hampered, beaten 1.5 lengths
    assert a.loc[2, "LR_easy_win"] == 1.0 and a.loc[2, "LR_c_trouble"] == 0.0
    assert a.loc[3, "LR_noexcuse_poor"] == 1.0                # weakened into 7th of 9, no excuse
    assert a.loc[3, "L3_c_trouble"] == pytest.approx(1 / 3)
    assert a.loc[1, "runs_since_trouble"] == 1 and a.loc[3, "runs_since_trouble"] == 3
    # changing a run's own comment never changes its own features
    d2 = d.copy()
    d2.loc[4, "comment"] = "hampered, slowly away, keen, wide"
    out2, _ = add_comment_features(d2)
    last = (out2["horse_name"] == "A") & (out2["race_date"] == pd.Timestamp("2025-04-01"))
    for c in names:
        assert np.array_equal(out.loc[last, c].to_numpy(), out2.loc[last, c].to_numpy(), equal_nan=True), c
    assert not set(names) & set(COMMENT_POST_RACE)
