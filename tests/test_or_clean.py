"""model/blocks/or_clean.py and at_weights.py (the owner's review of 20 Nov 2026: F2, the rating of 0 read as
unrated; the chase finding: the rating at the weights). Constructed frames test the arithmetic only."""
import numpy as np
import pandas as pd

from model.blocks import at_weights, or_clean


def _runs(rows):
    df = pd.DataFrame(rows, columns=["race_date", "track", "race_time", "horse_name", "official_rating",
                                     "placing_numerical", "pounds", "race_type", "race_name"])
    df["raceid"] = df["race_date"] + "_" + df["track"] + "_" + df["race_time"]
    return df


def test_a_rating_of_0_is_unrated_and_the_change_skips_unrated_runs():
    df = _runs([
        ("2025-01-01", "A", "1:00", "x", 0, 3, 150, "Novices Chase", "n"),     # unrated
        ("2025-01-01", "A", "1:00", "y", 120, 1, 160, "Novices Chase", "n"),
        ("2025-02-01", "A", "1:00", "x", 0, 2, 150, "Novices Chase", "n"),     # unrated again
        ("2025-02-01", "A", "1:00", "y", 125, 4, 160, "Novices Chase", "n"),
        ("2025-03-01", "A", "1:00", "x", 110, 1, 150, "Handicap Chase", "h"),  # first rating
        ("2025-03-01", "A", "1:00", "y", 124, 2, 160, "Handicap Chase", "h"),
        ("2025-04-01", "A", "1:00", "x", 116, 2, 150, "Handicap Chase", "h"),
        ("2025-04-01", "A", "1:00", "y", 0, 5, 160, "Handicap Chase", "h"),
    ])
    out = or_clean.build(df).set_index(["race_date", "horse_name"])
    assert np.isnan(out.loc[("2025-01-01", "x"), "orc_or"]) and out.loc[("2025-01-01", "x"), "orc_rated"] == 0
    # the first rated run has no earlier rating to change from (the old or_change read +110)
    assert np.isnan(out.loc[("2025-03-01", "x"), "orc_change"])
    assert out.loc[("2025-04-01", "x"), "orc_change"] == 6 and out.loc[("2025-04-01", "x"), "orc_best"] == 110
    assert out.loc[("2025-04-01", "x"), "orc_vs_last_win"] == 6          # it won off 110 on an earlier day
    # y's best before 1 Apr is 125; it is unrated on 1 Apr, so its readings against today are missing
    assert out.loc[("2025-04-01", "y"), "orc_best"] == 125 and np.isnan(out.loc[("2025-04-01", "y"), "orc_vs_best"])
    assert out.loc[("2025-03-01", "y"), "orc_change"] == -1 and out.loc[("2025-03-01", "y"), "orc_vs_last_win"] == 4
    # the race's mean ignores the unrated (the old Race_avgOR read (0 + 120) / 2 = 60)
    assert out.loc[("2025-01-01", "y"), "orc_race_mean"] == 120 and out.loc[("2025-01-01", "y"), "orc_rated_share"] == 0.5
    assert out.loc[("2025-04-01", "x"), "orc_runs_rated"] == 1 and out.loc[("2025-03-01", "x"), "orc_runs_rated"] == 0


def test_the_rating_at_the_weights_reorders_a_conditions_race():
    df = _runs([
        ("2025-01-01", "A", "2:00", "a", 150, 1, 168, "Chase", "Graded"),      # best mark, top weight
        ("2025-01-01", "A", "2:00", "b", 145, 2, 158, "Chase", "Graded"),      # 5 lb worse, 10 lb less
        ("2025-01-01", "A", "2:00", "c", 0, 3, 158, "Chase", "Graded"),        # unrated
    ])
    out = at_weights.build(df).set_index("horse_name")
    assert out.loc["a", "atw_or"] == 140 and out.loc["b", "atw_or"] == 145 and np.isnan(out.loc["c", "atw_or"])
    assert out.loc["b", "atw_rank"] == 1 and out.loc["b", "atw_vs_or"] == -1 and out.loc["a", "atw_vs_or"] == 1
    assert out.loc["b", "atw_gap"] == 5 and out.loc["a", "atw_gap"] == -5 and out.loc["a", "atw_hcap"] == 0
