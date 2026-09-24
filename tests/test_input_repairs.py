"""Regression tests for the inputs the feature review found broken."""

import numpy as np
import pandas as pd
import pytest

from model.custom_metrics import CustomMetricsEngine, _calculate_epf


@pytest.mark.parametrize("comment, figure", [
    # the spec's unanchored patterns scored all of these as front-runners (6.0)
    ("held up in rear, failed to pick up", 1.0),
    ("struggled to go pace, always rear", 1.0),
    ("pulled, held up", 1.0),
    ("travelled, keen, held up in touch", 3.0),
    # the first positional phrase is the early position
    ("tracked leaders, led over 1f out, won", 4.0),
    ("prominent, led 3f out", 4.0),
    ("held up in rear, led final 100yds", 1.0),
    ("led, headed 2f out, weakened", 6.0),
    ("made all, ran on", 6.0),
    ("disputed lead until 3f out", 5.5),
    ("chased leader, ridden 2f out", 5.0),
    ("midfield, no extra", 3.0),
    ("slowly away, soon behind", 1.0),
    # nothing positional, or nothing at all
    ("ran on final furlong", 3.0), ("", 3.0), (None, 3.0),
])
def test_epf_reads_the_early_position(comment, figure):
    assert _calculate_epf(comment) == figure


def test_beaten_lengths_parse_fractions_and_keep_blanks_unknown():
    d = pd.DataFrame({
        "raceid": ["r"] * 7, "race_date": pd.to_datetime(["2025-01-01"] * 7), "race_time": ["2.00"] * 7,
        "horse_name": list("abcdefg"), "number_of_runners": 7,
        "placing_numerical": [1, 2, 3, 4, 5, 6, 7],
        "total_dst_bt": ["", "3/4", "1½", "1 1/2", "2hd", None, "dist"],
    })
    out = CustomMetricsEngine()._calc_actual_lengths_beaten(d)
    got = dict(zip(out["horse_name"], out["LB"]))
    assert got["a"] == 0.0                     # the winner, whatever the field says
    assert got["b"] == 0.75                    # was 3.0
    assert got["c"] == 1.5 and got["d"] == 1.5 # were 1.0
    assert got["e"] == pytest.approx(2.2)
    assert np.isnan(got["f"])                  # was 0.0: a loser read as a winner
    assert got["g"] == 30.0


def test_debut_flag_is_on_the_horses_first_run():
    """The pedigree block re-sorts the frame by damsire and distance band before
    the debut flag is taken; a row count over that order flagged the first run
    at the shortest trip, not the debut."""
    rows = []
    for k, (date, dist) in enumerate([("2025-01-01", 8.0), ("2025-02-01", 6.0), ("2025-03-01", 7.0)]):
        rows.append({"raceid": f"r{k}", "race_date": pd.Timestamp(date), "race_time": "2.00",
                     "horse_name": "H", "stallion": "S", "dam_stallion": "DS", "dam": "D",
                     "dist_furlongs": dist, "going_description": "Good", "number_of_runners": 8,
                     "placing_numerical": 3, "trainer": "T", "jockey_name": "J"})
    d = pd.DataFrame(rows)
    eng = CustomMetricsEngine()
    d = eng._calc_nfp(d.assign(won=0.0, placed=1.0))
    d = eng._calc_xwinrand_wiv(d)
    out = eng._calc_pedigree(d)
    flagged = out.loc[out["debut_x_sire_nfp"].notna() & (out["debut_x_sire_nfp"] != 0), "race_date"]
    first_rows = out.sort_values("race_date")
    # debut_x_* is non-zero only on the debut row, whatever the sire value
    nonzero_dates = set(first_rows.loc[first_rows["debut_x_sire_wiv"].fillna(0) != 0, "race_date"])
    assert nonzero_dates <= {pd.Timestamp("2025-01-01")}
    assert set(flagged) <= {pd.Timestamp("2025-01-01")}
