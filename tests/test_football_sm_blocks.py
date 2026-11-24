"""The 24 Nov Sportmonks blocks: opponent-adjusted xG, shrunk finishing and luck, more statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football.blocks import sm_adj, sm_luck, sm_style2


def matches(rows):
    """rows: (date, home, away, npxg_h, npxg_a, goals_h, goals_a)."""
    out = []
    for i, (d, h, a, xh, xa, gh, ga) in enumerate(rows):
        out.append({"match_id": f"m{i}", "date": pd.Timestamp(d), "season": 2025, "competition_id": "ENG1",
                    "country": "ENG", "home_team_id": h, "away_team_id": a, "ft_home": gh, "ft_away": ga,
                    "smx_xg_npxg_h": xh, "smx_xg_npxg_a": xa, "smx_xg_xg_h": xh, "smx_xg_xg_a": xa,
                    "smx_xg_xgot_h": xh, "smx_xg_xgot_a": xa, "smx_xg_xpts_h": 1.5, "smx_xg_xpts_a": 1.0})
    return pd.DataFrame(out)


def test_xg_against_a_side_that_concedes_a_lot_counts_for_less():
    # Saturdays, a week apart. C concedes 2.0 to everyone; A and B each make 2.0 xG, but A against C, B against D
    # (who concedes 0.5). Adjusted, B's attack is the better one.
    rows = [("2025-08-02", "C", "X", 0.5, 2.0, 0, 1), ("2025-08-02", "D", "Y", 1.0, 0.5, 1, 0),
            ("2025-08-09", "A", "C", 2.0, 1.0, 2, 1), ("2025-08-09", "B", "D", 2.0, 1.0, 2, 1),
            ("2025-08-16", "A", "B", 1.0, 1.0, 1, 1)]
    got = sm_adj.build(matches(rows)).set_index("match_id")
    assert got.loc["m4", "sa_att_diff"] == pytest.approx((2.0 - 2.0) - (2.0 - 0.5))      # A's 0.0 less B's 1.5
    assert np.isnan(got.loc["m0", "sa_att_diff"])                                          # no history yet


def test_nothing_from_the_match_itself_or_its_collection_window():
    rows = [("2025-08-02", "C", "X", 0.5, 2.0, 0, 1), ("2025-08-09", "A", "C", 2.0, 1.0, 2, 1),
            ("2025-08-16", "A", "C", 1.0, 1.0, 1, 1)]
    a = sm_adj.build(matches(rows)).set_index("match_id").loc["m2"]
    changed = matches(rows)
    changed.loc[2, ["smx_xg_npxg_h", "smx_xg_npxg_a", "ft_home", "ft_away"]] = [9.0, 0.0, 9, 0]
    b = sm_adj.build(changed).set_index("match_id").loc["m2"]
    pd.testing.assert_series_equal(a, b)
    l1 = sm_luck.build(matches(rows)).set_index("match_id").loc["m2"]
    l2 = sm_luck.build(changed).set_index("match_id").loc["m2"]
    pd.testing.assert_series_equal(l1, l2)


def test_finishing_is_shrunk_by_its_evidence():
    # A scores 3 from 1.0 xG once (+2 finishing); Y scores 1 from 1.0 (0). One match of evidence each, so A's +2 is
    # shrunk by about 1 / (1 + K) (the week's decay takes a little off the evidence)
    rows = [("2025-08-02", "A", "X", 1.0, 1.0, 3, 1), ("2025-08-02", "Y", "Z", 1.0, 1.0, 1, 1),
            ("2025-08-09", "A", "Y", 1.0, 1.0, 1, 1)]
    got = sm_luck.build(matches(rows)).set_index("match_id").loc["m2"]
    assert 0.9 * 2.0 / (1 + sm_luck.K) < got["sl_fin_diff"] <= 2.0 / (1 + sm_luck.K)


def test_style_block_builds_blank_without_its_columns():
    got = sm_style2.build(matches([("2025-08-02", "A", "X", 1.0, 1.0, 1, 1)]))
    assert list(got.columns) == ["match_id"] + sm_style2.FEATURES and got[sm_style2.FEATURES].isna().all().all()
