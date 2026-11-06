"""Cup form check: form is point in time (a match never sees its own statistics)."""

from __future__ import annotations

import pandas as pd

from research.open_close import football_cup_form_check as cf


def test_team_form_uses_only_earlier_matches():
    rows, fx = [], []
    for i in range(6):
        fid = 100 + i
        fx.append({"apif_fixture_id": fid, "kickoff_utc": f"2026-01-{i + 1:02d}T15:00:00Z"})
        rows.append({"apif_fixture_id": fid, "apif_team_id": 1, "shots_on": i, "shots": 10, "xg": 1.0})
        rows.append({"apif_fixture_id": fid, "apif_team_id": 2, "shots_on": 0, "shots": 10, "xg": 1.0})
    f = cf.team_form(pd.DataFrame(rows), pd.DataFrame(fx))
    t1 = f[f["apif_team_id"] == 1].sort_values("apif_fixture_id")
    assert t1["sotd"].isna().tolist()[:3] == [True, True, True]          # fewer than 3 earlier matches
    assert t1["sotd"].tolist()[3] == (0 + 1 + 2) / 3                      # matches 0-2 only, not match 3 itself
    assert t1["n_prior"].tolist() == list(range(6))
