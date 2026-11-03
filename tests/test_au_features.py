"""The Australian model's Betfair-only features (au/features.py): the horse is its name, history is earlier days only.

The rows are format fixtures written for these tests; they are not data and no figure from them is used anywhere.
"""

import numpy as np
import pandas as pd
import pytest

from au.features import FORBIDDEN, build_features, leak_check


def _t():
    rows = []
    # one horse, four runs with a different saddlecloth (so a different selection id) each time, rivals around it
    days = ["2024-01-06", "2024-01-20", "2024-02-03", "2024-02-17"]
    for i, day in enumerate(days):
        m = 100 + i
        rows.append(dict(market_id=m, selection_id=10 + i, horse="Aardvark", meeting_date=day, off_uk=f"{day} 03:00",
                         track="Flemington", race_no=1, distance_m=1200, race_type="Hcap", race_family="handicap",
                         bsp=[4.0, 3.0, 6.0, 5.0][i], won=[0, 1, 0, 0][i]))
        for j, b in enumerate([2.5, 8.0, 20.0]):
            rows.append(dict(market_id=m, selection_id=900 + 10 * i + j, horse=f"Rival {i}{j}", meeting_date=day,
                             off_uk=f"{day} 03:00", track="Flemington", race_no=1, distance_m=1200, race_type="Hcap",
                             race_family="handicap", bsp=b, won=0))
    return pd.DataFrame(rows)


def test_the_horse_is_followed_by_name_across_selection_ids():
    d, feats = build_features(_t())
    a = d[d["horse"] == "Aardvark"].sort_values("meeting_date")
    assert list(a["h_n"]) == [0, 1, 2, 3]
    assert list(a["h_wins"]) == [0, 0, 1, 1]                      # the win on 20 Jan counts from the next run
    assert np.isnan(a["h_last_lp"].iloc[0]) and a["h_days"].iloc[1] == 14
    assert a["h_last_won"].iloc[2] == 1
    assert not (FORBIDDEN & set(feats))


def test_no_feature_reads_the_future():
    rep = leak_check(_t(), ["2024-01-20", "2024-02-17"])
    assert all(v["ok"] for v in rep.values()), rep


def test_a_forbidden_feature_is_refused(monkeypatch):
    import au.features as f
    monkeypatch.setattr(f, "FORBIDDEN", f.FORBIDDEN | {"field"})
    with pytest.raises(AssertionError):
        f.build_features(_t())
