"""greyhound/pit/irish_market.py: the Irish market's view of a dog, read from earlier days only."""

import numpy as np
import pandas as pd

from greyhound.pit import irish_market as im


def _ie():
    rows = []
    # two Irish races on two days; "Fast Dog" is the 2.0 favourite and wins the first, loses the second at 4.0
    for rid, day, sps, pos in ((1, "2026-01-05", [2.0, 4.0, 4.0], [1, 2, 3]), (2, "2026-02-05", [4.0, 2.0, 4.0], [3, 1, 2])):
        for k, (sp, p) in enumerate(zip(sps, pos)):
            rows.append({"race_id": rid, "dog_id": 100 + k, "dog_name": ["Fast Dog", "Other One", "Third Dog"][k],
                         "track": "Shelbourne", "start_local": f"{day}T20:00:00", "trap": k + 1, "position": p,
                         "run_time_s": 29.5, "adjusted_time_s": None, "sectional_s": None, "distance_m": 525,
                         "class_number": 3, "sp_decimal": sp, "weight_kg": 30.0 + k})
    return pd.DataFrame(rows)


def _runs():
    return pd.DataFrame({"race_id": [9, 9, 10], "dog_id": [7, 8, 7], "dog_name": ["Fast Dog", "Gb Only", "Fast Dog"],
                         "race_date": ["2026-02-05", "2026-02-05", "2026-03-01"], "race_time": ["19:00"] * 3,
                         "track": ["Romford"] * 3, "trap": [1, 2, 1]})


def test_reads_earlier_irish_days_only():
    tab, notes = im.build(_runs(), ctx={"api_ie": _ie(), "api_gb": None})
    # the GB race on 5 Feb sees only the 5 Jan Irish run (the same day's Irish run is not read)
    a = tab.iloc[0]
    assert a.im_n == 1 and np.isclose(a.im_ae, (1 - 0.5) / (1 + im.K)) and np.isclose(a.im_lsp_last, np.log(2.0))
    assert a.im_wt_last == 30.0
    b = tab.iloc[2]                       # 1 Mar: both Irish runs
    assert b.im_n == 2 and np.isclose(b.im_ae, ((1 - 0.5) + (0 - 0.25)) / (2 + im.K))
    assert np.isclose(b.im_beat_mean3, ((1 - 1) + (2 - 3)) / 2)
    assert tab.iloc[1].im_n == 0 and np.isnan(tab.iloc[1].im_ae)          # no Irish runs
    assert notes["irish_runs_with_market"] == 2


def test_no_irish_results_is_said():
    tab, notes = im.build(_runs(), ctx={})
    assert "no data found" in notes["note"] and (tab.im_n == 0).all()
