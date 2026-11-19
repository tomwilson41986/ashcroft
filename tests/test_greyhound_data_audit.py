"""research/greyhound/data_audit.py on constructed GBGB rows (unit test only): each drop counted by its reason."""
import numpy as np
import pandas as pd

from research.greyhound.data_audit import gbgb_year


def test_drops_are_counted_by_reason():
    rows = []
    for race, pos in ((1, [1, 2, 3, 4, 5, 6]), (2, [0, 0, 0, 0, 0, 0]), (3, [1, 2, 3, 4, np.nan, 0])):
        for i, p in enumerate(pos):
            rows.append({"dog_id": race * 10 + i, "race_id": race, "race_date": "2026-11-01", "race_time": "19:00",
                         "position": p, "race_class": "A5", "track": "Romford", "trap": i + 1, "sp_decimal": 4.0})
    rows.append({**rows[0], "dog_id": np.nan, "race_id": 4})                 # no dog id
    rows.append({**rows[0], "race_time": "xx", "race_id": 5})                # no parseable off time
    out, c = gbgb_year(pd.DataFrame(rows))
    assert out["rows_in"] == 20 and out["drop_no_id_or_date"] == 1 and out["drop_no_position"] == 1
    assert out["drop_no_off_time"] == 1 and out["drop_void_races"] == 1 and out["drop_void_race_rows"] == 6
    assert out["rows_out"] == 11 and out["non_finisher_rows"] == 1 and out["field_size"] == {"5": 1, "6": 1}
