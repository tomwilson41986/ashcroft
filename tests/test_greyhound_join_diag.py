"""research/greyhound/join_diag.py: the track-alias learner on constructed tables shaped as the stored ones (a price
table with its own country column, GBGB runs) - unit test only."""
import io

import pandas as pd

from research.greyhound import join_diag as jd


class _Store:
    def __init__(self, d):
        self.d = d

    def get(self, k):
        return self.d.get(k)

    def get_parquet(self, k):
        return self.d.get("pq:" + k)


def test_the_alias_is_learnt_by_date_and_off_time():
    rows = []
    for day in range(25):
        for i, t in enumerate(["19:00", "19:20"]):
            rid = day * 10 + i
            for trap in range(1, 7):
                rows.append({"race_id": rid, "dog_id": rid * 10 + trap, "race_date": f"2019-01-{day + 1:02d}",
                             "race_time": t + ":00", "position": trap, "race_class": "A5",
                             "track": "Harlow" if i == 0 else "Romford", "trap": trap, "sp_decimal": 3.0})
    buf = io.BytesIO()
    pd.DataFrame(rows).to_parquet(buf)
    prices = pd.DataFrame([{"market": "win", "country": "GB", "menu_hint": f"SIS / Harl {d + 1}st Jan",
                            "race_date": f"2019-01-{d + 1:02d}", "race_time": "19:00", "trap": 1, "track": "Harl"}
                           for d in range(25)])
    s = _Store({"gbgb/runs_2019.parquet": buf.getvalue(), "pq:greyhound_prices/prices_2019.parquet": prices})
    out = jd.aliases(s, [2019])
    assert out["kept"] == {"Harl": "Harlow"} and out["labels"]["Harl"]["markets"] == 25
