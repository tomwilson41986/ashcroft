"""greyhound/pit/world.py on planted data: a GB runner's sire and dam joined by name to the API pedigree, the progeny's
Irish and Australian runs read only before the day; no API data gives empty columns, not an error."""
import pandas as pd

from greyhound.pit import world


def _gb():
    return pd.DataFrame({"race_id": [1, 1, 2], "dog_id": [11, 12, 11], "race_date": ["2026-03-01", "2026-03-01", "2026-03-20"],
                         "sire": ["Kinloch Brae", "Unknown Sire", "Kinloch Brae"], "dam": ["Mary's Pet", "X", "Mary's Pet"],
                         "is_card": False})


def _ped():
    return pd.DataFrame({"dog_id": ["a", "b", "c"], "dog_name": ["A", "B", "C"], "sire": ["KINLOCH BRAE"] * 3,
                         "dam": ["Marys Pet", "Other", "Marys Pet"], "sire_id": 1})


def _runs(region_days):
    rows = []
    for k, (day, dog, pos) in enumerate(region_days):
        rows.append({"race_id": k, "dog_id": dog, "track": "T", "distance_m": 480, "start_local": f"{day}T20:00:00+00:00",
                     "trap": 1, "position": pos, "run_time_s": 29.0, "adjusted_time_s": None, "sectional_s": 3.6,
                     "class_number": 3})
    return pd.DataFrame(rows)


def test_progeny_abroad_read_before_the_day():
    ie = _runs([("2026-02-01", "a", 1), ("2026-02-10", "b", 2), ("2026-03-05", "c", 1)])
    au = _runs([("2026-01-15", "a", 3)])
    tab, notes = world.build(_gb(), None, {"api_pedigree": _ped(), "api_ie_all": ie, "api_au_all": au})
    t = tab.set_index(["race_id", "dog_id"])
    assert t.loc[(1, 11), "wd_sire_ie_n"] == 2 and t.loc[(2, 11), "wd_sire_ie_n"] == 3   # the 5 Mar run read on 20 Mar
    assert t.loc[(1, 11), "wd_dam_ie_n"] == 1 and t.loc[(1, 11), "wd_sire_au_n"] == 1
    assert abs(t.loc[(1, 11), "wd_sire_au_share"] - 1 / 3) < 1e-6
    assert t.loc[(1, 12), "wd_sire_ie_n"] == 0 and pd.isna(t.loc[(1, 12), "wd_sire_ie_win"])


def test_no_api_data():
    tab, notes = world.build(_gb(), None, {})
    assert "no data found" in notes["note"] and (tab.wd_sire_ie_n == 0).all()
