"""greyhound/pit/irish.py on planted data: the API's GB rows link its dog ids to GBGB's, the Irish runs read only
before the day, the import flag; no API data gives NaN columns, not an error."""
import numpy as np
import pandas as pd

from greyhound.pit import irish


def _gb_runs():
    rows = []
    for k, day in enumerate(["2026-03-01", "2026-03-08", "2026-03-15"]):
        for trap, dog in ((1, 101), (2, 102)):
            rows.append({"race_id": 10 + k, "dog_id": dog, "race_date": day, "race_time": "19:00", "track": "Romford",
                         "trap": trap, "position": trap, "is_card": False})
    return pd.DataFrame(rows)


def _api_gb():
    return pd.DataFrame([{"race_id": 900 + k, "dog_id": api, "track": "Romford", "start_local": f"{day}T19:00:00+00:00",
                          "trap": trap, "position": trap}
                         for k, day in enumerate(["2026-03-01", "2026-03-08"]) for trap, api in ((1, "A"), (2, "B"))])


def _api_ie():
    rows = []
    for k, day in enumerate(["2025-12-01", "2026-01-05", "2026-02-02", "2026-03-08"]):
        for trap, api in ((1, "A"), (2, "B"), (3, "C")):
            rows.append({"race_id": 500 + k, "dog_id": api, "track": "Shelbourne", "distance_m": 525,
                         "start_local": f"{day}T20:00:00+00:00", "trap": trap, "position": trap,
                         "run_time_s": 28.5 + 0.1 * trap, "adjusted_time_s": None, "sectional_s": 3.5 + 0.05 * trap,
                         "class_number": 4})
    return pd.DataFrame(rows)


def test_dog_map_links_by_day_track_time_trap():
    mp, notes = irish.dog_map(_api_gb(), _gb_runs())
    assert mp == {"A": 101, "B": 102} and notes["dogs_mapped"] == 2


def test_irish_form_reads_only_earlier_days():
    tab, notes = irish.build(_gb_runs(), None, {"api_ie": _api_ie(), "api_gb": _api_gb()})
    t = tab.set_index(["race_id", "dog_id"])
    assert t.loc[(10, 101), "ie_n"] == 3                         # 1 Mar: the Dec, Jan and Feb runs
    assert t.loc[(11, 101), "ie_n"] == 3                         # 8 Mar: the 8 Mar Irish run is the same day, not read
    assert t.loc[(12, 101), "ie_n"] == 4                         # 15 Mar: it is read now
    assert t.loc[(10, 101), "ie_win_rate"] > t.loc[(10, 102), "ie_win_rate"]
    assert t.loc[(10, 101), "ie_import"] == 1 and t.loc[(10, 101), "ie_days_since"] == 27
    assert notes["dogs_mapped"] == 2 and notes["gb_rows_with_irish_form"] == 6


def test_no_api_data_is_not_an_error():
    tab, notes = irish.build(_gb_runs(), None, {})
    assert "no data found" in notes["note"] and tab.ie_fig_mean3.isna().all() and (tab.ie_n == 0).all()


def test_name_link_when_the_ids_differ():
    gb = _gb_runs()
    gb["dog_name"] = gb.dog_id.map({101: "Swift Blade", 102: "Other Dog"})
    ie = _api_ie().assign(dog_id=lambda d: d.dog_id.map({"A": "ie-A", "B": "ie-B", "C": "ie-C"}),
                          dog_name=lambda d: d.dog_id.map({"ie-A": "SWIFT BLADE", "ie-B": "Somebody", "ie-C": "Nobody"}))
    tab, notes = irish.build(gb, None, {"api_ie": ie, "api_gb": _api_gb()})
    t = tab.set_index(["race_id", "dog_id"])
    assert notes["name_links"] == 1 and t.loc[(10, 101), "ie_n"] == 3 and t.loc[(10, 102), "ie_n"] == 0


def test_a_name_reused_long_before_is_not_read():
    gb = _gb_runs()
    gb["dog_name"] = gb.dog_id.map({101: "Swift Blade", 102: "Other Dog"})
    ie = _api_ie().assign(dog_id="ie-A", dog_name="Swift Blade", start_local="2010-01-01T20:00:00+00:00")
    ie["race_id"] = range(len(ie))
    tab, _ = irish.build(gb, None, {"api_ie": ie, "api_gb": _api_gb()})
    assert (tab.ie_n == 0).all()
