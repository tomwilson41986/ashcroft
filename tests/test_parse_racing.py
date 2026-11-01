"""sources/parse_racing.py: the JRA and HKJC backfills skip what is held, find the meetings and build the tables.

Hand-made payloads in the shapes race.netkeiba.com and racing.hkjc.com returned through Parse on 1 Nov 2026."""
from datetime import date

import pandas as pd

import sources.parse_odds as po
import sources.parse_racing as pr
from sources.common import Store
from tests.test_parse_odds import FakeHTTP, ok


def test_jra_lists_then_fetches_results_and_skips_held(tmp_path, monkeypatch):
    monkeypatch.setattr(po.time, "sleep", lambda s: None)
    store = Store(root=tmp_path)

    def list_races(params):
        if params["date"] != "2025-11-01":
            return ok({"date": params["date"], "meetings": []})
        return ok({"date": "2025-11-01", "meetings": [{"kaisai_id": "k", "meeting": "4th Tokyo", "venue": "Tokyo",
                                                      "turf_condition": "good", "dirt_condition": "heavy",
                                                      "races": [{"race_id": "R1", "race_number": 1, "surface": "dirt",
                                                                 "distance_m": 1600, "has_result": True},
                                                                {"race_id": "R2", "race_number": 2, "surface": "turf",
                                                                 "distance_m": 1800, "has_result": False}]}]})

    def result(params):
        return ok({"race_id": params["race_id"], "race_number": 1, "venue": "Tokyo", "surface": "dirt",
                   "distance_m": 1600, "race_class": ["2yo", "maiden"], "prize_money_man_yen": [560, 220],
                   "results": [{"finish_position": 1, "horse_id": "h1", "frame_number": 7, "win_odds": 2.7,
                                "corner_positions": [1, 1]},
                               {"finish_position": 2, "horse_id": "h2", "frame_number": 6, "win_odds": 4.3,
                                "corner_positions": [8, 8]}]})

    http = FakeHTTP({"list_races": list_races, "get_race_result": result})
    p = po.Parse(key="k", floor=0, http=http)
    got = pr.jra(store, p, date(2025, 11, 1), date(2025, 11, 2), workers=1)   # Saturday and Sunday
    assert got == {"days_listed": 2, "races_with_result": 1, "results_fetched": 1}
    n = len(http.calls)
    assert pr.jra(store, p, date(2025, 11, 1), date(2025, 11, 2), workers=1)["results_fetched"] == 0
    assert len(http.calls) == n                                                # nothing fetched twice
    t = pr.table_jra_results(store)
    assert len(t) == 2 and set(t["going"]) == {"heavy"} and t["race_class"].iloc[0] == "2yo|maiden"
    assert t["race_date"].iloc[0] == "2025-11-01" and t["prize_1st_man_yen"].iloc[0] == 560


def test_hk_finds_meetings_stops_at_the_last_race_and_takes_sectionals(tmp_path, monkeypatch):
    monkeypatch.setattr(po.time, "sleep", lambda s: None)
    store = Store(root=tmp_path)

    def results(params):
        if params["date"] == "20251102" and params["race_no"] <= 2:
            return ok({"date": "20251102", "race_no": params["race_no"], "venue": "", "finishers": [
                {"placing": "1", "horse_id": f"H{params['race_no']}a", "draw": "10", "win_odds": "7.4"},
                {"placing": "2", "horse_id": f"H{params['race_no']}b", "draw": "7", "win_odds": "3"}]})
        return ok({"date": params["date"], "race_no": params["race_no"], "venue": "", "finishers": []})

    def sectionals(params):
        return ok({"date": params["date"], "race_no": params["race_no"], "venue": "Happy Valley", "class": "Class 5",
                   "distance": "1200", "going": "Good To Firm", "race_times": ["23.75", "1:09.60"],
                   "runners": [{"horse_id": "x", "finishing_order": "1",
                                "sections": [{"section": 1, "position": "8", "margin": "2-1/2", "time": "24.15"},
                                             {"section": 2, "position": "1", "margin": "N", "time": "22.98"}]}]})

    http = FakeHTTP({"get_race_results": results, "get_sectional_times": sectionals})
    p = po.Parse(key="k", floor=0, http=http)
    got = pr.hk(store, p, date(2025, 11, 1), date(2025, 11, 2), workers=1)    # Saturday (no meeting), Sunday
    assert got["meetings"] == 1 and got["races"] == 2 and got["sectionals_fetched"] == 2
    calls = [c[1] for c in http.calls if c[0].endswith("get_race_results")]
    # race 1 of both days, then races 2 and 3 of the Sunday (3 comes back empty: the meeting's end)
    assert sorted((c["date"], c["race_no"]) for c in calls) == [("20251101", 1), ("20251102", 1),
                                                               ("20251102", 2), ("20251102", 3)]
    t = pr.table_hk_results(store)
    assert len(t) == 4 and set(t["venue"]) == {"Happy Valley"} and t["race_date"].iloc[0] == "2025-11-02"
    s = pr.table_hk_sectionals(store)
    assert len(s) == 4 and set(s["section"]) == {1, 2}
    counts = pr.build(store)
    assert counts["hk_results"] == 4 and isinstance(store.get_parquet("parse/hk_results.parquet"), pd.DataFrame)
