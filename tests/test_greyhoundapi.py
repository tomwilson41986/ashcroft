"""sources/greyhoundapi.py with a fake API (no network, no key): the cursor paging, the months held are skipped and
the recent ones refetched, the runner rows. Planted data only."""
from datetime import date

import pytest

from sources import greyhoundapi as ga
from sources.common import Store


class FakeClient(ga.Client):
    def __init__(self, races_by_month):
        self.key, self.calls, self.by = "x", 0, races_by_month
        self.interval, self.next = 0.0, None

    def get(self, path, params=None, tries=6):
        self.calls += 1
        month = params["date_from"][:7]
        races = self.by.get((params["region"], month), [])
        start = int(params.get("cursor") or 0)
        page = races[start:start + 2]
        nxt = str(start + 2) if start + 2 < len(races) else None
        return {"meta": {"next_cursor": nxt, "total": len(races)}, "data": page}


def _race(i, region="GB"):
    return {"race_id": i, "meeting_id": 7, "region": region, "track": {"track_id": 1, "name": "Hove"},
            "scheduled_start": {"utc": "2026-10-01T18:00:00Z"}, "distance_m": 515, "grade": "A3",
            "result": {"result_status": "final", "weather": {"temp_c": 9},
                       "positions": [{"position": 1, "trap": 3, "dog_id": 11, "dog_name": "A", "sp": {"decimal": 3.0},
                                      "sectional_s": 4.9, "run_time_s": 29.5, "weight_kg": 31.2,
                                      "weight": {"status": "weighed", "change_kg": 0.2, "last_race": {"kg": 31.0, "days_before": 6}},
                                      "betfair": {"selection_id": 5, "back_price": 3.1, "lay_price": 3.2}},
                                     {"position": 2, "trap": 5, "dog_id": 12, "dog_name": "B", "sp": None}]}}


def test_paging_and_rows(tmp_path):
    c = FakeClient({("GB", "2026-10"): [_race(i) for i in range(5)]})
    races, man = ga.fetch_month(c, "GB", 2026, 10)
    assert len(races) == 5 and man["pages"] == 3 and man["api_total"] == 5
    rows = ga.rows(races)
    assert len(rows) == 10
    r = rows[0]
    assert r["track"] == "Hove" and r["sp_decimal"] == 3.0 and r["weight_change_kg"] == 0.2 and r["bf_back"] == 3.1
    assert rows[1]["sp_decimal"] is None


def test_months_held_are_skipped_recent_refetched(tmp_path, monkeypatch):
    monkeypatch.setitem(ga.FIRST, "GB", date(2026, 7, 1))
    store = Store(root=tmp_path)
    c = FakeClient({("GB", m): [_race(1)] for m in ("2026-07", "2026-08", "2026-09", "2026-10", "2026-11")})
    r1 = ga.fetch_results(store, c, ["GB"], date(2026, 11, 16))
    assert r1["months_fetched"] == 5
    r2 = ga.fetch_results(store, c, ["GB"], date(2026, 11, 16))
    assert r2["months_fetched"] == ga.REFRESH_MONTHS                  # only Oct and Nov again
    out = ga.build(store, ["GB"])
    assert out == {"GB_2026": 2}                                     # one race, two runners, deduplicated


def test_no_key_no_run(monkeypatch):
    monkeypatch.delenv("GREYHOUNDAPI_KEY", raising=False)
    with pytest.raises(SystemExit):
        ga.Client()
