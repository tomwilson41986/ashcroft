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


class RunnersClient(FakeClient):
    def __init__(self, used=0):
        super().__init__({})
        self.used = used

    def get(self, path, params=None, tries=6):
        self.calls += 1
        if path == "/usage":
            return {"data": {"month": {"used": self.used}}}
        if path.endswith("/runners"):
            return {"data": [{"trap": 1, "runner_status": "scratched"}]}
        if path in ("/racecards/today", "/racecards/upcoming"):
            return {"meta": {}, "data": [{"race_id": 1}]}
        raise AssertionError(path)


def _held_month(store, region="GB", month="2026-10"):
    store.put(ga.raw_key(region, int(month[:4]), int(month[5:])), ga.gz(b'[{"race_id": 7}, {"race_id": 8}]'))


def test_runners_backfill_and_quota_guard(tmp_path):
    import gzip
    import json
    store = Store(root=tmp_path)
    _held_month(store)
    r = ga.fetch_runners(store, RunnersClient(), ["GB"])
    assert r["months_fetched"] == 1 and r["races"] == 2
    got = json.loads(gzip.decompress(store.get(ga.runners_key("GB", "2026-10"))))
    assert got["7"][0]["runner_status"] == "scratched"
    assert ga.fetch_runners(store, RunnersClient(), ["GB"])["months_to_fetch"] == 0     # held: skipped
    _held_month(store, month="2026-09")
    stopped = ga.fetch_runners(store, RunnersClient(used=ga.QUOTA_STOP), ["GB"])
    assert stopped["months_fetched"] == 0 and stopped["stopped"].startswith("quota")


def test_cards_snapshot(tmp_path):
    store = Store(root=tmp_path)
    assert ga.snapshot_cards(store, RunnersClient()) == {"today": 1, "upcoming": 1}
    assert any(k.endswith(".json.gz") for k in store.listing("greyhoundapi/raw/cards/"))


def test_comments_are_fetched_a_month_a_file_and_flattened(tmp_path):
    """The owner, 17 Nov 2026: "fetch the in-running comments for Irish runs"."""
    import gzip, json as _json
    from sources import greyhoundapi as g
    from sources.common import Store
    st = Store(root=tmp_path)
    st.put("greyhoundapi/raw/results/IE/2026-10.json.gz", gzip.compress(_json.dumps([{"race_id": 7}]).encode()))
    st.put("greyhoundapi/raw/results/IE/2010-01.json.gz", gzip.compress(_json.dumps([{"race_id": 8}]).encode()))

    class C:
        calls = 0

        def get(self, path, params=None, tries=6):
            self.calls += 1
            if path == "/usage":
                return {"data": {"month": {"used": 10}}}
            if path == "/comments/vocabulary":
                return {"data": {"led": "Led"}}
            assert path == "/races/7/comments", path            # 2010 is before the first month
            return {"data": {"scheduled_start": "2026-10-01T19:00:00Z", "track": {"name": "Shelbourne Park"},
                             "runners": [{"dog_id": 5, "dog_name": "A Dog", "trap": 1, "position": 1,
                                          "comments": {"in_running": {"summary": "Led.", "events": [
                                              {"event": "fast_start", "place": "", "text": "Quick away"},
                                              {"event": "led", "place": "bend_1", "text": "Led at the first bend"}]}}}]}}

    rep = g.fetch_comments(st, C(), ["IE"], first_month="2011-01")
    assert rep["months_fetched"] == 1 and rep["races"] == 1 and rep["races_without_comments"] == 0
    rows = g.comment_rows(st, "IE")
    assert list(rows.events) == ["fast_start|led"] and list(rows.places) == ["|bend_1"] and rows.n_events.iloc[0] == 2
    assert g.build_comments(st, ["IE"]) == {"IE": 1}
