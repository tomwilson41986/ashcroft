"""The warehouse's daily update from Betfair's Historic Data API: what is fetched, where it goes, a rerun is a no-op."""

from __future__ import annotations

import bz2
from datetime import date

from football.warehouse import historic_api as ha
from sources.common import Store


class FakeAPI:
    """The Historic Data API: lists the files of a range of days; serves each one."""

    def __init__(self, files: dict[str, bytes]):
        self.files_ = files
        self.bodies, self.downloads = [], []

    def files(self, body):
        self.bodies.append(body)
        lo = date(body["fromYear"], body["fromMonth"], body["fromDay"])
        hi = date(body["toYear"], body["toMonth"], body["toDay"])
        out = []
        for p in self.files_:
            y, m, d = p.split("/BASIC/")[1].split("/")[:3]
            day = date(int(y), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov",
                                "Dec"].index(m) + 1, int(d))
            if lo <= day <= hi:
                out.append(p)
        return out

    def download(self, p):
        self.downloads.append(p)
        return self.files_[p]


def path(day: str, market: str) -> str:
    return f"/xds_nfs/edp_processed/BASIC/{day}/35000001/{market}.bz2"


DATA = bz2.compress(b'{"op":"mcm","clk":"1","pt":1764000000000,"mc":[]}\n')


def test_the_last_days_are_fetched_once_kept_raw_and_added_to_bronze(tmp_path):
    store = Store(root=tmp_path)
    api = FakeAPI({path("2026/Nov/29", "1.1"): DATA, path("2026/Dec/1", "1.2"): DATA,
                   path("2026/Nov/10", "1.3"): DATA})                     # outside the 10 days: not asked for
    got = ha.update(store, api, days=10, today=date(2026, 12, 2))
    assert (got["listed"], got["fetched"], got["held"], got["failed"]) == (2, 2, 0, 0)
    assert got["bronze"]["new"] == 2
    assert [b["sport"] for b in api.bodies] == ["Soccer", "Soccer"]      # the range crosses a month: two lists
    assert all(b["marketTypesCollection"] == ha.MARKETS and b["countriesCollection"] == [] for b in api.bodies)
    assert store.get("betfair_historic/raw/soccer/xds_nfs/edp_processed/BASIC/2026/Nov/29/35000001/1.1.bz2") == DATA
    again = ha.update(store, api, days=10, today=date(2026, 12, 2))
    assert (again["fetched"], again["held"]) == (0, 2) and len(api.downloads) == 2


def test_a_spent_budget_stops_the_run_and_leaves_the_rest_for_the_next():
    import tempfile
    store = Store(root=tempfile.mkdtemp())
    api = FakeAPI({path("2026/Dec/1", "1.2"): DATA})
    got = ha.update(store, api, days=3, today=date(2026, 12, 2), max_minutes=0)
    assert got["fetched"] == 0 and got["stopped"] is True


def test_months_split_a_range_at_the_month_end():
    assert ha.months(date(2026, 11, 25), date(2026, 12, 3)) == [(date(2026, 11, 25), date(2026, 11, 30)),
                                                                (date(2026, 12, 1), date(2026, 12, 3))]
