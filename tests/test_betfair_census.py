"""sources/betfair_census.py: Betfair racing by country, apart from the live recorder (the owner, 2 Nov 2026)."""
import json
from datetime import date, datetime, timezone

import sources.betfair_census as bc

ALLOWED = ("listMarketCatalogue", "listMarketBook", "listClearedOrders", "listEventTypes", "listMarketTypes")


class Data:
    """As trading/exchange.py BetfairData: a call outside the read-only list raises (listCountries did, 2 Nov)."""

    def __init__(self, fail_day=None):
        self.calls = []
        self.fail_day = fail_day

    def _read(self, method, params):
        if method not in ALLOWED:
            raise ValueError(f"{method}: this client only reads markets, prices and settled bets")
        f = params["filter"]
        assert f["eventTypeIds"] == ["7"] and f["marketTypeCodes"] == ["WIN"] and "marketCountries" not in f
        self.calls.append(f["marketStartTime"]["from"])
        if self.fail_day and f["marketStartTime"]["from"].startswith(self.fail_day):
            raise RuntimeError("unread")
        return [{"totalMatched": 500.0, "event": {"countryCode": "HK", "venue": "Sha Tin"},
                 "marketStartTime": "2026-11-02T05:00:00Z"},
                {"totalMatched": 1000.0, "event": {"countryCode": "HK", "venue": "Sha Tin"},
                 "marketStartTime": "2026-11-02T04:30:00Z"},
                {"totalMatched": 100.0, "event": {"countryCode": "JP", "venue": "Tokyo"},
                 "marketStartTime": "2026-11-02T03:00:00Z"}]


class NoS3:
    def put_object(self, **kw):
        raise RuntimeError("no S3 here")


def test_the_census_uses_only_the_read_only_calls_and_groups_by_country(tmp_path):
    d = Data(fail_day="2026-11-03")
    got = bc.census_racing_countries(d, hours=72, now=datetime(2026, 11, 1, 5, 30, tzinfo=timezone.utc))
    assert len(d.calls) == 3 and got["calls"] == 3 and len(got["errors"]) == 1
    assert [r["country"] for r in got["countries"]] == ["HK", "JP"]
    hk = got["countries"][0]
    assert hk["win_markets"] == 4 and hk["win_matched_total"] == 3000.0 and hk["venues"] == ["Sha Tin"]
    assert hk["first_off_utc"] == "2026-11-02T04:30:00Z" and got["capped_days"] == 0
    where = bc.save(got, date(2026, 11, 1), s3=NoS3(), root=tmp_path, name="census_racing_0530.json")
    assert where.endswith("census_racing_0530.json")
    assert json.loads((tmp_path / "2026-11-01" / "census_racing_0530.json").read_text())["countries"][0]["country"] == "HK"


def test_the_census_names_only_calls_the_read_only_client_permits():
    import inspect
    src = inspect.getsource(bc)
    for m in ("listCountries", "listEvents", "listCompetitions", "placeOrders"):
        assert m not in src.replace("permits no listCountries", ""), m


def test_the_live_recorder_is_untouched_by_the_census():
    import betfair_recorder
    assert not hasattr(betfair_recorder, "census_racing_countries")
