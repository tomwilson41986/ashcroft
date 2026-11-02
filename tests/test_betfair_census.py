"""sources/betfair_census.py: Betfair racing by country, apart from the live recorder (the owner, 2 Nov 2026)."""
import json
from datetime import date, datetime, timezone

import sources.betfair_census as bc


class Data:
    def _read(self, method, params):
        if method == "listCountries":
            assert params["filter"]["eventTypeIds"] == ["7"]
            return [{"countryCode": "JP", "marketCount": 24}, {"countryCode": "HK", "marketCount": 20},
                    {"countryCode": "XX", "marketCount": 1}]
        cc = params["filter"]["marketCountries"][0]
        assert params["filter"]["marketTypeCodes"] == ["WIN"]
        if cc == "XX":
            raise RuntimeError("unread")
        m = 100.0 if cc == "JP" else 500.0
        return [{"totalMatched": m, "event": {"venue": "V" + cc}, "marketStartTime": "2026-11-02T04:30:00Z"},
                {"totalMatched": 2 * m, "event": {"venue": "V" + cc}, "marketStartTime": "2026-11-02T05:00:00Z"}]


class NoS3:
    def put_object(self, **kw):
        raise RuntimeError("no S3 here")


def test_the_census_sorts_by_money_and_survives_a_bad_country(tmp_path):
    got = bc.census_racing_countries(Data(), hours=168, now=datetime(2026, 11, 1, 5, 30, tzinfo=timezone.utc))
    assert got["hours"] == 168 and [r["country"] for r in got["countries"]][:2] == ["HK", "JP"]
    hk = got["countries"][0]
    assert hk["win_markets"] == 2 and hk["win_matched_total"] == 1500.0 and hk["venues"] == ["VHK"]
    assert hk["first_off_utc"] == "2026-11-02T04:30:00Z"
    assert "error" in [r for r in got["countries"] if r["country"] == "XX"][0]
    where = bc.save(got, date(2026, 11, 1), s3=NoS3(), root=tmp_path, name="census_racing_0530.json")
    assert where.endswith("census_racing_0530.json")
    assert json.loads((tmp_path / "2026-11-01" / "census_racing_0530.json").read_text())["countries"][0]["country"] == "HK"


def test_the_live_recorder_is_untouched_by_the_census():
    import betfair_recorder
    assert not hasattr(betfair_recorder, "census_racing_countries")
