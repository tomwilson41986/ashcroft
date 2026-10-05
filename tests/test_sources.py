"""The historic data for the other markets (the owner's ask, 3 Oct 2026): each source fetched once, resumably, into its
store, and read into tables the research can trust; Betfair's Historic Data files read to opening and closing prices;
the recorder's census of every sport, its ante-post snapshot and its records of their own."""

import bz2
import io
import json
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

import betfair_historic as bh
import betfair_recorder as br
from sources import football, gbgb, puntingform, tennis
from sources.common import Budget, Store


class Reply:
    def __init__(self, body=None, status=200, content=None):
        self.status_code = status
        self._body = body
        self.content = content if content is not None else json.dumps(body).encode()
        self.headers = {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


# --------------------------------------------------------------------------------------------------------------------
# GBGB
# --------------------------------------------------------------------------------------------------------------------

def _meeting(mid, day="01/10/2026"):
    return [{"meetingDate": day, "meetingId": mid, "trackName": "Romford", "races": [{
        "raceTime": "18:33:00", "raceDate": day, "raceId": mid * 10, "raceNumber": "1", "raceTitle": "A",
        "raceType": "Flat", "raceHandicap": False, "raceClass": "A5", "raceDistance": 400.0, "raceGoing": "-10",
        "raceForecast": "(5-3) £11.28", "raceTricast": "(5-3-2) £28.88", "racePrizes": "1st £300",
        "traps": [
            {"trapNumber": "5", "dogId": 1, "dogName": "Hoffa", "SP": "11/4", "resultPosition": 1,
             "resultPriceNumerator": 11, "resultPriceDenominator": 4, "resultBtnDistance": "",
             "resultSectionalTime": "03.78", "resultComment": "QAw,AlwaysLed", "resultRunTime": "24.46",
             "resultDogWeight": "34.9", "resultAdjustedTime": "24.36", "trainerName": "P W Young"},
            {"trapNumber": "3", "dogId": 2, "dogName": "Sapphire", "SP": "EvsF", "resultPosition": 2,
             "resultBtnDistance": "1 3/4", "resultRunTime": "24.60"},
            {"trapNumber": "1", "dogId": 3, "dogName": "Siddy", "SP": "6/1", "resultPosition": 3,
             "resultBtnDistance": "Nk"}]}]}]


class GbgbSite:
    def __init__(self):
        self.gets = []

    def get(self, url, params=None, timeout=None):
        self.gets.append((url, dict(params or {})))
        if url.endswith("/results"):
            page = params["page"]
            items = [{"meetingId": 7}, {"meetingId": 8}] if page == 1 else [{"meetingId": 8}, {"meetingId": 9}]
            return Reply({"items": items, "meta": {"count": 4, "page": page, "pageCount": 2}})
        mid = int(url.rsplit("/", 1)[1])
        return Reply(_meeting(mid))


def test_gbgb_reads_every_meeting_of_a_day_and_flattens_each_dog_a_race(tmp_path):
    store, site = Store(root=tmp_path), GbgbSite()
    got = gbgb.fetch(store, date(2026, 10, 1), date(2026, 10, 1), s=site, workers=1)
    assert got["fetched"] == 1 and got["races"] == 3 and got["capped"] == 0
    meetings = sorted(int(u.rsplit("/", 1)[1]) for u, _ in site.gets if "/meeting/" in u)
    assert meetings == [7, 8, 9]                                  # each meeting once, across both pages
    gbgb.build(store)
    df = store.get_parquet("gbgb/runs_2026.parquet")
    assert len(df) == 9 and set(df["race_date"]) == {"2026-10-01"}
    first = df[(df["meeting_id"] == 7) & (df["trap"] == 5)].iloc[0]
    assert first["sp_decimal"] == 3.75 and first["sectional"] == 3.78 and first["comment"] == "QAw,AlwaysLed"
    assert first["going"] == -10.0 and first["runners"] == 3
    assert df[(df["meeting_id"] == 7) & (df["trap"] == 3)].iloc[0]["sp_decimal"] == 2.0      # evens, from the text
    assert df[(df["meeting_id"] == 7) & (df["trap"] == 3)].iloc[0]["beaten_lengths"] == 1.75


def test_gbgb_fetches_only_what_the_store_lacks_and_the_last_days_again(tmp_path):
    store = Store(root=tmp_path)
    store.put(gbgb.raw_key(date(2026, 9, 20)), b"x")
    store.put(gbgb.raw_key(date(2026, 9, 30)), b"x")
    site = GbgbSite()
    got = gbgb.fetch(store, date(2026, 9, 19), date(2026, 10, 1), s=site, refresh_days=3, workers=1)
    # 13 days: 20 Sep is held and old, so skipped; 30 Sep is held but recent, so fetched again
    assert got["skipped"] == 1 and got["fetched"] == 12


def test_gbgb_distances_beaten():
    assert gbgb.lengths("5 1/2") == 5.5 and gbgb.lengths("3/4") == 0.75 and gbgb.lengths("Hd") == 0.1
    assert gbgb.lengths("") == 0.0 and gbgb.lengths("Dis") is None and gbgb.lengths("abc") is None


def test_a_budget_runs_out_after_its_minutes():
    t = iter([0.0, 30.0, 61.0])
    budget = Budget(1.0, clock=lambda: next(t))
    assert budget.left() and not budget.left()
    assert Budget(None).left()


# --------------------------------------------------------------------------------------------------------------------
# football-data.co.uk
# --------------------------------------------------------------------------------------------------------------------

MAIN_CSV = ("﻿Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,PSCH,PSCD,PSCA,BFECH,BFECD,BFECA,,\n"
            "E0,15/08/2025,20:00,Liverpool,Bournemouth,4,2,H,1.30,6.0,9.0,1.31,6.2,9.6,,\n"
            "E0,16/08/25,15:00,Villa,Newcastle,0,0,D,2.5,3.4,2.9,,,,,\n")
NEW_CSV = ("Country,League,Season,Date,Time,Home,Away,HG,AG,Res,PSCH,PSCD,PSCA\n"
           "Brazil,Serie A,2012,19/05/2012,22:30,Palmeiras,Portuguesa,1,1,D,1.8,3.5,4.9\n")


class FootballSite:
    def __init__(self):
        self.urls = []

    def get(self, url, params=None, timeout=None):
        self.urls.append(url)
        if url.endswith("/mmz4281/2526/E0.csv"):
            return Reply(content=MAIN_CSV.encode())
        if url.endswith("/new/BRA.csv"):
            return Reply(content=NEW_CSV.encode())
        return Reply(status=404, content=b"")


def test_football_fetches_each_league_season_and_names_both_layouts_alike(tmp_path, monkeypatch):
    monkeypatch.setattr("sources.common.time.sleep", lambda s: None)
    store, site = Store(root=tmp_path), FootballSite()
    got = football.fetch(store, first_season=2025, s=site, today=date(2026, 3, 1))
    assert got["fetched"] == 2 and got["missing"] == len(football.EXTRA) + len(football.MAIN) - 2
    out = football.build(store)
    m = store.get_parquet("football/matches.parquet")
    assert out["matches"] == 3 and out["with_betfair_close"] == 1
    bra = m[m["Div"] == "BRA"].iloc[0]
    assert (bra["HomeTeam"], bra["FTHG"], bra["FTR"], bra["match_date"]) == ("Palmeiras", 1, "D", "2012-05-19")
    e0 = m[m["Div"] == "E0"].sort_values("match_date")
    assert list(e0["match_date"]) == ["2025-08-15", "2025-08-16"] and e0.iloc[0]["BFECH"] == 1.31
    assert set(m["season"]) == {"2025", "2012"}
    # a second run takes the current season and the extra leagues again, and only what the store lacks before it
    store.put("football/raw/2425/E0.csv", MAIN_CSV.encode())
    site2 = FootballSite()
    football.fetch(store, first_season=2024, s=site2, today=date(2026, 3, 1))
    assert any(u.endswith("/2526/E0.csv") for u in site2.urls) and any(u.endswith("/new/BRA.csv") for u in site2.urls)
    assert not any(u.endswith("/2425/E0.csv") for u in site2.urls)


def test_the_football_season_turns_in_july():
    assert football.current_season(date(2026, 6, 30)) == 2025 and football.current_season(date(2026, 7, 1)) == 2026
    assert football.season_code(1999) == "9900" and football.season_code(2025) == "2526"


# --------------------------------------------------------------------------------------------------------------------
# Tennis
# --------------------------------------------------------------------------------------------------------------------

SACK = ("tourney_id,tourney_name,surface,tourney_level,tourney_date,winner_id,winner_name,loser_id,loser_name,score,"
        "round,w_svpt,w_1stIn,l_svpt,winner_rank\n"
        "2024-0339,Brisbane,Hard,A,20240101,1,Dimitrov,2,Rune,7-6(5) 6-4,F,70,45,75,14\n")


def test_tennis_builds_sackmann_matches_with_tour_and_level(tmp_path):
    store = Store(root=tmp_path)
    store.put("tennis/raw/sackmann/atp_matches_2024.csv", SACK.encode())
    store.put("tennis/raw/sackmann/atp_matches_qual_chall_2024.csv", SACK.encode())
    store.put("tennis/raw/sackmann/atp_players.csv", b"player_id,name_first\n1,Grigor\n")
    buf = io.BytesIO()
    pd.DataFrame({"Date": ["2024-01-07"], "Winner": ["Dimitrov G."], "Loser": ["Rune H."], "PSW": ["1.9"],
                  "PSL": ["1.95"], "Surface": ["Hard"]}).to_excel(buf, index=False)
    store.put("tennis/raw/tennis_data/atp_2024.xlsx", buf.getvalue())
    out = tennis.build(store)
    m = store.get_parquet("tennis/sackmann_matches.parquet")
    assert out["sackmann"]["matches"] == 2 and set(m["level_file"]) == {"tour", "challenger_qualifying"}
    assert m.iloc[0]["match_date"] == "2024-01-01" and m["w_svpt"].iloc[0] == 70
    o = store.get_parquet("tennis/odds_matches.parquet")
    assert out["tennis_data"]["with_pinnacle"] == 1 and o.iloc[0]["PSW"] == 1.9 and o.iloc[0]["tour"] == "atp"


class TennisSite:
    """The mirror has the tour files, TML its ATP years; tennis-data's Cloudflare refuses everything."""

    def __init__(self):
        self.urls = []

    def get(self, url, params=None, timeout=None):
        self.urls.append(url)
        if "tennis-data.co.uk" in url:
            return Reply(status=403, content=b"<html>blocked</html>")
        if url.endswith("/atp/atp_matches_2026.csv") or url.endswith("/wta/wta_matches_2025.csv"):
            return Reply(content=SACK.encode())
        if "TML-Database" in url and url.endswith("/2026.csv"):
            return Reply(content=SACK.replace("Dimitrov", "Sinner").encode())
        return Reply(status=404, content=b"404: Not Found")


def test_tennis_reads_the_sackmann_mirror_and_tml_and_keeps_one_row_a_match(tmp_path, monkeypatch):
    monkeypatch.setattr("sources.common.time.sleep", lambda s: None)
    store, site = Store(root=tmp_path), TennisSite()
    got = tennis.fetch(store, first_year=2025, s=site, today=date(2026, 10, 3))
    assert got["fetched"] == 3 and got["failed"] == 0
    assert all("JeffSackmann" not in u for u in site.urls)                   # the taken-down repositories
    assert not any("tennis-data" in u for u in site.urls)                   # the odds are their own source
    store.put("tennis/raw/tml/2025.csv", SACK.encode())                     # the same match as the mirror's 2025
    tennis.build(store)
    m = store.get_parquet("tennis/sackmann_matches.parquet")
    assert sorted(m["winner_name"]) == ["Dimitrov", "Dimitrov", "Sinner"]
    assert set(m["source"]) == {"sackmann", "tml"}


def test_tennis_odds_stop_at_the_first_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr("sources.common.time.sleep", lambda s: None)
    site = TennisSite()
    got = tennis.fetch_odds(Store(root=tmp_path), first_year=2000, s=site, today=date(2026, 10, 3))
    assert got["refused"] and got["fetched"] == 0
    assert len(site.urls) <= 4                                               # not 94 refused requests


def test_tennis_lists_wta_odds_only_from_2007():
    stems = [s for s, _ in tennis.tennis_data_files(2005, 2008)]
    assert "wta_2006" not in stems and "wta_2007" in stems and "atp_2005" in stems
    assert tennis.tennis_data_files(2014, 2014)[0][1][0].endswith("/2014/2014.xlsx")


# --------------------------------------------------------------------------------------------------------------------
# Punting Form
# --------------------------------------------------------------------------------------------------------------------

def test_puntingform_without_a_key_does_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("PUNTINGFORM_API_KEY", raising=False)
    assert puntingform.fetch(Store(root=tmp_path))["fetched"] == 0


def test_puntingform_never_logs_its_key(tmp_path, caplog):
    class Refusing:
        def get(self, url, params=None, timeout=None):
            raise RuntimeError(f"401 for {url}?apiKey={params['apiKey']}")
    got = puntingform.fetch(Store(root=tmp_path), date(2026, 9, 1), date(2026, 9, 30), s=Refusing(), key="SECRET123")
    assert got["fetched"] == 0 and got["failed"] == 5                    # gives up on a refused key
    assert "SECRET123" not in caplog.text


# --------------------------------------------------------------------------------------------------------------------
# Betfair Historic Data
# --------------------------------------------------------------------------------------------------------------------

OFF = datetime(2026, 9, 1, 15, 0, tzinfo=timezone.utc)


def _ms(t):
    return int(t.timestamp() * 1000)


def _stream():
    defn = {"eventId": "35", "eventName": "Alpha v Bravo", "eventTypeId": "2", "marketType": "MATCH_ODDS",
            "name": "Match Odds", "countryCode": "GB", "marketTime": "2026-09-01T15:00:00.000Z", "inPlay": False,
            "runners": [{"id": 11, "name": "Alpha", "status": "ACTIVE", "sortPriority": 1},
                        {"id": 22, "name": "Bravo", "status": "ACTIVE", "sortPriority": 2}]}
    msgs = [
        {"op": "mcm", "pt": _ms(OFF - timedelta(days=2)), "mc": [{"id": "1.5", "marketDefinition": defn,
                                                                  "rc": [{"ltp": 2.0, "id": 11}, {"ltp": 2.0, "id": 22}]}]},
        {"op": "mcm", "pt": _ms(OFF - timedelta(minutes=90)), "mc": [{"id": "1.5", "rc": [{"ltp": 1.9, "id": 11}]}]},
        {"op": "mcm", "pt": _ms(OFF - timedelta(minutes=30)), "mc": [{"id": "1.5", "rc": [{"ltp": 1.8, "id": 11}]}]},
        {"op": "mcm", "pt": _ms(OFF - timedelta(seconds=30)), "mc": [{"id": "1.5", "rc": [{"ltp": 1.7, "id": 11}]}]},
        {"op": "mcm", "pt": _ms(OFF + timedelta(minutes=1)), "mc": [{"id": "1.5", "marketDefinition": {**defn, "inPlay": True}}]},
        {"op": "mcm", "pt": _ms(OFF + timedelta(minutes=50)), "mc": [{"id": "1.5", "rc": [{"ltp": 1.1, "id": 11}]}]},
        {"op": "mcm", "pt": _ms(OFF + timedelta(minutes=99)), "mc": [{"id": "1.5", "marketDefinition": {
            **defn, "inPlay": True, "status": "CLOSED",
            "runners": [{"id": 11, "name": "Alpha", "status": "WINNER"}, {"id": 22, "name": "Bravo", "status": "LOSER"}]}}]},
    ]
    return bz2.compress("\n".join(json.dumps(m) for m in msgs).encode())


def test_a_historic_market_file_reads_to_the_prices_before_the_off_and_the_result():
    rows = {r["selection_id"]: r for r in bh.parse_market(_stream())}
    a = rows[11]
    assert a["ltp_first"] == 2.0 and a["ltp_close"] == 1.7                   # the last price before it went in play
    assert a["ltp_t_1440"] == 2.0 and a["ltp_t_60"] == 1.9 and a["ltp_t_15"] == 1.8 and a["ltp_t_1"] == 1.8
    assert a["won"] == 1 and rows[22]["won"] == 0 and a["market_type"] == "MATCH_ODDS" and a["event_type_id"] == "2"
    assert a["went_in_play_utc"].startswith("2026-09-01T15:01")


def test_historic_months_run_newest_first_and_the_path_names_the_year():
    got = list(bh.months_back(date(2026, 2, 10), date(2025, 12, 15)))
    assert got == [(date(2026, 2, 1), date(2026, 2, 10)), (date(2026, 1, 1), date(2026, 1, 31)),
                   (date(2025, 12, 15), date(2025, 12, 31))]
    assert bh.path_year("betfair_historic/raw/tennis/xds_nfs/edp_processed/BASIC/2023/Jan/1/32/1.2.bz2") == 2023
    assert bh.held_sports([{"sport": "Tennis", "plan": "Basic Plan"}, {"sport": "Soccer", "plan": "Pro Plan"}]) == ["Tennis"]


class FakeHistoric:
    def __init__(self):
        self.downloads = []

    def files(self, body):
        if body["sport"] == "Tennis" and (body["fromYear"], body["fromMonth"]) == (2026, 9):
            return ["/xds/BASIC/2026/Sep/1/35/1.5.bz2", "/xds/BASIC/2026/Sep/1/35/1.6.bz2"]
        return []

    def download(self, path):
        self.downloads.append(path)
        return _stream()


def test_historic_fetch_keeps_each_file_once_by_sport_and_builds_its_table(tmp_path):
    store, hd = Store(root=tmp_path), FakeHistoric()
    got = bh.fetch(store, hd, ["Tennis"], date(2026, 8, 1), date(2026, 9, 30), workers=1)
    assert got["fetched"] == 2 and got["sports"]["Tennis"]["months"] == 2
    again = bh.fetch(store, FakeHistoric(), ["Tennis"], date(2026, 8, 1), date(2026, 9, 30), workers=1)
    assert again["fetched"] == 0 and again["skipped"] == 2
    out = bh.build(store, ["Tennis"])
    assert out["tennis_2026"]["files"] == 2 and out["tennis_2026"]["with_close"] == 2       # 1.5 read twice: one market
    t = store.get_parquet("betfair_historic/markets_tennis_2026.parquet")
    assert len(t) == 2


def test_historic_fetch_without_a_held_sport_says_so(tmp_path):
    class Empty(FakeHistoric):
        def my_data(self):
            return []
    assert bh.fetch(Store(root=tmp_path), Empty()) == {"fetched": 0, "sports": []}


# --------------------------------------------------------------------------------------------------------------------
# The recorder: every sport's census, the ante-post snapshot, records of their own
# --------------------------------------------------------------------------------------------------------------------

class SportsData:
    def __init__(self):
        self.calls = []

    def _read(self, method, params):
        self.calls.append((method, params))
        if method == "listEventTypes":
            return [{"eventType": {"id": "1", "name": "Soccer"}, "marketCount": 900},
                    {"eventType": {"id": "2", "name": "Tennis"}, "marketCount": 300}]
        if method == "listMarketTypes":
            return [{"marketType": "MATCH_ODDS", "marketCount": 100}, {"marketType": "OVER_UNDER_25", "marketCount": 90}]
        if method == "listMarketCatalogue":
            big = 100000.0 if params["filter"]["eventTypeIds"] == ["1"] else 3000.0
            return [{"marketId": "1.1", "marketName": "Match Odds", "totalMatched": big,
                     "description": {"marketType": "MATCH_ODDS"}, "event": {"name": "A v B"}},
                    {"marketId": "1.2", "marketName": "Over/Under 2.5", "totalMatched": big / 10,
                     "description": {"marketType": "OVER_UNDER_25"}, "event": {"name": "A v B"}}]
        raise AssertionError(method)


def test_the_census_reads_every_sport_its_market_types_and_matched_money(tmp_path):
    data = SportsData()
    got = br.census_sports(data, now=datetime(2026, 10, 3, 20, 0, tzinfo=timezone.utc))
    assert [r["sport"] for r in got["sports"]] == ["Soccer", "Tennis"]                 # busiest first
    soccer = got["sports"][0]
    assert soccer["top_matched_total"] == 110000.0 and soccer["market_types"]["MATCH_ODDS"] == 100
    assert soccer["top_matched_by_type"] == {"MATCH_ODDS": 100000.0, "OVER_UNDER_25": 10000.0}
    assert {m for m, _ in data.calls} <= {"listEventTypes", "listMarketTypes", "listMarketCatalogue"}
    flt = data.calls[0][1]["filter"]["marketStartTime"]
    assert flt == {"from": "2026-10-03T20:00:00Z", "to": "2026-10-04T20:00:00Z"}

    class S3:
        def __init__(self):
            self.put = {}

        def put_object(self, Bucket, Key, Body):
            self.put[Key] = Body
    s3 = S3()
    where = br.save_census(got, date(2026, 10, 3), s3=s3, root=tmp_path)
    assert where.endswith("betfair_live/2026-10-03/census_sports.json")
    assert json.loads(s3.put["betfair_live/2026-10-03/census_sports.json"])["sports"][1]["sport"] == "Tennis"


def test_the_exchange_reader_allows_the_census_and_still_refuses_orders():
    from trading.exchange import BetfairData

    class Client:
        def _api_call(self, method, params):
            return [method]
    data = BetfairData(client=Client())
    assert data._read("listEventTypes", {}) == ["listEventTypes"]
    assert data._read("listMarketTypes", {}) == ["listMarketTypes"]
    with pytest.raises(ValueError):
        data._read("placeOrders", {})


def test_the_antepost_snapshot_reads_far_ahead_and_keeps_its_own_files(tmp_path):
    class Data:
        def __init__(self):
            self.calls = []

        def _read(self, method, params):
            self.calls.append((method, params))
            if method == "listMarketCatalogue":
                if params["filter"]["marketTypeCodes"] == ["ANTEPOST_WIN"] and len(self.calls) == 1:
                    return [{"marketId": "1.9", "marketName": "Grand National", "marketStartTime": "2027-04-10T15:00:00Z",
                             "event": {"venue": "Aintree", "countryCode": "GB"}, "runners": [
                                 {"selectionId": 5, "runnerName": "Alpha", "sortPriority": 1}],
                             "description": {"marketType": "ANTEPOST_WIN"}}]
                return []
            return [{"marketId": "1.9", "status": "OPEN", "runners": [
                {"selectionId": 5, "status": "ACTIVE", "ex": {"availableToBack": [{"price": 34.0, "size": 20}]}}]}]

    rec = br.DayRecorder(date(2026, 10, 3), root=tmp_path, tag="antepost", background=False, s3=_NullS3())
    data = Data()
    got = br.antepost(data, rec, now=datetime(2026, 10, 3, 21, 0, tzinfo=timezone.utc))
    assert got == {"markets": 1, "rows": 1}
    assert (tmp_path / "2026-10-03" / "books_antepost.csv").exists()
    assert not (tmp_path / "2026-10-03" / "books.csv").exists()            # the nightly load's files untouched
    cats = [p for m, p in data.calls if m == "listMarketCatalogue"]
    assert cats[-1]["filter"]["marketStartTime"]["to"].startswith("2027-11")  # 400 days ahead


class _NullS3:
    def put_object(self, **kw):
        pass


def test_a_record_of_its_own_settles_its_own_markets(tmp_path):
    day = date(2026, 10, 3)
    main = br.DayRecorder(day, root=tmp_path, background=False, s3=_NullS3())
    main.record_catalogue([{"marketId": "1.1", "runners": [{"selectionId": 1, "runnerName": "A"}]}])
    dogs = br.DayRecorder(day, root=tmp_path, tag="greyhound", background=False, s3=_NullS3())
    dogs.record_catalogue([{"marketId": "1.77", "runners": [{"selectionId": 2, "runnerName": "B"}]}])

    class Data:
        def __init__(self):
            self.asked = []

        def _read(self, method, params):
            self.asked += params["marketIds"]
            return []
    data = Data()
    br.final(day, data, dogs)
    assert data.asked == ["1.77"]
    data2 = Data()
    br.final(day, data2, main)
    assert data2.asked == ["1.1"]


def test_greyhounds_need_their_own_tag_and_read_their_event_type(monkeypatch, tmp_path):
    seen = {}

    def fake_record(day, until, data, rec, *a, event_type="7", **kw):
        seen["event_type"], seen["books"] = event_type, rec.books_path.name
        return {}

    class Data:
        def login(self):
            pass
    monkeypatch.setattr(br, "record", fake_record)
    monkeypatch.setattr(br, "ROOT", tmp_path)
    monkeypatch.setattr("trading.exchange.BetfairData", lambda *a, **k: Data())
    br.main(["--record", "--event-type", "4339", "--date", "2026-10-03"])
    assert seen == {"event_type": "4339", "books": "books_greyhound.csv"}
    with pytest.raises(SystemExit):
        br.main(["--record", "--event-type", "1", "--date", "2026-10-03"])


# --------------------------------------------------------------------------------------------------------------------
# A bundle downloaded from the Historic Data website (the owner's data.tar of 5 Oct)
# --------------------------------------------------------------------------------------------------------------------

def _event_file():
    """An event's own file: two markets in one stream (an ante-post market and a win market)."""
    msgs = [{"op": "mcm", "pt": _ms(OFF - timedelta(days=1)), "mc": [
                {"id": "1.7", "marketDefinition": {"eventId": "35", "eventTypeId": "4339", "marketType": "ANTEPOST_WIN",
                                                   "marketTime": "2026-09-01T15:00:00.000Z",
                                                   "runners": [{"id": 1, "status": "ACTIVE"}]},
                 "rc": [{"ltp": 5.0, "id": 1}]},
                {"id": "1.8", "marketDefinition": {"eventId": "35", "eventTypeId": "4339", "marketType": "WIN",
                                                   "marketTime": "2026-09-01T15:00:00.000Z",
                                                   "runners": [{"id": 1, "status": "ACTIVE"}, {"id": 2, "status": "ACTIVE"}]},
                 "rc": [{"ltp": 3.0, "id": 1}, {"ltp": 1.5, "id": 2}]}]}]
    return bz2.compress("\n".join(json.dumps(m) for m in msgs).encode())


def _bundle(path):
    import tarfile
    with tarfile.open(path, "w") as t:
        for name, body in (("BASIC/2026/Sep/1/35/1.5.bz2", _stream()), ("BASIC/2026/Sep/1/35/35.bz2", _event_file()),
                           ("BASIC/2026/Sep/1/35/1.9.bz2", b"not a stream")):
            info = tarfile.TarInfo(name)
            info.size = len(body)
            t.addfile(info, io.BytesIO(body))


def test_an_event_file_reads_each_of_its_markets_apart():
    rows = bh.parse_market(_event_file())
    assert sorted((r["market_id"], r["selection_id"]) for r in rows) == [("1.7", 1), ("1.8", 1), ("1.8", 2)]
    assert {r["market_id"]: r["market_type"] for r in rows} == {"1.7": "ANTEPOST_WIN", "1.8": "WIN"}
    assert [r["ltp_close"] for r in rows if r["market_id"] == "1.8"] == [3.0, 1.5]


def test_a_website_bundle_is_stored_as_the_api_would_and_its_table_built(tmp_path):
    _bundle(tmp_path / "data.tar")
    store = Store(root=tmp_path / "store")
    got = bh.import_tar(store, str(tmp_path / "data.tar"), "Greyhound Racing", workers=2)
    assert got["files"] == 3 and got["stored"] == 3 and got["unreadable"] == 1
    assert got["years"][2026]["markets"] == 3                       # 1.5 (its own file), 1.7 and 1.8 (the event's)
    raw = store.listing("betfair_historic/raw/greyhound_racing/")
    assert "betfair_historic/raw/greyhound_racing/BASIC/2026/Sep/1/35/1.5.bz2" in raw
    t = store.get_parquet("betfair_historic/markets_greyhound_racing_2026.parquet")
    assert set(t["market_id"]) == {"1.5", "1.7", "1.8"}
    # the nightly build finds the table built from these very files and leaves it
    assert bh.build(store, ["Greyhound Racing"])["greyhound_racing_2026"] == {"files": 3, "unchanged": True}
    # imported again: nothing stored twice, the table the same
    again = bh.import_tar(store, str(tmp_path / "data.tar"), "Greyhound Racing", workers=2)
    assert again["stored"] == 0 and again["already"] == 3 and again["years"][2026]["markets"] == 3


def test_the_api_fetch_skips_a_file_a_bundle_already_brought(tmp_path):
    store = Store(root=tmp_path)
    store.put("betfair_historic/raw/tennis/BASIC/2026/Sep/1/35/1.5.bz2", _stream())
    hd = FakeHistoric()
    got = bh.fetch(store, hd, ["Tennis"], date(2026, 9, 1), date(2026, 9, 30), workers=1)
    assert hd.downloads == ["/xds/BASIC/2026/Sep/1/35/1.6.bz2"] and got["skipped"] == 1


def test_the_import_queue_takes_each_bundle_once(tmp_path, monkeypatch):
    _bundle(tmp_path / "data.tar")
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps([{"name": "dogs", "sport": "Greyhound Racing", "url": "https://example/dl"}]))
    runs = []

    def fake_run(cmd, check):
        runs.append(cmd)
        import shutil
        shutil.copy(tmp_path / "data.tar", cmd[cmd.index("-o") + 1])
    monkeypatch.setattr("subprocess.run", fake_run)
    store = Store(root=tmp_path / "store")
    first = bh.import_queue(store, str(queue), dest=str(tmp_path))
    assert first["dogs"]["stored"] == 3 and len(runs) == 1
    assert bh.import_queue(store, str(queue), dest=str(tmp_path)) == {"dogs": "already imported"} and len(runs) == 1


def test_the_queued_bundles_name_a_sport_the_service_knows():
    items = json.loads(open("sources/betfair_imports.json").read())
    assert items and all(it["sport"] in bh.PLAN_SCOPE and it["url"].startswith("https://") for it in items)
    assert len({it["name"] for it in items}) == len(items)


def test_a_price_betfair_writes_as_text_is_read_as_missing(tmp_path):
    df = bh._tidy(pd.DataFrame({"bsp": [2.5, "NaN", "Infinity", None], "market_id": ["1.1"] * 4,
                                "turned_in_play": [True, None, False, True]}))
    assert df["bsp"].iloc[0] == 2.5 and df["bsp"].iloc[1:].isna().iloc[0]
    store = Store(root=tmp_path)
    store.put_parquet("t.parquet", df)                                # the write that failed on 5 Oct
    assert len(store.get_parquet("t.parquet")) == 4
