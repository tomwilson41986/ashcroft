"""The market recorder (betfair_recorder.py), its hook in the trader's reads, and the S3 archive of the price files.

Stand-ins throughout: a fake S3 and fake Betfair replies shaped like the API's. Nothing here reaches Betfair or S3,
and nothing here is used to train or evaluate a model; the tests only check the plumbing.
"""

import csv
import gzip
import io
import sqlite3
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

import betfair_prices as bp
import betfair_recorder as br
from trading.exchange import BetfairData

DAY = date(2026, 10, 1)


class FakeS3:
    def __init__(self, fail=False):
        self.objects, self.fail = {}, fail

    def put_object(self, Bucket, Key, Body):
        if self.fail:
            raise RuntimeError("S3 unreachable")
        self.objects[(Bucket, Key)] = bytes(Body)

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise RuntimeError("NoSuchKey")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise RuntimeError("404")
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        keys = sorted(k for b, k in self.objects if b == Bucket and k.startswith(Prefix))
        return {"Contents": [{"Key": k, "Size": len(self.objects[(Bucket, k)])} for k in keys], "IsTruncated": False}


def raw_book(mid="1.100", status="OPEN", inplay=False, runners=None, total=5000.0):
    runners = runners or [
        {"selectionId": 11, "status": "ACTIVE", "lastPriceTraded": 3.5, "totalMatched": 0.0,
         "ex": {"availableToBack": [{"price": 3.4, "size": 20.0}, {"price": 3.3, "size": 11.5}],
                "availableToLay": [{"price": 3.6, "size": 9.0}]},
         "sp": {"nearPrice": "NaN", "farPrice": "Infinity"}},
        {"selectionId": 12, "status": "REMOVED", "adjustmentFactor": 12.5, "removalDate": "2026-10-01T09:00:00Z"},
    ]
    return {"marketId": mid, "status": status, "inplay": inplay, "totalMatched": total, "numberOfRunners": 2,
            "numberOfActiveRunners": 1, "isMarketDataDelayed": True, "betDelay": 0, "runners": runners}


def raw_catalogue(mid="1.100", start="2026-10-01T14:30:00.000Z"):
    return {"marketId": mid, "marketName": "7f Hcap", "marketStartTime": start,
            "event": {"id": "999", "name": "Kempton 1st Oct", "venue": "Kempton", "countryCode": "GB"},
            "description": {"marketType": "WIN", "raceType": "Flat"},
            "runners": [{"selectionId": 11, "runnerName": "Real Bullet", "sortPriority": 1,
                         "metadata": {"CLOTH_NUMBER": "3", "STALL_DRAW": "7", "JOCKEY_NAME": "A Rider",
                                      "TRAINER_NAME": "B Trainer", "FORECASTPRICE_NUMERATOR": "5",
                                      "FORECASTPRICE_DENOMINATOR": "2"}},
                        {"selectionId": 12, "runnerName": "Pit Boss", "sortPriority": 2}]}


# ---------------------------------------------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------------------------------------------

def test_book_rows_keep_three_levels_and_blank_the_rest():
    rows = br.book_rows([raw_book()], datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc), "trader")
    assert len(rows) == 2
    a, b = rows
    assert (a["back1"], a["back1_size"], a["back2"], a["back3"]) == (3.4, 20.0, 3.3, None)
    assert (a["lay1"], a["lay2"]) == (3.6, None)
    assert a["sp_near"] is None and a["sp_far"] is None          # Betfair's "NaN" / "Infinity" projections
    assert a["market_total_matched"] == 5000.0 and a["is_delayed"] == 1 and a["source"] == "trader"
    assert a["polled_utc"] == "2026-10-01T09:00:00Z"
    assert (b["runner_status"], b["adjustment_factor"], b["back1"]) == ("REMOVED", 12.5, None)


def test_catalogue_rows_carry_the_card_metadata():
    rows = br.catalogue_rows([raw_catalogue()], datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc))
    assert [r["runner_name"] for r in rows] == ["Real Bullet", "Pit Boss"]
    r = rows[0]
    assert (r["venue"], r["country"], r["market_type"], r["race_type"]) == ("Kempton", "GB", "WIN", "Flat")
    assert (r["cloth_number"], r["stall_draw"], r["jockey"], r["trainer"]) == ("3", "7", "A Rider", "B Trainer")
    assert (r["forecast_num"], r["forecast_den"]) == ("5", "2")
    assert rows[1]["jockey"] is None                              # no metadata asked for, or none sent


def test_the_places_paid_and_the_each_way_divisor_are_recorded():
    book = raw_book(mid="1.300")
    book["numberOfWinners"] = 3
    rows = br.book_rows([book, raw_book()], datetime(2026, 10, 7, 9, 0, tzinfo=timezone.utc), "recorder")
    assert [r["number_of_winners"] for r in rows] == [3, 3, None, None]       # a book that does not say: blank
    ew = raw_catalogue(mid="1.301")
    ew["marketName"], ew["description"] = "Each Way", {"marketType": "EACH_WAY", "eachWayDivisor": 5.0}
    rows = br.catalogue_rows([ew, raw_catalogue()], datetime(2026, 10, 7, 8, 0, tzinfo=timezone.utc))
    assert [r["each_way_divisor"] for r in rows] == [5.0, 5.0, None, None]
    assert "number_of_winners" in br.BOOK_FIELDS and "each_way_divisor" in br.MARKET_FIELDS


# ---------------------------------------------------------------------------------------------------------------
# The day's files
# ---------------------------------------------------------------------------------------------------------------

def test_the_day_files_append_with_one_header_and_upload_gzipped(tmp_path):
    s3 = FakeS3()
    rec = br.DayRecorder(DAY, root=tmp_path, s3=s3, background=False)
    assert rec.record_catalogue([raw_catalogue()]) == 2
    assert rec.record_catalogue([raw_catalogue()]) == 0           # a market is catalogued once
    rec.record_books([raw_book()], "trader")
    rec.record_books([raw_book()], "trader")
    with rec.books_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 and rows[0]["market_id"] == "1.100"
    assert rec.books_path.read_text().count("polled_utc") == 1
    rec.maybe_upload(force=True)
    key = ("ashcroft", f"betfair_live/{DAY}/books.csv.gz")
    assert gzip.decompress(s3.objects[key]) == rec.books_path.read_bytes()
    assert ("ashcroft", f"betfair_live/{DAY}/markets.csv.gz") in s3.objects


def test_a_day_file_begun_by_an_older_recorder_keeps_its_own_columns(tmp_path):
    old_fields = [f for f in br.BOOK_FIELDS if f != "number_of_winners"]
    rec = br.DayRecorder(DAY, root=tmp_path, background=False)
    with rec.books_path.open("w", newline="") as f:                   # the morning's file, before the change
        csv.DictWriter(f, fieldnames=old_fields, lineterminator="\n").writeheader()
    book = raw_book()
    book["numberOfWinners"] = 1
    rec.record_books([book], "recorder")                              # a restart on the new code, the same day
    with rec.books_path.open() as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == old_fields                                # one header, the rows under it in line
    assert (rows[0]["market_id"], rows[0]["back1"], rows[0]["sp_lay_taken"]) == ("1.100", "3.4", "")
    fresh = br.DayRecorder(DAY + timedelta(days=1), root=tmp_path, background=False)
    fresh.record_books([book], "recorder")
    with fresh.books_path.open() as f:
        assert next(csv.DictReader(f))["number_of_winners"] == "1"    # a new day's file carries the new column


def test_the_nightly_load_adds_new_columns_to_an_older_table_and_reads_older_files(tmp_path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    old_cols = [c for c in br.LIVE_MARKET_FIELDS if c != "each_way_divisor"]
    conn.execute(f"CREATE TABLE betfair_live_markets ({', '.join(old_cols)}, loaded_at TEXT, "
                 "PRIMARY KEY (market_id, selection_id))")
    conn.commit()
    br.ensure_tables(conn)
    assert "each_way_divisor" in {r[1] for r in conn.execute("PRAGMA table_info(betfair_live_markets)")}
    br.ensure_tables(conn)                                            # twice is harmless
    conn.close()
    books, markets = _books_frame()
    d = tmp_path / "live" / f"{DAY}"
    d.mkdir(parents=True)
    books.drop(columns=["number_of_winners"]).to_csv(d / "books.csv", index=False)     # files of before 7 Oct
    markets.drop(columns=["each_way_divisor"]).to_csv(d / "markets.csv", index=False)
    out = br.load_db(str(db), days=1, day_to=DAY, s3=FakeS3(), root=tmp_path / "live")
    assert out[0]["markets"] == 2 and out[0]["marks"] > 0
    ew = raw_catalogue(mid="1.301")
    ew["description"] = {"marketType": "EACH_WAY", "eachWayDivisor": 4.0}
    pd.concat([markets, pd.DataFrame(br.catalogue_rows([ew], datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)))]
              ).to_csv(d / "markets.csv", index=False)                  # a file of the new recorder's
    br.load_db(str(db), days=1, day_to=DAY, s3=FakeS3(), root=tmp_path / "live")
    conn = sqlite3.connect(db)
    got = conn.execute("SELECT market_type, each_way_divisor FROM betfair_live_markets WHERE market_id = '1.301'")
    assert {tuple(r) for r in got} == {("EACH_WAY", "4.0")}
    conn.close()


def test_a_new_recorder_takes_up_the_markets_already_on_file(tmp_path):
    br.DayRecorder(DAY, root=tmp_path, s3=FakeS3()).record_catalogue([raw_catalogue()])
    again = br.DayRecorder(DAY, root=tmp_path, s3=FakeS3())
    assert again.record_catalogue([raw_catalogue(), raw_catalogue("1.200")]) == 2   # only the new market


def test_a_market_catalogued_without_metadata_is_catalogued_again_with_it(tmp_path):
    plain = raw_catalogue()
    for r in plain["runners"]:
        r.pop("metadata", None)                                   # the trader's catalogue call asks for none
    rec = br.DayRecorder(DAY, root=tmp_path, s3=FakeS3())
    assert rec.record_catalogue([plain]) == 2
    assert rec.record_catalogue([raw_catalogue()]) == 2           # the recorder's, with jockey and draw
    assert rec.record_catalogue([raw_catalogue()]) == 0
    again = br.DayRecorder(DAY, root=tmp_path, s3=FakeS3())       # and a restart knows it has them
    assert again.record_catalogue([raw_catalogue()]) == 0
    markets = pd.read_csv(rec.markets_path, dtype=str)
    last = markets.drop_duplicates(["market_id", "selection_id"], keep="last")
    assert last.loc[last["selection_id"] == "11", "jockey"].item() == "A Rider"


def test_uploads_wait_their_interval_and_a_failed_upload_is_counted_not_raised(tmp_path):
    clock = [0.0]
    s3 = FakeS3(fail=True)
    rec = br.DayRecorder(DAY, root=tmp_path, s3=s3, upload_every=900, clock=lambda: clock[0], background=False)
    rec.record_books([raw_book()], "trader")                      # first write: an upload is tried, and fails
    assert rec.errors == 1
    clock[0] = 100.0
    assert rec.maybe_upload() is False                            # not due yet
    s3.fail = False
    clock[0] = 1000.0
    assert rec.maybe_upload() is True
    assert ("ashcroft", f"betfair_live/{DAY}/books.csv.gz") in s3.objects


def test_a_background_upload_is_finished_before_a_forced_one(tmp_path):
    s3 = FakeS3()
    rec = br.DayRecorder(DAY, root=tmp_path, s3=s3, background=True)
    rec.record_books([raw_book()], "trader")
    assert rec.maybe_upload(force=True) is True
    assert ("ashcroft", f"betfair_live/{DAY}/books.csv.gz") in s3.objects


def test_bad_replies_are_logged_not_raised(tmp_path):
    rec = br.DayRecorder(DAY, root=tmp_path, s3=FakeS3(), background=False)
    assert rec.record_books([{"runners": "not a list"}], "trader") == 0
    assert rec.errors == 1


# ---------------------------------------------------------------------------------------------------------------
# The hook in the trader's reads
# ---------------------------------------------------------------------------------------------------------------

class FakeClient:
    def __init__(self):
        self.calls = []

    def login(self):
        pass

    def _api_call(self, method, params):
        self.calls.append(method)
        if method == "listMarketCatalogue":
            return [raw_catalogue()]
        return [raw_book(mid) for mid in params["marketIds"]]


class BrokenRecorder:
    def record_books(self, *a, **k):
        raise RuntimeError("disk full")

    def record_catalogue(self, *a, **k):
        raise RuntimeError("disk full")


def test_the_trader_keeps_every_book_it_reads(tmp_path):
    rec = br.DayRecorder(DAY, root=tmp_path, s3=FakeS3(), background=False)
    data = BetfairData(client=FakeClient(), recorder=rec)
    data.markets(datetime(2026, 10, 1, tzinfo=timezone.utc), datetime(2026, 10, 2, tzinfo=timezone.utc))
    books = data.books(["1.100", "1.200"])
    assert set(books) == {"1.100", "1.200"}
    with rec.books_path.open() as f:
        rows = list(csv.DictReader(f))
    assert {r["market_id"] for r in rows} == {"1.100", "1.200"} and {r["source"] for r in rows} == {"trader"}
    assert rec.markets_path.exists()


def test_a_broken_recorder_never_stops_the_trader():
    data = BetfairData(client=FakeClient(), recorder=BrokenRecorder())
    markets = data.markets(datetime(2026, 10, 1, tzinfo=timezone.utc), datetime(2026, 10, 2, tzinfo=timezone.utc))
    books = data.books(["1.100"])
    assert [m.market_id for m in markets] == ["1.100"] and books["1.100"].runners[11].back[0] == (3.4, 20.0)


def test_the_trader_records_only_on_the_uk_runner(tmp_path, monkeypatch):
    import auto_trade
    monkeypatch.setattr(br, "ROOT", tmp_path)
    assert auto_trade.book_recorder(DAY, env={}) is None
    assert auto_trade.book_recorder(DAY, env={"BETFAIR_RECORD": "0"}) is None
    rec = auto_trade.book_recorder(DAY, env={"BETFAIR_RECORD": "1"})
    assert isinstance(rec, br.DayRecorder) and rec.dir == tmp_path / f"{DAY}"
    assert rec.books_path.name == "books_trader.csv"                   # its own files: the recorder runs beside it


def test_the_trader_and_the_recorder_write_side_by_side_and_the_nightly_load_reads_both(tmp_path):
    s3 = FakeS3()
    trader = br.DayRecorder(DAY, root=tmp_path / "server", s3=s3, background=False, tag="trader")
    recorder = br.DayRecorder(DAY, root=tmp_path / "server", s3=s3, background=False)
    morning = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)                    # 10:00 UK, before the offs
    trader.record_catalogue([raw_catalogue()])
    trader.record_books([raw_book()], "trader", polled_at=morning)
    recorder.record_catalogue([raw_catalogue(), raw_catalogue("1.200", "2026-10-01T15:00:00.000Z")])
    recorder.record_books([raw_book(), raw_book("1.200")], "recorder", polled_at=morning)
    trader.maybe_upload(force=True)
    recorder.maybe_upload(force=True)
    keys = {k for _, k in s3.objects}
    assert {f"betfair_live/{DAY}/{n}.csv.gz" for n in ("books", "markets", "books_trader", "markets_trader")} <= keys
    assert br.market_ids_on_file(DAY, root=tmp_path / "server") == ["1.100", "1.200"]
    books = br._read_day_files(DAY, "books", s3, root=tmp_path / "nothing-here")       # the S3 copies, both writers
    assert len(books) == 6 and set(books["source"]) == {"trader", "recorder"}
    out = br.load_db(str(tmp_path / "t.db"), days=1, day_to=DAY, s3=s3, root=tmp_path / "nothing-here")
    assert out[0]["markets"] == 4 and out[0]["marks"] > 0     # each runner once, whichever writer caught it


# ---------------------------------------------------------------------------------------------------------------
# The recording loop
# ---------------------------------------------------------------------------------------------------------------

class LoopData:
    """Two markets: one off at 12:00 UTC, one at 12:30; each goes in play at its off."""

    def __init__(self, clock):
        self.clock, self.reads = clock, []
        self.offs = {"1.1": datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc),
                     "1.2": datetime(2026, 10, 1, 12, 30, tzinfo=timezone.utc)}

    def _read(self, method, params):
        if method == "listMarketCatalogue":
            return [raw_catalogue(m, t.strftime("%Y-%m-%dT%H:%M:%S.000Z")) for m, t in self.offs.items()]
        now = self.clock[0]
        self.reads.append((now, tuple(params["marketIds"])))
        return [raw_book(m, inplay=now >= self.offs[m]) for m in params["marketIds"]]


def test_the_recorder_snapshots_far_then_near_and_leaves_a_market_at_the_off(tmp_path):
    clock = [datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)]
    data = LoopData(clock)
    rec = br.DayRecorder(DAY, root=tmp_path, s3=FakeS3(), background=False)

    def sleep(seconds):
        assert 5.0 <= seconds <= 60.0
        clock[0] += timedelta(seconds=seconds)

    out = br.record(DAY, datetime(2026, 10, 1, 13, 0, tzinfo=timezone.utc), data, rec, every_far=300, every_near=60,
                    near_minutes=60, now=lambda: clock[0], sleep=sleep)
    first = [t for t, ids in data.reads if "1.1" in ids]
    far = [t for t in first if t < datetime(2026, 10, 1, 11, 0, tzinfo=timezone.utc)]
    near = [t for t in first if t >= datetime(2026, 10, 1, 11, 0, tzinfo=timezone.utc)]
    gaps_far = {round((b - a).total_seconds()) for a, b in zip(far, far[1:])}
    gaps_near = {round((b - a).total_seconds()) for a, b in zip(near, near[1:])}
    assert max(gaps_far) <= 300 + 60 and min(gaps_far) >= 299
    assert max(gaps_near) <= 61
    assert max(first) >= data.offs["1.1"]                       # read at the off, seen in play, then left
    assert all(t <= data.offs["1.1"] + timedelta(minutes=2) for t in first)
    assert out["left"] == 0 and out["rows"] > 0


class SpData(LoopData):
    """LoopData whose books carry the BSP once asked with SP_TRADED in play: 1.1 from its second such read, 1.2
    never (a market without a BSP of its own)."""

    def __init__(self, clock):
        super().__init__(clock)
        self.sp_reads = []

    def _read(self, method, params):
        out = super()._read(method, params)
        if method == "listMarketBook" and "SP_TRADED" in params["priceProjection"]["priceData"]:
            for b in out:
                m = b["marketId"]
                self.sp_reads.append((self.clock[0], m))
                b["bspReconciled"] = True
                if m == "1.1" and sum(1 for _, x in self.sp_reads if x == m) >= 2:
                    for r in b["runners"]:
                        r["sp"] = {"actualSP": 3.75 if r["selectionId"] == 11 else "NaN"}
        return out


def test_the_bsp_is_read_as_each_market_reconciles_it_at_the_off(tmp_path):
    clock = [datetime(2026, 10, 1, 11, 50, tzinfo=timezone.utc)]
    data = SpData(clock)
    rec = br.DayRecorder(DAY, root=tmp_path, s3=FakeS3(), background=False)

    def sleep(seconds):
        clock[0] += timedelta(seconds=seconds)

    out = br.record(DAY, datetime(2026, 10, 1, 13, 0, tzinfo=timezone.utc), data, rec, every_far=300, every_near=60,
                    near_minutes=60, now=lambda: clock[0], sleep=sleep)
    reads = {m: [t for t, x in data.sp_reads if x == m] for m in ("1.1", "1.2")}
    assert len(reads["1.1"]) == 2 and len(reads["1.2"]) == br.BSP_TRIES        # found on the second; given up after 3
    assert all(data.offs[m] <= t <= data.offs[m] + timedelta(minutes=2) for m in reads for t in reads[m])
    with rec.books_path.open() as f:
        rows = [r for r in csv.DictReader(f) if r["source"] == "bsp"]
    assert {(r["market_id"], r["selection_id"], r["sp_actual"]) for r in rows} == {("1.1", "11", "3.75"),
                                                                                  ("1.1", "12", "")}
    assert out["left"] == 0


def test_the_read_at_the_off_asks_for_the_bsp_within_betfairs_weight_limit():
    """SP_TRADED alone brought no BSP (ledger bsp-at-off-1007): the read at the off asks SP_AVAILABLE too, a call's
    markets times their weight (EX_BEST_OFFERS 5, SP_AVAILABLE 3, SP_TRADED 7) never over 200."""
    weight = {"EX_BEST_OFFERS": 5, "SP_AVAILABLE": 3, "SP_TRADED": 7}
    asked = []

    class Data:
        def _read(self, method, params):
            asked.append(params)
            return [raw_book(m) for m in params["marketIds"]]

    got = br.read_books(Data(), [f"1.{i}" for i in range(40)], settled=True)
    assert len(got) == 40
    for params in asked:
        kinds = params["priceProjection"]["priceData"]
        assert {"SP_AVAILABLE", "SP_TRADED"} <= set(kinds)
        assert len(params["marketIds"]) * sum(weight[k] for k in kinds) <= 200
    asked.clear()
    br.read_books(Data(), ["1.1"])
    assert asked[0]["priceProjection"]["priceData"] == ["EX_BEST_OFFERS"]


def test_the_final_mark_takes_the_bsp_from_the_traders_read_after_the_off():
    """The trader's own reads as it settles its races (the live key) carry the BSP once Betfair sends it: the final
    mark takes it when neither the settled book nor the recorder's reads at the off have one."""
    books, markets = _books_frame()
    books.loc[books["source"] == "final", "sp_actual"] = None              # as Betfair sends a closed market
    trader = pd.DataFrame(br.book_rows([raw_book(inplay=True, runners=[
        {"selectionId": 11, "status": "ACTIVE", "sp": {"actualSP": 4.2, "backStakeTaken": [{"price": 4.2,
                                                                                            "size": 50.0}]}}])],
        datetime(2026, 10, 1, 14, 35, 10, tzinfo=timezone.utc), "trader"))
    m = br.marks(pd.concat([books, trader], ignore_index=True), markets, DAY).set_index(["selection_id", "mark"])
    assert m.loc[(11, "final"), "bsp"] == pytest.approx(4.2)
    assert m.loc[(11, "last"), "polled_utc"] == "2026-10-01T14:29:00Z"   # the in-play read is no pre-off mark


def test_the_final_mark_takes_the_bsp_read_at_the_off_when_the_settled_book_has_none():
    books, markets = _books_frame()
    books.loc[books["source"] == "final", "sp_actual"] = None              # as Betfair sends a closed market
    at_off = pd.DataFrame(br.book_rows([raw_book(inplay=True, runners=[
        {"selectionId": 11, "status": "ACTIVE", "sp": {"actualSP": 4.1}}])],
        datetime(2026, 10, 1, 14, 30, 20, tzinfo=timezone.utc), "bsp"))
    m = br.marks(pd.concat([books, at_off], ignore_index=True), markets, DAY).set_index(["selection_id", "mark"])
    assert m.loc[(11, "final"), "bsp"] == pytest.approx(4.1)
    assert m.loc[(11, "last"), "polled_utc"] == "2026-10-01T14:29:00Z"   # the in-play read is no pre-off mark
    assert "final" not in m.loc[12].index


def test_the_final_pass_carries_the_bsp_read_at_the_off(tmp_path):
    rec = br.DayRecorder(DAY, root=tmp_path, s3=FakeS3(), background=False)
    rec.record_books([raw_book(inplay=True, runners=[{"selectionId": 11, "status": "ACTIVE", "sp": {"actualSP": 4.4}},
                                                     {"selectionId": 12, "status": "ACTIVE"}])],
                     source="bsp", polled_at=datetime(2026, 10, 1, 14, 30, 20, tzinfo=timezone.utc))

    class Closed:
        def _read(self, method, params):                       # a closed market's book: results, no BSP
            return [raw_book(status="CLOSED", runners=[{"selectionId": 11, "status": "WINNER", "sp": {}},
                                                       {"selectionId": 12, "status": "LOSER"}])]

    assert br.final(DAY, Closed(), rec, market_ids=["1.100"]) == 2
    with rec.books_path.open() as f:
        fin = {r["selection_id"]: r for r in csv.DictReader(f) if r["source"] == "final"}
    assert (fin["11"]["runner_status"], fin["11"]["sp_actual"]) == ("WINNER", "4.4")
    assert fin["12"]["sp_actual"] == ""                                 # none read at the off: left blank


def test_the_catalogue_is_read_in_pieces_under_betfairs_100_market_limit():
    calls = []

    class Data:
        def _read(self, method, params):
            calls.append(params)
            f = params["filter"]
            return [raw_catalogue(f"{f['marketTypeCodes'][0]}-{f['marketStartTime']['from']}")]

    start = datetime(2026, 9, 30, 23, 0, tzinfo=timezone.utc)
    cat = br.read_catalogue(Data(), start, start + timedelta(days=1))
    assert len(calls) == 12 and all(p["maxResults"] <= 100 for p in calls)
    assert all(len(p["filter"]["marketTypeCodes"]) == 1 for p in calls)
    assert "RUNNER_METADATA" in calls[0]["marketProjection"] and len({m["marketId"] for m in cat}) == 12


# ---------------------------------------------------------------------------------------------------------------
# The marks and the nightly load
# ---------------------------------------------------------------------------------------------------------------

def _books_frame():
    """One runner of a 14:30 UTC (15:30 UK) race polled every minute from 07:55 UTC, then a settled book."""
    rows = []
    t = datetime(2026, 10, 1, 7, 55, tzinfo=timezone.utc)
    off = datetime(2026, 10, 1, 14, 30, tzinfo=timezone.utc)
    while t < off:
        rows += br.book_rows([raw_book()], t, "recorder")
        t += timedelta(minutes=1)
    rows += br.book_rows([raw_book(inplay=True)], off, "recorder")
    settled = raw_book(status="CLOSED", runners=[{"selectionId": 11, "status": "WINNER", "sp": {"actualSP": 3.9}}])
    rows += br.book_rows([settled], off + timedelta(hours=8), "final")
    return pd.DataFrame(rows), pd.DataFrame(br.catalogue_rows([raw_catalogue()], off))


def test_marks_pick_the_book_nearest_each_mark_before_the_off():
    books, markets = _books_frame()
    m = br.marks(books, markets, DAY).set_index(["selection_id", "mark"])
    got = m.loc[11]
    # 08:00 UK is 07:00 UTC, before the first poll: no clock_08:00 mark
    assert "clock_08:00" not in got.index
    assert got.loc["clock_09:00", "polled_utc"] == "2026-10-01T08:00:00Z"
    assert got.loc["off_5", "minutes_to_off"] == pytest.approx(5.0)
    assert got.loc["off_1", "minutes_to_off"] == pytest.approx(1.0)
    assert got.loc["last", "polled_utc"] == "2026-10-01T14:29:00Z"
    assert got.loc["final", "bsp"] == 3.9 and got.loc["final", "runner_status"] == "WINNER"
    assert "final" not in m.loc[12].index                        # the removed runner was not in the settled book


def test_a_final_book_read_before_the_market_settled_is_not_the_final_mark():
    books, markets = _books_frame()
    books = books[books["source"] != "final"]
    early = pd.DataFrame(br.book_rows([raw_book(status="OPEN")], datetime(2026, 10, 1, 21, 0, tzinfo=timezone.utc),
                                      "final"))
    m = br.marks(pd.concat([books, early]), markets, DAY)
    assert "final" not in set(m["mark"])


def test_the_nightly_load_fills_the_three_tables(tmp_path):
    books, markets = _books_frame()
    d = tmp_path / "live" / f"{DAY}"
    d.mkdir(parents=True)
    books.to_csv(d / "books.csv", index=False)
    markets.to_csv(d / "markets.csv", index=False)
    s3 = FakeS3()
    ledger = "ts,mode,event,market_id,selection_id,runner,matched,avg_price,status\n" \
             "2026-10-01T09:01:00Z,live,back,1.100,11,Real Bullet,125.0,3.4,SUCCESS\n"
    s3.put_object("ashcroft", f"trading/live/{DAY}/ledger.csv", ledger.encode())
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE race_results (id INTEGER, race_date TEXT, race_time TEXT, track TEXT, horse_name TEXT)")
    conn.execute("INSERT INTO race_results VALUES (7, '2026-10-01', '3.30', 'Kempton', 'Real Bullet')")
    conn.commit(); conn.close()
    out = br.load_db(str(db), days=1, day_to=DAY, s3=s3, root=tmp_path / "live")
    assert out[0]["markets"] == 2 and out[0]["marks"] > 0 and out[0]["orders"] == 1
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT race_results_id FROM betfair_live_markets WHERE selection_id = 11").fetchone() == (7,)
    assert conn.execute("SELECT bsp FROM betfair_live_marks WHERE mark = 'final'").fetchone() == (3.9,)
    assert conn.execute("SELECT runner, matched FROM live_orders").fetchone() == ("Real Bullet", "125.0")
    conn.close()
    again = br.load_db(str(db), days=1, day_to=DAY, s3=s3, root=tmp_path / "live")   # a reload replaces, not adds
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM live_orders").fetchone() == (1,)
    assert conn.execute("SELECT COUNT(*) FROM betfair_live_marks").fetchone() == (again[0]["marks"],)
    conn.close()


def test_the_nightly_load_reads_the_s3_copy_when_the_server_has_none(tmp_path):
    books, markets = _books_frame()
    s3 = FakeS3()
    s3.put_object("ashcroft", f"betfair_live/{DAY}/books.csv.gz", gzip.compress(books.to_csv(index=False).encode()))
    s3.put_object("ashcroft", f"betfair_live/{DAY}/markets.csv.gz",
                  gzip.compress(markets.to_csv(index=False).encode()))
    out = br.load_db(str(tmp_path / "t.db"), days=2, day_to=DAY, s3=s3, root=tmp_path / "nothing-here")
    assert out[-1]["markets"] == 2 and out[-1]["marks"] > 0 and out[0]["markets"] == 0


# ---------------------------------------------------------------------------------------------------------------
# The price files' archive
# ---------------------------------------------------------------------------------------------------------------

PRICE_FILE = ("EVENT_ID,MENU_HINT,EVENT_NAME,EVENT_DT,SELECTION_ID,SELECTION_NAME,WIN_LOSE,BSP,PPWAP,MORNINGWAP,PPMAX,"
              "PPMIN,IPMAX,IPMIN,MORNINGTRADEDVOL,PPTRADEDVOL,IPTRADEDVOL\n"
              "1,GB / Kemp 1st Oct,7f Hcap,01-10-2026 15:30,11,Real Bullet,1,3.9,3.8,3.4,4.2,3.3,4,1.01,812.5,9000,300\n")


def test_the_price_files_go_to_s3_once_and_come_back_only_when_not_loaded_or_still_being_rewritten(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    name = bp.file_name("uk", "win", DAY)
    (raw / name).write_text(PRICE_FILE)
    s3 = FakeS3()
    assert bp.push_s3(raw, s3=s3, bucket="ashcroft") == {"uploaded": 1, "present": 0}
    assert bp.push_s3(raw, s3=s3, bucket="ashcroft") == {"uploaded": 0, "present": 1}
    db = tmp_path / "t.db"
    pulled = bp.pull_s3(tmp_path / "dl", str(db), s3=s3, bucket="ashcroft")
    assert [p.name for p in pulled] == [name]
    bp.load_files(pulled, str(db))
    # loaded: not pulled again, unless it is one of the last week's, which Betfair rewrites
    assert bp.pull_s3(tmp_path / "dl", str(db), s3=s3, bucket="ashcroft", refresh_days=0) == []
    assert [p.name for p in bp.pull_s3(tmp_path / "dl", str(db), s3=s3, bucket="ashcroft")] == [name]
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT morningwap, morning_vol, bsp FROM betfair_prices").fetchone() == (3.4, 812.5, 3.9)
    conn.close()


# ---------------------------------------------------------------------------------------------------------------
# Every file Betfair publishes, all markets (the owner's ask, 1 Oct 2026)
# ---------------------------------------------------------------------------------------------------------------

class _Reply:
    def __init__(self, code, content, headers=None):
        self.status_code, self.content, self.headers = code, content, dict(headers or {})
        self.text = content.decode("latin-1")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSite:
    """promo.betfair.com as the UK server sees it: the listing page, and each file (or the status it answers with)."""

    def __init__(self, files, status=None, listing=None):
        self.files, self.status, self.listing, self.gets = dict(files), dict(status or {}), listing, []

    def get(self, url, timeout=None):
        self.gets.append(url)
        if url == bp.BASE_URL:
            page = self.listing if self.listing is not None else "".join(
                f'<a href="{bp.BASE_URL}/{n}">{n}</a><br>\n' for n in sorted(self.files))
            return _Reply(self.status.get("", 200), page.encode())
        name = url.rsplit("/", 1)[-1]
        code = self.status.get(name, 200 if name in self.files else 404)
        if isinstance(code, list):                                   # a run of answers, one a request
            code = code.pop(0) if len(code) > 1 else code[0]
        body = self.files.get(name, b"") if code == 200 else b"<html>Betfair Restricted, Region: US</html>"
        return _Reply(code, body, {"Retry-After": "120"} if code == 429 else None)


def _csv(tag):
    return (PRICE_FILE + f"2,GB / Kemp 1st Oct,7f Hcap,01-10-2026 15:30,12,{tag},0,9,9,9,9,9,9,9,1,1,1\n").encode()


def test_every_listed_file_is_named_by_its_market_and_day():
    assert bp.parse_listed_name("dwbfpricesukwin14092026.csv") == ("ukwin", date(2026, 9, 14))
    assert bp.parse_listed_name("dwbfpricesireplace01012019.csv") == ("ireplace", date(2019, 1, 1))
    assert bp.parse_listed_name("dwbfgreyhoundwin30092026.csv") == ("greyhoundwin", date(2026, 9, 30))
    assert bp.parse_listed_name("DWBFPRICESAUSWIN05052021.csv") == ("auswin", date(2021, 5, 5))
    assert bp.parse_listed_name("dwbfpricesuk4tbp01102026.csv") == ("uk4tbp", date(2026, 10, 1))
    assert bp.parse_listed_name("dwbfpricesukwin31022026.csv") is None          # no such day
    assert bp.parse_listed_name("notes.csv") is None and bp.parse_listed_name("dwbfpricesukwin2026.csv") is None


def test_the_listing_is_read_from_its_links_or_its_text_and_a_refusal_is_an_error():
    names = ["dwbfpricesukwin30092026.csv", "dwbfgreyhoundwin01102026.csv", "dwbfpricesirewin01012019.csv"]
    got = bp.list_published(FakeSite({n: b"x" for n in names}))
    assert [f["name"] for f in got] == ["dwbfgreyhoundwin01102026.csv", "dwbfpricesukwin30092026.csv",
                                        "dwbfpricesirewin01012019.csv"]                    # newest first
    assert got[0]["url"] == f"{bp.BASE_URL}/dwbfgreyhoundwin01102026.csv" and got[0]["market"] == "greyhoundwin"
    relative = FakeSite({}, listing='<a href="dwbfpricesukplace30092026.csv">x</a>')
    assert bp.list_published(relative)[0]["url"] == f"{bp.BASE_URL}/dwbfpricesukplace30092026.csv"
    text = FakeSite({}, listing="dwbfpricesukwin30092026.csv dwbfpricesukplace30092026.csv")
    assert len(bp.list_published(text)) == 2                                               # names without links
    with pytest.raises(RuntimeError, match="refused"):
        bp.list_published(FakeSite({}, status={"": 403}, listing="Betfair Restricted, Region: US"))


def _site_and_archive():
    old_uk, old_aus = "dwbfpricesukwin01012019.csv", "dwbfpricesauswin01012019.csv"
    new_uk, new_place, new_dogs = "dwbfpricesukwin01102026.csv", "dwbfpricesukplace01102026.csv", "dwbfgreyhoundwin01102026.csv"
    last_week, gap = "dwbfpricesirewin28092026.csv", "dwbfpricesukwin15062024.csv"
    site = FakeSite({old_uk: _csv("a"), old_aus: _csv("b"), new_uk: _csv("rewritten"), new_place: _csv("c"),
                     new_dogs: _csv("d"), last_week: _csv("e"), gap: _csv("f")})
    s3 = FakeS3()
    s3.put_object(Bucket="ashcroft", Key=f"{bp.S3_PREFIX}/{old_uk}", Body=_csv("a"))       # archived already
    s3.put_object(Bucket="ashcroft", Key=f"{bp.S3_PREFIX}/{new_uk}", Body=_csv("first"))   # since rewritten by Betfair
    return site, s3


def test_every_market_is_archived_slowly_the_last_week_first_and_rewritten_files_again():
    site, s3 = _site_and_archive()
    pauses = []
    out = bp.archive_published(s3=s3, bucket="ashcroft", session=site, sleep=pauses.append)
    fetched = [u.rsplit("/", 1)[-1] for u in site.gets[1:]]
    # the last week first (every market), then the UK and Irish racing files, then the rest, each newest first;
    # the 2019 UK file the archive holds is not fetched again
    assert fetched == ["dwbfgreyhoundwin01102026.csv", "dwbfpricesukplace01102026.csv", "dwbfpricesukwin01102026.csv",
                       "dwbfpricesirewin28092026.csv", "dwbfpricesukwin15062024.csv", "dwbfpricesauswin01012019.csv"]
    assert out["listed"] == 7 and out["markets"] == 5 and out["stored"] == 6 and out["unchanged"] == 0
    assert s3.objects[("ashcroft", f"{bp.S3_PREFIX}/dwbfpricesukwin01102026.csv")] == _csv("rewritten")
    assert pauses == [1.5] * 5                                       # one file at a time, a pause between each
    again = bp.archive_published(s3=s3, bucket="ashcroft", session=FakeSite(site.files), sleep=lambda _: None)
    assert again["to_fetch"] == 4 and again["stored"] == 0 and again["unchanged"] == 4    # only the last week


def test_the_other_markets_from_2018_come_before_anything_older():
    """The nightly load reads UK and Irish racing from 2018, and the owner's research of 2 Oct wants the other
    markets: after the last week, UK/IE from 2018, then the other markets from 2018, then every older file."""
    names = ["dwbfpricesukwin01012012.csv", "dwbfpricesauswin01012012.csv", "dwbfpricesauswin01012020.csv",
             "dwbfpricesusaplace01012019.csv", "dwbfpricesireplace01012018.csv", "dwbfpricesukwin31122017.csv",
             "dwbfpricesukwin01102026.csv"]
    site = FakeSite({n: _csv(n) for n in names})
    bp.archive_published(s3=FakeS3(), bucket="ashcroft", session=site, sleep=lambda _: None)
    fetched = [u.rsplit("/", 1)[-1] for u in site.gets[1:]]
    assert fetched == ["dwbfpricesukwin01102026.csv",                                      # the last week
                       "dwbfpricesireplace01012018.csv",                                   # UK/IE from 2018
                       "dwbfpricesauswin01012020.csv", "dwbfpricesusaplace01012019.csv",   # the rest from 2018
                       "dwbfpricesukwin31122017.csv", "dwbfpricesauswin01012012.csv",      # then the older files
                       "dwbfpricesukwin01012012.csv"]


def test_a_backfill_of_old_days_fetches_only_what_the_archive_lacks():
    site, s3 = _site_and_archive()
    # the last week is Betfair's last week, not the backfill's: the 2019 UK file the archive holds stays as it is
    out = bp.archive_published(s3=s3, bucket="ashcroft", session=site, date_to=date(2019, 1, 1), sleep=lambda _: None)
    assert out["listed"] == 2 and out["to_fetch"] == 1 and out["stored"] == 1
    assert [u.rsplit("/", 1)[-1] for u in site.gets[1:]] == ["dwbfpricesauswin01012019.csv"]
    only = bp.archive_published(s3=s3, bucket="ashcroft", session=FakeSite(site.files), markets=["ukplace"],
                                sleep=lambda _: None)
    assert only["listed"] == 1 and only["markets"] == 1 and only["unchanged"] == 0 and only["stored"] == 1


def test_the_archive_stops_at_its_time_and_the_next_run_takes_up_the_rest():
    site, s3 = _site_and_archive()
    t = {"now": 0.0}

    def clock():
        t["now"] += 60.0                                             # a minute a file
        return t["now"]
    out = bp.archive_published(s3=s3, bucket="ashcroft", session=site, max_minutes=3, clock=clock,
                               sleep=lambda _: None)
    assert out["stored"] == 2 and out["left"] == 4
    rest = bp.archive_published(s3=s3, bucket="ashcroft", session=FakeSite(site.files), sleep=lambda _: None)
    # the rewritten UK file, the Irish one, the 2024 and the 2019 files; the two stored before are unchanged
    assert rest["stored"] == 4 and rest["unchanged"] == 2 and rest["left"] == 0
    assert len([k for b, k in s3.objects if b == "ashcroft"]) == 7


def test_a_block_page_is_never_archived_and_a_refusal_stops_the_run():
    site, s3 = _site_and_archive()
    site.files["dwbfgreyhoundwin01102026.csv"] = b"<html><body>Betfair Restricted</body></html>"
    site.status["dwbfpricesukwin15062024.csv"] = 403
    out = bp.archive_published(s3=s3, bucket="ashcroft", session=site, sleep=lambda _: None)
    assert ("ashcroft", f"{bp.S3_PREFIX}/dwbfgreyhoundwin01102026.csv") not in s3.objects
    assert out["failed"] == 2 and out["stored"] == 3 and out["left"] == 1      # the 2019 Australian file waits
    assert ("ashcroft", f"{bp.S3_PREFIX}/dwbfpricesauswin01012019.csv") not in s3.objects


def test_asked_to_slow_down_the_archive_waits_as_long_as_asked_and_stops_if_asked_again_and_again():
    site, s3 = _site_and_archive()
    site.status["dwbfgreyhoundwin01102026.csv"] = [429, 200]                # once: wait, then the file
    site.status["dwbfpricesukwin15062024.csv"] = [429]                      # every time: stop for the night
    waits = []
    out = bp.archive_published(s3=s3, bucket="ashcroft", session=site, sleep=waits.append)
    assert waits[0] == 120.0 and waits.count(120.0) == 3                    # as long as asked (Retry-After)
    assert ("ashcroft", f"{bp.S3_PREFIX}/dwbfgreyhoundwin01102026.csv") in s3.objects
    assert out["stored"] == 4 and out["failed"] == 1 and out["left"] == 1   # the 2019 Australian file waits


class _DownS3(FakeS3):
    def put_object(self, Bucket, Key, Body):
        raise RuntimeError("S3 is down")


def test_s3_refusing_five_files_running_stops_the_run():
    site, _ = _site_and_archive()
    names = [f"dwbfpricesukwin{d:02d}092026.csv" for d in range(1, 11)]
    site.files.update({n: _csv(n) for n in names})
    out = bp.archive_published(s3=_DownS3(), bucket="ashcroft", session=site, sleep=lambda _: None)
    assert out["failed"] == 5 and out["stored"] == 0 and out["left"] == out["to_fetch"] - 5


def test_the_nightly_load_takes_the_last_week_first_then_the_newest_within_its_cap(tmp_path):
    s3 = FakeS3()
    days = [date(2017, 12, 31), date(2018, 1, 1), date(2024, 6, 15), date(2026, 9, 25), date(2026, 10, 1)]
    for i, d in enumerate(days):                                     # each day its own race
        body = PRICE_FILE.replace("\n1,GB", f"\n{100 + i},GB").replace("01-10-2026", d.strftime("%d-%m-%Y"))
        s3.put_object(Bucket="ashcroft", Key=f"{bp.S3_PREFIX}/{bp.file_name('uk', 'win', d)}", Body=body.encode())
    s3.put_object(Bucket="ashcroft", Key=f"{bp.S3_PREFIX}/dwbfgreyhoundwin01102026.csv", Body=b"x")   # stays raw
    db = tmp_path / "t.db"
    first = bp.pull_s3(tmp_path / "dl", str(db), s3=s3, bucket="ashcroft", load_from=date(2018, 1, 1), max_files=3)
    assert sorted(bp.parse_file_name(p.name)[2] for p in first) == [date(2024, 6, 15), date(2026, 9, 25),
                                                                    date(2026, 10, 1)]
    bp.load_files(first, str(db))
    second = bp.pull_s3(tmp_path / "dl", str(db), s3=s3, bucket="ashcroft", load_from=date(2018, 1, 1), max_files=3)
    # the last week again (Betfair rewrites it), then what is left from 2018 on; 2017 and the greyhounds never
    assert sorted(bp.parse_file_name(p.name)[2] for p in second) == [date(2018, 1, 1), date(2026, 9, 25),
                                                                     date(2026, 10, 1)]


def test_a_day_without_racing_or_a_file_that_cannot_be_read_is_taken_once_and_old_names_read_in_latin_1(tmp_path):
    s3 = FakeS3()
    header = PRICE_FILE.split("\n", 1)[0] + "\n"
    empty, broken, latin = (bp.file_name("uk", "win", date(2020, 4, 1)), bp.file_name("ire", "win", date(2019, 5, 1)),
                            bp.file_name("uk", "place", date(2019, 5, 1)))
    s3.put_object(Bucket="ashcroft", Key=f"{bp.S3_PREFIX}/{empty}", Body=header.encode())          # no racing that day
    s3.put_object(Bucket="ashcroft", Key=f"{bp.S3_PREFIX}/{broken}", Body=b"EVENT_ID,MENU_HINT\n1,x\n")
    body = PRICE_FILE.replace("Real Bullet", "Sí Señor").replace("01-10-2026", "01-05-2019").replace("1st Oct", "1st May")
    s3.put_object(Bucket="ashcroft", Key=f"{bp.S3_PREFIX}/{latin}", Body=body.encode("latin-1"))
    recent = bp.file_name("uk", "win", DAY)
    s3.put_object(Bucket="ashcroft", Key=f"{bp.S3_PREFIX}/{recent}", Body=PRICE_FILE.encode())
    db = tmp_path / "t.db"
    first = bp.pull_s3(tmp_path / "dl", str(db), s3=s3, bucket="ashcroft")
    assert sorted(p.name for p in first) == sorted([empty, broken, latin, recent])
    assert bp.load_files(first, str(db)) == {"files": 3, "rows": 2}
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT selection_name FROM betfair_prices WHERE market_type = 'place'").fetchone() == ("Sí Señor",)
    noted = dict(conn.execute("SELECT source_file, status FROM betfair_prices_files").fetchall())
    conn.close()
    assert noted[empty] == "loaded" and noted[latin] == "loaded" and noted[broken].startswith("skipped: ")
    # none of the three old files is pulled again; only the last week, which Betfair rewrites
    assert [p.name for p in bp.pull_s3(tmp_path / "dl", str(db), s3=s3, bucket="ashcroft")] == [recent]


def test_the_census_counts_every_market_type_on_the_day_once():
    """The owner asked about match bets (AvB), which Betfair's price files do not hold: the census reads every market
    on a day's racing, any type, read-only, so the recording can be told which to add."""
    def market(mid, kind, name, venue="Kempton"):
        return {"marketId": mid, "marketName": name, "description": {"marketType": kind}, "event": {"venue": venue}}

    class Data:
        def __init__(self):
            self.calls = []

        def _read(self, method, params):
            self.calls.append((method, params))
            if len(self.calls) == 1:
                return [market("1.1", "WIN", "7f Hcap"), market("1.2", "PLACE", "To Be Placed"),
                        market("1.3", "MATCH_BET", "Alpha v Bravo")]
            if len(self.calls) == 2:                                     # the next window: one new, one seen
                return [market("1.3", "MATCH_BET", "Alpha v Bravo"), market("1.4", "MATCH_BET", "Charlie v Delta")]
            return []

    data = Data()
    got = br.census(data, DAY)
    assert got["MATCH_BET"] == {"markets": 2, "e.g.": ["Kempton: Alpha v Bravo", "Kempton: Charlie v Delta"]}
    assert got["WIN"]["markets"] == 1 and got["PLACE"]["markets"] == 1 and "_note" not in got
    assert {m for m, _ in data.calls} == {"listMarketCatalogue"}                         # reads only
    assert all("marketTypeCodes" not in p["filter"] for _, p in data.calls)              # every type
    assert len(data.calls) == 12                                                       # 2-hour windows of the day


def test_the_command_line_archives_with_its_markets_dates_pause_and_time(monkeypatch, capsys):
    site, s3 = _site_and_archive()
    monkeypatch.setattr(bp, "_capture", lambda: (s3, "ashcroft"))
    monkeypatch.setattr(bp, "make_session", lambda proxy=None: site)
    real, seen, pauses = bp.archive_published, {}, []

    def archive(**kw):
        seen.update(kw)
        return real(**kw, sleep=pauses.append)
    monkeypatch.setattr(bp, "archive_published", archive)
    assert bp.main(["--archive", "--market", "ukwin, ukplace", "--from", "2024-01-01", "--pause", "0.5",
                    "--max-minutes", "30", "--refresh-days", "0"]) == 0
    assert seen["markets"] == ["ukwin", " ukplace"] and seen["date_from"] == date(2024, 1, 1)
    assert seen["date_to"] is None and seen["max_minutes"] == 30 and seen["refresh_days"] == 0
    out = capsys.readouterr().out
    assert "'listed': 3" in out and "'stored': 2" in out                 # 2026 ukwin (held), ukplace, the 2024 ukwin
    assert [u.rsplit("/", 1)[-1] for u in site.gets[1:]] == ["dwbfpricesukplace01102026.csv",
                                                             "dwbfpricesukwin15062024.csv"]
    assert pauses == [0.5]


def test_a_load_is_matched_run_by_run_not_across_the_years_between():
    runs = bp.day_runs([date(2026, 10, 1), date(2026, 9, 29), date(2026, 9, 30), date(2019, 3, 2), date(2019, 3, 1)])
    assert runs == [(date(2019, 3, 1), date(2019, 3, 2)), (date(2026, 9, 29), date(2026, 10, 1))]
    assert bp.day_runs([]) == []


def test_tomorrows_markets_recorded_the_evening_before_go_to_their_own_files(tmp_path, monkeypatch):
    """The owner's ask (4 Oct): tomorrow's markets recorded in the evening, for minutes from now, kept apart."""
    seen = {}

    class FakeData:
        def login(self):
            pass

    def fake_record(day, until, data, rec, *args, wait_for_markets=False):
        seen.update(day=day, until=until, books=rec.books_path.name, markets=rec.markets_path.name,
                    wait=wait_for_markets)
        return {"markets": 0}

    monkeypatch.setattr("trading.exchange.BetfairData", FakeData)
    monkeypatch.setattr(br, "ROOT", tmp_path)
    monkeypatch.setattr(br, "record", fake_record)
    before = datetime.now(timezone.utc)
    assert br.main(["--record", "--date", "2026-10-05", "--tag", "evening", "--for-minutes", "45",
                    "--market-types", "WIN"]) == 0
    assert seen["day"] == date(2026, 10, 5)
    assert (seen["books"], seen["markets"]) == ("books_evening.csv", "markets_evening.csv")
    assert timedelta(minutes=44) < seen["until"] - before < timedelta(minutes=46)
    assert seen["wait"] is False                    # a record for minutes from now ends when nothing is listed
    # the nightly load reads only the day's own writers, so an evening record never moves the race-day marks
    assert "evening" not in br.DAY_FILE_TAGS


class EveningData:
    """Tomorrow's two markets, listed only from 15:20 UTC the evening before; neither goes in play this evening."""

    def __init__(self, clock):
        self.clock, self.reads, self.catalogue_reads = clock, [], 0
        self.listed = datetime(2026, 10, 4, 15, 20, tzinfo=timezone.utc)

    def _read(self, method, params):
        if method == "listMarketCatalogue":
            self.catalogue_reads += 1
            if self.clock[0] < self.listed:
                return []
            return [raw_catalogue(m, f"2026-10-05T{h}:00:00.000Z") for m, h in (("1.1", "13"), ("1.2", "14"))]
        self.reads.append((self.clock[0], tuple(params["marketIds"])))
        return [raw_book(m) for m in params["marketIds"]]


def test_an_evening_record_waits_for_tomorrows_markets_to_be_listed(tmp_path):
    clock = [datetime(2026, 10, 4, 15, 0, tzinfo=timezone.utc)]

    def sleep(seconds):
        assert 5.0 <= seconds <= 60.0
        clock[0] += timedelta(seconds=seconds)

    until = datetime(2026, 10, 4, 18, 0, tzinfo=timezone.utc)
    rec = br.DayRecorder(date(2026, 10, 5), root=tmp_path, s3=FakeS3(), background=False, tag="evening")
    data = EveningData(clock)
    out = br.record(date(2026, 10, 5), until, data, rec, every_far=900, market_types=("WIN",),
                    now=lambda: clock[0], sleep=sleep, wait_for_markets=True)
    first = min(t for t, _ in data.reads)
    assert data.listed <= first <= data.listed + timedelta(minutes=31)      # found at the next catalogue read
    gaps = {round((b - a).total_seconds()) for (a, _), (b, _) in zip(data.reads, data.reads[1:])}
    assert min(gaps) >= 899 and max(gaps) <= 900 + 60                       # then every 15 minutes, to the end
    assert max(t for t, _ in data.reads) >= until - timedelta(minutes=16) and clock[0] >= until
    assert out["markets"] == 2 and out["left"] == 2 and out["rows"] > 0
    assert (tmp_path / "2026-10-05" / "books_evening.csv").exists()
    # the day's own record still ends when nothing is listed: a day without racing does not hold the runner
    clock[0] = datetime(2026, 10, 4, 15, 0, tzinfo=timezone.utc)
    quiet = EveningData(clock)
    rec2 = br.DayRecorder(date(2026, 10, 5), root=tmp_path / "day", s3=FakeS3(), background=False)
    out2 = br.record(date(2026, 10, 5), until, quiet, rec2, now=lambda: clock[0], sleep=sleep)
    assert out2["markets"] == 0 and quiet.reads == [] and clock[0] < until


def test_the_evening_before_is_read_in_uk_time_on_the_day_before():
    assert br.evening_window(date(2026, 10, 6), "17:00-21:30") == (
        datetime(2026, 10, 5, 16, 0, tzinfo=timezone.utc), datetime(2026, 10, 5, 20, 30, tzinfo=timezone.utc))
    assert br.evening_window(date(2026, 12, 2), "17:00-21:30") == (                 # GMT in winter
        datetime(2026, 12, 1, 17, 0, tzinfo=timezone.utc), datetime(2026, 12, 1, 21, 30, tzinfo=timezone.utc))
    for bad in ("21:30-17:00", "17:00", "5pm-9pm", "25:00-26:00"):
        with pytest.raises(ValueError):
            br.evening_window(date(2026, 10, 6), bad)


def test_an_evening_record_sleeps_to_its_start_before_logging_in(tmp_path, monkeypatch):
    """The trader's job starts it in the morning: it waits for 17:00 UK the evening before, logs in, and records
    tomorrow's markets to its own files until 21:30, waiting for them to be listed."""
    events, seen = [], {}

    class FakeData:
        def login(self):
            events.append("login")

    def fake_record(day, until, data, rec, *args, wait_for_markets=False):
        seen.update(day=day, until=until, books=rec.books_path.name, wait=wait_for_markets)
        return {"markets": 0}

    monkeypatch.setattr("trading.exchange.BetfairData", FakeData)
    monkeypatch.setattr(br, "ROOT", tmp_path)
    monkeypatch.setattr(br, "record", fake_record)
    monkeypatch.setattr(br.time, "sleep", lambda s: events.append(("sleep", s)))
    start, end = br.evening_window(date(2099, 7, 2), "17:00-21:30")
    assert br.main(["--record", "--date", "2099-07-02", "--tag", "evening", "--market-types", "WIN",
                    "--evening", "17:00-21:30"]) == 0
    assert events[0][0] == "sleep" and events[1] == "login"                  # the wait comes before the login
    assert abs(events[0][1] - (start - datetime.now(timezone.utc)).total_seconds()) < 60
    assert seen == {"day": date(2099, 7, 2), "until": end, "books": "books_evening.csv", "wait": True}
    # an evening already over records nothing and does not log in
    events.clear()
    assert br.main(["--record", "--date", "2020-07-02", "--tag", "evening", "--evening", "17:00-21:30"]) == 0
    assert events == []
    # and the evening is never written over the day's own files, nor settled
    for argv in (["--record", "--date", "2099-07-02", "--evening", "17:00-21:30"],
                 ["--record", "--final", "--date", "2099-07-02", "--tag", "evening", "--evening", "17:00-21:30"],
                 ["--record", "--date", "2099-07-02", "--tag", "evening", "--evening", "21:30-17:00"]):
        with pytest.raises(SystemExit):
            br.main(argv)


def test_a_record_asked_to_stop_ends_cleanly_and_copies_its_files_first(tmp_path, monkeypatch):
    """The recorder giving way to the trader (scripts/give_way.py), or its job cancelled: SIGTERM ends the record at
    once, between two writes, the day's files go to S3 on the way out, and the exit is clean. The handler is the
    record's only while it records; the trader's job then writes on to the same files."""
    import os
    import signal
    import time

    forced = []
    before = signal.getsignal(signal.SIGTERM)

    class FakeData:
        def login(self):
            pass

    def fake_record(day, until, data, rec, *args, **kw):
        rec.record_books([raw_book()], source="recorder")
        assert signal.getsignal(signal.SIGTERM) is not before, "no SIGTERM handler while recording"
        os.kill(os.getpid(), signal.SIGTERM)          # the watch's signal, mid-record
        time.sleep(5)                                 # cut short by it
        raise AssertionError("the record was not stopped")

    monkeypatch.setattr("trading.exchange.BetfairData", FakeData)
    monkeypatch.setattr(br, "ROOT", tmp_path)
    monkeypatch.setattr(br, "record", fake_record)
    monkeypatch.setattr(br.DayRecorder, "maybe_upload", lambda self, force=False: forced.append(force) or True)
    t0 = time.monotonic()
    with pytest.raises(SystemExit) as stop:
        br.main(["--record", "--date", "2026-10-07"])
    assert stop.value.code == 0 and time.monotonic() - t0 < 4
    assert forced and forced[-1] is True                     # the day's files copied on the way out
    assert signal.getsignal(signal.SIGTERM) is before        # and the process's own handler back
    rows = list(csv.DictReader((tmp_path / "2026-10-07" / "books.csv").open()))
    assert rows and {r["market_id"] for r in rows} == {"1.100"}
    # a record that ends by itself leaves no handler behind either
    monkeypatch.setattr(br, "record", lambda *a, **k: {"markets": 0})
    assert br.main(["--record", "--date", "2026-10-07"]) == 0
    assert signal.getsignal(signal.SIGTERM) is before
