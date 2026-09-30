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


def test_the_price_files_go_to_s3_once_and_come_back_only_when_not_loaded(tmp_path):
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
    assert bp.pull_s3(tmp_path / "dl", str(db), s3=s3, bucket="ashcroft") == []     # loaded: not pulled again
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT morningwap, morning_vol, bsp FROM betfair_prices").fetchone() == (3.4, 812.5, 3.9)
    conn.close()
