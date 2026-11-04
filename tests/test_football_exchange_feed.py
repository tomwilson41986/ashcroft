"""Every football match on Betfair, recorded and paper-traded: the catalogue, the cadence, the record, the paper book
(a fake Betfair; synthetic books only, never a reported figure)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from football import exchange_feed as ef
from football import paper_book as pb

T0 = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)


def raw_book(mid, runners, status="OPEN", inplay=False):
    """runners: {sid: (backs [(p, s)], lays [(p, s)], traded {p: v}, runner status)}."""
    out = []
    for sid, (backs, lays, traded, *st) in runners.items():
        out.append({"selectionId": sid, "status": st[0] if st else "ACTIVE",
                    "lastPriceTraded": backs[0][0] if backs else None, "totalMatched": sum(traded.values()),
                    "ex": {"availableToBack": [{"price": p, "size": s} for p, s in backs],
                           "availableToLay": [{"price": p, "size": s} for p, s in lays],
                           "tradedVolume": [{"price": p, "size": v} for p, v in traded.items()]}})
    return {"marketId": mid, "status": status, "inplay": inplay, "totalMatched": 1000.0, "runners": out}


def catalogue_entry(mid, start, names=("Arsenal", "Chelsea", "The Draw")):
    return {"marketId": mid, "marketName": "Match Odds", "marketStartTime": start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "description": {"marketType": "MATCH_ODDS"}, "event": {"id": "e1", "name": "Arsenal v Chelsea",
                                                                  "countryCode": "GB"},
            "competition": {"id": "10932509", "name": "English Premier League"},
            "runners": [{"selectionId": 11 + i, "runnerName": n, "sortPriority": i + 1} for i, n in enumerate(names)]}


def test_cadence_tightens_towards_the_off_and_batches_respect_the_weight():
    assert ef.cadence_seconds(600) == 1800 and ef.cadence_seconds(200) == 600 and ef.cadence_seconds(90) == 300
    assert ef.cadence_seconds(30) == 120 and ef.cadence_seconds(5) == 60
    assert ef.batch_size(3, False) == 40 and ef.batch_size(5, True) == 6 and ef.batch_size(10, True) == 3
    assert ef.weight_per_market(5, True) * ef.batch_size(5, True) <= 200


def test_the_catalogue_splits_windows_betfair_fills_and_keeps_each_market_once():
    calls = []

    def read(method, params):
        a = datetime.strptime(params["filter"]["marketStartTime"]["from"], "%Y-%m-%dT%H:%M:%SZ")
        b = datetime.strptime(params["filter"]["marketStartTime"]["to"], "%Y-%m-%dT%H:%M:%SZ")
        calls.append((a, b))
        hours = (b - a).total_seconds() / 3600
        n = params["maxResults"] if hours > 3 else 3                  # a full answer until the window is 3h or less
        return [{"marketId": f"1.{int(a.timestamp()) // 60 + i}"} for i in range(n)] + [{"marketId": "1.dup"}]

    got = ef.catalogue_all(read, T0, T0 + timedelta(hours=12), max_results=50)
    assert len(calls) > 1 and all(params is not None for params in calls)
    ids = [m["marketId"] for m in got]
    assert len(ids) == len(set(ids)) and "1.dup" in ids


def test_book_rows_keep_depth_and_the_money_traded_since_the_last_read():
    prev = {}
    r1 = ef.book_rows([raw_book("1.1", {11: ([(2.0, 50), (1.99, 20)], [(2.02, 30)], {2.0: 100.0})})], T0, 3, prev)
    assert r1[0]["back2"] == 1.99 and r1[0]["back3"] is None and r1[0]["traded_delta"] == ""
    r2 = ef.book_rows([raw_book("1.1", {11: ([(2.02, 50)], [(2.04, 30)], {2.0: 130.0, 2.02: 15.0})})],
                      T0 + timedelta(minutes=2), 3, prev)
    assert ef.parse_ladder(r2[0]["traded_delta"]) == {2.0: 30.0, 2.02: 15.0} and r2[0]["traded_total"] == 145.0


def test_walk_takes_the_depth_in_order():
    px, got = pb.walk([(2.0, 5.0), (1.98, 10.0)], 10.0)
    assert got == 10.0 and px == pytest.approx((5 * 2.0 + 5 * 1.98) / 10)
    px, got = pb.walk([(2.0, 3.0)], 10.0)
    assert got == 3.0 and px == 2.0


def make_book(start):
    book = pb.PaperBook(checkpoints=(60.0, 2.0), rest_checkpoints=(60.0,), stakes=(2.0, 10.0))
    book.add_market(catalogue_entry("1.1", start))
    return book


def test_a_resting_back_waits_behind_the_queue_then_fills_from_traded_money():
    start = T0 + timedelta(minutes=60)
    book = make_book(start)
    # T-60: home 2.00 / 2.04 (two-tick spread): the back posts at 2.02, inside, nobody ahead; the lay at 2.02 too
    book.on_book(raw_book("1.1", {11: ([(2.0, 50)], [(2.04, 40)], {2.0: 100.0}),
                                  12: ([(4.0, 20)], [(4.1, 20)], {4.0: 10.0}),
                                  13: ([(3.4, 20)], [(3.5, 20)], {3.4: 10.0})}), start, T0)
    rests = [ln for ln in book.lines if ln["kind"] == "rest" and ln["selection_id"] == 11]
    back = next(ln for ln in rests if ln["side"] == "back")
    assert back["post_price"] == 2.02 and back["queue_ahead"] == 0.0
    away_back = next(ln for ln in book.lines if ln["kind"] == "rest" and ln["selection_id"] == 12 and ln["side"] == "back")
    assert away_back["post_price"] == 4.1 and away_back["queue_ahead"] == 20.0           # one tick: joins the lay
    # GBP6 traded at 2.02: the home back is GBP6 filled
    book.on_book(raw_book("1.1", {11: ([(2.0, 50)], [(2.02, 10)], {2.0: 100.0, 2.02: 6.0}),
                                  12: ([(4.0, 20)], [(4.1, 20)], {4.0: 10.0, 4.1: 15.0}),
                                  13: ([(3.4, 20)], [(3.5, 20)], {3.4: 10.0})}), start, T0 + timedelta(minutes=5))
    assert back["filled"] == 6.0 and back["fill_how"] == "queue"
    assert away_back["filled"] == 0.0 and away_back["queue_ahead"] == 5.0                # 15 of 20 ahead traded
    # the away book trades through 4.1 (best back 4.2): filled outright
    book.on_book(raw_book("1.1", {11: ([(2.0, 50)], [(2.02, 10)], {2.0: 100.0, 2.02: 6.0}),
                                  12: ([(4.2, 20)], [(4.3, 20)], {4.0: 10.0, 4.1: 15.0}),
                                  13: ([(3.4, 20)], [(3.5, 20)], {3.4: 10.0})}), start, T0 + timedelta(minutes=10))
    assert away_back["filled"] == 10.0 and away_back["fill_how"] == "traded through"


def test_checkpoint_lines_are_sized_on_the_depth_valued_at_the_close_and_settled_on_betfair_result():
    start = T0 + timedelta(minutes=60)
    book = make_book(start)
    runners = {11: ([(2.0, 5), (1.98, 100)], [(2.04, 100)], {}), 12: ([(4.0, 50)], [(4.1, 50)], {}),
               13: ([(3.4, 50)], [(3.5, 50)], {})}
    book.on_book(raw_book("1.1", runners), start, T0)
    home_back = next(ln for ln in book.lines if ln["kind"] == "cross" and ln["selection_id"] == 11
                     and ln["side"] == "back" and ln["checkpoint_min"] == 60.0)
    assert home_back["px_2"] == 2.0 and home_back["px_10"] == pytest.approx(1.99) and home_back["got_10"] == 10.0
    assert home_back["spread_ticks"] == 2 and home_back["book_over"] == pytest.approx(1 / 2 + 1 / 4 + 1 / 3.4, abs=1e-4)
    # nothing at T-30 (no checkpoint); the close at T-1: home 1.90 / 1.92
    close = {11: ([(1.9, 50)], [(1.92, 50)], {}), 12: ([(4.4, 50)], [(4.5, 50)], {}),
             13: ([(3.5, 50)], [(3.6, 50)], {})}
    book.on_book(raw_book("1.1", close), start, start - timedelta(minutes=1))
    assert len([ln for ln in book.lines if ln["checkpoint_min"] == 2.0]) == 6                # T-1 inside T-2's window
    res = {11: ([], [], {}, "WINNER"), 12: ([], [], {}, "LOSER"), 13: ([], [], {}, "LOSER")}
    book.on_book(raw_book("1.1", res, status="CLOSED"), start, start + timedelta(hours=2))
    assert home_back["close_mid"] == pytest.approx(1.91) and home_back["won"] == 1
    assert home_back["green_close"] == pytest.approx(1.99 / 1.92 - 1, abs=1e-5)
    assert home_back["pnl_10"] == pytest.approx(10 * 0.99 * 0.98)
    home_lay = next(ln for ln in book.lines if ln["kind"] == "cross" and ln["selection_id"] == 11
                    and ln["side"] == "lay" and ln["checkpoint_min"] == 60.0)
    assert home_lay["pnl_10"] == pytest.approx(-10 * 1.04)
    files = book.files()
    assert [(d, n) for d, n, _ in files] == [("2026-11-07", "paper_lines.csv")]


def test_the_feed_records_schedules_and_settles_end_to_end(tmp_path):
    start = T0 + timedelta(minutes=15)
    state = {"phase": "pre"}

    def read(method, params):
        if method == "listMarketCatalogue":
            return [catalogue_entry("1.1", start)]
        if state["phase"] == "pre":
            return [raw_book("1.1", {11: ([(2.0, 50)], [(2.02, 50)], {2.0: 10.0}),
                                     12: ([(4.0, 50)], [(4.1, 50)], {}), 13: ([(3.4, 50)], [(3.5, 50)], {})})]
        return [raw_book("1.1", {11: ([], [], {}, "WINNER"), 12: ([], [], {}, "LOSER"),
                                 13: ([], [], {}, "LOSER")}, status="CLOSED")]

    w = ef.DayWriter(store=None, root=tmp_path, depth=3)
    feed = ef.Feed(read, w, depth=3, paper=pb.PaperBook())
    got = feed.step(T0)
    assert got["markets"] == 1 and got["rows"] == 3
    assert feed.next_due["1.1"] == T0 + timedelta(seconds=120)                               # 15 min out: every 2
    assert feed.step(T0 + timedelta(seconds=60))["rows"] == 0                                # not due yet
    state["phase"] = "post"
    feed.step(start + timedelta(minutes=101))
    assert "1.1" in feed.settled
    feed.save()
    day = tmp_path / "2026-11-07"
    assert (day / "markets.csv").exists() and list(day.glob("books_*.csv")) and (day / "paper_lines.csv").exists()
    mk = ef.read_day(None, "2026-11-07", "markets", tmp_path)
    assert set(mk["runner_name"]) == {"Arsenal", "Chelsea", "The Draw"}
    s = ef.summary(None, 1, today=datetime(2026, 11, 7).date(), root=tmp_path)["2026-11-07"]
    assert s["markets"] == 1 and s["matches"] == 1 and s["last_book_spread_ticks_median"] >= 1


def test_real_stakes_take_the_live_limits():
    from football.live import live_limits, load_config
    cfg = load_config()
    live = live_limits(cfg)
    assert live["limits"]["level_stake"] == 2.0 and live["limits"]["max_daily_turnover"] == 20.0
    assert live["limits"]["max_match_stake"] == 2.0 and live["limits"]["min_price"] == cfg["limits"]["min_price"]
    assert cfg["limits"]["max_daily_turnover"] == 100.0                     # paper keeps its own


def test_every_catalogue_call_stays_under_betfairs_200_points_and_names_its_market_type():
    """4 Nov 2026: MARKET_DESCRIPTION (weight 1) at 1,000 results was refused TOO_MUCH_DATA on every call."""
    seen = []

    def read(method, params):
        w = sum(ef.CATALOGUE_WEIGHTS.get(p, 1) for p in params["marketProjection"])
        assert w * params["maxResults"] <= 200 and params["maxResults"] <= 1000
        assert len(params["filter"]["marketTypeCodes"]) == 1
        t = params["filter"]["marketTypeCodes"][0]
        seen.append(t)
        return [{"marketId": f"{t}-1", "runners": []}]
    got = ef.catalogue_all(read, T0, T0 + timedelta(hours=2))
    assert sorted(seen) == sorted(ef.DEFAULT_TYPES)
    assert {m["marketType"] for m in got} == set(ef.DEFAULT_TYPES)
    assert ef.catalogue_limit(["EVENT", "MARKET_DESCRIPTION"]) == 200        # a weighted projection is capped
