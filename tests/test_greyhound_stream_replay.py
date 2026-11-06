"""research/greyhound/stream_replay.py on hand-built stream messages: the book rebuilt from images and deltas, a lay
taken at once, the queue cleared before our fill, a trade through our price, cancellations shortening the queue, the
close at the best price and at the SP."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from research.greyhound import stream_replay as sr

OFF = datetime(2026, 11, 5, 14, 0, tzinfo=timezone.utc)
MID = "1.1"


def line(t, mc):
    return json.dumps({"rt": t.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                       "m": json.dumps({"op": "mcm", "pt": int(t.timestamp() * 1000), "mc": [mc]})})


def defn(inplay=False, bsp=None):
    return {"marketTime": OFF.strftime("%Y-%m-%dT%H:%M:%S.000Z"), "inPlay": inplay, "status": "OPEN",
            "runners": [{"id": 7, "status": "ACTIVE", **({"bsp": bsp} if bsp else {})}]}


def image(t, atb, atl, trd):
    return line(t, {"id": MID, "img": True, "marketDefinition": defn(),
                    "rc": [{"id": 7, "atb": atb, "atl": atl, "trd": trd, "ltp": 5.0}]})


def delta(t, **rc):
    return line(t, {"id": MID, "rc": [{"id": 7, **rc}]})


T = OFF - timedelta(minutes=5)


def test_the_book_is_rebuilt_from_an_image_and_deltas():
    markets = {}
    for _, mc in sr.events([image(T, [[5.0, 10], [4.9, 20]], [[5.1, 8]], [[5.0, 30]]),
                            delta(T + timedelta(seconds=1), atb=[[5.0, 0], [4.8, 5]], trd=[[5.0, 35]])]):
        sr.apply(markets, mc)
    r = markets[MID].runners[7]
    assert r.atb == {4.9: 20, 4.8: 5} and r.atl == {5.1: 8} and r.trd == {5.0: 35}
    assert markets[MID].off == OFF and r.best_back() == (4.9, 20)


def test_a_lay_takes_what_is_offered_at_or_under_its_price():
    lines = [image(T, [[4.9, 10]], [[5.0, 3], [5.1, 50]], [])]
    o = sr.SimOrder(MID, 7, "LAY", 5.0, 5.0, T)
    df = sr.replay(lines, [o])
    assert o.taken == 3 and o.matched == 3 and o.queue_at_place == 0      # 3 taken at 5.0; the rest rests


def test_the_queue_ahead_is_cleared_before_our_order_fills():
    lines = [image(T, [[5.0, 10], [4.9, 20]], [[5.2, 8]], [[5.0, 100]]),
             delta(T + timedelta(seconds=5), trd=[[5.0, 106]], atb=[[5.0, 4]]),      # 6 traded: 10 -> 4 ahead
             delta(T + timedelta(seconds=10), trd=[[5.0, 112]], atb=[[5.0, 0.01]]),  # 6 more: 4 ahead cleared, 2 ours
             line(OFF - timedelta(seconds=4), {"id": MID, "rc": [{"id": 7, "atb": [[5.4, 9]]}]})]
    o = sr.SimOrder(MID, 7, "LAY", 5.0, 5.0, T + timedelta(seconds=1))
    df = sr.replay(lines, [o])
    row = df.iloc[0]
    assert row.queue_at_place == 10 and row.matched == 2 and row.close_price == 5.4
    assert row.pnl == pytest.approx(2 * (1 - 5.0 / 5.4), abs=1e-4)
    half = sr.replay(lines, [sr.SimOrder(MID, 7, "LAY", 5.0, 5.0, T + timedelta(seconds=1))], vol_factor=0.5)
    assert half.iloc[0].matched == 0                                          # half the volume never clears the queue


def test_cancellations_shorten_the_queue_and_a_trade_through_fills_in_full():
    lines = [image(T, [[5.0, 50]], [[5.2, 8]], [[5.0, 0]]),
             delta(T + timedelta(seconds=5), atb=[[5.0, 2]]),                        # 48 cancelled: 2 ahead at most
             delta(T + timedelta(seconds=6), trd=[[5.0, 3]]),                         # 3 traded: 2 ahead, 1 ours
             delta(T + timedelta(seconds=7), trd=[[4.9, 4]])]                        # traded under our 5.0: through
    o = sr.SimOrder(MID, 7, "LAY", 5.0, 5.0, T + timedelta(seconds=1))
    sr.replay(lines, [o])
    assert o.matched == 5.0 and o.first_fill == T + timedelta(seconds=6)


def test_unmatched_lapses_at_the_close_and_the_sp_close_is_read_when_settled():
    lines = [image(T, [[5.0, 10]], [[5.2, 8]], []),
             line(OFF + timedelta(minutes=2), {"id": MID, "marketDefinition": defn(inplay=True, bsp=6.0)})]
    o = sr.SimOrder(MID, 7, "LAY", 5.0, 5.0, T + timedelta(seconds=1))
    df = sr.replay(lines, [o])
    assert df.iloc[0].matched == 0 and df.iloc[0].pnl is None
    s = sr.summary(df)
    assert s["orders"] == 1 and s["matched"] == 0 and s["filled_share"] == 0


def test_a_back_rests_among_the_lay_offers():
    lines = [image(T, [[4.8, 10]], [[5.0, 7]], [[5.0, 0]]),
             delta(T + timedelta(seconds=5), trd=[[5.0, 10]]),
             delta(OFF - timedelta(seconds=4), atl=[[4.6, 3]])]
    o = sr.SimOrder(MID, 7, "BACK", 5.0, 4.0, T + timedelta(seconds=1))
    df = sr.replay(lines, [o])
    assert o.queue_at_place == 7 and o.matched == 3
    assert df.iloc[0].close_price == 4.6 and df.iloc[0].pnl == pytest.approx(3 * (5.0 / 4.6 - 1), abs=1e-4)



def test_the_sp_close_reads_the_settled_definition():
    lines = [image(T, [[5.0, 10]], [[5.2, 8]], [[5.0, 0]]),
             delta(T + timedelta(seconds=5), trd=[[4.9, 2]]),                       # through: filled in full at 5.0
             line(OFF - timedelta(seconds=4), {"id": MID, "rc": [{"id": 7, "atb": [[5.4, 9]]}]}),
             line(OFF + timedelta(minutes=3), {"id": MID, "marketDefinition": {**defn(inplay=False, bsp=6.0),
                                                                                "status": "CLOSED"}})]
    df = sr.replay(lines, [sr.SimOrder(MID, 7, "LAY", 5.0, 4.0, T + timedelta(seconds=1))])
    row = df.iloc[0]
    assert row.matched == 4 and row.sp == 6.0 and row.pnl_sp == pytest.approx(4 * (1 - 5.0 / 6.0), abs=1e-4)


def test_the_result_target_reads_the_settled_status():
    settled = {**defn(inplay=False, bsp=6.0), "status": "CLOSED"}
    for status, want in (("LOSER", 4.0), ("WINNER", -16.0)):
        d = {**settled, "runners": [{"id": 7, "status": status, "bsp": 6.0}]}
        lines = [image(T, [[5.0, 10]], [[5.2, 8]], [[5.0, 0]]),
                 delta(T + timedelta(seconds=5), trd=[[4.9, 2]]),
                 line(OFF + timedelta(minutes=3), {"id": MID, "marketDefinition": d})]
        row = sr.replay(lines, [sr.SimOrder(MID, 7, "LAY", 5.0, 4.0, T + timedelta(seconds=1))]).iloc[0]
        assert row.won == (status == "WINNER") and row.pnl_result == pytest.approx(want)   # lay 4 at 5.0
    assert sr.result_pnl("BACK", 3.0, 2.0, 1, comm=0.02) == pytest.approx(2 * 2 * 0.98)
    assert sr.result_pnl("BACK", 3.0, 2.0, None) is None                                  # no result: not scored
