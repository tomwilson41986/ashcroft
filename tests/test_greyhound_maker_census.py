"""research/greyhound/maker_census.py: who was resting, what it earned to the SP, and the reconciliation."""

import json

import pytest

from research.greyhound import maker_census as mc


def _line(pt, mcs):
    return json.dumps({"rt": "2026-11-10T19:00:00Z", "m": json.dumps({"op": "mcm", "pt": pt, "mc": mcs})})


def _defn(status="OPEN", bsp=None, inplay=False):
    r = {"id": 1, "status": "ACTIVE"}
    if bsp:
        r = {"id": 1, "status": "WINNER", "bsp": bsp}
    return {"marketTime": "2026-11-10T19:30:00Z", "marketType": "WIN", "venue": "Romford", "status": status,
            "inPlay": inplay, "runners": [r]}


T0 = 1794339000000            # 2026-11-10T19:30:00Z in ms, the scheduled off


def _stream():
    return [
        # first sight, 100 s out: GBP50 already traded at 3.0; best to back 3.0, best to lay 3.2
        _line(T0 - 100_000, [{"id": "1.1", "img": True, "marketDefinition": _defn(),
                              "rc": [{"id": 1, "atb": [[3.0, 10]], "atl": [[3.2, 10]], "trd": [[3.0, 50]], "tv": 50}]}]),
        # 50 s out: 20 more at 3.0 (a backer took a resting lay)
        _line(T0 - 50_000, [{"id": "1.1", "rc": [{"id": 1, "trd": [[3.0, 70]], "tv": 70}]}]),
        # 20 s out: 30 at 3.2 (a layer took a resting back) and 5 at 3.1, inside the spread
        _line(T0 - 20_000, [{"id": "1.1", "rc": [{"id": 1, "trd": [[3.2, 30], [3.1, 5]], "tv": 105}]}]),
        # a resubscription image with 10 more at 3.0: the book before it is not known
        _line(T0 - 15_000, [{"id": "1.1", "img": True, "marketDefinition": _defn(),
                             "rc": [{"id": 1, "atb": [[3.0, 10]], "atl": [[3.2, 10]],
                                     "trd": [[3.0, 80], [3.2, 30], [3.1, 5]], "tv": 115}]}]),
        # closed with the SP
        _line(T0 + 60_000, [{"id": "1.1", "marketDefinition": _defn("CLOSED", bsp=3.1)}]),
    ]


def test_each_trade_is_read_against_the_book_before_it():
    tr, notes = mc.day_trades(_stream())
    by = {(r.side, r.price): r for r in tr.itertuples()}
    lay, back = by[("resting lay", 3.0)], by[("resting back", 3.2)]
    assert lay.vol == 20 and lay.t_bucket == "60-30" and lay.ret == pytest.approx(1 - 3.0 / 3.1)
    assert back.vol == 30 and back.t_bucket == "30-10" and back.ret == pytest.approx(3.2 / 3.1 - 1)
    assert lay.spread == "3-4 ticks" and lay.band == "3-5" and len(tr) == 2
    # nothing dropped silently: the inside-the-spread 5 and the image's 10 are counted, and the volume reconciles
    assert notes["unsigned_volume"] == {"inside the spread or no book": 5.0, "image": 10.0}
    assert notes["runners_with_volume"] == 1 and notes["reconciled_share"] == 1.0 and notes["signed_volume"] == 50.0


def test_the_cells_net_two_percent_off_a_gain_only():
    tr, _ = mc.day_trades(_stream())
    c = mc.cells(tr, "2026-11-10", 0.02)
    row = c[c.side == "resting lay"].iloc[0]
    assert row.gross == pytest.approx(20 * (1 - 3.0 / 3.1)) and row.net == pytest.approx(0.98 * row.gross)
    tr2 = tr.assign(ret=-tr.ret)
    c2 = mc.cells(tr2, "2026-11-10", 0.02)
    assert (c2.net == c2.gross).all()


def test_buckets():
    assert mc.t_bucket(1000) == "900+" and mc.t_bucket(59.9) == "60-30" and mc.t_bucket(-3) == "after the off"
    assert mc.band(1.9) == "under 2" and mc.band(15.0) == "15+"
    assert mc.spread_bucket(3.0, 3.05) == "1 tick" and mc.spread_bucket(None, 3.0) == "one side empty"
