"""The Betfair Historic Data test's mechanics: the collection moment, the prices read around it, the trade."""

from __future__ import annotations

import bz2
import json

import numpy as np
import pandas as pd

from research.open_close import football_bf_historic_test as bt


def test_the_collection_moment_is_friday_17_or_tuesday_13_uk():
    sat = bt.collection_instant(pd.Timestamp("2026-08-15T14:00Z"))           # Saturday: Friday 17:00 BST
    assert sat == pd.Timestamp("2026-08-14T16:00Z")
    wed = bt.collection_instant(pd.Timestamp("2026-08-19T18:45Z"))           # Wednesday: Tuesday 13:00 BST
    assert wed == pd.Timestamp("2026-08-18T12:00Z")


def _stream():
    defn = {"eventId": "1", "eventName": "Alpha v Beta", "countryCode": "GB", "marketType": "MATCH_ODDS",
            "marketTime": "2026-08-15T14:00:00.000Z", "inPlay": False,
            "runners": [{"id": 1, "name": "Alpha", "status": "WINNER"}, {"id": 2, "name": "Beta", "status": "LOSER"},
                        {"id": 3, "name": "The Draw", "status": "LOSER"}]}
    ms = lambda t: int(pd.Timestamp(t).value // 10**6)                       # noqa: E731
    lines = [{"pt": ms("2026-08-13T10:00Z"), "mc": [{"id": "1.5", "marketDefinition": defn,
                                                     "rc": [{"id": 1, "ltp": 2.5}]}]},
             {"pt": ms("2026-08-14T15:30Z"), "mc": [{"id": "1.5", "rc": [{"id": 1, "ltp": 2.4}]}]},   # before 16:00
             {"pt": ms("2026-08-14T18:00Z"), "mc": [{"id": "1.5", "rc": [{"id": 1, "ltp": 2.3}]}]},   # after
             {"pt": ms("2026-08-15T13:59Z"), "mc": [{"id": "1.5", "rc": [{"id": 1, "ltp": 2.1}]}]},
             {"pt": ms("2026-08-15T14:02Z"), "mc": [{"id": "1.5", "marketDefinition": {**defn, "inPlay": True}}]},
             {"pt": ms("2026-08-15T14:10Z"), "mc": [{"id": "1.5", "rc": [{"id": 1, "ltp": 1.5}]}]}]     # in play
    return bz2.compress("\n".join(json.dumps(x) for x in lines).encode())


def test_the_prices_around_the_collection_moment_and_the_close_ignore_in_play():
    rows = {r["selection_id"]: r for r in bt.parse_match_odds(_stream())}
    a = rows[1]
    assert a["ltp_coll"] == 2.4 and abs(a["coll_age_h"] - 0.5) < 1e-9 and a["ltp_after_coll"] == 2.3
    assert a["ltp_first"] == 2.5 and a["ltp_close"] == 2.1 and a["won"] == 1      # in-play 1.5 never read


def test_a_back_first_trade_gains_when_the_price_shortens_and_pays_a_tick_each_end():
    p = pd.DataFrame({"match_id": ["m"], "entry_h": [2.5], "ltp_close_h": [2.0], "entry_a": [3.0],
                      "ltp_close_a": [3.0], "yb_h": [0.2], "yb_a": [0.0], "fd_code": ["E1"], "week": ["w"]})
    t = bt.trades(p, [0.1], [0.0], np.array([1]), np.array([0]), "ltp")
    assert len(t) == 1 and abs(t["profit"].iloc[0] - (2.5 / 2.0 - 1)) < 1e-12 and t["right_way"].iloc[0]
    tk = bt.trades(p, [0.1], [0.0], np.array([1]), np.array([0]), "tick")
    assert abs(tk["profit"].iloc[0] - (2.48 / 2.02 - 1)) < 1e-12                  # 2.5 -> 2.48, 2.0 -> 2.02
