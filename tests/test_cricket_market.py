"""cricket/market: the read-only wrapper refuses every betting method; the recorder's catalogue, cadence and record.

The Betfair replies here are hand-written in Betfair's reply shape to drive the recorder's bookkeeping; they are
not market data and nothing is scored on them."""

import datetime as dt
import gzip
import io

import pandas as pd
import pytest

from cricket.market.betfair_read import READ_METHODS, ReadOnly
from cricket.market.recorder import PER_CALL, Recorder

T0 = dt.datetime(2026, 11, 4, 12, 0, tzinfo=dt.timezone.utc)


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t


class FakeClient:
    def __init__(self, clock):
        self.clock, self.calls, self.inplay = clock, [], True

    def login(self):
        pass

    def _api_call(self, method, params):
        self.calls.append((method, params))
        if method == "listMarketTypes":
            return [{"marketType": "MATCH_ODDS", "marketCount": 2}]
        if method == "listMarketCatalogue":
            return [{"marketId": f"1.{i}", "marketName": "Match Odds", "marketStartTime": "2026-11-04T11:00:00.000Z",
                     "description": {"marketType": "MATCH_ODDS"}, "event": {"id": f"e{i}", "name": "A v B"},
                     "runners": [{"selectionId": 1, "runnerName": "A"}, {"selectionId": 2, "runnerName": "B"}]}
                    for i in range(12)]
        if method == "listMarketBook":
            return [{"marketId": m, "status": "OPEN", "inplay": self.inplay, "betDelay": 5, "totalMatched": 1000.0,
                     "runners": [{"selectionId": 1, "status": "ACTIVE", "lastPriceTraded": 1.8,
                                  "ex": {"availableToBack": [{"price": 1.79, "size": 50.0}],
                                         "availableToLay": [{"price": 1.81, "size": 40.0}],
                                         "tradedVolume": [{"price": 1.8, "size": 900.0}]}}]}
                    for m in params["marketIds"]]
        raise AssertionError(method)


@pytest.mark.parametrize("method", ["placeOrders", "cancelOrders", "replaceOrders", "updateOrders"])
def test_read_only_refuses_every_betting_method(method):
    c = FakeClient(Clock())
    with pytest.raises(PermissionError):
        ReadOnly(c).read(method, {})
    assert c.calls == []
    assert method not in READ_METHODS


def test_recorder_reads_in_batches_on_the_in_play_cadence(tmp_path):
    clock = Clock()
    api = ReadOnly(FakeClient(clock))
    rec = Recorder(api, store=None, root=tmp_path, clock=clock, inplay_s=5.0)
    assert rec.census() == {"MATCH_ODDS": 2}
    assert rec.catalogue() == 12
    n = rec.poll()
    books = [p for m, p in api.client.calls if m == "listMarketBook"]
    assert [len(p["marketIds"]) for p in books] == [PER_CALL, 12 - PER_CALL]
    assert n == 12
    assert rec.due(clock.t) == []
    clock.t += dt.timedelta(seconds=5)
    assert len(rec.due(clock.t)) == 12          # in play: due again five seconds on
    rec.save()
    f = next(tmp_path.glob("2026-11-04/books_12_*.csv.gz"))
    df = pd.read_csv(io.BytesIO(gzip.decompress(f.read_bytes())))
    assert len(df) == 12 and set(df.bet_delay) == {5}
    assert df.back1_p.iloc[0] == 1.79 and df.traded.iloc[0] == "1.8:900.0"
