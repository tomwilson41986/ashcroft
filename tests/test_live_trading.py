"""The live trader (the owner's rule, real orders): the closing-CLV strategy, the live exchange's orders, the
session with it, and the owner's switch. Hand-made markets and a stand-in for Betfair: nothing here reaches the
exchange, and nothing is evaluated on these numbers."""

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from trading import config as tc
from trading.exchange import Book, LiveExchange, Market, Quote
from trading.session import Session
from trading.strategy import RunnerView, plan_race

UTC = timezone.utc
DAY = date(2026, 10, 1)
LIVE = tc.load_config("trading/config_live.json", env={})


def _live_cfg(**limits):
    cfg = tc.load_config("trading/config_live.json", env={})
    for k, v in limits.items():
        setattr(cfg.limits, k, v)
    return cfg


def _views(backs, ours, traded, lays=None):
    lays = lays or [b * 1.04 for b in backs]
    return [RunnerView(selection_id=i + 1, name=f"h{i + 1}", model_price=ours[i], back=backs[i], back_size=500.0,
                       lay=lays[i], traded=traded[i]) for i in range(len(backs))]


# ---------------------------------------------------------------- the strategy

def test_the_rule_backs_the_value_to_win_250_at_most_300_a_bet():
    # books like a real market's: the back prices add to about 100%
    got = plan_race(_views([7.0, 2.3, 3.6, 6.0], [4.0, 2.6, 4.2, 8.0], [500, 900, 800, 300]), LIVE, 4000)
    assert [d.selection_id for d in got] == [1]
    assert got[0].edge >= 0.03 and got[0].target == pytest.approx(250 / 6.0)
    short = plan_race(_views([1.4, 5.0, 12.0], [1.15, 12.0, 20.0], [5000, 900, 800]), LIVE, 4000)
    assert [d.selection_id for d in short] == [1]
    assert short[0].target == pytest.approx(300.0)                # to win 250 at 1.4 would be GBP625


def test_a_race_not_wholly_priced_is_left_alone():
    views = _views([7.0, 2.3, 3.6, 6.0], [4.0, 2.6, 4.2, 8.0], [500, 900, 800, 300])
    views[2].model_price = None                                    # a reserve that got in: the model never priced it
    assert plan_race(views, LIVE, 4000) == []


def test_matched_money_is_required_unless_the_feed_has_none():
    thin = _views([7.0, 2.3, 3.6, 6.0], [4.0, 2.6, 4.2, 8.0], [50, 900, 800, 300])   # the value horse: GBP50
    assert plan_race(thin, LIVE, 4000) == []
    blind = _views([7.0, 2.3, 3.6, 6.0], [4.0, 2.6, 4.2, 8.0], [0, 0, 0, 0])          # the delayed key's feed
    got = plan_race(blind, LIVE, 4000)
    assert [d.selection_id for d in got] == [1] and "without volume" in got[0].reason


# ---------------------------------------------------------------- the live exchange

class _Client:
    """A stand-in for betfair_client: records each call, answers placeOrders as filled at the limit (or as told)."""

    def __init__(self, fill=True, lose_reply=False, orders=None, orders_fail=False):
        self.calls, self.fill, self.lose_reply = [], fill, lose_reply
        self.current, self.orders_fail = orders or [], orders_fail

    def login(self):
        self.calls.append(("login", None))

    def _api_call(self, method, params):
        self.calls.append((method, params))
        if method == "placeOrders":
            if self.lose_reply:
                raise TimeoutError("read timed out")
            ins = params["instructions"][0]
            if ins["orderType"] == "MARKET_ON_CLOSE":
                return {"status": "SUCCESS", "instructionReports": [{"status": "SUCCESS", "betId": "L1"}]}
            if not self.fill:
                return {"status": "SUCCESS", "instructionReports": [
                    {"status": "SUCCESS", "betId": "B0", "sizeMatched": 0.0, "orderStatus": "EXPIRED"}]}
            lo = ins["limitOrder"]
            return {"status": "SUCCESS", "instructionReports": [
                {"status": "SUCCESS", "betId": "B1", "sizeMatched": lo["size"], "averagePriceMatched": lo["price"]}]}
        if method == "listCurrentOrders":
            if self.orders_fail:
                raise ConnectionError("no route")
            return {"currentOrders": self.current, "moreAvailable": False}
        return {}


class _Data:
    def __init__(self, client, markets=(), books=None):
        self.client, self._markets, self._books = client, list(markets), books or {}

    def login(self):
        self.client.login()

    def markets(self, start, end, countries=("GB", "IE")):
        return list(self._markets)

    def books(self, ids, with_sp=False):
        return {i: self._books[i] for i in ids if i in self._books}


def test_a_back_is_a_fill_or_kill_limit_order_at_the_price_read():
    c = _Client()
    f = LiveExchange(_Data(c)).back("1.5", 7, 10.0, 27.777, "ash10017")
    method, params = c.calls[-1]
    ins = params["instructions"][0]
    assert method == "placeOrders" and params["marketId"] == "1.5"
    assert ins["side"] == "BACK" and ins["orderType"] == "LIMIT" and ins["customerOrderRef"] == "ash10017"
    assert ins["limitOrder"] == {"size": 27.77, "price": 10.0, "persistenceType": "LAPSE",
                                 "timeInForce": "FILL_OR_KILL", "minFillSize": 2.0}
    assert len(params["customerRef"]) <= 32
    assert f.status == "SUCCESS" and f.matched == 27.77 and f.avg_price == 10.0 and f.bet_id == "B1"
    miss = LiveExchange(_Data(_Client(fill=False))).back("1.5", 7, 10.0, 27.77, "ash10017")
    assert miss.status == "FAILURE" and miss.matched == 0 and miss.error == "EXPIRED"


def test_the_trade_out_is_a_lay_at_the_sp_for_a_liability_and_nothing_is_ever_cancelled():
    c = _Client()
    x = LiveExchange(_Data(c))
    f = x.lay_at_bsp("1.5", 7, 249.93, "ash10017x")
    ins = c.calls[-1][1]["instructions"][0]
    assert ins["side"] == "LAY" and ins["orderType"] == "MARKET_ON_CLOSE"
    assert ins["marketOnCloseOrder"] == {"liability": 249.93}
    assert f.status == "PENDING"
    n = len(c.calls)
    x.cancel_all()
    assert len(c.calls) == n                                       # the hedges stay


def test_a_lost_reply_is_settled_by_betfairs_own_record():
    placed = [{"betId": "B9", "marketId": "1.5", "selectionId": 7, "side": "BACK", "sizeMatched": 27.77,
               "averagePriceMatched": 10.0, "customerOrderRef": "ash10017", "placedDate": "2026-10-01T08:00:00Z"}]
    got = LiveExchange(_Data(_Client(lose_reply=True, orders=placed))).back("1.5", 7, 10.0, 27.77, "ash10017")
    assert got.status == "SUCCESS" and got.matched == 27.77 and got.bet_id == "B9"
    none = LiveExchange(_Data(_Client(lose_reply=True))).back("1.5", 7, 10.0, 27.77, "ash10017")
    assert none.status == "FAILURE" and none.matched == 0
    blind = LiveExchange(_Data(_Client(lose_reply=True, orders_fail=True))).back("1.5", 7, 10.0, 27.77, "ash10017")
    assert blind.status == "UNKNOWN"


# ---------------------------------------------------------------- a live session

class _Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def _race(mid, off, backs, ours, traded):
    n = len(backs)
    market = Market(mid, "York", "GB", off, runners={i + 1: f"h{i + 1}" for i in range(n)})
    book = Book(market_id=mid, status="OPEN", inplay=False, total_matched=None, runners={
        i + 1: Quote(i + 1, back=[(backs[i], 500.0)], lay=[(round(backs[i] * 1.04, 2), 500.0)], traded=traded[i])
        for i in range(n)})
    preds = pd.DataFrame({"market_id": [mid] * n, "selection_id": list(range(1, n + 1)), "predicted_bfsp": ours})
    return market, book, preds


def _session(client, races, cfg=None):
    markets = [r[0] for r in races]
    data = _Data(client, markets, {r[0].market_id: r[1] for r in races})
    preds = pd.concat([r[2] for r in races], ignore_index=True)
    clock = _Clock(datetime(2026, 10, 1, 8, 0, tzinfo=UTC))       # 09:00 UK, inside the window
    s = Session(LiveExchange(data), preds, cfg or _live_cfg(), DAY, clock=clock, sleep=lambda _: None)
    s.load_markets()
    return s, clock


def test_a_live_day_backs_the_value_once_and_lays_it_at_the_sp():
    off = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)
    c = _Client()
    s, clock = _session(c, [_race("1.7", off, [7.0, 2.3, 3.6, 6.0], [4.0, 2.6, 4.2, 8.0], [500, 900, 800, 300])])
    s.step(clock())
    s.step(clock())                                                # the next poll adds nothing
    orders = [p["instructions"][0] for m, p in c.calls if m == "placeOrders"]
    assert [o["side"] for o in orders] == ["BACK", "LAY"]
    assert orders[0]["limitOrder"]["size"] == 41.66                  # to win GBP250 at 7.0, to the penny below
    assert orders[1]["marketOnCloseOrder"]["liability"] == pytest.approx(41.66 * 6.0, abs=0.01)
    rows = [r for r in s.ledger if r["event"] in ("back", "trade_out")]
    assert {r["mode"] for r in rows} == {"live"} and s.positions[("1.7", 1)].hedged


def test_the_days_limit_goes_to_the_best_value_first():
    early, late = datetime(2026, 10, 1, 12, 0, tzinfo=UTC), datetime(2026, 10, 1, 15, 0, tzinfo=UTC)
    small = _race("1.8", early, [6.0, 2.0, 3.2], [4.5, 2.2, 3.5], [500, 900, 800])     # the earlier race, less value
    big = _race("1.9", late, [10.0, 2.0, 2.8], [4.0, 2.4, 3.2], [500, 900, 800])
    c = _Client()
    s, clock = _session(c, [small, big], cfg=_live_cfg(max_daily_turnover=29.0))
    s.step(clock())
    backs = [p for m, p in c.calls if m == "placeOrders" and p["instructions"][0]["side"] == "BACK"]
    assert [p["marketId"] for p in backs] == ["1.9"]               # 27.77 on the best; 1.23 left is under GBP2


def test_a_session_that_starts_again_takes_its_positions_from_betfair():
    off = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)
    held = [{"marketId": "1.7", "selectionId": 1, "side": "BACK", "orderType": "LIMIT", "sizeMatched": 41.66,
             "averagePriceMatched": 7.0, "customerOrderRef": "ash10011"},
            {"marketId": "1.7", "selectionId": 1, "side": "LAY", "orderType": "MARKET_ON_CLOSE",
             "bspLiability": 249.96, "customerOrderRef": "ash10011x"},
            {"marketId": "1.7", "selectionId": 2, "side": "BACK", "orderType": "LIMIT", "sizeMatched": 5.0,
             "averagePriceMatched": 3.0, "customerOrderRef": "someone-else"}]   # not ours: left alone
    c = _Client(orders=held)
    s, clock = _session(c, [_race("1.7", off, [7.0, 2.3, 3.6, 6.0], [4.0, 2.6, 4.2, 8.0], [500, 900, 800, 300])])
    s.sync(s.x.orders(["1.7"]))
    assert s.positions[("1.7", 1)].matched == pytest.approx(41.66) and s.positions[("1.7", 1)].hedged
    assert ("1.7", 2) not in s.positions and s.state.turnover == pytest.approx(41.66)
    s.step(clock())
    assert not [p for m, p in c.calls if m == "placeOrders"]        # the horse is not backed a second time


def test_an_order_whose_fate_is_unknown_stops_the_day():
    off = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)
    c = _Client(lose_reply=True, orders_fail=True)
    s, clock = _session(c, [_race("1.7", off, [7.0, 2.3, 3.6, 6.0], [4.0, 2.6, 4.2, 8.0], [500, 900, 800, 300])])
    s.step(clock())
    assert s.state.stopped.startswith("order state unknown")
    n = sum(1 for m, _ in c.calls if m == "placeOrders")
    s.step(clock())
    assert sum(1 for m, _ in c.calls if m == "placeOrders") == n


def test_a_refused_trade_out_is_sent_again_and_then_left_to_the_result():
    class Refuse(_Client):
        def _api_call(self, method, params):
            if method == "placeOrders" and params["instructions"][0]["orderType"] == "MARKET_ON_CLOSE":
                self.calls.append((method, params))
                return {"status": "FAILURE", "errorCode": "BET_ACTION_ERROR",
                        "instructionReports": [{"status": "FAILURE", "errorCode": "INVALID_BSP_BET"}]}
            return super()._api_call(method, params)
    off = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)
    c = Refuse()
    s, clock = _session(c, [_race("1.7", off, [7.0, 2.3, 3.6, 6.0], [4.0, 2.6, 4.2, 8.0], [500, 900, 800, 300])])
    for _ in range(8):
        s.step(clock())
        clock.t += timedelta(minutes=1)
    lays = [p for m, p in c.calls if m == "placeOrders" and p["instructions"][0]["side"] == "LAY"]
    assert len(lays) == 5 and not s.positions[("1.7", 1)].hedged   # five tries, then left to the result


# ---------------------------------------------------------------- the owner's switch

def test_live_trading_needs_the_owners_switch(monkeypatch, tmp_path):
    import auto_trade
    for k in ("BETFAIR_USERNAME", "BETFAIR_PASSWORD", "BETFAIR_APP_KEY"):
        monkeypatch.setenv(k, "x")
    monkeypatch.setenv("TRADING_LIVE", "no")
    monkeypatch.setattr(auto_trade, "BetfairData", lambda: pytest.fail("no Betfair call without the switch"))
    out = auto_trade.main(["--live", "--date", "2026-10-01", "--out", str(tmp_path)])
    assert out["traded"] is False and out["mode"] == "live" and "TRADING_LIVE" in out["reason"]
    assert auto_trade.live_switch_on({"TRADING_LIVE": "yes"}) and not auto_trade.live_switch_on({})


def test_a_horse_backed_already_is_topped_up_only_to_what_is_left_to_win():
    # backed at 7.0 for GBP41.66 (to win 249.96); the price is now 6.0, still value: nothing more to back
    off = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)
    held = [{"marketId": "1.7", "selectionId": 1, "side": "BACK", "orderType": "LIMIT", "sizeMatched": 41.66,
             "averagePriceMatched": 7.0, "customerOrderRef": "ash10011"},
            {"marketId": "1.7", "selectionId": 1, "side": "LAY", "orderType": "MARKET_ON_CLOSE",
             "bspLiability": 249.96, "customerOrderRef": "ash10011x"}]
    c = _Client(orders=held)
    s, clock = _session(c, [_race("1.7", off, [6.0, 2.3, 3.6, 6.0], [3.5, 2.6, 4.2, 8.0], [500, 900, 800, 300])])
    s.sync(s.x.orders(["1.7"]))
    s.step(clock())
    assert not [p for m, p in c.calls if m == "placeOrders" and p["instructions"][0]["selectionId"] == 1]


def test_a_lost_session_is_logged_in_again_and_the_order_sent_once_more():
    class Expired(_Client):
        def _api_call(self, method, params):
            if method == "placeOrders" and not any(m == "login" for m, _ in self.calls):
                self.calls.append((method, params))
                raise RuntimeError("400 Client Error: {\"faultcode\":\"Client\",\"detail\":{\"APINGException\":"
                                   "{\"errorCode\":\"INVALID_SESSION_INFORMATION\"}}}")
            return super()._api_call(method, params)
    c = Expired()
    f = LiveExchange(_Data(c)).back("1.5", 7, 7.0, 41.66, "ash10017")
    assert [m for m, _ in c.calls] == ["placeOrders", "login", "placeOrders"] and f.status == "SUCCESS"


def test_the_trader_waits_for_a_late_0600_record(monkeypatch):
    import io as _io
    import auto_trade
    tries = {"n": 0}

    class S3:
        def get_object(self, Bucket, Key):
            tries["n"] += 1
            if tries["n"] < 3:
                raise KeyError("NoSuchKey")
            return {"Body": _io.BytesIO(b"race_time,venue,runner_name,predicted_bfsp\n1:30,York,Alpha,3.5\n")}
    monkeypatch.setattr(auto_trade, "_s3", lambda: S3())
    later = datetime.now(UTC) + timedelta(hours=1)
    df = auto_trade.load_predictions(DAY, None, wait_until=later, sleep=lambda _: None)
    assert tries["n"] == 3 and len(df) == 1
