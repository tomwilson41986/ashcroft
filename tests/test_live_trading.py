"""The live trader (the owner's rule, real orders): the closing-CLV strategy, the live exchange's orders, the
session with it, and the owner's switch. Hand-made markets and a stand-in for Betfair: nothing here reaches the
exchange, and nothing is evaluated on these numbers."""

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from trading import config as tc
from trading.exchange import BetfairData, Book, LiveExchange, Market, PaperExchange, Quote
from trading.session import LEDGER_FIELDS, SETTLED_BY_BETFAIR, Session
from trading.strategy import RunnerView, plan_race

UTC = timezone.utc
DAY = date(2026, 10, 1)
LIVE = tc.load_config("trading/config_live.json", env={})
STAKE = 250.0      # the mechanics below are written to win GBP250 (the live target until 7 Oct); the owner's own
                   # target and limits are tested on LIVE itself (test_the_owners_commission_stake_and_limits)


def _live_cfg(**limits):
    cfg = tc.load_config("trading/config_live.json", env={})
    cfg.target = STAKE
    cfg.closing_volume, cfg.topups = True, True   # the mechanics below read the feed's volume and top up; the owner's
                                                  # rule from 9 Oct (neither) is tested on LIVE (test_the_owners_rule_*)
    for k, v in limits.items():
        setattr(cfg.limits, k, v)
    return cfg


MECH = _live_cfg()


def _views(backs, ours, traded, lays=None):
    lays = lays or [b * 1.04 for b in backs]
    return [RunnerView(selection_id=i + 1, name=f"h{i + 1}", model_price=ours[i], back=backs[i], back_size=500.0,
                       lay=lays[i], traded=traded[i]) for i in range(len(backs))]


# ---------------------------------------------------------------- the strategy

def test_the_rule_backs_the_value_to_win_250_at_most_300_a_bet():
    # books like a real market's: the back prices add to about 100%
    got = plan_race(_views([7.0, 2.3, 3.6, 6.0], [4.0, 2.6, 4.2, 8.0], [500, 900, 800, 300]), MECH, 4000)
    assert [d.selection_id for d in got] == [1]
    assert got[0].edge >= 0.03 and got[0].target == pytest.approx(250 / 6.0)
    short = plan_race(_views([1.4, 5.0, 12.0], [1.15, 12.0, 20.0], [5000, 900, 800]), MECH, 4000)
    assert [d.selection_id for d in short] == [1]
    assert short[0].target == pytest.approx(300.0)                # to win 250 at 1.4 would be GBP625


def test_a_race_not_wholly_priced_is_left_alone():
    views = _views([7.0, 2.3, 3.6, 6.0], [4.0, 2.6, 4.2, 8.0], [500, 900, 800, 300])
    views[2].model_price = None                                    # a reserve that got in: the model never priced it
    assert plan_race(views, MECH, 4000) == []


def test_matched_money_is_required_unless_the_feed_has_none():
    thin = _views([7.0, 2.3, 3.6, 6.0], [4.0, 2.6, 4.2, 8.0], [50, 900, 800, 300])   # the value horse: GBP50
    assert plan_race(thin, MECH, 4000) == []
    blind = _views([7.0, 2.3, 3.6, 6.0], [4.0, 2.6, 4.2, 8.0], [0, 0, 0, 0])          # the delayed key's feed
    got = plan_race(blind, MECH, 4000)
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
    def __init__(self, client, markets=(), books=None, cleared=None, cleared_fail=False):
        self.client, self._markets, self._books = client, list(markets), books or {}
        self._cleared, self.cleared_fail = list(cleared or []), cleared_fail

    def login(self):
        self.client.login()

    def markets(self, start, end, countries=("GB", "IE")):
        return list(self._markets)

    def books(self, ids, with_sp=False):
        return {i: self._books[i] for i in ids if i in self._books}

    def cleared(self, ids):                                         # Betfair's record of the settled bets
        if self.cleared_fail:
            raise ConnectionError("no route")
        return [o for o in self._cleared if o["marketId"] in ids]


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


def test_the_owners_rule_from_9_oct_reads_every_race_without_volume_and_backs_each_horse_once():
    """The owner, 9 Oct: "No-volume model, first back only". The live record 30 Sep - 8 Oct: first backs priced by
    the model fitted without volume +7.9% CLV against the BSP; the volume model on the live key's feed -3.9%, and
    top-ups -2.2% against first backs +5.7%."""
    assert LIVE.closing_volume is False and LIVE.topups is False
    thin = _views([7.0, 2.3, 3.6, 6.0], [4.0, 2.6, 4.2, 8.0], [50, 900, 800, 300])   # the live key's feed
    blind = _views([7.0, 2.3, 3.6, 6.0], [4.0, 2.6, 4.2, 8.0], [0, 0, 0, 0])         # the delayed key's
    cfg = _live_cfg()
    cfg.closing_volume = False
    got, same = plan_race(thin, cfg, 4000), plan_race(blind, cfg, 4000)
    assert [d.selection_id for d in got] == [1] and "no-volume model" in got[0].reason   # no GBP100 floor
    assert [(d.selection_id, d.edge, d.p_pool) for d in got] == [(d.selection_id, d.edge, d.p_pool) for d in same]


@pytest.mark.parametrize("topups,more", [(True, True), (False, False)])
def test_with_top_ups_off_a_horse_backed_already_is_not_backed_again(topups, more):
    # backed at 7.0 for GBP20 (to win 120 of 250); the price is now 6.0 and still value
    off = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)
    held = [{"marketId": "1.7", "selectionId": 1, "side": "BACK", "orderType": "LIMIT", "sizeMatched": 20.0,
             "averagePriceMatched": 7.0, "customerOrderRef": "ash10011"},
            {"marketId": "1.7", "selectionId": 1, "side": "LAY", "orderType": "MARKET_ON_CLOSE",
             "bspLiability": 120.0, "customerOrderRef": "ash10011x"}]
    c = _Client(orders=held)
    cfg = _live_cfg()
    cfg.topups = topups
    s, clock = _session(c, [_race("1.7", off, [6.0, 2.3, 3.6, 6.0], [3.5, 2.6, 4.2, 8.0], [500, 900, 800, 300])],
                        cfg)
    s.sync(s.x.orders(["1.7"]))
    s.step(clock())
    again = [p for m, p in c.calls if m == "placeOrders" and p["instructions"][0]["selectionId"] == 1
             and p["instructions"][0]["side"] == "BACK"]
    assert bool(again) is more


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


# ---------------------------------------------------------------- settling from Betfair's record

def _settled(mid, sid, side, profit, price, size, ref=None, bet="B"):
    """One bet as listClearedOrders reports it once the market is settled."""
    return {"marketId": mid, "selectionId": sid, "side": side, "betId": f"{bet}{sid}{side[0]}",
            "orderType": "MARKET_ON_CLOSE" if side == "LAY" else "LIMIT", "priceMatched": price,
            "sizeSettled": size, "profit": profit, "customerOrderRef": ref or f"ash1001{sid}{'x' if side == 'LAY' else ''}"}


def _closed(mid, winner, n=4):
    """The market after the race as the delayed feed shows it: results, and no SP."""
    return Book(market_id=mid, status="CLOSED", inplay=False, total_matched=None, runners={
        i: Quote(i, status="WINNER" if i == winner else "LOSER") for i in range(1, n + 1)})


def test_a_live_race_is_settled_from_betfairs_record_and_waits_for_it():
    """30 Sep: the delayed feed carries no SP, so the lays at SP were counted as nothing (+523.80 reported for three
    races whose winners were each laid back at the SP). A live race is settled from what Betfair paid on each bet."""
    off = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)
    c = _Client()
    s, clock = _session(c, [_race("1.7", off, [7.0, 2.3, 3.6, 6.0], [4.0, 2.6, 4.2, 8.0], [500, 900, 800, 300])])
    s.step(clock())                                                # backs horse 1 at 7.0, lays GBP249.96 at the SP
    assert s.positions[("1.7", 1)].hedge_liability == pytest.approx(249.96)
    s.x.data._books["1.7"] = _closed("1.7", winner=2)
    clock.t = off + timedelta(minutes=10)
    s.step(clock())                                                # the race is over, Betfair has not settled it
    assert not [r for r in s.ledger if r["event"] == "settle"] and not s.positions[("1.7", 1)].settled
    s.x.data._cleared = [_settled("1.7", 1, "BACK", -41.66, 7.0, 41.66),
                         _settled("1.7", 1, "LAY", 62.49, 5.0, 62.49),       # 249.96 laid at an SP of 5.0
                         _settled("1.7", 2, "BACK", 400.0, 3.0, 200.0, ref="the-owners-own-bet", bet="X")]
    s.step(clock())
    settle = [r for r in s.ledger if r["event"] == "settle"]
    assert len(settle) == 1 and settle[0]["result"] == "LOSER" and settle[0]["reason"] == SETTLED_BY_BETFAIR
    assert settle[0]["pnl_back"] == -41.66 and settle[0]["pnl_lay"] == 62.49 and settle[0]["pnl"] == 20.83
    assert settle[0]["bsp"] == 5.0 and settle[0]["clv"] == pytest.approx(7.0 / 5.0 - 1)
    commission = [r for r in s.ledger if r["event"] == "commission"][0]["commission"]
    assert commission == pytest.approx(0.02 * 20.83, abs=0.01)                # the account's rate
    assert s.summary()["settled_pnl"] == pytest.approx(20.83 - commission, abs=0.01)
    assert s.summary()["stake_weighted_clv"] == pytest.approx(0.4)
    s.step(clock())
    assert len([r for r in s.ledger if r["event"] == "settle"]) == 1     # settled once


def test_a_winner_laid_at_the_sp_nets_nothing_and_a_missing_lay_is_named():
    off = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)
    s, clock = _session(_Client(), [_race("1.7", off, [7.0, 2.3, 3.6, 6.0], [4.0, 2.6, 4.2, 8.0], [500, 900, 800, 300])])
    s.step(clock())
    s.x.data._books["1.7"] = _closed("1.7", winner=1)
    s.x.data._cleared = [_settled("1.7", 1, "BACK", 249.96, 7.0, 41.66), _settled("1.7", 1, "LAY", -249.96, 5.0, 62.49)]
    clock.t = off + timedelta(minutes=10)
    s.step(clock())
    settle = [r for r in s.ledger if r["event"] == "settle"][0]
    assert settle["result"] == "WINNER" and settle["pnl"] == 0.0 and settle["error"] == ""
    assert s.summary()["settled_pnl"] == 0.0
    # a back whose lay Betfair never settled stands alone, and the row says so
    s2, clock2 = _session(_Client(), [_race("1.7", off, [7.0, 2.3, 3.6, 6.0], [4.0, 2.6, 4.2, 8.0], [500, 900, 800, 300])])
    s2.step(clock2())
    s2.x.data._books["1.7"] = _closed("1.7", winner=1)
    s2.x.data._cleared = [_settled("1.7", 1, "BACK", 249.96, 7.0, 41.66)]
    clock2.t = off + timedelta(minutes=10)
    s2.step(clock2())
    settle = [r for r in s2.ledger if r["event"] == "settle"][0]
    assert settle["pnl"] == 249.96 and "no lay at SP settled" in settle["error"]


def test_a_back_reduced_for_a_non_runner_is_scored_at_the_price_betfair_settled():
    """1 Oct: a horse withdrawn after our bets made Betfair reduce the prices matched on the others, and the BSP is the
    smaller field's; set against it, the price as matched overstated the day's CLV by about half a point."""
    off = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)
    s, clock = _session(_Client(), [_race("1.7", off, [7.0, 2.3, 3.6, 6.0], [4.0, 2.6, 4.2, 8.0], [500, 900, 800, 300])])
    s.step(clock())                                                # backs horse 1 at 7.0
    s.x.data._books["1.7"] = _closed("1.7", winner=2)
    reduced = {**_settled("1.7", 1, "BACK", -41.66, 6.3, 41.66), "priceReduced": True}   # a 10% reduction factor
    s.x.data._cleared = [reduced, _settled("1.7", 1, "LAY", 56.23, 5.0, 56.23)]
    clock.t = off + timedelta(minutes=10)
    s.step(clock())
    settle = [r for r in s.ledger if r["event"] == "settle"][0]
    assert settle["avg_price"] == 6.3 and settle["clv"] == pytest.approx(6.3 / 5.0 - 1)
    assert "reduced the price for a non-runner" in settle["error"] and "matched at 7.00" in settle["error"]
    assert settle["pnl"] == pytest.approx(-41.66 + 56.23)                       # what Betfair paid, as before
    assert s.summary()["stake_weighted_clv"] == pytest.approx(0.26)


def test_the_owners_commission_stake_and_limits():
    """The owner, 2 Oct. Betfair charges the account 2% of each race's net winnings: on 1 Oct the settled result at 2%
    met the account's balance to the penny (+GBP188.19; 5% gave 178.62). The number of bets is not limited (a cap of
    250 stopped seven backs on 1 Oct). The owner, 7 Oct: GBP400 to win (GBP250 until then), GBP300 a bet, no limit on
    the day's stakes and no stop time: "We should just trade if we think the price is right" (test_the_owners_window_*)."""
    assert LIVE.commission == 0.02
    assert LIVE.limits.max_bets_per_day >= 1_000_000
    assert LIVE.limits.max_daily_turnover >= 1_000_000
    assert LIVE.limits.max_stake == 300.0 and LIVE.target == 400.0


def test_a_failed_read_of_the_settled_bets_never_stops_the_day():
    off = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)
    s, clock = _session(_Client(), [_race("1.7", off, [7.0, 2.3, 3.6, 6.0], [4.0, 2.6, 4.2, 8.0], [500, 900, 800, 300])])
    s.step(clock())
    s.x.data._books["1.7"] = _closed("1.7", winner=2)
    s.x.data.cleared_fail = True
    clock.t = off + timedelta(minutes=10)
    s.step(clock())                                                # no exception: the race waits
    assert s.state.stopped == "" and not s.positions[("1.7", 1)].settled


def _old_rows():
    """The first live day's ledger as the evening job finds it: a back, its lay at SP, and a settlement written from
    the delayed feed, which had no SP, so the lay counted as nothing (strings, as read from the CSV)."""
    base = {k: "" for k in LEDGER_FIELDS}
    common = {"mode": "live", "market_id": "1.263045597", "venue": "Kempton", "off": "19:00", "selection_id": "85907638",
              "runner": "Elan Dor"}
    return [
        {**base, **common, "ts": "2026-09-30T17:31:49Z", "event": "back", "asked": "8.62", "matched": "8.62",
         "avg_price": "30.0", "bet_id": "445070659999", "status": "SUCCESS"},
        {**base, **common, "ts": "2026-09-30T17:31:49Z", "event": "trade_out", "asked": "249.98", "matched": "8.62",
         "avg_price": "30.0", "bet_id": "445070660000", "status": "PENDING", "hedged": "True",
         "hedge_liability": "249.98"},
        {**base, **common, "ts": "2026-09-30T18:08:42Z", "event": "settle", "matched": "8.62", "avg_price": "30.0",
         "hedged": "True", "hedge_liability": "249.98", "result": "WINNER", "pnl_back": "249.98", "pnl_lay": "0.0",
         "pnl": "249.98"},
        {**base, **common, "ts": "2026-09-30T18:08:42Z", "event": "commission", "selection_id": "0", "runner": "",
         "commission": "12.5", "pnl": "-12.5"},
    ]


def _paid(mid="1.263045597"):
    """Betfair's record of that race: the winner's back paid 249.98, its lay at SP (21.0) cost 249.98. The bet ids
    are the ledger's (the references are left off: the ids alone must find them)."""
    return [{"marketId": mid, "selectionId": 85907638, "side": "BACK", "betId": "445070659999", "orderType": "LIMIT",
             "priceMatched": 30.0, "sizeSettled": 8.62, "profit": 249.98},
            {"marketId": mid, "selectionId": 85907638, "side": "LAY", "betId": "445070660000",
             "orderType": "MARKET_ON_CLOSE", "priceMatched": 21.0, "sizeSettled": 12.5, "profit": -249.98}]


def test_a_day_settled_without_its_lays_is_settled_again_from_betfairs_record():
    data = _Data(_Client(), markets=[], books={}, cleared=_paid())       # the closed market is off the catalogue
    s = Session(PaperExchange(data), pd.DataFrame(columns=["market_id", "selection_id", "predicted_bfsp"]),
                _live_cfg(), date(2026, 9, 30), clock=lambda: datetime(2026, 9, 30, 21, 45, tzinfo=UTC))
    s.mode = "live"
    s.restore(_old_rows(), [])
    assert [r["event"] for r in s.ledger] == ["back", "trade_out", "unsettled", "unsettled"]
    assert s.state.settled_pnl == 0.0 and not s.positions[("1.263045597", 85907638)].settled
    assert s.markets["1.263045597"].venue == "Kempton"                  # course and off kept from the ledger
    s.settle_all()
    settle = [r for r in s.ledger if r["event"] == "settle"]
    assert len(settle) == 1 and settle[0]["pnl"] == 0.0 and settle[0]["bsp"] == 21.0 and settle[0]["off"] == "19:00"
    assert settle[0]["clv"] == pytest.approx(30.0 / 21.0 - 1)
    assert s.summary()["settled_pnl"] == 0.0 and s.summary()["races_settled"] == 1
    # the next evening's job takes the day up again: settled from Betfair's record, it stays settled
    again = Session(PaperExchange(data), pd.DataFrame(columns=["market_id", "selection_id", "predicted_bfsp"]),
                    _live_cfg(), date(2026, 9, 30))
    again.mode = "live"
    again.restore([{k: ("" if v is None else v) for k, v in r.items()} for r in s.ledger], [])
    again.settle_all()
    assert len([r for r in again.ledger if r["event"] == "settle"]) == 1 and again.state.settled_pnl == 0.0


def test_the_evening_job_settles_the_live_day_from_betfairs_record(monkeypatch, tmp_path):
    import csv as _csv
    import auto_trade
    for k in ("BETFAIR_USERNAME", "BETFAIR_PASSWORD", "BETFAIR_APP_KEY"):
        monkeypatch.setenv(k, "x")
    ledger = tmp_path / "ledger_live_2026-09-30.csv"
    with ledger.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        w.writeheader()
        w.writerows(_old_rows())
    data = _Data(_Client(), markets=[], books={}, cleared=_paid())
    monkeypatch.setattr(auto_trade, "BetfairData", lambda recorder=None: data)
    sent = []
    monkeypatch.setattr(auto_trade, "publish", lambda *a, **k: None)
    monkeypatch.setattr(auto_trade, "running_totals", lambda mode: None)
    monkeypatch.setattr(auto_trade, "email", lambda summary, races=None, totals=None: sent.append((summary, races)))
    out = auto_trade.main(["--live", "--settle", "--date", "2026-09-30", "--out", str(tmp_path)])
    assert out["settled_pnl"] == 0.0 and out["races_settled"] == 1 and sent and sent[0][0]["settled_pnl"] == 0.0
    assert [(r["off"], r["venue"], r["winner"], r["pnl"]) for r in sent[0][1]] == [("19:00", "Kempton", "Elan Dor", 0.0)]
    with ledger.open() as f:
        events = [r["event"] for r in _csv.DictReader(f)]
    assert events == ["back", "trade_out", "unsettled", "unsettled", "settle", "commission"]


def test_the_settled_bets_are_read_page_by_page_and_only_read():
    class Pages:
        def __init__(self):
            self.calls = []

        def _api_call(self, method, params):
            self.calls.append((method, params))
            if params["fromRecord"] == 0:
                return {"clearedOrders": [{"betId": "1"}], "moreAvailable": True}
            return {"clearedOrders": [{"betId": "2"}], "moreAvailable": False}
    c = Pages()
    got = BetfairData(client=c).cleared(["1.5", "1.6"])
    assert [o["betId"] for o in got] == ["1", "2"]
    assert c.calls[0] == ("listClearedOrders", {"betStatus": "SETTLED", "marketIds": ["1.5", "1.6"],
                                                "fromRecord": 0, "recordCount": 1000})
    with pytest.raises(ValueError):
        BetfairData(client=c)._read("placeOrders", {})


# ---------------------------------------------------------------- the evening email

def test_the_email_shows_each_race_and_the_running_total(monkeypatch):
    import io as _io
    import json as _json
    import auto_trade
    rows = [
        {"event": "settle", "market_id": "1.1", "venue": "Kempton", "off": "19:00", "runner": "A", "matched": "10.0",
         "result": "LOSER", "clv": "0.5", "pnl": "8.0"},
        {"event": "settle", "market_id": "1.1", "venue": "Kempton", "off": "19:00", "runner": "B", "matched": "30.0",
         "result": "WINNER", "clv": "0.1", "pnl": "0.0"},
        {"event": "commission", "market_id": "1.1", "venue": "Kempton", "off": "19:00", "pnl": "-0.4"},
        {"event": "unsettled", "market_id": "1.2", "venue": "Kempton", "off": "19:30", "matched": "9", "pnl": "99"},
        {"event": "settle", "market_id": "1.3", "venue": "Kempton", "off": "20:00", "runner": "C", "matched": "5.0",
         "result": "LOSER", "clv": "", "pnl": "-5.0"},
        {"event": "commission", "market_id": "1.3", "venue": "Kempton", "off": "20:00", "pnl": "0"},
    ]
    races = auto_trade.race_lines(rows)
    assert [(r["off"], r["horses"], r["staked"], r["winner"], r["pnl"]) for r in races] == [
        ("19:00", 2, 40.0, "B", 7.6), ("20:00", 1, 5.0, "", -5.0)]
    assert races[0]["clv"] == pytest.approx((0.5 * 10 + 0.1 * 30) / 40) and races[1]["clv"] is None

    class S3:
        def list_objects_v2(self, **kw):
            return {"Contents": [{"Key": "trading/live/2026-09-30/summary.json"},
                                 {"Key": "trading/live/2026-09-30/ledger.csv"},
                                 {"Key": "trading/live/2026-10-01/summary.json"}], "IsTruncated": False}

        def get_object(self, Bucket, Key):
            day = Key.split("/")[2]
            got = {"2026-09-30": {"bets_matched": 38, "turnover": 548.99, "settled_pnl": 37.47},
                   "2026-10-01": {"bets_matched": 10, "turnover": 100.0, "settled_pnl": -2.5}}[day]
            return {"Body": _io.BytesIO(_json.dumps(got).encode())}
    tot = auto_trade.running_totals("live", s3=S3())
    assert tot == {"since": "2026-09-30", "days": 2, "bets": 48, "staked": pytest.approx(648.99),
                   "pnl": pytest.approx(34.97)}

    sent = []
    import ultra_betting.reporting.daily_report as dr
    monkeypatch.setattr(dr, "send_email_report", lambda html, subject=None: sent.append((html, subject)))
    summary = {"mode": "live", "day": "2026-10-01", "bets_matched": 10, "turnover": 100.0, "races_settled": 2,
               "settled_pnl": -2.5, "stake_weighted_clv": 0.2}
    auto_trade.email(summary, races=races, totals=tot)
    html, subject = sent[0]
    assert "19:00 Kempton" in html and "+7.60" in html and "+20.0%" in html and "Since 2026-09-30: 2 days" in html
    assert subject == "Ashcroft live trading 2026-10-01: settled -2.50 on GBP100.00 (10 bets), CLV +20.0%"
    auto_trade.email({**summary, "stake_weighted_clv": None})       # no CLV yet, no races: still sent
    assert sent[1][1].endswith("CLV n/a")


# ---------------------------------------------------------------- short of funds (1 Oct: the morning's exposure)

class _Short(_Client):
    """Betfair refusing every order as a whole for want of funds while `short`, as it did from 10:19 UK on 1 Oct."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.short = True

    def _api_call(self, method, params):
        if method == "placeOrders" and self.short:
            self.calls.append((method, params))
            return {"status": "FAILURE", "errorCode": "INSUFFICIENT_FUNDS",
                    "instructionReports": [{"status": "FAILURE", "errorCode": "ERROR_IN_ORDER"}]}
        return super()._api_call(method, params)


class _Funded(_Data):
    def __init__(self, *a, funds=None, **kw):
        super().__init__(*a, **kw)
        self.available, self.reads = funds, 0

    def funds(self):
        self.reads += 1
        return None if self.available is None else {"available": self.available, "exposure": -1490.84}


def test_a_refused_order_keeps_betfairs_own_reason():
    x = LiveExchange(_Data(_Short()))
    assert x.back("1.5", 7, 10.0, 20.0, "ash10017").error == "INSUFFICIENT_FUNDS (ERROR_IN_ORDER)"
    assert x.lay_at_bsp("1.5", 7, 250.0, "ash10017x").error == "INSUFFICIENT_FUNDS (ERROR_IN_ORDER)"
    why = LiveExchange._reason
    assert why({"errorCode": "PROCESSED_WITH_ERRORS"}, {"errorCode": "INVALID_BET_SIZE"}, "-") == \
        "INVALID_BET_SIZE (PROCESSED_WITH_ERRORS)"
    assert why({"errorCode": "MARKET_SUSPENDED"}, {}, "-") == "MARKET_SUSPENDED"
    assert why({}, {"errorCode": "ERROR_IN_ORDER"}, "-") == "ERROR_IN_ORDER"
    assert why({}, {}, "EXPIRED") == "EXPIRED"


def _orders(c):
    return [p["instructions"][0] for m, p in c.calls if m == "placeOrders"]


def test_short_of_funds_the_day_holds_and_trades_again_when_the_funds_are_back():
    off = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)
    c = _Short()
    market, book, preds = _race("1.7", off, [7.0, 2.3, 3.6, 6.0], [4.0, 2.6, 4.2, 8.0], [500, 900, 800, 300])
    data = _Funded(c, [market], {"1.7": book}, funds=1.0)
    clock = _Clock(datetime(2026, 10, 1, 8, 0, tzinfo=UTC))
    s = Session(LiveExchange(data), preds, _live_cfg(), DAY, clock=clock, sleep=lambda _: None)
    s.load_markets()
    s.step(clock())                                             # refused: the account's funds are short
    assert len(_orders(c)) == 1 and s.hold_until == clock() + timedelta(minutes=5)
    for minute in range(1, 5):                                  # held: nothing more is sent
        clock.t = datetime(2026, 10, 1, 8, minute, tzinfo=UTC)
        s.step(clock())
    assert len(_orders(c)) == 1
    clock.t = datetime(2026, 10, 1, 8, 5, tzinfo=UTC)          # GBP1 back: still short, held again
    s.step(clock())
    assert len(_orders(c)) == 1 and s.hold_until == clock() + timedelta(minutes=5)
    c.short, data.available = False, 30.0                       # a race run, a deposit: the funds are back
    clock.t = datetime(2026, 10, 1, 8, 10, tzinfo=UTC)
    s.step(clock())
    back, lay = _orders(c)[1:]
    assert back["limitOrder"]["size"] == 30.0                   # no more than the account has
    assert lay["marketOnCloseOrder"]["liability"] == pytest.approx(30.0 * 6.0, abs=0.01)
    clock.t = datetime(2026, 10, 1, 8, 11, tzinfo=UTC)          # the funds read are spent: held, nothing sent
    s.step(clock())
    assert len(_orders(c)) == 3 and s.hold_until is not None
    out = s.summary()
    assert out["backs_refused"] == {"INSUFFICIENT_FUNDS (ERROR_IN_ORDER)": 1} and out["held_for_funds"] == 2
    assert out["horses_held_for_funds"] == 1 and out["horses_held_never_backed"] == 0   # held, then backed
    again = Session(LiveExchange(data), preds, _live_cfg(), DAY, clock=clock, sleep=lambda _: None)
    again.restore(s.ledger, [market])                           # the settling run: it never held, the ledger did
    restored = again.summary()
    assert restored["held_for_funds"] == 0 and restored["horses_held_for_funds"] == 1
    skips = [r["error"] for r in s.ledger if r["event"] == "skip"]
    assert skips and set(skips) == {"held: the account's funds are short"}


def test_funds_too_low_for_a_layable_fill_hold_the_day_rather_than_skip_the_horse():
    off = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)
    c = _Short()
    market, book, preds = _race("1.7", off, [4.6, 2.3, 3.6, 6.0], [2.8, 2.6, 4.2, 8.0], [500, 900, 800, 300])
    data = _Funded(c, [market], {"1.7": book}, funds=2.5)      # GBP2.50 at 4.6 wins 9.00: not enough to lay
    clock = _Clock(datetime(2026, 10, 1, 8, 0, tzinfo=UTC))
    s = Session(LiveExchange(data), preds, _live_cfg(), DAY, clock=clock, sleep=lambda _: None)
    s.load_markets()
    s.step(clock())                                             # refused, held
    c.short = False
    clock.t = datetime(2026, 10, 1, 8, 5, tzinfo=UTC)
    s.step(clock())                                             # GBP2.50 back: too little to lay, so held again
    assert len(_orders(c)) == 1 and s.hold_until == clock() + timedelta(minutes=5)
    data.available = 500.0
    clock.t = datetime(2026, 10, 1, 8, 10, tzinfo=UTC)
    s.step(clock())
    assert [o["side"] for o in _orders(c)] == ["BACK", "BACK", "LAY"] and _orders(c)[1]["limitOrder"]["size"] == 69.44


def test_funds_that_cannot_be_read_hold_the_day_for_five_minutes_only():
    off = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)
    c = _Short()
    market, book, preds = _race("1.7", off, [7.0, 2.3, 3.6, 6.0], [4.0, 2.6, 4.2, 8.0], [500, 900, 800, 300])
    data = _Funded(c, [market], {"1.7": book}, funds=None)
    clock = _Clock(datetime(2026, 10, 1, 8, 0, tzinfo=UTC))
    s = Session(LiveExchange(data), preds, _live_cfg(), DAY, clock=clock, sleep=lambda _: None)
    s.load_markets()
    s.step(clock())
    c.short = False
    clock.t = datetime(2026, 10, 1, 8, 5, tzinfo=UTC)
    s.step(clock())                                             # tried again: Betfair's answer decides
    assert [o["side"] for o in _orders(c)] == ["BACK", "BACK", "LAY"] and s.hold_until is None
    assert _orders(c)[1]["limitOrder"]["size"] == 41.66        # the full stake: no funds were read to cap it


def test_a_back_too_small_to_lay_at_the_sp_is_never_sent_and_a_fill_is_always_one_it_can_lay():
    off = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)
    c = _Client()
    s, clock = _session(c, [_race("1.7", off, [4.6, 2.3, 3.6, 6.0], [2.8, 2.6, 4.2, 8.0], [500, 900, 800, 300])])
    s.step(clock())
    back = _orders(c)[0]["limitOrder"]
    assert back["size"] == 69.44 and back["minFillSize"] == 2.78   # GBP2.78 at 4.6 wins GBP10, the least laid at SP
    c2 = _Client()
    market, book, preds = _race("1.7", off, [4.6, 2.3, 3.6, 6.0], [2.8, 2.6, 4.2, 8.0], [500, 900, 800, 300])
    book.runners[1] = Quote(1, back=[(4.6, 2.27)], lay=[(4.8, 500.0)], traded=500)   # GBP2.27 on offer: wins 8.17
    s2, clock2 = _session(c2, [(market, book, preds)])
    s2.step(clock2())
    assert not _orders(c2)
    assert [r["error"] for r in s2.ledger if r["event"] == "skip"] == ["too small to lay at the SP (winnings under GBP10)"]


def test_the_live_summary_carries_the_accounts_funds_at_the_start_and_the_end(monkeypatch, tmp_path):
    import csv as _csv
    import auto_trade
    for k in ("BETFAIR_USERNAME", "BETFAIR_PASSWORD", "BETFAIR_APP_KEY"):
        monkeypatch.setenv(k, "x")
    ledger = tmp_path / "ledger_live_2026-09-30.csv"
    with ledger.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        w.writeheader()
        w.writerows(_old_rows())
    data = _Funded(_Client(), markets=[], books={}, cleared=_paid(), funds=1520.0)
    monkeypatch.setattr(auto_trade, "BetfairData", lambda recorder=None: data)
    monkeypatch.setattr(auto_trade, "publish", lambda *a, **k: None)
    monkeypatch.setattr(auto_trade, "running_totals", lambda mode: None)
    monkeypatch.setattr(auto_trade, "email", lambda summary, races=None, totals=None: None)
    out = auto_trade.main(["--live", "--settle", "--date", "2026-09-30", "--out", str(tmp_path)])
    assert out["funds_start"] == {"available": 1520.0, "exposure": -1490.84}
    assert out["funds_end"] == out["funds_start"] and data.reads == 2


def test_the_summary_scores_the_bets_entered_by_11_apart_from_the_later_ones():
    s = Session(LiveExchange(_Data(_Client())), pd.DataFrame(columns=["market_id", "selection_id", "predicted_bfsp"]),
                _live_cfg(), DAY)
    s.ledger += [
        {"event": "back", "market_id": "1.1", "selection_id": 1, "ts": "2026-10-01T08:30:00Z", "matched": 10.0},
        {"event": "back", "market_id": "1.1", "selection_id": 1, "ts": "2026-10-01T10:30:00Z", "matched": 5.0},
        {"event": "back", "market_id": "1.2", "selection_id": 2, "ts": "2026-10-01T13:00:00Z", "matched": 20.0},
        {"event": "settle", "market_id": "1.1", "selection_id": 1, "matched": 15.0, "clv": 0.10},
        {"event": "settle", "market_id": "1.2", "selection_id": 2, "matched": 20.0, "clv": -0.05},
    ]
    out = s.summary()
    # a horse is scored by its first back: 09:30 UK for the first (topped up at 11:30), 14:00 UK for the second
    assert out["staked_entered_by_11_uk"] == 15.0 and out["clv_entered_by_11_uk"] == 0.10
    assert out["staked_entered_after_11_uk"] == 20.0 and out["clv_entered_after_11_uk"] == -0.05
    assert out["stake_weighted_clv"] == round((15 * 0.10 - 20 * 0.05) / 35, 4)


def test_the_owners_window_runs_to_15_minutes_before_each_off():
    """The owner, 1 Oct, and again 7 Oct ("Let's not stop. We should just trade if we think the price is right"): every
    race from 08:00 UK until 15 minutes before its off, however late the day."""
    assert LIVE.trade_from == "08:00" and LIVE.trade_until == "21:30" and LIVE.stop_before_off == 15
    race = [7.0, 2.3, 3.6, 6.0], [4.0, 2.6, 4.2, 8.0], [500, 900, 800, 300]
    off = datetime(2026, 10, 1, 17, 30, tzinfo=UTC)                 # 18:30 UK
    c = _Client()
    s, clock = _session(c, [_race("1.7", off, *race)])
    clock.t = datetime(2026, 10, 1, 17, 14, tzinfo=UTC)             # 16 minutes before: still on
    s.step(clock())
    assert [o["side"] for o in _orders(c)] == ["BACK", "LAY"]
    c2 = _Client()
    s2, clock2 = _session(c2, [_race("1.7", off, *race)])
    clock2.t = datetime(2026, 10, 1, 17, 15, tzinfo=UTC)            # 15 minutes before: no more bets
    s2.step(clock2())
    assert not _orders(c2)
