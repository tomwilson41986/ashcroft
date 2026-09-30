"""The paper trader: read-only data, simulated fills, strategy, limits and a simulated day (trading/).

Hand-made markets and books only: these test mechanics, nothing is evaluated on them.
"""
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from trading import config as tc
from trading.exchange import (Book, BetfairData, Market, PaperExchange, Quote, parse_book, tick_down,
                              tick_up)
from trading.risk import DayState, allowed_stake
from trading.session import Session
from trading.strategy import RunnerView, plan_race

UTC = timezone.utc


# ---------------------------------------------------------------- exchange

def test_prices_round_onto_betfairs_ladder():
    assert tick_up(2.013) == 2.02 and tick_down(2.013) == 2.0      # 2 to 3 goes in steps of 0.02
    assert tick_up(1.994) == 1.995 or tick_up(1.994) == 2.0             # below 2 in steps of 0.01
    assert tick_down(1.994) == 1.99
    assert tick_up(3.01) == 3.05 and tick_down(3.04) == 3.0
    assert tick_up(10.1) == 10.5 and tick_down(10.4) == 10.0
    assert tick_up(1.0) == 1.01 and tick_down(1001) == 1000.0
    assert tick_up(4.0) == 4.0 and tick_down(4.0) == 4.0


def _book(mid="1.1", status="OPEN", inplay=False, matched=5000.0, runners=None, reconciled=False):
    return Book(market_id=mid, status=status, inplay=inplay, total_matched=matched, runners=runners or {},
                bsp_reconciled=reconciled)


class _Data:
    """A read-only exchange: markets and a book per market, both set by the test."""
    live = True

    def __init__(self, markets, books):
        self._markets, self._books = markets, books
        self.logged_in = False

    def login(self):
        self.logged_in = True

    def markets(self, start, end, countries=("GB", "IE")):
        return list(self._markets)

    def books(self, ids, with_sp=False):
        return {i: self._books[i] for i in ids if i in self._books}


def test_a_paper_back_fills_at_the_offered_prices_at_or_above_its_limit_and_the_rest_lapses():
    q = Quote(7, back=[(5.0, 10.0), (4.8, 20.0), (4.6, 50.0)], lay=[(5.1, 30.0)])
    x = PaperExchange(_Data([], {"1.1": _book(runners={7: q})}))
    x.books(["1.1"])
    f = x.back("1.1", 7, 4.8, 25.0, "r")
    assert f.status == "SUCCESS" and f.matched == 25.0
    assert f.avg_price == pytest.approx((10 * 5.0 + 15 * 4.8) / 25)
    f = x.back("1.1", 7, 5.2, 25.0, "r")                          # nothing offered at 5.2 or better
    assert f.status == "FAILURE" and f.matched == 0
    x._books["1.1"] = _book(inplay=True, runners={7: q})
    assert x.back("1.1", 7, 4.6, 5.0, "r").status == "FAILURE"    # never in play


class _Client:
    """Records every call made to Betfair and answers with canned replies."""
    app_key, session_token = "key", "token"

    def __init__(self, replies=None):
        self.calls, self.replies = [], replies or {}

    def login(self):
        self.calls.append(("login", None))

    def _api_call(self, method, params):
        self.calls.append((method, params))
        return self.replies.get(method, {})


def test_the_data_client_only_reads_markets_and_prices():
    d = BetfairData(client=_Client())
    for method in ("placeOrders", "cancelOrders", "replaceOrders", "updateOrders"):
        with pytest.raises(ValueError, match="only reads"):
            d._read(method, {})
    assert not any(hasattr(d, name) for name in ("back", "lay", "lay_at_bsp", "place", "cancel_all"))
    assert d.client.calls == []


def test_a_lost_session_logs_in_again_once():
    class Flaky(_Client):
        def _api_call(self, method, params):
            self.calls.append((method, params))
            if sum(1 for m, _ in self.calls if m == method) == 1:
                raise RuntimeError("Betfair API error: INVALID_SESSION_INFORMATION")
            return []
    c = Flaky()
    assert BetfairData(client=c).books(["1.1"]) == {}
    assert [m for m, _ in c.calls] == ["listMarketBook", "login", "listMarketBook"]


def test_a_book_is_read_from_betfairs_reply():
    b = parse_book({"marketId": "1.2", "status": "CLOSED", "inplay": False, "totalMatched": 1234.5,
                    "bspReconciled": True, "runners": [
                        {"selectionId": 5, "status": "WINNER", "lastPriceTraded": 3.1, "totalMatched": 100,
                         "ex": {"availableToBack": [{"price": 3.0, "size": 12}], "availableToLay": []},
                         "sp": {"actualSP": 3.25}},
                        {"selectionId": 6, "status": "REMOVED", "sp": {"actualSP": "NaN"}}]})
    assert b.status == "CLOSED" and b.runners[5].bsp == 3.25 and b.runners[5].best_back == (3.0, 12.0)
    assert b.runners[6].status == "REMOVED" and b.runners[6].bsp is None


# ---------------------------------------------------------------- strategy and limits

def _cfg(**kw):
    limits = kw.pop("limits", {})
    return tc.load_config(None, env={}, overrides={**kw}) if not limits else _with_limits(kw, limits)


def _with_limits(kw, limits):
    cfg = tc.load_config(None, env={}, overrides=kw)
    for k, v in limits.items():
        setattr(cfg.limits, k, v)
    return cfg


def _race(prices, backs, lays=None, sizes=None, status=None):
    n = len(prices)
    lays = lays or [b * 1.02 for b in backs]
    return [RunnerView(selection_id=i + 1, name=f"h{i + 1}", model_price=prices[i], back=backs[i],
                       back_size=(sizes or [100.0] * n)[i], lay=lays[i],
                       status=(status or ["ACTIVE"] * n)[i]) for i in range(n)]


def test_the_rule_backs_where_the_model_is_at_least_22_percent_shorter_than_the_best_back():
    # the model's book is exactly 1 over these runners, so its prices stand as given
    model = [2.5, 5.0, 5.0, 5.0]                        # 0.4 + 3 x 0.2: a book of exactly 1
    backs = [2.6, 6.5, 5.2, 5.5]                        # runner 2: ln(6.5 / 5.0) = 0.26; the rest < 0.2
    got = plan_race(_race(model, backs), _cfg(strategy="rule", staking="level", unit=5.0), bank=1000)
    assert [d.selection_id for d in got] == [2] and got[0].target == 5.0
    assert got[0].move == pytest.approx(np.log(6.5 / 5.0))


def test_to_win_stakes_win_the_target_and_kelly_stakes_scale_with_the_bank():
    model = [2.5, 5.0, 5.0, 5.0]
    backs = [2.6, 6.5, 5.2, 5.5]
    d = plan_race(_race(model, backs), _cfg(strategy="rule", staking="to_win", target=10.0), 1000)[0]
    assert d.target * (6.5 - 1) * 0.95 == pytest.approx(10.0)
    k1 = plan_race(_race(model, backs), _cfg(strategy="rule", staking="kelly"), 1000)
    k2 = plan_race(_race(model, backs), _cfg(strategy="rule", staking="kelly"), 2000)
    if k1:
        assert k2[0].target == pytest.approx(2 * k1[0].target)


def test_race_kelly_on_the_pooled_price_backs_overlays_and_can_add_a_hedging_underlay():
    model = [2.2, 4.0, 8.0, 12.0, 20.0]
    backs = [2.6, 4.4, 7.0, 11.0, 18.0]
    got = plan_race(_race(model, backs), _cfg(strategy="race_kelly", model_weight=1.0), 1000)
    assert got and all(d.target > 0 for d in got)
    assert {d.reason.split(":")[0] for d in got} == {"race Kelly"}
    none = plan_race(_race(model, [p * 0.9 for p in model]), _cfg(strategy="race_kelly", model_weight=1.0), 1000)
    assert none == []                                   # every price shorter than fair: nothing to back


def test_a_race_the_model_cannot_price_well_enough_is_left_alone():
    views = _race([2.5, 5.0, None, None], [2.6, 6.5, 5.2, 10.5])
    assert plan_race(views, _cfg(strategy="rule"), 1000) == []


def test_non_runners_are_dropped_and_the_model_renormalised_over_the_rest():
    model = [2.0, 4.0, 4.0]                              # a book of 1
    views = _race(model, [2.1, 4.2, 4.2], status=["ACTIVE", "ACTIVE", "REMOVED"])
    got = plan_race(views, _cfg(strategy="value", staking="level", model_weight=1.0, min_edge=0.02), 1000)
    # without the third runner the model has 1/2 and 1/4 renormalised to 2/3 and 1/3: both overlays at 2.1 and 4.2
    assert {d.selection_id for d in got} == {1, 2}
    assert got[0].p_model + got[1].p_model == pytest.approx(1.0)


def test_a_market_without_its_matched_volume_is_not_held_to_the_volume_floor():
    """A delayed application key may leave totalMatched out: unknown is not thin."""
    from trading.config import Limits
    from trading.risk import DayState, allowed_stake
    b = parse_book({"marketId": "1.9", "status": "OPEN", "inplay": False, "runners": []})
    assert b.total_matched is None
    assert parse_book({"marketId": "1.9", "totalMatched": 0, "runners": []}).total_matched == 0.0
    lim = Limits()
    stake, why = allowed_stake(10.0, 5.0, "1.9", None, 100.0, DayState(), lim)
    assert (stake, why) == (10.0, "")
    stake, why = allowed_stake(10.0, 5.0, "1.9", 0.0, 100.0, DayState(), lim)
    assert stake == 0.0 and "below" in why
    stake, _ = allowed_stake(10.0, 5.0, "1.9", None, 6.0, DayState(), lim)
    assert stake == 3.0                              # half the size on offer still caps it


def test_the_limits_cap_every_stake_and_the_stop_loss_ends_the_day():
    cfg = _cfg(limits={"max_stake": 20.0, "max_race_stake": 30.0, "liquidity_share": 0.5})
    st = DayState()
    assert allowed_stake(50, 5.0, "m", 5000, 100, st, cfg.limits) == (20.0, "")
    assert allowed_stake(50, 5.0, "m", 5000, 10, st, cfg.limits)[0] == 5.0        # half the offered size
    st.record("m", 25.0)
    assert allowed_stake(50, 5.0, "m", 5000, 100, st, cfg.limits)[0] == 5.0       # the race's cap
    assert allowed_stake(50, 5.0, "m", 100, 100, st, cfg.limits)[0] == 0          # a thin market
    assert allowed_stake(50, 50.0, "n", 5000, 100, st, cfg.limits)[0] == 0        # price outside the range
    st.settled_pnl = -cfg.limits.daily_stop_loss
    assert allowed_stake(5, 5.0, "n", 5000, 100, st, cfg.limits)[0] == 0 and "stop-loss" in st.stopped


# ---------------------------------------------------------------- settings

def test_settings_come_from_the_file_then_the_environment_then_the_flags(tmp_path):
    f = tmp_path / "c.json"
    f.write_text('{"strategy": "value", "limits": {"max_stake": 10.0}}')
    cfg = tc.load_config(str(f), env={"TRADING_STAKING": "level", "TRADING_MAX_STAKE": "8"},
                         overrides={"trade_out": True})
    assert (cfg.strategy, cfg.staking, cfg.limits.max_stake, cfg.trade_out) == ("value", "level", 8.0, True)
    with pytest.raises(ValueError):
        tc.load_config(None, env={"TRADING_STRATEGY": "martingale"})


# ---------------------------------------------------------------- a simulated day

class _Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def _day(trade_out=False, kill=None, trade_out_at="before_off"):
    off = datetime(2026, 9, 28, 13, 30, tzinfo=UTC)                  # 14:30 UK
    market = Market("1.9", "York", "GB", off, runners={1: "Alpha", 2: "Beta", 3: "Gamma"})
    # the model's book here is 0.855, so its prices for the three are 1.80, 4.27 and 4.79: only Beta's
    # best back (7.0) is at least 22% longer (ln(7.0 / 4.27) = 0.49); Alpha's ln(2.0 / 1.80) = 0.11
    open_book = _book("1.9", runners={1: Quote(1, back=[(2.0, 200.0)], lay=[(2.04, 200.0)]),
                                      2: Quote(2, back=[(7.0, 60.0)], lay=[(7.2, 60.0)]),
                                      3: Quote(3, back=[(5.0, 80.0)], lay=[(5.1, 80.0)])})
    data = _Data([market], {"1.9": open_book})
    preds = pd.DataFrame({"market_id": ["1.9"] * 3, "selection_id": [1, 2, 3], "predicted_bfsp": [2.1, 5.0, 5.6]})
    cfg = _cfg(strategy="rule", staking="to_win", target=20.0, trade_from="08:00", trade_until="21:00",
               trade_out=trade_out, trade_out_at=trade_out_at)
    clock = _Clock(datetime(2026, 9, 28, 9, 0, tzinfo=UTC))
    s = Session(PaperExchange(data, bank=1000), preds, cfg, date(2026, 9, 28), clock=clock, sleep=lambda _: None,
                kill=kill)
    s.load_markets()
    return s, data, clock, off


def test_a_day_backs_the_rule_once_and_settles_the_winner_after_the_off():
    s, data, clock, off = _day()
    s.step(clock())
    s.step(clock())                                                  # the second read adds nothing
    backs = [r for r in s.ledger if r["event"] == "back"]
    assert len(backs) == 1 and backs[0]["selection_id"] == 2 and backs[0]["matched"] > 0
    stake = backs[0]["matched"]
    assert stake * (7.0 - 1) * 0.95 == pytest.approx(20.0, abs=0.1)       # to win £20, rounded down to the penny
    data._books["1.9"] = _book("1.9", status="CLOSED", runners={
        1: Quote(1, status="LOSER", bsp=2.0), 2: Quote(2, status="WINNER", bsp=5.5), 3: Quote(3, status="LOSER", bsp=6.0)})
    clock.t = off + timedelta(minutes=10)
    s.step(clock())
    settle = [r for r in s.ledger if r["event"] == "settle"]
    assert len(settle) == 1 and settle[0]["result"] == "WINNER"
    assert settle[0]["pnl"] == pytest.approx(stake * 6.0, abs=0.02)
    assert settle[0]["clv"] == pytest.approx(7.0 / 5.5 - 1)
    assert s.state.settled_pnl == pytest.approx(stake * 6.0 * 0.95, abs=0.02)   # after commission
    assert s.summary()["bets_matched"] == 1


def test_trading_out_lays_the_winnings_at_bsp_so_a_loser_keeps_the_move():
    s, data, clock, off = _day(trade_out=True)
    s.step(clock())
    stake = [r for r in s.ledger if r["event"] == "back"][0]["matched"]
    clock.t = off - timedelta(minutes=2)
    s.step(clock())
    out = [r for r in s.ledger if r["event"] == "trade_out"]
    assert len(out) == 1 and out[0]["hedge_liability"] == pytest.approx(round(stake * 6.0, 2))
    data._books["1.9"] = _book("1.9", status="CLOSED", runners={
        1: Quote(1, status="WINNER", bsp=2.0), 2: Quote(2, status="LOSER", bsp=5.5), 3: Quote(3, status="LOSER", bsp=6.0)})
    clock.t = off + timedelta(minutes=10)
    s.step(clock())
    settle = [r for r in s.ledger if r["event"] == "settle"][0]
    # lost the back stake, won the lay: liability / (bsp - 1)
    assert settle["pnl"] == pytest.approx(-stake + stake * 6.0 / 4.5, abs=0.02)
    assert settle["pnl"] > 0


def test_closing_on_the_fill_lays_at_bsp_at_once_and_the_day_settles_from_its_ledger():
    s, data, clock, off = _day(trade_out=True, trade_out_at="fill")
    s.step(clock())
    stake = [r for r in s.ledger if r["event"] == "back"][0]["matched"]
    out = [r for r in s.ledger if r["event"] == "trade_out"]
    assert len(out) == 1 and out[0]["hedge_liability"] == pytest.approx(round(stake * 6.0, 2))
    assert out[0]["minutes_to_off"] > 60                             # laid in the morning, settled at the off
    # the morning job ends here; the evening one takes the day up from the ledger and settles it
    rows = [{k: ("" if v is None else v) for k, v in r.items()} for r in s.ledger]
    data._books["1.9"] = _book("1.9", status="CLOSED", runners={
        1: Quote(1, status="LOSER", bsp=2.0), 2: Quote(2, status="WINNER", bsp=5.5), 3: Quote(3, status="LOSER", bsp=6.0)})
    later = Session(PaperExchange(data), pd.DataFrame(columns=["market_id", "selection_id", "predicted_bfsp"]),
                    s.cfg, s.day, clock=lambda: off + timedelta(hours=3))
    later.restore(rows, data.markets(None, None))
    later.settle_all()
    settle = [r for r in later.ledger if r["event"] == "settle"]
    assert len(settle) == 1 and settle[0]["result"] == "WINNER"
    # won the back, paid the lay's liability: the winnings are laid off exactly
    assert settle[0]["pnl"] == pytest.approx(0.0, abs=0.02)
    later.settle_all()                                               # a second settle adds nothing
    assert len([r for r in later.ledger if r["event"] == "settle"]) == 1


def test_paper_waits_for_the_sp_before_settling_a_runner_laid_at_it():
    """A feed without the SP (Betfair's delayed key) must never count a lay at SP as nothing: the race waits."""
    s, data, clock, off = _day(trade_out=True, trade_out_at="fill")
    s.step(clock())
    data._books["1.9"] = _book("1.9", status="CLOSED", runners={
        1: Quote(1, status="LOSER"), 2: Quote(2, status="WINNER"), 3: Quote(3, status="LOSER")})
    clock.t = off + timedelta(minutes=10)
    s.step(clock())
    assert not [r for r in s.ledger if r["event"] == "settle"] and s.state.settled_pnl == 0.0
    data._books["1.9"] = _book("1.9", status="CLOSED", runners={
        1: Quote(1, status="LOSER", bsp=2.0), 2: Quote(2, status="WINNER", bsp=5.5), 3: Quote(3, status="LOSER", bsp=6.0)})
    s.step(clock())
    settle = [r for r in s.ledger if r["event"] == "settle"]
    assert len(settle) == 1 and settle[0]["result"] == "WINNER" and settle[0]["pnl"] == pytest.approx(0.0, abs=0.02)


def test_the_kill_switch_stops_new_bets():
    s, data, clock, off = _day(kill=lambda: True)
    s.step(clock())
    assert not [r for r in s.ledger if r["event"] == "back"] and s.state.stopped == "kill switch"


def test_outside_the_window_nothing_is_bet():
    s, data, clock, off = _day()
    s.cfg.trade_until = "09:30"
    clock.t = datetime(2026, 9, 28, 9, 0, tzinfo=UTC)                # 10:00 UK: after the window
    s.step(clock())
    assert not [r for r in s.ledger if r["event"] == "back"]


def test_paper_never_sends_an_order_to_betfair():
    c = _Client({"listMarketCatalogue": [], "listMarketBook": []})
    x = PaperExchange(BetfairData(client=c))
    x.books(["1.1"])
    x.back("1.1", 1, 3.0, 5.0, "r")
    x.lay_at_bsp("1.1", 1, 5.0, "r")
    x.cancel_all()
    assert {m for m, _ in c.calls} <= {"listMarketBook", "listMarketCatalogue", "login"}


def test_the_job_stands_down_without_credentials_and_settles_nothing_untraded(monkeypatch, tmp_path):
    """Until the owner stores the Betfair secrets the scheduled job says why and exits cleanly,
    and an evening with no morning ledger has nothing to settle."""
    import auto_trade

    for k in ("BETFAIR_USERNAME", "BETFAIR_PASSWORD", "BETFAIR_APP_KEY"):
        monkeypatch.delenv(k, raising=False)
    out = auto_trade.main(["--date", "2026-09-28", "--out", str(tmp_path)])
    assert out["traded"] is False and "BETFAIR_APP_KEY" in out["reason"]

    monkeypatch.setattr(auto_trade, "_s3", lambda: (_ for _ in ()).throw(RuntimeError("no bucket here")))
    assert auto_trade.read_ledger(date(2026, 9, 28), tmp_path / "absent.csv") is None
