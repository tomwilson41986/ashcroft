"""The horse paper desk (the owner, 4 Nov 2026: paper trade backs, lays and both sides of the line): the two-sided
quotes' fills on recorded books without a traded ladder, their score at the SP, and the desk's frozen strategies and
gate. Fixtures only."""

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research.arms import core, quotes
from research.paper import horses
from trading.exchange import Book, Market, Quote

MID = "1.950"
START = datetime(2026, 11, 5, 13, 0, tzinfo=timezone.utc)
A, B = 21, 22


def _book(b_lt=2.0, b_traded=1000.0, a_lt=4.0, a_traded=500.0):
    return Book(MID, "OPEN", False, 9000.0, {
        A: Quote(A, back=[(4.0, 50.0)], lay=[(4.2, 50.0)], last_traded=a_lt, traded=a_traded),
        B: Quote(B, back=[(2.0, 300.0)], lay=[(2.02, 300.0)], last_traded=b_lt, traded=b_traded)})


def _day(polls_spec):
    m = Market(MID, "Kempton", "GB", START, name="1m Hcap", runners={A: "Alpha", B: "Bravo"})
    t0 = datetime(2026, 11, 5, 6, 0, tzinfo=timezone.utc)
    polls = [(t0 + timedelta(minutes=i), _book(**kw)) for i, kw in enumerate(polls_spec)]
    settled = {A: (4.0, "WINNER"), B: (2.5, "LOSER")}
    polls.append((datetime(2026, 11, 5, 23, 0, tzinfo=timezone.utc),
                  Book(MID, "CLOSED", False, None, {s: Quote(s, status=st, bsp=b) for s, (b, st) in settled.items()},
                       True)))
    preds = pd.DataFrame({"market_id": [MID] * 2, "selection_id": [A, B], "predicted_bfsp": [4.0, 2.4]})
    return SimpleNamespace(day=date(2026, 11, 5), kind="win", markets=[m], polls={MID: polls},
                           poll_times={MID: [t for t, _ in polls]}, starts={MID: START}, real_fills={}, preds=preds,
                           settled={(MID, s): v for s, v in settled.items()},
                           removed=pd.DataFrame(columns=["market_id", "selection_id", "af", "rt"]), places={},
                           ledger=pd.DataFrame())


def _quote(D, **policy):
    cfg = core.make_cfg(clv_draws=2000)
    core.set_plan_cfg(core.make_cfg(clv_draws=2000), reset=True)
    led, s, x = core.run_session(D, quotes.QuoteSession, cfg, {"fair": "mid", "margin": 0.02, **policy, "_D": D},
                                 datetime(2026, 11, 5, 5, 58, tzinfo=timezone.utc))
    return s, s.fills_frame()


def test_a_resting_back_fills_when_the_market_trades_through_it():
    # the mid of 2.00 / 2.02 is about 2.01: the back rests at 2.06, the lay at 1.96 (neither crosses the book)
    D = _day([{}, {"b_lt": 2.08, "b_traded": 1400.0}, {}])
    s, f = _quote(D)
    b = f[(f.selection_id == B)]
    assert list(b.side) == ["back"] and b.price.iloc[0] == 2.06 and b.matched.iloc[0] == 10.0
    assert b.how.iloc[0] == "traded through"
    r = quotes.score_quotes(f, D, "quotes_mid2")
    lose = -10 + 10 * 2.06 / 2.5                                   # the back loses, the SP lay of 10 x 2.06 / 2.5 wins
    assert r["locked"] == pytest.approx(round(10 * (2.06 / 2.5 - 1), 2), abs=0.02)
    assert len(f[f.selection_id == A]) == 0                         # Alpha never traded: nothing on it
    assert r["result"] == pytest.approx(lose, abs=0.01)             # a loss on the day: no commission


def test_a_lay_deeper_than_the_book_shows_fills_only_when_the_market_trades_below_it():
    D = _day([{}, {"b_lt": 1.98, "b_traded": 1500.0}, {"b_lt": 1.94, "b_traded": 1800.0}, {}])
    s, f = _quote(D)
    lay = f[(f.selection_id == B) & (f.side == "lay")]
    assert len(lay) == 1 and lay.price.iloc[0] == 1.96 and lay.how.iloc[0] == "traded through"
    assert lay.ts.iloc[0] == "2026-11-05T06:02:00Z"                # not at 1.98: its queue was not shown, so unknown


def test_a_quote_at_our_price_counts_half_the_money_beyond_the_queue_and_nothing_when_nothing_trades():
    D = _day([{}, {}, {}])
    s, f = _quote(D)
    assert f.empty or not len(f[f.selection_id == B])               # nothing traded: nothing filled
    o = {"price": 2.06, "left": 10.0, "queue": 0.0, "traded": 1000.0}
    s.resting[(MID, B, "back")] = o
    m = D.markets[0]
    s._advance(datetime(2026, 11, 5, 7, 0, tzinfo=timezone.utc), m, B,
               Quote(B, back=[(2.0, 300.0)], lay=[(2.06, 5.0)], last_traded=2.06, traded=1012.0))
    rows = [r for r in s.quote_rows if r["selection_id"] == B and r["how"] == "at our price"]
    assert rows and rows[-1]["matched"] == 6.0                     # half of the 12 traded, no queue ahead


def test_the_score_is_the_hedge_at_the_sp_after_the_cuts():
    D = _day([{}])
    f = pd.DataFrame([{"ts": "2026-11-05T06:00:00Z", "market_id": MID, "selection_id": A, "side": "lay",
                       "price": 3.8, "matched": 20.0},
                      {"ts": "2026-11-05T06:00:00Z", "market_id": MID, "selection_id": A, "side": "back",
                       "price": 4.3, "matched": 20.0}])
    r = quotes.score_quotes(f, D, "q")
    assert r["two_sided"] == 1 and r["runners"] == 1
    assert r["locked"] == pytest.approx(20 * (1 - 3.8 / 4.0) + 20 * (4.3 / 4.0 - 1), abs=0.01)
    # Alpha won: the lay pays 20 x 2.8, its SP back of 19 wins 57; the back wins 66, its SP lay of 21.5 loses 64.5
    assert r["result"] == pytest.approx(0.98 * (20 * (1 - 3.8 / 4) + 20 * (4.3 / 4 - 1)), abs=0.01)
    D.removed = pd.DataFrame([{"market_id": MID, "selection_id": B, "af": 10.0, "rt": pd.Timestamp("2026-11-05T09:00Z")}])
    r2 = quotes.score_quotes(f, D, "q")                             # a 10% cut: the lay's liability and the back's win
    assert r2["result_reduced"] == pytest.approx(r["result"] + 0.98 * (20 * (3.8 - 3.42) - 20 * (4.3 - 3.87)), abs=0.02)


def test_the_desk_s_strategies_and_gate_are_the_frozen_ones():
    assert set(horses.STRATEGIES) == {"live_rule", "backs_floor50", "backs_win1000", "lays", "quotes_model3",
                                      "quotes_mid2"}
    assert horses.STRATEGIES["backs_floor50"][1] == {"floor_long": (21.0, 50.0)}
    assert horses.STRATEGIES["lays"][1] == {"lay_bar": 0.03, "max_lay_price": 20.0}
    assert horses.FORWARD_FROM == date(2026, 11, 5) and horses.GATE_DAYS == 20
    rows = []
    rng = np.random.default_rng(1)
    for i in range(20):
        d = (date(2026, 11, 5) + timedelta(i)).isoformat()
        rows += [{"day": d, "arm": "live_rule", "forward": True, "result_reduced": 200.0},
                 {"day": d, "arm": "backs_win1000", "forward": True, "result_reduced": 230.0 + rng.normal(0, 5)},
                 {"day": d, "arm": "lays", "forward": True, "result_reduced": -10.0 + rng.normal(0, 5)}]
    g = horses.gate(pd.DataFrame(rows)).set_index("strategy")
    assert g.loc["backs_win1000", "gate"] == "PASSED" and g.loc["lays", "gate"] == "failed"
    assert horses.gate(pd.DataFrame(rows[:9])).set_index("strategy").loc["lays", "gate"] == "running"


def test_the_desk_runs_on_github_s_runners_and_never_reaches_betfair():
    import yaml
    from pathlib import Path
    text = (Path(__file__).resolve().parent.parent / ".github/workflows/paper-horses.yml").read_text()
    wf = yaml.safe_load(text)
    job = wf["jobs"]["desk"]
    assert job["runs-on"] == "ubuntu-latest"
    assert "BETFAIR" not in text and "TRADING_LIVE" not in text and "auto_trade" not in text
    src = (Path(__file__).resolve().parent.parent / "research/paper/horses.py").read_text()
    src += (Path(__file__).resolve().parent.parent / "research/arms/quotes.py").read_text()
    assert "placeOrders" not in src and "LiveExchange" not in src and "TRADING_LIVE" not in src
