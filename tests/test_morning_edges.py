"""The morning edges paper trader (the owner, 4 Nov 2026: "Ensure it doesn't bet before 10:00 so it doesnt take prices
from the live trader"): no order before 10:00 UK, the live trader's horses left alone, to-win staking and its caps,
the ladder walk, settlement with reduction factors, the card view, and no order-placing code. Fixtures only."""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research.arms import core
from research.paper import morning_edges as ME
from research.paper import morning_features as MF
from trading.exchange import Book, Market, Quote

DAY = date(2026, 11, 5)
MID = "1.990"
START = datetime(2026, 11, 5, 13, 30, tzinfo=timezone.utc)        # a 13:30 off (UK is UTC in November)
A, B, C = 31, 32, 33


def _book(a_back=((6.0, 40.0), (5.8, 100.0)), b_back=((2.0, 500.0),), c_back=((11.0, 50.0),)):
    return Book(MID, "OPEN", False, 5000.0, {A: Quote(A, back=list(a_back)), B: Quote(B, back=list(b_back)),
                                              C: Quote(C, back=list(c_back))})


def _day(first_poll_utc=datetime(2026, 11, 5, 6, 0, tzinfo=timezone.utc), minutes=330):
    m = Market(MID, "Kempton", "GB", START, name="7f Hcap", runners={A: "Alpha", B: "Bravo", C: "Charlie"})
    polls = [(first_poll_utc + timedelta(minutes=i), _book()) for i in range(minutes)]
    settled = {A: (5.5, "WINNER"), B: (2.2, "LOSER"), C: (12.0, "LOSER")}
    polls.append((datetime(2026, 11, 5, 23, 0, tzinfo=timezone.utc),
                  Book(MID, "CLOSED", False, None, {s: Quote(s, status=st, bsp=b) for s, (b, st) in settled.items()},
                       True)))
    return SimpleNamespace(day=DAY, kind="win", markets=[m], polls={MID: polls},
                           poll_times={MID: [t for t, _ in polls]}, starts={MID: START}, real_fills={},
                           preds=ME.NO_PREDS.copy(), settled={(MID, s): v for s, v in settled.items()},
                           removed=pd.DataFrame(columns=["market_id", "selection_id", "af", "rt"]), places={},
                           ledger=pd.DataFrame())


def _run(D, sel_ids, live=frozenset(), t0=datetime(2026, 11, 5, 5, 0, tzinfo=timezone.utc)):
    sel = pd.DataFrame({"market_id": [MID] * len(sel_ids), "selection_id": sel_ids, "label": ["x"] * len(sel_ids),
                        "race_time": ["1.30"] * len(sel_ids)})
    specs = [("qr_am", ME.MorningEdgeSession, core.make_cfg(), {"selections": sel, "live_backed": set(live)},
              ME.BANK, False)]
    out = core.run_agents(D, specs, t0, t_end=datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc))
    return out["qr_am"][0]


def test_no_order_before_ten_uk_even_with_books_all_morning():
    led = _run(_day(), [A, B, C])
    backs = led[led.event == "back"]
    assert len(backs) == 3
    ts = pd.to_datetime(led.ts, utc=True)
    assert (ts.dt.tz_convert("Europe/London").dt.time >= ME.ENTRY_UK).all()
    assert ts.min() == pd.Timestamp("2026-11-05T10:00:00Z")


def test_ten_uk_is_nine_utc_in_summer():
    assert ME.entry_utc(date(2026, 7, 1)) == datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc)
    assert ME.entry_utc(date(2026, 11, 5)) == datetime(2026, 11, 5, 10, 0, tzinfo=timezone.utc)


def test_the_session_does_nothing_at_any_minute_before_ten():
    D = _day()
    calls = []
    x = SimpleNamespace(data=SimpleNamespace(D=D), books=lambda ids: calls.append(("books", ids)) or {},
                        back=lambda *a, **k: calls.append(("back", a)))
    sel = pd.DataFrame({"market_id": [MID], "selection_id": [A], "label": ["Alpha"]})
    s = ME.MorningEdgeSession(x, None, None, DAY, policy={"selections": sel})
    s.load_markets()
    t = datetime(2026, 11, 5, 5, 0, tzinfo=timezone.utc)
    while t < ME.entry_utc(DAY):
        s.step(t)
        t += timedelta(minutes=1)
    assert calls == [] and s.ledger == []


def test_the_live_traders_horse_is_left_alone():
    led = _run(_day(), [A, B], live={(MID, A)})
    assert led.set_index("selection_id").loc[A, "event"] == "skipped_live"
    assert led.set_index("selection_id").loc[A, "matched"] == 0
    assert led.set_index("selection_id").loc[B, "event"] == "back"


def test_to_win_staking_walks_the_ladder_and_caps_the_stake():
    led = _run(_day(), [A, B]).set_index("selection_id")
    # Alpha: best 6.0, to win 100 -> 20.00; 6.0 offers 40 -> all at 6.0
    assert led.loc[A, "size"] == pytest.approx(20.0) and led.loc[A, "price"] == pytest.approx(6.0)
    # Bravo: best 2.0, to win 100 -> 100 (under the 200 cap)
    assert led.loc[B, "size"] == pytest.approx(100.0)
    D = _day()
    for t, b in D.polls[MID][:-1]:
        b.runners[A] = Quote(A, back=[(6.0, 10.0), (5.8, 100.0), (5.0, 500.0)])
    led = _run(D, [A]).set_index("selection_id")
    # 10 at 6.0 then 10 at 5.8 (5.0 is under 95% of 6.0): average 5.9, all 20 filled
    assert led.loc[A, "matched"] == pytest.approx(20.0) and led.loc[A, "price"] == pytest.approx(5.9, abs=1e-3)


def test_a_short_price_hits_the_200_cap():
    D = _day()
    for t, b in D.polls[MID][:-1]:
        b.runners[B] = Quote(B, back=[(1.3, 5000.0)])
    led = _run(D, [B]).set_index("selection_id")
    assert led.loc[B, "size"] == pytest.approx(ME.MAX_STAKE)


def test_settle_holds_to_the_result_and_cuts_a_winner_for_a_late_non_runner():
    D = _day()
    bets = pd.DataFrame({"ts": ["2026-11-05T10:00:00+00:00"] * 2, "market_id": [MID, MID], "selection_id": [A, B],
                         "price": [6.0, 2.0], "matched": [20.0, 100.0]})
    st = ME.settle(bets, D).set_index("selection_id")
    assert st.loc[A, "pnl"] == pytest.approx(100.0) and st.loc[B, "pnl"] == pytest.approx(-100.0)
    D.removed = pd.DataFrame({"market_id": [MID], "selection_id": [99], "af": [20.0],
                              "rt": [pd.Timestamp("2026-11-05T11:00:00Z")]})
    st = ME.settle(bets, D).set_index("selection_id")
    assert st.loc[A, "pnl"] == pytest.approx(20.0 * 5.0 * 0.8)


def test_card_view_blanks_the_days_results_and_prices_only():
    df = pd.DataFrame({"race_date": ["2026-11-04", "2026-11-05", "2026-11-06"], "bsp": [3.0, 4.0, 5.0],
                       "won": [1.0, 0.0, 1.0], "morningwap": [3.2, 4.4, 5.1], "days_since_lr": [10, 3, 2]})
    cv = MF.card_view(df, "2026-11-05")
    assert len(cv) == 2                                            # the next day is not there at all
    assert cv.loc[0, "bsp"] == 3.0 and cv.loc[0, "won"] == 1.0
    assert np.isnan(cv.loc[1, "bsp"]) and np.isnan(cv.loc[1, "won"]) and np.isnan(cv.loc[1, "morningwap"])
    assert cv.loc[1, "days_since_lr"] == 3                         # a card fact stays


def test_no_order_placing_code_is_reachable():
    src = Path(ME.__file__).read_text() + Path(MF.__file__).read_text()
    for word in ("LiveExchange", "placeOrders", "place_orders", "TRADING_LIVE", "betfair_client"):
        assert word not in src


def test_the_gate_reads_running_until_a_thousand_bets_and_sixty_days():
    rows = []
    for i in range(10):
        d = (DAY + timedelta(i)).isoformat()
        for mk in ("win", "place"):
            rows.append({"day": d, "forward": True, "strategy": "qr_am", "market": mk, "bets": 10, "wins": 3,
                         "expected": 2.0, "staked": 100.0, "wanted": 100.0, "pnl_5pc": 50.0, "skipped_live": 0,
                         "wap_gap_median": 0.0})
    g = ME.gate(pd.DataFrame(rows)).set_index("strategy")
    assert g.loc["qr_am", "gate"] == "running"


def test_the_workflow_has_no_betfair_key_and_no_live_switch():
    wf = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "paper-morning-edges.yml").read_text()
    for word in ("BETFAIR", "TRADING_LIVE", "--live", "self-hosted", "PAPER_RUNNER", "LIVE_RUNNER"):
        assert word not in wf
    assert "runs-on: ubuntu-latest" in wf
