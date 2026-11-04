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


def _book(b_lt=2.0, b_traded=1000.0, a_lt=4.0, a_traded=500.0, b_lay=2.02):
    return Book(MID, "OPEN", False, 9000.0, {
        A: Quote(A, back=[(4.0, 50.0)], lay=[(4.2, 50.0)], last_traded=a_lt, traded=a_traded),
        B: Quote(B, back=[(2.0, 300.0)], lay=[(b_lay, 300.0)], last_traded=b_lt, traded=b_traded)})


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


# --- the agents on one book: a price one takes is gone for the others (the owner, 4 Nov 2026) ----------------------

def _thin_day(size=50.0):
    """One race, Alpha long in the market with GBP``size`` offered at 10.0; the rule wants it (to win GBP400:
    GBP44.44 at 10)."""
    m = Market(MID, "Kempton", "GB", START, name="1m Hcap", runners={A: "Alpha", B: "Bravo", 23: "Charlie"})
    t0 = datetime(2026, 11, 5, 6, 0, tzinfo=timezone.utc)
    book = lambda: Book(MID, "OPEN", False, 9000.0, {
        A: Quote(A, back=[(10.0, size)], lay=[(11.0, 100.0)], last_traded=10.0, traded=500.0),
        B: Quote(B, back=[(2.0, 300.0)], lay=[(2.02, 300.0)], last_traded=2.0, traded=1000.0),
        23: Quote(23, back=[(2.4, 300.0)], lay=[(2.46, 300.0)], last_traded=2.4, traded=1000.0)})
    polls = [(t0 + timedelta(minutes=i), book()) for i in range(3)]
    settled = {A: (8.0, "LOSER"), B: (2.5, "WINNER"), 23: (2.4, "LOSER")}
    polls.append((datetime(2026, 11, 5, 23, 0, tzinfo=timezone.utc),
                  Book(MID, "CLOSED", False, None, {s: Quote(s, status=st, bsp=b) for s, (b, st) in settled.items()},
                       True)))
    preds = pd.DataFrame({"market_id": [MID] * 3, "selection_id": [A, B, 23], "predicted_bfsp": [5.0, 2.4, 2.6]})
    return SimpleNamespace(day=date(2026, 11, 5), kind="win", markets=[m], polls={MID: polls},
                           poll_times={MID: [t for t, _ in polls]}, starts={MID: START}, real_fills={}, preds=preds,
                           settled={(MID, s): v for s, v in settled.items()},
                           removed=pd.DataFrame(columns=["market_id", "selection_id", "af", "rt"]), places={},
                           ledger=pd.DataFrame())


def _alpha(led):
    b = led[(led.event == "back") & (pd.to_numeric(led.matched, errors="coerce") > 0)] if len(led) else led
    return float(pd.to_numeric(b[b.selection_id == A].matched).sum()) if len(b) else 0.0


def test_two_agents_on_one_book_cannot_both_take_the_same_price():
    from research.arms import sessions
    D = _thin_day(50.0)
    cfg = lambda: core.make_cfg(clv_draws=2000)
    core.set_plan_cfg(cfg(), reset=True)
    t0 = datetime(2026, 11, 5, 5, 58, tzinfo=timezone.utc)
    alone, _, _ = core.run_session(D, sessions.ArmSession, cfg(), {"_D": D}, t0)
    assert _alpha(alone) == pytest.approx(44.44, abs=0.01)                   # alone, each would take GBP44.44
    out = core.run_agents(D, [("one", sessions.ArmSession, cfg(), {"_D": D}, core.BALANCE, False),
                              ("two", sessions.ArmSession, cfg(), {"_D": D}, core.BALANCE, False)], t0)
    got = sorted([_alpha(out["one"][0]), _alpha(out["two"][0])])
    assert sum(got) == pytest.approx(50.0, abs=0.02)                         # together, no more than the GBP50 shown
    assert got[1] == pytest.approx(44.44, abs=0.01) and got[0] == pytest.approx(5.56, abs=0.02)


def test_the_live_rule_s_replay_goes_first_and_the_order_turns_for_the_rest():
    from research.arms import sessions
    D = _thin_day(50.0)
    cfg = lambda: core.make_cfg(clv_draws=2000)
    core.set_plan_cfg(cfg(), reset=True)
    t0 = datetime(2026, 11, 5, 5, 58, tzinfo=timezone.utc)
    out = core.run_agents(D, [("agent", sessions.ArmSession, cfg(), {"_D": D}, core.BALANCE, False),
                              ("live_rule", sessions.ArmSession, cfg(), {"_D": D}, core.BALANCE, True)], t0)
    assert _alpha(out["live_rule"][0]) == pytest.approx(44.44, abs=0.01)    # the control's pick comes first
    assert _alpha(out["agent"][0]) == pytest.approx(5.56, abs=0.02)


def test_the_paper_agents_takes_never_cost_the_live_rule_s_replay():
    D = _thin_day(50.0)
    data = core.SharedReplayData(D)
    data.now = datetime(2026, 11, 5, 6, 1, tzinfo=timezone.utc)
    data.agent_taken[(MID, A, 10.0)] += 30.0                                  # an agent took GBP30 at 10.0
    data.agent_taken_lay[(MID, A, 11.0)] += 60.0                              # and GBP60 of the lay side at 11.0
    seen = data.books([MID])[MID].runners[A]
    assert seen.back == [(10.0, 20.0)] and seen.lay == [(11.0, 40.0)]       # the other agents see what is left
    data.real_view = True
    real = data.books([MID])[MID].runners[A]
    assert real.back == [(10.0, 50.0)] and real.lay == [(11.0, 100.0)]      # the control sees the book it met


def test_two_quotes_at_one_price_queue_one_behind_the_other():
    from model import race_book as rb
    from trading.exchange import tick_up
    mid = float(rb.market_now(np.array([2.0]), np.array([2.2]))[0])
    P = tick_up(mid * 1.02)
    assert P < 2.2                                                           # the back improves the best offer to layers
    D = _day([{"b_lay": 2.2}, {"b_lay": 2.2, "b_lt": P, "b_traded": 1012.0}, {"b_lay": 2.2}])
    cfg = lambda: core.make_cfg(clv_draws=2000)
    core.set_plan_cfg(cfg(), reset=True)
    t0 = datetime(2026, 11, 5, 5, 58, tzinfo=timezone.utc)
    pol = {"fair": "mid", "margin": 0.02}
    out = core.run_agents(D, [("q1", quotes.QuoteSession, cfg(), {**pol, "_D": D}, core.BALANCE, False),
                              ("q2", quotes.QuoteSession, cfg(), {**pol, "_D": D}, core.BALANCE, False)], t0)
    # both rest a back at P: the first is first in line, the second queues behind its GBP10
    f1, f2 = (out[n][1].fills_frame() for n in ("q1", "q2"))
    m1 = float(f1[f1.selection_id == B].matched.sum()) if len(f1) else 0.0
    m2 = float(f2[f2.selection_id == B].matched.sum()) if len(f2) else 0.0
    assert m1 + m2 == pytest.approx(6.0, abs=0.01)                           # half the GBP12 traded at P, once
    assert sorted([m1, m2]) == [0.0, 6.0]                                    # all of it the first quote's


def test_an_agent_s_day_reads_on_the_dashboard(tmp_path):
    from dashboard import traders as tr
    D = _day([{}, {"b_lt": 2.08, "b_traded": 1400.0}, {}])
    s, f = _quote(D)
    fills = f.assign(venue="Kempton", off="13:00")
    fp = quotes.fill_pnl(fills, D)
    m = horses.markets_table(fp)
    assert list(m.columns) == tr.MARKET_COLS and m.settled.all()
    p = tmp_path / "pred" / "paper/horses/shared/agents/quotes_mid2" / "2026-11-05"
    p.mkdir(parents=True)
    (p / "markets.csv").write_text(m.to_csv(index=False))
    src = tr.LocalSource(tmp_path)
    t = [x for x in tr.TRADERS if x.id == "horse-agent-quotes-mid2"][0]
    got, _ = t.reader(src, t.kind, t.base, "2026-11-05")
    assert got.pnl.sum() == pytest.approx(horses.summary("quotes_mid2", fp, D)["result_reduced"], abs=0.01)
    assert {x.id for x in tr.TRADERS} >= {f"horse-agent-{k.replace('_', '-')}" for k in horses.AGENTS}


def test_intraday_settles_the_races_run_on_the_recorder_s_bsps(monkeypatch):
    """The owner, 4 Nov 2026: the paper agents trading as if live, results the same day. With no price file yet,
    load_day(intraday=True) settles the races run so far from the recorder's books (each runner's last BSP read and
    its result where read), marks the day provisional and gives the last book's time as the run's end; without
    intraday the day still waits for the price file."""
    import pandas as pd
    from research.arms import core
    day = date(2026, 11, 4)
    start = "2026-11-04T13:00:00Z"
    cat = pd.DataFrame([{"market_id": "1.9", "selection_id": s, "market_type": "WIN", "country": "GB",
                         "market_start_utc": start, "venue": "Kempton", "market_name": "1m Hcap", "runner_name": n}
                        for s, n in ((1, "Alpha"), (2, "Bravo"))])
    rows = []
    for t, status, inplay, sp1, sp2, rs1 in (("2026-11-04T12:55:00Z", "OPEN", False, None, None, "ACTIVE"),
                                             ("2026-11-04T13:02:00Z", "OPEN", True, 4.2, 1.9, "ACTIVE"),
                                             ("2026-11-04T13:06:00Z", "CLOSED", False, 4.2, 1.9, "WINNER")):
        rows += [{"polled_utc": t, "market_id": "1.9", "selection_id": 1, "status": status, "inplay": inplay,
                  "runner_status": rs1, "sp_actual": sp1, "back1": 4.0, "back1_size": 50.0},
                 {"polled_utc": t, "market_id": "1.9", "selection_id": 2, "status": status, "inplay": inplay,
                  "runner_status": "LOSER" if rs1 == "WINNER" else "ACTIVE", "sp_actual": sp2, "back1": 2.0,
                  "back1_size": 80.0}]
    books = pd.DataFrame(rows)

    def fake_csv(key, **kw):
        if "/books_trader" in key:
            return books
        if "/markets_trader" in key:
            return cat
        return pd.DataFrame()
    monkeypatch.setattr(core, "csv", fake_csv)
    monkeypatch.setattr(core, "price_files", lambda d, m: pd.DataFrame())
    preds = pd.DataFrame({"market_id": ["1.9", "1.9"], "selection_id": [1, 2], "predicted_bfsp": [4.0, 2.0]})
    assert core.load_day(day, "win", preds=preds) is None                       # the nightly path still waits
    D = core.load_day(day, "win", preds=preds, intraday=True)
    assert D.provisional is True
    assert D.settled == {("1.9", 1): (4.2, "WINNER"), ("1.9", 2): (1.9, "LOSER")}
    assert D.last_poll == datetime(2026, 11, 4, 13, 6, tzinfo=timezone.utc)


def test_a_backs_variant_trades_its_own_book_in_the_live_trader_s_place():
    """The first shared run (4 Nov): three copies of the rule on one book took each other's leftovers. A backs variant
    is an alternative to the live rule: it trades a book of its own, standing in for the live trader."""
    assert horses.WORLDS["live"] == ["live_rule", "lays", "quotes_model3", "quotes_mid2"]
    for v in ("backs_floor50", "backs_win1000"):
        assert horses.WORLDS[v] == [v] and v in horses.STANDS_IN
    every = [n for ms in horses.WORLDS.values() for n in ms]
    assert sorted(every) == sorted(horses.STRATEGIES) and len(every) == len(set(every))
    specs = {sp[0]: sp for sp in horses.agent_specs(2000)}
    assert specs["backs_win1000"][5] and specs["live_rule"][5] and not specs["lays"][5]


def test_the_day_so_far_runs_in_its_own_queue_on_github_s_runners_and_never_reaches_betfair():
    """The owner, 4 Nov 2026: the paper agents as if live, results through the day. The intraday run has a workflow
    and a queue of its own, so a long backfill in paper-horses.yml never holds the hourly results up; like the desk,
    it reads S3 only on GitHub's runners and places no order."""
    import yaml
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / ".github/workflows"
    text = (root / "paper-horses-intraday.yml").read_text()
    wf = yaml.safe_load(text)
    job = wf["jobs"]["desk"]
    assert job["runs-on"] == "ubuntu-latest"
    assert wf["concurrency"]["group"] == "paper-horses-intraday"
    assert yaml.safe_load((root / "paper-horses.yml").read_text())["concurrency"]["group"] != "paper-horses-intraday"
    assert "--intraday" in str(job["steps"]) and "--intraday" not in (root / "paper-horses.yml").read_text()
    assert "BETFAIR" not in text and "TRADING_LIVE" not in text and "auto_trade" not in text
