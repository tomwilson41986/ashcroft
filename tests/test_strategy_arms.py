"""The strategy arms' machinery (research/arms; the owner, 3 Nov 2026) on a made-up day: the policies, the lay
side, the race book, the exits and the place chances. Fixtures only; the arms are evaluated on the recorded books."""

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research.arms import core, exits, place, sessions  # noqa: E402
from trading import session as trading_session  # noqa: E402
from trading.exchange import Book, Market, Quote  # noqa: E402

DAY = date(2026, 11, 2)
MID = "1.900"
START = datetime(2026, 11, 2, 13, 0, tzinfo=timezone.utc)
A, B, C, D_, E = 11, 12, 13, 14, 15


def _book(t_shift: int, lay_a=11.0, lay_a_size=100.0):
    runners = {                                                       # the market's book sums to 1.0
        A: Quote(A, back=[(10.0, 100.0)], lay=[(lay_a, lay_a_size)], last_traded=10.0),       # long in the market
        B: Quote(B, back=[(2.0, 300.0)], lay=[(2.02, 300.0), (2.04, 200.0)], last_traded=2.0),  # short: a lay
        C: Quote(C, back=[(4.0, 100.0)], lay=[(4.2, 100.0)], last_traded=4.0),
        D_: Quote(D_, back=[(40.0, 100.0)], lay=[(44.0, 50.0)], last_traded=40.0),
        E: Quote(E, back=[(8.0, 100.0)], lay=[(8.4, 100.0)], last_traded=8.0),
    }
    return Book(MID, "OPEN", False, 5000.0, runners)


@pytest.fixture
def day():
    m = Market(MID, "Plumpton", "GB", START, name="2m Hcap Hrd",
               runners={A: "Alpha", B: "Bravo", C: "Charlie", D_: "Delta", E: "Echo"})
    t0 = datetime(2026, 11, 2, 6, 20, tzinfo=timezone.utc)
    polls = [(t0, _book(0)), (t0 + timedelta(minutes=1), _book(1, lay_a=9.0)),
             (t0 + timedelta(minutes=2), _book(2, lay_a=10.6))]
    settled = {A: (8.0, "WINNER"), B: (2.5, "LOSER"), C: (4.0, "LOSER"), D_: (35.0, "LOSER"), E: (9.0, "LOSER")}
    closed = Book(MID, "CLOSED", False, None, {s: Quote(s, status=st, bsp=b) for s, (b, st) in settled.items()}, True)
    polls.append((datetime(2026, 11, 2, 23, 0, tzinfo=timezone.utc), closed))
    preds = pd.DataFrame({"market_id": [MID] * 5, "selection_id": [A, B, C, D_, E],
                          "predicted_bfsp": [5.0, 4.0, 4.0, 20.0, 8.0]})
    ledger = pd.DataFrame([{"ts": "2026-11-02T06:20:00Z", "event": "back", "market_id": MID, "selection_id": A,
                            "matched": 40.0, "avg_price": 10.0, "best_back": 10.0}])
    return SimpleNamespace(day=DAY, kind="win", markets=[m], polls={MID: polls},
                           poll_times={MID: [t for t, _ in polls]}, starts={MID: START}, real_fills={},
                           preds=preds, settled={(MID, s): v for s, v in settled.items()},
                           removed=pd.DataFrame(columns=["market_id", "selection_id", "af", "rt"]), places={},
                           ledger=ledger)


def _run(D, cls, policy=None, **over):
    cfg = core.make_cfg(clv_draws=2000, **over)
    core.set_plan_cfg(core.make_cfg(clv_draws=2000), reset=True)
    pol = {**(policy or {}), "_D": D}
    led, s, x = core.run_session(D, cls, cfg, pol, core.session_start(D, "win"))
    return led, s


def _backs(led):
    if led.empty or "event" not in led:
        return {}
    b = led[(led.event == "back") & (pd.to_numeric(led.matched, errors="coerce") > 0)]
    return {int(r.selection_id): (float(r.avg_price), float(r.matched)) for r in b.itertuples()}


def test_the_rule_backs_the_long_horses_and_the_score_reads_the_hedge(day):
    led, s = _run(day, sessions.ArmSession)
    backs = _backs(led)
    assert A in backs and D_ in backs and B not in backs          # the market has A and D long, B short
    assert backs[D_][1] == 15.0                                      # to win GBP400 at 40 is GBP10: the GBP15 floor
    r = core.score(led, day, "baseline", s.state.settled_pnl, False)
    assert r["horses"] == len(backs) and r["clv"] > 0.03             # A: 10 / 8 - 1; D: 40 / 35 - 1
    assert r["result"] == pytest.approx(s.state.settled_pnl, abs=0.01)
    assert r["first_staked"] == pytest.approx(sum(m for _, m in backs.values()))


def test_the_stake_floor_on_long_prices(day):
    led, _ = _run(day, sessions.ArmSession, {"floor_long": (21.0, 30.0)})
    backs = _backs(led)
    assert backs[D_][1] == 30.0 and backs[A][1] == pytest.approx(400 / 9, abs=0.05)


def test_a_bar_that_rises_and_a_stop_hold_the_backs(day):
    led, _ = _run(day, sessions.ArmSession, {"bar_slope": 5.0})      # 0.03 + 5 x 0.08 h: nothing clears it
    assert not _backs(led)
    led, _ = _run(day, sessions.ArmSession, {"stop_after": "06:00"})
    assert not _backs(led)


def test_the_lay_side_lays_the_short_horse_and_closes_at_the_bsp(day):
    led, s = _run(day, sessions.LaySession, {"lay_bar": 0.03, "max_lay_price": 20.0})
    assert not _backs(led)                                           # lays only
    fills = [r for r in s.lay_rows if r["event"] == "lay" and r["matched"] > 0]
    assert fills and all(r["selection_id"] == B for r in fills)
    assert fills[0]["matched"] == 200.0 and fills[0]["avg_price"] == 2.02   # GBP400 at risk capped at GBP200 a stake
    f = s.lay_positions_frame()
    r = core.score_side(f, day, "lays", side="lay")
    assert r["clv_x_stake"] == pytest.approx(200 * (1 - 2.02 / 2.5), abs=0.01)
    assert r["result"] == pytest.approx(0.98 * 200 * (1 - 2.02 / 2.5), abs=0.01)


def test_the_race_book_takes_positions_within_its_exposure(day):
    led, s = _run(day, sessions.RaceBookSession, {"lays": True, "exposure": 300.0, "draws": 1000})
    rows = s.book_positions_frame()
    assert len(rows) >= 1 and rows.liability.sum() <= 300.0 + 1e-6
    r = sessions.score_book(rows, day, "book_lays")
    assert r["races"] == 1 and r["positions"] == len(rows)


def test_the_exits_fire_on_the_right_poll(day):
    rows = exits.exits_day(day)
    by = rows.set_index("rule")
    assert by.loc["sp", "profit"] == pytest.approx(40 * (10 / 8 - 1))
    assert by.loc["tp8", "fired"] and by.loc["tp8", "exit"] == 9.0           # 06:21: the lay at 9.0 is 10% under
    assert not by.loc["tp12", "fired"]
    assert by.loc["sl5", "fired"] and by.loc["sl5", "exit"] == 10.6          # 06:22: 6% over
    assert by.loc["t15", "exit"] == 10.6 and by.loc["t15", "profit"] < 0     # the last book before T-15
    assert len(exits.summarise_exits(rows)) == len(exits.RULES)


def test_harville_place_chances():
    p = harv = place.harville_place([0.5, 0.3, 0.2], 2)
    assert harv.sum() == pytest.approx(2.0)
    assert p[0] == pytest.approx(0.5 + 0.3 * 0.5 / 0.7 + 0.2 * 0.5 / 0.8)
    assert place.harville_place([0.5, 0.3, 0.2], 3).tolist() == [1.0, 1.0, 1.0]
    assert place.places_for(6, None) == 2 and place.places_for(10, None) == 3 and place.places_for(16, 4) == 4


def test_the_lay_fill_takes_the_levels_and_keeps_them_taken(day):
    data = core.ReplayData(day)
    data.now = datetime(2026, 11, 2, 6, 20, tzinfo=timezone.utc)
    x = core.ReplayExchange(data, bank=4000.0, balance=5000.0)
    x.books([MID])
    f = x.lay(MID, B, 2.04, 400.0, ref="t")
    assert f.status == "SUCCESS" and f.matched == 400.0
    assert f.avg_price == pytest.approx((300 * 2.02 + 100 * 2.04) / 400, abs=1e-4)
    book = x.books([MID])[MID]
    assert book.runners[B].lay == [(2.04, 100.0)]                     # 300 at 2.02 and 100 at 2.04 are gone
    g = x.lay(MID, B, 2.02, 50.0, ref="t")
    assert g.status == "FAILURE"


def test_importing_the_arms_leaves_the_traders_plan_alone(day):
    """The package patches trading.session.plan_race only for the span of a replay (the full suite collects these
    tests before the live trader's: 3 Nov)."""
    from trading import strategy
    assert trading_session.plan_race is strategy.plan_race
    _run(day, sessions.ArmSession)
    assert trading_session.plan_race is strategy.plan_race


def test_the_timing_study_scores_each_fill_and_cuts_at_any_day_limit(day, tmp_path, capsys):
    """research/arms/timing.py (the owner, 4 Nov: when to bet): each back's locked value at the BSP, and the result
    at a day limit read off the uncapped session."""
    from research.arms import timing
    led, s = _run(day, sessions.ArmSession)
    f = timing.fills(led, day, "uncapped")
    assert len(f) and set(f.selection_id) == set(_backs(led))
    for r in f.itertuples():
        assert r.locked == pytest.approx(r.matched * (r.avg_price / r.bsp - 1))
        assert r.locked_red == pytest.approx(r.locked)                     # nobody taken out: no cut
        assert not r.evening and r.mins_to_off > 0
    rows = pd.DataFrame([core.score(led, day, "uncapped", s.state.settled_pnl, False)])
    fs = f.assign(uk=f.uk.astype(str), t=f.t.astype(str))
    timing.report(rows, fs, {}, tmp_path)
    out = capsys.readouterr().out
    assert "the result at each day limit" in out and "by UK hour" in out
    assert (tmp_path / "fills.csv.gz").exists()
    assert timing._start(day, "evening_morning", "06:15").astimezone(core.UK).hour == 17
