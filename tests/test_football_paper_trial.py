"""The football paper trial's scorer: mechanics on synthetic rows only (never a reported figure)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from football.reference import team_id
from research.open_close import football_exchange as fx
from research.open_close import football_paper_trial as pt
from sources.common import Store

from tests.test_football_exchange import synthetic_gold


# ---------------------------------------------------------------------------
# the calendar
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("match_day,coll,moment", [
    ("2026-11-06", "2026-11-06", "2026-11-06 17:00"),     # Friday: that day's collection (GMT)
    ("2026-11-07", "2026-11-06", "2026-11-06 17:00"),     # Saturday
    ("2026-11-09", "2026-11-06", "2026-11-06 17:00"),     # Monday: still Friday's
    ("2026-11-10", "2026-11-10", "2026-11-10 13:00"),     # Tuesday
    ("2026-11-12", "2026-11-10", "2026-11-10 13:00"),     # Thursday
    ("2026-10-03", "2026-10-02", "2026-10-02 16:00"),     # BST: 17:00 UK is 16:00 UTC
])
def test_collection_of(match_day, coll, moment):
    c, m = pt.collection_of(match_day)
    assert c.isoformat() == coll
    assert m == pd.Timestamp(moment, tz="UTC")


def test_trial_start_is_the_first_friday_on_or_after_the_recorder():
    assert pt.trial_start(date(2026, 11, 2)) == date(2026, 11, 6)
    assert pt.trial_start(date(2026, 11, 6)) == date(2026, 11, 6)


def test_fixtures_file_is_the_first_fetch_after_the_collection_day():
    stamps = ["2026-11-06", "2026-11-07", "2026-11-08", "2026-11-11"]
    # the 6 Nov folder was fetched at 03:20 UTC on 6 Nov, before Friday's collection: never used for it
    assert pt.fixtures_stamp(stamps, date(2026, 11, 6)) == "2026-11-07"
    assert pt.fixtures_stamp(stamps, date(2026, 11, 10)) == "2026-11-11"
    assert pt.fixtures_stamp(["2026-11-10"], date(2026, 11, 10)) is None


# ---------------------------------------------------------------------------
# the recorded books
# ---------------------------------------------------------------------------

def book(polled, sel, back, lay, size=50.0, inplay=False, status="OPEN", market="1.1"):
    return {"polled_utc": polled, "market_id": market, "status": status, "inplay": inplay, "selection_id": sel,
            "back1": back, "back1_size": size, "lay1": lay, "lay1_size": size}


def test_early_book_is_the_first_full_snapshot_at_or_after_the_moment():
    b = pt.clean_books(pd.DataFrame([
        book("2026-11-06T16:50:00Z", 1, 1.80, 1.82), book("2026-11-06T16:50:00Z", 2, 2.20, 2.24),   # before
        book("2026-11-06T17:05:00Z", 1, 1.90, 1.92),                                                  # one leg only
        book("2026-11-06T17:20:00Z", 1, 1.91, 1.93), book("2026-11-06T17:20:00Z", 2, 2.02, 2.06),
    ]))
    t, snap = pt.early_book(b, ["1", "2"], pd.Timestamp("2026-11-06 17:00", tz="UTC"), 60)
    assert t == pd.Timestamp("2026-11-06 17:20", tz="UTC")
    assert snap.loc["1", "back1"] == 1.91
    t, snap = pt.early_book(b, ["1", "2"], pd.Timestamp("2026-11-06 17:00", tz="UTC"), 10)
    assert t is None and snap is None


def test_close_is_the_last_two_sided_book_before_in_play():
    b = pt.clean_books(pd.DataFrame([
        book("2026-11-07T14:50:00Z", 1, 1.80, 1.82), book("2026-11-07T14:58:00Z", 1, 1.78, 1.80),
        book("2026-11-07T14:59:00Z", 1, 1.77, np.nan),                 # one-sided: not a close
        book("2026-11-07T15:02:00Z", 1, 1.60, 1.62, inplay=True),
    ]))
    c = pt.close_book(b)
    assert c.loc["1", "back1"] == 1.78 and c.loc["1", "lay1"] == 1.80


def test_fill_trade_back_lay_size_and_clv():
    early = pd.Series({"back1": 2.0, "back1_size": 50.0, "lay1": 2.04, "lay1_size": 50.0})
    close = pd.Series({"back1": 1.80, "back1_size": 50.0, "lay1": 1.82, "lay1_size": 50.0})
    r = pt.fill_trade(1, early, close, 10.0)
    assert r["skip"] is None and r["fill_open"] == 2.0 and r["fill_close"] == 1.82
    assert r["profit"] == pytest.approx(2.0 / 1.82 - 1)
    assert r["clv"] == pytest.approx(2.0 / 1.81 - 1) and r["right_way"]
    lay = pt.fill_trade(-1, early, close, 10.0)
    assert lay["fill_open"] == 2.04 and lay["fill_close"] == 1.80
    assert lay["profit"] == pytest.approx(1 - 2.04 / 1.80) and not lay["right_way"]
    short = pt.fill_trade(1, pd.Series({**early, "back1_size": 4.0}), close, 10.0)
    assert short["skip"] == "size short at the early book"
    assert pt.fill_trade(1, early, None, 10.0)["skip"] == "no two-sided close"


# ---------------------------------------------------------------------------
# the window and the gate
# ---------------------------------------------------------------------------

def trades_on(days):
    return pd.DataFrame({"day": days, "market_id": [f"m{i}" for i in range(len(days))], "profit": 0.01,
                         "stake": 1.0, "right_way": True, "side": 1, "clv": 0.01, "close_size_short": False})


def test_window_running_complete_capped():
    start = date(2026, 11, 6)
    few = trades_on(["2026-11-07"] * 5)
    w = pt.window(few, start, date(2026, 12, 1), min_bets=300, min_days=90)
    assert w["status"] == "running" and w["end"] == "2026-12-01"
    many = trades_on(["2026-11-07"] * 300)
    w = pt.window(many, start, date(2027, 3, 1), min_bets=300, min_days=90)
    assert w["status"] == "complete" and w["end"] == "2027-02-04"           # start + 90 days, the later of the two
    w = pt.window(few, start, date(2027, 11, 10), min_bets=300, min_days=90)
    assert w["status"] == "capped (underpowered)" and w["end"] == "2027-11-06"
    assert pt.window(few, start, date(2026, 11, 1), 300, 90)["status"] == "not started"


def test_gate_is_running_until_the_window_is_complete():
    res = {"headline": {"bets": 5, "roi_lo90": 0.1, "clv_mean": 0.01}, "window": {"status": "running",
           "calendar_days": 20}, "halves": {"first": {"roi": 0.1}, "second": {"roi": 0.1}}}
    g = pt.gate(res, {"min_bets": 300, "min_days": 90})
    assert g["verdict"] == "running" and g["passed"] is None
    res["headline"]["bets"], res["window"] = 310, {"status": "complete", "calendar_days": 95}
    assert pt.gate(res, {"min_bets": 300, "min_days": 90})["verdict"] == "pass"
    res["halves"]["second"]["roi"] = -0.01
    assert pt.gate(res, {"min_bets": 300, "min_days": 90})["verdict"] == "fail"


# ---------------------------------------------------------------------------
# markets to matches
# ---------------------------------------------------------------------------

def test_map_markets_pairs_both_sides_and_the_legs():
    fixtures = pd.DataFrame({"match_id": ["a", "b"], "HomeTeam": ["Arsenal", "Everton"],
                             "AwayTeam": ["Chelsea", "Fulham"], "country": ["ENG", "ENG"]})
    markets = pd.DataFrame([
        {"market_id": "1.1", "market_type": "OVER_UNDER_25", "event_name": "Arsenal v Chelsea", "country": "GB",
         "selection_id": 47972, "runner_name": "Under 2.5 Goals", "market_start_utc": "2026-11-07T15:00:00Z"},
        {"market_id": "1.1", "market_type": "OVER_UNDER_25", "event_name": "Arsenal v Chelsea", "country": "GB",
         "selection_id": 47973, "runner_name": "Over 2.5 Goals", "market_start_utc": "2026-11-07T15:00:00Z"},
        {"market_id": "1.2", "market_type": "MATCH_ODDS", "event_name": "Everton v Fulham", "country": "GB",
         "selection_id": 11, "runner_name": "Everton", "market_start_utc": "2026-11-07T15:00:00Z"},
        {"market_id": "1.2", "market_type": "MATCH_ODDS", "event_name": "Everton v Fulham", "country": "GB",
         "selection_id": 12, "runner_name": "Fulham", "market_start_utc": "2026-11-07T15:00:00Z"},
        {"market_id": "1.2", "market_type": "MATCH_ODDS", "event_name": "Everton v Fulham", "country": "GB",
         "selection_id": 58805, "runner_name": "The Draw", "market_start_utc": "2026-11-07T15:00:00Z"},
        {"market_id": "1.3", "market_type": "MATCH_ODDS", "event_name": "Lazio v Roma", "country": "IT",
         "selection_id": 1, "runner_name": "Lazio", "market_start_utc": "2026-11-07T15:00:00Z"},
    ])
    m = pt.map_markets(markets, fixtures)
    got = {(r.market_id, r.selection_id): (r.match_id, r.leg) for r in m.itertuples()}
    assert got == {("1.1", "47972"): ("a", "Under 2.5"), ("1.1", "47973"): ("a", "Over 2.5"),
                   ("1.2", "11"): ("b", "H"), ("1.2", "12"): ("b", "A"), ("1.2", "58805"): ("b", "D")}


# ---------------------------------------------------------------------------
# end to end: a synthetic panel, a fixtures file, a recorder's day
# ---------------------------------------------------------------------------

TEAMS = [("Arsenal", "Chelsea"), ("Everton", "Fulham"), ("Brentford", "Burnley")]


def trial_inputs(tmp_path):
    g0, silver = synthetic_gold()
    dc0 = pd.DataFrame({"match_id": g0["match_id"], "p_h": 0.45, "p_d": 0.27, "p_a": 0.28, "p_o25": 0.5})
    panel, _ = fx.exchange_panel(g0, silver, dc0)
    rows, fx_rows = [], []
    for i, (h, a) in enumerate(TEAMS):
        rows.append({"match_id": f"t{i}", "status": "SCHEDULED", "match_date": "2025-11-08",
                     "kickoff_utc": pd.Timestamp("2025-11-08 15:00", tz="UTC"), "country": "ENG", "home_team": h,
                     "away_team": a, "home_team_id": team_id("ENG", h), "away_team_id": team_id("ENG", a),
                     "season": 2025, "competition_id": "ENG1", "fd_code": "E0"})
        fx_rows.append({"Div": "E0", "Date": "08/11/2025", "Time": "15:00", "HomeTeam": h, "AwayTeam": a,
                        "BFEH": 2.1, "BFED": 3.4, "BFEA": 3.9, "BFE>2.5": 1.95, "BFE<2.5": 2.0,
                        "AvgH": 2.0, "AvgD": 3.3, "AvgA": 3.7, "Avg>2.5": 1.85, "Avg<2.5": 1.95,
                        "MaxH": 2.15, "MaxD": 3.5, "MaxA": 4.0, "Max>2.5": 1.98, "Max<2.5": 2.05,
                        "B365H": 2.0, "B365D": 3.3, "B365A": 3.6, "B365>2.5": 1.83, "B365<2.5": 1.95})
    gold = pd.concat([g0, pd.DataFrame(rows)], ignore_index=True)
    store = Store(root=tmp_path)
    p = tmp_path / "football/raw/fixtures/2025-11-08/fixtures.csv"
    p.parent.mkdir(parents=True)
    pd.DataFrame(fx_rows).to_csv(p, index=False)
    markets, books = [], []
    early_at, close_at, off_at = "2025-11-07T17:05:00Z", "2025-11-08T14:55:00Z", "2025-11-08T15:01:00Z"
    quotes = {0: ((1.90, 1.92, 50.0), (2.02, 2.06, 50.0)),      # tight, sizes cover GBP10: trades
              1: ((1.70, 1.72, 50.0), (1.80, 1.83, 50.0)),      # wide (1.144): skipped
              2: ((1.90, 1.92, 2.0), (2.02, 2.06, 2.0))}        # tight, GBP2 shown: skipped for size
    for i, (h, a) in enumerate(TEAMS):
        mid = f"1.{i}"
        for j, name in enumerate(("Over 2.5 Goals", "Under 2.5 Goals")):
            sid = 100 * i + j
            markets.append({"market_id": mid, "market_type": "OVER_UNDER_25", "event_name": f"{h} v {a}",
                            "country": "GB", "selection_id": sid, "runner_name": name,
                            "market_start_utc": "2025-11-08T15:00:00Z"})
            back, lay, size = quotes[i][j]
            books.append(book(early_at, sid, back, lay, size, market=mid))
            books.append(book(close_at, sid, back - 0.1 if j == 0 else back + 0.1,
                              lay - 0.1 if j == 0 else lay + 0.1, 50.0, market=mid))
            books.append(book(off_at, sid, 1.5, 1.52, 50.0, inplay=True, market=mid))
    files = {(date(2025, 11, 8), "markets_football_eve_GB.csv"): pd.DataFrame(markets),
             (date(2025, 11, 8), "books_football_eve_GB.csv"): pd.DataFrame(books)}
    read_day = lambda d, name: files.get((d, name))                         # noqa: E731
    dc_fn = lambda gold, rows, as_of: pd.DataFrame({"match_id": rows["match_id"], "p_h": 0.45, "p_d": 0.27,  # noqa
                                                    "p_a": 0.28, "p_o25": 0.5})
    return gold, panel, store, read_day, dc_fn


def test_score_trial_end_to_end(tmp_path):
    gold, panel, store, read_day, dc_fn = trial_inputs(tmp_path)
    rule = {"model": "gbm_nopin", "market_types": ["OVER_UNDER_25"], "leagues": "all",
            "book_width": ["tight <1.02", "medium 1.02-1.05"], "threshold": 0.0, "commission": 0.02,
            "gate": {"min_bets": 300, "min_days": 90, "roi_lo90_above": 0, "clv_mean_above": 0}}
    t = pt.Trial(gold=gold, panel=panel, store=store, read_day=read_day, rule=rule, start=date(2025, 11, 7),
                 as_of=date(2025, 11, 9), n_jobs=1, n_boot=200, dc_fn=dc_fn)
    out_csv = tmp_path / "trades.csv"
    res = pt.score_trial(t, "sha", trades_out=str(out_csv))
    r = res["rules"][0]
    assert r["funnel"]["fixture_rows_with_match_id"] == 3
    assert r["forecasts"] == 6 and r["signals"] == 6                        # threshold 0: every leg signals
    assert r["headline"]["bets"] == 2                                       # the tight, well-sized market only
    assert r["skipped"] == {"recorded book wide": 2, "size short at the early book": 2}
    assert r["wide_share_of_signals_at_a_recorded_book"] == pytest.approx(2 / 6)
    assert res["gate"]["verdict"] == "running"
    tr = pd.read_csv(out_csv)
    traded = tr[tr["traded"]]
    assert set(traded["market_id"].astype(str)) == {"1.0"}
    for x in traded.itertuples():
        expect = x.fill_open / x.fill_close - 1 if x.side > 0 else 1 - x.fill_open / x.fill_close
        assert x.profit == pytest.approx(expect)
        assert x.early_utc.startswith("2025-11-07T17:05")
