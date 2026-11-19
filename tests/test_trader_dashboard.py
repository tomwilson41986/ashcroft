"""The traders' dashboard (dashboard/): each trader's ledger read as its own settle step reads it, the forecast by the
owner's day-report method, the page and the job summary. The ledgers here are small hand-made test inputs laid out as
the traders write them; nothing is shown as a real figure."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from dashboard.__main__ import main, summary_md
from dashboard.page import render
from dashboard.traders import TRADERS, LocalSource, build, summarise, trader_days

TODAY = date(2026, 11, 4)


def _put(root, kind, key, df):
    p = root / kind / key
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def horse_ledger(day, races):
    """races: [(market_id, matched, edge, pnl or None, commission or None)]"""
    rows = []
    for mid, matched, edge, pnl, com in races:
        rows.append({"ts": f"{day}T09:00:00Z", "event": "back", "market_id": mid, "venue": "Kempton",
                     "off": f"{day} 14:00", "selection_id": 1, "matched": matched, "edge": edge})
        if pnl is not None:
            rows.append({"ts": f"{day}T15:00:00Z", "event": "settle", "market_id": mid, "selection_id": 1,
                         "pnl": pnl, "matched": matched})
        if com is not None:
            rows.append({"ts": f"{day}T15:00:00Z", "event": "commission", "market_id": mid, "commission": com})
    return pd.DataFrame(rows)


def T(tid):
    return next(t for t in TRADERS if t.id == tid)


def EL():
    """The early lay's reader without its fresh start (11 Nov 2026): the ledger tests below are dated 4 Nov."""
    return replace(T("greyhound-early-lay"), start_from=None)


def test_horse_settled_pnl_is_the_day_reports(tmp_path):
    day = TODAY.isoformat()
    # a: won 50, commission row 1.00 -> 49; b: won 20, no commission row -> 2% = 0.40 -> 19.60; c: lost; d: open
    _put(tmp_path, "pred", f"trading/live/{day}/ledger.csv",
         horse_ledger(day, [("1.a", 100, 0.1, 50, 1.0), ("1.b", 100, 0.1, 20, None), ("1.c", 50, 0.2, -50, None),
                            ("1.d", 30, 0.1, None, None)]))
    src = LocalSource(tmp_path)
    days, last = trader_days(src, T("horse-live"), TODAY, cache=False)
    m = days[day].set_index("market_id")
    assert m.loc["1.a", "pnl"] == pytest.approx(49.0)
    assert m.loc["1.b", "pnl"] == pytest.approx(19.6)
    assert m.loc["1.c", "pnl"] == pytest.approx(-50.0)
    assert pd.isna(m.loc["1.d", "pnl"]) and not m.loc["1.d", "settled"]
    s = summarise(T("horse-live"), days, TODAY, last)
    assert s["today"]["pnl"] == pytest.approx(18.6)
    assert s["today"]["volume"] == pytest.approx(280.0)
    assert s["today"]["settled"] == 3 and s["today"]["markets"] == 4
    assert s["today"]["forecast"] is None                        # no record before today: no forecast
    assert s["all_time"]["pnl"] == pytest.approx(18.6)
    assert last == f"{day}T15:00:00Z"


def test_a_trader_without_marks_is_forecast_by_its_records_mean_a_market_never_the_expected_value(tmp_path):
    # five settled days from 28 Oct, results 4, 6, 2, 8, 5 on one market each; the expected value at the fill is
    # ignored (the owner, 4 Nov: never scale the forecast by the record's result over the expected value)
    results = [4, 6, 2, 8, 5]
    for i, r in enumerate(results):
        d = (TODAY - timedelta(days=7 - i)).isoformat()
        _put(tmp_path, "pred", f"trading/live/{d}/ledger.csv", horse_ledger(d, [("1.x", 100, 0.1, r, 0.0001)]))
    day = TODAY.isoformat()
    _put(tmp_path, "pred", f"trading/live/{day}/ledger.csv",
         horse_ledger(day, [("1.s", 100, 0.1, 10, 0.0001), ("1.o", 200, 0.2, None, None)]))
    t = replace(T("horse-live"), forecast_by="record")
    days, last = trader_days(LocalSource(tmp_path), t, TODAY, cache=False)
    s = summarise(t, days, TODAY, last)
    per_mkt = sum(r - 0.0001 for r in results) / 5                # a market a day: the mean a market is the mean day
    assert s["record"]["days"] == 5
    assert s["today"]["forecast"] == pytest.approx(round(10 - 0.0001 + 1 * per_mkt, 2))
    assert "expected value" not in s["today"]["method"]
    assert s["today"]["forecast_lo"] < s["today"]["forecast"] < s["today"]["forecast_hi"]
    nov_settled = (5 - 0.0001) + (10 - 0.0001)
    assert s["month"]["days_left"] == 26
    assert s["month"]["forecast"] == pytest.approx(round(nov_settled + per_mkt + 26 * per_mkt, 2), abs=0.02)


def _books(day, rows):
    """rows: [(market_id, selection_id, polled 'HH:MM', back1, lay1, sp)] as the recorder writes them (test inputs)."""
    return pd.DataFrame([{"polled_utc": f"{day}T{hm}:00Z", "market_id": m, "selection_id": sel, "status": "OPEN",
                          "inplay": False, "runner_status": "ACTIVE", "back1": b, "lay1": lay, "sp_actual": sp,
                          "adjustment_factor": 0.0, "removal_utc": None} for m, sel, hm, b, lay, sp in rows])


def _put_gz(root, kind, key, df):
    p = root / kind / key
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, compression="gzip")


def _horse_fill(day, mid, matched, price, ts_hm, off_hm, pnl=None):
    rows = [{"ts": f"{day}T{ts_hm}:00Z", "event": "back", "market_id": mid, "venue": "Kempton",
             "off": f"{day} {off_hm}", "selection_id": 7, "matched": matched, "avg_price": price, "edge": 0.5}]
    if pnl is not None:
        rows.append({"ts": f"{day}T21:00:00Z", "event": "settle", "market_id": mid, "selection_id": 7, "pnl": pnl,
                     "matched": matched})
        rows.append({"ts": f"{day}T21:00:00Z", "event": "commission", "market_id": mid,
                     "commission": max(pnl, 0) * 0.02 or 0.0001})
    return rows


def test_the_horse_forecast_is_the_day_reports_mark_and_drift(tmp_path, monkeypatch):
    import math
    import dashboard.day_forecast as dfc
    monkeypatch.setattr(dfc, "MIN_POOL", 1)
    t = replace(T("horse-live"), record_from="2026-11-01")
    # three past days, two races a day: 10 staked at 4.0, the book 5 min before the off 3.0 / 3.2, the BSP 3.0,
    # each race lost (-10, no commission row due): every race drifts the same, so the range closes on the forecast
    for d in ("2026-11-01", "2026-11-02", "2026-11-03"):
        a, b = f"1.a{d[-2:]}", f"1.b{d[-2:]}"                    # Betfair's market ids: a new one each race
        rows = _horse_fill(d, a, 10, 4.0, "13:00", "14:00", -10) + _horse_fill(d, b, 10, 4.0, "15:00", "16:00", -10)
        _put(tmp_path, "pred", f"trading/live/{d}/ledger.csv", pd.DataFrame(rows))
        _put_gz(tmp_path, "capture", f"betfair_live/{d}/books.csv.gz",
                _books(d, [(a, 7, "13:50", 3.0, 3.2, None), (a, 7, "14:01", 3.0, 3.2, 3.0),
                           (b, 7, "15:50", 3.0, 3.2, None), (b, 7, "16:01", 3.0, 3.2, 3.0)]))
    day = TODAY.isoformat()
    # today: one race settled (+20 gross, 2% commission -> 19.60), one to run, 10 at 4.0, the book now 3.0 / 3.2
    rows = _horse_fill(day, "1.s", 10, 3.0, "11:00", "12:00", 20) + _horse_fill(day, "1.o", 10, 4.0, "13:00", "15:00")
    _put(tmp_path, "pred", f"trading/live/{day}/ledger.csv", pd.DataFrame(rows))
    _put_gz(tmp_path, "capture", f"betfair_live/{day}/books.csv.gz", _books(day, [("1.o", 7, "14:50", 3.0, 3.2, None)]))
    data = build(LocalSource(tmp_path), today=TODAY, cache=False, traders=[t])
    h = data["sports"][0]["traders"][0]
    d = h["today"]
    mid = math.sqrt(3.0 * 3.2)
    mark_past = 10 * (4.0 / mid - 1)                    # each past race 5 min before the off
    drift5 = (-10 - mark_past) / 10                     # a pound staked, mark to result
    mark_now = 10 * (4.0 / mid - 1)
    assert d["pnl"] == pytest.approx(19.6)
    assert d["races_left"] == 1
    assert d["mark_mid"] == pytest.approx(round(mark_now, 2))
    assert d["cash_out"] == pytest.approx(round(10 * (4.0 / 3.2 - 1), 2))
    assert d["forecast"] == pytest.approx(round(19.6 + mark_now + drift5 * 10, 2), abs=0.01)
    assert d["forecast_lo"] == pytest.approx(d["forecast"], abs=0.01) == d["forecast_hi"]
    drift5_co = (-10 - 10 * (4.0 / 3.2 - 1)) / 10
    assert d["forecast_cash_out"] == pytest.approx(round(19.6 + 10 * (4.0 / 3.2 - 1) + drift5_co * 10, 2), abs=0.01)
    assert d["method"].startswith("the day report") and "expected value" not in d["method"]
    assert d["forecast_past_days"] == 3 and d["forecast_past_races"] == 6
    assert h["forecast_races"][0]["market_id"] == "1.o"


def test_no_past_races_marked_means_no_forecast_never_the_expected_value(tmp_path):
    day = TODAY.isoformat()
    rows = _horse_fill(day, "1.o", 10, 4.0, "13:00", "15:00")
    _put(tmp_path, "pred", f"trading/live/{day}/ledger.csv", pd.DataFrame(rows))
    _put_gz(tmp_path, "capture", f"betfair_live/{day}/books.csv.gz", _books(day, [("1.o", 7, "14:50", 3.0, 3.2, None)]))
    data = build(LocalSource(tmp_path), today=TODAY, cache=False, traders=[T("horse-live")])
    d = data["sports"][0]["traders"][0]["today"]
    assert d["forecast"] is None and d["method"].startswith("no forecast: too few past races")


def test_a_final_past_day_is_cached_and_read_back(tmp_path, monkeypatch):
    import dashboard.day_forecast as dfc
    monkeypatch.setattr(dfc, "MIN_POOL", 1)
    d = "2026-11-03"
    _put(tmp_path, "pred", f"trading/live/{d}/ledger.csv", pd.DataFrame(_horse_fill(d, "1.a", 10, 4.0, "13:00", "14:00", -10)))
    _put_gz(tmp_path, "capture", f"betfair_live/{d}/books.csv.gz",
            _books(d, [("1.a", 7, "13:50", 3.0, 3.2, None), ("1.a", 7, "14:01", 3.0, 3.2, 3.0)]))
    src = LocalSource(tmp_path)
    o1 = dfc.past_obs(src, "horse-live", "pred", "trading/live", [d])
    assert (tmp_path / "pred" / dfc.CACHE / "horse-live" / f"{d}.csv").exists()
    (tmp_path / "capture" / "betfair_live" / d / "books.csv.gz").unlink()        # the cache, not the books, is read
    o2 = dfc.past_obs(src, "horse-live", "pred", "trading/live", [d])
    assert len(o1) == len(o2) == 2 and sorted(o2.g) == [0, 5]


def test_the_record_starts_at_the_traders_rule_change(tmp_path):
    t = T("horse-live")
    assert t.record_from == "2026-10-10"
    for d in ("2026-10-08", "2026-10-09", "2026-10-11"):
        _put(tmp_path, "pred", f"trading/live/{d}/ledger.csv", horse_ledger(d, [("1.x", 100, 0.1, 5, 0.1)]))
    days, _ = trader_days(LocalSource(tmp_path), t, date(2026, 10, 12), cache=False)
    s = summarise(t, days, date(2026, 10, 12))
    assert s["record"]["days"] == 1                              # 10 Oct on: only the 11th
    assert s["all_time"]["days"] == 3                            # but every day counts in the P&L


def test_greyhound_paper_reads_its_settled_file(tmp_path):
    d = (TODAY - timedelta(days=1)).isoformat()
    base = f"sources/greyhound/live/paper/{d}"
    _put(tmp_path, "capture", f"{base}/ledger.csv", pd.DataFrame(
        [{"time_utc": f"{d}T12:00:00Z", "market_id": "1.g", "selection_id": 5, "track": "Romford", "race_time": "12:00",
          "action": "back", "status": "SUCCESS", "matched": 5, "edge": 0.3}]))
    _put(tmp_path, "capture", f"{base}/settled.csv", pd.DataFrame(
        [{"market_id": "1.g", "selection_id": 5, "track": "Romford", "race_time": "12:00", "matched": 5, "edge": 0.3,
          "pnl": 7.5}, {"market_id": "1.g", "selection_id": 6, "track": "Romford", "race_time": "12:00",
                        "matched": 5, "edge": 0.2, "pnl": None}]))
    days, _ = trader_days(LocalSource(tmp_path), T("greyhound-paper"), TODAY, cache=False)
    m = days[d].iloc[0]
    assert m.pnl == pytest.approx(7.5) and m.staked == pytest.approx(10) and m.ev == pytest.approx(2.5)


def test_early_lay_counts_the_locked_pnl_and_settles_the_held_lays_with_the_result(tmp_path):
    d = TODAY.isoformat()
    base = f"sources/greyhound/early_lay/live/{d}"
    rows = [
        {"time_utc": f"{d}T12:00:00Z", "market_id": "1.a", "selection_id": 1, "track": "Hove", "race_time": "12:00",
         "action": "lay", "price": 6.0, "size": 4, "matched": 4},
        {"time_utc": f"{d}T12:01:00Z", "market_id": "1.a", "selection_id": 1, "action": "green", "lay_matched": 4,
         "locked_pnl": 0.8},
        {"time_utc": f"{d}T13:00:00Z", "market_id": "1.b", "selection_id": 2, "track": "Hove", "race_time": "13:00",
         "action": "lay", "price": 5.0, "size": 2, "matched": 2},
        {"time_utc": f"{d}T13:01:00Z", "market_id": "1.b", "selection_id": 2, "action": "hold", "size": 2,
         "avg_price": 5.0, "liability": 8.0, "lay_matched": 2},
    ]
    _put(tmp_path, "capture", f"{base}/ledger.csv", pd.DataFrame(rows))
    src = LocalSource(tmp_path)
    days, _ = trader_days(src, EL(), TODAY, cache=False)
    m = days[d].set_index("market_id")
    assert m.loc["1.a", "pnl"] == pytest.approx(0.8) and m.loc["1.a", "settled"]
    assert not m.loc["1.b", "settled"] and m.loc["1.b", "open_liability"] == pytest.approx(8.0)
    # the recorder's settled books arrive: the held lay's dog lost, so the lay keeps its stake less 2%
    books = pd.DataFrame([{"polled_utc": f"{d}T13:10:00Z", "source": "final", "market_id": "1.b", "selection_id": 2,
                           "runner_status": "LOSER", "sp_actual": 4.2}])
    p = tmp_path / "capture" / f"betfair_live/{d}/books_greyhound.csv.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    books.to_csv(p, index=False, compression="gzip")
    days, _ = trader_days(src, EL(), TODAY, cache=False)
    m = days[d].set_index("market_id")
    assert m.loc["1.b", "settled"] and m.loc["1.b", "pnl"] == pytest.approx(2 * 0.98)
    s = summarise(EL(), days, TODAY)
    assert s["today"]["pnl"] == pytest.approx(0.8 + 1.96) and s["today"]["volume"] == pytest.approx(6.0)


def test_football_stays_open_until_the_nightly_settle(tmp_path):
    d = TODAY.isoformat()
    base = f"sources/football/live/paper/{d}"
    _put(tmp_path, "capture", f"{base}/ledger.csv", pd.DataFrame(
        [{"time_utc": f"{d}T10:00:00Z", "market_id": "1.f", "event": "A v B", "kickoff_utc": f"{d}T15:00:00Z",
          "action": "back", "status": "SUCCESS", "matched": 2, "ev": 0.1}]))
    src = LocalSource(tmp_path)
    days, _ = trader_days(src, T("football-paper"), TODAY, cache=False)
    assert not days[d].settled.iloc[0] and pd.isna(days[d].pnl.iloc[0])
    _put(tmp_path, "capture", f"{base}/settled.csv", pd.DataFrame(
        [{"market_id": "1.f", "event": "A v B", "kickoff_utc": f"{d}T15:00:00Z", "matched": 2, "ev": 0.1,
          "settled": True, "pnl": 3.1}]))
    days, _ = trader_days(src, T("football-paper"), TODAY, cache=False)
    assert days[d].settled.iloc[0] and days[d].pnl.iloc[0] == pytest.approx(3.1)


def test_the_cache_holds_settled_days_and_rereads_open_ones(tmp_path):
    old = (TODAY - timedelta(days=6)).isoformat()
    _put(tmp_path, "pred", f"trading/live/{old}/ledger.csv", horse_ledger(old, [("1.x", 100, 0.1, 5, 0.1)]))
    src = LocalSource(tmp_path)
    trader_days(src, T("horse-live"), TODAY)
    assert (tmp_path / "pred" / f"dashboard/cache/v2/horse-live/{old}.csv").exists()
    # the ledger goes; the settled day still comes from the cache
    (tmp_path / "pred" / f"trading/live/{old}/ledger.csv").unlink()
    days, _ = trader_days(src, T("horse-live"), TODAY)
    assert days[old].pnl.iloc[0] == pytest.approx(4.9)


def test_every_sport_is_on_the_page_and_cricket_says_no_data_found(tmp_path):
    data = build(LocalSource(tmp_path), TODAY, cache=False)
    assert [s["id"] for s in data["sports"]] == ["horse_racing", "greyhounds", "football", "cricket"]
    cricket = data["sports"][-1]
    assert cricket["traders"] == []
    assert all(not t["has_data"] for s in data["sports"] for t in s["traders"])
    md = summary_md(data)
    assert "No data found: no trader for this sport." in md and "no data found" in md
    html = render(data)
    assert "<title>Ashcroft Traders</title>" in html and "No data found" in html
    assert "—" not in html                                       # hyphens, never em dashes
    # the JSON cannot close its own script tag
    data["sports"][0]["label"] = "</script><b>"
    assert "</script><b>" not in render(data)


def test_one_traders_fault_does_not_blank_the_page(tmp_path):
    d = TODAY.isoformat()
    p = tmp_path / "pred" / f"trading/live/{d}/ledger.csv"
    p.parent.mkdir(parents=True)
    p.write_text("event,market_id,matched\nback,1.a,not-a-number\n")
    data = build(LocalSource(tmp_path), TODAY, cache=False)
    assert data["sports"][0]["traders"][0]["id"] == "horse-live"


def test_the_cli_writes_both_files(tmp_path):
    d = TODAY.isoformat()
    _put(tmp_path / "src", "pred", f"trading/live/{d}/ledger.csv", horse_ledger(d, [("1.a", 10, 0.1, 5, 0.1)]))
    out = tmp_path / "out"
    assert main(["--root", str(tmp_path / "src"), "--date", d, "--out", str(out), "--no-cache"]) == 0
    j = json.loads((out / "traders.json").read_text())
    assert j["sports"][0]["traders"][0]["today"]["pnl"] == pytest.approx(4.9)
    assert (out / "index.html").read_text().startswith("<!doctype html>")


def test_clv_and_the_edge_flag(tmp_path):
    """CLV is the stake-weighted mean of each settled position's CLV; the flag reads the last 7 days against the
    record: too few bets grey, at or under zero red, a fading or uncertain edge amber, else green."""
    from dashboard.traders import MIN_FLAG_BETS

    def ledger(day, clvs, stake=10):
        rows = []
        for i, c in enumerate(clvs):
            mid = f"1.{i}"
            rows.append({"ts": f"{day}T09:00:00Z", "event": "back", "market_id": mid, "venue": "Ascot",
                         "off": f"{day} 14:00", "selection_id": 1, "matched": stake, "edge": 0.05})
            rows.append({"ts": f"{day}T15:00:00Z", "event": "settle", "market_id": mid, "selection_id": 1,
                         "pnl": -stake, "matched": stake, "clv": c})
        return pd.DataFrame(rows)

    t = T("horse-live")
    # 40 bets a day at +10% CLV for 7 days: green
    for i in range(7):
        d = (TODAY - timedelta(days=i)).isoformat()
        _put(tmp_path, "pred", f"trading/live/{d}/ledger.csv", ledger(d, [0.10 + 0.01 * (j % 3 - 1) for j in range(40)]))
    days, last = trader_days(LocalSource(tmp_path), t, TODAY, cache=False)
    e = summarise(t, days, TODAY, last)["edge"]
    assert e["clv_today"] == pytest.approx(0.10, abs=0.002) and e["clv_7d_n"] == 280
    assert e["flag"] == "green" and e["clv_7d_lo"] > 0

    # the last 7 days at -2%: red
    neg = tmp_path / "neg"
    for i in range(7):
        d = (TODAY - timedelta(days=i)).isoformat()
        _put(neg, "pred", f"trading/live/{d}/ledger.csv", ledger(d, [-0.02] * 40))
    days, _ = trader_days(LocalSource(neg), t, TODAY, cache=False)
    assert summarise(t, days, TODAY)["edge"]["flag"] == "red"

    # an edge at a third of the record's: amber
    fade = tmp_path / "fade"
    for i in range(20):
        d = (TODAY - timedelta(days=i)).isoformat()
        _put(fade, "pred", f"trading/live/{d}/ledger.csv", ledger(d, [0.02 if i < 7 else 0.30] * 40))
    days, _ = trader_days(LocalSource(fade), t, TODAY, cache=False)
    e = summarise(t, days, TODAY)["edge"]
    assert e["flag"] == "amber" and e["clv_record"] > e["clv_7d"] > 0

    # too few bets: grey
    few = tmp_path / "few"
    _put(few, "pred", f"trading/live/{TODAY.isoformat()}/ledger.csv", ledger(TODAY.isoformat(), [0.1] * (MIN_FLAG_BETS - 1)))
    days, _ = trader_days(LocalSource(few), t, TODAY, cache=False)
    e = summarise(t, days, TODAY)["edge"]
    assert e["flag"] == "grey" and e["ev_today"] == pytest.approx(0.05 * 10 * (MIN_FLAG_BETS - 1))
    assert e["ev_today_pct"] == pytest.approx(0.05)


def test_the_clv_by_day_since_the_first_day(tmp_path):
    """The owner, 5 Nov 2026: "show the CLV per day since we started". Each traded day's stake-weighted CLV, its bets
    and stake, oldest first; the page folds it into the trader's details."""
    from dashboard.page import render
    t = T("horse-live")
    for i, c in enumerate([0.12, -0.03, 0.05]):
        d = (TODAY - timedelta(days=2 - i)).isoformat()
        rows = [{"ts": f"{d}T09:00:00Z", "event": "back", "market_id": "1.1", "venue": "Ascot", "off": f"{d} 14:00",
                 "selection_id": 1, "matched": 20, "edge": 0.05},
                {"ts": f"{d}T09:01:00Z", "event": "back", "market_id": "1.2", "venue": "Ascot", "off": f"{d} 15:00",
                 "selection_id": 2, "matched": 10, "edge": 0.05},
                {"ts": f"{d}T15:00:00Z", "event": "settle", "market_id": "1.1", "selection_id": 1, "pnl": 3,
                 "matched": 20, "clv": c},
                {"ts": f"{d}T16:00:00Z", "event": "settle", "market_id": "1.2", "selection_id": 2, "pnl": -1,
                 "matched": 10, "clv": c + 0.03}]
        _put(tmp_path, "pred", f"trading/live/{d}/ledger.csv", pd.DataFrame(rows))
    days, last = trader_days(LocalSource(tmp_path), t, TODAY, cache=False)
    cd = summarise(t, days, TODAY, last)["clv_daily"]
    assert [r["day"] for r in cd] == [(TODAY - timedelta(days=k)).isoformat() for k in (2, 1, 0)]
    for r, c in zip(cd, [0.12, -0.03, 0.05]):
        assert r["clv"] == pytest.approx((20 * c + 10 * (c + 0.03)) / 30, abs=1e-4) and r["bets"] == 2
        assert r["staked"] == pytest.approx(30)
    assert "function clvDays" in render({"sports": [], "today": TODAY.isoformat(), "generated_utc": "x"})


def test_an_early_lays_clv_is_the_bsp_over_the_price_laid(tmp_path):
    d = TODAY.isoformat()
    rows = [{"time_utc": f"{d}T12:00:00Z", "market_id": "1.a", "selection_id": 1, "track": "Hove", "race_time": "12:00",
             "action": "lay", "price": 4.0, "size": 5, "matched": 5},
            {"time_utc": f"{d}T12:01:00Z", "market_id": "1.a", "selection_id": 1, "action": "green", "lay_matched": 5,
             "lay_price": 4.0, "locked_pnl": 0.5}]
    _put(tmp_path, "capture", f"sources/greyhound/early_lay/live/{d}/ledger.csv", pd.DataFrame(rows))
    books = pd.DataFrame([{"polled_utc": f"{d}T12:10:00Z", "source": "final", "market_id": "1.a", "selection_id": 1,
                           "runner_status": "LOSER", "sp_actual": 5.0}])
    p = tmp_path / "capture" / f"betfair_live/{d}/books_greyhound.csv.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    books.to_csv(p, index=False, compression="gzip")
    days, _ = trader_days(LocalSource(tmp_path), EL(), TODAY, cache=False)
    e = summarise(EL(), days, TODAY)["edge"]
    assert e["clv_today"] == pytest.approx(5.0 / 4.0 - 1) and e["ev_today"] is None


def test_a_trader_counts_only_from_its_start_day(tmp_path):
    """The owner, 4 Nov 2026: the paper agents only start from now. Days in a ledger before a trader's start_from
    (a replay written before it went on) are not its P&L."""
    from dataclasses import replace
    from dashboard.traders import AGENTS_FROM
    assert AGENTS_FROM == "2026-11-04"
    assert not [t for t in TRADERS if t.id.startswith("horse-agent-")]      # retired, the owner, 4 Nov 2026
    t = replace(T("horse-live"), start_from=TODAY.isoformat(), record_from=None)
    for i, r in enumerate([100, -50, 25]):
        d = (TODAY - timedelta(days=3 - i)).isoformat()
        _put(tmp_path, "pred", f"trading/live/{d}/ledger.csv", horse_ledger(d, [("1.x", 100, 0.1, r, 0.1)]))
    days, _ = trader_days(LocalSource(tmp_path), t, TODAY, cache=False)
    s = summarise(t, days, TODAY)
    assert days == {} and s["has_data"] is False and s["all_time"]["pnl"] is None and s["daily"] == []
    d = TODAY.isoformat()
    _put(tmp_path, "pred", f"trading/live/{d}/ledger.csv", horse_ledger(d, [("1.y", 10, 0.1, None, None)]))
    days, _ = trader_days(LocalSource(tmp_path), t, TODAY, cache=False)
    s = summarise(t, days, TODAY)
    assert list(days) == [d] and s["has_data"] and s["all_time"]["pnl"] == 0.0 and s["month"]["forecast"] is None


def test_the_latest_days_race_by_race(tmp_path):
    """The owner, 4 Nov 2026: score the paper models race by race. Each trader gives its latest RACE_DAYS traded days
    as a row a race: course, off, staked, P&L once settled, CLV."""
    from dashboard.traders import RACE_DAYS
    t = T("horse-live")
    for i in range(RACE_DAYS + 1):
        d = (TODAY - timedelta(days=RACE_DAYS - i)).isoformat()
        _put(tmp_path, "pred", f"trading/live/{d}/ledger.csv",
             horse_ledger(d, [("1.a", 100, 0.1, 20, 0.4), ("1.b", 50, 0.1, None, None)]))
    days, _ = trader_days(LocalSource(tmp_path), t, TODAY, cache=False)
    rs = summarise(t, days, TODAY)["races"]
    assert sorted({r["day"] for r in rs}) == sorted(days)[-RACE_DAYS:]
    a = next(r for r in rs if r["market_id"] == "1.a" and r["day"] == TODAY.isoformat())
    b = next(r for r in rs if r["market_id"] == "1.b" and r["day"] == TODAY.isoformat())
    assert a["course"] == "Kempton" and a["staked"] == 100 and a["pnl"] == pytest.approx(19.6)
    assert b["pnl"] is None and b["staked"] == 50          # still to run: no P&L yet


def test_the_morning_models_are_separate_lines_from_one_ledger(tmp_path):
    """One live ledger holds both morning models (morning/live.py); each model is its own trader on the page, its P&L
    after 2% on the market's net winnings, settled when the market's backs have results, CLV at the BSP."""
    day = TODAY.isoformat()
    rows = [
        {"ts": f"{day}T10:00:05Z", "model": "steamer", "event": "back", "market_id": "1.a", "venue": "Kempton",
         "off": "14:00", "selection_id": 1, "matched": 10.0, "avg_price": 6.0, "result": "WINNER", "pnl": 50.0,
         "bsp": 5.0, "clv": 0.2},
        {"ts": f"{day}T10:00:06Z", "model": "steamer", "event": "back", "market_id": "1.b", "venue": "Lingfield",
         "off": "15:00", "selection_id": 2, "matched": 5.0, "avg_price": 11.0, "result": "UNSETTLED", "pnl": None},
        {"ts": f"{day}T10:00:07Z", "model": "steamer", "event": "not_filled", "market_id": "1.c", "selection_id": 3,
         "matched": 0.0},
        {"ts": f"{day}T10:00:08Z", "model": "quick_return", "event": "back", "market_id": "1.a", "venue": "Kempton",
         "off": "14:00", "selection_id": 4, "matched": 12.5, "avg_price": 5.0, "result": "LOSER", "pnl": -12.5,
         "bsp": 4.0},
    ]
    _put(tmp_path, "pred", f"morning/live/{day}/ledger.csv", pd.DataFrame(rows))
    src = LocalSource(tmp_path)
    st, _ = trader_days(src, T("horse-morning-steamer"), TODAY, cache=False)
    m = st[day].set_index("market_id")
    assert sorted(m.index) == ["1.a", "1.b"]                    # the unfilled order is not a market traded
    assert m.loc["1.a", "pnl"] == pytest.approx(49.0) and m.loc["1.a", "settled"]
    assert pd.isna(m.loc["1.b", "pnl"]) and not m.loc["1.b", "settled"]
    assert m.loc["1.a", "clv_ws"] / m.loc["1.a", "clv_s"] == pytest.approx(0.2)
    qr, _ = trader_days(src, T("horse-morning-quick-return"), TODAY, cache=False)
    q = qr[day].set_index("market_id")
    assert list(q.index) == ["1.a"] and q.loc["1.a", "pnl"] == pytest.approx(-12.5)
    assert q.loc["1.a", "clv_ws"] / q.loc["1.a", "clv_s"] == pytest.approx(0.25)   # 5.0 / 4.0 - 1, from the BSP


def test_the_morning_paper_forward_test_is_read_a_file_a_day(tmp_path):
    """research/paper/morning_edges.py writes bets/<day>.csv: one strategy's WIN backs only, from 5 Nov 2026."""
    rows = [
        {"strategy": "steam_am", "market": "win", "event": "back", "market_id": "1.x", "label": "Wolverhampton",
         "off": "18:00", "matched": 20.0, "price": 6.0, "bsp": 6.0, "result": "WINNER", "pnl": 100.0},
        {"strategy": "steam_am", "market": "place", "event": "back", "market_id": "1.y", "label": "Wolverhampton",
         "off": "18:00", "matched": 50.0, "price": 3.0, "bsp": 3.0, "result": "LOSER", "pnl": -50.0},
        {"strategy": "qr_am", "market": "win", "event": "back", "market_id": "1.x", "label": "Wolverhampton",
         "off": "18:00", "matched": 25.0, "price": 5.0, "bsp": 5.0, "result": "LOSER", "pnl": -25.0},
    ]
    for d in ("2026-11-04", "2026-11-05"):
        _put(tmp_path, "pred", f"paper/morning_edges/bets/{d}.csv", pd.DataFrame(rows))
    src = LocalSource(tmp_path)
    days, _ = trader_days(src, T("horse-morning-steamer-paper"), date(2026, 11, 6), cache=False)
    assert list(days) == ["2026-11-05"]                         # the forward test starts 5 Nov
    m = days["2026-11-05"].set_index("market_id")
    assert list(m.index) == ["1.x"] and m.loc["1.x", "pnl"] == pytest.approx(98.0)
    q, _ = trader_days(src, T("horse-morning-quick-return-paper"), date(2026, 11, 6), cache=False)
    assert q["2026-11-05"].pnl.sum() == pytest.approx(-25.0)


# --------------------------------------------------------------------------------------------------------------------
# The early lay's switch on its tile, and its record from 11 Nov (the owner, 11 Nov 2026: "Show it. Start tracking
# from fresh")
# --------------------------------------------------------------------------------------------------------------------

def test_the_live_early_lay_is_tracked_from_11_nov_and_its_earlier_days_are_left_out(tmp_path):
    from dashboard.traders import EARLY_LAY_FROM
    assert EARLY_LAY_FROM == "2026-11-11" and T("greyhound-early-lay").start_from == EARLY_LAY_FROM
    assert T("greyhound-early-lay-paper").start_from is None          # the paper record is not reset
    led = pd.DataFrame([{"time_utc": "2026-11-04T12:00:00Z", "market_id": "1.1", "selection_id": 1, "track": "Romford",
                         "race_time": "12:05", "action": "lay", "matched": 0.0}])
    for day in ("2026-11-04", "2026-11-11"):
        _put(tmp_path, "capture", f"sources/greyhound/early_lay/live/{day}/ledger.csv", led)
    days, _ = trader_days(LocalSource(tmp_path), T("greyhound-early-lay"), date(2026, 11, 11), cache=False)
    assert sorted(days) == ["2026-11-11"]


def test_the_switch_is_read_as_the_early_lay_itself_reads_it(tmp_path):
    from dashboard.traders import switch_state
    from greyhound import early_lay as el
    t, src = T("greyhound-early-lay"), LocalSource(tmp_path)
    assert t.switch == "GREYHOUND_EARLY_LAY"
    for env in ({}, {"GREYHOUND_EARLY_LAY": "paper"}, {"GREYHOUND_EARLY_LAY": "live"},
                {"GREYHOUND_EARLY_LAY": " LIVE ", "GREYHOUND_EARLY_LAY_DAILY_LOSS": "50"},
                {"GREYHOUND_EARLY_LAY": "live", "GREYHOUND_EARLY_LAY_DAILY_LOSS": "-5"},
                {"GREYHOUND_EARLY_LAY": "live", "GREYHOUND_EARLY_LAY_DAILY_LOSS": "abc"}):
        w = switch_state(src, t, env)
        assert w["loss_limit"] == el.daily_loss_limit(env)
        if env:
            assert (w["state"] == "live") == el.switch_on(env)
        # real orders exactly when the trader would send them: the switch, a loss limit and no STOP
        assert w["armed"] == (el.switch_on(env) and el.daily_loss_limit(env) is not None)
    assert switch_state(src, t, {})["state"] is None                     # not in this run's environment: unread
    assert switch_state(src, t, {"GREYHOUND_EARLY_LAY": "paper"})["label"].startswith("Paper")
    assert switch_state(src, T("greyhound-early-lay-paper"), {"GREYHOUND_EARLY_LAY": "live"}) is None


def test_a_stop_object_disarms_the_switch(tmp_path):
    from dashboard.traders import switch_state
    env = {"GREYHOUND_EARLY_LAY": "live", "GREYHOUND_EARLY_LAY_DAILY_LOSS": "50"}
    t = T("greyhound-early-lay")
    assert switch_state(LocalSource(tmp_path), t, env)["armed"] is True
    for key in ("greyhound/STOP", "trading/STOP"):
        p = tmp_path / "capture" / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("stop")
        w = switch_state(LocalSource(tmp_path), t, env)
        assert w["stop"] is True and w["armed"] is False and w["label"].startswith("Stopped")
        p.unlink()


def test_the_switch_reaches_the_page_and_the_job_reads_the_variables_as_paper_models_does(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.setenv("GREYHOUND_EARLY_LAY", "paper")
    monkeypatch.delenv("GREYHOUND_EARLY_LAY_DAILY_LOSS", raising=False)
    d = build(LocalSource(tmp_path), today=date(2026, 11, 11), cache=False)
    el = next(t for s in d["sports"] for t in s["traders"] if t["id"] == "greyhound-early-lay")
    assert el["switch"]["state"] == "paper" and el["switch"]["armed"] is False and el["switch"]["from"] == "2026-11-11"
    switched = {t.id for t in TRADERS if t.switch}
    assert switched == {"greyhound-early-lay", "greyhound-take-close"}
    others = [t for s in d["sports"] for t in s["traders"] if t["id"] not in switched]
    assert all("switch" not in t for t in others)
    html = render(d)
    assert "function switchLine" in html and "switchLine(t) + edgeLine(t)" in html
    wf = (Path(__file__).resolve().parents[1] / ".github/workflows/trader-dashboard.yml").read_text()
    pm = (Path(__file__).resolve().parents[1] / ".github/workflows/paper-models.yml").read_text()
    for ln in ("GREYHOUND_EARLY_LAY: ${{ vars.GREYHOUND_EARLY_LAY || 'paper' }}",
               "GREYHOUND_EARLY_LAY_DAILY_LOSS: ${{ vars.GREYHOUND_EARLY_LAY_DAILY_LOSS }}"):
        assert ln in wf
    assert "vars.GREYHOUND_EARLY_LAY || 'paper'" in pm                   # the same default as the trader's job
    assert "BETFAIR" not in wf and "--live" not in wf                   # the dashboard never trades


# The take and close (the owner, 18 Nov 2026: "Run the take-and-close LIVE"; 19 Nov: "add a tile"): its backs closed
# at the SP, its own settle, its own STOP

def TC():
    """The take and close's reader without its start (18 Nov 2026): the ledger tests below are dated 4 Nov."""
    return replace(T("greyhound-take-close"), start_from=None)


def test_take_close_reads_its_own_settle_and_else_the_settled_books_by_its_own_rule(tmp_path):
    from greyhound.take_close import close_pnl
    d = TODAY.isoformat()
    base = f"sources/greyhound/take_close/live/{d}"
    rows = [
        {"time_utc": f"{d}T12:00:00Z", "market_id": "1.a", "action": "decided"},
        {"time_utc": f"{d}T12:00:01Z", "market_id": "1.a", "selection_id": 1, "track": "Romford",
         "race_time": "12:05", "action": "back_matched", "price": 6.0, "matched": 40},
        {"time_utc": f"{d}T12:00:02Z", "market_id": "1.a", "selection_id": 1, "action": "sp_lay", "liability": 200},
        {"time_utc": f"{d}T12:20:00Z", "market_id": "1.a", "action": "settled", "pnl": 6.17},
        {"time_utc": f"{d}T13:00:00Z", "market_id": "1.b", "action": "decided"},
        {"time_utc": f"{d}T13:00:01Z", "market_id": "1.b", "selection_id": 2, "track": "Romford",
         "race_time": "13:05", "action": "back_matched", "price": 4.0, "matched": 20},
        {"time_utc": f"{d}T13:00:02Z", "market_id": "1.b", "selection_id": 2, "action": "sp_lay", "liability": 60},
        {"time_utc": f"{d}T14:00:00Z", "market_id": "1.c", "action": "no_back", "note": "stopped (S3 STOP)"},
    ]
    _put(tmp_path, "capture", f"{base}/ledger.csv", pd.DataFrame(rows))
    src = LocalSource(tmp_path)
    days, last = trader_days(src, TC(), TODAY, cache=False)
    m = days[d].set_index("market_id")
    assert set(m.index) == {"1.a", "1.b"}                                # a market with no back is no market traded
    assert m.loc["1.a", "settled"] and m.loc["1.a", "pnl"] == pytest.approx(6.17)
    assert not m.loc["1.b", "settled"] and m.loc["1.b", "open_liability"] == pytest.approx(20)
    assert m.loc["1.a", "staked"] == pytest.approx(40) and last == f"{d}T14:00:00Z"
    # the recorder's settled books: 1.b's dog lost at a BSP of 3.5, so the back's winnings laid at the SP net
    books = pd.DataFrame([{"polled_utc": f"{d}T13:10:00Z", "source": "final", "market_id": "1.b", "selection_id": 2,
                           "runner_status": "LOSER", "sp_actual": 3.5},
                          {"polled_utc": f"{d}T12:10:00Z", "source": "final", "market_id": "1.a", "selection_id": 1,
                           "runner_status": "LOSER", "sp_actual": 5.0}])
    _put_gz(tmp_path, "capture", f"betfair_live/{d}/books_greyhound.csv.gz", books)
    days, _ = trader_days(src, TC(), TODAY, cache=False)
    m = days[d].set_index("market_id")
    gross = close_pnl(20, 4.0, 3.5, False)                               # the trader's own rule
    assert m.loc["1.b", "settled"] and m.loc["1.b", "pnl"] == pytest.approx(gross - 0.02 * max(gross, 0))
    assert m.loc["1.a", "pnl"] == pytest.approx(6.17)                    # its own settle row stands
    # CLV: the price taken over the BSP, less 1, stake-weighted
    s = summarise(TC(), days, TODAY)
    assert s["edge"]["clv_today"] == pytest.approx((40 * (6 / 5 - 1) + 20 * (4 / 3.5 - 1)) / 60, abs=1e-3)
    assert s["today"]["volume"] == pytest.approx(60)


def test_the_take_close_switch_is_read_as_the_trader_reads_it_with_its_own_stop(tmp_path):
    from dashboard.traders import switch_state
    from greyhound import take_close as tc
    t = T("greyhound-take-close")
    assert t.switch == "GREYHOUND_TAKE_CLOSE" and t.start_from == "2026-11-18" and t.mode == "live"
    assert tuple(t.stop_keys) == tc.STOP_KEYS                            # the trader's own STOP objects
    src = LocalSource(tmp_path)
    for env in ({}, {"GREYHOUND_TAKE_CLOSE": "paper"}, {"GREYHOUND_TAKE_CLOSE": "live"},
                {"GREYHOUND_TAKE_CLOSE": "live", "GREYHOUND_TAKE_CLOSE_DAILY_LOSS": "200"}):
        w = switch_state(src, t, env)
        assert w["armed"] == (tc.switch_on(env) and tc.daily_loss_limit(env) is not None)
    env = {"GREYHOUND_TAKE_CLOSE": "live", "GREYHOUND_TAKE_CLOSE_DAILY_LOSS": "200"}
    p = tmp_path / "capture" / "greyhound/STOP"                          # the early lay's STOP does not stop it
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("stop")
    assert switch_state(src, t, env)["armed"] is True
    assert switch_state(src, T("greyhound-early-lay"),
                        {"GREYHOUND_EARLY_LAY": "live", "GREYHOUND_EARLY_LAY_DAILY_LOSS": "50"})["stop"] is True
    for key in tc.STOP_KEYS:
        q = tmp_path / "capture" / key
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text("stop")
        w = switch_state(src, t, env)
        assert w["stop"] is True and w["armed"] is False and w["label"].endswith("no new backs")
        q.unlink()
    wf = (Path(__file__).resolve().parents[1] / ".github/workflows/trader-dashboard.yml").read_text()
    pm = (Path(__file__).resolve().parents[1] / ".github/workflows/paper-models.yml").read_text()
    for ln in ("GREYHOUND_TAKE_CLOSE: ${{ vars.GREYHOUND_TAKE_CLOSE || 'paper' }}",
               "GREYHOUND_TAKE_CLOSE_DAILY_LOSS: ${{ vars.GREYHOUND_TAKE_CLOSE_DAILY_LOSS }}"):
        assert ln in wf
    assert "vars.GREYHOUND_TAKE_CLOSE || 'paper'" in pm                  # the same default as the trader's job
