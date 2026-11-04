"""The traders' dashboard (dashboard/): each trader's ledger read as its own settle step reads it, the forecast by the
owner's day-report method, the page and the job summary. The ledgers here are small hand-made test inputs laid out as
the traders write them; nothing is shown as a real figure."""
from __future__ import annotations

import json
from datetime import date, timedelta

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


def test_the_forecast_scales_the_expected_value_by_the_record(tmp_path):
    # five settled days from 28 Oct, each expected value 10 (100 x 0.1) and results 4, 6, 2, 8, 5 -> 25 / 50 = 0.5
    results = [4, 6, 2, 8, 5]
    for i, r in enumerate(results):
        d = (TODAY - timedelta(days=7 - i)).isoformat()
        _put(tmp_path, "pred", f"trading/live/{d}/ledger.csv", horse_ledger(d, [("1.x", 100, 0.1, r, 0.0001)]))
    day = TODAY.isoformat()
    # today: one settled (+10 after commission 0.0001 -> rounded), one open with expected value 40 x 0.5 = 20
    _put(tmp_path, "pred", f"trading/live/{day}/ledger.csv",
         horse_ledger(day, [("1.s", 100, 0.1, 10, 0.0001), ("1.o", 200, 0.2, None, None)]))
    src = LocalSource(tmp_path)
    t = T("horse-live")
    days, last = trader_days(src, t, TODAY, cache=False)
    s = summarise(t, days, TODAY, last)
    k = sum(r - 0.0001 for r in results) / 50.0
    assert s["record"]["days"] == 5
    assert s["today"]["forecast"] == pytest.approx(round(10 - 0.0001 + 40 * k, 2))
    assert s["today"]["forecast_lo"] < s["today"]["forecast"] < s["today"]["forecast_hi"]
    # the month: November to date (1 Nov's 5 and today's settled 10), today's rest, 26 days left x the mean day
    mean = sum(r - 0.0001 for r in results) / 5
    assert s["month"]["days_left"] == 26
    nov_settled = (5 - 0.0001) + (10 - 0.0001)
    assert s["month"]["forecast"] == pytest.approx(round(nov_settled + 40 * k + 26 * mean, 2), abs=0.02)
    assert s["month"]["pnl"] == pytest.approx(round(nov_settled, 2))
    assert s["all_time"]["pnl"] == pytest.approx(round(sum(results) + 10 - 6 * 0.0001, 2))
    assert s["all_time"]["days"] == 6 and s["all_time"]["days_up"] == 6


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
    days, _ = trader_days(src, T("greyhound-early-lay"), TODAY, cache=False)
    m = days[d].set_index("market_id")
    assert m.loc["1.a", "pnl"] == pytest.approx(0.8) and m.loc["1.a", "settled"]
    assert not m.loc["1.b", "settled"] and m.loc["1.b", "open_liability"] == pytest.approx(8.0)
    # the recorder's settled books arrive: the held lay's dog lost, so the lay keeps its stake less 2%
    books = pd.DataFrame([{"polled_utc": f"{d}T13:10:00Z", "source": "final", "market_id": "1.b", "selection_id": 2,
                           "runner_status": "LOSER", "sp_actual": 4.2}])
    p = tmp_path / "capture" / f"betfair_live/{d}/books_greyhound.csv.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    books.to_csv(p, index=False, compression="gzip")
    days, _ = trader_days(src, T("greyhound-early-lay"), TODAY, cache=False)
    m = days[d].set_index("market_id")
    assert m.loc["1.b", "settled"] and m.loc["1.b", "pnl"] == pytest.approx(2 * 0.98)
    s = summarise(T("greyhound-early-lay"), days, TODAY)
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
    assert (tmp_path / "pred" / f"dashboard/cache/horse-live/{old}.csv").exists()
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
    assert "<title>racing² Traders</title>" in html and "No data found" in html
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
