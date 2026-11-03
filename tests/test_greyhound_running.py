"""The greyhound paper trader's running table (greyhound.live.running): a row per market looked at, P&L only once the
market's settled book is in, the running total in race order."""

from __future__ import annotations

import io
from datetime import date

import pandas as pd

import betfair_recorder
from greyhound import live
from sources.common import Store


def _ledger():
    rows = [
        # market 1: a back of GBP5 at 6.0, matched; its winnings laid at SP (liability 25)
        {"market_id": "1.1", "selection_id": 11, "track": "Romford", "race_time": "12:03", "action": "back",
         "status": "SUCCESS", "matched": 5, "avg_price": 6.0, "lay_liability": 25},
        {"market_id": "1.1", "selection_id": 12, "track": "Romford", "race_time": "12:03", "action": "none",
         "status": "", "matched": 0, "avg_price": None, "lay_liability": None},
        # market 2: looked at, nothing backed
        {"market_id": "1.2", "selection_id": 21, "track": "Hove", "race_time": "12:10", "action": "none",
         "status": "", "matched": 0, "avg_price": None, "lay_liability": None},
        # market 3: backed, not settled yet
        {"market_id": "1.3", "selection_id": 31, "track": "Towcester", "race_time": "12:20", "action": "back",
         "status": "SUCCESS", "matched": 5, "avg_price": 4.0, "lay_liability": 15},
    ]
    return pd.DataFrame(rows, columns=live.LEDGER_FIELDS)


def test_running_table(tmp_path, monkeypatch):
    store = Store(root=str(tmp_path))
    buf = io.StringIO()
    _ledger().to_csv(buf, index=False)
    store.put("greyhound/live/paper/2026-11-03/ledger.csv", buf.getvalue().encode())
    final = pd.DataFrame([{"source": "final", "market_id": "1.1", "selection_id": 11, "runner_status": "LOSER",
                           "sp_actual": 5.0, "polled_utc": "2026-11-03T12:06:00Z"}])
    monkeypatch.setattr(betfair_recorder, "_read_day_csv", lambda day, name, s3=None: final)
    r = live.running(store, date(2026, 11, 3))
    assert r["markets_seen"] == 3 and r["markets_bet"] == 2 and r["bets"] == 2
    m = {x["market_id"]: x for x in r["markets"]}
    # lost: -5 on the back, +25/(5-1) x 0.98 on the lay at SP
    assert abs(m["1.1"]["pnl"] - (-5 + 25 / 4 * 0.98)) <= 0.005 + 1e-9 and m["1.1"]["pnl_held"] == -5.0
    assert m["1.1"]["clv%"] == 20.0 and m["1.1"]["result"] == "LOSER"
    assert m["1.2"]["bets"] == 0 and m["1.3"]["pnl"] is None and m["1.3"]["result"] == "pending"
    assert r["markets_settled"] == 1 and r["pending"] == 1
    md = live.running_markdown(r)
    assert "| 12:03 | Romford | 1 |" in md and "Hove" not in md.split("|---|")[1]


def test_running_without_a_ledger(tmp_path):
    r = live.running(Store(root=str(tmp_path)), date(2026, 11, 3))
    assert r["markets"] == [] and "no ledger" in r["note"]
    assert "no ledger" in live.running_markdown(r)


def test_the_cli_writes_the_table(tmp_path, monkeypatch):
    store = Store(root=str(tmp_path))
    buf = io.StringIO()
    _ledger().to_csv(buf, index=False)
    store.put("greyhound/live/paper/2026-11-03/ledger.csv", buf.getvalue().encode())
    monkeypatch.setattr(betfair_recorder, "_read_day_csv", lambda day, name, s3=None: None)
    out = tmp_path / "running.md"
    assert live.main(["--store", str(tmp_path), "--running", "--date", "2026-11-03", "--out", str(out)]) == 0
    assert "Romford" in out.read_text() and store.get("greyhound/live/paper/2026-11-03/running.json")
