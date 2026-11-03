"""The early-lay forward test (greyhound/laytest.py): frozen once, never a day before its first, the primary and the
recorded variants scored with the limit study's own code, the gate held back until enough days."""

from __future__ import annotations

import json
from datetime import date

import pytest

from greyhound import laytest as lt
from research.greyhound import limit_study as ls
from sources.common import Store
from tests.test_greyhound_limit_study import synthetic_panel


@pytest.fixture()
def panel(monkeypatch):
    p = synthetic_panel(n_days=34, races=8)                 # 6 Oct to 8 Nov 2026
    monkeypatch.setattr(ls, "load", lambda *a, **k: p)
    return p


def test_freeze_once_and_never_a_day_before_the_first(tmp_path, panel):
    store = Store(root=str(tmp_path))
    f = lt.freeze(store, date(2026, 11, 3))
    assert f["first_test_day"] == "2026-11-04" and f["primary"]["gate"] == "drift"
    assert lt.freeze(store, date(2026, 11, 4))["first_test_day"] == "2026-11-04"    # the rule was fixed on 3 Nov
    assert lt.freeze(store, date(2026, 11, 20))["first_test_day"] == "2026-11-21"   # later: never before tomorrow
    with pytest.raises(ValueError):
        lt.track(store, f, [date(2026, 11, 3)])


def test_track_and_summary(tmp_path, panel, monkeypatch):
    store = Store(root=str(tmp_path))
    f = lt.freeze(store, date(2026, 11, 3))
    days = [date(2026, 11, d) for d in range(4, 9)]
    rep = lt.track(store, f, days)
    assert [r["day"] for r in rep] == [f"2026-11-0{d}" for d in range(4, 9)]
    assert all(r["primary_orders"] > 0 for r in rep)
    o = store.get_parquet("greyhound/laytest/days/2026-11-05.parquet")
    assert set(o.order) == {"bid", "join"} and set(o["mark"]) == {lt.MARK}
    s = lt.summary(store, f)
    assert s["days"] == 5 and s["gate"].startswith("not yet judged: 5 of 20 days")
    assert set(s["rules"]) == {"primary", *lt.SECONDARY} and s["rules"]["primary"]["orders"] > 0
    md = lt.markdown(s)
    assert "early-lay forward test" in md and "| primary |" in md


def test_the_gate_is_judged_after_enough_days(tmp_path, panel, monkeypatch):
    store = Store(root=str(tmp_path))
    f = lt.freeze(store, date(2026, 11, 3))
    lt.track(store, f, [date(2026, 11, d) for d in range(4, 9)])
    monkeypatch.setattr(lt, "READ_DAYS", 5)
    monkeypatch.setattr(lt, "MIN_TRADES", 1)
    assert lt.summary(store, f)["gate"] in ("PASS", "FAIL")


def test_the_cli(tmp_path, panel, monkeypatch):
    import betfair_recorder
    monkeypatch.setattr(betfair_recorder, "uk_today", lambda: date(2026, 11, 3))
    out = tmp_path / "rep.json"
    assert lt.main(["--store", str(tmp_path), "--freeze", "--summary", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["frozen"]["first_test_day"] == "2026-11-04"
    monkeypatch.setattr(betfair_recorder, "uk_today", lambda: date(2026, 11, 7))
    assert lt.main(["--store", str(tmp_path), "--freeze", "--days", "3", "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["frozen"].startswith("already frozen") and [d["day"] for d in r["days"]] == ["2026-11-04", "2026-11-05",
                                                                                         "2026-11-06"]
