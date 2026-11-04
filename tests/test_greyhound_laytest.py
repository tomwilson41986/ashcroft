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


def test_the_day_fit_is_made_once_and_kept(tmp_path, panel, monkeypatch):
    store = Store(root=str(tmp_path))
    f = lt.day_fit(store, date(2026, 11, 8))
    assert f is not None and f["fit_to"] == "2026-11-07" and len(f["beta"]) == len(f["cols"])
    monkeypatch.setattr(ls, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("refitted")))
    g = lt.day_fit(store, date(2026, 11, 8))                 # from S3, not refitted
    assert list(g["beta"]) == list(f["beta"])


def test_intraday_scores_today_race_by_race(tmp_path, panel, monkeypatch):
    store = Store(root=str(tmp_path))
    rows = panel[(panel["mark"] == lt.MARK) & (panel.race_date == "2026-11-08")].assign(
        race_time="12:00", track="Romford")
    monkeypatch.setattr(lt, "today_rows", lambda store, day: (rows, "test books"))
    r = lt.intraday(store, date(2026, 11, 8))
    assert r["races"] > 0 and r["source"] == "test books" and r["lays_placed"] >= r["matched"]
    assert abs(r["markets"][-1]["running"] - r["pnl"]) < 0.02
    md = lt.intraday_markdown(r)
    assert "### Early lay" in md and "| 12:00 | Romford |" in md


def test_intraday_says_why_when_nothing_is_settled(tmp_path, panel, monkeypatch):
    import pandas as pd
    monkeypatch.setattr(lt, "today_rows", lambda store, day: (pd.DataFrame(), "no settled race yet"))
    r = lt.intraday(Store(root=str(tmp_path)), date(2026, 11, 8))
    assert r["markets"] == [] and r["note"] == "no settled race yet"
    assert "no settled race yet" in lt.intraday_markdown(r)


def test_today_rows_fall_back_to_the_live_prices_with_the_winner_from_the_result(tmp_path, monkeypatch):
    import io
    import pandas as pd
    import betfair_recorder
    from greyhound import tradeout as to
    store = Store(root=str(tmp_path))
    preds = pd.DataFrame({"race_id": ["r1", "r1", "r2"], "race_date": ["2026-11-03"] * 3,
                          "race_time": ["12:00", "12:00", "12:10"], "track": ["Hove"] * 3, "trap": [1, 2, 1],
                          "field": [2, 2, 1], "p_model": [0.6, 0.4, 1.0], "market_id": ["1.1", "1.1", "1.2"],
                          "selection_id": [11, 12, 21]})
    buf = io.StringIO()
    preds.to_csv(buf, index=False)
    store.put("greyhound/live/predictions/2026-11-03.csv", buf.getvalue().encode())
    final = pd.DataFrame({"source": ["final", "final"], "market_id": ["1.1", "1.1"], "selection_id": [11, 12],
                          "runner_status": ["LOSER", "WINNER"]})
    monkeypatch.setattr(betfair_recorder, "_read_day_csv",
                        lambda day, name, s3=None: final if name == to.ENTRY_BOOKS["live"] else None)
    seen = {}

    def panel(pred, entry, markets, day, closing_books=None, retimed=False):
        seen["pred"] = pred
        return pd.DataFrame()
    monkeypatch.setattr(to, "day_panel", panel)
    monkeypatch.setattr(betfair_recorder, "_read_day_csv",
                        lambda day, name, s3=None: final if name == to.ENTRY_BOOKS["live"] else (
                            pd.DataFrame({"market_id": ["1.1"]}) if name.startswith("markets") else None))
    rows, note = lt.today_rows(store, date(2026, 11, 3))
    p = seen["pred"]
    assert set(p.market_id) == {"1.1"}                          # only the settled race
    assert p.set_index("selection_id").won.to_dict() == {11: 0, 12: 1}


def test_the_rule_revised_before_the_first_test_day_is_frozen_again(tmp_path, panel, monkeypatch):
    import betfair_recorder
    store = Store(root=str(tmp_path))
    old = lt.freeze(store, date(2026, 11, 3))
    old["primary"] = dict(lt.PRIMARY, exit="sp")                  # the first version, as frozen at 11:41
    store.put(lt.FROZEN_KEY, json.dumps(old).encode())
    monkeypatch.setattr(betfair_recorder, "uk_today", lambda: date(2026, 11, 3))
    out = tmp_path / "rep.json"
    assert lt.main(["--store", str(tmp_path), "--freeze", "--out", str(out)]) == 0
    f = json.loads(store.get(lt.FROZEN_KEY))
    assert f["primary"]["exit"] == "last" and f["revised_from"]["primary"]["exit"] == "sp"
    assert f["first_test_day"] == "2026-11-04"
    # from the first test day on, the frozen rule stays as it is
    f["primary"] = dict(lt.PRIMARY, exit="sp")
    store.put(lt.FROZEN_KEY, json.dumps(f).encode())
    monkeypatch.setattr(betfair_recorder, "uk_today", lambda: date(2026, 11, 4))
    assert lt.main(["--store", str(tmp_path), "--freeze", "--out", str(out)]) == 0
    assert json.loads(store.get(lt.FROZEN_KEY))["primary"]["exit"] == "sp"


def test_the_primary_greens_up(tmp_path, panel, monkeypatch):
    store = Store(root=str(tmp_path))
    rows = panel[(panel["mark"] == lt.MARK) & (panel.race_date == "2026-11-08")].assign(
        race_time="12:00", track="Romford")
    monkeypatch.setattr(lt, "today_rows", lambda store, day: (rows, "test books"))
    r = lt.intraday(store, date(2026, 11, 8))
    assert lt.PRIMARY["exit"] == "last"
    assert any(m["greened_at"] is not None for m in r["markets"] if m["matched"])
    assert "Greened at" in lt.intraday_markdown(r)


def test_the_forecast_fit_matches_the_live_model(tmp_path, monkeypatch):
    """The early lay's forecast is fitted on the prices of the model that made today's predictions: the log-BSP
    model's out-of-sample prices when it is live, else the 5 Oct model's; each fit kept under its own key."""
    import pandas as pd
    from datetime import date
    from greyhound import laytest as lt
    from sources.common import Store
    store = Store(root=tmp_path)
    day = date(2026, 11, 4)
    assert lt.live_prices(store, day) is None                                   # no predictions yet
    store.put(f"greyhound/live/predictions/{day}.csv", pd.DataFrame({"market_id": ["1.1"], "p_model": [0.2]})
              .to_csv(index=False).encode())
    assert lt.live_prices(store, day) is None                                   # an older file: the 5 Oct model
    store.put(f"greyhound/live/predictions/{day}.csv", pd.DataFrame({"market_id": ["1.1"], "p_model": [0.2],
                                                                    "model_objective": ["logbsp"]})
              .to_csv(index=False).encode())
    assert lt.live_prices(store, day) == lt.LOGBSP_PRICES
    seen = {}

    def fake_load(store_, first, last, source="retimed", pred_dir=None, books_out=None):
        seen["pred_dir"] = pred_dir
        return pd.DataFrame()
    from research.greyhound import limit_study as ls
    monkeypatch.setattr(ls, "load", fake_load)
    assert lt.live_fit(store, day) is None and seen["pred_dir"] == f"s3:{lt.LOGBSP_PRICES}"


def test_the_panel_reads_prices_from_the_store(tmp_path):
    import pandas as pd
    from datetime import date
    from research.greyhound import limit_study as ls
    from sources.common import Store
    store = Store(root=tmp_path)
    p = pd.DataFrame({"race_id": ["r"], "p_model": [0.5]})
    store.put_parquet = None
    import io
    buf = io.BytesIO()
    p.to_parquet(buf, index=False)
    store.put("greyhound/x/2026-10-06.parquet", buf.getvalue())
    got = Store(root=tmp_path).get_parquet("greyhound/x/2026-10-06.parquet")
    assert got is not None and len(got) == 1
    # with no books for the day the loader notes it and moves on: the S3 branch is taken without error
    assert ls.load(Store(root=tmp_path), date(2026, 10, 6), date(2026, 10, 6), pred_dir="s3:greyhound/x").empty
