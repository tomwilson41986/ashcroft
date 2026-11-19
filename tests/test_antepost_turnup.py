"""The ante-post turn-up model (antepost/turnup.py). Every frame here is a format fixture written for the test, drawn
from a seeded generator where a pattern is needed; no figure from it is used anywhere."""

import numpy as np
import pandas as pd

from antepost import turnup as tu


def _fixture(n_markets=60, horses=12, seed=3):
    """Markets across 2024-26, each horse listed at every mark with a price; shorter-priced horses run more often."""
    rng = np.random.default_rng(seed)
    mk, rn = [], []
    days = pd.date_range("2024-02-01", "2026-09-30", periods=n_markets)
    for m, day in enumerate(days):
        mid = f"1.{m}"
        mk.append({"market_id": mid, "market_time": f"{day:%Y-%m-%d}T14:00:00Z", "country": "GB",
                   "event_name": f"Ascot {day.day}th {day:%b}", "market_name": "Some Handicap"})
        for h in range(horses):
            price = float(np.exp(rng.uniform(np.log(2), np.log(100))))
            ran = rng.random() < 1 / (1 + price / 15)
            row = {"market_id": mid, "selection_id": m * 100 + h, "runner_name": f"Horse {m}-{h}",
                   "name_norm": f"horse{m}{h}", "race_date": f"{day:%Y-%m-%d}",
                   "final_status": "REMOVED" if (not ran and rng.random() < 0.1) else ("LOSER"),
                   "label": "ran" if ran else "no_run", "first_seen": f"{day - pd.Timedelta(days=200):%Y-%m-%d}T00:00:00Z"}
            for k in tu.ORDER:
                row[f"listed_{k}"] = True
                row[f"ltp_{k}"] = price * (1 + tu.MARKS[k] / 100)
                row[f"age_{k}"], row[f"ntr_{k}"], row[f"susp_{k}"] = 1.0, 5, 1
            rn.append(row)
    return pd.DataFrame(rn), pd.DataFrame(mk)


def test_the_outcome_follows_the_rules():
    assert tu.outcome("REMOVED", "ran") == "void" and tu.outcome("REMOVED", "no_run") == "void"
    assert tu.outcome("WINNER", "ran") == "ran" and tu.outcome("LOSER", "ran_elsewhere") == "lost"
    assert tu.outcome("LOSER", "no_run") == "lost"


def test_the_panel_keeps_what_was_known_at_the_mark():
    rn, mk = _fixture(n_markets=4, horses=3)
    rn.loc[0, "listed_d3"] = False                         # removed before 3 days out: no row there
    rn.loc[1, "ltp_d7"] = None                              # no price at 7 days: a row without one
    df, rep = tu.build_panel(rn, mk)
    assert rep["markets_used"] == 4 and rep["duplicates"] == 0
    assert len(df) == 4 * 3 * len(tu.ORDER) - 1
    r = df[(df["selection_id"] == rn.loc[1, "selection_id"]) & (df["mark"] == "d7")].iloc[0]
    assert r["has_price"] == 0 and np.isnan(r["log_ltp"])
    one = df[(df["market_id"] == "1.0") & (df["mark"] == "d30")]
    assert abs(one["share"].sum() - 1) < 1e-9 and set(one["n_listed"]) == {3}
    assert set(df.columns) >= set(tu.FEATURES) - {"days_since_run", "runs_365", "ran_14"}


def test_wholly_removed_markets_and_unfound_races_are_left_out():
    rn, mk = _fixture(n_markets=3, horses=3)
    rn.loc[rn["market_id"] == "1.0", "final_status"] = "REMOVED"
    rn.loc[rn["market_id"] == "1.1", "label"] = "race_not_found"
    _, rep = tu.build_panel(rn, mk)
    assert rep["markets_used"] == 1 and rep["dropped_wholly_removed"] == 1


def test_the_bootstrap_resamples_whole_races():
    diff = np.array([-1.0, -1.0, 1.0, 1.0])
    lo, hi = tu.race_bootstrap(diff, np.array(["a", "a", "b", "b"]), draws=500)
    assert -1 <= lo <= hi <= 1
    lo, hi = tu.race_bootstrap(np.full(6, -0.2), np.array(list("aabbcc")), draws=200)
    assert lo == hi == -0.2


def test_the_evaluation_runs_end_to_end_and_reads_as_registered():
    rn, mk = _fixture()
    df, _ = tu.build_panel(rn, mk)
    df, rr = tu.add_recent_runs(df, None)
    assert rr["skipped"].startswith("no database")
    rep, preds = tu.evaluate(df)
    assert set(rep["test"]) == {"model", "b1_table", "b2_logit"} and rep["rounds"] >= 1
    assert set(rep["paired"]) == {"model_less_b1_table", "model_less_b2_logit"}
    assert len(preds) == rep["split"]["test"]["rows"]
    p = preds[[f"model_{c}" for c in tu.CLASSES]].to_numpy()
    assert np.allclose(p.sum(axis=1), 1.0)
    r = tu.reading(rep)
    assert r["verdict"] in ("PASS", "FAIL") and "marks_no_worse_than_b2" in r


def test_recent_runs_count_only_days_before_the_mark(tmp_path):
    import sqlite3
    db = tmp_path / "h.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE betfair_prices (race_date TEXT, market_type TEXT, horse_norm TEXT)")
    con.executemany("INSERT INTO betfair_prices VALUES (?,?,?)", [
        ("2025-01-01", "win", "alpha"), ("2025-02-20", "win", "alpha"), ("2025-03-01", "win", "alpha"),
        ("2025-02-25", "place", "alpha"), ("2024-01-01", "win", "alpha")])
    con.commit()
    con.close()
    df = pd.DataFrame({"name_norm": ["alpha", "beta"],
                       "t": pd.to_datetime(["2025-03-01T10:00:00Z", "2025-03-01T10:00:00Z"])})
    out, rep = tu.add_recent_runs(df, str(db))
    naive = df.assign(t=pd.to_datetime(df["t"]).dt.tz_convert(None))          # as build_panel keeps it
    assert tu.add_recent_runs(naive, str(db))[0]["days_since_run"].iloc[0] == 9
    a = out.iloc[0]
    assert a["days_since_run"] == 9 and a["runs_365"] == 2 and a["ran_14"] == 1     # 20 Feb and 1 Jan; not 1 Mar
    assert np.isnan(out.iloc[1]["days_since_run"]) and out.iloc[1]["runs_365"] == 0


def test_every_row_gets_a_price_band_key():
    df = pd.DataFrame({"mark": ["d30"] * 5, "has_price": [1, 1, 1, 1, 0],
                       "log_ltp": np.log([1.0, 1.01, 1000.0, 1200.0, np.nan])})
    keys = tu.band_of(df)
    assert list(keys) == ["d30|b0", "d30|b0", "d30|b5", "d30|b5", "d30|no_price"]
    train = df.assign(y=["ran", "lost", "lost", "void", "lost"])
    p = tu.fit_table(train)(df)
    assert p.shape == (5, 3) and np.allclose(p.sum(axis=1), 1.0)
