"""Forward closing-line value: the prices captured at prediction time.

`race_results.odds` is the returned SP, so the database holds two closing
prices and no early one -- there is nothing historic to measure CLV against.
It has to accrue forward from the morning job. These tests cover the three
places that can silently drop a price: the racecard parse, the exchange
snapshot's key, and the join back to the settled result."""

import sqlite3
import sys
import types

import pandas as pd
import pytest

from betfair_prices import race_time_to_24h, utc_iso_to_uk_hhmm
from daily_predictions import parse_odds_text


# --- the racecard price ----------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("4/1", 5.0), ("  7/2  ", 4.5), ("11/8", 2.375),
    ("4/1F", 5.0), ("9/2JF", 5.5), ("5/1CF", 6.0),      # favourite markers
    ("Evs", 2.0), ("EVS", 2.0), ("Evens", 2.0), ("evsF", 2.0),
    ("5.5", 5.5),                                        # already decimal
    ("SP", None), ("-", None), ("", None), (None, None),
    ("1/0", None), ("0.5", None), ("no", None),
])
def test_parse_odds_text(text, expected):
    assert parse_odds_text(text) == expected


# --- the exchange's clock --------------------------------------------------

def test_utc_iso_to_uk_hhmm_handles_bst():
    """British Summer Time is the whole point: in July the exchange says 13:30
    for the race the card calls 2.30, and a naive read drops every summer
    meeting from the join."""
    assert utc_iso_to_uk_hhmm("2026-07-04T13:30:00.000Z") == "14:30"
    assert utc_iso_to_uk_hhmm("2026-01-04T13:30:00.000Z") == "13:30"
    assert utc_iso_to_uk_hhmm("2026-07-04T13:30:00+00:00") == "14:30"
    assert utc_iso_to_uk_hhmm("") is None and utc_iso_to_uk_hhmm("2.30") is None


def test_race_time_to_24h_takes_either_form():
    assert race_time_to_24h("2.30") == "14:30"           # racecard, pm implied
    assert race_time_to_24h("11.15") == "11:15"
    assert race_time_to_24h("2026-07-04T13:30:00.000Z") == "14:30"


# --- the snapshot ----------------------------------------------------------

def _pred(**kw):
    from ultra_betting.data.schemas import Prediction
    base = dict(date="2026-07-04", venue="Ascot", race_time="2.30",
                runner_name="Horse One", predicted_bfsp=4.0, predicted_win_prob=0.25)
    base.update(kw)
    return Prediction(**base)


def _stub_betfair(monkeypatch, runners):
    """Stand in for the two modules attach_exchange_prices imports."""
    auth = types.ModuleType("ultra_betting.betfair.auth")
    auth.ensure_session = lambda: None
    client_mod = types.ModuleType("ultra_betting.betfair.client")

    class _Err(Exception):
        pass

    class _Client:
        def get_live_odds_for_date(self, d):
            return runners

    client_mod.get_client = lambda: _Client()
    client_mod.BetfairAPIError = _Err
    monkeypatch.setitem(sys.modules, "ultra_betting.betfair.auth", auth)
    monkeypatch.setitem(sys.modules, "ultra_betting.betfair.client", client_mod)


def test_attach_exchange_prices_keys_on_market_and_selection(monkeypatch):
    from pipeline.predict import attach_exchange_prices
    preds = [_pred(market_id="1.1", selection_id=11),
             _pred(runner_name="Horse Two", market_id="1.1", selection_id=22)]
    _stub_betfair(monkeypatch, [
        {"market_id": "1.1", "selection_id": 11, "best_back_price": 4.2, "best_lay_price": 4.4,
         "sp_near_price": 4.1, "sp_far_price": 4.3, "last_traded_price": 4.2,
         "runner_matched": 900.0, "total_matched": 50000.0},
        {"market_id": "1.1", "selection_id": 99, "best_back_price": 9.9},   # not ours
    ])
    out = attach_exchange_prices(preds, "2026-07-04")
    assert out[0].bf_best_back == 4.2 and out[0].bf_sp_near == 4.1
    # the RUNNER's volume, not the market's 50,000
    assert out[0].bf_total_matched == 900.0 and out[0].price_snapshot_at is not None
    assert out[1].bf_best_back is None      # no row for selection 22, left empty


def test_attach_exchange_prices_treats_zero_as_no_price(monkeypatch):
    """Betfair reports 0 for "nothing on offer", and 0 read as a price makes
    a 100% CLV out of a market nobody was in."""
    from pipeline.predict import attach_exchange_prices
    _stub_betfair(monkeypatch, [{"market_id": "1.1", "selection_id": 11,
                                 "best_back_price": 0.0, "sp_near_price": None,
                                 "runner_matched": 0.0, "total_matched": 0.0}])
    out = attach_exchange_prices([_pred(market_id="1.1", selection_id=11)], "2026-07-04")
    assert out[0].bf_best_back is None and out[0].bf_sp_near is None
    assert out[0].bf_total_matched == 0.0    # volume of zero is a fact, not a missing price


def test_attach_exchange_prices_never_fails_the_job(monkeypatch):
    from pipeline.predict import attach_exchange_prices
    auth = types.ModuleType("ultra_betting.betfair.auth")

    def _boom():
        raise RuntimeError("no session")

    auth.ensure_session = _boom
    client_mod = types.ModuleType("ultra_betting.betfair.client")
    client_mod.get_client = lambda: None
    client_mod.BetfairAPIError = RuntimeError
    monkeypatch.setitem(sys.modules, "ultra_betting.betfair.auth", auth)
    monkeypatch.setitem(sys.modules, "ultra_betting.betfair.client", client_mod)
    preds = [_pred(market_id="1.1", selection_id=11)]
    assert attach_exchange_prices(preds, "2026-07-04")[0].bf_best_back is None


# --- the join back to the result ------------------------------------------

def test_live_loader_joins_on_normalised_keys(tmp_path):
    """The card and the database disagree about spelling and about time
    format; the join has to survive both, and must not invent rows."""
    from research_lab import _live_clv_frame

    live = tmp_path / "predictions"
    live.mkdir()
    pd.DataFrame([
        {"date": "2026-07-04", "venue": "Newmarket (July)", "race_time": "2.30",
         "runner_name": "1. Casts Tasha (IRE)", "predicted_bfsp": 4.0,
         "predicted_win_prob": 0.25, "racecard_odds": 5.0, "bf_total_matched": 900.0},
        {"date": "2026-07-04", "venue": "Newmarket (July)", "race_time": "2.30",
         "runner_name": "Never Ran", "predicted_bfsp": 9.0,
         "predicted_win_prob": 0.1, "racecard_odds": 11.0, "bf_total_matched": 10.0},
    ]).to_csv(live / "2026-07-04.csv", index=False)

    db = tmp_path / "h.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE race_results (race_date TEXT, track TEXT, race_time TEXT, "
                 "horse_name TEXT, bfsp REAL, placing_numerical INTEGER)")
    conn.executemany("INSERT INTO race_results VALUES (?,?,?,?,?,?)", [
        ("2026-07-04", "Newmarket", "2.30", "Casts Tasha", 4.6, 1),
        ("2026-07-04", "Newmarket", "3.05", "Other Horse", 3.0, 1),
    ])
    conn.commit()
    conn.close()

    d = _live_clv_frame(str(live), str(db))
    assert len(d) == 1                       # the unsettled runner is not invented
    row = d.iloc[0]
    assert row["bfsp"] == 4.6 and bool(row["won"]) is True
    assert row["racecard_odds"] == 5.0 and row["morning_vol"] == 900.0


def test_live_loader_is_empty_without_files(tmp_path):
    from research_lab import _live_clv_frame
    assert _live_clv_frame(str(tmp_path), "nonexistent.db").empty


def test_market_volume_is_not_stored_as_the_runner_s(monkeypatch):
    """`total_matched` is the market's and identical across the field.

    Storing it per runner makes everyone's share of the race's money exactly
    1/n, which is what `prepare_clv_frame`'s `vol_share` would then compute --
    a column that looks like liquidity data and carries none."""
    from pipeline.predict import attach_exchange_prices
    _stub_betfair(monkeypatch, [
        {"market_id": "1.1", "selection_id": 11, "total_matched": 50000.0, "runner_matched": 900.0},
        {"market_id": "1.1", "selection_id": 22, "total_matched": 50000.0, "runner_matched": 120.0},
    ])
    out = attach_exchange_prices(
        [_pred(market_id="1.1", selection_id=11),
         _pred(runner_name="Horse Two", market_id="1.1", selection_id=22)], "2026-07-04")
    vols = [p.bf_total_matched for p in out]
    assert vols == [900.0, 120.0], "the market total was stored instead of the runner's"
    assert len(set(vols)) == 2, "every runner carrying the same volume is the bug"


def test_the_predictions_are_written_before_the_price_snapshot():
    """Order matters: the price call is live and the client has no timeout.

    A hang while reaching for prices must cost the prices, not the day's
    predictions, so the CSV is written first and rewritten afterwards."""
    import inspect

    from pipeline import predict as mod

    src = inspect.getsource(mod.main)
    first_write = src.index('write_csv("predictions"')
    snapshot = src.index("attach_exchange_prices(")
    assert first_write < snapshot, (
        "attach_exchange_prices runs before the predictions are written; a hung "
        "Betfair socket would cost the whole card"
    )


# --- the morning card, kept as fetched ----------------------------------------

def test_the_morning_card_is_kept_as_fetched_and_a_failed_write_costs_nothing(monkeypatch):
    """The result rows later overwrite the going, the jockeys and the field, so
    the card as fetched is the only record of what was known at 06:00."""
    from datetime import date

    import predict_bfsp_today as pbt
    from ultra_betting.model import predict as mod

    card = pd.DataFrame({"race_date": ["2026-09-24"], "track": ["Kempton"], "race_time": ["6:30"],
                         "horse_name": ["Lady Luck"], "going_description": ["Standard"],
                         "jockey_name": ["A Rider(3)"]})
    history = pd.DataFrame({"race_date": pd.to_datetime(["2026-09-01"] * 200)})

    def predict_in_place(hist, runners, *a, **k):
        runners["jockey_name"] = "A Rider"            # the fill strips the claim from the frame it is given
        runners["predicted_bfsp"] = 3.0
        runners["predicted_win_prob_norm"] = 1 / 3
        return runners

    monkeypatch.setenv("HRB_USERNAME", "someone")
    monkeypatch.setattr(pbt, "load_bfsp_model", lambda model_dir: (None, [], {}))
    monkeypatch.setattr(pbt, "load_historical", lambda *a, **k: history)
    monkeypatch.setattr(pbt, "fetch_racecard_from_hrb", lambda d: card.copy())
    monkeypatch.setattr(pbt, "get_runners_from_db", lambda *a: card.copy())
    monkeypatch.setattr(pbt, "prepare_and_predict", predict_in_place)
    day = date(2026, 9, 24)

    kept = []
    assert len(mod.run_predictions(target_date=day, card_sink=kept.append)) == 1
    assert len(kept) == 1 and kept[0]["jockey_name"].iloc[0] == "A Rider(3)"      # as fetched

    def s3_down(frame):
        raise OSError("S3 unreachable")
    assert len(mod.run_predictions(target_date=day, card_sink=s3_down)) == 1      # costs the record only

    kept.clear()
    mod.run_predictions(target_date=day, from_db=True, card_sink=kept.append)
    assert kept == []                                                              # database rows are no card


def test_the_morning_card_is_written_under_its_fetch_time():
    """A re-run later in the day adds a file; it never replaces the morning's."""
    import inspect

    from pipeline import predict as mod

    src = inspect.getsource(mod.main)
    assert 'write_csv("racecards", f"{target_date}_{fetched_at}"' in src
    assert src.index("card_sink=") < src.index('write_csv("predictions"')


# --- the early-price diagnostics -------------------------------------------------

def test_clv_diagnostics_clock_and_commission():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "clv_diagnostics", Path(__file__).resolve().parent.parent / "scripts" / "clv_diagnostics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.minutes("1.50") == 13 * 60 + 50          # card times 1-9 are afternoon
    assert mod.minutes("11:05") == 11 * 60 + 5
    assert mod.minutes("1.50.") == 13 * 60 + 50         # the table's trailing dot
    assert pd.isna(mod.minutes("tba"))
    net = mod.net_clv(pd.Series([4.4, 3.0]), pd.Series([4.0, 4.0]))
    assert net[0] == pytest.approx(0.1 * 0.95)          # commission only on a gain
    assert net[1] == pytest.approx(-0.25)


def test_serving_builds_history_from_where_training_did(tmp_path):
    """QA review M6: serving from 2020 while training from 2021 gave live rows a
    year of history no training row had."""
    import json

    from ultra_betting.model.predict import TRAINING_START_FALLBACK, training_start

    (tmp_path / "bfsp_training_summary.json").write_text(json.dumps({"data_range": {"min_date": "2022-03-01"}}))
    assert training_start(tmp_path) == "2022-03-01"
    assert training_start(tmp_path / "absent") == TRAINING_START_FALLBACK
    # the committed model says where its own history began
    import os
    committed = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "models")
    assert training_start(committed) == "2021-01-01"
