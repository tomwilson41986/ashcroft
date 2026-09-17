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
