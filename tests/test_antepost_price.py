"""The ante-post calibration and price stages (antepost/price.py). The frames are format fixtures from a seeded
generator, written for the test; no figure from them is used anywhere."""

import numpy as np
import pandas as pd

from antepost import price as pr
from antepost import turnup as tu
from tests.test_antepost_turnup import _fixture


def _panel(seed=5):
    rn, mk = _fixture()
    rng = np.random.default_rng(seed)
    # the horses that ran start at their 7-day price moved by a little noise
    rn["bsp"] = np.where(rn["label"] == "ran", rn["ltp_d7"] * np.exp(rng.normal(-0.1, 0.2, len(rn))), np.nan)
    df, _ = tu.build_panel(rn, mk)
    df, _ = tu.add_recent_runs(df, None)
    return df


def test_the_panel_carries_the_sp_as_a_target_not_a_feature():
    df = _panel()
    assert "bsp" in df and "bsp" not in tu.FEATURES
    assert df.loc[df["y"] != "ran", "bsp"].isna().all() and df.loc[df["y"] == "ran", "bsp"].notna().all()


def test_price_rows_are_runners_with_a_price_and_an_sp():
    df = _panel()
    r = pr.price_rows(df)
    assert set(r["y"]) == {"ran"} and (r["has_price"] == 1).all()
    assert np.allclose(r["z"], np.log(r["bsp"]))


def test_the_shift_baseline_learns_the_median_move():
    tr = pd.DataFrame({"mark": ["d30"] * 40, "has_price": 1, "log_ltp": np.log(np.full(40, 4.0)),
                       "z": np.log(np.full(40, 2.0))})
    p = pr.fit_shift(tr)(tr.iloc[:3])
    assert np.allclose(p, np.log(2.0))
    lin = pr.fit_linear(tr.assign(days_out=30))(tr.iloc[:3].assign(days_out=30))
    assert np.allclose(lin, np.log(2.0), atol=1e-6)


def test_concordance_is_one_for_a_perfect_order():
    df = pd.DataFrame({"market_id": ["a"] * 4, "mark": ["d7"] * 4, "z": [1.0, 2.0, 3.0, 4.0]})
    assert pr.concordance(df, np.array([0.1, 0.2, 0.3, 0.4])) == 1.0
    assert pr.concordance(df, np.array([0.4, 0.3, 0.2, 0.1])) == -1.0


def test_both_stages_run_end_to_end_and_read_as_registered():
    df = _panel()
    cal = pr.calibration(df)
    assert {"uncalibrated", "calibrated", "calibrated_less_uncalibrated"} <= set(cal)
    rep, preds = pr.price(df)
    assert set(rep["test"]) == {"model", "p0_price", "p1_shift", "p2_linear"}
    assert len(preds) == rep["rows"]["test"] and rep["residual_sd_by_mark_valid"]
    r = pr.reading(cal, rep)
    assert r["calibration"]["verdict"] in ("PASS", "FAIL") and r["price"]["verdict"] in ("PASS", "FAIL")
    assert pr.reading({"skipped": "x"}, {"skipped": "y"})["price"]["verdict"] == "NOT RUN"
