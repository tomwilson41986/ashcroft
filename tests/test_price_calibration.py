"""Tests for calibrating the BFSP forecast against realised BFSP."""

import numpy as np
import pandas as pd

from model.price_calibration import (BSPPriceCalibrator, build_design, calibration_report,
                                     quantile_fit, walk_forward_calibrate)


def _frame(n_races=600, seed=0, compress=0.8, level=1.15):
    """Races whose forecast prices are deliberately compressed and too long.

    `compress` < 1 squeezes the forecast spread inside the race, which is what a
    squared-error fit on log price does; `level` lengthens every price. Both are
    exactly the defects the calibrator is supposed to undo."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        n = int(rng.integers(6, 15))
        s = rng.normal(0, 1.0, n)
        p_true = np.exp(s) / np.exp(s).sum()
        bsp = 1.0 / p_true
        lb = np.log(bsp)
        pred = np.exp(lb.mean() + compress * (lb - lb.mean()) + np.log(level)
                      + rng.normal(0, 0.15, n))
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=r // 3)
        for i in range(n):
            rows.append({"raceid": f"r{r}", "race_date": day.strftime("%Y-%m-%d"),
                         "horse_name": f"h{i}", "predicted_bfsp": float(pred[i]),
                         "bfsp": float(bsp[i])})
    return pd.DataFrame(rows)


def test_design_is_known_before_the_off():
    d = _frame(30)
    X, aug = build_design(d)
    assert X.shape == (len(d), 7) and np.isfinite(X).all()
    # nothing in the design touches the realised price
    assert aug["_dev"].abs().max() > 0
    assert np.allclose(aug.groupby("raceid")["_dev"].sum(), 0, atol=1e-9)
    assert aug["_u"].min() == 0 and abs(aug["_u"].max() - 1.0) < 1e-9
    d2 = d.copy(); d2["bfsp"] = 999.0
    assert np.allclose(build_design(d2)[0], X)          # BSP cannot influence it


def test_quantile_fit_recovers_a_known_quantile():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 6000)
    y = 2.0 + 0.5 * x + rng.normal(0, 1, 6000)
    X = np.column_stack([np.ones_like(x), x])
    for q, z in ((0.5, 0.0), (0.25, -0.6745), (0.75, 0.6745)):
        b = quantile_fit(X, y, q)
        assert abs(b[0] - (2.0 + z)) < 0.12 and abs(b[1] - 0.5) < 0.08


def test_calibration_removes_compression_and_level_bias():
    d = _frame(700, seed=1)
    cal = BSPPriceCalibrator(quantiles=(0.5,)).fit(d)
    out = cal.transform(d)
    raw_ratio = (d["predicted_bfsp"] / d["bfsp"]).median()
    cal_ratio = (out["bsp_forecast"] / out["bfsp"]).median()
    assert raw_ratio > 1.10 and abs(cal_ratio - 1.0) < 0.02
    # the top pick is where compression bites hardest
    rank = d.groupby("raceid")["predicted_bfsp"].rank(method="first")
    top = rank == 1
    assert (d.loc[top, "predicted_bfsp"] / d.loc[top, "bfsp"]).median() > 1.15
    assert abs((out.loc[top.values, "bsp_forecast"] / out.loc[top.values, "bfsp"]).median() - 1.0) < 0.05
    # and accuracy improves, it is not just a level shift
    assert (np.abs(np.log(out["bsp_forecast"] / out["bfsp"])).mean()
            < np.abs(np.log(d["predicted_bfsp"] / d["bfsp"])).mean())


def test_quantiles_are_ordered_and_cover_nominally():
    d = _frame(700, seed=2)
    out = BSPPriceCalibrator().fit(d).transform(d)
    assert (out["bsp_forecast_q25"] <= out["bsp_forecast"] + 1e-6).all()
    assert (out["bsp_forecast"] <= out["bsp_forecast_q75"] + 1e-6).all()
    for col, nominal in (("bsp_forecast_q25", 25), ("bsp_forecast", 50), ("bsp_forecast_q75", 75)):
        assert abs(100 * (out["bfsp"] <= out[col]).mean() - nominal) < 6


def test_walk_forward_never_fits_on_the_block_it_scores():
    d = _frame(900, seed=3)
    wf = walk_forward_calibrate(d, n_folds=5)
    scored = wf[wf["bsp_forecast"].notna()]
    assert 0 < len(scored) < len(wf)                 # the first block is training only
    assert scored["race_date"].min() > wf["race_date"].min()
    rep = calibration_report(scored)
    s = rep["summary"].set_index("forecast")
    assert s.loc["cal", "median_ratio"] == rep["summary"].set_index("forecast").loc["cal", "median_ratio"]
    assert abs(s.loc["cal", "median_ratio"] - 1.0) < abs(s.loc["raw", "median_ratio"] - 1.0)
    assert s.loc["cal", "mean_abs_log_err"] < s.loc["raw", "mean_abs_log_err"]
    by = rep["by_rank"]
    assert (by["ratio_cal"] - 1.0).abs().max() < (by["ratio_raw"] - 1.0).abs().max()


def test_round_trip_through_json(tmp_path):
    d = _frame(200, seed=4)
    cal = BSPPriceCalibrator().fit(d)
    p = tmp_path / "cal.json"
    cal.save(str(p))
    back = BSPPriceCalibrator.load(str(p))
    assert set(back.coef_) == set(cal.coef_)
    assert np.allclose(back.predict(d, 0.5), cal.predict(d, 0.5))
    assert back.meta_["rows"] == len(d)
