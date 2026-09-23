"""Tests for the closing-line-value evaluation (synthetic prices; exercises code paths only)."""

import numpy as np
import pandas as pd

from model.clv import clv_report, early_bet_rule, prepare_clv_frame, price_move_model, steam_predictability


def _frame(n_races=800, seed=0, fundamentals_informative=True):
    rng = np.random.default_rng(seed); rows = []
    for r in range(n_races):
        n = int(rng.integers(6, 12)); true = rng.dirichlet(np.ones(n) * 0.8)
        bsp = 1 / true * 1.02
        morning = bsp * np.exp(rng.normal(0, 0.25, n))                 # morning price = BSP + noise
        pred = bsp * np.exp(rng.normal(0, 0.45, n)) if fundamentals_informative else morning * np.exp(rng.normal(0, 0.45, n))
        w = rng.choice(n, p=true)
        for i in range(n):
            rows.append({"raceid": f"r{r}", "race_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=r // 5), "bfsp": bsp[i],
                         "morningwap": morning[i], "predicted_bfsp": pred[i], "morning_vol": rng.uniform(10, 500), "won": i == w})
    return pd.DataFrame(rows)


def test_prepare_and_price_move_model_detects_informative_fundamentals():
    d = prepare_clv_frame(_frame())
    assert np.allclose(d.groupby("raceid")["p_early"].sum(), 1.0) and "realised_clv" in d.columns
    d, coefs = price_move_model(d, n_folds=4, extra_cols=["vol_share"])
    assert d["bsp_hat"].notna().sum() > 0 and (coefs["ln_p_pred"] < -0.1).all()  # log price on log prob: negative, and material
    sp = steam_predictability(d)
    assert sp["corr_signal_move"] > 0.2 and sp["direction_hit_rate_strong_signal"] > 0.6
    d2, coefs2 = price_move_model(prepare_clv_frame(_frame(fundamentals_informative=False)), n_folds=4)
    assert coefs2["ln_p_pred"].abs().mean() < coefs["ln_p_pred"].abs().mean()


def test_early_bet_rule_and_report():
    d = prepare_clv_frame(_frame())
    d, _ = price_move_model(d, n_folds=4)
    bets = early_bet_rule(d, min_pred_clv=0.10, commission=0.05, max_early_odds=50)
    assert len(bets) > 0 and (bets["pred_clv"] >= 0.10).all()
    rep = clv_report(d, bets, commission=0.05)
    assert rep["mean_net_clv"] > 0 and rep["clv_ci90"][0] <= rep["mean_net_clv"] <= rep["clv_ci90"][1]
    assert "hold_roi_net" in rep and "bsp_implied_ev_net" in rep and len(rep["by_pred_clv_tier"]) >= 3
    # greening up: realised CLV equals the locked-in profit rate o/b - 1
    assert np.allclose(bets["realised_clv"], np.exp(bets["ln_early"]) / np.exp(bets["ln_bsp"]) - 1)


def test_commission_is_charged_on_winning_trades_only():
    """A drifted price greened up at BSP is a loss, and Betfair takes nothing
    from a loss -- so net CLV must not shrink it."""
    import pandas as pd
    import pytest
    d = pd.DataFrame({"pred_clv": [0.3, 0.3], "ln_early": [np.log(10.0), np.log(10.0)],
                      "realised_clv": [0.20, -0.20]})
    bets = early_bet_rule(d, min_pred_clv=0.10, commission=0.05, max_early_odds=50)
    assert bets["net_clv"].tolist() == pytest.approx([0.19, -0.20])
