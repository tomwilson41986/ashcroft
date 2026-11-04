"""Staking mechanics: capital-limited Kelly, the implied draw and under forecasts, the bank path."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.open_close import football_staking as fs


def test_kelly_shares_respect_the_budget_and_cap_and_prefer_edge_per_risk():
    mu, s2 = np.array([0.02, 0.02, 0.005]), np.array([0.01, 0.04, 0.01])
    f = fs.kelly_shares(mu, s2, budget=0.5, cap=0.3)
    assert f.sum() <= 0.5 + 1e-9 and (f <= 0.3 + 1e-12).all()
    assert f[0] > f[1] > 0 and f[2] < f[1]                    # same edge, less risk: more capital


def test_the_implied_draw_and_under_moves_balance_the_book():
    qh, qd, qa = np.array([0.5]), np.array([0.25]), np.array([0.25])
    d = fs._implied(qd, qh, qa, np.log(0.55 / 0.5), np.array([0.0]))           # home up 5 points: draw down
    assert abs(np.exp(d[0]) * 0.25 - 0.20) < 1e-12
    u = fs._under(np.array([0.5]), np.log(0.6 / 0.5))
    assert abs(np.exp(u[0]) * 0.5 - 0.4) < 1e-12


def test_the_bank_compounds_window_by_window_and_charges_commission_on_wins():
    sel = pd.DataFrame({"window": ["w1", "w2"], "profit": [0.10, -0.05], "cap_unit": [1.0, 1.0],
                        "mu": [0.02, 0.02], "s2": [0.01, 0.01], "ev": [0.02, 0.02]})
    r = fs.simulate(sel, "level", budget=0.5, cap=0.5)
    b1 = 1 + 0.5 * 0.10 * 0.98
    assert abs(r["bank_end"] - round(b1 - b1 * 0.5 * 0.05, 4)) < 1e-4 and r["windows"] == 2
