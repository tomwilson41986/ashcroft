"""The pre-registered early-lay test's arithmetic (au/lay_test.py). The rows are format fixtures written for the test;
no figure from them is used anywhere."""

import pandas as pd
import pytest

from au.lay_test import bets, score


def _j():
    return pd.DataFrame({
        "market_id": [1, 1, 2, 2], "selection_id": [11, 12, 21, 22], "year": [2024, 2024, 2025, 2025],
        "market_time": ["2024-05-01T04:00:00Z"] * 2 + ["2025-05-01T04:00:00Z"] * 2,
        "won": [1, 0, 0, 1], "bsp": [4.0, 6.0, 5.0, 2.0],
        "ltp_t_60": [3.6, 5.0, 4.0, 30.0], "p_model": [0.2, 0.25, 0.15, 0.4]})


def test_the_rule_picks_prices_under_the_models_and_closes_level():
    s = bets(_j(), "ltp_t_60", True, 1.5, 21.0)
    # 3.6*0.2=0.72 < 1, 5.0*0.25=1.25 not, 4.0*0.15=0.6 < 1, 30 over 21
    assert list(s["selection_id"]) == [11, 21]
    assert s["green"].tolist() == pytest.approx([1 - 3.6 / 4.0, 1 - 4.0 / 5.0])
    assert s["result"].tolist() == [-2.6, 1.0]


def test_commission_is_taken_from_each_markets_net_profit():
    s = bets(_j(), "ltp_t_60", True, 1.5, 21.0)
    r = score(s, "green", boot=50)
    # market 1: +0.1 -> 0.098; market 2: +0.2 -> 0.196; 0.294 over 2 units
    assert r["per_unit_pct"] == pytest.approx(14.7, abs=1e-3)
    assert r["by_year_pct"] == {2024: pytest.approx(9.8, abs=1e-3), 2025: pytest.approx(19.6, abs=1e-3)}


def test_only_the_yearly_tables_are_read():
    import re
    keys = ["betfair_historic/horse_racing_win_2024.parquet", "betfair_historic/horse_racing_win_ages_au.parquet",
            "betfair_historic/ages/au_t60_ages.parquet"]
    assert [k for k in keys if re.fullmatch(r"betfair_historic/horse_racing_win_\d{4}\.parquet", k)] == keys[:1]
