"""The football strategy scan's pieces: the worse price, a cell's reading, and a price jump read from minute rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.open_close import football_strategy_scan as sc


def test_one_tick_worse_follows_betfairs_ladder():
    assert sc.tick_worse([1.05, 2.0, 2.5, 3.5, 5.0, 8.0, 15.0, 25.0, 40.0, 80.0, 200.0, 1.01]).tolist() == \
        pytest.approx([1.04, 1.99, 2.48, 3.45, 4.9, 7.8, 14.5, 24.0, 38.0, 75.0, 190.0, 1.01])


def test_a_lead_needs_both_halves_above_zero_and_the_even_months_interval_too():
    rng = np.random.default_rng(0)

    def frame(win_rate_odd, win_rate_even, n=3000):
        day = pd.to_datetime("2026-01-01") + pd.to_timedelta(rng.integers(0, 300, n), "D")
        odd = day.month % 2 == 1
        won = rng.uniform(size=n) < np.where(odd, win_rate_odd, win_rate_even)
        return pd.DataFrame({"day": day.strftime("%Y-%m-%d"), "price": 2.0, "won": won})

    good = frame(0.56, 0.56)                                  # a back at 2.0 that wins 56 per cent of the time
    c = sc.cell(good, sc.tick_worse(good["price"]))
    assert c["lead"] and c["even_months"]["at_the_worse_price"]["lo90"] > 0
    assert c["all"]["at_the_price"]["return"] > c["all"]["at_the_worse_price"]["return"]
    one_half = frame(0.56, 0.48)                              # found in the odd months, gone in the even ones
    assert not sc.cell(one_half, sc.tick_worse(one_half["price"]))["lead"]
    assert not sc.cell(good.head(20), sc.tick_worse(good["price"].head(20)))["lead"]      # too few to read


def test_a_jump_is_read_from_two_fresh_rows_and_the_bet_waits_two_minutes():
    m = pd.DataFrame({"event_id": 1, "minute": [10, 11, 12, 13, 14, 15, 30, 40, 41],
                      "raw_home": [0.40, 0.40, 0.62, 0.63, 0.64, 0.65, 0.66, 0.30, 0.31],
                      "raw_away": [0.30, 0.30, 0.15, 0.14, 0.13, 0.12, 0.11, 0.45, 0.46]})
    j = sc.jumps(m)
    # the home side jumps at minute 12 (0.40 to 0.62 between rows a minute apart); the bet is the first row from 14
    assert j[["minute", "side", "bet_minute"]].to_numpy().tolist() == [[12, "home", 14]]
    assert (j["raw_gainer"].iloc[0], j["raw_other"].iloc[0]) == (0.64, 0.13)
    # 30 to 40 is a jump for the away side but the rows are ten minutes apart: not read as one
    assert sc.jumps(m[m["minute"] >= 30]).empty
