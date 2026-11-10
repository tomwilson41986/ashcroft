"""Betfair's Expert Fee as research/expert_fee.py models it (the repository's reading of Betfair's terms)."""
import pandas as pd
import pytest

from research import expert_fee as ef


def _weeks(gross, commission=None):
    return pd.DataFrame({"gross": gross, "commission": commission or [0.0] * len(gross)})


def test_the_rate_by_the_52_week_gross_profit():
    assert ef.fee_rate(24_999.99) == 0.0 and ef.fee_rate(25_000.0) == 0.0
    assert ef.fee_rate(25_000.01) == 0.20 and ef.fee_rate(100_000.0) == 0.20
    assert ef.fee_rate(100_000.01) == 0.40


def test_no_fee_under_the_threshold_and_the_repositorys_own_example_over_it():
    """reports/open_close/markets.md 5.1: at 2% commission and a 20% rate a week of GBP1,000 gross pays GBP20
    commission and then GBP180 fee."""
    w = ef.weekly_fees(_weeks([1000.0] * 30, [20.0] * 30))
    assert (w.fee.iloc[:25] == 0).all()                       # 25 weeks: GBP25,000, not over it
    assert w.rate.iloc[25] == 0.20 and w.fee.iloc[25] == pytest.approx(180.0)
    assert w.kept.iloc[25] == pytest.approx(800.0)            # the account keeps 80% of the week's gross


def test_commission_is_credited_and_the_fee_is_never_negative():
    w = ef.weekly_fees(_weeks([30_000.0, 100.0], [0.0, 50.0]))
    assert w.fee.iloc[1] == 0.0                               # 20% of GBP100 is GBP20, under the GBP50 commission
    w = ef.weekly_fees(_weeks([30_000.0, 1000.0], [0.0, 50.0]))
    assert w.fee.iloc[1] == pytest.approx(150.0)


def test_a_losing_week_builds_the_buffer_and_later_profit_uses_it():
    w = ef.weekly_fees(_weeks([30_000.0, -400.0, 300.0, 600.0]))
    assert w.fee.iloc[0] == pytest.approx(6000.0)
    assert w.buffer.iloc[1] == 400.0 and w.fee.iloc[1] == 0.0
    assert w.buffer.iloc[2] == 100.0 and w.fee.iloc[2] == 0.0  # GBP300 of profit, all inside the buffer
    assert w.buffer.iloc[3] == 0.0 and w.fee.iloc[3] == pytest.approx(0.20 * 500.0)


def test_not_in_lifetime_profit_no_fee():
    w = ef.weekly_fees(_weeks([30_000.0]), lifetime_before=-40_000.0)
    assert w.rate.iloc[0] == 0.0 and w.fee.iloc[0] == 0.0


def test_the_window_is_52_active_weeks():
    w = ef.weekly_fees(_weeks([30_000.0] + [0.0] * 52 + [1000.0]))
    assert w.gross_52w.iloc[51] == 30_000.0 and w.gross_52w.iloc[52] == 0.0
    assert w.fee.iloc[-1] == 0.0                              # the big week has left the window


def test_the_excess_reading_charges_only_the_profit_over_each_threshold():
    w = ef.weekly_fees(_weeks([24_000.0, 2000.0, 80_000.0]), band="excess")
    assert w.fee.iloc[0] == 0.0
    assert w.fee.iloc[1] == pytest.approx(0.20 * 1000.0)      # GBP1,000 of the week lies over GBP25,000
    # 26,000 -> 106,000: GBP74,000 in the 20% band and GBP6,000 in the 40% band
    assert w.fee.iloc[2] == pytest.approx(0.20 * 74_000.0 + 0.40 * 6000.0)
    whole = ef.weekly_fees(_weeks([24_000.0, 2000.0, 80_000.0]))
    assert whole.fee.iloc[1] == pytest.approx(0.20 * 2000.0) and whole.fee.iloc[2] == pytest.approx(0.40 * 80_000.0)
    with pytest.raises(ValueError):
        ef.weekly_fees(_weeks([1.0]), band="other")


def test_the_projection_carries_the_mean_week_forward():
    p = ef.project(_weeks([1000.0] * 5, [20.0] * 5), n_weeks=30)
    assert p["weeks_on_record"] == 5 and p["mean_week_gross"] == 1000.0
    assert p["first_fee_week"] == 21                           # week 26 of the account is the first over GBP25,000
    assert p["projected_fee"] == pytest.approx(10 * 180.0)
    assert p["projected_kept"] == pytest.approx(30 * 980.0 - 1800.0)
    assert ef.project(_weeks([]))["first_fee_week"] is None
    assert ef.project(_weeks([100.0] * 4), n_weeks=10)["first_fee_week"] is None
