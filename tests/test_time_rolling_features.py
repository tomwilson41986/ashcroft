"""Regression tests for time-windowed rolling form features.

Covers two production bugs from the vectorization of the loop-based
implementations:
- _calc_win_density crashed with "need at least one array to concatenate"
  (all group keys became NaN via DataFrame-constructor index alignment).
- _calc_hot_form silently produced all-NaN trainer/jockey form features
  (same alignment bug nulled the value columns).
"""

import numpy as np
import pandas as pd
import pytest

from model.custom_metrics import CustomMetricsEngine, _time_window_prior_stats
from model.financial_features import _calc_win_density


def _brute_force_prior_stats(df, entity_col, window_days):
    """Reference implementation: literal loops over the entity's EARLIER DAYS.

    The reference used to scan `for j in range(i)` and admit `dts[j] <= dts[i]`,
    which counts the rows sitting above this one in the sorted frame on today's
    own date. A yard runs several across a card and two in one race, so that
    reference asserted a feature that reads results from races yet to be run —
    the row-lag leak, written into the test. Today is out in full."""
    n = len(df)
    wins = np.full(n, np.nan)
    runs = np.full(n, np.nan)
    dts = pd.to_datetime(df["race_date"]).values
    won = df["won"].values.astype(float)
    ents = df[entity_col].values
    for i in range(n):
        if pd.isna(ents[i]):
            continue
        cutoff = dts[i] - np.timedelta64(window_days, "D")
        cnt, tot = 0, 0.0
        for j in range(n):
            if pd.isna(ents[j]) or ents[j] != ents[i]:
                continue
            if cutoff <= dts[j] < dts[i]:
                cnt += 1
                tot += won[j]
        if cnt > 0:
            wins[i] = tot
            runs[i] = float(cnt)
    return wins, runs


def _assert_same(actual, expected):
    np.testing.assert_allclose(
        np.nan_to_num(actual, nan=-1), np.nan_to_num(expected, nan=-1)
    )


@pytest.fixture
def form_df():
    rng = np.random.RandomState(7)
    n = 400
    df = pd.DataFrame({
        "trainer": rng.choice(
            ["A", "B", "C", "D", None], n, p=[0.27, 0.27, 0.22, 0.14, 0.1]
        ),
        "race_date": pd.to_datetime("2026-01-01")
        + pd.to_timedelta(rng.randint(0, 120, n), unit="D"),
        "race_time": [f"{h:02d}:00" for h in rng.randint(10, 20, n)],
        "won": (rng.rand(n) < 0.15).astype(float),
    })
    # Duplicate some rows so entities have several runs on the same day
    df = pd.concat([df, df.head(40)], ignore_index=True)
    return df.sort_values(
        ["trainer", "race_date", "race_time"]
    ).reset_index(drop=True)


@pytest.mark.parametrize("window_days", [14, 30, 60])
@pytest.mark.parametrize("closed", ["right", "both"])
def test_time_window_prior_stats_matches_brute_force(form_df, window_days, closed):
    wins, runs = _time_window_prior_stats(form_df, "trainer", window_days, closed)
    exp_wins, exp_runs = _brute_force_prior_stats(form_df, "trainer", window_days)
    _assert_same(wins, exp_wins)
    _assert_same(runs, exp_runs)


def test_time_window_right_edge_is_always_open(form_df):
    """`closed` no longer moves the right edge, whatever a caller passes.

    The callers pass 'right' and 'both' and used to mean by it whether today's
    own runs counted. They never may, so the argument is inert and the test
    says so rather than leaving the next reader to assume it still works."""
    a = _time_window_prior_stats(form_df, "trainer", 30, "right")
    b = _time_window_prior_stats(form_df, "trainer", 30, "both")
    for x, y in zip(a, b):
        _assert_same(x, y)


def test_a_yards_other_runners_today_are_not_in_its_form(form_df):
    """The leak this feature had: same-day runs inside the trailing window."""
    df = pd.DataFrame({
        "trainer": ["A", "A", "A"],
        "race_date": pd.to_datetime(["2026-02-01", "2026-02-20", "2026-02-20"]),
        "race_time": ["14:00", "14:00", "14:30"],
        "won": [0.0, 1.0, 0.0],
    })
    wins, runs = _time_window_prior_stats(df, "trainer", 30, "right")
    # The 14:30 runner must not see the 14:00 winner from the same afternoon.
    assert runs[2] == 1.0 and wins[2] == 0.0
    assert runs[1] == 1.0 and wins[1] == 0.0


def test_time_window_prior_stats_empty_frame(form_df):
    wins, runs = _time_window_prior_stats(form_df.head(0), "trainer", 14, "right")
    assert len(wins) == 0 and len(runs) == 0


def test_time_window_prior_stats_all_nan_entities(form_df):
    df = form_df.copy()
    df["trainer"] = None
    wins, runs = _time_window_prior_stats(df, "trainer", 14, "right")
    assert np.all(np.isnan(wins)) and np.all(np.isnan(runs))


def test_win_density_on_racecard_frame():
    """History rows plus today's racecard (the daily-predictions crash case)."""
    hist = pd.DataFrame({
        "horse_name": ["Alpha"] * 3 + ["Beta"] * 2,
        "race_date": pd.to_datetime(
            ["2026-06-01", "2026-06-15", "2026-06-25", "2026-06-10", "2026-07-01"]
        ),
        "race_time": ["14:00"] * 5,
        "won": [1.0, 0.0, 1.0, 0.0, 1.0],
    })
    today = pd.DataFrame({
        "horse_name": ["Alpha", "Beta", "Newbie"],
        "race_date": pd.to_datetime(["2026-07-13"] * 3),
        "race_time": ["15:00"] * 3,
        "won": [0.0, 0.0, 0.0],
    })
    df = pd.concat([hist, today], ignore_index=True)
    df = df.sort_values(
        ["horse_name", "race_date", "race_time"]
    ).reset_index(drop=True)

    out = _calc_win_density(df)

    def val(horse, date, col):
        row = out[(out["horse_name"] == horse) & (out["race_date"] == date)]
        return row[col].iloc[0]

    # Alpha on 07-13: prior runs within 30 days (>= 06-13): 06-15 (0), 06-25 (1)
    assert val("Alpha", "2026-07-13", "win_density_30d") == pytest.approx(0.5)
    # Alpha 60d window covers all three prior runs: 2 wins / 3 runs
    assert val("Alpha", "2026-07-13", "win_density_60d") == pytest.approx(2 / 3)
    # Beta on 07-13: only 07-01 in the 30d window, a win
    assert val("Beta", "2026-07-13", "win_density_30d") == pytest.approx(1.0)
    # First-ever run and debutants have no prior data
    assert np.isnan(val("Alpha", "2026-06-01", "win_density_30d"))
    assert np.isnan(val("Newbie", "2026-07-13", "win_density_30d"))


def test_hot_form_produces_real_values(form_df):
    """Trainer/jockey hot form must not be all-NaN and must match brute force."""
    df = form_df.dropna(subset=["trainer"]).copy()
    df["jockey_name"] = df["trainer"].map(
        {"A": "J1", "B": "J2", "C": "J1", "D": "J2"}
    )
    engine = CustomMetricsEngine()
    out = engine._calc_hot_form(df.copy())

    for prefix in ["trainer", "jockey"]:
        assert out[f"{prefix}_wins_14d"].notna().any(), (
            f"{prefix} hot form is all-NaN"
        )

    # Compare trainer 14d against brute force on identically sorted data
    check = out.sort_values(
        ["trainer", "race_date", "race_time"]
    ).reset_index(drop=True)
    exp_wins, exp_runs = _brute_force_prior_stats(check, "trainer", 14)
    _assert_same(check["trainer_wins_14d"].values, exp_wins)
    _assert_same(check["trainer_runs_14d"].values, exp_runs)
    expected_sr = exp_wins / exp_runs
    _assert_same(check["trainer_sr_14d"].values, expected_sr)
