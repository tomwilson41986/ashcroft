"""The Betfair-native closing-price model: its memory features read earlier matches only; its scores are what
they say."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.open_close import football_bf_close_model as cm


def test_prior_mean_reads_only_rows_strictly_before_the_moment_and_only_the_window():
    group = np.array([1, 1, 1, 1, 2])
    when = np.array([10, 20, 30, 40, 15])
    value = np.array([1.0, 2.0, np.nan, 4.0, 10.0])
    q_group = np.array([1, 1, 1, 1, 2, 3])
    q_when = np.array([10, 21, 41, 100, 15, 50])
    mean, n = cm.prior_mean(group, when, value, q_group, q_when, window=2, k=0.0)
    assert n.tolist() == [0, 2, 2, 2, 0, 0]                  # at 10: nothing strictly before; group 3: unseen
    assert np.isnan(mean[0])                                 # nothing before, and no shrinkage: no mean
    assert mean[1] == 1.5 and mean[2] == 3.0 and mean[3] == 3.0   # the last two with a value: 2 and 4
    shrunk, _ = cm.prior_mean(group, when, value, np.array([1]), np.array([21]), window=2, k=2.0)
    assert shrunk[0] == 3.0 / 4.0                            # (1 + 2) / (2 + 2)


def _frame() -> pd.DataFrame:
    rows = []
    t0 = pd.Timestamp("2026-01-03 15:00", tz="UTC")
    for i in range(6):                                        # team 7 plays at home each week
        for role, team_h, team_a in (("home", 7, 100 + i), ("away", 7, 100 + i), ("draw", 7, 100 + i)):
            rows.append({"kickoff": t0 + pd.Timedelta(days=7 * i), "cp_min": 60, "role": role, "home_id": team_h,
                         "away_id": team_a, "usable": True, "y_logit": 0.1 * (i + 1) if role == "home" else -0.05,
                         "country_code": "GB"})
    return pd.DataFrame(rows)


def test_memory_uses_a_sides_earlier_matches_and_never_the_match_itself():
    d = cm.memory(_frame())
    home = d[d["role"] == "home"].reset_index(drop=True)
    assert home["mm_own_n"].tolist() == [0, 1, 2, 3, 4, 5]
    # before the third match: the first two home moves (0.1, 0.2), shrunk by k = 5
    assert abs(home["mm_own"].iloc[2] - (0.1 + 0.2) / (2 + cm.MEMORY_K)) < 1e-12
    # changing a match's own move leaves its own memory as it was
    f = _frame()
    f.loc[(f["role"] == "home") & (f["kickoff"] == f["kickoff"].max()), "y_logit"] = 99.0
    again = cm.memory(f)
    last = again[(again["role"] == "home")].iloc[-1]
    assert abs(last["mm_own"] - home["mm_own"].iloc[-1]) < 1e-12
    # an opponent never seen has no memory; the country's earlier home moves are there from the second match
    assert (home["mm_opp_n"] == 0).all() and home["mm_country_n"].tolist() == [0, 1, 2, 3, 4, 5]


def test_memory_waits_for_a_late_kick_off_to_close_before_it_reads_the_match():
    # team 7's first match was due at 15:00 and went in play at 16:30; its second match's T-60 checkpoint is
    # 16:00 the same day: the first match had not closed, so its move is not known yet
    t0 = pd.Timestamp("2026-01-03 15:00", tz="UTC")
    rows = []
    for ko, actual, opp, y in ((t0, t0 + pd.Timedelta(minutes=90), 101, 0.3),
                               (t0 + pd.Timedelta(hours=2), t0 + pd.Timedelta(hours=2), 102, 0.0),
                               (t0 + pd.Timedelta(days=7), t0 + pd.Timedelta(days=7), 103, 0.0)):
        rows.append({"kickoff": ko, "kickoff_actual": actual, "cp_min": 60, "role": "home", "home_id": 7,
                     "away_id": opp, "usable": True, "y_logit": y, "country_code": "GB"})
    late = cm.memory(pd.DataFrame(rows))
    assert late["mm_own_n"].tolist() == [0, 0, 2]            # the 16:00 decision does not see the 16:30 close
    on_time = pd.DataFrame(rows)
    on_time.loc[0, "kickoff_actual"] = t0 + pd.Timedelta(seconds=70)
    assert cm.memory(on_time)["mm_own_n"].tolist() == [0, 1, 2]
    never = pd.DataFrame(rows)
    never["kickoff_actual"] = pd.NaT                         # not in play: the scheduled kick-off stands
    assert cm.memory(never)["mm_own_n"].tolist() == [0, 1, 2]


def test_skill_is_one_minus_sse_over_sse_of_no_move_and_pairs_against_a_base():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.05, 3000)
    day = np.repeat(np.arange(300), 10).astype(str)
    perfect, half, zero = y.copy(), 0.5 * y, np.zeros_like(y)
    assert cm.skill_ci(day, y, zero)["skill"] == 0.0
    assert abs(cm.skill_ci(day, y, perfect)["skill"] - 1.0) < 1e-12
    s = cm.skill_ci(day, y, half, base=zero)
    assert abs(s["skill"] - 0.75) < 1e-12 and abs(s["vs_base"] - 0.75) < 1e-12 and s["vs_base_lo95"] > 0.7
    assert s["days"] == 300 and s["rows"] == 3000


def test_trades_take_the_forecasts_side_and_pay_the_round_trip():
    t = pd.DataFrame({"pred": [0.10, -0.10, 0.01] * 20, "price_ratio": [1.10, 0.90, 2.0] * 20,
                      "day": [f"d{i // 3}" for i in range(60)]})
    r = cm.trades(t, 0.05)
    gross = np.log(1.10)                                      # both trades right: +ln(1.1) and -ln(0.9)
    assert r["bets"] == 40 and r["right_way"] == 1.0
    assert abs(r["gross"] - (np.log(1.10) - np.log(0.90)) / 2) < 1e-12
    want = np.mean([(gross - cm.ROUND_TRIP) * (1 - cm.COMMISSION), (-np.log(0.90) - cm.ROUND_TRIP) * (1 - cm.COMMISSION)])
    assert abs(r["net"] - want) < 1e-12
    assert cm.trades(t, 0.5) == {"bets": 0}


def test_liquid_arm_says_no_data_when_too_few_busy_rows():
    d = pd.DataFrame({"n_ticks_before": [5, 80, 200], "usable": [True, True, False]})
    out = cm.liquid_arm(d, ["x_cp"], ["x_cp"])
    assert out["rows"] == 2 and out["status"].startswith("no data found")
