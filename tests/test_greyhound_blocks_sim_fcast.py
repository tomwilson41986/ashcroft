"""The race simulator (greyhound/blocks/sim.py) and Betfair's forecast price (greyhound/forecast_price.py,
greyhound/blocks/fcast.py). Both are also in the fast loop's leakage test (tests/test_greyhound_fastloop.py)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from greyhound import blocks, forecast_price as fp
from greyhound.blocks import fcast, sim
from tests.test_greyhound_fastloop import synthetic_cache


def race(esr, fsr=None, traps=None, rid=1):
    k = len(esr)
    return pd.DataFrame({"race_id": rid, "race_date": "2026-11-03", "race_time": "14:00", "track": "Romford",
                         "trap": traps or list(range(1, k + 1)), "esr_mean3": esr,
                         "fsr_mean3": fsr if fsr is not None else [0.0] * k, "gsr_std6": [2.0] * k})


def test_a_much_faster_dog_wins_most_simulations():
    df = race([4.0, 0.0, 0.0, 0.0, 0.0, 0.0], fsr=[4.0, 0, 0, 0, 0, 0])
    sim.build(df)
    assert df.sim_win_p.iloc[0] > 0.8 and df.sim_lead_p.iloc[0] > 0.8
    assert np.isclose(df.sim_win_p.sum(), 1.0) and np.isclose(df.sim_lead_p.sum(), 1.0)
    assert np.isclose(df.sim_top2_p.sum(), 2.0) and df.sim_win_vs_best.max() == 1.0


def test_the_checks_fall_on_the_dog_behind_a_neighbour():
    # traps 1 and 2 level on early pace and next to each other; the trap 8 dog has no neighbour within one trap
    df = race([2.0, 2.0, -3.0, -3.0, -3.0, 2.0], traps=[1, 2, 4, 5, 6, 8])
    sim.build(df)
    assert df.sim_lost.iloc[0] > 0 and df.sim_lost.iloc[1] > 0 and df.sim_lost.iloc[5] == 0.0


def test_the_simulation_is_repeatable_and_handles_debutants():
    df = race([1.0, np.nan, 0.0, -1.0])
    a, b = df.copy(), df.copy()
    sim.build(a)
    sim.build(b)
    pd.testing.assert_frame_equal(a, b)
    assert a.sim_win_p.notna().all() and np.isclose(a.sim_win_p.sum(), 1.0)


def test_the_simulator_on_the_fast_loop_matrix():
    df, _ = synthetic_cache(n_days=4, races_a_day=5)
    feats = blocks.attach(df, "sim")
    assert all(f.startswith("sim_") for f in feats) and df[feats].notna().all().all()
    assert np.allclose(df.groupby("race_id").sim_win_p.sum(), 1.0)
    # the simulation follows early pace: the race's fastest early dog wins more often than the slowest
    g = df.sort_values("esr_mean3").groupby("race_id")
    assert (g.sim_win_p.last() >= g.sim_win_p.first()).mean() > 0.8


def cat_rows():
    rows = []
    for i, (num, den) in enumerate([(2, 1), (3, 1), (5, 1), (6, 4)]):
        rows.append({"market_id": "1.9", "market_type": "WIN", "market_start_utc": "2026-11-03T14:00:00Z",
                     "venue": "Romford", "selection_id": 100 + i, "runner_name": f"{i + 1}. Dog {i}",
                     "forecast_num": num, "forecast_den": den, "recorded_utc": "2026-11-03T08:00:00Z"})
    rows.append({**rows[0], "market_type": "PLACE", "market_id": "1.10"})        # a place market: not a forecast
    return pd.DataFrame(rows)


def test_the_forecast_price_from_the_catalogue():
    t = fp.from_catalogue(cat_rows())
    assert len(t) == 4 and t.race_time.iloc[0] == "14:00" and t.race_date.iloc[0] == "2026-11-03"
    assert t.sort_values("trap").fc_price.tolist() == [3.0, 4.0, 6.0, 2.5]
    assert len(fp.from_catalogue(cat_rows().assign(forecast_num=np.nan))) == 0     # none recorded: none made up


def test_the_forecast_loads_recorded_days_only():
    seen = []

    def read(d, name):
        seen.append(d)
        return cat_rows() if d == date(2026, 11, 3) else None
    t = fp.load(date(2026, 9, 1), date(2026, 11, 3), read=read)
    assert min(seen) == fp.FIRST_DAY and len(t) == 4


def test_the_forecast_block_joins_by_track_date_time_trap():
    df = race([0.0] * 4)
    fcast.set_table(fp.from_catalogue(cat_rows()))
    try:
        cols = fcast.build(df)
    finally:
        fcast.set_table(None)
    assert cols == ["fc_p", "fc_logp_z", "fc_rank", "fc_cover"]
    assert np.isclose(df.fc_p.sum(), 1.0) and df.fc_rank.tolist() == [2.0, 3.0, 4.0, 1.0]
    assert (df.fc_cover == 1.0).all()
    other = race([0.0] * 4).assign(track="Hove")
    fcast.set_table(fp.from_catalogue(cat_rows()))
    try:
        fcast.build(other)
    finally:
        fcast.set_table(None)
    assert other.fc_p.isna().all() and (other.fc_cover == 0).all()          # no recorded forecast: NaN
