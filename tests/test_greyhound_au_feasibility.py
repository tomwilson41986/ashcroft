"""research/greyhound/au_feasibility.py on planted data: the BSP join on name, track and UTC day; features read only
earlier days; the conditional logit recovers a planted coefficient; the value rule's arithmetic. Planted data only."""
import numpy as np
import pandas as pd

from research.greyhound import au_feasibility as au


def test_join_on_name_track_utc_day():
    api = pd.DataFrame({"dog_name": ["Fast One"], "track": ["Albion Park"], "start_utc": ["2026-03-01T09:30:00Z"],
                        "race_id": [1], "dog_id": [5]})
    bf = pd.DataFrame({"race_date": ["2026-03-01"], "race_time": ["09:30"], "track": ["Albion Park"], "trap": [1],
                       "dog": ["FAST ONE"], "bsp": [4.2], "win_lose": [1], "event_id": [9]})
    m, notes = au.join_bsp(api, bf)
    assert m.bsp.iloc[0] == 4.2 and notes["matched"] == 1


def _runs(n_days=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(n_days):
        for r in range(3):
            speed = rng.normal(size=6)
            order = np.argsort(-speed - rng.normal(scale=0.5, size=6))
            for box in range(6):
                pos = int(np.where(order == box)[0][0]) + 1
                rows.append({"race_id": day * 10 + r, "dog_id": box + 6 * r, "dog_name": f"d{box}{r}", "track": "T",
                             "start_local": f"2025-{1 + day // 28:02d}-{1 + day % 28:02d}T19:00:00",
                             "trap": box + 1, "position": pos, "run_time_s": 29.5 - 0.1 * speed[box],
                             "adjusted_time_s": None, "sectional_s": 5.3, "distance_m": 520, "class_number": 5,
                             "weight_kg": 30.0, "bf_back": 6.0})
    return pd.DataFrame(rows)


def test_features_read_only_earlier_days():
    d = au.figures(_runs())
    X, cols = au.features(d)
    d2 = d.copy()
    last = d2.race_date == d2.race_date.max()
    d2.loc[last, ["position", "won", "placed", "fig"]] = [6, 0.0, 0.0, -9.0]
    X2, _ = au.features(au.figures(d2.drop(columns=["won", "placed", "fig", "sec", "mkt_share", "mkt_resid", "pos"])))
    assert np.allclose(X[last.to_numpy()].to_numpy(), X2[last.to_numpy()].to_numpy(), equal_nan=True)
    assert "sec_mean3_rank" in cols and X.n_runs.max() > 0


def test_clogit_recovers_a_coefficient():
    rng = np.random.default_rng(3)
    races, k = 3000, 6
    x = rng.normal(size=(races * k, 1))
    u = 1.5 * x[:, 0] + rng.gumbel(size=races * k)
    race = np.repeat(np.arange(races), k)
    won = (pd.Series(u).groupby(race).transform("max").to_numpy() == u).astype(float)
    b, se = au.clogit(x, won, race)
    assert abs(b[0] - 1.5) < 4 * se[0]


def test_value_rule_arithmetic():
    p = np.array([0.5, 0.1])
    bsp = np.array([3.0, 5.0])
    v = au.value_rule(p, bsp, np.array([1, 0]), np.array(["d1", "d1"]), 0.0)
    assert v["bets"] == 1 and abs(v["roi%"] - 100 * (2.0 * 0.98)) < 1e-6
