"""research/greyhound/trap_residual.py on planted races: the market frame keeps only full six-runner races with one
winner, the BSP's chance sums to 1 a race, A/E and the return by trap, the model's bias by trap. Planted data only."""
import numpy as np
import pandas as pd

from research.greyhound import trap_residual as tr


def _prices(n_races=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_races):
        bsp = rng.uniform(2.0, 12.0, 6)
        p = (1 / bsp) / (1 / bsp).sum()
        w = rng.choice(6, p=p)
        for t in range(6):
            rows.append({"race_date": f"2025-01-{1 + i % 28:02d}", "race_time": f"{10 + i // 28}:00", "track": "Hove",
                         "trap": t + 1, "bsp": bsp[t], "win_lose": int(t == w), "market": "win", "country": "GB"})
    return pd.DataFrame(rows)


def test_market_frame_keeps_full_races_and_normalises():
    p = _prices()
    bad = p.iloc[:5].assign(race_time="23:59")          # a five-runner race: dropped
    m = tr.market_frame(pd.concat([p, bad]))
    assert len(m) == len(p)
    s = m.groupby(["race_date", "race_time", "track"]).p_mkt.sum()
    assert np.allclose(s, 1.0)
    rows = tr.by_trap(m)
    assert [r["trap"] for r in rows] == [1, 2, 3, 4, 5, 6]
    assert all(0.5 < r["ae"] < 1.5 and r["ae_90"][0] <= r["ae"] <= r["ae_90"][1] for r in rows)


def test_model_bias_by_trap():
    pred = pd.DataFrame({"race_id": np.repeat(np.arange(100), 6), "dog_id": np.arange(600),
                         "race_date": "2026-10-10", "bsp": 5.0, "won": np.tile([1, 0, 0, 0, 0, 0], 100),
                         "pred_bsp": np.tile([5.0, 5.0, 5.0, 4.0, 5.0, 5.0], 100), "p_model": 1 / 6})
    traps = pd.DataFrame({"race_id": pred.race_id, "dog_id": pred.dog_id, "trap": np.tile(np.arange(1, 7), 100)})
    rows = {r["trap"]: r for r in tr.model_by_trap(pred, traps)}
    assert abs(rows[1]["bias_log_bsp"]) < 1e-9
    assert np.isclose(rows[4]["bias_log_bsp"], np.log(5 / 4), atol=1e-4)


def test_model_bias_when_the_predictions_carry_a_trap():
    pred = pd.DataFrame({"race_id": [1] * 6, "dog_id": range(6), "race_date": "2026-10-10", "bsp": 5.0,
                         "won": [1, 0, 0, 0, 0, 0], "pred_bsp": 5.0, "p_model": 1 / 6, "trap": 0})
    traps = pd.DataFrame({"race_id": [1] * 6, "dog_id": range(6), "trap": range(1, 7)})
    assert [r["trap"] for r in tr.model_by_trap(pred, traps)] == [1, 2, 3, 4, 5, 6]
