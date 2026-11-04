"""The live pre-race model's retrain path (greyhound/live.py, 4 Nov 2026): the log-BSP objective rescaled to the SP
book, the matrix blocks attached before pricing, and the binary path kept."""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound import live


def frame(n_races=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=r // 4)
        strength = rng.normal(0, 1, 6)
        p = np.exp(strength) / np.exp(strength).sum()
        won = rng.choice(6, p=p)
        for trap in range(6):
            rows.append({"race_id": f"r{r}", "t": day, "race_date": str(day.date()), "race_time": "14:00",
                         "track": "T", "trap": trap + 1, "dog_id": f"d{r}{trap}", "dog_name": "x", "race_class": "A3",
                         "distance_m": 480, "field": 6, "is_trial": False, "is_card": False,  # the SP book 1.15 over
                         "won": int(trap == won), "x1": strength[trap] + rng.normal(0, 0.3),
                         "x2": rng.normal(), "bsp": 1.0 / (p[trap] * 1.15)})
    return pd.DataFrame(rows)


def test_the_logbsp_fit_and_its_prices():
    df = frame()
    m, meta = live.fit(df, ["x1", "x2"], before="2024-04-01", date_from="2024-01-01", objective="logbsp")
    assert meta["objective"] == "logbsp" and 1.0 < meta["book"] < 1.3 and meta["rounds"] > 0
    card = df[df.t >= "2024-04-01"].head(12).assign(is_card=True)
    key = card[["race_id", "trap"]].assign(market_id="1.1", selection_id=1, matched_history=True)
    out = live.price_cards(card, m, ["x1", "x2"], key, meta)
    assert np.allclose(out.groupby("race_id").p_model.sum(), 1.0)                 # a whole book of chances
    assert np.allclose((1.0 / out.pred_bsp).groupby(out.race_id).sum(), meta["book"], atol=0.01)
    x1 = out.merge(card[["race_id", "trap", "x1"]], on=["race_id", "trap"]).x1
    assert np.corrcoef(x1, np.log(out.pred_bsp))[0, 1] < -0.5                       # stronger dogs shorter
    mb, metab = live.fit(df, ["x1", "x2"], before="2024-04-01", date_from="2024-01-01")
    outb = live.price_cards(card, mb, ["x1", "x2"], key, metab)
    assert "pred_bsp" not in outb and np.allclose(outb.groupby("race_id").p_model.sum(), 1.0)


def test_the_matrix_blocks_attach_once(monkeypatch):
    import greyhound.blocks as gblocks
    calls = []

    def fake_attach(df, name):
        calls.append(name)
        df[f"{name}_f1"] = 1.0
        return [f"{name}_f1"]
    monkeypatch.setattr(gblocks, "attach", fake_attach)
    df = pd.DataFrame({"a": [1.0]})
    assert live.attach_blocks(df, ("sim",)) == ["sim_f1"] and calls == ["sim"]
    assert live.attach_blocks(df, ()) == []
    assert live.LIVE_OBJECTIVE == "logbsp" and "times" in live.LIVE_BLOCKS and live.LIVE_MATRIX_BLOCKS == ("sim",)
