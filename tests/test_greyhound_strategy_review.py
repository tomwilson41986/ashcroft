"""research/greyhound/strategy_review.py: the profit and loss of each bet, the join's guards, and a planted edge found
while noise is not."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.greyhound import strategy_review as sr
from sources.common import Store


def test_the_profit_and_loss_a_unit():
    c = sr.COM
    assert np.allclose(sr.back_hold(np.array([3.0, 3.0]), np.array([1, 0])), [2 * (1 - c), -1])
    assert np.allclose(sr.lay_hold(np.array([3.0, 3.0]), np.array([1, 0])), [-1.0, (1 - c) / 2])
    # backed at 4, closed at 2: 4/2 - 1 = 1 a unit whatever wins; laid at 2, closed at 4: 1 - 2/4 = 0.5 on 1 liability
    assert np.allclose(sr.back_green(np.array([4.0]), np.array([2.0])), [1 - c])
    assert np.allclose(sr.lay_green(np.array([2.0]), np.array([4.0])), [0.5 * (1 - c)])
    assert np.allclose(sr.back_green(np.array([2.0]), np.array([4.0])), [-0.5])          # a loss pays no commission


def synthetic(days=160, seed=0):
    """Races of six; the SP known to the model only partly; the morning price the SP with noise, and for a third of
    the dogs the model knows which way it will move."""
    rng = np.random.default_rng(seed)
    pred, prices, rid = [], [], 0
    for d in range(days):
        day = (pd.Timestamp("2026-06-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        for r in range(12):
            rid += 1
            s = np.exp(rng.normal(0, 0.6, 6))
            q = s / s.sum()
            sp = 1.0 / (q * 1.02)
            won = rng.choice(6, p=q)
            move = rng.normal(0, 0.25, 6)
            morning = sp * np.exp(move)
            knows = rng.random(6) < 0.33
            fc = np.where(knows, sp, morning) * np.exp(rng.normal(0, 0.02, 6))
            t = f"{12 + r // 4}:{(r % 4) * 15:02d}"
            for k in range(6):
                pred.append({"race_id": rid, "race_date": day, "race_time": t, "track": "Romford", "trap": k + 1,
                             "field": 6, "won": int(k == won), "bsp": sp[k], "p_model": q[k], "pred_bsp": fc[k]})
                prices.append({"market": "win", "country": "GB", "race_date": day, "race_time": t,
                               "track": "Romford", "trap": float(k + 1), "event_id": rid, "bsp": sp[k],
                               "ppwap": sp[k] * np.exp(move[k] / 4), "morningwap": morning[k], "morning_vol": 50.0,
                               "pp_vol": 500.0, "win_lose": float(k == won)})
    return pd.DataFrame(pred), pd.DataFrame(prices)


def test_the_join_drops_a_race_on_the_wrong_result(tmp_path):
    pred, prices = synthetic(days=3)
    bad = prices.race_date.eq("2026-06-01") & prices.race_time.eq("12:00")
    prices.loc[bad, "win_lose"] = 1.0 - prices.loc[bad, "win_lose"]
    m, why = sr.join(pred, prices.assign(bsp_place=np.nan, placed=np.nan))
    assert why["kept_races"] == pred.race_id.nunique() - 1


def test_a_planted_move_is_found_and_the_rest_is_not(tmp_path):
    pred, prices = synthetic()
    store = Store(root=tmp_path)
    store.put_parquet("greyhound_prices/prices_2026.parquet", prices)
    out = sr.run(store, pred, "syn", tmp_path / "out")
    rows = pd.read_json(tmp_path / "out" / "review_syn.json", typ="series")["rules"]
    got = {(r["family"], r["rule"]): r for r in rows}
    lay = got[("morning move: lay, closed at the SP", "forecast above the morning price by 0.10 (log), morning volume >= 0, "
                                                     "0 ticks")]
    assert lay["passes"] and lay["holdout"]["return%"] > 5
    # the model's chance is the SP's own: no value at the SP beyond the commission
    sp = [r for r in rows if r["family"].startswith("SP back (the model") and r["holdout"]["bets"] >= 100]
    assert not any(r["passes"] for r in sp)
    assert out["rules_tried"] == len([r for r in rows if not r.get("control")]) and (tmp_path / "out" / "review_syn.md").exists()


def test_the_control_and_the_models_lift(tmp_path):
    pred, prices = synthetic()
    store = Store(root=tmp_path)
    store.put_parquet("greyhound_prices/prices_2026.parquet", prices)
    sr.run(store, pred, "syn", tmp_path / "out")
    out = pd.read_json(tmp_path / "out" / "review_syn.json", typ="series")
    rows = {(r["family"], r["rule"]): r for r in out["rules"]}
    ctrl = rows[("morning control: lay every runner, closed at the SP", "no model, morning volume >= 0, 0 ticks")]
    assert ctrl["control"] and abs(ctrl["holdout"]["return%"]) < 3          # no drift on average: no free money
    lay = rows[("morning move: lay, closed at the SP", "forecast above the morning price by 0.10 (log), morning volume "
                                                        ">= 0, 0 ticks")]
    assert lay["lift_holdout"]["ci90%"][0] > 0 and out["passes_beating_the_control"] >= 1
