"""research/greyhound/residual_memory.py: the layer that carries a dog's earlier misses forward. A planted lasting
offset is found and removed out of sample; with none planted the layer does nothing; a runner reads earlier days only."""
import numpy as np
import pandas as pd

from research.greyhound import residual_memory as rm


def frame(offset_sd=0.25, days=120, seed=0):
    """Races of six from 240 dogs. The model's price is the dog's true chance but for a lasting offset a dog (the
    market knows it, the model does not) and noise a race."""
    rng = np.random.default_rng(seed)
    dogs = np.arange(240)
    ability = rng.normal(0, 0.5, len(dogs))
    hidden = rng.normal(0, offset_sd, len(dogs))                     # what the market knows and the model does not
    rows, rid = [], 0
    for d in range(days):
        day = (pd.Timestamp("2026-06-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        for field in rng.permutation(dogs).reshape(-1, 6)[:12]:
            rid += 1
            true = ability[field] + hidden[field] + rng.normal(0, 0.15, 6)
            pm = np.exp(ability[field]) / np.exp(ability[field]).sum()
            pt = np.exp(true) / np.exp(true).sum()
            win = rng.choice(6, p=pt)
            for k, dog in enumerate(field):
                rows.append({"race_id": rid, "dog_id": int(dog), "race_date": day, "race_time": "18:00", "won": float(k == win),
                             "p_model": pm[k], "pred_bsp": 1.0 / pm[k], "bsp": 1.0 / pt[k]})
    return pd.DataFrame(rows)


def test_a_lasting_offset_is_found_and_taken_out_of_sample():
    res = rm.run(frame(), "2026-08-15")
    f, s = res["fit"], res["scored"]
    assert 0.5 < f["b"] < 1.2 and f["k"] in rm.KS and f["c"] > 0.3
    assert s["rmse_layer"] < s["rmse"] - 0.03 and s["rmse_z"] < -5 and s["logloss_layer"] < s["logloss"]
    assert res["by_memory"]["5 or more"]["rmse_diff"] < -0.03


def test_no_lasting_offset_no_change():
    res = rm.run(frame(offset_sd=0.0), "2026-08-15")
    s = res["scored"]
    assert abs(s["rmse_diff"]) < 0.004 and abs(res["fit"]["b"]) < 0.25


def test_a_runner_reads_earlier_days_only():
    p = frame(days=40)
    a = rm.misses(p)
    last = p.race_date.max()
    q = p.copy()
    q.loc[q.race_date == last, "bsp"] = q.loc[q.race_date == last, "bsp"] * 3.0          # the last day's results changed
    b = rm.misses(q)
    assert np.allclose(a.sum_before, b.sum_before) and np.array_equal(a.n_before, b.n_before)
    one = a[a.dog_id == 7].sort_values("race_date")
    assert one.n_before.tolist() == list(range(len(one))) and one.sum_before.iloc[0] == 0
    assert np.isclose(one.sum_before.iloc[2], one.miss.iloc[:2].sum())
    # the layer keeps each race's mean log price and its chances sum to one
    out = rm.apply(a, {"k": 2.0, "b": 0.5, "c": 0.5})
    g = out.groupby("race_id")
    assert np.allclose(g.p_model_m.sum(), 1.0) and np.allclose(np.log(out.pred_bsp_m).groupby(out.race_id).mean(),
                                                               np.log(out.pred_bsp).groupby(out.race_id).mean())
