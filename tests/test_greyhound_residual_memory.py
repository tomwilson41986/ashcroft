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


def test_the_decayed_memory_reads_earlier_days_only_and_fades():
    import numpy as np
    import pandas as pd

    from research.greyhound import residual_memory as rm
    rows = []
    for k, day in enumerate(["2026-06-01", "2026-06-11", "2026-06-12"]):
        for dog, bsp in ((1, 4.0), (2, 2.0)):
            rows.append({"race_id": k, "race_date": day, "dog_id": dog, "won": float(dog == 2), "p_model": 0.5,
                         "bsp": bsp, "pred_bsp": 2.0 if dog == 2 else 2.0})
    p = rm.misses(pd.DataFrame(rows))
    q = rm.decayed(p, 10.0)
    d1 = q[q.dog_id == 1].sort_values("race_date")
    m = float(p[p.dog_id == 1].miss.iloc[0])
    assert d1.n_before.iloc[0] == 0 and d1.sum_before.iloc[0] == 0
    assert np.isclose(d1.n_before.iloc[1], np.exp(-1.0)) and np.isclose(d1.sum_before.iloc[1], m * np.exp(-1.0))
    assert np.isclose(d1.n_before.iloc[2], (np.exp(-1.0) + 1) * np.exp(-0.1))
    # a later day's result rewritten: the earlier days' memories do not move
    r2 = pd.DataFrame(rows)
    r2.loc[r2.race_date == "2026-06-12", "bsp"] = [9.0, 1.5]
    q2 = rm.decayed(rm.misses(r2), 10.0)
    assert np.allclose(q2.sort_values(["race_id", "dog_id"]).sum_before, q.sort_values(["race_id", "dog_id"]).sum_before)
