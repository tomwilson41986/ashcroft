"""The served residual memory (greyhound/memory.py) matches the layer the ledger scored (research/greyhound/
residual_memory.py, time-decay form) on constructed prices; it reads earlier days only; it keeps each race's book;
it writes the out-of-sample prices with the layer walk-forward. Constructed data for the unit tests only."""

import io
from datetime import date

import numpy as np
import pandas as pd

from greyhound import memory as gm
from research.greyhound import residual_memory as rm
from sources.common import Store


def frame(n_days=20, races=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    dogs = np.arange(1, 121)
    bias = rng.normal(0, 0.3, len(dogs) + 1)
    for d in range(n_days):
        ds = str(date(2026, 10, 1 + d))
        for r in range(races):
            field = rng.choice(dogs, 6, replace=False)
            p = rng.dirichlet(np.ones(6) * 3)
            won = rng.choice(6, p=p)
            for i, dog in enumerate(field):
                pb = 1 / p[i] * 1.02
                rows.append({"race_id": f"{ds}-{r}", "race_date": ds, "race_time": f"{12 + r // 6}:{(r % 6) * 9:02d}",
                             "track": "Romford", "trap": i + 1, "dog_id": int(dog), "p_model": p[i], "pred_bsp": pb,
                             "won": int(i == won), "bsp": pb * np.exp(bias[dog] + rng.normal(0, 0.2))})
    return pd.DataFrame(rows)


def test_matches_the_scored_layer():
    f = frame()
    ref = rm.decayed(rm.misses(f), gm.TAU)
    ref = rm.apply(ref, {"k": gm.K, "b": gm.B, "c": gm.C})
    dm = gm.misses(f.assign(pred_bsp_base=f.pred_bsp))
    for ds in ("2026-10-05", "2026-10-20"):
        day = f[f.race_date == ds]
        out = gm.apply(day, gm.memory_on(dm, date.fromisoformat(ds)))
        r = ref[ref.race_date == ds].set_index(["race_id", "dog_id"]).loc[out.set_index(["race_id", "dog_id"]).index]
        assert np.allclose(out.pred_bsp.to_numpy(), r.pred_bsp_m.to_numpy(), rtol=1e-9)
        assert np.allclose(out.p_model.to_numpy(), r.p_model_m.to_numpy(), rtol=1e-9)


def test_earlier_days_only_and_the_book_kept():
    f = frame()
    dm = gm.misses(f.assign(pred_bsp_base=f.pred_bsp))
    ds = "2026-10-12"
    later = f.copy()
    later.loc[later.race_date >= ds, "bsp"] *= 3.0                       # the day's own and later BSPs rewritten
    dm2 = gm.misses(later.assign(pred_bsp_base=later.pred_bsp))
    a = gm.memory_on(dm, date.fromisoformat(ds)).set_index("dog_id").m
    b = gm.memory_on(dm2, date.fromisoformat(ds)).set_index("dog_id").m
    assert np.allclose(a.sort_index(), b.sort_index())
    day = f[f.race_date == ds]
    out = gm.apply(day, gm.memory_on(dm, date.fromisoformat(ds)))
    assert np.allclose(out.groupby("race_id").p_model.sum(), 1.0)
    lb, la = np.log(day.pred_bsp).groupby(day.race_id).mean(), np.log(out.pred_bsp).groupby(out.race_id).mean()
    assert np.allclose(lb, la)
    assert (out.memory_m != 0).any() and np.allclose(out.pred_bsp_base, day.pred_bsp)


def test_no_memory_leaves_the_price(monkeypatch):
    f = frame(n_days=2)
    out = gm.apply(f, pd.DataFrame(columns=["dog_id", "m", "n"]))
    assert np.allclose(out.pred_bsp, f.pred_bsp) and np.allclose(out.p_model, f.p_model)
    monkeypatch.setenv("GREYHOUND_MEMORY", "off")
    same, notes = gm.serve(None, date(2026, 10, 3), f, None)
    assert same is f and "off" in notes["memory"]


def test_history_and_oos_files(tmp_path):
    f = frame(n_days=6)
    store = Store(root=tmp_path)
    base = "greyhound/live/oos_prices_test"
    for ds, g in f.groupby("race_date"):
        if ds < "2026-10-05":
            buf = io.BytesIO()
            g.drop(columns=["bsp", "won"]).to_parquet(buf, index=False)
            store.put(f"{base}/{ds}.parquet", buf.getvalue())
        else:                                   # a served day: the predictions file
            store.put(f"{gm.PRED_PREFIX}/{ds}.csv", g.drop(columns=["bsp", "won"]).to_csv(index=False).encode())
    prices = f.assign(market="win", country="GB")[["market", "country", "race_date", "race_time", "track", "trap", "bsp"]]
    h = gm.history(store, date(2026, 10, 7), base, prices=prices)
    assert set(h.race_date) == {f"2026-10-0{i}" for i in range(1, 7)} and h.bsp.notna().all()
    w = gm.ensure_oos(store, base, prices=prices)
    assert w["written"] == 4 and gm.ensure_oos(store, base, prices=prices)["written"] == 0
    first = store.get_parquet(f"{base}_mem/2026-10-01.parquet")
    assert np.allclose(first.pred_bsp, first.pred_bsp_base)               # the first day has no earlier misses
    fourth = store.get_parquet(f"{base}_mem/2026-10-04.parquet")
    assert (fourth.memory_m != 0).any()
