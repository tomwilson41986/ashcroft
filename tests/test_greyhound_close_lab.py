"""research/greyhound/close_lab.py: the checkpoints rebuilt from the stream with the place market joined and the SP
as the target; each spec fitted on earlier days only; the baselines; the champion only on fresh days."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from research.greyhound import close_lab as cl

OFF = datetime(2026, 11, 5, 14, 0, tzinfo=timezone.utc)


def line(t, mc):
    return json.dumps({"rt": t.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                       "m": json.dumps({"op": "mcm", "pt": int(t.timestamp() * 1000), "mc": [mc]})})


def defn(mtype, status="OPEN", bsp=None):
    rs = [{"id": i, "status": "ACTIVE", **({"bsp": bsp[i]} if bsp else {})} for i in (1, 2)]
    return {"marketTime": OFF.strftime("%Y-%m-%dT%H:%M:%S.000Z"), "inPlay": False, "status": status,
            "marketType": mtype, "eventId": "e1", "venue": "Romford", "runners": rs}


def test_snapshots_and_the_frame():
    t0 = OFF - timedelta(seconds=400)
    lines = [line(t0, {"id": "1.1", "img": True, "marketDefinition": defn("WIN"), "rc": [
                 {"id": 1, "atb": [[2.0, 50], [1.99, 20]], "atl": [[2.02, 40]], "spn": 2.0, "tv": 100, "ltp": 2.0},
                 {"id": 2, "atb": [[3.0, 30]], "atl": [[3.1, 30]], "spn": 3.1, "tv": 50, "ltp": 3.05}]}),
             line(t0, {"id": "1.2", "img": True, "marketDefinition": defn("PLACE"), "rc": [
                 {"id": 1, "atb": [[1.3, 10]], "atl": [[1.32, 10]]}, {"id": 2, "atb": [[1.6, 10]], "atl": [[1.65, 10]]}]}),
             line(OFF - timedelta(seconds=100), {"id": "1.1", "rc": [{"id": 1, "tv": 160}]}),
             line(OFF + timedelta(minutes=2), {"id": "1.1", "marketDefinition": defn("WIN", "CLOSED",
                                                                                      bsp={1: 1.9, 2: 3.3})})]
    snap = cl.snapshots(lines)
    assert set(snap.check) == set(cl.CHECKS) and set(snap.mtype) == {"WIN", "PLACE"}
    f = cl.frame(snap)
    assert set(f.sid) == {1, 2} and np.isclose(f.y.iloc[0], np.log(f.bsp.iloc[0]))
    r = f[(f.sid == 1) & (f.check == 60)].iloc[0]
    assert r.p_mid > 1 and r.tv == 160 and r.tv_d60 == 60                     # 100 at 120 s out, 160 at 60 s out
    assert np.isclose(r.spn_vs_mid, np.log(2.0) - r.l_mid)


def synth(day_seed, n=600):
    """A day where log(SP) = log(mid) + 0.5 x spn_vs_mid + noise: a model reading spn_vs_mid should beat the mid."""
    rng = np.random.default_rng(day_seed)
    mid = np.exp(rng.uniform(0.3, 2.5, n))
    gap = rng.normal(0, 0.1, n)
    d = pd.DataFrame({"market": [f"m{day_seed}_{i // 6}" for i in range(n)], "sid": np.arange(n) % 6,
                      "secs": rng.choice([10, 60, 120, 200], n), "l_mid": np.log(mid), "l_ltp": np.log(mid),
                      "l_spn": np.log(mid) + gap, "spn_vs_mid": gap})
    for fam in cl.FEATURES.values():
        for c in fam:
            if c not in d:
                d[c] = 0.0
    d["y"] = d.l_mid + 0.5 * gap + rng.normal(0, 0.02, n)
    return d


def test_specs_are_fitted_on_earlier_days_and_the_champion_needs_fresh_days():
    days = {"2026-11-05": synth(1), "2026-11-06": synth(2), "2026-11-07": synth(3)}
    specs = [{"name": "ridge_sp", "registered": "2026-11-05", "features": ["sp"], "model": {"kind": "ridge", "lam": 1.0}},
             {"name": "late", "registered": "2026-11-07", "features": ["sp"], "model": {"kind": "ridge", "lam": 1.0}}]
    res = cl.evaluate(days, specs)
    first = res[res.day == "2026-11-05"]
    assert set(first.spec) == set(cl.BASELINES)                             # nothing to train on yet
    later = res[res.day == "2026-11-07"].set_index("spec").rmse
    assert later["ridge_sp"] < later["mid"]
    ch = cl.champion(res, specs, days, incumbent=None)
    assert ch["challenger"] == "ridge_sp" and ch["champion"] == "ridge_sp"   # 'late' has no fresh day yet
    assert ch["paired"]["lo90"] > 0


def test_the_registered_specs_load_with_a_hash():
    specs = cl.load_specs()
    assert len({s["name"] for s in specs}) == len(specs) and all(len(s["hash"]) == 10 for s in specs)
    feats = {f for fam in cl.FEATURES.values() for f in fam}
    assert all(set(s["features"]) <= set(cl.FEATURES) for s in specs) and feats
