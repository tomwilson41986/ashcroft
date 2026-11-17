"""A constructed card for the breeding2 and form2 tests (unit tests only: nothing is evaluated on it). 2 tracks a
day, 6 races of 6 from ``n_dogs`` dogs; each dog's ability its sire's planted effect plus its own."""
import numpy as np
import pandas as pd

from greyhound.data import clean
from greyhound.pit import core


def synth(days=60, seed=0, n_dogs=120, n_sires=8, sire_sd=2.0, trial_every=7):
    rng = np.random.default_rng(seed)
    dogs = np.arange(1, n_dogs + 1)
    sire_eff = rng.normal(0, sire_sd, n_sires)
    sire = dogs % n_sires
    skill = np.r_[0.0, sire_eff[sire] + rng.normal(0, 0.5, n_dogs)]
    months = ["Jan-2022", "Jul-2022", "Jan-2023", "Jul-2023"]
    rows, rid = [], 1000
    for d in range(days):
        day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        for ti, track in enumerate(("Romford", "Hove")):
            for r in range(6):
                rid += 1
                field = rng.choice(dogs, 6, replace=False)
                perf = skill[field] + rng.normal(0, 1.0, 6)
                time_ = 29.0 - perf * core.LENGTH_S
                order = np.argsort(-perf)
                pos = np.empty(6, int)
                pos[order] = np.arange(1, 7)
                sp = np.exp(-skill[field])
                sp = 1 + 5 * sp / sp.sum() * 6
                trial = trial_every and (r == 0 and d % trial_every == 0)
                for k, dog in enumerate(field):
                    rows.append({"race_date": day, "race_time": f"{18 + r}:{10 * ti:02d}", "meeting_id": d * 10 + ti,
                                 "track": track, "race_id": rid, "race_class": "T3" if trial else f"A{1 + dog % 5}",
                                 "distance_m": 480.0 if r < 4 else 640.0, "trap": k + 1, "dog_id": int(dog),
                                 "dog_name": f"Dog {dog}", "sire": f"Sire {sire[dog - 1]}", "dam": f"Dam {dog % 11}",
                                 "born": months[dog % 4], "sex": "b" if dog % 2 else "d", "season": None,
                                 "trainer": f"Trainer {dog % 5}", "sp_decimal": float(sp[k]),
                                 "market_pos": int(np.argsort(np.argsort(sp))[k]) + 1, "position": int(pos[k]),
                                 "beaten_lengths": 0.5, "sectional": 4.8 - skill[dog] * 0.02,
                                 "run_time": float(time_[k]), "adjusted_time": float(time_[k]),
                                 "weight_kg": 30.0 + dog % 5, "comment": "QAw,Rls"})
    return clean(pd.DataFrame(rows)), sire_eff
