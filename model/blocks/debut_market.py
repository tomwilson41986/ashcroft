"""How the market has priced the yard's unknowns: its debutants, second-time-outers and lightly raced runners.

The price error is largest where public form is thin (error-by-segment-958:
maiden, novice and bumper races 0.52 against 0.36 in handicaps). What prices a
debutant is the yard, the rider and the pedigree, and the engine reads the yard
only by results (strike rates, NFP, the intent block's wins less price on
debuts). Bookings (bookings.py) reads how the market rated the yard's runners,
but all of them together: a yard whose handicappers go off short is not one
whose debutants do. For the trainer, over its runners on days before today, by
how many runs they had had (the table's career_runs; a card's is filled from
history, 0 for a debutant), each run as the market rated it (the run's BSP
normalised in its race: model/form_windows.run_measures "mkt", 0 for a runner at
the field's average price, positive for one backed):

    dm_tr_debut_mkt     debutants, decayed (half-life 365 days), shrunk over
                        twenty to all debutants' level on the same days
    dm_tr_debut_n       the effective number of those debutants
    dm_tr_second_mkt    second-time-outers, the same way
    dm_tr_early_mkt     third- and fourth-time-outers, the same way
    dm_trc_debut_mkt    debutants in today's race code (flat, all-weather,
                        hurdle, chase, bumper), shrunk over ten to the yard's own
                        debutant level
    dm_jk_debut_mkt     the rider's debut rides, as dm_tr_debut_mkt
    dm_stage_mkt        the reading for the runner's own stage: the code's
                        debutant level for a debutant, the second-run level for a
                        second-time-outer, the early level for a third or fourth
                        run; missing for a horse with more runs
    dm_stage_mkt_z      dm_stage_mkt against the field's, over the runners with one
    dm_race_debut_share the share of today's field making its debut
    dm_race_unexposed_share  the share with three runs or fewer

Days before today only (model.race_shape.asof_decayed_mean), so a 06:00 card and
the same rows with their results in agree to the bit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks.race_relative_wide import _z_gap
from model.form_windows import run_measures
from model.race_shape import _codes, asof_decayed_mean, day_index, race_code, race_key
from model.shrinkage import shrunk_mean

FEATURES = ["dm_tr_debut_mkt", "dm_tr_debut_n", "dm_tr_second_mkt", "dm_tr_early_mkt", "dm_trc_debut_mkt",
            "dm_jk_debut_mkt", "dm_stage_mkt", "dm_stage_mkt_z", "dm_race_debut_share", "dm_race_unexposed_share"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "trainer", "jockey_name", "career_runs",
         "race_type", "surface_type", "number_of_runners", "placing_numerical", "bfsp"]
HALFLIFE = 365.0
K = 20.0            # shrinkage of a yard's or rider's stage level to all runners' at that stage, in runners
K_CODE = 10.0       # shrinkage of a yard's debutants in one code to its debutants in all


def _key(s: pd.Series) -> np.ndarray:
    s = s.fillna("").astype(str).str.strip().str.lower()
    return np.where(s.isin(["", "nan", "none"]).to_numpy(), -1, _codes(s))


def build(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    day = day_index(df)
    mkt = run_measures(df)["mkt"]
    runs = pd.to_numeric(df["career_runs"], errors="coerce").to_numpy(dtype=float) if "career_runs" in df.columns \
        else np.full(n, np.nan)
    tr = _key(df["trainer"]) if "trainer" in df.columns else np.full(n, -1)
    jk = _key(df["jockey_name"]) if "jockey_name" in df.columns else np.full(n, -1)
    code = pd.factorize(race_code(df), sort=True)[0].astype(np.int64)
    zero = np.zeros(n, dtype=np.int64)
    stages = {"debut": runs == 0, "second": runs == 1, "early": (runs >= 2) & (runs <= 3)}

    def level(key, at_stage, prior, k):
        v = np.where(at_stage, mkt, np.nan)
        m, c = asof_decayed_mean(key, day, v, key, day, HALFLIFE)
        return shrunk_mean(np.nan_to_num(m) * c, c, prior, k), c

    cols = {}
    pop = {}
    for s, at in stages.items():
        pm, _ = asof_decayed_mean(zero, day, np.where(at, mkt, np.nan), zero, day, HALFLIFE)
        pop[s] = np.where(np.isfinite(pm), pm, 0.0)
    tr_debut, tr_debut_n = level(tr, stages["debut"], pop["debut"], K)
    cols["dm_tr_debut_mkt"] = np.where(tr >= 0, tr_debut, np.nan)
    cols["dm_tr_debut_n"] = np.where(tr >= 0, tr_debut_n, np.nan)
    for s in ("second", "early"):
        v, _ = level(tr, stages[s], pop[s], K)
        cols[f"dm_tr_{s}_mkt"] = np.where(tr >= 0, v, np.nan)
    trc = np.where(tr >= 0, tr * (int(code.max()) + 2) + code, -1).astype(np.int64)
    trc_debut, _ = level(trc, stages["debut"], np.where(np.isfinite(tr_debut), tr_debut, pop["debut"]), K_CODE)
    cols["dm_trc_debut_mkt"] = np.where(tr >= 0, trc_debut, np.nan)
    jk_debut, _ = level(jk, stages["debut"], pop["debut"], K)
    cols["dm_jk_debut_mkt"] = np.where(jk >= 0, jk_debut, np.nan)

    stage_mkt = np.select([stages["debut"], stages["second"], stages["early"]],
                          [cols["dm_trc_debut_mkt"], cols["dm_tr_second_mkt"], cols["dm_tr_early_mkt"]], np.nan)
    cols["dm_stage_mkt"] = stage_mkt
    rk = race_key(df).to_numpy()
    cols["dm_stage_mkt_z"], _ = _z_gap(pd.Series(stage_mkt, index=df.index), rk)
    known = pd.Series(np.isfinite(runs), index=df.index)
    per_race = pd.DataFrame({"r": rk, "known": known.to_numpy(), "debut": stages["debut"],
                             "unexposed": np.isfinite(runs) & (runs <= 3)})
    g = per_race.groupby("r")
    n_known = g["known"].transform("sum").to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        cols["dm_race_debut_share"] = np.where(n_known > 0, g["debut"].transform("sum") / n_known, np.nan)
        cols["dm_race_unexposed_share"] = np.where(n_known > 0, g["unexposed"].transform("sum") / n_known, np.nan)
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    assert len(new) == n
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
