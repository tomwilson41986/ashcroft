"""How the market has priced a horse's relations: the sire's runners, the damsire's grandchildren and the dam's other foals.

The price error is largest where a horse's own form is thin (maiden, novice and
bumper races 0.52 against 0.36 in handicaps), and what prices a horse with little or
no form, besides its yard (debut_market.py), is its pedigree. The engine reads the
pedigree only by results: the sire's and damsire's NFP, win rate and WIV, by going
and trip. How the market regards a sire's stock is a different thing, and it is what
the BSP is made of. Over earlier days, each run as the market rated it (the run's BSP
normalised in its race: model/form_windows.run_measures "mkt", 0 for a runner at the
field's average price, positive for one backed), decayed with a half-life of 365
days:

    pdm_sire_debut_mkt     the sire's debutants, shrunk over twenty to all
                           debutants' level on the same days
    pdm_sire_debut_n       the effective number of those debutants
    pdm_sire_young_mkt     the sire's runners at their second to fourth run, the
                           same way
    pdm_sire_mkt           every run of the sire's stock, shrunk over twenty to all
                           runners' level
    pdm_damsire_debut_mkt  the damsire's grandchildren on debut, as the sire's
    pdm_dam_mkt            the dam's other foals, every run, shrunk over five to the
                           sire's stock (the horse's own runs left out)
    pdm_dam_n              the effective number of those runs
    pdm_stage_mkt          the reading for the runner's own stage: the sire's
                           debutants for a debutant, its young runners for a second
                           to fourth run; missing for a horse with more runs
    pdm_stage_mkt_z        pdm_stage_mkt against the field's, over the runners
                           with one

Days before today only (model.race_shape.asof_decayed_mean), so a 06:00 card and the
same rows with their results in agree to the bit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks.race_relative_wide import _z_gap
from model.form_windows import run_measures
from model.race_shape import _codes, asof_decayed_mean, day_index, race_key
from model.shrinkage import shrunk_mean

FEATURES = ["pdm_sire_debut_mkt", "pdm_sire_debut_n", "pdm_sire_young_mkt", "pdm_sire_mkt",
            "pdm_damsire_debut_mkt", "pdm_dam_mkt", "pdm_dam_n", "pdm_stage_mkt", "pdm_stage_mkt_z"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "stallion", "dam_stallion", "dam",
         "career_runs", "number_of_runners", "placing_numerical", "bfsp"]
HALFLIFE = 365.0
K = 20.0            # a sire's (or damsire's) level shrunk to everyone's, in runners
K_DAM = 5.0         # the dam's other foals shrunk to the sire's stock, in runs


def _key(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), -1, dtype=np.int64)
    s = df[col].fillna("").astype(str).str.strip().str.lower()
    return np.where(s.isin(["", "nan", "none"]).to_numpy(), -1, _codes(s)).astype(np.int64)


def build(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    day = day_index(df)
    mkt = run_measures(df)["mkt"]
    runs = pd.to_numeric(df["career_runs"], errors="coerce").to_numpy(dtype=float) if "career_runs" in df.columns \
        else np.full(n, np.nan)
    sire, damsire, dam, horse = _key(df, "stallion"), _key(df, "dam_stallion"), _key(df, "dam"), _key(df, "horse_name")
    zero = np.zeros(n, dtype=np.int64)
    debut, young = runs == 0, (runs >= 1) & (runs <= 3)

    def pop(at):
        m, _ = asof_decayed_mean(zero, day, np.where(at, mkt, np.nan), zero, day, HALFLIFE)
        return np.where(np.isfinite(m), m, 0.0)

    def level(key, at, prior, k):
        m, c = asof_decayed_mean(key, day, np.where(at, mkt, np.nan), key, day, HALFLIFE)
        return np.where(key >= 0, shrunk_mean(np.nan_to_num(m) * c, c, prior, k), np.nan), c

    everyone = np.ones(n, dtype=bool)
    cols = {}
    cols["pdm_sire_debut_mkt"], sire_debut_n = level(sire, debut, pop(debut), K)
    cols["pdm_sire_debut_n"] = np.where(sire >= 0, sire_debut_n, np.nan)
    cols["pdm_sire_young_mkt"], _ = level(sire, young, pop(young), K)
    sire_all, _ = level(sire, everyone, pop(everyone), K)
    cols["pdm_sire_mkt"] = sire_all
    cols["pdm_damsire_debut_mkt"], _ = level(damsire, debut, pop(debut), K)

    # the dam's other foals: the dam's runs less the horse's own, both decayed alike
    dm_m, dm_c = asof_decayed_mean(dam, day, mkt, dam, day, HALFLIFE)
    own_m, own_c = asof_decayed_mean(horse, day, np.where(dam >= 0, mkt, np.nan), horse, day, HALFLIFE)
    others_c = np.clip(dm_c - np.where(np.isfinite(own_m), own_c, 0.0), 0.0, None)
    others_s = np.nan_to_num(dm_m) * dm_c - np.where(np.isfinite(own_m), own_m * own_c, 0.0)
    others_c = np.where(others_c > 1e-9, others_c, 0.0)
    prior = np.where(np.isfinite(sire_all), sire_all, pop(everyone))
    cols["pdm_dam_mkt"] = np.where(dam >= 0, shrunk_mean(np.where(others_c > 0, others_s, 0.0), others_c, prior,
                                                         K_DAM), np.nan)
    cols["pdm_dam_n"] = np.where(dam >= 0, others_c, np.nan)

    stage = np.select([debut, young], [cols["pdm_sire_debut_mkt"], cols["pdm_sire_young_mkt"]], np.nan)
    cols["pdm_stage_mkt"] = stage
    cols["pdm_stage_mkt_z"], _ = _z_gap(pd.Series(stage, index=df.index), race_key(df).to_numpy())
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    assert len(new) == n
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
