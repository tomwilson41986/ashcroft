"""Each runner's strongest form measures against this field: within-race z-scores and gaps to the best.

From Lessmann, Sung and Johnson (2009, eq. 14; syndicate reading of 25 Sep):
standardising each input within the race as well as over the database was worth
more than either alone. A price is a within-race quantity -- a horse is short
because its rivals are weak -- but the model sees each runner on its own, with
within-race ranks for about sixty metrics and nothing else about the field. A
rank keeps the order and throws away the distance: first by a street and first
by a whisker rank the same.

For thirty of the most informative per-runner measures (the served model's top
continuous inputs by gain, the race-constant and already-relative ones left out,
and six of the form windows), two readings against today's field:

    rr_<m>_z     (value - the race's mean) / the race's standard deviation
    rr_<m>_gap   value - the race's best (highest) value

over the runners with a value. Pre-race inputs only, so a card row gets exactly
what the same row gets with its results in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.race_shape import race_key

BASE = [
    "LR_ORR2", "official_rating", "LR3_RWO", "LR_LB", "PMW10", "fr_days_since_placed", "LRNFP", "LR3_ORR2",
    "OFS1", "LR10_RWO", "LR5_RWO", "LR_RACE_RB", "fr_days_since_code", "PMW3", "OFS3", "career_runs",
    "trainerjockeycareerNFP", "days_since_lr", "OFS5", "fr_gap_vs_usual", "horse_age", "trainer_sr_30d",
    "trainer_track_win_rate", "course_avg_nfp",
    "fw_mkt_w5", "fw_perf_w5", "fw_lbs_w5", "fw_nfp_w5", "fw_wax_w5", "fw_ae_w5",
]
FEATURES = [f"rr_{m}_{k}" for m in BASE for k in ("z", "gap")]
POST_RACE: set[str] = set()
#: Every column build() reads, and the engine blocks that make some of them.
READS = ["raceid", "race_date", "race_time", "track"] + BASE
ENGINE = ("form_windows", "freshness")


def build(df: pd.DataFrame) -> pd.DataFrame:
    missing = [m for m in BASE if m not in df.columns]
    if missing:
        raise ValueError(f"race_relative needs the engine's {missing[:5]}")
    rk = race_key(df).to_numpy()
    cols = {}
    for m in BASE:
        v = pd.to_numeric(df[m], errors="coerce").astype(float)
        g = v.groupby(rk)
        mean, sd, top = g.transform("mean"), g.transform("std"), g.transform("max")
        with np.errstate(invalid="ignore", divide="ignore"):
            cols[f"rr_{m}_z"] = np.where(sd > 0, (v - mean) / sd, np.where(v.notna() & (sd == 0), 0.0, np.nan))
        cols[f"rr_{m}_gap"] = (v - top).to_numpy(dtype=float)
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
