"""The owner's normalised finishing position (6 Oct 2026): a place as a z-score over its field, divided by 3.

    NFPz = (N + 1 - 2F) / (3 sqrt((1/3)(N + 1)/(N - 1)) (N - 1))       = (2 NFP - 1) sqrt((N - 1)/(3 (N + 1)))

Each race's finishers average 0 with a spread of 1/3, and a place counts for more in a big field: a win scores 0.33
in a two-runner race and 0.55 in a twenty-runner one, where the engine's NFP, (N - F)/(N - 1), scores both 1. On
61,423 GB/IE races from 2022 it predicted the winner a little better than the engine's NFP over every window
(+0.5 to +0.9 millinats a race, research/queries/done/nfp_formula_check.py), and neither adds anything beside the
BSP; this block puts it in front of the served model, which forecasts the BSP.

    nz_car  nz_l1  nz_m3  nz_m5  nz_m10  nz_w5     career, last run, the mean of the last 3, 5 and 10 runs, and
                                                    the last 5 weighted 5..1 (form_variants' ladder)
    nz_e3  nz_e6                                    exponential in runs, half-lives 3 and 6
    nz_d365                                         exponential in days, a half-life of a year

Non-finishers are missing (as the engine's NFP). Earlier days only: every window reads the horse's earlier racing
days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks.form_variants import _ladder
from model.freshness_features import _Days, _col
from model.race_shape import _codes, asof_decayed_mean, day_index, field_size, horse_decayed_prior

PREFIX = "nz_"
WINDOWS = ("car", "l1", "m3", "m5", "m10", "w5", "e3", "e6", "d365")
FEATURES = [f"{PREFIX}{w}" for w in WINDOWS]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "number_of_runners", "placing_numerical"]


def nfp_z(n, pos) -> np.ndarray:
    """The owner's NFP of each run: (N + 1 - 2F)/(3 sqrt((N + 1)/(3 (N - 1))) (N - 1)); missing for a
    non-finisher or a field of one."""
    n = np.asarray(n, float)
    pos = np.asarray(pos, float)
    ok = np.isfinite(n) & np.isfinite(pos) & (n >= 2) & (pos >= 1) & (pos <= n)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (n + 1 - 2 * pos) / (3 * np.sqrt((n + 1) / (3 * (n - 1))) * (n - 1))
    return np.where(ok, z, np.nan)


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    name = df["horse_name"].fillna("").astype(str).str.strip().str.lower()
    bad = name.isin(["", "nan", "none"]).to_numpy()
    horse = np.where(bad, -1, _codes(name))
    H = _Days(horse, day)
    pos = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(dtype=float)
    v = nfp_z(field_size(df).to_numpy(dtype=float), np.where(pos > 0, pos, np.nan))

    cols: dict[str, np.ndarray] = {}
    lad = _ladder(H, H.per_block(v))
    for w in ("car", "l1", "m3", "m5", "m10", "w5"):
        cols[f"{PREFIX}{w}"] = H.to_rows(lad[w])
    for half in (3, 6):
        sums, cnts, _ = horse_decayed_prior(np.where(bad, "", name), day, {"v": v}, halflife_runs=half)
        with np.errstate(invalid="ignore", divide="ignore"):
            cols[f"{PREFIX}e{half}"] = np.where(bad | ~(cnts["v"] > 0), np.nan, sums["v"] / cnts["v"])
    cols[f"{PREFIX}d365"], _ = asof_decayed_mean(horse, day, v, horse, day, halflife_days=365.0)

    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
