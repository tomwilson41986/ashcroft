"""The newer blocks' strongest measures against this field: within-race z-scores and gaps to the best.

The within-race readings (race_relative, iteration 35 and 36: the largest gain
after the form windows) read thirty engine inputs against today's field. The
blocks built since read each horse on its own: its time figure, its exposure
and improvement, and the form windows the readings do not cover (the three-run
weighted window, the served model's strongest). A figure means little until it
is set against the figures it has to beat today. For each measure below, two
readings over the runners with a value:

    rn_<m>_z     (value - the race's mean) / the race's standard deviation
    rn_<m>_gap   value - the race's best (highest) value

Measures: the time figure (tf_best5, tf_w5, tf_l1, tf_best), exposure
(ex_perf_best3, ex_headroom_w5) and the three-run weighted form windows
(fw_mkt_w3, fw_lbsc_w3, fw_perf_w3, fw_nres_w3; fw_lbs_w3 negated, so higher
is better as for the rest). Pre-race inputs only: a card row gets exactly what
the same row gets with its results in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.race_shape import race_key

#: (source, sign): the sign makes higher better for every measure
SOURCES = [
    ("tf_best5", 1.0), ("tf_w5", 1.0), ("tf_l1", 1.0), ("tf_best", 1.0),
    ("ex_perf_best3", 1.0), ("ex_headroom_w5", 1.0),
    ("fw_mkt_w3", 1.0), ("fw_lbsc_w3", 1.0), ("fw_perf_w3", 1.0), ("fw_nres_w3", 1.0), ("fw_lbs_w3", -1.0),
]
FEATURES = [f"rn_{m}_{k}" for m, _ in SOURCES for k in ("z", "gap")]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track"] + [m for m, _ in SOURCES]
ENGINE = ("form_windows",)
AFTER = ("time_figure", "exposure")


def build(df: pd.DataFrame) -> pd.DataFrame:
    missing = [m for m, _ in SOURCES if m not in df.columns]
    if missing:
        raise ValueError(f"race_relative_new needs {missing[:5]} (time_figure and exposure built first)")
    rk = race_key(df).to_numpy()
    cols = {}
    for m, sign in SOURCES:
        v = sign * pd.to_numeric(df[m], errors="coerce").astype(float)
        g = v.groupby(rk)
        mean, sd, top = g.transform("mean"), g.transform("std"), g.transform("max")
        with np.errstate(invalid="ignore", divide="ignore"):
            cols[f"rn_{m}_z"] = np.where(sd > 0, (v - mean) / sd, np.where(v.notna() & (sd == 0), 0.0, np.nan))
        cols[f"rn_{m}_gap"] = (v - top).to_numpy(dtype=float)
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
