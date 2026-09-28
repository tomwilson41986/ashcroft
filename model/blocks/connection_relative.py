"""The connections' recent form against today's field: connection windows as within-race readings.

Connection windows (connection_windows.py) read the trainer's and jockey's
runners over a window ladder and passed on the 944 (iteration 68). They reach the
model as values for each runner alone, with one within-race rank (the career
NFP's). The within-race readings of the horse's own measures (race_relative.py)
were among the largest gains: a price is a within-race quantity, and a rank
keeps the order but throws away the distance. The same two readings, for the
connection windows' recent measures:

    cwr_<e>_<m>_z     (value - the race's mean) / the race's standard deviation
    cwr_<e>_<m>_gap   value - the race's highest value

for the trainer (tr) and the jockey (jk), and <m> the last-20 and last-100 NFP,
the career NFP, the last-100 pounds beaten and the last-100 wins less BSP
chances, over the runners with a value. Built from connection windows' own
columns, which read days before today only, so a card row gets exactly what the
same row gets with its results in.
"""

from __future__ import annotations

import pandas as pd

from model.blocks.connection_windows import FEATURES as CW_FEATURES
from model.blocks.race_relative_wide import _z_gap
from model.race_shape import race_key

MEASURES = ("nfp_l20", "nfp_l100", "nfp_car", "lbs_l100", "ae_l100")
SOURCES = [f"cw_{e}_{m}" for e in ("tr", "jk") for m in MEASURES]
FEATURES = [f"cwr_{e}_{m}_{k}" for e in ("tr", "jk") for m in MEASURES for k in ("z", "gap")]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track"] + SOURCES
AFTER = ("connection_windows",)
assert set(SOURCES) <= set(CW_FEATURES)


def build(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in SOURCES if c not in df.columns]
    if missing:
        raise ValueError(f"connection_relative needs connection windows' {missing[:5]}")
    rk = race_key(df).to_numpy()
    cols = {}
    for c in SOURCES:
        z, gap = _z_gap(pd.to_numeric(df[c], errors="coerce").astype(float), rk)
        name = "cwr_" + c[len("cw_"):]
        cols[f"{name}_z"], cols[f"{name}_gap"] = z, gap
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
