"""The strongest ratings and form windows read against the field: each one's z-score in the race, rank and gap to
the race's best. The engine reads only a dozen base measures this way (``GreyhoundMetricsEngine._race_relative``);
the Kalman rating, Elo, the normalised-finish and speed-figure windows and the market history are the top features
(reports/greyhound_model.md sections 11, 14) and are read only as they stand. Pre-race values of the same race only."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

PREFIX = "rr2_"
EXACT = ("lb_kr", "lb_elo", "lb_kr_vs_par", "lb_fl3_wr", "trap_bias", "days_since", "runs_before")
PATTERNS = (r"^gp_(nfpz|gsr|ep|lbsc|ae|mkt|bsp_mkt|rs|shape|wax)_(w3|w5|w10|car)$",)


def columns(df: pd.DataFrame) -> list[str]:
    """The inputs, in a fixed order whatever the frame's. Until 11 Nov 2026 the list was cut at 30 in the frame's
    column order, which dropped 17 of 47 - every market-history window the block says it reads - and could keep
    different columns in the cached matrix and in a live frame (the audit of 11 Nov 2026, finding 45)."""
    cols = [c for c in EXACT if c in df.columns]
    for pat in PATTERNS:
        cols += sorted(c for c in df.columns if re.match(pat, c) and c not in cols)
    return cols


def build(df: pd.DataFrame) -> list[str]:
    new = {}
    race = df.race_id
    for c in columns(df):
        x = pd.to_numeric(df[c], errors="coerce").astype(float)
        g = x.groupby(race)
        sd = g.transform("std").replace(0, np.nan)
        new[f"{PREFIX}{c}_z"] = ((x - g.transform("mean")) / sd).astype(np.float32)
        new[f"{PREFIX}{c}_rank"] = g.rank(ascending=False, method="average").astype(np.float32)
        new[f"{PREFIX}{c}_gap"] = (x - g.transform("max")).astype(np.float32)
    for k, v in new.items():
        df[k] = v
    return list(new)
