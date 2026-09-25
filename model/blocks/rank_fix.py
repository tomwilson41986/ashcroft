"""The four within-race ranks the metric audit found pointing the wrong way for missing values, ranked right.

reports/metric_audit.md, finding 2: the engine ranks every measure highest
first with missing values last (custom_metrics.py, rank(ascending=False,
na_option="bottom")). For a measure where lower is better that puts a runner
with no figure beside the best in the race: a debutant ranks with the horse
beaten least. Here the same four sources ranked lowest first, missing values
last, so a runner with no figure ranks behind the worst:

    rfix_LB          career lengths beaten (preracehorsecareerLB), for rLB
    rfix_Consistency the standard deviation of NFP (career_nfp_std), for rConsistency
    rfix_DistApt     distance from the preferred trip (dist_from_preferred), for rDistApt
    rfix_GoingPref   distance from the preferred going (going_from_preferred), for rGoingPref

1 = the best in the race (the lowest value), ties at the lowest rank, as the
engine's ranks. A research variant adds this block and withholds the four
originals (--drop-features rLB,rConsistency,rDistApt,rGoingPref); if it passes,
the engine is fixed at the source and the model retrained.

Pre-race inputs only: the sources are the engine's lagged career figures, so a
card row gets exactly what the same row gets with its result in.
"""

from __future__ import annotations

import pandas as pd

from model.race_shape import race_key

SOURCES = {
    "rfix_LB": "preracehorsecareerLB",
    "rfix_Consistency": "career_nfp_std",
    "rfix_DistApt": "dist_from_preferred",
    "rfix_GoingPref": "going_from_preferred",
}
FEATURES = list(SOURCES)
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track"] + list(SOURCES.values())


def build(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in SOURCES.values() if c not in df.columns]
    if missing:
        raise ValueError(f"rank_fix needs the engine's {missing}")
    rk = race_key(df).to_numpy()
    cols = {}
    for name, src in SOURCES.items():
        v = pd.to_numeric(df[src], errors="coerce").astype(float)
        cols[name] = v.groupby(rk).rank(ascending=True, method="min", na_option="bottom").to_numpy(dtype=float)
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
