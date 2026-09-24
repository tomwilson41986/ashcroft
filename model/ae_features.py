"""Who does the market misprice? Actual-minus-expected against the price, by entity.

For every earlier race, a runner's residual against the market is
won - pi, with pi its race-normalised BSP probability. Summed over an entity's
earlier runs it says whether the market has been under- or over-pricing that
entity: a trainer whose runners win more often than their prices say, a sire
the market has not caught up with, a draw the market underrates at a course.
If such mispricing persists, the lagged residual predicts the next residual,
which is exactly information beyond the price.

Every aggregate steps back whole days (model.lagsafe), is time-decayed (a
half-life, so an old pattern fades) and is shrunk toward zero by its effective
sample size, so a trainer with three runners does not look like a genius.
All inputs are past results and past prices; nothing about the race being
priced enters its own features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.draw_metrics import race_lagged_decayed_mean

HALF_LIFE_DAYS = 365.0
SHRINK_N = 50.0          # pseudo-runners at a zero residual

ENTITIES = {
    "trainer": ["trainer"],
    "jockey": ["jockey_name"],
    "sire": ["stallion"],
    "trainer_jockey": ["trainer", "jockey_name"],
    "horse": ["horse_name"],
    "trainer_track": ["trainer", "track"],
    "track_draw": ["track", "_dist_band", "_draw_third"],
    "track_front": ["track", "_dist_band", "_price_band"],
}

#: The entities a forecast made BEFORE the off may use. `track_front` keys its cell on
#: today's market rank -- known at the off, which is when the outcome model runs, but it
#: is the very thing a forecast of BSP is trying to predict.
PRE_OFF_ENTITIES = {k: v for k, v in ENTITIES.items() if "_price_band" not in v}


def _race_key(df: pd.DataFrame) -> pd.Series:
    if "raceid" in df.columns:
        return df["raceid"].astype(str)
    return (pd.to_datetime(df["race_date"]).dt.strftime("%Y-%m-%d") + "|" + df["track"].astype(str)
            + "|" + df["race_time"].astype(str))


def add_ae_features(df: pd.DataFrame, half_life_days: float = HALF_LIFE_DAYS,
                    shrink_n: float = SHRINK_N, entities: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    d = df.copy()
    race = _race_key(d)
    d["raceid"] = race
    bsp = pd.to_numeric(d["bfsp"], errors="coerce")
    inv = (1.0 / bsp).where(bsp > 1)
    pi = inv / inv.groupby(race).transform("sum")
    pos = pd.to_numeric(d.get("placing_numerical"), errors="coerce")
    won = (pos == 1).astype(float).where(pos.notna())
    # a residual needs both a price and a result; a runner lacking either contributes nothing
    d["_ae_resid"] = (won - pi).where(pi.notna() & won.notna())
    dist = pd.to_numeric(d.get("dist_furlongs"), errors="coerce")
    d["_dist_band"] = pd.cut(dist, [0, 6.5, 9.5, 13.5, 100], labels=["sprint", "mile", "middle", "staying"]).astype(str)
    stall = pd.to_numeric(d.get("stall"), errors="coerce")
    n = pd.to_numeric(d.get("number_of_runners"), errors="coerce")
    rel = (stall - 1) / (n - 1).replace(0, np.nan)
    d["_draw_third"] = pd.cut(rel, [-0.01, 1 / 3, 2 / 3, 1.01], labels=["low", "mid", "high"]).astype(str)
    # the price band uses TODAY's market rank only as a cell label for PAST residuals of that band,
    # i.e. "how have favourites / outsiders fared at this course and trip" -- known before the off
    rank = bsp.groupby(race).rank(method="min")
    d["_price_band"] = pd.cut(rank, [0, 1, 3, 100], labels=["fav", "2-3", "rest"]).astype(str)

    names = []
    for name, keys in (ENTITIES if entities is None else entities).items():
        if any(k not in d.columns for k in keys):
            continue
        mean, n_eff = race_lagged_decayed_mean(d, keys, "_ae_resid", halflife_days=half_life_days)
        shrunk = (mean.fillna(0.0) * n_eff) / (n_eff + shrink_n)
        d[f"ae_{name}"] = shrunk.where(n_eff > 0)
        d[f"ae_{name}_n"] = n_eff
        names += [f"ae_{name}", f"ae_{name}_n"]
    return d.drop(columns=["_ae_resid", "_dist_band", "_draw_third", "_price_band"]), names
