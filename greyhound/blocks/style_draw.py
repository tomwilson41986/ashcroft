"""Running style by trap at each track and distance (reports/greyhound_roadmap.md F4; Butler et al. 2025, Gaine 2022:
the middle traps' disadvantage, crowding between styles). A dog's style from its pre-race running line
(``path_mean6``: 1 rails, 2 middle, 3 wide, rounded); the wins of dogs of that style from this trap at this track and
distance on **earlier days** against their fair share (1 / field), shrunk by 5 fair wins of prior, and the same against
the trap's own record, so the feature is what the style adds to the draw."""

from __future__ import annotations

import numpy as np
import pandas as pd

PREFIX = "sd_"
PRIOR = 5.0


def _earlier_days(key: pd.Series, day: pd.Series, won: pd.Series, fair: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Each row's key's wins and fair wins summed over the days before the row's own."""
    g = pd.DataFrame({"key": key.to_numpy(), "day": day.to_numpy(), "w": won.to_numpy(float), "f": fair.to_numpy(float)})
    daily = g.groupby(["key", "day"], sort=True)[["w", "f"]].sum()
    cum = daily.groupby(level=0).cumsum() - daily                 # strictly before the day
    idx = pd.MultiIndex.from_arrays([g.key, g.day])
    c = cum.reindex(idx)
    return pd.Series(c.w.to_numpy(), index=key.index), pd.Series(c.f.to_numpy(), index=key.index)


def build(df: pd.DataFrame) -> list[str]:
    style = pd.to_numeric(df.get("path_mean6"), errors="coerce").round().clip(1, 3).fillna(2).astype(int).astype(str)
    dist = pd.to_numeric(df.get("distance_m"), errors="coerce").fillna(-1).astype(int).astype(str)
    trap = pd.to_numeric(df.trap, errors="coerce").fillna(-1).astype(int).astype(str)
    base = df.track.astype(str) + "|" + dist + "|" + trap
    day = pd.to_datetime(df.t).dt.normalize()
    fair = 1.0 / pd.to_numeric(df.field, errors="coerce").clip(lower=1)
    won = pd.to_numeric(df.won, errors="coerce").fillna(0.0)
    sw, sf = _earlier_days(base + "|" + style, day, won, fair)
    tw, tf = _earlier_days(base, day, won, fair)
    rate = (sw.fillna(0) + PRIOR) / (sf.fillna(0) + PRIOR)
    trap_rate = (tw.fillna(0) + PRIOR) / (tf.fillna(0) + PRIOR)
    df[f"{PREFIX}rate"] = rate.astype(np.float32)
    df[f"{PREFIX}vs_trap"] = (rate / trap_rate).astype(np.float32)
    df[f"{PREFIX}n"] = sf.fillna(0).astype(np.float32)
    df[f"{PREFIX}style"] = style.astype(float).astype(np.float32)
    return [f"{PREFIX}{n}" for n in ("rate", "vs_trap", "n", "style")]
