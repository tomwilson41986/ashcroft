"""The run to the first bend (reports/greyhound_roadmap.md F1; Butler, Butler & Gaine 2025: the bend-1 position
correlates 0.334 with the finish and the trap barely predicts it). From each dog's pre-race early speed (``esr_mean3``,
``esr_max3``), break (``brk_mean6``) and leading record (``led_mean6``), an early-pace score standardised in the race;
then each dog's chance of leading at the bend (a softmax of the score), its projected order, the crowding around it
(dogs within two traps of it with a close score, the ones that reach the bend together) and how much faster the
quickest dog drawn inside it is. Pre-race values of the same race only."""

from __future__ import annotations

import numpy as np
import pandas as pd

PREFIX = "fb2_"
TEMPERATURE = 0.7
CLOSE = 0.5


def _z(x: pd.Series, race: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    x = x.fillna(x.groupby(race).transform("median")).fillna(0.0)
    g = x.groupby(race)
    return ((x - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)


def build(df: pd.DataFrame) -> list[str]:
    race = df.race_id
    parts = [("esr_mean3", 1.0), ("esr_max3", 0.5), ("brk_mean6", 0.5), ("led_mean6", 0.5)]
    score = sum(w * _z(df[c], race) for c, w in parts if c in df.columns)
    if isinstance(score, int):                                   # none of the inputs present
        score = pd.Series(0.0, index=df.index)
    e = np.exp(score / TEMPERATURE - (score / TEMPERATURE).groupby(race).transform("max"))
    lead_p = e / e.groupby(race).transform("sum")
    rank = score.groupby(race).rank(ascending=False, method="average")
    trap = pd.to_numeric(df.trap, errors="coerce").astype(float)
    # crowding: for each dog, the other dogs within two traps, weighted by how close their early-pace score is
    o = pd.DataFrame({"race": race.to_numpy(), "trap": trap.to_numpy(), "s": score.to_numpy()}, index=df.index)
    crowd = pd.Series(0.0, index=df.index)
    inside_gap = pd.Series(0.0, index=df.index)
    srt = o.sort_values(["race", "trap"])
    for k in (1, 2):
        for sgn in (1, -1):
            nb = srt.groupby("race")[["trap", "s"]].shift(sgn * k)
            near = (nb.trap - srt.trap).abs() <= 2
            crowd = crowd.add((np.exp(-(nb.s - srt.s).abs() / CLOSE) * near).fillna(0.0), fill_value=0.0)
    best_inside = srt.groupby("race").s.transform(lambda s: s.shift(1).cummax())
    inside_gap = (best_inside - srt.s).clip(lower=0).fillna(0.0)
    df[f"{PREFIX}score"] = score.astype(np.float32)
    df[f"{PREFIX}lead_p"] = lead_p.astype(np.float32)
    df[f"{PREFIX}rank"] = rank.astype(np.float32)
    df[f"{PREFIX}crowd"] = crowd.reindex(df.index).astype(np.float32)
    df[f"{PREFIX}inside_faster"] = inside_gap.reindex(df.index).astype(np.float32)
    df[f"{PREFIX}lead_x_crowd"] = (lead_p * (1.0 + crowd.reindex(df.index))).astype(np.float32)
    return [f"{PREFIX}{n}" for n in ("score", "lead_p", "rank", "crowd", "inside_faster", "lead_x_crowd")]
