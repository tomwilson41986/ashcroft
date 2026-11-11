"""Trainer intent (reports/greyhound_roadmap.md F5; the horse model's ``model/intent_features.py`` family): the
connections' choices for today's run, and each trainer's record when making that choice, against the SP.

The situations, each read from the card against the dog's last day (all on the card: grade, distance, trainer, and
whether the last run was a trial):

- ``class_down``: into a weaker grade than last time (a higher grade rank); ``class_up``: into a stronger one.
- ``dist_up`` / ``dist_down``: a longer / shorter trip than last time.
- ``trial_return``: the last run was a trial that came after a break of ``BREAK_DAYS`` or more (a dog trialled back).
- ``new_trainer``: the first run for a trainer other than last time's (a kennel change).

For each, the trainer's **earlier days** in that situation: runs, and actual against expected winners, the expected
from the SP (``sp_p_norm``), shrunk by ``PRIOR`` expected winners toward 1 (as the engine's ``trainer_ae``). The intent
signal is the flag times that A/E's lift over the trainer's overall A/E: a trainer who wins more than the market
expects when dropping a dog in class, dropping one today.

    ti_<situation> (flag), ti_<situation>_ae, ti_<situation>_n, ti_<situation>_lift, ti_any
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.lagsafe import race_lagged_expanding_count, race_lagged_expanding_sum

PREFIX = "ti_"
BREAK_DAYS = 28
PRIOR = 5.0
#: the situations. Until 11 Nov 2026 there was a fifth, "trial_return" (the dog's last outing a trial after a break):
#: no trial reaches the engine's frame (data.clean), so it fired only after an intertrack race, which is a race. A
#: dog's trials are read on their own channel (greyhound/pit/trials.py)
SITUATIONS = ("class_down", "class_up", "dist_up", "dist_down", "new_trainer")


def flags(df: pd.DataFrame, dog_days) -> pd.DataFrame:
    """Each run's situations against the dog's last day (NaN for a dog's first day)."""
    tmp = df[["dog_id", "race_date"]].copy()
    tmp["_grade"] = pd.to_numeric(df.grade, errors="coerce").to_numpy()
    # the ladder the grade is a rung of (data.GRADE_SCALE): a class move is read on one ladder only
    scale = df.grade_scale if "grade_scale" in df else df.grade_family if "grade_family" in df else pd.Series("A", index=df.index)
    tmp["_scale"] = pd.factorize(scale.astype(object).where(scale.notna()))[0].astype(float)
    tmp.loc[tmp._scale < 0, "_scale"] = np.nan
    tmp["_dist"] = pd.to_numeric(df.distance_m, errors="coerce").to_numpy()
    tmp["_trainer"] = pd.factorize(df.trainer.astype(str))[0].astype(float)
    dd = dog_days(tmp, [], ["_grade", "_scale", "_dist", "_trainer"])
    prev_grade = dd.back(dd.prev("_grade"))
    prev_scale = dd.back(dd.prev("_scale"))
    prev_dist = dd.back(dd.prev("_dist"))
    prev_trainer = dd.back(dd.prev("_trainer"))
    first_ever = dd.back(dd.prev("_d").isna().astype(float)) == 1
    grade, dist, trainer = tmp._grade.to_numpy(), tmp._dist.to_numpy(), tmp._trainer.to_numpy()
    out = pd.DataFrame(index=df.index)
    with np.errstate(invalid="ignore"):
        prev_grade = np.where(tmp._scale.to_numpy() == prev_scale, prev_grade, np.nan)
        out["class_down"] = (grade > prev_grade).astype(float)
        out["class_up"] = (grade < prev_grade).astype(float)
        out["dist_up"] = (dist > prev_dist + 1).astype(float)
        out["dist_down"] = (dist < prev_dist - 1).astype(float)
        out["new_trainer"] = (trainer != prev_trainer).astype(float)
    known = {"class_down": prev_grade + grade, "class_up": prev_grade + grade, "dist_up": prev_dist,
             "dist_down": prev_dist, "new_trainer": prev_trainer}
    for s, base in known.items():
        out[s] = out[s].where(~np.isnan(base) & ~first_ever)
    return out


def calculate(df: pd.DataFrame, dog_days) -> tuple[pd.DataFrame, list[str]]:
    f = flags(df, dog_days)
    won = pd.to_numeric(df.won, errors="coerce")
    exp = pd.to_numeric(df.sp_p_norm, errors="coerce")
    ok = won.notna() & exp.notna() & ~df.is_trial.astype(bool)     # a trial has no SP and is not a target
    aw_all = race_lagged_expanding_sum(df.assign(_w=won.where(ok)), "trainer", "_w")
    ae_all = race_lagged_expanding_sum(df.assign(_e=exp.where(ok)), "trainer", "_e")
    overall = (aw_all.fillna(0) + PRIOR) / (ae_all.fillna(0) + PRIOR)
    cols = {}
    for s in SITUATIONS:
        on = (f[s] == 1) & ok
        w = race_lagged_expanding_sum(df.assign(_w=won.where(on)), "trainer", "_w")
        e = race_lagged_expanding_sum(df.assign(_e=exp.where(on)), "trainer", "_e")
        n = race_lagged_expanding_count(df.assign(_e=exp.where(on)), "trainer", "_e")
        ae = (w.fillna(0) + PRIOR) / (e.fillna(0) + PRIOR)
        cols[f"{PREFIX}{s}"] = f[s]
        cols[f"{PREFIX}{s}_ae"] = ae
        cols[f"{PREFIX}{s}_n"] = n
        cols[f"{PREFIX}{s}_lift"] = f[s] * (ae - overall)
    cols[f"{PREFIX}any"] = f[list(SITUATIONS)].sum(axis=1, min_count=1)
    out = pd.DataFrame(cols, index=df.index).astype(np.float32)
    return out, list(out.columns)
