"""The connections: what the trainer's runners do, and above all what they do in the situation today's dog is in.

Each run's situation, known before the off: back from a break (28 days or more since the dog's last race), down in
grade, up in grade, the first run for a new trainer, the debut. For the trainer, over the 365 and 30 days before the
run's day (earlier days only; trials excluded, no SP prices them): runners, winners and the market's expected
winners (the SP's chances normalised within the race), as A/E shrunk towards 1 (``K`` expected winners of prior), in
all and in today's situation.

The trainer is the trainer at the race only where GBGB was read within ``core.FRESH_DAYS`` of it (GBGB serves each
dog's CURRENT trainer on past results: core.trainer_at_the_time); elsewhere the trainer is unknown and the trainer's
columns are NaN. The table grows point-in-time from the nightly fetch on; a backfilled day adds nothing to it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core

PREFIX = "cx_"
K = 5.0
SOURCE = "GBGB results (trainer, grade, SP: sources/gbgb.py)"
SITUATIONS = ("layoff", "drop", "rise", "new_trainer", "debut")


def situations(df: pd.DataFrame) -> pd.DataFrame:
    """The run's situation flags, from the dog's previous race (trials excluded) on an earlier day."""
    races = df[~df.is_trial.fillna(False)] if "is_trial" in df else df
    prev = core.prev_day_values(races, "dog_id", ["t", "grade", "trainer", "grade_family"]).reindex(df.index)
    gap = (df.t - prev.t).dt.total_seconds() / 86400.0
    same_fam = prev.grade_family == df.grade_family
    out = pd.DataFrame(index=df.index)
    out["layoff"] = (gap >= 28).astype(float)
    out["drop"] = ((df.grade > prev.grade) & same_fam).astype(float)        # a higher number is a lower grade
    out["rise"] = ((df.grade < prev.grade) & same_fam).astype(float)
    out["new_trainer"] = (prev.trainer.notna() & (prev.trainer != df.trainer)).astype(float)
    out["debut"] = prev.t.isna().astype(float)
    out["gap"] = gap
    return out


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = runs.copy()
    df["trainer"] = core.trainer_at_the_time(df, (ctx or {}).get("fetch_lag"))
    sit = situations(df)
    racing = ~df.is_trial.fillna(False) & df.sp_p_norm.notna()
    df["_w"] = df.won.where(racing)
    df["_e"] = df.sp_p_norm.where(racing)
    df["_w"], df["_e"] = df._w.fillna(0.0), df._e.fillna(0.0)
    df["_r"] = racing.astype(float)
    out = pd.DataFrame({"race_id": df.race_id, "dog_id": df.dog_id})
    for days in (365, 30):
        w = core.prior_window(df, ["trainer"], ["_w", "_e", "_r"], days)
        out[f"{PREFIX}tr_ae{days}"] = (w._w + K) / (w._e + K)
        out[f"{PREFIX}tr_runs{days}"] = w._r
    for s in SITUATIONS:
        out[f"{PREFIX}{s}"] = sit[s]
        cell = df.assign(_s=sit[s])
        for c in ("_w", "_e", "_r"):
            cell[c] = cell[c] * cell._s
        w = core.prior_window(cell, ["trainer"], ["_w", "_e", "_r"], 365)
        out[f"{PREFIX}tr_{s}_ae"] = ((w._w + K) / (w._e + K)).where(sit[s] == 1)
        out[f"{PREFIX}tr_{s}_runs"] = w._r.where(sit[s] == 1)
    known = df.trainer.notna()
    for c in [c for c in out.columns if c.startswith(f"{PREFIX}tr_")] + [f"{PREFIX}new_trainer"]:
        out[c] = out[c].where(known)
    out[f"{PREFIX}gap_days"] = sit.gap
    return out.reset_index(drop=True), {"k": K, "runs_with_trainer_at_the_time": int(known.sum()),
                                        "share": round(float(known.mean()), 4)}


COLUMNS = {
    f"{PREFIX}tr_ae365": "the trainer's winners over the SP's expected winners, last 365 days (shrunk to 1)",
    f"{PREFIX}tr_ae30": "the same, last 30 days (the yard's form)",
    f"{PREFIX}tr_runs365": "the trainer's runners, last 365 days",
    f"{PREFIX}tr_runs30": "the trainer's runners, last 30 days",
    **{f"{PREFIX}{s}": f"today's dog: {s.replace('_', ' ')} (1/0)" for s in SITUATIONS},
    **{f"{PREFIX}tr_{s}_ae": f"the trainer's A/E against the SP with runners in the same situation ({s}), 365 days"
       for s in SITUATIONS},
    **{f"{PREFIX}tr_{s}_runs": f"the trainer's runners in that situation ({s}), 365 days" for s in SITUATIONS},
    f"{PREFIX}gap_days": "days since the dog's last race (trials excluded)",
}
