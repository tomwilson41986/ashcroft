"""The GBGB run table, cleaned for modelling: one row a dog a race, in time order.

Reads ``gbgb/runs_<year>.parquet`` from the sources store (S3, or a local folder with ``root``). Drops the rows a model
cannot use (vacant traps, no dog, no finishing position), marks trials, and adds the race key, the off time and the field.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

from sources.common import Store

#: GBGB grades, lower is better: OR (open race) 0, then A1..A11, the sprint (D), stayers (S), hurdle (H) grades by
#: number; B grades are a notch below the A grade of the same number; P (puppy) as A
GRADE_LETTERS = {"OR": 0.0, "A": 0.0, "D": 0.0, "S": 0.0, "H": 0.0, "P": 0.0, "B": 2.0, "HP": 0.0, "IV": 0.0}


def grade_rank(c) -> float:
    m = re.match(r"([A-Z]+)(\d+)?$", str(c or "").strip().upper())
    if not m or m.group(1) not in GRADE_LETTERS:
        return np.nan
    return GRADE_LETTERS[m.group(1)] + float(m.group(2) or 0)


def grade_family(c) -> str:
    """The grade's letter: A (standard), D (sprint), S (stayers), H (hurdles), OR, ..."""
    m = re.match(r"([A-Z]+)", str(c or "").strip().upper())
    return m.group(1) if m else "?"


def load_runs(store: Store | None = None, years=None, frames=None) -> pd.DataFrame:
    """Every GBGB run (the years held, or ``years``), or ``frames`` given directly (tests)."""
    if frames is None:
        store = store or Store()
        keys = sorted(k for k in store.listing("gbgb/") if re.search(r"gbgb/runs_\d{4}\.parquet$", k))
        if years:
            keys = [k for k in keys if int(re.search(r"(\d{4})", k).group(1)) in set(years)]
        frames = [pd.read_parquet(io.BytesIO(store.get(k))) for k in keys]
    df = pd.concat(frames, ignore_index=True)
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # a race card (greyhound/card.py: today's runners, before the off) has no result yet and is kept as it is
    card = df["is_card"].fillna(False).astype(bool) if "is_card" in df else pd.Series(False, index=df.index)
    df = df.assign(is_card=card)
    df = df[df.dog_id.notna() & (df.position.notna() | card) & df.race_id.notna() & df.race_date.notna()]
    # trials (T1..T4, IT) stay: they are form a model reads, but no market prices them, so they are never a target
    df["is_trial"] = df.race_class.astype(str).str.upper().str.match(r"^(T\d*|IT)$")
    df["dog_id"] = df.dog_id.astype("int64")
    df["race_id"] = df.race_id.astype("int64")
    df["raceid"] = df.race_id.astype(str)                                   # the key model.lagsafe groups races on
    df["race_time"] = df.race_time.astype(str).str[:5]
    df["t"] = pd.to_datetime(df.race_date + " " + df.race_time, errors="coerce")
    df = df.dropna(subset=["t"])
    df["field"] = df.groupby("race_id").dog_id.transform("size")
    df["won"] = (df.position == 1).astype(float)
    df["placed2"] = (df.position <= 2).astype(float)
    df.loc[df.is_card, ["won", "placed2"]] = np.nan
    df["grade"] = df.race_class.map(grade_rank)
    df["grade_family"] = df.race_class.map(grade_family)
    df["sp_p"] = 1.0 / df.sp_decimal
    df.loc[~np.isfinite(df.sp_p), "sp_p"] = np.nan
    df["sp_p_norm"] = df.sp_p / df.groupby("race_id").sp_p.transform("sum")
    return df.sort_values(["t", "race_id", "trap"], kind="stable").reset_index(drop=True)
