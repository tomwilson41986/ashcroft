"""
Same-day features from the user's HRB Ratings Machine sets (hrb_ratings).

For each rating set present, adds (all legitimately pre-race):
    hrb_<set>          the rating
    hrb_<set>_rank     within-race rank (1 = top rated)
    hrb_<set>_gap_top  top rating in the race minus this horse's
    hrb_<set>_z        within-race z-score
"""

from __future__ import annotations

import re
import sqlite3

import numpy as np
import pandas as pd


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def load_hrb_frame(db_path: str, sets=None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    q = "SELECT race_results_id, set_id, set_name, rating FROM hrb_ratings WHERE race_results_id IS NOT NULL"
    if sets:
        q += " AND set_id IN (%s)" % ",".join(str(int(s)) for s in sets)
    df = pd.read_sql_query(q, conn); conn.close()
    return df


def pivot_ratings(frame: pd.DataFrame) -> pd.DataFrame:
    """(race_results_id, set) long frame -> one column per set."""
    frame = frame.copy()
    frame["col"] = "hrb_" + frame["set_name"].map(slug)
    return frame.pivot_table(index="race_results_id", columns="col", values="rating", aggfunc="first").reset_index()


def add_hrb_features(df: pd.DataFrame, frame: pd.DataFrame | None = None, db_path: str | None = None, sets=None):
    """Attach per-set rating, rank, gap-to-top and z-score. Returns (df, feature_names)."""
    if frame is None:
        if db_path is None:
            raise ValueError("pass frame= or db_path=")
        frame = load_hrb_frame(db_path, sets)
    if "id" not in df.columns:
        raise ValueError("race_results frame needs the `id` column")
    wide = pivot_ratings(frame)
    d = df.merge(wide, left_on="id", right_on="race_results_id", how="left").drop(columns=["race_results_id"])
    if "raceid" not in d.columns:
        from model.perf_figures import ensure_raceid
        d = ensure_raceid(d)
    feats = []
    for c in [c for c in wide.columns if c.startswith("hrb_")]:
        g = d.groupby("raceid")[c]
        d[f"{c}_rank"] = g.rank(ascending=False, method="min")
        d[f"{c}_gap_top"] = g.transform("max") - d[c]
        d[f"{c}_z"] = (d[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        feats += [c, f"{c}_rank", f"{c}_gap_top", f"{c}_z"]
    return d, feats
