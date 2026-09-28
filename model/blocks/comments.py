"""What the in-running comments of a horse's earlier runs say: trouble, slow starts, keenness, a kind ride, a fade.

A form figure records where a horse finished, not why. One that was hampered,
denied a clear run or slowly away and still finished close ran better than its
figure; one that was eased, or ran green, was not asked for its best; one that
raced keen or wide spent energy the figure does not show; one that weakened
out of it may have found the trip too far. The form book is read for these
"eyecatchers", and the market reads it too, so what the price forecaster can
learn from them is how the market prices them. The classes and their patterns
are model/comment_features.py's (COMMENT_CLASSES); this block reads them the
way a drop-in block has to, from EARLIER DAYS only:

    cm_lr_<k>            the class in the horse's last run (on an earlier day)
    cm_l3_<k>, cm_l6_<k> its share of the horse's last 3 and last 6 runs with a
                         comment
        for k in trouble, switched, slow_start, keen, wide, finished_well,
        tender, weakened, no_finish, problem, easy_win, and excuse (any of
        trouble, slow start, keen, wide, problem)
    cm_excuse_close      an excuse last time and beaten 5 lengths or less
    cm_excuse_nfp        an excuse last time x its normalised finishing position
    cm_noexcuse_poor     no excuse last time and in the bottom half of the field
    cm_tender_close      a kind ride last time and beaten 5 lengths or less
    cm_finished_well_nfp ran on last time x its normalised finishing position
    cm_easy_win          won easily last time
    cm_runs_since_trouble, cm_runs_since_excuse
                         runs since the last one with trouble (with an excuse);
                         NaN when there has been none

The windows count a horse's rows on earlier days, so a second row for the
same horse on the same day (a duplicate in the feed) reads nothing of the
first. The shares are sums of 0/1 flags, exact in floating point, so no
feature moves by rounding when another horse's day changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.comment_features import COMMENT_CLASSES, EXCUSES, comment_flags
from model.lagsafe import race_minutes
from model.perf_figures import parse_beaten_lengths
from model.race_shape import _codes, day_index

CLASSES = list(COMMENT_CLASSES) + ["excuse"]
WINDOWS = (3, 6)
FEATURES = ([f"cm_lr_{k}" for k in CLASSES]
            + [f"cm_l{w}_{k}" for w in WINDOWS for k in CLASSES]
            + ["cm_excuse_close", "cm_excuse_nfp", "cm_noexcuse_poor", "cm_tender_close",
               "cm_finished_well_nfp", "cm_easy_win", "cm_runs_since_trouble", "cm_runs_since_excuse"])
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "comment", "placing_numerical",
         "number_of_runners", "total_dst_bt"]


def _lengths(s: pd.Series) -> np.ndarray:
    """Beaten lengths, parsed once per distinct string."""
    codes, uniq = pd.factorize(s.astype("string").fillna(""))
    vals = np.array([parse_beaten_lengths(u) for u in uniq], dtype=float)
    return np.where(codes >= 0, vals[np.clip(codes, 0, None)] if len(vals) else np.nan, np.nan)


def build(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    name = df["horse_name"].fillna("").astype(str).str.strip().str.lower()
    day = day_index(df)
    horse = np.where(name.isin(["", "nan", "none"]).to_numpy() | (day < 0), -1, _codes(name))
    t = race_minutes(df["race_time"]).to_numpy(dtype=float) if "race_time" in df.columns else np.zeros(n)
    # a tiebreak that is a function of the row, not of where it arrives
    tie = pd.factorize(df["raceid" if "raceid" in df.columns else "track"].astype(str), sort=True)[0]

    comments = df["comment"] if "comment" in df.columns else pd.Series(np.nan, index=df.index)
    flags = comment_flags(comments).reset_index(drop=True)
    flags = flags.rename(columns=lambda c: c[3:])                 # "_c_trouble" -> "trouble"
    pos = pd.to_numeric(df["placing_numerical"], errors="coerce").to_numpy(dtype=float)
    runners = pd.to_numeric(df["number_of_runners"], errors="coerce").to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        nfp = np.clip((runners - pos) / np.where(runners > 1, runners - 1, np.nan), 0, 1)
    lb = _lengths(df["total_dst_bt"]) if "total_dst_bt" in df.columns else np.full(n, np.nan)
    lb = np.where(pos == 1, 0.0, lb)

    # the horse's rows in order: horse, day, off time, then the race as a tiebreak
    order = np.lexsort((tie, t, day, horse))
    hs, ds_day = horse[order], day[order]
    new_horse = np.r_[True, hs[1:] != hs[:-1]]
    new_day = new_horse | np.r_[True, ds_day[1:] != ds_day[:-1]]
    idx = np.arange(n)
    gs = np.maximum.accumulate(np.where(new_horse, idx, 0))        # the horse's first row
    ds = np.maximum.accumulate(np.where(new_day, idx, 0))          # the first row of its day
    prev = ds - 1                                                  # its last row on an earlier day
    has_prev = (prev >= gs) & (hs >= 0)
    p = np.where(has_prev, prev, 0)

    def last(v: np.ndarray) -> np.ndarray:
        return np.where(has_prev, v[p], np.nan)

    out = {}
    for k in CLASSES:
        f = flags[k].to_numpy(dtype=float)[order]
        ok = np.isfinite(f)
        out[f"cm_lr_{k}"] = last(np.where(ok, f, np.nan))
        cv = np.r_[0.0, np.cumsum(np.where(ok, f, 0.0))]
        cm = np.r_[0.0, np.cumsum(ok.astype(float))]
        for w in WINDOWS:
            lo = np.maximum(ds - w, gs)
            s, c = cv[ds] - cv[lo], cm[ds] - cm[lo]
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"cm_l{w}_{k}"] = np.where((c > 0) & (hs >= 0), s / np.where(c > 0, c, 1.0), np.nan)
        if k in ("trouble", "excuse"):
            mark = np.where(ok & (f > 0), idx, -1)
            latest = np.maximum.accumulate(mark)                   # the latest flagged row so far
            before = np.where(has_prev, latest[p], -1)
            out[f"cm_runs_since_{k}"] = np.where((before >= gs) & has_prev, (ds - before).astype(float), np.nan)

    lr_nfp, lr_lb = last(nfp[order]), last(lb[order])
    exc, tender = out["cm_lr_excuse"], out["cm_lr_tender"]
    both = lambda a, b: np.isfinite(a) & np.isfinite(b)            # noqa: E731
    with np.errstate(invalid="ignore"):
        out["cm_excuse_close"] = np.where(both(exc, lr_lb), exc * (lr_lb <= 5), np.nan)
        out["cm_excuse_nfp"] = exc * lr_nfp
        out["cm_noexcuse_poor"] = np.where(both(exc, lr_nfp), (1 - exc) * (lr_nfp < 0.5), np.nan)
        out["cm_tender_close"] = np.where(both(tender, lr_lb), tender * (lr_lb <= 5), np.nan)
        out["cm_finished_well_nfp"] = out["cm_lr_finished_well"] * lr_nfp
        out["cm_easy_win"] = np.where(both(out["cm_lr_easy_win"], lr_nfp),
                                      out["cm_lr_easy_win"] * (lr_nfp == 1), np.nan)

    cols = {}
    for c in FEATURES:
        v = np.full(n, np.nan)
        v[order] = out[c]
        cols[c] = v
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
