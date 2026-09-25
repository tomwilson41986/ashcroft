"""Freshness: how long since the horse ran, against what is usual for it, its yard and its rivals.

The served model already has the days since the last run (`days_since_lr_num`, HRB's
figure, 16th of 501 features by gain) and form discounted by it. What it lacks is the
context that makes a gap mean something:

    fr_usual_gap         the horse's usual spacing: recency-weighted mean of log(1 + gap)
                         before its earlier runs
    fr_gap_vs_usual      today's log gap less the usual one: + = a longer break than usual,
                         - = back quicker than usual
    fr_gap_vs_trainer    today's log gap less the yard's usual gap on earlier days
    fr_gap_rel_race      today's log gap less the field's mean: fresher or busier than rivals
    fr_runs_since_break  0 = first run back from a break of BREAK_DAYS or more (or the
                         debut), 1 = second run back, ... capped at RUNS_CAP; unknown when
                         the horse's history held here starts inside a sequence
    fr_break_len         log(1 + days) of that break (unknown after a debut)
    fr_runs_30d/60d/90d  workload: the horse's runs on earlier days in the last 30/60/90 days
    fr_quick_after_win   back within QUICK_DAYS of a win (in a handicap, likely under a penalty)
    fr_days_since_win    days since its last win held here; fr_runs_since_win, runs since
    fr_days_since_placed days since its last top-three finish held here
    fr_days_since_code   days since its last run in today's code (flat, all-weather,
                         hurdle, chase, bumper)

Today's gap is HRB's `days_since_lr`, the figure on the 06:00 card, where it is given.
It counts runs the database here may not hold (the history has missing months), so it
is the better measure; where it is missing, the gap is taken from the dates held.

Every feature reads the horse's runs on EARLIER DAYS and today's card only. Nothing
of today's results -- nor of today's other races -- enters, so a 06:00 card and the
same day with its results in get identical values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.race_shape import _codes, asof_decayed_mean, day_index, horse_decayed_prior, race_code, race_key

#: A gap of this many days or more is a break: the next run is the first back.
BREAK_DAYS = 60

#: A run within this many days of a win is a quick return.
QUICK_DAYS = 14

#: Half-life, in the horse's runs, of a past gap in its usual spacing.
USUAL_HALFLIFE_RUNS = 4.0

#: Half-life in days of a past runner in a yard's usual spacing.
TRAINER_HALFLIFE_DAYS = 365.0

#: fr_runs_since_break is capped here: the tenth run back is not different from the eighth.
RUNS_CAP = 8

WORKLOAD_DAYS = (30, 60, 90)

FRESHNESS_FEATURES = [
    "fr_usual_gap", "fr_gap_vs_usual", "fr_gap_vs_trainer", "fr_gap_rel_race",
    "fr_runs_since_break", "fr_break_len",
    *[f"fr_runs_{w}d" for w in WORKLOAD_DAYS],
    "fr_quick_after_win", "fr_days_since_win", "fr_runs_since_win", "fr_days_since_placed",
    "fr_days_since_code",
]

_DAY_BITS = 20


class _Days:
    """A key's racing days in order: one block per (key, day), whatever the rows.

    `inv` maps each row to its block; `pos` is the block's place among its key's
    days (0 = the first held); `prev` the key's previous day's block, -1 if none.
    Statistics of a block come only from blocks before it, so two rows of one horse
    on one day see the same history and never each other."""

    def __init__(self, key: np.ndarray, day: np.ndarray):
        ok = (key >= 0) & (day >= 0)
        base = int(day[ok].min()) if ok.any() else 0
        code = np.where(ok, (key.astype(np.int64) << _DAY_BITS) | np.clip(day - base, 0, None), -1)
        self.u, self.inv = np.unique(code, return_inverse=True)
        self.valid = self.u >= 0
        self.key = np.where(self.valid, self.u >> _DAY_BITS, -1)
        self.day = np.where(self.valid, (self.u & ((1 << _DAY_BITS) - 1)) + base, -1)
        new = np.r_[True, self.key[1:] != self.key[:-1]]
        self.start = np.maximum.accumulate(np.where(new, np.arange(len(self.u)), 0))
        self.pos = np.arange(len(self.u)) - self.start
        self.prev = np.where(self.valid & (self.pos > 0), np.arange(len(self.u)) - 1, -1)
        self.row_ok = ok

    def per_block(self, values, how: str = "first") -> np.ndarray:
        """One value per block from its rows (first non-missing, or any/max)."""
        s = pd.Series(np.asarray(values, dtype=float)).groupby(self.inv)
        out = (s.first() if how == "first" else s.max()).reindex(range(len(self.u)))
        return out.to_numpy(dtype=float)

    def to_rows(self, block_values: np.ndarray) -> np.ndarray:
        return np.where(self.row_ok, np.asarray(block_values, dtype=float)[self.inv], np.nan)

    def last_before(self, flag: np.ndarray) -> np.ndarray:
        """Index of the key's last block BEFORE each block where `flag` holds (-1 if none)."""
        idx = np.where(flag & self.valid, np.arange(len(self.u)), -1)
        run = pd.Series(idx).groupby(self.key).cummax().to_numpy()
        # shift by one block within the key: the block itself does not count
        before = np.r_[-1, run[:-1]]
        return np.where((self.pos > 0) & (before >= self.start), before, -1)

    def count_since(self, lookback: int) -> np.ndarray:
        """How many of the key's blocks before each block lie within `lookback` days."""
        lo = np.searchsorted(self.u, np.where(self.valid, self.u - lookback, -1), side="left")
        lo = np.maximum(lo, self.start)
        return np.where(self.valid, np.arange(len(self.u)) - lo, np.nan)


def _col(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    return df[name] if name in df.columns else pd.Series(default, index=df.index)


def add_freshness_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The freshness block on a frame of history plus (optionally) today's card.

    Needs race_date, race_time, track, horse_name; uses days_since_lr, trainer,
    placing_numerical, career_runs, race_type and surface_type where present. Card
    rows need no result: they read the history before their day and add nothing."""
    day = day_index(df)
    name = df["horse_name"].fillna("").astype(str).str.strip().str.lower()
    horse = np.where(name.isin(["", "nan", "none"]).to_numpy(), -1, _codes(name))
    H = _Days(horse, day)

    # --- today's gap ---------------------------------------------------------------------
    own = np.where(H.prev >= 0, H.day - H.day[np.clip(H.prev, 0, None)], np.nan)
    hrb = pd.to_numeric(_col(df, "days_since_lr"), errors="coerce").to_numpy(dtype=float)
    hrb = H.per_block(np.where(hrb > 0, hrb, np.nan))
    gap = np.where(np.isfinite(hrb), hrb, own)                         # per block
    lgap = np.log1p(gap)

    # --- against the horse's usual spacing -------------------------------------------------
    sums, cnt, runs = horse_decayed_prior(horse, day, {"lg": H.to_rows(lgap)}, USUAL_HALFLIFE_RUNS)
    with np.errstate(invalid="ignore", divide="ignore"):
        usual = np.where((runs >= 2) & (cnt["lg"] > 0), sums["lg"] / cnt["lg"], np.nan)
    lgap_rows = H.to_rows(lgap)
    usual = np.where(H.row_ok, usual, np.nan)
    df["fr_usual_gap"] = usual
    df["fr_gap_vs_usual"] = lgap_rows - usual

    # --- against the yard's usual spacing, on earlier days --------------------------------
    tr = _col(df, "trainer", "").fillna("").astype(str).str.strip().str.lower()
    tkey = np.where(tr.to_numpy() != "", _codes(tr), -1)
    tmean, _ = asof_decayed_mean(tkey, day, lgap_rows, tkey, day, TRAINER_HALFLIFE_DAYS)
    df["fr_gap_vs_trainer"] = lgap_rows - tmean

    # --- against the field (every runner's gap is on the card) -----------------------------
    g = pd.Series(lgap_rows, index=df.index)
    df["fr_gap_rel_race"] = g - g.groupby(race_key(df)).transform("mean")

    # --- runs since a break ---------------------------------------------------------------
    cr = H.per_block(pd.to_numeric(_col(df, "career_runs"), errors="coerce").to_numpy(dtype=float))
    debut = np.where(np.isfinite(cr), cr <= 0, (H.pos == 0) & ~np.isfinite(hrb))
    starts = H.valid & (debut | (gap >= BREAK_DAYS))
    idx = np.where(starts, np.arange(len(H.u)), -1)
    last_start = pd.Series(idx).groupby(H.key).cummax().to_numpy()   # this block counts
    known = (last_start >= 0) & (last_start >= H.start)
    since = np.where(known, np.minimum(np.arange(len(H.u)) - last_start, RUNS_CAP), np.nan)
    blen = np.where(known, gap[np.clip(last_start, 0, None)], np.nan)
    df["fr_runs_since_break"] = H.to_rows(since)
    df["fr_break_len"] = H.to_rows(np.log1p(blen))

    # --- workload -------------------------------------------------------------------------
    for w in WORKLOAD_DAYS:
        df[f"fr_runs_{w}d"] = H.to_rows(H.count_since(w))

    # --- since the last win, the last good run -----------------------------------------------
    place = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(dtype=float)
    won = H.per_block(np.where(np.isfinite(place), (place == 1).astype(float), np.nan), how="max") == 1
    placed = H.per_block(np.where(np.isfinite(place), (place <= 3).astype(float), np.nan), how="max") == 1
    lw = H.last_before(won)
    lp = H.last_before(placed)
    blocks = np.arange(len(H.u))
    df["fr_days_since_win"] = H.to_rows(np.where(lw >= 0, H.day - H.day[np.clip(lw, 0, None)], np.nan))
    df["fr_runs_since_win"] = H.to_rows(np.where(lw >= 0, blocks - lw, np.nan))
    df["fr_days_since_placed"] = H.to_rows(np.where(lp >= 0, H.day - H.day[np.clip(lp, 0, None)], np.nan))
    # the gap to the run held, not HRB's: when HRB's is shorter, the run held is not the last one
    prev_won = np.where(H.prev >= 0, won[np.clip(H.prev, 0, None)], False)
    df["fr_quick_after_win"] = H.to_rows(np.where(H.prev >= 0, (prev_won & (own <= QUICK_DAYS)).astype(float),
                                                  np.nan))

    # --- since the last run in today's code ------------------------------------------------
    code = race_code(df).to_numpy()
    hc = np.where(H.row_ok, _codes(horse, code), -1)
    C = _Days(hc, day)
    df["fr_days_since_code"] = C.to_rows(np.where(C.prev >= 0, C.day - C.day[np.clip(C.prev, 0, None)], np.nan))
    return df, list(FRESHNESS_FEATURES)
