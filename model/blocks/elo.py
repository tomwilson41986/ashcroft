"""A rating from finishing orders: every horse's strength inferred from whom it beat and who beat it, across the whole race network.

Collateral form (head_to_head) reads direct meetings with today's rivals and
helped (iteration 41: -0.0020). Most rivals have never met; a rating carries
the comparison through everyone they have met. Official ratings do that for
handicappers, but maidens, novices and bumper horses have none, and there the
forecast errs most (iteration 42's candidate: 0.50-0.57 mean absolute log
error against 0.34-0.37 in handicaps). A multi-runner Elo:

    before a race, each finisher's expected share of the field beaten is
        E_i = mean over rivals j of 1 / (1 + 10 ** ((R_j - R_i) / 400))
    its actual share S_i = (finishers - its place) / (finishers - 1), and
        R_i += K * (S_i - E_i)

Ratings live in two pools, the Flat (turf and all-weather) and jumps, start
at 1500, move only on finishers (a fall says nothing about ability) and are
read as they stood at the start of the day, so a day's own results never
reach its features:

    elo          the rating (NaN until the horse has finished in the pool)
    elo_runs     finishes behind it (against at least one other finisher)
    elo_z        (rating - the race's mean) / its standard deviation, over rated runners
    elo_gap      rating - the best rating in the race
    elo_trend    rating now - rating three finishes ago
    elo_field    the race's mean rating (field strength)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks.head_to_head import pairs
from model.race_shape import _codes, day_index, race_key

FEATURES = ["elo", "elo_runs", "elo_z", "elo_gap", "elo_trend", "elo_field"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "placing_numerical", "race_code"]
K = 24.0
START = 1500.0
TREND_LAG = 3


def _pool(df: pd.DataFrame) -> np.ndarray:
    """0 for the Flat (turf and all-weather), 1 for jumps."""
    if "race_code" not in df.columns:
        return np.zeros(len(df), dtype=np.int64)
    code = df["race_code"].fillna("").astype(str).str.lower()
    return np.where(code.str.contains("hunt|hurdle|chase|jump").to_numpy(), 1, 0).astype(np.int64)


def build(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    name = df["horse_name"].fillna("").astype(str).str.strip().str.lower()
    day = day_index(df)
    pool = _pool(df)
    horse = np.where(name.isin(["", "nan", "none"]).to_numpy() | (day < 0), -1, _codes(name))
    key = np.where(horse >= 0, horse * 2 + pool, -1)               # one rating per horse and pool
    race = _codes(race_key(df))
    pos = pd.to_numeric(df["placing_numerical"], errors="coerce").to_numpy(dtype=float)

    nk = int(key.max()) + 1 if (key >= 0).any() else 0
    rating = np.full(nk, np.nan)
    runs = np.zeros(nk)
    hist = np.full((nk, TREND_LAG), np.nan)                          # ratings before the last three finishes

    out = {c: np.full(n, np.nan) for c in FEATURES}
    order = np.argsort(day, kind="stable")
    days = day[order]
    starts = np.flatnonzero(np.r_[True, days[1:] != days[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for s, e in zip(starts, ends):
        rows = order[s:e]
        if days[s] < 0:
            continue
        rows = rows[key[rows] >= 0]
        if not len(rows):
            continue
        k = key[rows]
        r_now = rating[k]
        out["elo"][rows] = r_now
        out["elo_runs"][rows] = runs[k]
        out["elo_trend"][rows] = r_now - hist[k, 0]

        # the field as rated at the start of the day
        rc = race[rows]
        rs = pd.Series(r_now)
        g = rs.groupby(rc)
        mean, sd, top = g.transform("mean").to_numpy(), g.transform("std").to_numpy(), g.transform("max").to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            out["elo_z"][rows] = np.where(sd > 0, (r_now - mean) / sd, np.where(np.isfinite(r_now) & (sd == 0), 0.0,
                                                                                np.nan))
        out["elo_gap"][rows] = r_now - top
        out["elo_field"][rows] = mean

        # the day's results update the ratings, all from the day's opening values
        fin = np.isfinite(pos[rows])
        if fin.sum() < 2:
            continue
        fr, fk, frc = rows[fin], k[fin], rc[fin]
        before = np.where(np.isfinite(rating[fk]), rating[fk], START)
        fpos = pos[fr]
        a, b = pairs(frc)                                            # every ordered pair of finishers in a race
        p_beat = 1.0 / (1.0 + 10.0 ** ((before[b] - before[a]) / 400.0))
        rivals = np.bincount(a, minlength=len(fr)).astype(float)
        expected = np.bincount(a, weights=p_beat, minlength=len(fr))
        # a dead heat shares the credit: rivals strictly behind, plus half those level
        actual = (np.bincount(a, weights=(fpos[b] > fpos[a]).astype(float), minlength=len(fr))
                  + 0.5 * np.bincount(a, weights=(fpos[b] == fpos[a]).astype(float), minlength=len(fr)))
        with np.errstate(invalid="ignore", divide="ignore"):
            delta = np.where(rivals > 0, K * (actual - expected) / np.where(rivals > 0, rivals, 1.0), 0.0)
        # a horse finishes once a day in a pool; were the feed to repeat one, its moves add up.
        # A lone finisher beat nobody and lost to nobody: no move, and not a rated run.
        has = rivals > 0
        fk, before, delta = fk[has], before[has], delta[has]
        if not len(fk):
            continue
        upd = pd.DataFrame({"k": fk, "before": before, "delta": delta}).groupby("k", sort=True)
        uk = upd.size().index.to_numpy()
        rating[uk] = upd["before"].first().to_numpy() + upd["delta"].sum().to_numpy()
        runs[uk] += upd.size().to_numpy()
        hist[uk] = np.column_stack([hist[uk, 1:], upd["before"].first().to_numpy()])

    new = pd.DataFrame(out, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
