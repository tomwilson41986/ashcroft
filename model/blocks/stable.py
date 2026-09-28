"""A trainer's other runners today: how many, where this one stands among them, and which one gets the yard's jockey.

When a yard runs two in a race, the market reads the yard's own view of them:
the stable's first-choice jockey rides the one the yard prefers, and the
higher-rated one is usually the better fancied. A yard sending several to one
meeting has picked the day. Nothing in the engine sets a runner against its
stablemates; each horse is read on its own. All of it is on the 06:00 card
(declarations, bookings, ratings), and the jockey's standing with the yard is
read from earlier days only:

    st_n_race     the trainer's runners in this race (1: the only one)
    st_n_meeting  the trainer's runners at this meeting today
    st_n_day      the trainer's runners anywhere today
    st_or_rank    rank by official rating among the trainer's runners in the
                  race (1 = highest); NaN for a lone runner or no rating
    st_or_gap     official rating less the best of the trainer's runners in the
                  race (0 for the best); NaN as st_or_rank
    st_jk_share   the share of the trainer's rides on earlier days (decayed,
                  half-life 180 days) that today's jockey took
    st_jk_first   with two or more of the trainer's runners in the race: 1 when
                  this one's jockey has the largest share among them, else 0;
                  NaN when none of their jockeys has ridden for the yard
    st_jk_gap     this one's jockey share less the largest among the trainer's
                  runners in the race (0 for the first choice); NaN as st_jk_first

An official rating of 0 is the table's "unrated" and counts as missing. No
result is read, so a day's own results cannot move anything here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.race_shape import _codes, asof_decayed_mean, day_index, race_key

FEATURES = ["st_n_race", "st_n_meeting", "st_n_day", "st_or_rank", "st_or_gap", "st_jk_share", "st_jk_first",
            "st_jk_gap"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "trainer", "jockey_name", "official_rating"]
READS_RESULTS = False
HALFLIFE_DAYS = 180.0


def _key(s: pd.Series) -> np.ndarray:
    s = s.fillna("").astype(str).str.strip().str.lower()
    return np.where(s.isin(["", "nan", "none"]).to_numpy(), -1, _codes(s))


def build(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    day = day_index(df)
    tr = _key(df["trainer"]) if "trainer" in df.columns else np.full(n, -1)
    jk = _key(df["jockey_name"]) if "jockey_name" in df.columns else np.full(n, -1)
    race = _codes(race_key(df))
    track = _codes(df["track"].astype(str).str.strip().str.lower()) if "track" in df.columns else np.zeros(n, int)
    has_tr = (tr >= 0) & (day >= 0)

    def count(*keys):
        codes = _codes(*keys)
        c = pd.Series(1.0, index=np.arange(n)).groupby(codes).transform("sum").to_numpy()
        return np.where(has_tr, c, np.nan)

    n_race = count(race, tr)
    n_meet = count(day, track, tr)
    n_day = count(day, tr)

    rating = pd.to_numeric(df["official_rating"], errors="coerce") if "official_rating" in df.columns \
        else pd.Series(np.nan, index=df.index)
    rating = rating.where(rating > 0).to_numpy(dtype=float)
    grp = _codes(race, tr)
    rs = pd.Series(rating)
    rank = rs.groupby(grp).rank(ascending=False, method="min").to_numpy()
    best = rs.groupby(grp).transform("max").to_numpy()
    several = has_tr & (n_race >= 2)
    or_ok = several & np.isfinite(rating)

    # the jockey's standing with the yard: decayed rides by the pair over the yard's, earlier days only
    pair = np.where(has_tr & (jk >= 0), _codes(tr, jk), -1)
    one = np.ones(n)
    _, c_pair = asof_decayed_mean(pair, day, one, pair, day, halflife_days=HALFLIFE_DAYS)
    _, c_tr = asof_decayed_mean(np.where(has_tr, tr, -1), day, one, np.where(has_tr, tr, -1), day,
                                halflife_days=HALFLIFE_DAYS)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(has_tr & (jk >= 0) & (c_tr > 0), c_pair / np.where(c_tr > 0, c_tr, 1.0), np.nan)
    top = pd.Series(share).groupby(grp).transform("max").to_numpy()
    # a yard none of whose jockeys here has ridden for it before says nothing about its choice
    sh_ok = several & np.isfinite(share) & np.isfinite(top) & (top > 0)

    cols = {
        "st_n_race": n_race,
        "st_n_meeting": n_meet,
        "st_n_day": n_day,
        "st_or_rank": np.where(or_ok, rank, np.nan),
        "st_or_gap": np.where(or_ok, rating - best, np.nan),
        "st_jk_share": share,
        "st_jk_first": np.where(sh_ok, (share >= top).astype(float), np.nan),
        "st_jk_gap": np.where(sh_ok, share - top, np.nan),
    }
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
