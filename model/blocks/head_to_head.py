"""Collateral form: how each runner has fared against today's rivals when they met before.

Every other block reads a horse's form on its own or against its field's
averages. None reads the form lines between today's actual runners: a horse
that has finished ahead of four of today's rivals in earlier races, by a few
pounds each time, is being told something about today that its ratings only
approximate. Handicappers read these lines by hand; the model has never seen
them.

For every pair of today's runners, their earlier meetings in which both
finished (EARLIER DAYS only), and from those, for each runner:

    h2h_rivals_met   how many of today's rivals it has met before
    h2h_meetings     how many such meetings, over all of today's rivals
    h2h_win_share    the share of them in which it finished ahead, shrunk to one
                     half by one meeting each way ((ahead + 1) / (meetings + 2))
    h2h_net          meetings finished ahead less meetings finished behind
    h2h_lbs          its average margin over the rivals in pounds (positive:
                     ahead), shrunk toward 0 by two meetings
    h2h_last_lbs     the margin of each pair's latest meeting only, averaged over
                     the rivals met
    h2h_last_net     rivals it finished ahead of at their latest meeting less
                     rivals that finished ahead of it

Margins are the engine's lengths beaten turned into pounds at the race's trip,
capped at 20 lb a runner (model/form_windows.run_measures), so the margin
between two runners in one race is at most 20 lb.

A run's own result never enters its own features: each pair's meetings are
summed over the days before the row's day, so a 06:00 card and the same rows
with results in get the same values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.form_windows import run_measures
from model.race_shape import day_index, race_key

FEATURES = ["h2h_rivals_met", "h2h_meetings", "h2h_win_share", "h2h_net", "h2h_lbs", "h2h_last_lbs",
            "h2h_last_net"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "number_of_runners", "placing_numerical",
         "LB", "total_dst_bt", "dist_furlongs"]


def pairs(race: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every ordered pair (a, b), a != b, of rows in the same race, as row positions."""
    order = np.argsort(race, kind="stable")
    rc = race[order]
    starts = np.flatnonzero(np.r_[True, rc[1:] != rc[:-1]])
    sizes = np.diff(np.r_[starts, len(rc)])
    size_of = np.repeat(sizes, sizes)                 # each sorted row: its race's size
    start_of = np.repeat(starts, sizes)               # ... and its race's first sorted row
    a = np.repeat(np.arange(len(rc)), size_of)
    k = np.arange(len(a)) - np.repeat(np.cumsum(size_of) - size_of, size_of)
    b = np.repeat(start_of, size_of) + k
    ok = a != b
    return order[a[ok]], order[b[ok]]


def _sum_before(v: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Within each group (a pair's days, in order), the sum of the entries before.

    The running sum stepped back one entry, not the running sum less the entry: a
    day's own value then never enters its sum, not even as rounding."""
    g = pd.Series(v).groupby(group, sort=False)
    return g.cumsum().groupby(group, sort=False).shift(1).fillna(0.0).to_numpy(dtype=float)


def build(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    race = pd.factorize(race_key(df))[0].astype(np.int64)
    day = day_index(df)
    horse = pd.factorize(df["horse_name"].fillna("").astype(str).str.strip().str.lower())[0].astype(np.int64)
    lbs = run_measures(df)["lbs"]                     # pounds behind the winner; NaN = no finish
    a, b = pairs(race)

    # the pair's meeting on this row's day, when both finished: the margin, a's view
    margin = lbs[b] - lbs[a]                          # positive: a ahead; NaN unless both finished
    fin = np.isfinite(margin)
    met = fin.astype(float)
    ahead = (fin & (margin > 0)).astype(float)
    behind = (fin & (margin < 0)).astype(float)
    m0 = np.where(fin, margin, 0.0)

    # one entry per (pair, day), in pair then day order
    pair = horse[a] * (horse.max() + 1) + horse[b]
    grp = pd.DataFrame({"pair": pair, "day": day[a]}).groupby(["pair", "day"], sort=True)
    code = grp.ngroup().to_numpy()
    keys = grp.size().index
    gpair = keys.get_level_values(0).to_numpy()
    G = len(keys)
    first = np.r_[True, gpair[1:] != gpair[:-1]]

    def gsum(v):
        return np.bincount(code, weights=v, minlength=G)

    met_g, ahead_g, behind_g, m_g = gsum(met), gsum(ahead), gsum(behind), gsum(m0)
    # the meetings on the days before each entry's day
    p_met = _sum_before(met_g, gpair)[code]
    p_ahead = _sum_before(ahead_g, gpair)[code]
    p_behind = _sum_before(behind_g, gpair)[code]
    p_margin = _sum_before(m_g, gpair)[code]
    # the latest earlier meeting's margin: step back one day within the pair, carry forward
    day_margin = np.where(met_g > 0, m_g / np.maximum(met_g, 1.0), np.nan)
    prev = np.r_[np.nan, day_margin[:-1]]
    prev[first] = np.nan
    p_last = pd.Series(prev).groupby(gpair).ffill().to_numpy(dtype=float)[code]

    # each row, over today's rivals
    def rsum(v):
        return np.bincount(a, weights=v, minlength=n)

    meetings = rsum(p_met)
    has_last = np.isfinite(p_last)
    n_last = rsum(has_last.astype(float))
    cols = {
        "h2h_rivals_met": rsum((p_met > 0).astype(float)),
        "h2h_meetings": meetings,
        "h2h_win_share": (rsum(p_ahead) + 1.0) / (meetings + 2.0),
        "h2h_net": rsum(p_ahead) - rsum(p_behind),
        "h2h_lbs": rsum(p_margin) / (meetings + 2.0),
        "h2h_last_lbs": np.where(n_last > 0, rsum(np.where(has_last, p_last, 0.0)) / np.maximum(n_last, 1.0),
                                 np.nan),
        "h2h_last_net": rsum(np.where(has_last, np.sign(p_last), 0.0)),
    }
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
