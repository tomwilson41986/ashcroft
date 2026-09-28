"""The connections' recent form at finer grains: the last fortnight, today's race code and course, the pairing.

Connection windows (connection_windows.py) read the trainer's and jockey's runners
over a ladder of runner counts, and gave the largest gain since the form windows
(iteration 67). The engine's other readings of the connections are career
rates (trainer and jockey at the track, the trainer-jockey pair) and 14-day
strike rates. This block reads the same runners as connection windows (a
non-finisher's normalised finishing position 0; wins less BSP chances) at the
grains a race reader weighs:

    cg_<e>_d14_n        the trainer's (jockey's) runners in the 14 days before today
    cg_<e>_d14_nfp      their mean NFP, shrunk to the connection's own last-100 NFP
                        by five runners: in form or out of it now
    cg_<e>_d14_dnfp     that less the last-100 NFP
    cg_<e>_d14_ae       their wins less BSP chances, per run, shrunk to 0 by five
    cg_tr_code_n        the trainer's runners in today's race code so far (flat,
                        all-weather, hurdle, chase, bumper)
    cg_tr_code_nfp      the last 50 of them, NFP, shrunk to the last-100 NFP by ten
    cg_tr_code_dnfp     that less the last-100 NFP
    cg_<e>_trk_n        the trainer's (jockey's) runners at today's course so far
    cg_<e>_trk_nfp      the last 20 there, NFP, shrunk to the last-100 NFP by five
    cg_tj_n             the trainer and jockey together so far
    cg_tj_nfp           their last 20 together, NFP, shrunk to the trainer's
                        last-100 NFP by five
    cg_tj_ae            their last 20 together, wins less BSP chances, shrunk by five

with <e> the trainer (tr) and the jockey (jk); a connection with no runner in
its last 100 is shrunk to one half. Today's results never enter: every window
closes the day before the row's day, and every sum is a running sum within its
own key in a fixed order (by day, race, horse), so a 06:00 card and the same
rows with results in agree to the bit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks.connection_windows import NFP_PRIOR, _codes, _runner_values
from model.freshness_features import _col
from model.race_shape import day_index, race_code, race_key

FEATURES = (
    [f"cg_{e}_d14_{s}" for e in ("tr", "jk") for s in ("n", "nfp", "dnfp", "ae")]
    + ["cg_tr_code_n", "cg_tr_code_nfp", "cg_tr_code_dnfp"]
    + [f"cg_{e}_trk_{s}" for e in ("tr", "jk") for s in ("n", "nfp")]
    + ["cg_tj_n", "cg_tj_nfp", "cg_tj_ae"]
)
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "trainer", "jockey_name", "number_of_runners",
         "placing_numerical", "bfsp", "LB", "total_dst_bt", "dist_furlongs", "race_type", "surface_type"]
DAYS = 14           # the recent window, in days before today
K_RECENT = 5.0      # shrinkage of the fortnight, the course, the pairing, in runners
K_CODE = 10.0       # shrinkage of the race code's last 50, in runners
CODES = {"flat": 0, "aw": 1, "hurdle": 2, "chase": 3, "nhflat": 4}


class _Keyed:
    """One key's rows in a fixed order (day, then the tiebreak), with running sums that restart at each key."""

    def __init__(self, key: np.ndarray, day: np.ndarray, tie: np.ndarray):
        n = len(key)
        self.ok = (key >= 0) & (day >= 0)
        self.order = np.lexsort((tie, day, key))
        self.k, self.d = key[self.order], day[self.order]
        idx = np.arange(n)
        new_key = np.r_[True, self.k[1:] != self.k[:-1]]
        new_day = new_key | np.r_[True, self.d[1:] != self.d[:-1]]
        self.key_start = np.maximum.accumulate(np.where(new_key, idx, 0))
        self.day_start = np.maximum.accumulate(np.where(new_day, idx, 0))     # today's first row: windows end here
        self.n = n

    def _running(self, x: np.ndarray) -> np.ndarray:
        return pd.Series(x).groupby(self.k, sort=False).cumsum().to_numpy(dtype=float)

    def _upto(self, cs: np.ndarray, i: np.ndarray) -> np.ndarray:
        """The key's running sum over its rows before position i (0 when there are none)."""
        return np.where(i > self.key_start, cs[np.maximum(i - 1, 0)], 0.0)

    def last_runners(self, k: int | None) -> np.ndarray:
        """The first position of the window of the key's last k rows before today (all of them when k is None)."""
        return self.key_start if k is None else np.maximum(self.key_start, self.day_start - k)

    def last_days(self, days: int) -> np.ndarray:
        """The first position of the key's rows on the `days` days before today."""
        span = int(self.d.max()) + days + 2 if self.n else 1
        comp = self.k * span + self.d                                    # nondecreasing in this order
        lo = np.searchsorted(comp, self.k * span + (self.d - days), side="left")
        return np.maximum(self.key_start, lo)

    def window(self, v: np.ndarray, lo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sum and count of v's finite values over positions [lo, today's first row), back in row order."""
        v_s = v[self.order]
        f = np.isfinite(v_s)
        cx, cc = self._running(np.where(f, v_s, 0.0)), self._running(f.astype(float))
        s = self._upto(cx, self.day_start) - self._upto(cx, lo)
        c = self._upto(cc, self.day_start) - self._upto(cc, lo)
        out_s, out_c = np.empty(self.n), np.empty(self.n)
        out_s[self.order], out_c[self.order] = s, c
        return np.where(self.ok, out_s, np.nan), np.where(self.ok, out_c, np.nan)


def _pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """One code for each (a, b), -1 where either is missing."""
    return np.where((a >= 0) & (b >= 0), a * (int(b.max()) + 2) + b, -1).astype(np.int64)


def build(df: pd.DataFrame) -> pd.DataFrame:
    n_rows = len(df)
    day = day_index(df)
    race = pd.factorize(race_key(df), sort=True)[0].astype(np.int64)
    horse = _codes(_col(df, "horse_name"))
    tie = race * (int(horse.max()) + 2) + (horse + 1)          # a fixed order within a day
    vals = _runner_values(df)
    nfp, ae = vals["nfp"], vals["ae"]
    ones = np.ones(n_rows)
    tr, jk = _codes(_col(df, "trainer")), _codes(_col(df, "jockey_name"))
    trk = _codes(_col(df, "track"))
    code = race_code(df).map(CODES).fillna(-1).to_numpy(dtype=np.int64)

    cols = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        base = {}
        for e, key in (("tr", tr), ("jk", jk)):
            K = _Keyed(key, day, tie)
            s, c = K.window(nfp, K.last_runners(100))
            base[e] = np.where(c > 0, s / c, NFP_PRIOR)                   # the connection's own last-100 level
            lo = K.last_days(DAYS)
            cols[f"cg_{e}_d14_n"] = K.window(ones, lo)[1]
            s, c = K.window(nfp, lo)
            cols[f"cg_{e}_d14_nfp"] = (s + K_RECENT * base[e]) / (c + K_RECENT)
            cols[f"cg_{e}_d14_dnfp"] = cols[f"cg_{e}_d14_nfp"] - base[e]
            s, c = K.window(ae, lo)
            cols[f"cg_{e}_d14_ae"] = s / (c + K_RECENT)
            T = _Keyed(_pair(key, trk), day, tie)
            cols[f"cg_{e}_trk_n"] = T.window(ones, T.last_runners(None))[1]
            s, c = T.window(nfp, T.last_runners(20))
            cols[f"cg_{e}_trk_nfp"] = (s + K_RECENT * base[e]) / (c + K_RECENT)
        C = _Keyed(_pair(tr, code), day, tie)
        cols["cg_tr_code_n"] = C.window(ones, C.last_runners(None))[1]
        s, c = C.window(nfp, C.last_runners(50))
        cols["cg_tr_code_nfp"] = (s + K_CODE * base["tr"]) / (c + K_CODE)
        cols["cg_tr_code_dnfp"] = cols["cg_tr_code_nfp"] - base["tr"]
        P = _Keyed(_pair(tr, jk), day, tie)
        cols["cg_tj_n"] = P.window(ones, P.last_runners(None))[1]
        lo = P.last_runners(20)
        s, c = P.window(nfp, lo)
        cols["cg_tj_nfp"] = (s + K_RECENT * base["tr"]) / (c + K_RECENT)
        s, c = P.window(ae, lo)
        cols["cg_tj_ae"] = s / (c + K_RECENT)
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    assert len(new) == n_rows
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
