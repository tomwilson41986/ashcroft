"""Head-to-head with expectancy: how a jockey (a trainer) has fared against each of today's rivals, beyond the price.

From the owner's reading of 12 Nov 2026: a rider's record against another rider in the races both rode in says
little until expectancy is added. With it: over the earlier races where jockeys A and B both rode, A finished ahead
of B so many times; the Betfair SP said to expect p_A / (p_A + p_B) each time (the two prices as a match); the
difference is A's head-to-head wins above expectation against B. A positive figure against most of today's rivals
is a rider who beats the riders he is up against by more than the market allows.

Every pair of runners in every race is one meeting of two jockeys (and of two trainers; the same on both sides is
left out). For each directed pair the record over earlier days is decayed (half-life HALF days) and shrunk to zero
by K meetings; today's runner reads the mean of it over today's rivals.

    hx_jk  hx_tr          the mean head-to-head edge against today's rival jockeys (trainers), per meeting
    hx_jk_n  hx_tr_n      the mean number of earlier meetings behind it
    hx_jk_min  hx_tr_min  the worst of them: the one rival who has had the better of this rider (yard)
    hx_jk_raw             the same record without the price: finished ahead less a half (what expectancy removes)

A non-finisher is behind every finisher; two non-finishers, and a dead heat, are no result. Earlier days only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.freshness_features import _col
from model.race_shape import asof_decayed_mean, day_index, race_key

FEATURES = ["hx_jk", "hx_tr", "hx_jk_n", "hx_tr_n", "hx_jk_min", "hx_tr_min", "hx_jk_raw"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "trainer", "jockey_name", "placing_numerical", "bfsp"]
HALF = 3 * 365.0
K = 20.0
MAX_FIELD = 30


def _codes(s: pd.Series) -> np.ndarray:
    t = s.fillna("").astype(str).str.strip().str.lower()
    return np.where(t.isin(["", "nan", "none"]).to_numpy(), -1, pd.factorize(t, sort=True)[0]).astype(np.int64)


def _pairs(race: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every ordered pair (i, j), i != j, of rows in one race (fields over MAX_FIELD are left out)."""
    order = np.argsort(race, kind="stable")
    r = race[order]
    starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
    sizes = np.diff(np.r_[starts, len(r)])
    left, right = [], []
    for n in np.unique(sizes):
        if n < 2 or n > MAX_FIELD:
            continue
        idx = order[(starts[sizes == n][:, None] + np.arange(n)[None, :])]            # races of this size, by row
        a, b = np.repeat(idx[:, :, None], n, axis=2), np.repeat(idx[:, None, :], n, axis=1)
        off = ~np.eye(n, dtype=bool)
        left.append(a[:, off].ravel())
        right.append(b[:, off].ravel())
    if not left:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return np.concatenate(left), np.concatenate(right)


def build(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    day = day_index(df)
    race = pd.factorize(race_key(df), sort=True)[0].astype(np.int64)
    pos = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(float)
    bsp = pd.to_numeric(_col(df, "bfsp"), errors="coerce").to_numpy(float)
    p = np.where(np.isfinite(bsp) & (bsp > 1.0), 1.0 / np.where(bsp > 1.0, bsp, 1.0), np.nan)
    fin = np.isfinite(pos) & (pos >= 1)
    rank = np.where(fin, pos, np.inf)
    i, j = _pairs(race)
    cols = {c: np.full(n, np.nan) for c in FEATURES}
    if len(i):
        both = fin[i] | fin[j]
        ahead = np.where(both & (rank[i] != rank[j]), (rank[i] < rank[j]).astype(np.float32), np.nan).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            exp = (p[i] / (p[i] + p[j])).astype(np.float32)
        d = day[i]
        cnt = np.bincount(i, minlength=n).astype(float)
        for tag, code in (("jk", _codes(_col(df, "jockey_name"))), ("tr", _codes(_col(df, "trainer")))):
            a, b = code[i], code[j]
            ok = (a >= 0) & (b >= 0) & (a != b)
            key = np.where(ok, pd.factorize(a * (int(code.max()) + 2) + b, sort=True)[0], -1).astype(np.int64)
            m, k = asof_decayed_mean(key, d, (ahead - exp).astype(float), key, d, halflife_days=HALF)
            edge = np.where(ok, np.nan_to_num(m) * np.nan_to_num(k) / (np.nan_to_num(k) + K), np.nan)
            good = np.isfinite(edge)
            c = np.bincount(i[good], minlength=n).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                cols[f"hx_{tag}"] = np.where(c > 0, np.bincount(i[good], weights=edge[good], minlength=n) / c, np.nan)
                cols[f"hx_{tag}_n"] = np.where(c > 0, np.bincount(i[good], weights=np.nan_to_num(k)[good], minlength=n) / c, np.nan)
            worst = np.full(n, np.inf)
            np.minimum.at(worst, i[good], edge[good])
            cols[f"hx_{tag}_min"] = np.where(np.isfinite(worst), worst, np.nan)
            if tag == "jk":
                m0, k0 = asof_decayed_mean(key, d, (ahead - 0.5).astype(float), key, d, halflife_days=HALF)
                raw = np.where(ok, np.nan_to_num(m0) * np.nan_to_num(k0) / (np.nan_to_num(k0) + K), np.nan)
                with np.errstate(invalid="ignore", divide="ignore"):
                    cols["hx_jk_raw"] = np.where(c > 0, np.bincount(i[good], weights=np.nan_to_num(raw)[good], minlength=n) / c, np.nan)
        del cnt
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
