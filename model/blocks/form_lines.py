"""How the form of a horse's recent races has worked out: what the rivals it met there have done since.

Race readers check whether a run's form "has worked out": whether the horses
that finished around it have won or run well since. A fourth behind three
subsequent winners is worth more than a fourth in a race nobody has come out
of. The model reads a horse's own results and its direct meetings with today's
rivals (collateral form, head_to_head), not how its races have been franked by
everyone else in them.

For the horse's last race, and its last three together: its rivals' next runs
after that race (each rival's next three at most) that came BEFORE today, and

    fl_<w>_n      how many such runs
    fl_<w>_wins   how many of them were won
    fl_<w>_wr     the rivals' win rate since, shrunk to 1 in 10 by five runs
    fl_<w>_plc    their place rate (first three), shrunk to 3 in 10 by five runs
    fl_<w>_ae     their wins less the market's chances at BSP, per run, shrunk to 0
                  by five runs: whether the race's form has run above its prices

w = l1 (the last race) or l3 (the last three; the horse's own later runs are
left out, and a rival met twice counts twice). NaN before a horse's first run.
A non-finisher counts as a run that was not won or placed.

Today's results never enter: only runs on days before the row's day are
counted, each race's runs summed in date order from zero, so a 06:00 card and
the same rows with results in get the same values, to the bit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.freshness_features import _Days, _col
from model.race_shape import day_index, race_key

WINDOWS = ("l1", "l3")
STATS = ("n", "wins", "wr", "plc", "ae")
FEATURES = [f"fl_{w}_{s}" for w in WINDOWS for s in STATS]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "placing_numerical", "bfsp"]
NEXT = 3            # each rival's next runs after the race that count
LAST = 3            # the horse's races pooled in l3
K = 5.0             # shrinkage, in runs
P_WIN, P_PLC = 0.1, 0.3


def _values(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each run's win, place and win less the market's chance (post-race)."""
    pos = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(dtype=float)
    won = (pos == 1).astype(float)
    plc = ((pos >= 1) & (pos <= 3)).astype(float)
    race = pd.factorize(race_key(df), sort=True)[0]
    bsp = pd.to_numeric(_col(df, "bfsp"), errors="coerce").to_numpy(dtype=float)
    p = pd.Series(np.where(bsp > 1, 1.0 / bsp, np.nan))
    pn = (p / p.groupby(race).transform("sum")).to_numpy(dtype=float)
    return {"n": np.ones(len(df)), "wins": won, "plc": plc, "ae": np.where(np.isfinite(pn), won - pn, 0.0)}


def build(df: pd.DataFrame) -> pd.DataFrame:
    n_rows = len(df)
    day = day_index(df)
    name = df["horse_name"].fillna("").astype(str).str.strip().str.lower()
    bad = name.isin(["", "nan", "none"]).to_numpy() | (day < 0)
    horse = np.where(bad, -1, pd.factorize(name, sort=True)[0]).astype(np.int64)   # codes independent of row order
    race_rows = pd.factorize(race_key(df), sort=True)[0].astype(np.int64)
    vals = _values(df)

    H = _Days(horse, day)
    nb = len(H.u)
    b = np.arange(nb)
    # a horse twice on a day (rare): its later-coded race, and either run's result, whatever the row order
    race = np.nan_to_num(H.per_block(race_rows, how="max"), nan=-1).astype(np.int64)
    bv = {s: H.per_block(v, how="max") for s, v in vals.items()}

    # events: for every horse-day block, the horse's next NEXT days, filed under the block's race
    ev_r, ev_d, ev_h, ev_v = [], [], [], {s: [] for s in bv}
    for k in range(1, NEXT + 1):
        j = b + k
        ok = H.valid & (j < nb)
        ok[ok] = H.key[j[ok]] == H.key[b[ok]]
        ev_r.append(race[b[ok]])
        ev_d.append(H.day[j[ok]])
        ev_h.append(H.key[j[ok]])
        for s in bv:
            ev_v[s].append(bv[s][j[ok]])
    ev_r, ev_d, ev_h = np.concatenate(ev_r), np.concatenate(ev_d), np.concatenate(ev_h)
    order = np.lexsort((ev_h, ev_d, ev_r))                  # race, then day, then horse: no row-order effect
    ev_r, ev_d = ev_r[order], ev_d[order]
    ev_v = {s: np.nan_to_num(np.concatenate(v)[order]) for s, v in ev_v.items()}
    dmin = int(H.day[H.valid].min()) if H.valid.any() else 0
    span = int(H.day[H.valid].max()) - dmin + 2 if H.valid.any() else 2
    code = ev_r * span + (ev_d - dmin)
    csum = {s: pd.Series(v).groupby(ev_r, sort=False).cumsum().to_numpy() for s, v in ev_v.items()}

    def before(r: np.ndarray, t: np.ndarray, ok: np.ndarray) -> dict[str, np.ndarray]:
        """Per block: each sum over race r's events on days before t (0 where none)."""
        q = np.where(ok, r * span + (t - dmin), -1)
        hi = np.searchsorted(code, q, side="left")
        lo = np.searchsorted(code, np.where(ok, r * span, -1), side="left")
        has = ok & (hi > lo)
        last = np.clip(hi - 1, 0, None)
        return {s: np.where(has, c[last], 0.0) if len(c) else np.zeros(len(r)) for s, c in csum.items()}

    out = {}
    tot = {s: np.zeros(nb) for s in bv}
    for k in range(1, LAST + 1):
        ok = H.valid & (H.pos >= k)
        rk = np.where(ok, race[np.clip(b - k, 0, None)], -1)
        got = before(rk, H.day, ok)
        for s in bv:
            own = np.zeros(nb)
            for i in range(1, k):                            # its own runs since that race, all before today
                if i <= NEXT:
                    own += np.where(ok, np.nan_to_num(bv[s][np.clip(b - i, 0, None)]), 0.0)
            tot[s] += np.where(ok, got[s] - own, 0.0)
        if k == 1:
            out["l1"] = ({s: v.copy() for s, v in got.items()}, H.valid & (H.pos >= 1))
    out["l3"] = (tot, H.valid & (H.pos >= 1))

    cols = {}
    for w, (sums, has) in out.items():
        n = sums["n"]
        with np.errstate(invalid="ignore", divide="ignore"):
            stats = {"n": n, "wins": sums["wins"], "wr": (sums["wins"] + K * P_WIN) / (n + K),
                     "plc": (sums["plc"] + K * P_PLC) / (n + K), "ae": sums["ae"] / (n + K)}
        for s in STATS:
            cols[f"fl_{w}_{s}"] = H.to_rows(np.where(has, stats[s], np.nan))
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    assert len(new) == n_rows
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
