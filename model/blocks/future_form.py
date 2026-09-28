"""Future form: how the rivals from a horse's recent races ran next, over every window back and forward.

A race reader asks of each run whether its form has worked out: what the horses
it met there did next. form_lines reads that for the last race and the last
three, through the rivals' next three runs, in wins and places. This block reads
the whole grid the owner specified, in the measures the price model rates
highest:

    back     l1, l3, l5   the horse's last race, last three, last five (pooled;
                          a rival met twice counts twice; the horse's own later
                          runs are left out)
    forward  n1, n2, n3, n5   each rival's next 1, 2, 3, 5 runs after that race,
                              on days before today

and for each pair (b, f):

    ff_<b>_<f>_n     the rivals' runs counted
    ff_<b>_<f>_wr    their win rate, shrunk to 1 in 10 by five runs
    ff_<b>_<f>_ae    their wins less the market's chances at BSP, per run, shrunk
                     to 0 by five runs: whether the race's form has run above its
                     prices
    ff_<b>_<f>_nfp   their normalised finishing position (1 won, 0 last), shrunk
                     to 0.5 by five runs
    ff_<b>_<f>_perf  their performance figures (official-rating scale, lb) less
                     theirs in the race, per run, shrunk to 0 by five runs: how
                     much better (+) the race's runners have run since

and for the last race, the horse's own figure there revised by how it has worked out:

    ff_l1_<f>_adj    its performance figure in the race + ff_l1_<f>_perf

A non-finisher counts as a run not won; it has no finishing position or figure,
so those two count only the runs that have one. NaN before a horse's first run.

Today's results never enter: only runs on days before the row's day are counted,
each race's runs summed in a fixed order (race, then day, then horse) from zero,
so a 06:00 card and the same rows with results in get the same values, to the bit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.freshness_features import _Days, _col
from model.race_shape import day_index, race_key

BACK = (1, 3, 5)
FORWARD = (1, 2, 3, 5)
STATS = ("n", "wr", "ae", "nfp", "perf")
FEATURES = ([f"ff_l{b}_n{f}_{s}" for b in BACK for f in FORWARD for s in STATS]
            + [f"ff_l1_n{f}_adj" for f in FORWARD])
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "number_of_runners", "placing_numerical",
         "bfsp", "official_rating", "LB", "total_dst_bt", "dist_furlongs"]
K = 5.0             # shrinkage, in runs
P_WIN, NFP0 = 0.1, 0.5
#: the per-event sums: (value, count) pairs
PARTS = ("n", "wins", "ae", "nfp_v", "nfp_n", "perf_v", "perf_n")


def _values(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each run's readings from its own race (post-race): won (a non-finisher 0), the market's
    chance at BSP normalised in the race, normalised finishing position and performance figure
    (both NaN for a non-finisher)."""
    from model.form_windows import run_measures

    m = run_measures(df)
    pos = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(dtype=float)
    won = (pos == 1).astype(float)
    race = pd.factorize(race_key(df), sort=True)[0]
    bsp = pd.to_numeric(_col(df, "bfsp"), errors="coerce").to_numpy(dtype=float)
    p = pd.Series(np.where(bsp > 1, 1.0 / bsp, np.nan))
    pn = (p / p.groupby(race).transform("sum")).to_numpy(dtype=float)
    return {"won": won, "pn": pn, "nfp": m["nfp"], "perf": m["perf"]}


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
    # a horse twice on a day (rare): its later-coded race, and the larger of either run's readings
    race = np.nan_to_num(H.per_block(race_rows, how="max"), nan=-1).astype(np.int64)
    bv = {s: H.per_block(v, how="max") for s, v in vals.items()}

    def event(later: np.ndarray, then: np.ndarray) -> dict[str, np.ndarray]:
        """One run `later` after the race of block `then`: its parts, zero where unknown."""
        won = np.nan_to_num(bv["won"][later])
        pn = bv["pn"][later]
        nfp = bv["nfp"][later]
        dperf = bv["perf"][later] - bv["perf"][then]
        return {"n": np.ones(len(later)), "wins": won, "ae": np.where(np.isfinite(pn), won - pn, 0.0),
                "nfp_v": np.nan_to_num(nfp), "nfp_n": np.isfinite(nfp).astype(float),
                "perf_v": np.nan_to_num(dperf), "perf_n": np.isfinite(dperf).astype(float)}

    # events: for every horse-day block, the horse's next max(FORWARD) days, filed under the block's race,
    # each with its rank k (the k-th run after the race)
    ev_r, ev_d, ev_h, ev_k = [], [], [], []
    ev_v = {p: [] for p in PARTS}
    for k in range(1, max(FORWARD) + 1):
        j = b + k
        ok = H.valid & (j < nb)
        ok[ok] = H.key[j[ok]] == H.key[b[ok]]
        ev_r.append(race[b[ok]])
        ev_d.append(H.day[j[ok]])
        ev_h.append(H.key[j[ok]])
        ev_k.append(np.full(int(ok.sum()), k))
        e = event(j[ok], b[ok])
        for p in PARTS:
            ev_v[p].append(e[p])
    ev_r, ev_d, ev_h, ev_k = (np.concatenate(x) for x in (ev_r, ev_d, ev_h, ev_k))
    order = np.lexsort((ev_k, ev_h, ev_d, ev_r))            # race, day, horse, rank: no row-order effect
    ev_r, ev_d, ev_k = ev_r[order], ev_d[order], ev_k[order]
    ev_v = {p: np.concatenate(v)[order] for p, v in ev_v.items()}
    dmin = int(H.day[H.valid].min()) if H.valid.any() else 0
    span = int(H.day[H.valid].max()) - dmin + 2 if H.valid.any() else 2
    code = ev_r * span + (ev_d - dmin)

    def sums_before(csum: dict, r: np.ndarray, t: np.ndarray, ok: np.ndarray) -> dict[str, np.ndarray]:
        """Per block: each cumulative sum over race r's events on days before t (0 where none)."""
        q = np.where(ok, r * span + (t - dmin), -1)
        hi = np.searchsorted(code, q, side="left")
        lo = np.searchsorted(code, np.where(ok, r * span, -1), side="left")
        has = ok & (hi > lo)
        last = np.clip(hi - 1, 0, None)
        return {p: np.where(has, c[last], 0.0) if len(c) else np.zeros(len(r)) for p, c in csum.items()}

    has_run = H.valid & (H.pos >= 1)
    cols = {}
    for f in FORWARD:
        inside = (ev_k <= f).astype(float)
        csum = {p: pd.Series(ev_v[p] * inside).groupby(ev_r, sort=False).cumsum().to_numpy() for p in PARTS}
        tot = {p: np.zeros(nb) for p in PARTS}
        for k in range(1, max(BACK) + 1):
            ok = H.valid & (H.pos >= k)
            then = np.clip(b - k, 0, None)
            got = sums_before(csum, np.where(ok, race[then], -1), H.day, ok)
            own = {p: np.zeros(nb) for p in PARTS}
            for i in range(1, k):                            # its own runs since that race, all before today
                if k - i <= f:
                    e = event(np.clip(b - i, 0, None), then)
                    for p in PARTS:
                        own[p] += np.where(ok, e[p], 0.0)
            for p in PARTS:
                tot[p] += np.where(ok, got[p] - own[p], 0.0)
            if k in BACK:
                s = tot
                with np.errstate(invalid="ignore", divide="ignore"):
                    stat = {"n": s["n"], "wr": (s["wins"] + K * P_WIN) / (s["n"] + K), "ae": s["ae"] / (s["n"] + K),
                            "nfp": (s["nfp_v"] + K * NFP0) / (s["nfp_n"] + K), "perf": s["perf_v"] / (s["perf_n"] + K)}
                for st in STATS:
                    cols[f"ff_l{k}_n{f}_{st}"] = np.where(has_run, stat[st], np.nan)
        prev = np.clip(b - 1, 0, None)
        with np.errstate(invalid="ignore"):
            cols[f"ff_l1_n{f}_adj"] = np.where(has_run, bv["perf"][prev] + cols[f"ff_l1_n{f}_perf"], np.nan)
    rows = {c: H.to_rows(v) for c, v in cols.items()}
    new = pd.DataFrame(rows, index=df.index)[FEATURES]
    assert len(new) == n_rows
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
