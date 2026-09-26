"""How a horse's recent races have worked out, in figures: what its rivals have run since against what they ran there.

form_lines reads a race through its rivals' wins since. A win is one bit a
run, and most runs are not wins. The rivals' later figures say more per run: if
the runners of a race have since run 5 lb better than they did in it, the race
was better than its figures said, and so was every run in it. A handicapper
revises a race's form this way, and so, on its own evidence, does the market.

For the horse's last race, and its last three together: its rivals' next runs
after that race (each rival's next three at most) on days before today, and

    fu_<w>_perf  their performance figures since (official-rating scale, lb) less
                 theirs in the race, per run, shrunk to 0 by five runs
    fu_<w>_or    their official ratings since less theirs that day: the
                 handicapper's reassessment of them
    fu_<w>_mkt   the market's view of them since less its view there (log of
                 field size x the BSP's normalised chance)
    fu_<w>_nres  their finishing positions since against the market's order
                 (+ = beat their prices), per run, shrunk to 0

and the horse's figure in its last race, revised by how that race has worked
out, against today's mark:

    fu_l1_adj_vs_or  its performance figure there + fu_l1_perf, less today's
                     official rating (+ = has run to more than it is rated)

w = l1 (the last race) or l3 (the last three; the horse's own later runs are
left out). A pair of readings counts where both are known (a non-finisher has
no figure; a race without ratings has no marks). NaN before a horse's first run.

Today's results never enter: only runs on days before the row's day count,
each race's runs summed in a fixed order (by day, then horse) from zero, so a
06:00 card and the same rows with results in get the same values, to the bit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.freshness_features import _Days, _col
from model.race_shape import day_index, race_key

WINDOWS = ("l1", "l3")
MEASURES = ("perf", "or", "mkt", "nres")
FEATURES = [f"fu_{w}_{m}" for w in WINDOWS for m in MEASURES] + ["fu_l1_adj_vs_or"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "number_of_runners", "placing_numerical",
         "bfsp", "official_rating", "LB", "total_dst_bt", "dist_furlongs"]
NEXT = 3            # each rival's next runs after the race that count
LAST = 3            # the horse's races pooled in l3
K = 5.0             # shrinkage, in runs
#: read as the rival's later run less its run in the race; nres is the later run's own
CHANGE = ("perf", "or", "mkt")


def _values(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each run's readings, from its own race (post-race except the mark)."""
    from model.form_windows import run_measures

    m = run_measures(df)
    orr = pd.to_numeric(_col(df, "official_rating"), errors="coerce").to_numpy(dtype=float)
    return {"perf": m["perf"], "or": np.where(orr > 0, orr, np.nan), "mkt": m["mkt"], "nres": m["nres"]}


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

    def reading(s: str, later: np.ndarray, then: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The event's value and whether it is known: the later run's reading, less the race's for a change."""
        x = bv[s][later] - bv[s][then] if s in CHANGE else bv[s][later]
        f = np.isfinite(x)
        return np.where(f, x, 0.0), f.astype(float)

    # events: for every horse-day block, the horse's next NEXT days, filed under the block's race
    ev_r, ev_d, ev_h = [], [], []
    ev_v = {s: [] for s in MEASURES}
    ev_n = {s: [] for s in MEASURES}
    for k in range(1, NEXT + 1):
        j = b + k
        ok = H.valid & (j < nb)
        ok[ok] = H.key[j[ok]] == H.key[b[ok]]
        ev_r.append(race[b[ok]])
        ev_d.append(H.day[j[ok]])
        ev_h.append(H.key[j[ok]])
        for s in MEASURES:
            x, f = reading(s, j[ok], b[ok])
            ev_v[s].append(x)
            ev_n[s].append(f)
    ev_r, ev_d, ev_h = np.concatenate(ev_r), np.concatenate(ev_d), np.concatenate(ev_h)
    order = np.lexsort((ev_h, ev_d, ev_r))                  # race, then day, then horse: no row-order effect
    ev_r, ev_d = ev_r[order], ev_d[order]
    dmin = int(H.day[H.valid].min()) if H.valid.any() else 0
    span = int(H.day[H.valid].max()) - dmin + 2 if H.valid.any() else 2
    code = ev_r * span + (ev_d - dmin)
    csum = {}
    for s in MEASURES:
        for part, ev in (("v", ev_v), ("n", ev_n)):
            v = np.concatenate(ev[s])[order]
            csum[(s, part)] = pd.Series(v).groupby(ev_r, sort=False).cumsum().to_numpy()

    def before(r: np.ndarray, t: np.ndarray, ok: np.ndarray) -> dict:
        """Per block: each sum over race r's events on days before t (0 where none)."""
        q = np.where(ok, r * span + (t - dmin), -1)
        hi = np.searchsorted(code, q, side="left")
        lo = np.searchsorted(code, np.where(ok, r * span, -1), side="left")
        has = ok & (hi > lo)
        last = np.clip(hi - 1, 0, None)
        return {key: np.where(has, c[last], 0.0) if len(c) else np.zeros(len(r)) for key, c in csum.items()}

    has_run = H.valid & (H.pos >= 1)
    l1 = None
    tot = {key: np.zeros(nb) for key in csum}
    for k in range(1, LAST + 1):
        ok = H.valid & (H.pos >= k)
        then = np.clip(b - k, 0, None)
        got = before(np.where(ok, race[then], -1), H.day, ok)
        own = {key: np.zeros(nb) for key in csum}
        for i in range(1, k):                                # its own runs since that race, all before today
            if k - i <= NEXT:
                later = np.clip(b - i, 0, None)
                for s in MEASURES:
                    x, f = reading(s, later, then)
                    own[(s, "v")] += np.where(ok, x, 0.0)
                    own[(s, "n")] += np.where(ok, f, 0.0)
        for key in csum:
            tot[key] += np.where(ok, got[key] - own[key], 0.0)
        if k == 1:
            l1 = {key: v.copy() for key, v in got.items()}

    cols = {}
    for w, sums in (("l1", l1), ("l3", tot)):
        for s in MEASURES:
            with np.errstate(invalid="ignore", divide="ignore"):
                cols[f"fu_{w}_{s}"] = np.where(has_run, sums[(s, "v")] / (sums[(s, "n")] + K), np.nan)
    prev = np.clip(b - 1, 0, None)
    with np.errstate(invalid="ignore"):
        cols["fu_l1_adj_vs_or"] = np.where(has_run, bv["perf"][prev] + cols["fu_l1_perf"] - bv["or"], np.nan)
    rows = {c: H.to_rows(v) for c, v in cols.items()}
    new = pd.DataFrame(rows, index=df.index)[FEATURES]
    assert len(new) == n_rows
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
