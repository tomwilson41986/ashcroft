"""Form lines weighted by how close each rival finished to the horse.

form_lines counts every rival from a horse's recent races alike. A race reader
does not: a horse beaten a neck by a subsequent winner has run to that
winner's level, one beaten twenty lengths has not. So each rival's runs since
count by how near it finished to the horse, exp(-|lengths between them| / L),
L = 3 lengths (1 beside it, 0.37 three lengths away, 0.04 ten away):

    flc_<w>_n     the rivals' runs since, weighted
    flc_<w>_wins  their wins since, weighted
    flc_<w>_wr    their weighted win rate, shrunk to 1 in 10 by five runs
    flc_<w>_ae    their weighted wins less the market's chances at BSP, per run,
                  shrunk to 0 by five runs

w = l1 (the last race) or l3 (the last three together). Each rival's next three
runs, on days before today, as form_lines. A rival, or the horse, without a
finishing distance in the race (a non-finisher; lengths not parsed) counts for
nothing there. NaN before a horse's first run.

Today's results never enter: only runs on days before the row's day are
counted, summed in a fixed order (by race, rival run date, then horse), so a
card and the same rows with results in agree to the bit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks import form_lines
from model.freshness_features import _Days, _col
from model.race_shape import day_index, race_key

WINDOWS = ("l1", "l3")
STATS = ("n", "wins", "wr", "ae")
FEATURES = [f"flc_{w}_{s}" for w in WINDOWS for s in STATS]
POST_RACE: set[str] = set()
READS = list(dict.fromkeys(form_lines.READS + ["LB", "total_dst_bt"]))
NEXT = form_lines.NEXT
LAST = form_lines.LAST
K = form_lines.K
P_WIN = form_lines.P_WIN
L = 3.0             # lengths at which a rival's weight falls to 1/e


def _lengths(df: pd.DataFrame) -> np.ndarray:
    """Lengths behind the winner (0 for the winner), NaN for a non-finisher or an unparsed distance."""
    from model.perf_figures import parse_beaten_lengths

    pos = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(dtype=float)
    if "LB" in df.columns:
        lb = pd.to_numeric(df["LB"], errors="coerce").to_numpy(dtype=float)
    else:
        lb = _col(df, "total_dst_bt").map(parse_beaten_lengths).to_numpy(dtype=float)
    return np.where(pos == 1, 0.0, np.where(pos > 1, lb, np.nan))


def build(df: pd.DataFrame) -> pd.DataFrame:
    n_rows = len(df)
    day = day_index(df)
    name = df["horse_name"].fillna("").astype(str).str.strip().str.lower()
    bad = name.isin(["", "nan", "none"]).to_numpy() | (day < 0)
    horse = np.where(bad, -1, pd.factorize(name, sort=True)[0]).astype(np.int64)
    race_rows = pd.factorize(race_key(df), sort=True)[0].astype(np.int64)
    vals = form_lines._values(df)

    H = _Days(horse, day)
    nb = len(H.u)
    b = np.arange(nb)
    race = np.nan_to_num(H.per_block(race_rows, how="max"), nan=-1).astype(np.int64)
    lb = H.per_block(_lengths(df), how="min")
    won = np.nan_to_num(H.per_block(vals["wins"], how="max"))
    ae = np.nan_to_num(H.per_block(vals["ae"], how="max"))

    # every rival run: the race it came out of, the rival's distance there, the run's day, horse and result
    parts = []
    for k in range(1, NEXT + 1):
        j = b + k
        ok = H.valid & (j < nb)
        ok[ok] = H.key[j[ok]] == H.key[b[ok]]
        ok &= np.isfinite(lb)
        parts.append(pd.DataFrame({"R": race[b[ok]], "rlb": lb[b[ok]], "ds": H.day[j[ok]], "h": H.key[j[ok]],
                                   "won": won[j[ok]], "ae": ae[j[ok]]}))
    ev = pd.concat(parts, ignore_index=True)

    sums = {w: {s: np.zeros(nb) for s in ("n", "wins", "ae")} for w in WINDOWS}
    has = H.valid & (H.pos >= 1)
    for k in range(1, LAST + 1):
        okq = H.valid & (H.pos >= k)
        qb = b[okq]
        then = qb - k
        q = pd.DataFrame({"qid": qb, "R": race[then], "qlb": lb[then], "t": H.day[qb], "qh": H.key[qb]})
        q = q[np.isfinite(q["qlb"].to_numpy())]
        m = q.merge(ev, on="R", how="inner")
        m = m[(m["ds"].to_numpy() < m["t"].to_numpy()) & (m["h"].to_numpy() != m["qh"].to_numpy())]
        m = m.sort_values(["qid", "ds", "h"], kind="stable")
        wgt = np.exp(-np.abs(m["rlb"].to_numpy() - m["qlb"].to_numpy()) / L)
        agg = pd.DataFrame({"qid": m["qid"].to_numpy(), "n": wgt, "wins": wgt * m["won"].to_numpy(),
                            "ae": wgt * m["ae"].to_numpy()}).groupby("qid", sort=True).sum()
        idx = agg.index.to_numpy()
        for s in ("n", "wins", "ae"):
            add = np.zeros(nb)
            add[idx] = agg[s].to_numpy()
            if k == 1:
                sums["l1"][s] += add
            sums["l3"][s] += add

    cols = {}
    for w in WINDOWS:
        n, wins, a = sums[w]["n"], sums[w]["wins"], sums[w]["ae"]
        stats = {"n": n, "wins": wins, "wr": (wins + K * P_WIN) / (n + K), "ae": a / (n + K)}
        for s in STATS:
            cols[f"flc_{w}_{s}"] = H.to_rows(np.where(has, stats[s], np.nan))
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    assert len(new) == n_rows
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
