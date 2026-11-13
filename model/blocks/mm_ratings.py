"""Strengths by Hunter's MM algorithm: who beat whom, credited for the opposition, refitted on earlier races only.

The owner, 12 Nov 2026: "what about Hunter's Min-Max? Discover how we could use this." The ratings so far (Elo, the
market rating, the connections rating) are filters: each race is read once, and an old field is never re-rated when
its horses run again. The MM algorithm (minorisation-maximisation; Hunter 2004, generalised Bradley-Terry models)
fits every strength at once, so a race is credited for what its field went on to do. Here it is the Plackett-Luce
form on the front of each race: a race is a sequence of choices (the winner from the field, the second from the
rest, ... to STAGES places), each runner chosen with a chance in proportion to its strength.

Every SNAP_DAYS a fit is made on the races of the WINDOW days before the snapshot, a race weighted by
2^(-age / HALF), each strength with a gamma prior worth PRIOR runs at 1; the rows of the days from the snapshot to
the next read that fit. So a row never sees its own day or a later one, and its strengths are up to SNAP_DAYS old.

Three fits:
  horses        strength = the horse's
  connections   strength = the runner's share of the Betfair SP book x trainer x jockey x sire: what each adds
                beyond the price, fitted together (the exact form of crediting a jockey net of his trainers)
  the draw      strength = the BSP share x the stall's third of the field at this course and trip band (races with
                stalls): the draw the price has not allowed for

    mmr_h  mmr_h_w            the horse's log strength, and the weighted number of races behind it
    mmr_share  mmr_lshare     its share of today's field's strengths (an unrated horse at the prior), and
                              log(runners x share)
    mmr_z  mmr_field          its log strength within the race, standardised; the field's mean log strength
    mmr_tr  mmr_jk  mmr_sire  the trainer's, jockey's and sire's log multipliers beyond the price
    mmr_conn  mmr_conn_z      their sum, and that standardised within the race
    mmr_draw                  today's stall third's log multiplier at this course and trip band
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.freshness_features import _col
from model.race_shape import DIST_BANDS, bands, day_index, race_key

FEATURES = ["mmr_h", "mmr_h_w", "mmr_share", "mmr_lshare", "mmr_z", "mmr_field", "mmr_tr", "mmr_jk", "mmr_sire",
            "mmr_conn", "mmr_conn_z", "mmr_draw"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "trainer", "jockey_name", "stallion",
         "placing_numerical", "number_of_runners", "bfsp", "stall", "dist_furlongs"]
STAGES = 4
SNAP_DAYS = 30
WINDOW = 3 * 365
HALF = 365.0
PRIOR = 3.0
ITERS = 40
TEST_REACH = {"SNAP_DAYS": 5, "WINDOW": 60, "HALF": 30.0, "ITERS": 15}


def _codes(s: pd.Series) -> np.ndarray:
    t = s.fillna("").astype(str).str.strip().str.lower()
    return np.where(t.isin(["", "nan", "none"]).to_numpy(), -1, pd.factorize(t, sort=True)[0]).astype(np.int64)


def fit(race, pos, weight, offset, entities, iters=None, prior=None):
    """MM for a Plackett-Luce model on the first STAGES places, the runner's strength offset x the product of its
    entities' strengths. race: contiguous race codes; pos: finishing position (NaN: not placed); weight: per row;
    offset: per row (> 0); entities: a list of code arrays (-1: none, strength 1). Returns one strength array an
    entity list, and the weighted number of stage choices each entity took part in."""
    iters, prior = ITERS if iters is None else iters, PRIOR if prior is None else prior
    nr = int(race.max()) + 1 if len(race) else 0
    sizes = [int(e.max()) + 1 if len(e) and e.max() >= 0 else 0 for e in entities]
    gam = [np.ones(max(s, 1)) for s in sizes]
    rows = np.bincount(race, minlength=nr)
    stage_of = np.where(np.isfinite(pos) & (pos >= 1) & (pos <= STAGES), pos, 0).astype(int)
    # a stage counts when exactly one runner holds that place, every earlier stage counts, and a rival remains
    live = np.ones(nr, dtype=bool)
    exists = np.zeros((STAGES + 1, nr), dtype=bool)
    for t in range(1, STAGES + 1):
        cnt = np.bincount(race[stage_of == t], minlength=nr)
        live &= (cnt == 1) & (rows > t)
        exists[t] = live
    reach = np.where(stage_of > 0, stage_of, STAGES)                    # the last stage a runner is still in
    chosen = (stage_of > 0) & exists[stage_of, race]
    exposure = []
    for e in entities:
        k = np.where(e >= 0, e, 0)
        took = np.zeros(len(race))
        for t in range(1, STAGES + 1):
            took += (exists[t, race] & (reach >= t)) * weight
        exposure.append(np.bincount(k, weights=np.where(e >= 0, took, 0.0), minlength=len(gam[len(exposure)])))
    for _ in range(iters):
        for j, e in enumerate(entities):
            if not sizes[j]:
                continue
            v = offset.copy()
            for i, o in enumerate(entities):
                v = v * np.where(o >= 0, gam[i][np.where(o >= 0, o, 0)], 1.0)
            remaining = np.bincount(race, weights=v, minlength=nr)
            inv = np.zeros(len(race))
            for t in range(1, STAGES + 1):
                ok = exists[t] & (remaining > 0)
                inv += np.where(ok[race] & (reach >= t), 1.0 / np.where(ok, remaining, 1.0)[race], 0.0)
                remaining = remaining - np.bincount(race[stage_of == t], weights=v[stage_of == t], minlength=nr)
            k = np.where(e >= 0, e, 0)
            mine = gam[j][k]
            num = np.bincount(k, weights=np.where((e >= 0) & chosen, weight, 0.0), minlength=len(gam[j]))
            den = np.bincount(k, weights=np.where(e >= 0, weight * inv * v / mine, 0.0), minlength=len(gam[j]))
            gam[j] = (num + prior) / (den + prior)
    return gam, exposure


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    race = pd.factorize(race_key(df), sort=True)[0].astype(np.int64)
    rs = pd.Series(race, index=df.index)
    horse, trainer, jockey, sire = (_codes(_col(df, c)) for c in ("horse_name", "trainer", "jockey_name", "stallion"))
    pos = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(float)
    n = pd.to_numeric(_col(df, "number_of_runners"), errors="coerce").to_numpy(float)
    bsp = pd.to_numeric(_col(df, "bfsp"), errors="coerce").to_numpy(float)
    priced = np.isfinite(bsp) & (bsp > 1.0)
    inv = pd.Series(np.where(priced, 1.0 / np.where(priced, bsp, 1.0), 0.0), index=df.index)
    tot = inv.groupby(rs).transform("sum").to_numpy()
    rows = pd.Series(1.0, index=df.index).groupby(rs).transform("sum").to_numpy()
    npr = pd.Series(priced.astype(float), index=df.index).groupby(rs).transform("sum").to_numpy()
    whole = (npr == rows) & (tot > 0)                                   # the price fits need every runner priced
    share = np.where(whole, inv.to_numpy() / np.where(tot > 0, tot, 1.0), np.nan)
    stall = pd.to_numeric(_col(df, "stall"), errors="coerce").to_numpy(float)
    dist = pd.to_numeric(_col(df, "dist_furlongs"), errors="coerce").to_numpy(float)
    tb = pd.Series(bands(dist, DIST_BANDS)).fillna(-1).to_numpy().astype(np.int64)
    course = _codes(_col(df, "track"))
    with np.errstate(invalid="ignore", divide="ignore"):
        third = np.where((stall > 0) & (n > 2), np.minimum(np.floor(3.0 * (stall - 1) / n), 2), -1).astype(np.int64)
    cell = np.where((third >= 0) & (course >= 0) & (tb >= 0), (course * 16 + np.where(tb < 0, 0, tb)) * 3 + third, -1)
    draw = np.where(cell >= 0, pd.factorize(cell, sort=True)[0], -1).astype(np.int64)
    staged = pd.Series((draw >= 0).astype(float), index=df.index).groupby(rs).transform("min").to_numpy() == 1

    out = {c: np.full(len(df), np.nan) for c in ("h", "h_w", "tr", "jk", "sire", "draw")}
    ok_day = day >= 0
    if ok_day.any():
        d0, d1 = int(day[ok_day].min()), int(day[ok_day].max())
        for snap in range(d0 + SNAP_DAYS, d1 + 1, SNAP_DAYS):
            tr_rows = ok_day & (day < snap) & (day >= snap - WINDOW)
            te = ok_day & (day >= snap) & (day < snap + SNAP_DAYS)
            if not tr_rows.any() or not te.any():
                continue
            w = np.power(2.0, -(snap - day) / HALF)

            def sub(mask, ents):
                idx = np.flatnonzero(mask)
                r = pd.factorize(race[idx], sort=True)[0].astype(np.int64)
                return idx, r, [e[idx] for e in ents]

            idx, r, (hh,) = sub(tr_rows, [horse])
            (gh,), (eh,) = fit(r, pos[idx], w[idx], np.ones(len(idx)), [hh])
            k = horse[te]
            seen = (k >= 0) & (k < len(gh))
            out["h"][te] = np.where(seen, np.log(gh[np.where(seen, k, 0)]), np.nan)
            out["h_w"][te] = np.where(seen, eh[np.where(seen, k, 0)], np.nan)

            idx, r, ents = sub(tr_rows & whole, [trainer, jockey, sire])
            if len(idx):
                g, _ = fit(r, pos[idx], w[idx], share[idx], ents)
                for name, code, gg in (("tr", trainer, g[0]), ("jk", jockey, g[1]), ("sire", sire, g[2])):
                    k = code[te]
                    seen = (k >= 0) & (k < len(gg))
                    out[name][te] = np.where(seen, np.log(gg[np.where(seen, k, 0)]), np.where(k >= 0, 0.0, np.nan))
            idx, r, (dd,) = sub(tr_rows & whole & staged, [draw])
            if len(idx):
                (gd,), _ = fit(r, pos[idx], w[idx], share[idx], [dd])
                k = draw[te]
                seen = (k >= 0) & (k < len(gd))
                out["draw"][te] = np.where(seen, np.log(gd[np.where(seen, k, 0)]), np.where(k >= 0, 0.0, np.nan))

    cols: dict[str, np.ndarray] = {"mmr_h": out["h"], "mmr_h_w": out["h_w"], "mmr_tr": out["tr"], "mmr_jk": out["jk"],
                                   "mmr_sire": out["sire"], "mmr_draw": out["draw"]}
    fitted = np.isfinite(out["tr"]) | np.isfinite(out["h"]) | np.isfinite(out["jk"])        # a snapshot covers the row
    lg = pd.Series(np.where(np.isfinite(out["h"]), out["h"], np.where(fitted, 0.0, np.nan)), index=df.index)
    g = lg.groupby(rs)
    top = g.transform("max")
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        e = np.exp(lg - top)
        sh = (e / e.groupby(rs).transform("sum")).to_numpy(float)
        z = ((lg - g.transform("mean")) / g.transform("std")).where(g.transform("count") >= 3)
    cols["mmr_share"], cols["mmr_lshare"] = sh, np.log(rows * sh)
    cols["mmr_z"] = z.replace([np.inf, -np.inf], np.nan).to_numpy(float)
    cols["mmr_field"] = g.transform("mean").to_numpy(float)
    conn = pd.Series(np.nan_to_num(out["tr"]) + np.nan_to_num(out["jk"]) + np.nan_to_num(out["sire"]), index=df.index) \
        .where(np.isfinite(out["tr"]) | np.isfinite(out["jk"]) | np.isfinite(out["sire"]))
    gc = conn.groupby(rs)
    with np.errstate(invalid="ignore", divide="ignore"):
        zc = ((conn - gc.transform("mean")) / gc.transform("std")).where(gc.transform("count") >= 3)
    cols["mmr_conn"], cols["mmr_conn_z"] = conn.to_numpy(float), zc.replace([np.inf, -np.inf], np.nan).to_numpy(float)
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
