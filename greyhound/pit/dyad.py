"""Dyadic memory: what each pair of dogs has done to each other beyond their ratings (the owner, 11 Nov 2026:
"identify NEW features that have not been discovered yet").

Greyhounds race in graded pools at a home track, six to a race, about weekly, so the same two dogs meet again and
again. A rating (a time figure, an expected time) says how fast each runs alone; it does not say that A always
crosses B at the first bend, or that B will not pass A. This table keeps a memory for every PAIR.

For each run in a graded race: ``r`` = (its calculated time less its expected time before the race, the
expected-time table's ``xt_tot_time``) in lengths, above 0 slower than expected. For two dogs A and B in the same
race the pair's residual is ``adv(A over B) = r_B - r_A``: how many lengths A beat B by beyond what their expected
times said. The night's going, the track and the grade are common to both and cancel.

For every pair in a race the table reads the pair's EARLIER days only: the sum and count of ``adv`` and of who
finished in front. Shrunk by ``K`` meetings of prior to zero, and summed over a dog's rivals today:

- ``dy_edge``: the lengths its rivals of today have tended to give it beyond the ratings (above 0: it has had the
  better of them); ``dy_best`` / ``dy_worst``: the rival it most has the better of, and its nemesis;
- ``dy_ahead``: the same on finishing in front (share in front less a half, shrunk, summed);
- ``dy_pairs``, ``dy_meets``: how many of today's rivals it has met, and how many times in all (familiarity);
- ``dy_curl``: intransitivity around the dog. For three dogs that have all met, the cyclic sum
  adv(A,B) + adv(B,C) + adv(C,A) is zero when the three results are consistent with ANY ratings and non-zero when
  they are rock-paper-scissors (the curl of the margin flow in its Hodge decomposition). The mean absolute curl of
  the triangles through the dog in today's race, and ``dy_curl_race`` over the whole race: how much the result
  will depend on the match-ups, not the ratings.

Every column is read from earlier days only: a pair's meetings on the day itself, earlier or later, are not read
(tests/test_greyhound_dyad.py; tests/test_greyhound_pit.py).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from greyhound.pit import core

PREFIX = "dy_"
SOURCE = "GBGB results (calculated time, position: sources/gbgb.py) against the expected-time table (xt_tot_time)"
K = 2.0                      # meetings of prior toward no pair effect
CLIP = 15.0                  # lengths: a run further than this from its expected time is not a pair's doing
SLOTS = 6
PAIRS = list(combinations(range(SLOTS), 2))
TRIPLES = list(combinations(range(SLOTS), 3))


def residuals(df: pd.DataFrame, xt_time: pd.Series) -> pd.Series:
    """Each graded run's lengths slower than its expected time (NaN: a trial, a fall, a card row, no clock)."""
    bad = df.comment.fillna("").str.contains(r"\(Handslip\)|Fell|Fll|BrkLine", case=False) if "comment" in df else \
        pd.Series(False, index=df.index)
    trial = df.is_trial.fillna(False).astype(bool) if "is_trial" in df else pd.Series(False, index=df.index)
    r = (pd.to_numeric(df.adjusted_time, errors="coerce") - pd.to_numeric(xt_time, errors="coerce")) / core.LENGTH_S
    return r.where(~bad & ~trial & (r.abs() <= CLIP))


def earlier(key: np.ndarray, dn: np.ndarray, cols: list[np.ndarray]) -> list[np.ndarray]:
    """For rows keyed by ``key`` on day ``dn``: each column's sum over the key's rows on EARLIER days (the day's own
    rows, before or after, are left out). Aligned to the input order."""
    order = np.lexsort((dn, key))
    k, d = key[order], dn[order]
    new_key = np.r_[True, k[1:] != k[:-1]]
    new_day = new_key | np.r_[True, d[1:] != d[:-1]]
    start = np.flatnonzero(new_key)
    grp = np.cumsum(new_key) - 1
    first_of_day = np.maximum.accumulate(np.where(new_day, np.arange(len(k)), 0))
    out = []
    for c in cols:
        v = np.nan_to_num(c[order].astype(float))
        cs = np.cumsum(v)
        before_row = cs - v                                   # everything before this row, any key
        base = before_row[start][grp]                         # everything before the key's first row
        res = (before_row - base)[first_of_day]               # the key's rows before this DAY's first row
        back = np.empty(len(res))
        back[order] = res
        out.append(back)
    return out


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = figs if figs is not None else core.per_run_figures(runs)
    df = df.sort_values(["t", "race_id", "trap"], kind="stable")
    xt = (ctx or {}).get("xtime")
    if xt is None:
        from greyhound.pit import xtime
        xt, _ = xtime.build(runs, df)
    j = df[["race_id", "dog_id"]].merge(xt[["race_id", "dog_id", "xt_tot_time"]].drop_duplicates(["race_id", "dog_id"]),
                                        on=["race_id", "dog_id"], how="left")
    r = residuals(df, pd.Series(j.xt_tot_time.to_numpy(), index=df.index)).to_numpy(float)
    pos = pd.to_numeric(df.position, errors="coerce").to_numpy(float)
    rc = pd.factorize(df.race_id)[0]
    dog = pd.factorize(df.dog_id)[0].astype(np.int64)
    nd = int(dog.max()) + 1 if len(df) else 1
    nr = int(rc.max()) + 1 if len(df) else 0
    slot = df.groupby(rc).cumcount().to_numpy()
    dn_row = (core.day(df) - pd.Timestamp("2000-01-01")).dt.days.to_numpy()
    ok = slot < SLOTS
    DOG = np.full((nr, SLOTS), -1, dtype=np.int64)
    R = np.full((nr, SLOTS), np.nan)
    POS = np.full((nr, SLOTS), np.nan)
    DN = np.zeros(nr, dtype=np.int64)
    DOG[rc[ok], slot[ok]], R[rc[ok], slot[ok]], POS[rc[ok], slot[ok]] = dog[ok], r[ok], pos[ok]
    DN[rc] = dn_row
    # every pair of every race, oriented low dog code to high
    race_i, pi, key, sgn, adv, has, ahead, placed = [], [], [], [], [], [], [], []
    for p, (a, b) in enumerate(PAIRS):
        da, db = DOG[:, a], DOG[:, b]
        v = np.flatnonzero((da >= 0) & (db >= 0) & (da != db))
        lo, hi = np.minimum(da[v], db[v]), np.maximum(da[v], db[v])
        s = np.where(da[v] == lo, 1.0, -1.0)                       # +1: slot a is the low dog
        d = R[v, b] - R[v, a]                                      # a over b
        f = POS[v, a] < POS[v, b]
        pl = np.isfinite(POS[v, a]) & np.isfinite(POS[v, b]) & (POS[v, a] != POS[v, b])
        race_i.append(v)
        pi.append(np.full(len(v), p))
        key.append(lo * nd + hi)
        sgn.append(s)
        adv.append(s * d)                                           # low over high
        has.append(np.isfinite(d))
        ahead.append(np.where(s > 0, f, ~f) & pl)                  # the low dog in front
        placed.append(pl)
    race_i, pi, key, sgn = map(np.concatenate, (race_i, pi, key, sgn))
    adv, has, ahead, placed = map(np.concatenate, (adv, has, ahead, placed))
    S, N, A, M = earlier(key, DN[race_i], [np.where(has, adv, 0.0), has, ahead, placed])
    m_adv = sgn * S / (N + K)                                       # slot a over slot b, shrunk
    m_ahead = sgn * (A - 0.5 * M) / (M + K)
    met = N > 0
    a_slot = np.array([p[0] for p in PAIRS])[pi]
    b_slot = np.array([p[1] for p in PAIRS])[pi]
    edge, front, pairs, meets = (np.zeros((nr, SLOTS)) for _ in range(4))
    best, worst = np.full((nr, SLOTS), -np.inf), np.full((nr, SLOTS), np.inf)
    for sl, sign in ((a_slot, 1.0), (b_slot, -1.0)):
        np.add.at(edge, (race_i, sl), sign * m_adv)
        np.add.at(front, (race_i, sl), sign * m_ahead)
        np.add.at(pairs, (race_i, sl), met.astype(float))
        np.add.at(meets, (race_i, sl), N)
        v = np.where(met, sign * m_adv, np.nan)
        np.fmax.at(best, (race_i, sl), v)
        np.fmin.at(worst, (race_i, sl), v)
    # the curl of every triangle whose three pairs have all met
    Mx = np.zeros((nr, SLOTS, SLOTS), dtype=np.float32)
    W = np.zeros((nr, SLOTS, SLOTS), dtype=bool)
    Mx[race_i, a_slot, b_slot], Mx[race_i, b_slot, a_slot] = m_adv, -m_adv
    W[race_i, a_slot, b_slot] = W[race_i, b_slot, a_slot] = met
    c_sum, c_n = np.zeros((nr, SLOTS)), np.zeros((nr, SLOTS))
    for a, b, c in TRIPLES:
        full = W[:, a, b] & W[:, b, c] & W[:, a, c]
        curl = np.abs(Mx[:, a, b] + Mx[:, b, c] + Mx[:, c, a]) * full
        for s_ in (a, b, c):
            c_sum[:, s_] += curl
            c_n[:, s_] += full
    with np.errstate(invalid="ignore", divide="ignore"):
        curl_dog = np.where(c_n > 0, c_sum / c_n, np.nan)
        curl_race = np.where(c_n.sum(axis=1) > 0, c_sum.sum(axis=1) / np.maximum(c_n.sum(axis=1), 1), np.nan)
    out = pd.DataFrame({"race_id": df.race_id.to_numpy(), "dog_id": df.dog_id.to_numpy()}, index=df.index)

    def at(arr):
        v = np.full(len(df), np.nan)
        v[ok] = arr[rc[ok], slot[ok]]
        return v

    none = at(pairs) == 0
    out[f"{PREFIX}edge"] = np.where(none, np.nan, at(edge))
    out[f"{PREFIX}ahead"] = np.where(none, np.nan, at(front))
    out[f"{PREFIX}best"] = np.where(np.isfinite(at(best)), at(best), np.nan)
    out[f"{PREFIX}worst"] = np.where(np.isfinite(at(worst)), at(worst), np.nan)
    out[f"{PREFIX}pairs"] = at(pairs)
    out[f"{PREFIX}meets"] = at(meets)
    out[f"{PREFIX}curl"] = at(curl_dog)
    out[f"{PREFIX}curl_race"] = curl_race[rc]
    cols = [c for c in out.columns if c.startswith(PREFIX)]
    out[cols] = out[cols].astype(np.float32)
    graded = ~(df.is_trial.fillna(False).to_numpy(bool) if "is_trial" in df else np.zeros(len(df), bool))
    notes = {"k_prior": K, "pair_rows": int(len(key)), "distinct_pairs": int(len(np.unique(key))),
             "pair_rows_with_an_earlier_meeting": int(met.sum()),
             "share_of_pair_rows_with_an_earlier_meeting": round(float(met.mean()), 4) if len(met) else None,
             "graded_runs_with_a_rival_met_before": round(float((out[f"{PREFIX}pairs"].to_numpy()[graded] > 0).mean()), 4)
             if graded.any() else None,
             "mean_rivals_met_before": round(float(np.nanmean(out[f"{PREFIX}pairs"].to_numpy()[graded])), 3)
             if graded.any() else None,
             "triangles_all_met_share_of_races": round(float(np.isfinite(curl_race).mean()), 4) if nr else None}
    return out.sort_index().reset_index(drop=True), notes


COLUMNS = {
    f"{PREFIX}edge": "lengths today's rivals have given the dog beyond the ratings in their earlier meetings (each "
                     f"pair's mean shrunk by {K:g} meetings, summed; above 0 it has had the better of them)",
    f"{PREFIX}ahead": "the same on finishing in front: each pair's share in front less a half, shrunk, summed",
    f"{PREFIX}best": "the pair it most has the better of today (lengths, shrunk)",
    f"{PREFIX}worst": "its nemesis today: the pair it has had the worst of (lengths, shrunk)",
    f"{PREFIX}pairs": "how many of today's rivals it has met before",
    f"{PREFIX}meets": "how many earlier meetings with today's rivals in all",
    f"{PREFIX}curl": "intransitivity around the dog: the mean absolute cyclic sum of the pair effects over the "
                     "triangles of today's race through it whose three pairs have all met (lengths)",
    f"{PREFIX}curl_race": "the same over every such triangle of the race",
}
