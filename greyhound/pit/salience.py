"""Salience: each printed attribute of a dog set against today's field, and the largest standout.

A reader of the card does not read a dog alone: the eye goes to the one that stands out of the six, the only last-time
winner, the time a class apart from the others. This table measures, for each attribute the card prints, how far the
dog stands from the field's median and from the field's best in units of the field's own spread, whether it is alone
at the top, and over the attributes the largest standout for it and against it.

Screened on 12 Nov 2026 against what the stack still misses (the residual of the stacked model's forecast of the
Betfair SP; the discovery screen ``salience``: the set lowered that residual's RMSE from 0.2872 to 0.2863 on 49,612
runners, z -5.2, and from 0.2854 to 0.2850 on top of the optics table). A screen, not a registered trial: nothing here
has passed a gate.

**The attributes** (each the dog's state after its last run on an EARLIER day, a day's last run where it ran twice):

- ``lp``: the finishing position of that last run (NaN when it had none: a void race, a non-finisher);
- ``w6``: wins in its last six runs;
- ``dw``: its weight at that run less its weight at the run before (the weigh-in of today is NOT read);
- at today's track and distance only (times do not compare across courses): ``lt`` the calculated time of its last
  run there, ``bt3`` the fastest calculated time of its last three runs there, ``sec3`` the mean sectional of its
  last three runs there. A calculated time outside ``PACE`` seconds a metre is not a greyhound's time (GBGB's keying
  errors; the optics table's bounds) and is not read.

**Against the field.** The field is every row of ``runs`` with the race's id. Each attribute is signed so that higher
is better (times, sectionals and positions negated; ``dw`` as it is). Within the race, over the runners that have the
attribute: the spread is the sample sd, never below ``FLOOR`` (so a field of near-identical times does not make a
hundredth a standout), and NaN with fewer than ``MIN_FIELD`` runners with a value; ``_med`` = (value - the field's
median) / spread; ``_best`` = (value - the field's best) / spread (0 for the best, below 0 for the rest). A dog is
AT THE TOP of an attribute when its value is the field's best and some runner has a lower one; its lead is
(value - the best value under the top) / spread. As screened, two runners that share the best value are both at the
top (two last-time winners both top ``lp``): the standout below is the dog that tops more attributes than any other.

**The largest standout**, over the five form attributes (``lt``, ``bt3``, ``lp``, ``w6``, ``sec3``; not the weight):
``sl_max_for`` and ``sl_max_against`` the largest and the smallest ``_med``; ``sl_n_top`` how many it is at the
top of; ``sl_lead`` its largest lead (0 when it tops none); ``sl_is_standout`` 1 when it tops at least ``STANDOUT``
of them and no other runner tops as many.

Every column reads rows dated strictly before the race day and, of the race itself, the track, the distance and the
field's dogs (tests/test_greyhound_salience.py: a day's results rewritten, later days added, a card row). A race's
distance is its rows' own; where a row has none, the race's first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core

PREFIX = "sl_"
SOURCE = ("GBGB results (position, wins, calculated time, sectional, weight: sources/gbgb.py) of the dog's earlier "
          "runs, against the same of every dog in today's race")
#: the least spread of each attribute within a field (seconds, places, wins, seconds, kg): the screen's, not fitted
FLOOR = {"lt": 0.05, "bt3": 0.05, "lp": 0.5, "w6": 0.5, "sec3": 0.02, "dw": 0.15}
SIGN = {"lt": -1.0, "bt3": -1.0, "lp": -1.0, "w6": 1.0, "sec3": -1.0, "dw": 1.0}       # higher is better after the sign
FORM = ("lt", "bt3", "lp", "w6", "sec3")       # the attributes the standout is read over
MIN_FIELD = 3                # a spread needs this many runners with the attribute
N_WINS = 6                   # the card's six lines
N_COURSE = 3                 # runs at today's track and distance read for the best time and the sectional
STANDOUT = 3                 # alone at the top of this many attributes
PACE = (0.050, 0.080)        # seconds a metre a greyhound's calculated time lies in (greyhound/pit/optics.py)
BIG = 1 << 21                # day numbers are below this


def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").to_numpy(float) if col in df else np.full(len(df), np.nan)


def _before(group: np.ndarray, dn: np.ndarray, order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For rows sorted by (group, day, time) (``order``): the sorted position of each row's group's last row on an
    EARLIER day, and whether there is one. Both in sorted order."""
    g, d = group[order], dn[order]
    key = g * BIG + d
    q = np.searchsorted(key, key, side="left") - 1
    ok = q >= 0
    ok[ok] = g[q[ok]] == g[ok]
    return np.where(ok, q, 0), ok


def _race_stats(x: np.ndarray, rc: np.ndarray, nr: int, floor: float):
    """Within each race, over the rows with a value: (value - median) / sd, (value - best) / sd, at the top with a
    lower value to compare, the lead over the best value under the top / sd. The sd is the sample sd, at least ``floor``, NaN under ``MIN_FIELD`` values."""
    n = len(x)
    ok = np.isfinite(x)
    cnt = np.bincount(rc[ok], minlength=nr).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.bincount(rc[ok], weights=x[ok], minlength=nr) / cnt
        ss = np.bincount(rc[ok], weights=(x[ok] - mean[rc[ok]]) ** 2, minlength=nr)
        sd = np.sqrt(ss / (cnt - 1.0))
    sd = np.where(cnt >= MIN_FIELD, np.maximum(sd, floor), np.nan)
    # the race's values in order: median, best and second best
    o = np.lexsort((np.where(ok, x, np.inf), rc))                   # by race, then value, the rows without one last
    start = np.r_[0, np.cumsum(np.bincount(rc, minlength=nr))[:-1]]
    k = cnt.astype(np.int64)
    xs = x[o]

    def at(j):                                                      # the race's j-th smallest value (NaN if none)
        good = (j >= 0) & (j < k)
        idx = np.clip(start + np.where(good, j, 0), 0, max(n - 1, 0))
        return np.where(good, xs[idx] if n else np.nan, np.nan)

    med = 0.5 * (at((k - 1) // 2) + at(k // 2))
    best = at(k - 1)
    below = ok & (x < best[rc])
    second = np.full(nr, -np.inf)                                   # the best value under the top one
    np.maximum.at(second, rc[below], x[below])
    second = np.where(np.isfinite(second), second, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        z_med = (x - med[rc]) / sd[rc]
        z_best = (x - best[rc]) / sd[rc]
        top = ok & (x == best[rc]) & (x > second[rc])
        lead = np.where(top, (x - second[rc]) / sd[rc], np.nan)
    return z_med, z_best, top, lead


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = runs
    n = len(df)
    card = df.is_card.fillna(False).to_numpy(bool) if "is_card" in df else np.zeros(n, bool)
    dog = pd.factorize(df.dog_id)[0].astype(np.int64)
    dn = (core.day(df) - pd.Timestamp("2000-01-01")).dt.days.to_numpy().astype(np.int64)
    tt = pd.to_datetime(df.t, errors="coerce").to_numpy().astype("datetime64[s]").astype(np.int64) if "t" in df \
        else dn.copy()
    rc = pd.factorize(df.race_id)[0]
    nr = int(rc.max()) + 1 if n else 0

    # ---- each run's own result (a card row has none), to be read on LATER days only
    pos = _num(df, "position")
    pos = np.where((pos >= 1) & ~card, pos, np.nan)
    won = np.where(pos == 1, 1.0, 0.0)
    wt = np.where(card, np.nan, _num(df, "weight_kg"))
    wt = np.where(wt > 0, wt, np.nan)
    dist = _num(df, "distance_m")
    time = _num(df, "adjusted_time")
    time = np.where((time > 0) & ~card, time, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        pace = time / dist
        odd = np.isfinite(pace) & ((pace < PACE[0]) | (pace > PACE[1]))
    time = np.where(odd, np.nan, time)
    sec = _num(df, "sectional")
    sec = np.where((sec > 0) & ~card, sec, np.nan)

    # the race's distance: the row's own, else the race's first (the card prints one distance a race)
    first_d = pd.Series(dist).groupby(rc).transform("first").to_numpy(float) if n else dist
    no_dist = ~np.isfinite(dist)
    dist = np.where(no_dist, first_d, dist)

    # ---- the dog's state after each run: last position, wins in six runs, the weight's change
    o = np.lexsort((tt, dn, dog))
    g = dog[o]
    new = np.r_[True, g[1:] != g[:-1]] if n else np.zeros(0, bool)
    idx = np.arange(n)
    head = np.maximum.accumulate(np.where(new, idx, 0)) if n else idx        # the dog's first sorted row
    cw = np.cumsum(won[o])
    lo = idx - N_WINS                                                        # the row before the six
    base = np.where(lo >= head, cw[np.maximum(lo, 0)], np.where(head > 0, cw[np.maximum(head - 1, 0)], 0.0))
    w6_s = cw - base
    w_s = wt[o]
    dw_s = np.where(new, np.nan, w_s - np.r_[np.nan, w_s[:-1]]) if n else w_s
    q, ok = _before(dog, dn, o)
    lp, w6, dw = (np.full(n, np.nan) for _ in range(3))
    lp[o] = np.where(ok, pos[o][q], np.nan)
    w6[o] = np.where(ok, w6_s[q], np.nan)
    dw[o] = np.where(ok, dw_s[q], np.nan)

    # ---- the course (dog, track, distance): the last time, the best of three, the mean sectional of three
    has_d = np.isfinite(dist)
    ck = pd.factorize(pd.MultiIndex.from_arrays([dog, df.track.astype(str).to_numpy(), np.where(has_d, dist, -1.0)]))[0]
    ck = ck.astype(np.int64)
    oc = np.lexsort((tt, dn, ck))
    gc = ck[oc]
    t_s, s_s = time[oc], sec[oc]
    T, Sx = [t_s], [s_s]
    for i in range(1, N_COURSE):
        same = np.r_[np.zeros(i, bool), gc[i:] == gc[:-i]] if n >= i else np.zeros(n, bool)
        T.append(np.where(same, np.r_[np.full(i, np.nan), t_s[:-i]] if n >= i else t_s, np.nan))
        Sx.append(np.where(same, np.r_[np.full(i, np.nan), s_s[:-i]] if n >= i else s_s, np.nan))
    T, Sx = np.vstack(T), np.vstack(Sx)
    bt_s = np.fmin.reduce(T, axis=0)
    cnt_s = np.isfinite(Sx).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sec_s = np.where(cnt_s > 0, np.nansum(Sx, axis=0) / np.maximum(cnt_s, 1), np.nan)
    qc, okc = _before(ck, dn, oc)
    okc = okc & has_d[oc]
    lt, bt3, sec3 = (np.full(n, np.nan) for _ in range(3))
    lt[oc] = np.where(okc, t_s[qc], np.nan)
    bt3[oc] = np.where(okc, bt_s[qc], np.nan)
    sec3[oc] = np.where(okc, sec_s[qc], np.nan)

    # ---- against today's field
    val = {"lt": lt, "bt3": bt3, "lp": lp, "w6": w6, "sec3": sec3, "dw": dw}
    Z, ZB, TOP, LEAD = {}, {}, {}, {}
    for a in FLOOR:
        Z[a], ZB[a], TOP[a], LEAD[a] = _race_stats(val[a] * SIGN[a], rc, nr, FLOOR[a])
    zc = np.vstack([Z[a] for a in FORM])
    some = np.isfinite(zc).any(axis=0)
    with np.errstate(invalid="ignore"):
        max_for = np.where(some, np.fmax.reduce(zc, axis=0), np.nan)
        max_against = np.where(some, np.fmin.reduce(zc, axis=0), np.nan)
    n_top = np.where(some, np.vstack([TOP[a] for a in FORM]).sum(axis=0).astype(float), np.nan)
    lead = np.fmax.reduce(np.vstack([LEAD[a] for a in FORM]), axis=0)
    lead = np.where(some, np.nan_to_num(lead), np.nan)
    mx = np.full(nr, -np.inf)
    np.maximum.at(mx, rc[some], n_top[some])
    with_mx = np.bincount(rc[some & (n_top == mx[rc])], minlength=nr)
    one = (mx >= STANDOUT) & (with_mx == 1)
    standout = np.where(np.isfinite(mx)[rc], (one[rc] & (n_top == mx[rc])).astype(float), np.nan)

    out = pd.DataFrame({"race_id": df.race_id.to_numpy(), "dog_id": df.dog_id.to_numpy()})
    out[f"{PREFIX}lt_med"] = Z["lt"]
    out[f"{PREFIX}lt_best"] = ZB["lt"]
    out[f"{PREFIX}bt3_med"] = Z["bt3"]
    out[f"{PREFIX}bt3_best"] = ZB["bt3"]
    out[f"{PREFIX}lp_med"] = Z["lp"]
    out[f"{PREFIX}w6_med"] = Z["w6"]
    out[f"{PREFIX}sec3_med"] = Z["sec3"]
    out[f"{PREFIX}sec3_best"] = ZB["sec3"]
    out[f"{PREFIX}dw_med"] = Z["dw"]
    out[f"{PREFIX}dw_abs"] = np.abs(Z["dw"])
    out[f"{PREFIX}max_for"] = max_for
    out[f"{PREFIX}max_against"] = max_against
    out[f"{PREFIX}n_top"] = n_top
    out[f"{PREFIX}lead"] = lead
    out[f"{PREFIX}is_standout"] = standout
    cols = [c for c in out.columns if c.startswith(PREFIX)]
    out[cols] = out[cols].astype(np.float32)

    def share(m) -> float | None:
        return round(float(np.mean(m)), 4) if n else None

    notes = {"rows": int(n), "races": int(nr), "floor": FLOOR, "min_field": MIN_FIELD, "standout": STANDOUT,
             "rows_with_an_earlier_run": share(ok[np.argsort(o)]) if n else None,
             "rows_with_a_run_at_todays_course": share(np.isfinite(lt)),
             "rows_with_any_form_attribute": share(some),
             "rows_at_the_top_of_one": share(np.nan_to_num(n_top) >= 1),
             "races_with_a_standout": round(float(one.mean()), 4) if nr else None,
             "rows_with_no_distance_given_the_races": int((no_dist & has_d).sum()),
             "rows_with_no_distance": int((~has_d).sum()),
             "times_no_greyhound_runs_left_out": int(odd.sum()),
             "non_null": {c: share(out[c].notna()) for c in cols}}
    return out, notes


_VS = "over the field's spread of it (sample sd, floored; NaN with fewer than 3 runners that have one)"
COLUMNS = {
    f"{PREFIX}lt_med": f"the calculated time of its last run at today's track and distance, against the field's median, {_VS}; above 0 faster",
    f"{PREFIX}lt_best": "the same against the field's fastest (0 for the fastest, below 0 behind it)",
    f"{PREFIX}bt3_med": f"its fastest calculated time in its last {N_COURSE} runs at today's track and distance, against the field's median, {_VS}; above 0 faster",
    f"{PREFIX}bt3_best": "the same against the field's fastest (0 for the time-finder's pick)",
    f"{PREFIX}lp_med": f"its last finishing position against the field's median, {_VS}; above 0 a better position",
    f"{PREFIX}w6_med": f"its wins in its last {N_WINS} runs against the field's median, {_VS}",
    f"{PREFIX}sec3_med": f"its mean sectional over its last {N_COURSE} runs at today's track and distance, against the field's median, {_VS}; above 0 faster early",
    f"{PREFIX}sec3_best": "the same against the field's fastest sectional (0 for the fastest)",
    f"{PREFIX}dw_med": f"its weight change at its last run (that run's weight less the run before's, kg), against the field's median, {_VS}; today's weigh-in is not read",
    f"{PREFIX}dw_abs": "the size of that, either way: an unusual weight move for this field",
    f"{PREFIX}max_for": "the largest standout for it: the highest of its five form attributes (last time, best time, last position, wins, sectional) against the field's median",
    f"{PREFIX}max_against": "the largest standout against it: the lowest of the five",
    f"{PREFIX}n_top": "how many of the five it is at the top of (the field's best value, shared or not, with a lower value in the field to compare)",
    f"{PREFIX}lead": "its largest lead over the field's best value under the top among those it tops, in spreads (0 when it tops none)",
    f"{PREFIX}is_standout": f"1 when it is at the top of at least {STANDOUT} of the five and no other runner tops as many",
}
