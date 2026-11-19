"""The card-reader: one fitted reading of the printed card in place of the optics (op_) and salience (sl_) looks
(reports/greyhound_feature_review.md, "optics, salience": they overlap on 3 of 5 card attributes and weigh them
equally; one card table with a fitted card-reader).

**The card** (each a row's dog before today's DAY; a line is a dog-day with a finishing position, read by that day's
last run; nothing of today but what the card prints: track, distance, grade, trap):

- form: the finishing positions of its last three lines, the mean position over its last ``N_CARD`` lines;
- wins: wins in its last ``N_CARD`` lines;
- time: its fastest and its latest calculated time at today's track and distance among those lines (lengths; a time
  outside ``PACE`` seconds a metre is a keying error and is not read);
- sectional: its mean sectional at today's track and distance among those lines, and the mean of its sectional
  rank within each of its last three lines' races (1 the fastest away, as the card prints it);
- grade: today's grade number less its last line's (same grade family only; above 0 down in class);
- trap: today's trap (one indicator a trap, 2 to 6);
- rest: log days since its last line, and a layoff indicator (over ``LAYOFF`` days);
- experience: its lines (0 to ``N_CARD``), a debut indicator.

Each continuous attribute is centred on today's field (the runners that have it) and a runner without it is set to
the field's mean with a missing indicator (race-centred too): a conditional logit reads only differences within a
race.

**The reader** (review: "a fitted card-reader"). A conditional logit (``core.clogit_fit``) of each runner's
de-biased SP chance (``core.debiased_probs``, Shin) on those attributes, the soft target the card's crowd prices.
Refitted walk-forward each ``FREQ`` period (yearly) on the races before the period's first day (at most the
``MAX_FIT_RACES`` latest; races with every runner priced and at least 3 runners; none under ``MIN_FIT_RACES``,
then the period's rows are NaN). The same refit, on the expected-time table's ``xt_tot`` (centred, with its
missing indicator), is the ability reader: the dog's ability on the same scale.

- ``cr_share``: the card's implied chance (what a punter reading the card would price), ``cr_rating`` its utility
  less the field's mean, ``cr_rank`` its rank in the field (1 the card's pick);
- ``cr_ab_share``: the chance its ability (xt_tot, from earlier days) implies; ``cr_surplus`` = log ``cr_ab_share``
  less log ``cr_share`` (above 0 the dog is better than its card reads: the reverse sign of ``op_surplus``);
- ``cr_for`` / ``cr_against``: the largest and smallest contribution of one attribute group (form, wins, time,
  sectional, grade, trap, rest, experience: beta x the centred values, summed in the group) to its utility, and
  ``cr_for_attr`` / ``cr_against_attr`` the group's code (``GROUPS`` order, 0 form ... 7 experience).

Point in time: the attributes read earlier days; the fits read races dated before the period's first day; a card row
(``is_card``) reads what its run would. Every column is card-safe. The ability side inherits the expected-time
table's own caveat (optics.py: its filter constants on a store that starts after 2018).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core

PREFIX = "cr_"
SOURCE = ("GBGB results (positions, wins, calculated times, sectionals, grade, trap: sources/gbgb.py) of the dog's last "
          "six race days and today's card; the de-biased SP chance as the reader's target; xt_tot (the expected-time "
          "table) as the ability")
DEPENDS = ("xtime",)
N_CARD = 6
LAYOFF = 28                  # days
PACE = (0.050, 0.080)        # seconds a metre a greyhound's time lies in (optics.py's bounds; fixed, not fitted)
FREQ = "Y"                   # the refit period
MIN_FIT_RACES = 500
MAX_FIT_RACES = 150_000
MIN_FIELD = 3
LAM = 1.0
BIG = 1 << 21
GROUPS = ("form", "wins", "time", "sectional", "grade", "trap", "rest", "experience")


def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").to_numpy(float) if col in df else np.full(len(df), np.nan)


def _centre(v: np.ndarray, rc: np.ndarray, nr: int) -> np.ndarray:
    """``v`` less its race's mean over the runners that have it (NaN stays NaN)."""
    ok = np.isfinite(v)
    n = np.bincount(rc[ok], minlength=nr).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.bincount(rc[ok], weights=v[ok], minlength=nr) / n
    return v - mean[rc]


def attributes(runs: pd.DataFrame) -> tuple[dict, dict]:
    """The printed card's raw attributes a row (from earlier days and today's card), and counts for the notes."""
    n = len(runs)
    card = runs.is_card.fillna(False).to_numpy(bool) if "is_card" in runs else np.zeros(n, bool)
    dog = pd.factorize(runs.dog_id)[0].astype(np.int64)
    dn = (core.day(runs) - pd.Timestamp("2000-01-01")).dt.days.to_numpy().astype(np.int64)
    tt = pd.to_datetime(runs.t, errors="coerce").to_numpy().astype("datetime64[s]").astype(np.int64) if "t" in runs \
        else dn.copy()
    rc = pd.factorize(runs.race_id)[0]
    pos = _num(runs, "position")
    pos = np.where((pos >= 1) & ~card, pos, np.nan)
    dist = _num(runs, "distance_m")
    time = _num(runs, "adjusted_time")
    time = np.where((time > 0) & ~card, time, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        pace = time / dist
    time = np.where(np.isfinite(pace) & (pace >= PACE[0]) & (pace <= PACE[1]), time, np.nan)
    sec = _num(runs, "sectional")
    sec = np.where((sec > 0) & ~card, sec, np.nan)
    srank = pd.Series(sec).groupby(rc).rank(method="min").to_numpy(float)          # 1 the fastest away
    grade = _num(runs, "grade")
    fam = runs.grade_family.astype(object).to_numpy() if "grade_family" in runs else np.full(n, None, object)
    td = pd.factorize(pd.MultiIndex.from_arrays([runs.track.astype(str).to_numpy(), np.nan_to_num(dist, nan=-1.0)]))[0]

    # the lines: one a dog-day with a position (that day's last run), sorted by dog and time
    li = np.flatnonzero(np.isfinite(pos))
    li = li[np.lexsort((tt[li], dn[li], dog[li]))]
    key = dog[li] * BIG + dn[li]
    last = np.r_[key[1:] != key[:-1], True] if len(li) else np.zeros(0, bool)
    li, key = li[last], key[last]
    l_dog = dog[li]
    first = np.searchsorted(key, dog * BIG + dn, side="left")          # the dog's lines before today lie below this

    a = {k: np.full(n, np.nan) for k in ("pos1", "pos2", "pos3", "best", "ltime", "srank3", "gchg", "days")}
    lines = np.zeros(n)
    psum, wins = np.zeros(n), np.zeros(n)
    ssum, sn, rsum, rn = (np.zeros(n) for _ in range(4))
    for lag in range(1, N_CARD + 1):
        q = first - lag
        ok = q >= 0
        ok[ok] = l_dog[q[ok]] == dog[ok]
        src = li[np.where(ok, q, 0)] if len(li) else np.zeros(n, dtype=np.int64)
        lines += ok
        p = np.where(ok, pos[src], np.nan)
        psum += np.nan_to_num(p)
        wins += np.where(ok, (pos[src] == 1).astype(float), 0.0)
        here = ok & (td[src] == td)
        t_l = np.where(here, time[src], np.nan)
        a["best"] = np.fmin(a["best"], t_l)
        a["ltime"] = np.where(np.isnan(a["ltime"]), t_l, a["ltime"])         # the most recent line here first
        s_l = np.where(here, sec[src], np.nan)
        ssum += np.nan_to_num(s_l)
        sn += np.isfinite(s_l)
        if lag <= 3:
            a[f"pos{lag}"] = p
            r_l = np.where(ok, srank[src], np.nan)
            rsum += np.nan_to_num(r_l)
            rn += np.isfinite(r_l)
        if lag == 1:
            with np.errstate(invalid="ignore"):
                same = ok & (fam[src] == fam) & pd.notna(fam)
            a["gchg"] = np.where(same, grade - grade[src], np.nan)
            a["days"] = np.where(ok, (dn - dn[src]).astype(float), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        a["pos6"] = np.where(lines > 0, psum / np.maximum(lines, 1), np.nan)
        a["sec"] = np.where(sn > 0, ssum / np.maximum(sn, 1), np.nan)
        a["srank3"] = np.where(rn > 0, rsum / np.maximum(rn, 1), np.nan)
    a["wins6"] = np.where(lines > 0, wins, np.nan)
    a["lines"] = lines
    a["trap"] = _num(runs, "trap")
    return a, {"lines": int(len(li)), "rows_with_a_line": round(float((lines > 0).mean()), 4) if n else None}


def design(a: dict, rc: np.ndarray, nr: int) -> tuple[np.ndarray, list[str], list[int]]:
    """The reader's design matrix: each attribute centred on the field, a missing one at the field's mean with a
    (centred) indicator; the trap indicators; each column's group (``GROUPS``)."""
    cols, names, grp = [], [], []

    def add(v, name, g, flag=True):
        c = _centre(v, rc, nr)
        cols.append(np.nan_to_num(c))
        names.append(name)
        grp.append(g)
        if flag:                                    # always, so the design is the same on any frame
            cols.append(_centre((~np.isfinite(v)).astype(float), rc, nr))
            names.append(name + "_na")
            grp.append(g)

    L = core.LENGTH_S
    for k in ("pos1", "pos2", "pos3", "pos6"):
        add(a[k], k, 0)
    add(a["wins6"], "wins6", 1, flag=False)
    add(-a["best"] / L, "best", 2)
    add(-a["ltime"] / L, "ltime", 2)
    add(-a["sec"] / L, "sec", 3)
    add(a["srank3"], "srank3", 3)
    add(a["gchg"], "gchg", 4)
    for tr in range(2, 7):
        add((a["trap"] == tr).astype(float), f"trap{tr}", 5, flag=False)
    with np.errstate(invalid="ignore", divide="ignore"):
        add(np.log1p(a["days"]), "log_days", 6, flag=False)
        add((a["days"] > LAYOFF).astype(float), "layoff", 6, flag=False)
    add(a["lines"], "lines", 7, flag=False)
    add((a["lines"] == 0).astype(float), "debut", 7, flag=False)
    return np.column_stack(cols).astype(np.float32), names, grp


def walk_forward(X: np.ndarray, y: np.ndarray, rc: np.ndarray, per: np.ndarray, fit_race: np.ndarray,
                 race_per: np.ndarray, race_rank: np.ndarray) -> tuple[np.ndarray, dict]:
    """For each period: beta fitted on the races of earlier periods (``fit_race``, the latest ``MAX_FIT_RACES``),
    the utilities X beta for the period's rows (NaN with too few races). ``race_rank``: the race's order in time."""
    u = np.full(len(rc), np.nan)
    fits = {}
    for p in np.unique(per):
        ok_r = fit_race & (race_per < p)
        rid = np.flatnonzero(ok_r)
        if len(rid) < MIN_FIT_RACES:
            fits[int(p)] = None
            continue
        rid = rid[np.argsort(race_rank[rid])][-MAX_FIT_RACES:]
        use = np.zeros(len(fit_race), bool)
        use[rid] = True
        rows = use[rc]
        Xf = X[rows].astype(float)
        sd = Xf.std(axis=0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        gi = pd.factorize(rc[rows])[0]
        b = core.clogit_fit(Xf / sd, gi, y[rows], lam=LAM)
        beta = b / sd
        at = per == p
        u[at] = X[at] @ beta
        fits[int(p)] = {"races": int(len(rid)), "beta": beta}
    return u, fits


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    n = len(runs)
    ctx = ctx or {}
    rc = pd.factorize(runs.race_id)[0]
    nr = int(rc.max()) + 1 if n else 0
    card = runs.is_card.fillna(False).to_numpy(bool) if "is_card" in runs else np.zeros(n, bool)
    a, cnt = attributes(runs)
    X, names, grp = design(a, rc, nr)

    xt = ctx.get("xtime")
    if xt is None:
        from greyhound.pit import xtime
        xt, _ = xtime.build(runs, figs if figs is not None else core.per_run_figures(runs))
    xt_tot = runs[["race_id", "dog_id"]].merge(
        xt[["race_id", "dog_id", "xt_tot"]].drop_duplicates(["race_id", "dog_id"]), on=["race_id", "dog_id"],
        how="left").xt_tot.to_numpy(float)
    xa = _centre(xt_tot, rc, nr)
    XA = np.column_stack([np.nan_to_num(xa), _centre((~np.isfinite(xt_tot)).astype(float), rc, nr)])

    # the target: the de-biased SP chance; a race is fitted when every runner is priced, none is a card row
    y = core.debiased_probs(_num(runs, "sp_decimal"), runs.race_id.to_numpy())
    field = np.bincount(rc, minlength=nr)
    priced = np.bincount(rc, weights=(np.isfinite(y) & ~card).astype(float), minlength=nr)
    fit_race = (priced == field) & (field >= MIN_FIELD)
    y = np.nan_to_num(y)

    d = core.day(runs)
    per = d.dt.to_period(FREQ).astype("int64").to_numpy() if n else np.zeros(0, np.int64)
    race_per = np.zeros(nr, np.int64)
    race_per[rc] = per
    race_rank = np.zeros(nr)
    race_rank[rc] = pd.to_datetime(runs.t if "t" in runs else d, errors="coerce").to_numpy().astype("datetime64[s]") \
        .astype(np.int64)
    u, fits = walk_forward(X, y, rc, per, fit_race, race_per, race_rank)
    ua, fits_a = walk_forward(XA, y, rc, per, fit_race, race_per, race_rank)

    share = core.group_softmax(np.nan_to_num(u), rc) if n else np.zeros(0)
    share = np.where(np.isfinite(u), share, np.nan)
    ab = core.group_softmax(np.nan_to_num(ua), rc) if n else np.zeros(0)
    ab = np.where(np.isfinite(ua), ab, np.nan)
    rating = _centre(u, rc, nr)
    rank = pd.Series(-u).groupby(rc).rank(method="min").to_numpy(float)

    # each attribute group's contribution: beta x the centred values, summed in the group
    gl = np.asarray(grp)
    C = np.full((n, len(GROUPS)), np.nan)
    per_beta = {p: f["beta"] for p, f in fits.items() if f is not None}
    for p, beta in per_beta.items():
        at = per == p
        contrib = X[at] * beta
        C[at] = np.column_stack([contrib[:, gl == g].sum(axis=1) for g in range(len(GROUPS))])
    fitted = np.isfinite(u)
    Cf = np.where(np.isfinite(C), C, 0.0)
    for_attr = np.where(fitted, Cf.argmax(axis=1), np.nan)
    ag_attr = np.where(fitted, Cf.argmin(axis=1), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        surplus = np.log(ab) - np.log(share)

    out = pd.DataFrame({"race_id": runs.race_id.to_numpy(), "dog_id": runs.dog_id.to_numpy()})
    out[f"{PREFIX}share"] = share
    out[f"{PREFIX}rating"] = rating
    out[f"{PREFIX}rank"] = rank
    out[f"{PREFIX}ab_share"] = ab
    out[f"{PREFIX}surplus"] = surplus
    out[f"{PREFIX}for"] = np.where(fitted, Cf.max(axis=1), np.nan)
    out[f"{PREFIX}for_attr"] = for_attr
    out[f"{PREFIX}against"] = np.where(fitted, Cf.min(axis=1), np.nan)
    out[f"{PREFIX}against_attr"] = ag_attr
    cols = [c for c in out.columns if c.startswith(PREFIX)]
    out[cols] = out[cols].astype(np.float32)
    assert cols == list(COLUMNS)

    def fmt(f, nm):
        if f is None:
            return None
        return {"races": f["races"], "beta": {k: round(float(v), 5) for k, v in zip(nm, f["beta"])}}

    notes = {"rows": int(n), "card_rows": int(card.sum()), **cnt, "freq": FREQ, "min_fit_races": MIN_FIT_RACES,
             "max_fit_races": MAX_FIT_RACES, "races_fittable": int(fit_race.sum()),
             "rows_with_a_reading": round(float(fitted.mean()), 4) if n else None,
             "rows_with_an_ability_reading": round(float(np.isfinite(ua).mean()), 4) if n else None,
             "groups": list(GROUPS), "card_safe": cols,
             "fits": {str(pd.Period(ordinal=p, freq=FREQ)): fmt(f, names)
                      for p, f in fits.items()},
             "ability_fits": {str(pd.Period(ordinal=p, freq=FREQ)): fmt(f, ["xt_tot", "xt_tot_na"]) for p, f in fits_a.items()},
             "target": "core.debiased_probs(sp_decimal, method='shin')",
             "refit": f"each {FREQ} period on the races dated before its first day"}
    return out, notes


COLUMNS = {
    f"{PREFIX}share": "the card's implied chance: a conditional logit of the de-biased SP chance on the printed card "
                      "(form, wins, times and sectional at today's track and trip, sectional ranks, grade move, trap, "
                      "days off, lines), refitted yearly on earlier years",
    f"{PREFIX}rating": "the card-reader's utility less the field's mean (log-odds scale)",
    f"{PREFIX}rank": "its rank in the field by the card-reader (1 the card's pick)",
    f"{PREFIX}ab_share": "the chance its ability implies: the same refit on the expected-time table's xt_tot",
    f"{PREFIX}surplus": "log cr_ab_share less log cr_share (above 0 better than its card reads)",
    f"{PREFIX}for": "the largest attribute group's contribution to its card utility against the field (beta x centred)",
    f"{PREFIX}for_attr": "that group's code: 0 form, 1 wins, 2 time, 3 sectional, 4 grade, 5 trap, 6 rest, 7 experience",
    f"{PREFIX}against": "the smallest attribute group's contribution (the largest standout against it)",
    f"{PREFIX}against_attr": "that group's code (as cr_for_attr)",
}
