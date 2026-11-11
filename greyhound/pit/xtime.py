"""The expected time of each dog in each race (the owner, 11 Nov 2026: "a specific Expected Time model per dog per
race. Adjust based on priors, features, track adjusted, going adjusted, distance adjusted, age, recency, days off").

The review of 11 Nov (reports/greyhound_model.md, section 41) found that every time figure in the model rests on a
track-and-distance standard made of winners' times, with no link between tracks, one meeting variant in seconds from
the 10 to 14 winners of the night, and no mean effect of trap, age, days off, a change of trip or a first run at a
track on the time. This table is one model of the time itself, read before the race:

    100 x ln(time) = standard(track, distance, hurdles) + variant(meeting)
                     + the dog's level + its preference (this track and trip; this distance band)
                     + trap(track, distance, trap) + grade(track, distance, grade)
                     + x'b (trial, field size, age by sex, days off by sex, runs since a layoff by sex, experience,
                            first run at the track or trip, a change of track, a change of trip, last weight against
                            the dog's own mean)
                     + trouble(the run's comment codes) + noise

in percent of the time, so a level carries from one trip to another, and a variant scales with the distance run. It
is fitted walking forward, a day at a time:

- before a day's races every runner is priced from the state after earlier days (the columns of this table);
- after them, each run's error against its own expected time gives the **meeting variant** (the mean error of every
  runner of the meeting, not the winners only, shrunk by n / (n + k), k the within over between variance of earlier
  meetings), and the error less the variant and less the run's trouble cost updates the dog's level (a Kalman filter:
  drift a day off, noise a run and the spread of a new dog fitted on the early years, more for a trial or a troubled run, an outlier clipped at 3 sd), the standard (a
  slow running mean, so a track linked to the others by the dogs that move), and the trap, grade and preference
  effects (shrunk running means of the partial residual);
- ``b`` and the price of each interference code (crowded, bumped, baulked, checked, hampered: never a code that
  says where the dog ran, which is the run itself) are a ridge regression refitted every ``REFIT_MONTHS`` on the periods
  before, on the partial residuals those periods stored: a row is never priced by a fit that saw its period;
- a dog with no run has the mean level of the dogs it is graded to meet (the racing manager's opinion of it), else
  the mean level of its grade there on earlier days.

Trials (classes T, T1 to T4) are read as runs when the build is given them (``ctx["trial_runs"]``: the rows
``data.clean`` drops because GBGB serves a trial with no off time): each is a noisy observation of the dog's level
with its own trial and field-size adjustments, so a debutant comes to its first race with a level from its trials. A
trial moves no standard, no trap or grade effect, and is never a row of the table.

Three times are modelled the same way, each with its own states: the whole race (GBGB's calculated time), the
sectional (to the line the first time: early pace) and the run home (run time less sectional). Handicap races
(staggered starts) and a fall are read by none; hurdles have their own standard.

The experts' view (Timeform's expected time against the field, early pace first, the trap against the neighbours'
pace: section 41) is the race-relative columns: lengths against the field on each time, the expected sectional
against the dogs drawn inside and outside, the expected leader's margin.

Every column is read from earlier days only (tests/test_greyhound_pit.py; tests/test_greyhound_xtime.py).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from greyhound.pit import core, figs as figs_mod, form, trouble

PREFIX = "xt_"
SOURCE = ("GBGB results (sources/gbgb.py): calculated time, run time, sectional, grade, trap, born, sex, weight, "
          "comment")
FIT_BEFORE = "2018-01-01"
REFIT_MONTHS = 12
MIN_FIT = 50_000
MIN_COUNT = 200
LAM = 20.0
OUT = 8.0                    # a run more than 8% from its expected time is read by nothing
CLIP = 4.0
STD_CAP = 800.0              # the standard's memory, runs
K_TRAP, K_GRADE, K_PREF, K_VAR0 = 50.0, 30.0, 5.0, 5.0
NOISY = 1.5
TARGETS = ("tot", "sec", "run")
# the second build (run 43058158856) chose r 1.6 (the grid's top) and q 0.001 (its bottom) on all three times: the
# grid reaches further
GRID = [(r, q, p0) for r in (0.8, 1.6, 3.2, 6.4) for q in (0.0003, 0.001, 0.003) for p0 in (0.1, 0.3, 1.0)]
#: the comment codes that are interference (crowded, bumped, baulked, checked, hampered, impeded, stumbled). The first
#: build on the real data (run 43055884883, 11 Nov 2026) priced every code: "always led" came out at -3.9% of the time
#: and the model's error grew each year, because leading is what running fast looks like, not luck. Only interference
#: is taken out of a past run, it can only have cost time, and never more than ``COST_CAP`` percent
#: Until 11 Nov 2026 the pattern read "FcdToCk" but not GBGB's commonest spellings of forced to check ("FcdTCk",
#: "FcdCk", "ForcedToCheck", "FcdToBCk"), nor "Stb" (stumbled), "HitRls", "StruckInto": 19,662 of 198,439
#: interference tokens in 1 Jul 2025 to 5 Nov 2026 and 13,504 runs whose only interference went unread (the review
#: of the fear table, 11 Nov 2026)
TROUBLE = re.compile(r"crd|crowd|bmp|bump|blk|baulk|balk|hmp|hamp|imped|^imp\d|stumb|stb|fcd-?t?o?-?b?-?c(hec)?k"
                     r"|forcedt?o?b?c(hec)?k|^ck|checked|brtdn|broughtdown|knockedover|hitrls|struckinto")
COST_CAP = 1.5
BANDS = (350.0, 550.0, 750.0)
OFF_EDGES = (5, 8, 11, 15, 22, 36, 61, 101, 181)
EXP_EDGES = (1, 2, 3, 5, 10, 20, 40)


# ---- what is known before the race ---------------------------------------------------------------------------------

def _codes(*parts) -> np.ndarray:
    key = parts[0].astype(str)
    for p in parts[1:]:
        key = key + "|" + p.astype(str)
    return pd.factorize(key)[0]


def prepare(df: pd.DataFrame) -> dict:
    """The keys and the pre-race covariates of every row (``df`` in time order)."""
    n = len(df)
    dn = (core.day(df) - pd.Timestamp("2000-01-01")).dt.days.to_numpy()
    dist = pd.to_numeric(df.distance_m, errors="coerce")
    # hurdles: GBGB's race type, or the class H. Until 11 Nov 2026 the key read the class HP - the handicap, whose
    # times are not read at all (``targets``) - and the real hurdle races, classed OR, shared the flat standard. A
    # card does not know the race type: a hurdle race on the card is read as flat (22 races in 16 months)
    hurdle = df.hurdles.fillna(False).astype(bool) if "hurdles" in df else pd.Series(False, index=df.index)
    if "grade_family" in df:
        hurdle = hurdle | (df.grade_family == "H")
    track = df.track.astype(str)
    key_s = track + "|" + dist.round().astype("Int64").astype(str) + "|" + hurdle.astype(int).astype(str)
    trap = pd.to_numeric(df.trap, errors="coerce").fillna(0).astype(int).astype(str)
    cls = df.race_class.astype(str)
    band = pd.Series(np.digitize(dist.fillna(480.0), BANDS), index=df.index)
    d = df.assign(_dn=dn.astype(float), _key=key_s, _one=1.0, _dist=dist)
    prev = core.prev_day_values(d, "dog_id", ["_dn", "_dist", "track"])
    career = core.prior_window(d, ["dog_id"], ["_one"], None).n.to_numpy()
    at_track = core.prior_window(d, ["dog_id", "track"], ["_one"], None).n.to_numpy()
    at_key = core.prior_window(d, ["dog_id", "_key"], ["_one"], None).n.to_numpy()
    off = dn - prev._dn.to_numpy(float)
    # runs since a layoff of 28 days or more (1: the first run back), a day's runs counted once
    per = pd.DataFrame({"dog": df.dog_id.to_numpy(), "dn": dn}).drop_duplicates().sort_values(["dog", "dn"])
    gap = per.groupby("dog").dn.diff()
    block = (gap.isna() | (gap >= 28)).groupby(per.dog).cumsum()
    per["back"] = per.groupby([per.dog, block]).cumcount() + 1
    back = pd.DataFrame({"dog": df.dog_id.to_numpy(), "dn": dn}).merge(per, on=["dog", "dn"], how="left").back.to_numpy()
    w = pd.to_numeric(df.weight_kg, errors="coerce") if "weight_kg" in df else pd.Series(np.nan, index=df.index)
    w = w.where((w >= 15) & (w <= 50))
    d2 = d.assign(_w=w.fillna(0.0), _wn=w.notna().astype(float), _wv=w)
    ws = core.prior_window(d2, ["dog_id"], ["_w", "_wn"], None)
    wprev = core.prev_day_values(d2, "dog_id", ["_wv"])._wv.to_numpy(float)
    wdiff = wprev - (ws._w / ws._wn.replace(0, np.nan)).to_numpy(float)
    age = form._age_months(df).to_numpy(float)
    bitch = (form._bitch(df).to_numpy() >= 0.5).astype(int)
    trial = df.is_trial.fillna(False).to_numpy(bool) if "is_trial" in df else np.zeros(n, bool)
    field = pd.to_numeric(df.field, errors="coerce").fillna(6).clip(1, 6).to_numpy(int) if "field" in df else \
        np.full(n, 6)
    debut = career == 0
    # the design: one indicator a level (the ridge and the intercept settle the baseline)
    rows, cols, names = [], [], []

    def add(name: str, code: np.ndarray, labels: list[str], skip: int | None = None):
        for lv, lab in enumerate(labels):
            if lv == skip:
                continue
            r = np.flatnonzero(code == lv)
            rows.append(r)
            cols.append(np.full(len(r), len(names)))
            names.append(f"{name}:{lab}")

    add("trial", trial.astype(int), ["no", "yes"], skip=0)
    add("field", field - 1, [str(k) for k in range(1, 7)], skip=5)
    a = np.where(np.isfinite(age), np.clip(np.floor((age - 12) / 3), 0, 18), 19).astype(int)
    add("age", a * 2 + bitch, [f"{12 + 3 * k if k < 19 else 'unknown'}m {'bitch' if b else 'dog'}"
                              for k in range(20) for b in (0, 1)])
    o = np.where(debut, len(OFF_EDGES) + 1, np.digitize(np.nan_to_num(off, nan=7.0), OFF_EDGES))
    off_lab = [f"<{OFF_EDGES[0]}d"] + [f"{lo}-{hi - 1}d" for lo, hi in zip(OFF_EDGES[:-1], OFF_EDGES[1:])] + \
              [f"{OFF_EDGES[-1]}d+", "debut"]
    add("off", o * 2 + bitch, [f"{lab} {'bitch' if b else 'dog'}" for lab in off_lab for b in (0, 1)])
    bk = np.where(debut, -1, np.clip(np.nan_to_num(back, nan=6.0), 1, 6) - 1).astype(int)
    add("back", np.where(bk >= 0, bk * 2 + bitch, -1), [f"run {k} {'bitch' if b else 'dog'}" for k in range(1, 7)
                                                        for b in (0, 1)], skip=None)
    add("runs", np.digitize(career, EXP_EDGES), ["0"] + [f"{lo}-{hi - 1}" for lo, hi in zip(EXP_EDGES[:-1], EXP_EDGES[1:])]
        + [f"{EXP_EDGES[-1]}+"], skip=len(EXP_EDGES))
    add("first_track", (~debut & (at_track == 0)).astype(int), ["no", "yes"], skip=0)
    add("first_trip", (~debut & (at_key == 0)).astype(int), ["no", "yes"], skip=0)
    switch = ~debut & (prev.track.astype(str).to_numpy() != track.to_numpy())
    add("switch", switch.astype(int), ["no", "yes"], skip=0)
    ratio = dist.to_numpy(float) / prev._dist.to_numpy(float)
    add("trip", np.where(np.isfinite(ratio), np.digitize(ratio, (0.8, 0.95, 1.05, 1.25)), 2),
        ["much shorter", "shorter", "same", "longer", "much longer"], skip=2)
    add("weight", np.where(np.isfinite(wdiff), np.digitize(wdiff, (-1.0, -0.4, 0.4, 1.0)), 5),
        ["<-1kg", "-1..-0.4kg", "level", "0.4..1kg", ">1kg", "unknown"], skip=2)
    from scipy import sparse
    X = sparse.csr_matrix((np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, len(names)))
    # the comment codes (the run's own result: they price a past run, never today's)
    toks = [trouble.codes(c) for c in df.comment.fillna("").to_numpy()] if "comment" in df else [[] for _ in range(n)]
    vocab = {}
    cr, cc = [], []
    for i, ts in enumerate(toks):
        for c in set(ts):
            if not TROUBLE.search(c):
                continue
            cr.append(i)
            cc.append(vocab.setdefault(c, len(vocab)))
    C = sparse.csr_matrix((np.ones(len(cr)), (cr, cc)), shape=(n, max(len(vocab), 1)))
    day = core.day(df)
    period = ((day.dt.year * 12 + day.dt.month - 1) // REFIT_MONTHS).to_numpy()
    order = np.argsort(dn, kind="stable")
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(dn[order])) + 1, [n]])
    return {"n": n, "dn": dn, "dog": pd.factorize(df.dog_id)[0], "key": pd.factorize(key_s)[0],
            "kt": _codes(key_s, trap), "kc": _codes(key_s, cls), "pk": _codes(df.dog_id, key_s),
            "pb": _codes(df.dog_id, band), "race": pd.factorize(df.race_id)[0],
            "meeting": pd.factorize(df.meeting_id)[0] if "meeting_id" in df else _codes(track, pd.Series(dn)),
            "trial": trial, "X": X, "names": names, "C": C, "vocab": vocab, "period": period, "order": order,
            "bounds": bounds}


def targets(df: pd.DataFrame) -> dict:
    """100 x ln(seconds) of the three times; NaN for a fall, a handicap race, a card row or a missing clock."""
    bad = df.comment.fillna("").str.contains(r"\(Handslip\)|Fell|Fll|BrkLine", case=False) if "comment" in df else \
        pd.Series(False, index=df.index)
    hcp = df.handicap.fillna(False).astype(bool) if "handicap" in df else pd.Series(False, index=df.index)
    sec = pd.to_numeric(df.sectional, errors="coerce")
    t = {"tot": pd.to_numeric(df.adjusted_time, errors="coerce"), "sec": sec,
         "run": pd.to_numeric(df.run_time, errors="coerce") - sec}
    return {k: (100.0 * np.log(v.where((v > 0) & ~bad & ~hcp))).to_numpy(float) for k, v in t.items()}


# ---- the fits --------------------------------------------------------------------------------------------------------

def ridge(X, C, y: np.ndarray, rows: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
    """A ridge of ``y`` on the covariates and the comment codes seen ``MIN_COUNT`` times in ``rows``: the intercept,
    the covariates' effects, the codes' prices (0 for a rare code), and the mean price a run (so a clean time is a
    time at the average trouble)."""
    from scipy import sparse
    Xr, Cr = X[rows], C[rows]
    keep = np.flatnonzero(np.asarray(Cr.sum(axis=0)).ravel() >= MIN_COUNT)
    Z = sparse.hstack([sparse.csr_matrix(np.ones((len(rows), 1))), Xr, Cr[:, keep]], format="csr")
    reg = LAM * np.eye(Z.shape[1])
    reg[0, 0] = 0.0
    coef = np.linalg.solve((Z.T @ Z).toarray() + reg, Z.T @ y)
    k = X.shape[1]
    gamma = np.zeros(C.shape[1])
    gamma[keep] = np.clip(coef[1 + k:], 0.0, None)
    return float(coef[0]), coef[1:1 + k], gamma, float(np.minimum(Cr @ gamma, COST_CAP).mean())


def run_filter(P: dict, y: np.ndarray, prm: dict, upto: int | None = None, fit: bool = True) -> dict:
    """Walk the days: each row's expected time from the state after earlier days, then the day's runs update it.
    ``upto``: stop after that day number. ``fit``: refit the covariates' effects and the codes' prices each period
    (off for the grid over the filter's constants)."""
    n, dn, dog, key, kt, kc, pk, pb = P["n"], P["dn"], P["dog"], P["key"], P["kt"], P["kc"], P["pk"], P["pb"]
    race, meeting, trial, period = P["race"], P["meeting"], P["trial"], P["period"]
    r_, q_, P0 = prm["r"], prm["q"], prm["p0"]

    def state(codes):
        m = int(codes.max()) + 1 if n else 1
        return np.zeros(m), np.zeros(m)

    a, nkey = np.full(int(key.max()) + 1 if n else 1, np.nan), np.zeros(int(key.max()) + 1 if n else 1)
    nd = int(dog.max()) + 1 if n else 1
    M, V, CNT, LAST = np.zeros(nd), np.full(nd, P0), np.zeros(nd), np.full(nd, -1)
    (Ste, Nte), (Sge, Nge), (Sdk, Ndk), (Sdb, Ndb), (Scm, Ncm) = state(kt), state(kc), state(pk), state(pb), state(kc)
    (Skt, Nkt), (Skg, Nkg) = state(key), state(key)             # the track and trip's own mean of each: taken out
    vw = {"ss": 0.0, "df": 0.0, "m2": 0.0, "inv": 0.0, "m": 0.0}
    xb, cost = np.zeros(n), np.zeros(n)
    o = {k: np.full(n, np.nan) for k in ("a", "level", "sd", "pref", "trap", "grade", "e", "e1", "e2")}
    o["n"] = np.zeros(n)
    o["xb"] = xb
    fits, cur = {}, None
    order, bounds = P["order"], P["bounds"]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        i = order[lo:hi]
        today = int(dn[i[0]])
        if upto is not None and today > upto:
            break
        if fit and period[i[0]] != cur:
            cur = period[i[0]]
            # the target of the refit: the residual after the variant with the adjustments put back. The trouble is
            # still in e1 (only e2 has the cost taken out), so the cost is NOT added: until 11 Nov 2026 it was, and
            # each yearly refit priced a code at its true cost plus the cost already applied (the audit, finding 29)
            pr = o["e1"] + xb
            rows = np.flatnonzero((period < cur) & np.isfinite(pr))
            if len(rows) >= MIN_FIT:
                b0, beta, gamma, mean_cost = ridge(P["X"], P["C"], np.clip(pr[rows], -CLIP - 2, CLIP + 2), rows)
                nxt = np.flatnonzero(period == cur)
                xb[nxt] = b0 + mean_cost + P["X"][nxt] @ beta
                cost[nxt] = np.minimum(P["C"][nxt] @ gamma, COST_CAP) - mean_cost
                fits[int(cur)] = {"rows": int(len(rows)), "beta": beta, "gamma": gamma, "b0": b0 + mean_cost}
        d, k = dog[i], key[i]
        new = LAST[d] < 0
        if new.any():
            rc = np.unique(race[i], return_inverse=True)[1]
            s = np.bincount(rc, weights=np.where(new, 0.0, M[d]))
            c = np.bincount(rc, weights=(~new).astype(float))
            grade = np.where(Ncm[kc[i]] > 0, Scm[kc[i]] / np.maximum(Ncm[kc[i]], 1.0), 0.0)
            prior = np.where(c[rc] >= 2, s[rc] / np.maximum(c[rc], 1.0), grade)
            M[d[new]], V[d[new]] = prior[new], P0
        gap = np.where(new, 0.0, today - LAST[d])
        vpre = np.minimum(V[d] + q_ * gap, P0)
        th = M[d]
        dkv, dbv = Sdk[pk[i]] / (Ndk[pk[i]] + K_PREF), Sdb[pb[i]] / (Ndb[pb[i]] + K_PREF)
        # the trap and the grade against the track and trip's average (the standard carries the average)
        tev = Ste[kt[i]] / (Nte[kt[i]] + K_TRAP) - Skt[k] / (Nkt[k] + K_TRAP)
        gev = Sge[kc[i]] / (Nge[kc[i]] + K_GRADE) - Skg[k] / (Nkg[k] + K_GRADE)
        base = th + dkv + dbv + tev + gev + xb[i]
        o["a"][i], o["level"][i], o["sd"][i], o["n"][i] = a[k], th, np.sqrt(vpre + r_), CNT[d]
        o["pref"][i], o["trap"][i], o["grade"][i] = dkv + dbv, tev, gev
        yv = y[i]
        ok = np.isfinite(yv)
        if not ok.any():
            continue
        o["e"][i] = yv - a[k] - base                              # the forecast's error: NaN at a new track and trip
        cold = ok & np.isnan(a[k])
        if cold.any():
            ku, inv = np.unique(k[cold], return_inverse=True)
            a[ku] = np.bincount(inv, weights=(yv - base)[cold]) / np.bincount(inv)
        e = yv - a[k] - base
        ok &= np.abs(np.nan_to_num(e, nan=99.0)) <= OUT
        if not ok.any():
            continue
        # the meeting variant: every runner's error, shrunk by the evidence
        mu, mi = np.unique(meeting[i][ok], return_inverse=True)
        ec = np.clip(e[ok], -CLIP, CLIP)
        nm = np.bincount(mi).astype(float)
        mm = np.bincount(mi, weights=ec) / nm
        if vw["m"] >= 50 and vw["df"] > 0:
            s2w = vw["ss"] / vw["df"]
            s2b = max(vw["m2"] / vw["m"] - s2w * vw["inv"] / vw["m"], 1e-4)
            kv = s2w / s2b
        else:
            kv = K_VAR0
        v = (mm * nm / (nm + kv))[mi]
        ss = np.bincount(mi, weights=(ec - mm[mi]) ** 2)
        big = nm >= 2
        vw["ss"] += float(ss[big].sum())
        vw["df"] += float((nm[big] - 1).sum())
        vw["m2"] += float((mm[big] ** 2).sum())
        vw["inv"] += float((1.0 / nm[big]).sum())
        vw["m"] += float(big.sum())
        j = i[ok]
        e1 = e[ok] - v
        e2 = e1 - cost[j]
        o["e1"][j], o["e2"][j] = e1, e2
        dj, tj = d[ok], trial[j]
        noisy = np.where(tj | (P["C"][j].getnnz(axis=1) > 0), NOISY, 1.0)
        # the dog's level: one update a dog a day (a day's runs averaged)
        du, di = np.unique(dj, return_inverse=True)
        cnt = np.bincount(di).astype(float)
        vp = np.zeros(len(du))
        vp[di] = vpre[ok]
        R = r_ * (np.bincount(di, weights=noisy) / cnt) / cnt
        inn = np.bincount(di, weights=e2) / cnt
        lim = 3.0 * np.sqrt(vp + R)
        gain = vp / (vp + R)
        M[du] += gain * np.clip(inn, -lim, lim)
        V[du] = (1.0 - gain) * vp
        CNT[du] += cnt
        LAST[du] = today
        # the standard, the trap, the grade (graded races only), the preferences
        g = ~tj
        if g.any():
            ku, inv = np.unique(k[ok][g], return_inverse=True)
            c = np.bincount(inv).astype(float)
            # from the error BEFORE the variant: the variant is the night, the standard what lasts (the first build
            # updated it after the variant, which hid every lasting error from it: run 43055884883)
            a[ku] += (np.bincount(inv, weights=(e2 + v)[g]) / c) * c / (c + np.minimum(nkey[ku], STD_CAP) + 1.0)
            nkey[ku] += c
            np.add.at(Ste, kt[j][g], (e2 + tev[ok])[g])
            np.add.at(Nte, kt[j][g], 1.0)
            np.add.at(Sge, kc[j][g], (e2 + gev[ok])[g])
            np.add.at(Nge, kc[j][g], 1.0)
            np.add.at(Skt, k[ok][g], (e2 + tev[ok])[g])
            np.add.at(Nkt, k[ok][g], 1.0)
            np.add.at(Skg, k[ok][g], (e2 + gev[ok])[g])
            np.add.at(Nkg, k[ok][g], 1.0)
        np.add.at(Sdk, pk[j], e2 + dkv[ok])
        np.add.at(Ndk, pk[j], 1.0)
        np.add.at(Sdb, pb[j], e2 + dbv[ok])
        np.add.at(Ndb, pb[j], 1.0)
        known = ~new
        np.add.at(Scm, kc[i][known], th[known])
        np.add.at(Ncm, kc[i][known], 1.0)
    o["fits"] = fits
    return o


def fit_constants(P: dict, y: np.ndarray, early_to: int) -> tuple[dict, list]:
    """The filter's noise a run and drift a day: the pair with the smallest squared error of the forecasts over the
    second half of the early days, each against its race's mean error (graded runs of dogs with a run behind them)."""
    dn = P["dn"]
    seen = dn[dn <= early_to]
    if not len(seen):
        return {"r": 0.3, "q": 0.001, "p0": 0.3}, []
    half = int(np.median(seen))
    score = (dn > half) & (dn <= early_to) & ~P["trial"]
    grid, best = [], None
    for r, q, p0 in GRID:
        o = run_filter(P, y, {"r": r, "q": q, "p0": p0}, upto=early_to, fit=False)
        k = score & np.isfinite(o["e"]) & (o["n"] >= 1) & (np.abs(o["e"]) <= OUT)
        if k.sum() >= 100:                                       # against the race's mean error: the going is not the dog
            rc = np.unique(P["race"][k], return_inverse=True)[1]
            ek = o["e"][k]
            mse = float(np.mean((ek - (np.bincount(rc, weights=ek) / np.bincount(rc))[rc]) ** 2))
        else:
            mse = np.nan
        grid.append((r, q, p0, round(mse, 5) if np.isfinite(mse) else None))
        if np.isfinite(mse) and (best is None or mse < best[0]):
            best = (mse, {"r": r, "q": q, "p0": p0})
    return (best[1] if best else {"r": 0.3, "q": 0.001, "p0": 0.3}), grid


# ---- the table -----------------------------------------------------------------------------------------------------

def _vs(v: pd.Series, race: pd.Series) -> tuple[pd.Series, pd.Series]:
    g = v.groupby(race)
    others = (g.transform("sum") - v.fillna(0.0)) / (g.transform("count") - v.notna()).replace(0, np.nan)
    return v - others, v - g.transform("max")


def with_trials(df: pd.DataFrame, trial_runs: pd.DataFrame | None) -> pd.DataFrame:
    """``df`` with the trial rows ``data.clean`` drops (greyhound/pit/trials.py ``select``) added as runs the filter
    reads: a time, a sectional, a weight and a comment on a day, flagged as trials (their noise is larger, they move
    no standard, trap or grade effect, and they are never a row of the table). The audit of 11 Nov 2026: no trial had
    ever reached the model, so a debutant's level was a guess from the dogs it was graded to meet."""
    if trial_runs is None or not len(trial_runs):
        return df
    have = set(zip(df.race_id.to_numpy(), df.dog_id.to_numpy()))
    tr = trial_runs[[k not in have for k in zip(trial_runs.race_id.to_numpy(), trial_runs.dog_id.to_numpy())]]
    if not len(tr):
        return df
    day = pd.to_datetime(tr.race_date, errors="coerce")
    new = pd.DataFrame({
        "race_id": tr.race_id.to_numpy(), "dog_id": tr.dog_id.to_numpy(), "race_date": tr.race_date.astype(str).to_numpy(),
        "race_time": "00:00", "t": day.to_numpy(), "track": tr.track.to_numpy(),
        "distance_m": pd.to_numeric(tr.distance_m, errors="coerce").to_numpy(), "race_class": tr.race_class.to_numpy(),
        "trap": pd.to_numeric(tr.trap, errors="coerce").to_numpy(), "comment": tr.comment.fillna("").to_numpy(),
        "adjusted_time": pd.to_numeric(tr.adjusted_time, errors="coerce").to_numpy(),
        "run_time": pd.to_numeric(tr.run_time, errors="coerce").to_numpy(),
        "sectional": pd.to_numeric(tr.sectional, errors="coerce").to_numpy(),
        "weight_kg": pd.to_numeric(tr.weight_kg, errors="coerce").to_numpy(),
        "born": tr.born.to_numpy() if "born" in tr else np.nan, "sex": tr.sex.to_numpy() if "sex" in tr else np.nan,
        "position": pd.to_numeric(tr.position, errors="coerce").to_numpy(), "won": np.nan, "is_trial": True,
        "is_card": False, "grade_family": "T", "handicap": False,
        "hurdles": (tr.race_type.astype(str).str.strip().str.lower() == "hurdles").to_numpy() if "race_type" in tr else False,
        "field": tr.groupby("race_id").dog_id.transform("size").to_numpy(float)})
    # a trial session is its own meeting (never a race meeting's variant): negative ids clear of the cards'
    key = tr.track.astype(str) + "|" + tr.race_date.astype(str)
    new["meeting_id"] = -(1_000_000_000 + pd.factorize(key)[0])
    new = new.dropna(subset=["t"])
    new.index = np.arange(len(new)) + (int(df.index.max()) + 1 if len(df) else 0)
    return pd.concat([df, new[[c for c in new.columns if c in df.columns]]])


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = figs if figs is not None else core.per_run_figures(runs)
    rows = df.index                                                    # the table's rows: the trials are read, not written
    df = with_trials(df, (ctx or {}).get("trial_runs"))
    n_trials = len(df) - len(rows)
    df = df.sort_values(["t", "race_id", "trap"], kind="stable")
    race = df.race_id
    day = core.day(df)
    early = day < pd.Timestamp(FIT_BEFORE)
    if early.sum() < 0.05 * len(df) and len(df):
        early = day <= day.sort_values().iloc[int(len(df) * 0.2)]
    P = prepare(df)
    Y = targets(df)
    early_to = int(P["dn"][early.to_numpy()].max()) if early.any() else -1
    out = pd.DataFrame({"race_id": race.to_numpy(), "dog_id": df.dog_id.to_numpy()}, index=df.index)
    notes = {"refit_months": REFIT_MONTHS, "covariates": len(P["names"]), "codes": len(P["vocab"]),
             "trial_runs_read": int(n_trials)}
    year = day.dt.year.to_numpy()
    L = core.LENGTH_S
    exp_pct = {}
    for name in TARGETS:
        prm, grid = fit_constants(P, Y[name], early_to)
        o = run_filter(P, Y[name], prm)
        per_pct = pd.Series(np.exp(o["a"] / 100.0) / 100.0 / L, index=df.index)       # lengths in 1% of the standard
        pct = pd.Series(o["level"] + o["pref"] + o["trap"] + o["grade"] + o["xb"], index=df.index)
        exp_pct[name] = pct
        f = -pct * per_pct                                                         # lengths, higher is faster
        out[f"{PREFIX}{name}"] = f
        out[f"{PREFIX}{name}_sd"] = o["sd"] * per_pct
        vs, gap = _vs(f, race)
        out[f"{PREFIX}{name}_vs_field"] = vs
        out[f"{PREFIX}{name}_gap"] = gap
        out[f"{PREFIX}{name}_level"] = -pd.Series(o["level"], index=df.index) * per_pct
        out[f"{PREFIX}{name}_adj"] = -pd.Series(o["xb"], index=df.index) * per_pct
        out[f"{PREFIX}{name}_time"] = np.exp((o["a"] + pct) / 100.0)
        if name == "tot":
            out[f"{PREFIX}n"] = o["n"]
            out[f"{PREFIX}pref"] = -pd.Series(o["pref"], index=df.index) * per_pct
            out[f"{PREFIX}trap"] = -pd.Series(o["trap"], index=df.index) * per_pct
            sur = df.assign(_s=-pd.Series(o["e2"], index=df.index) * per_pct)
            out[f"{PREFIX}last_surprise"] = core.dog_lagged(sur, "_s", 1)
            out[f"{PREFIX}surprise3"] = core.dog_lagged(sur, "_s", 3)
        # how well the forecast read the time (out of sample by construction): graded runs, a run or more behind
        e_l = pd.Series(o["e"], index=df.index) * per_pct
        flat = pd.Series(o["e"] + o["level"] + o["pref"], index=df.index) * per_pct
        k = np.isfinite(o["e"]) & ~P["trial"] & (o["n"] >= 1) & (np.abs(o["e"]) <= OUT)
        acc = {}
        for yv in sorted(set(year[k])):
            m = k & (year == yv)
            fm = flat[m] - flat[m].groupby(race[m]).transform("mean")
            em = e_l[m] - e_l[m].groupby(race[m]).transform("mean")
            acc[int(yv)] = {"runs": int(m.sum()), "rmse_lengths": round(float(np.sqrt((e_l[m] ** 2).mean())), 3),
                            "rmse_after_variant": round(float(np.sqrt(((pd.Series(o["e1"], index=df.index) * per_pct)[m]
                                                                        ** 2).mean())), 3),
                            "rmse_within_race": round(float(np.sqrt((em ** 2).mean())), 3),
                            "rmse_within_race_no_dog": round(float(np.sqrt((fm ** 2).mean())), 3)}
        last = o["fits"][max(o["fits"])] if o["fits"] else None
        notes[name] = {"filter": prm, "grid": grid, "by_year": acc, "fits": {p: v["rows"] for p, v in o["fits"].items()}}
        if last is not None:
            inv = {v: c for c, v in P["vocab"].items()}
            price = sorted(((inv[j], round(float(g), 3)) for j, g in enumerate(last["gamma"]) if g != 0.0),
                           key=lambda kv: -kv[1])
            notes[name]["effects_pct"] = {nm: round(float(b), 3) for nm, b in zip(P["names"], last["beta"])}
            if price:                                                # the dearest code's price at every refit: steady
                j_top = P["vocab"][price[0][0]]
                notes[name]["top_code_by_fit"] = {int(p_): round(float(v["gamma"][j_top]), 3)
                                                  for p_, v in sorted(o["fits"].items())}
            notes[name]["code_cost_pct_top"] = dict(price[:15])
            notes[name]["code_cost_pct_bottom"] = dict(price[-10:])
    tau, grid = figs_mod.fit_tau(out[f"{PREFIX}tot"], race, pd.to_numeric(df.won, errors="coerce"), early)
    sh = figs_mod.race_softmax(out[f"{PREFIX}tot"].fillna(out[f"{PREFIX}tot"].groupby(race).transform("mean"))
                               .fillna(0.0), race, tau)
    out[f"{PREFIX}tot_share"] = sh
    out[f"{PREFIX}tot_lshare"] = np.log(sh.clip(1e-9))
    notes.update({"tau": tau, "tau_grid": grid})
    # the first bend as the experts read it: the expected sectional against the dogs drawn either side, and the
    # expected leader's margin over the next
    s = out[f"{PREFIX}sec"]
    trap = pd.to_numeric(df.trap, errors="coerce")
    nb = pd.DataFrame({"race": race.to_numpy(), "trap": trap.to_numpy(), "s": s.to_numpy()})
    for side, step in (("in", -1), ("out", 1)):
        other = nb.assign(trap=nb.trap - step).rename(columns={"s": "o"}).drop_duplicates(["race", "trap"])
        j = nb.merge(other, on=["race", "trap"], how="left")
        out[f"{PREFIX}sec_vs_{side}"] = (j.s - j.o).to_numpy()
    out[f"{PREFIX}sec_clear"] = out[[f"{PREFIX}sec_vs_in", f"{PREFIX}sec_vs_out"]].min(axis=1)
    g = s.groupby(race)
    top = g.transform("max")
    second = s.where(s < top).groupby(race).transform("max")
    out[f"{PREFIX}sec_lead"] = np.where(s >= top, s - second, s - top)
    out[f"{PREFIX}pace_shape"] = out[f"{PREFIX}sec_vs_field"] - out[f"{PREFIX}run_vs_field"]
    cols = [c for c in out.columns if c.startswith(PREFIX)]
    out[cols] = out[cols].astype(np.float32)
    return out.loc[rows].sort_index().reset_index(drop=True), notes


_LAB = {"tot": "whole race", "sec": "sectional (to the line the first time)", "run": "run home"}
COLUMNS = {
    **{f"{PREFIX}{n}": f"the dog's expected {lab} time today against the standard of the track and trip (lengths, "
                       "higher faster): its level, its preference for the track, trip and distance band, the trap, "
                       "the grade and today's adjustments (age, days off, change of trip ...)" for n, lab in _LAB.items()},
    **{f"{PREFIX}{n}_sd": "the spread of that forecast (the level's uncertainty and a run's noise; lengths)"
       for n in TARGETS},
    **{f"{PREFIX}{n}_vs_field": "it against the mean of the race's other runners (lengths)" for n in TARGETS},
    **{f"{PREFIX}{n}_gap": "lengths behind the race's fastest expected" for n in TARGETS},
    **{f"{PREFIX}{n}_level": "the dog's level alone (lengths): no trap, grade or adjustment" for n in TARGETS},
    **{f"{PREFIX}{n}_adj": "today's adjustments alone (lengths): trial, field size, age, days off, runs since a "
                           "layoff, experience, first run here, change of track or trip, weight" for n in TARGETS},
    **{f"{PREFIX}{n}_time": f"the expected {lab} time in seconds" for n, lab in _LAB.items()},
    f"{PREFIX}n": "runs behind the dog's level",
    f"{PREFIX}pref": "the dog's preference for this track and trip and this distance band (lengths)",
    f"{PREFIX}trap": "the trap's effect on the time at this track and trip (lengths)",
    f"{PREFIX}last_surprise": "the last run against its expected time, the variant and the trouble taken out (lengths)",
    f"{PREFIX}surprise3": "the mean of the last 3",
    f"{PREFIX}tot_share": "its share of the race from a softmax of the expected whole-race times",
    f"{PREFIX}tot_lshare": "the log of that share",
    f"{PREFIX}sec_vs_in": "expected sectional against the dog drawn one trap inside (lengths; above 0 it leads it)",
    f"{PREFIX}sec_vs_out": "expected sectional against the dog drawn one trap outside",
    f"{PREFIX}sec_clear": "the smaller of the two: how clear of its neighbours it should be at the first bend",
    f"{PREFIX}sec_lead": "the expected leader's margin over the next (the leader), or lengths behind the leader",
    f"{PREFIX}pace_shape": "early pace against the field less run home against the field (above 0 a front runner)",
}
