"""The league table: every track's grades on one scale, by Hunter's MM, and what that scale says a dog's record
hides (screened on 12 Nov 2026: scratchpad rob/mm_entities, mm_closed, mm_exposure, mm_smart; a screen, not a
registered trial).

Hunter's Minorisation-Maximisation rates a thing by its results against the strength of what it met, where that
strength is itself a rating, and repeats until nothing moves. Rob's reading ("closed form" against "open form"): a
rating is the dog's own metric times the strength of its opposition, iterated from a Laplace prior at the mean. The
four scratch builds found that the rating LEVEL adds nothing to the pricing model's remaining miss; the SHIFTS do.
This table is the shifts.

**1. Leagues** (``lg_str`` .. ``lg_move_t``). A league is a track's class: (track, ``race_class``), every class
GBGB writes (the A, B, D, S, P, H rungs of ``greyhound.data.grade_scale``, and OR, IV, IT, HP: a class with no
rung is its own league). A dog that runs in two leagues links them. The performance of a run is its race-centred
PRB (the share of the finishers it beat, less the race's mean; races of three finishers or more), so inside one
league the mean is zero and only dogs that move tell the leagues apart. The update, on the log scale of "rating =
performance x league strength" (the product is a sum there), each run weighted w = 0.5^(age / ``HALF_LIFE`` days):

    rating_dog    <- sum(w * y) / (sum(w) + KD)  +  the w-mean of the strengths of the leagues it ran in
    strength_lg   <- sum(w * rating of its runners) / (sum(w) + KL),  then centred (the w-mean league is 0)

repeated until no league moves ``TOL`` (1e-4), at most ``MAXIT`` passes. The Laplace prior: a dog has ``KD``
pseudo-runs at its leagues' mean, a league ``KL`` pseudo-runs at the mean league (zero), so a thin league sits near
zero and no fallback league is needed. The first term is the open form; what the iteration adds is the second, the
dog's usual league. ``lg_move_t`` is the same fit on lengths (the race's median run time less the dog's, / 0.08 s,
clipped at +/-10, race-centred; no handicaps).

**2. Strength of schedule** (``lg_sos_prb2``, ``lg_sos_nfp``). Rob's rule on a dog's results against its actual
rivals. PRB squared (multiplicative): R <- (K*mu + sum_r m_r * S_r / Sbar) / (K + n), S_r the mean rating of the
RIVALS in race r, Sbar the m-weighted mean of S, mu the window's mean. Normalised finishing position (signed,
additive): R <- sum_r (m_r + S_r - Sbar) / (K + n). The first pass from the flat start is the open form (the
metric as is, the same shrinkage); the column is closed less open: what the opposition adds.

**3. Market strength faced** (``lg_mkt_open``, ``lg_mkt_diff``). x = log(SP share x priced runners) of a run. Open
form: the dog's mean x over its last ``N_MKT`` priced runs in the window, one pseudo-run at an equal share. One MM
pass (not iterated: the scratch build's iterated rule did not converge at lambda 1): the mean over those runs of
its rivals' mean open form, centred. ``lg_mkt_diff`` is that pass less the open form = how fancied the dogs it has
met have been. No result enters either column.

**4. The track** (``lg_new_trk``, ``lg_n_trk``): the dog's runs at today's track on earlier days within 365 days;
1 when none. Read to the day, not to the Monday (a run's existence is known; its result is not read).

**5. The trap** (``lg_trap_wax``): at today's track, trip and trap, the winners less the SP's expected winners a
run (won - SP share, races with every runner priced), over the window, shrunk by ``K_TRAP`` runs at zero. The
scratch column that scored (``ms_today_ctx_wax``) added the kennel's effect to the trap's: the kennel is left out
of this table altogether, because GBGB serves a dog's CURRENT trainer on its past rows. Nothing here reads the
trainer, the owner or a kennel.

**The refit calendar - the leak that matters.** An iterated rating fitted on a window that holds the race being
priced, or a later one, knows the result. Every fitted quantity (1, 2, 3, 5) is refitted on each Monday from runs
dated strictly before it, within the 365 days before it, and is read by the races dated from that Monday to the
following Sunday. Each refit starts from the last one's ratings (a warm start reads earlier refits only). What a
race reads of its own week is card facts: its track, class, trip and trap, and which leagues the dog ran in on
earlier days (the dog's last league may be a run of last Tuesday; that league's strength is still the Monday's).
tests/test_greyhound_league.py rewrites every result on and after a Monday and requires that week's and every
earlier row to be unchanged.

**What it inherits.** The store's first year has shorter windows (the first week has none: the fitted columns are
empty there), and ``lg_n_trk`` counts only what the store holds. The fits stop at ``TOL``, so a rating depends on
its warm start within that tolerance: a store that starts later gives values that differ by about ``TOL``.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from greyhound.pit import KEY

PREFIX = "lg_"
SOURCE = ("GBGB results (finishing position, run time, SP, class, track, trip, trap: sources/gbgb.py), fitted by "
          "Hunter's MM on each Monday from the runs of the 365 days before it")
COLUMNS = {
    "lg_str": "strength of today's league (track and class) on the cross-track scale, race-centred PRB units; 0 the mean league",
    "lg_str_last": "strength of the league of the dog's last run on an earlier day, at this week's fit",
    "lg_move": "class move in league units: today's league strength less its last league's (above 0: up in class)",
    "lg_move3": "today's league strength less the mean strength of the leagues of its last three runs on earlier days",
    "lg_move_avg": "today's league strength less the dog's usual league (the weighted mean strength of the leagues it ran in over the window)",
    "lg_rating_vs": "the dog's league-adjusted (closed-form) rating less the mean of the field's rated dogs",
    "lg_move_t": "the class move (today less last) on the league scale fitted on lengths from run times",
    "lg_sos_prb2": "strength of schedule, PRB squared: the rating iterated against the rivals' ratings less the open form",
    "lg_sos_nfp": "strength of schedule, normalised finishing position: the iterated rating less the open form",
    "lg_mkt_open": "the dog's mean log(SP share x runners) over its last 7 priced runs in the window, shrunk by one run at an equal share",
    "lg_mkt_diff": "market strength faced: the mean open-form market rating of the rivals in those runs (one MM pass less the open form), centred",
    "lg_new_trk": "1 when the dog has no run at today's track on an earlier day within 365 days",
    "lg_n_trk": "the dog's runs at today's track on earlier days within 365 days",
    "lg_trap_wax": "winners less the SP's expected winners a run from today's trap at today's track and trip, over the window, shrunk",
}

WINDOW = 365                 # days of runs behind a refit
HALF_LIFE = 180.0            # days: a run's weight in the league fit
KD, KL = 5.0, 20.0           # Laplace pseudo-runs: a dog at its leagues' mean, a league at the mean league
K_SOS = 3.0                  # pseudo-runs at the window's mean (PRB squared, NFP)
N_MKT, K_MKT = 7, 1.0        # the priced runs read; pseudo-runs at an equal share
K_TRAP = 50.0                # pseudo-runs at zero for a trap cell (fixed, not fitted)
TOL, MAXIT = 1e-4, 300
LENGTH_S = 0.08
BIG = np.int64(1) << 21      # day numbers are below this


def monday(dn: np.ndarray) -> np.ndarray:
    """The day number of the Monday on or before each day number (days since 1970-01-01, a Thursday)."""
    return (dn + 3) // 7 * 7 - 3


def _compact(code: np.ndarray, n: int):
    """``code`` renumbered 0..m-1 over the values present; the present values (ascending)."""
    here = np.bincount(code, minlength=n) > 0
    idx = np.cumsum(here) - 1
    return idx[code], np.flatnonzero(here)


def league_mm(dog, lg, y, w, nl: int, s0=None):
    """The league fit (module docstring, 1) on rows (dog 0..nd-1, league code, performance, weight). Returns the
    dogs' closed ratings, their open form, the league strengths (NaN for a league with no run), the passes."""
    nd = int(dog.max()) + 1
    wd = np.bincount(dog, w, nd)
    wl = np.bincount(lg, w, nl)
    opn = np.bincount(dog, w * y, nd) / (wd + KD)
    tot = np.cumsum(wl)[-1]            # summed in league order: a league first seen later in the store adds nothing
    s = np.zeros(nl) if s0 is None else np.where(wl > 0, np.nan_to_num(s0), 0.0)
    it = 0
    for it in range(1, MAXIT + 1):
        r = opn + np.bincount(dog, w * s[lg], nd) / wd
        sn = np.bincount(lg, w * r[dog], nl) / (wl + KL)
        sn = np.where(wl > 0, sn - np.cumsum(sn * wl)[-1] / tot, 0.0)
        delta = np.abs(sn - s).max()
        s = sn
        if delta < TOL:
            break
    r = opn + np.bincount(dog, w * s[lg], nd) / wd
    return r, opn, np.where(wl > 0, s, np.nan), it


def closed_form(dog, race, m, mu: float, additive: bool, r0=None):
    """Rob's rule (module docstring, 2) on rows (dog 0..nd-1, race 0..nr-1, metric); every race has two rows or
    more. Returns the closed ratings, the open form, the passes."""
    nd, nr = int(dog.max()) + 1, int(race.max()) + 1
    cnt = np.bincount(race, minlength=nr)[race] - 1.0
    den = K_SOS + np.bincount(dog, minlength=nd)
    prior = 0.0 if additive else K_SOS * mu
    opn = (prior + np.bincount(dog, m, nd)) / den
    r = np.full(nd, 0.0 if additive else mu) if r0 is None else r0.copy()
    msum = m.sum()
    it = 0
    for it in range(1, MAXIT + 1):
        x = r[dog]
        s = (np.bincount(race, x, nr)[race] - x) / cnt
        if additive:
            new = np.bincount(dog, m + s - s.mean(), nd) / den
        else:
            new = (prior + np.bincount(dog, m * s * (msum / (m * s).sum()), nd)) / den
        delta = np.abs(new - r).max()
        r = new
        if delta < TOL:
            break
    return r, opn, it


def market_pass(dog, race, x, use):
    """One MM pass of the market rating (module docstring, 3) on the window's priced rows (dog 0..nd-1, race
    0..nr-1, x); ``use`` marks each dog's last ``N_MKT`` of them. Returns the open form and the strength faced
    (NaN for a dog with none)."""
    nd, nr = int(dog.max()) + 1, int(race.max()) + 1
    n = np.bincount(dog[use], minlength=nd).astype(float)
    opn = np.bincount(dog[use], x[use], nd) / (n + K_MKT)
    cr = np.bincount(race, minlength=nr)[race] - 1.0
    ok = use & (cr > 0)
    riv = (np.bincount(race, opn[dog], nr)[race] - opn[dog]) / np.maximum(cr, 1.0)
    k = np.bincount(dog[ok], minlength=nd).astype(float)
    faced = np.where(k > 0, np.bincount(dog[ok], riv[ok], nd) / np.maximum(k, 1.0), np.nan)
    if (k > 0).any():
        faced = faced - np.nanmean(faced)
    return np.where(n > 0, opn, np.nan), faced


def _codes(*parts) -> tuple[np.ndarray, int]:
    """One integer code a distinct combination of ``parts`` (-1 where a part is missing)."""
    ok = np.ones(len(parts[0]), bool)
    key = None
    for p in parts:
        c, u = pd.factorize(np.asarray(p, dtype=object), use_na_sentinel=True)
        ok &= c >= 0
        key = c.astype(np.int64) if key is None else key * (len(u) + 1) + c
    code, u = pd.factorize(np.where(ok, key, -1))
    code = code.astype(np.int64)
    if (~ok).any():
        code = np.where(ok, code, -1)
    return code, len(u)


def _earlier(dog, dn, val, valid, k: int):
    """For every row, ``val`` of the dog's last ``k`` valid runs on EARLIER days (one a day: the day's last run),
    most recent first; -1 where there is none. Rows are in time order."""
    n = len(dog)
    idx = np.flatnonzero(valid)
    key = dog[idx].astype(np.int64) * BIG + dn[idx]
    o = np.argsort(key, kind="stable")
    key, v, dg = key[o], val[idx][o], dog[idx][o]
    last = np.r_[key[1:] != key[:-1], True] if len(key) else np.zeros(0, bool)
    key, v, dg = key[last], v[last], dg[last]
    j = np.searchsorted(key, dog.astype(np.int64) * BIG + dn, "left") - 1
    out = np.full((k, n), -1, dtype=np.int64)
    for i in range(k):
        jj = j - i
        ok = jj >= 0
        ok[ok] = dg[jj[ok]] == dog[ok]
        out[i, ok] = v[jj[ok]]
    return out


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None):
    t0 = time.time()
    n = len(runs)
    card = runs.is_card.to_numpy(bool) if "is_card" in runs else np.zeros(n, bool)
    trial = runs.is_trial.to_numpy(bool) if "is_trial" in runs else np.zeros(n, bool)
    dn_in = (pd.to_datetime(runs.race_date).to_numpy("datetime64[D]") - np.datetime64("1970-01-01")).astype(np.int64)
    tt = runs.t.to_numpy("datetime64[s]").astype(np.int64) if "t" in runs else dn_in
    o = np.lexsort((runs.race_id.to_numpy(), tt, dn_in))               # time order; a race's rows together
    inv = np.empty(n, np.int64)
    inv[o] = np.arange(n)
    r = runs.iloc[o]
    dn = dn_in[o]
    card, ran = card[o], ~card[o] & ~trial[o]
    dog, ND = _codes(r.dog_id.to_numpy())
    rid = r.race_id.to_numpy()
    race = np.cumsum(np.r_[True, rid[1:] != rid[:-1]]) - 1 if n else np.zeros(0, np.int64)
    NR = int(race.max()) + 1 if n else 0
    lg, NL = _codes(r.track.to_numpy(), r.race_class.to_numpy())
    cell, NC = _codes(r.track.to_numpy(), r.distance_m.to_numpy(), r.trap.to_numpy())
    dtrk, _ = _codes(dog, r.track.to_numpy())
    pos = pd.to_numeric(r.position, errors="coerce").to_numpy(float)
    sp = pd.to_numeric(r.sp_p_norm, errors="coerce").to_numpy(float) if "sp_p_norm" in r else np.full(n, np.nan)

    # ---- the per-run measures (results: read only through a window that ends before a Monday)
    fin = ran & np.isfinite(pos)
    nf = np.bincount(race[fin], minlength=NR)[race].astype(float)
    rank = np.full(n, np.nan)
    rank[fin] = pd.Series(pos[fin]).groupby(race[fin]).rank(method="average").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        prb = np.where(fin & (nf >= 2), (nf - rank) / (nf - 1), np.nan)
        nfp = np.where(fin & (nf >= 2), (nf + 1 - 2 * rank) / (3 * np.sqrt((nf + 1) / (3 * (nf - 1))) * (nf - 1)), np.nan)
    prbc = np.where(nf >= 3, prb - 0.5, np.nan)                         # the race's mean PRB is 0.5 by construction
    ln = np.full(n, np.nan)
    if "run_time" in r:
        hcp = r.handicap.to_numpy(bool) if "handicap" in r else np.zeros(n, bool)
        tm = pd.Series(np.where(fin & ~hcp & (r.run_time.to_numpy(float) > 5), r.run_time.to_numpy(float), np.nan))
        g = tm.groupby(race)
        ln = ((g.transform("median") - tm) / LENGTH_S).clip(-10, 10)
        ln = (ln - ln.groupby(race).transform("mean")).to_numpy()
    priced = ran & np.isfinite(sp) & (sp > 0)
    npr = np.bincount(race[priced], minlength=NR)[race].astype(float)
    whole = priced & (npr == np.bincount(race[ran], minlength=NR)[race])
    won = (pos == 1).astype(float)

    def subset(mask):
        i = np.flatnonzero(mask)
        return i, dn[i]

    f1, d1 = subset(np.isfinite(prbc) & (lg >= 0))                     # league fit, PRB
    f2, d2 = subset(np.isfinite(ln) & (lg >= 0))                       # league fit, lengths
    f3, d3 = subset(np.isfinite(prb))                                  # schedule: finishers of races with two or more
    f4, d4 = subset(priced)                                            # market
    f5, d5 = subset(whole & (cell >= 0))                               # trap cells
    rc3 = np.cumsum(np.r_[True, race[f3][1:] != race[f3][:-1]]) - 1 if len(f3) else np.zeros(0, np.int64)
    rc4 = np.cumsum(np.r_[True, race[f4][1:] != race[f4][:-1]]) - 1 if len(f4) else np.zeros(0, np.int64)
    k4 = pd.Series(np.ones(len(f4))).groupby(dog[f4]).cumcount().to_numpy() if len(f4) else np.zeros(0, np.int64)
    x4 = np.log(sp[f4] * npr[f4])
    wax5 = won[f5] - sp[f5]

    # ---- card facts read to the day: the leagues of the last three earlier days, the runs at the track
    lastlg = _earlier(dog, dn, lg, ran & (lg >= 0), 3)
    i_t = np.flatnonzero(ran)
    kt = np.sort(dtrk[i_t] * BIG + dn[i_t])
    q = dtrk * BIG + dn
    n_trk = (np.searchsorted(kt, q, "left") - np.searchsorted(kt, q - WINDOW, "left")).astype(float)

    out = {c: np.full(n, np.nan) for c in COLUMNS}
    out["lg_n_trk"], out["lg_new_trk"] = n_trk, (n_trk == 0).astype(float)

    # ---- the walk-forward refits
    weeks = np.unique(monday(dn)) if n else np.zeros(0, np.int64)
    warm = {"s": None, "st": None, "p2": np.full(ND, np.nan), "nf": np.full(ND, np.nan)}
    cnt4 = np.zeros(ND, np.int64)
    hi4_prev = 0
    its = {"league": [], "league_lengths": [], "prb2": [], "nfp": []}
    secs = []
    fitted = 0

    def glob(ids, v):
        a = np.full(ND, np.nan)
        a[ids] = v
        return a

    for M in weeks:
        ts = time.time()
        a0, a1 = np.searchsorted(dn, M, "left"), np.searchsorted(dn, M + 7, "left")
        rows = slice(a0, a1)
        gq, lq = dog[rows], lg[rows]

        def window(d):
            return np.searchsorted(d, M - WINDOW, "left"), np.searchsorted(d, M, "left")

        def take(a, idx):
            return np.where(idx >= 0, a[np.maximum(idx, 0)], np.nan)

        # 1. leagues
        for name, fset, dset, y, key in (("league", f1, d1, prbc, "s"), ("league_lengths", f2, d2, ln, "st")):
            lo, hi = window(dset)
            if hi - lo < 2:
                continue
            i = fset[lo:hi]
            assert dn[i].max() < M
            dl, ids = _compact(dog[i], ND)
            w = 0.5 ** ((M - dn[i]) / HALF_LIFE)
            rc, opn, s, it = league_mm(dl, lg[i], y[i], w, NL, warm[key])
            warm[key] = s
            its[name].append(it)
            today, last = take(s, lq), take(s, lastlg[0, rows])
            if key == "st":
                out["lg_move_t"][rows] = today - last
                continue
            fitted += 1
            rg, og = glob(ids, rc), glob(ids, opn)
            out["lg_str"][rows], out["lg_str_last"][rows] = today, last
            out["lg_move"][rows] = today - last
            l3 = np.stack([take(s, lastlg[k, rows]) for k in range(3)])
            k3 = np.isfinite(l3).sum(0)
            out["lg_move3"][rows] = today - np.where(k3 > 0, np.nansum(l3, 0) / np.maximum(k3, 1), np.nan)
            out["lg_move_avg"][rows] = today - (rg - og)[gq]
            v = rg[gq]
            has = np.isfinite(v)
            rq = race[rows] - race[a0]
            m = np.bincount(rq[has], v[has], rq.max() + 1) / np.maximum(np.bincount(rq[has], minlength=rq.max() + 1), 1)
            out["lg_rating_vs"][rows] = v - m[rq]

        # 2. strength of schedule
        lo, hi = window(d3)
        if hi - lo >= 2:
            i = f3[lo:hi]
            assert dn[i].max() < M
            dl, ids = _compact(dog[i], ND)
            rl = rc3[lo:hi] - rc3[lo]
            for name, m, key, col, additive in (("prb2", prb[i] ** 2, "p2", "lg_sos_prb2", False),
                                                ("nfp", nfp[i], "nf", "lg_sos_nfp", True)):
                mu = float(m.mean())
                w0 = warm[key][ids]
                w0 = np.where(np.isfinite(w0), w0, 0.0 if additive else mu)
                rc, opn, it = closed_form(dl, rl, m, mu, additive, w0)
                warm[key] = glob(ids, rc)
                its[name].append(it)
                out[col][rows] = glob(ids, rc - opn)[gq]

        # 3. market strength faced
        lo, hi = window(d4)
        cnt4 += np.bincount(dog[f4[hi4_prev:hi]], minlength=ND)
        hi4_prev = hi
        if hi - lo >= 2:
            i = f4[lo:hi]
            assert dn[i].max() < M
            dl, ids = _compact(dog[i], ND)
            rl = rc4[lo:hi] - rc4[lo]
            opn, faced = market_pass(dl, rl, x4[lo:hi], k4[lo:hi] >= cnt4[dog[i]] - N_MKT)
            out["lg_mkt_open"][rows] = glob(ids, opn)[gq]
            out["lg_mkt_diff"][rows] = glob(ids, faced)[gq]

        # 5. the trap at the track and trip
        lo, hi = window(d5)
        if hi > lo:
            c = cell[f5[lo:hi]]
            nc = np.bincount(c, minlength=NC).astype(float)
            e = np.where(nc > 0, np.bincount(c, wax5[lo:hi], NC) / (nc + K_TRAP), np.nan)
            out["lg_trap_wax"][rows] = take(e, cell[rows])
        secs.append(time.time() - ts)

    tab = pd.DataFrame({"race_id": runs.race_id.to_numpy(), "dog_id": runs.dog_id.to_numpy()})
    for c in COLUMNS:
        tab[c] = out[c][inv].astype(np.float32)
    assert list(tab.columns) == KEY + list(COLUMNS)

    def stat(v):
        return {"median": int(np.median(v)), "max": int(max(v)), "at_the_cap": int(sum(x >= MAXIT for x in v))} if v else None

    notes = {"rows": int(n), "card_rows": int(card.sum()), "dogs": int(ND), "leagues": int(NL), "trap_cells": int(NC),
             "weeks": int(len(weeks)), "weeks_with_a_league_fit": int(fitted),
             "passes": {k: stat(v) for k, v in its.items()},
             "seconds": round(time.time() - t0, 1), "seconds_in_refits": round(float(sum(secs)), 1),
             "seconds_a_refit_last_ten": round(float(np.mean(secs[-10:])), 3) if secs else None,
             "refit": f"each Monday on runs dated before it within {WINDOW} days, warm-started, tolerance {TOL}, "
                      f"at most {MAXIT} passes; read by that Monday to the following Sunday"}
    return tab, notes
