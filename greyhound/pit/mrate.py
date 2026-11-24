"""The market-implied rating, rebuilt (the feature review of 17 Nov 2026, "figs fg_mr", the strongest family, and B5).

greyhound/pit/figs.py ``market_rating`` reads each race's SP as a Kalman observation of every runner's rating: the
race-relative log of its proportional SP share, re-centred on the field's mean rating, one noise level a race, a
drift a day. This table keeps that filter and changes four things.

(a) **The trap is taken out of the observation.** The observation is the dog's log de-biased (Shin) SP share, less
    the race's mean, less the track-trip-trap mean of that race-centred log share over EARLIER days (shrunk to 0 by
    ``TRAP_K`` races), so a strong trap does not raise the dog's rating. The rating's forecast share puts today's trap
    back (softmax of rating + trap mean).
(b) **Share-dependent noise.** A longshot's share is a noisier reading than a favourite's: the observation variance is
    fitted by share band on the early slice (the innovations' variance less the pre-race rating variance), then
    scaled as a whole by the cross-entropy. Two runs of a dog in a day are combined by precision.
(c) **The market's error memory.** A second filter a dog, fed by the result surprise (won less the de-biased SP's
    chance, noise p(1 - p)): the market's running error on the dog, with its sd and z-score.
(d) **Mean reversion after layoffs.** Beyond ``GAP0`` days off, the rating is pulled toward the mean rating of the
    field it last met by exp(-(gap - GAP0) / tau), tau fitted (inf: none).

Constants are fitted on the first 20% of the days (never on or after ``TEST_FROM``) by the cross-entropy of the SP
shares under the ratings before each race (figs' score) and, for (c), by the surprise's predictive log-likelihood.
Every column of a row reads the dog's EARLIER days only; a card row reads what the historic row would.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core
from greyhound.pit import figs as _figs
from greyhound.pit.market2 import shin_probs

DEPENDS = ("figs", "market2")
PREFIX = "mv_"
SOURCE = ("GBGB results (SP); Shin de-biased SP (core.debiased_probs); Betfair's greyhound price files for the BSP "
          "variant (sources/greyhound_prices.py)")
TEST_FROM = "2026-06-01"
EARLY_FRAC = 0.2
TRAP_K = 50.0
GAP0 = 14
BANDS = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 1.01])
MIN_BAND = 50
R_FLOOR = 0.02
GRID_R = (0.1, 0.2, 0.4, 0.8)
GRID_Q = (0.0005, 0.002, 0.008, 0.02)
GRID_V0 = (0.3, 1.0)
GRID_SCALE = (0.5, 0.75, 1.0, 1.5)
GRID_TAU = (np.inf, 60.0, 180.0, 540.0)
GRID_EQ = (0.0, 1e-6, 1e-5, 1e-4)
GRID_EV0 = (0.0003, 0.001, 0.003, 0.01)
DEFAULT = {"r_band": [0.3] * (len(BANDS) - 1), "q": 0.002, "v0": 1.0, "tau": np.inf}


def _early_cut(dn: np.ndarray, have: np.ndarray) -> int | None:
    """The first day after the first ``EARLY_FRAC`` of the days from the first with data, never reaching
    ``TEST_FROM`` (days counted on every row, so a card day counts as its run's day would)."""
    test = (pd.Timestamp(TEST_FROM) - pd.Timestamp("2000-01-01")).days
    if not have.any():
        return None
    days = np.unique(dn[(dn >= dn[have].min()) & (dn < test)])
    if not len(days):
        return None
    return int(max(days[min(int(len(days) * EARLY_FRAC), len(days) - 1)], days[0] + 1))


def band_of(q: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(BANDS, np.nan_to_num(q), side="right") - 1, 0, len(BANDS) - 2)


def trap_effect(df: pd.DataFrame, q: np.ndarray, full: np.ndarray) -> np.ndarray:
    """Each row's track-trip-trap mean of the race-centred log share over EARLIER days, shrunk to 0 by ``TRAP_K``."""
    race = df.race_id.to_numpy()
    lq = pd.Series(np.where(full, np.log(np.where(full, q, 1.0)), np.nan))
    c = lq - lq.groupby(race).transform("mean")
    t = pd.DataFrame({"race_date": df.race_date.to_numpy(), "track": df.track.to_numpy(),
                      "dist": pd.to_numeric(df.distance_m, errors="coerce").round().to_numpy(),
                      "trap": pd.to_numeric(df.trap, errors="coerce").to_numpy(),
                      "_c": c.fillna(0.0).to_numpy(), "_k": c.notna().astype(float).to_numpy()})
    w = core.prior_window(t, ["track", "dist", "trap"], ["_c", "_k"], None)
    return (w._c / (w._k + TRAP_K)).to_numpy()


def rating(df: pd.DataFrame, q: np.ndarray, te: np.ndarray, prm: dict, upto: int | None = None,
           collect: bool = False) -> tuple[pd.DataFrame, float, dict | None]:
    """The ratings before each row's day (mean, sd, races, the layoff pull) and the mean cross-entropy of the observed
    shares ``q`` under softmax(rating + trap) before each race. Days in order: each day's rows read the state after
    the dog's EARLIER days, then the day's whole races update it (figs.market_rating's loop, with (a), (b), (d))."""
    dog = pd.factorize(df.dog_id)[0]
    dn = _figs._dayn(df)
    race = pd.factorize(df.race_id)[0]
    trial = df.is_trial.fillna(False).to_numpy(bool) if "is_trial" in df else np.zeros(len(df), bool)
    use = np.isfinite(q) & (q > 0) & ~trial
    full = pd.Series(use).groupby(race).transform("all").to_numpy() & (df.field.to_numpy(float) >= 2)
    rb = np.asarray(prm["r_band"], float)[band_of(q)]
    tau = prm.get("tau", np.inf)
    nd = dog.max() + 1 if len(dog) else 0
    R, V, N, LAST, MU = np.zeros(nd), np.full(nd, prm["v0"]), np.zeros(nd), np.full(nd, -1), np.zeros(nd)
    m_out, sd_out, n_out, pull = (np.full(len(df), np.nan), np.full(len(df), np.nan), np.zeros(len(df)),
                                  np.full(len(df), np.nan))
    col = {"innov": [], "vpre": [], "q": []} if collect else None
    order = np.argsort(dn, kind="stable")
    bounds = np.flatnonzero(np.diff(dn[order])) + 1
    ce_sum, ce_n = 0.0, 0

    def pre(dg, today):
        gap = np.where(LAST[dg] >= 0, today - LAST[dg], 0)
        rho = np.exp(-np.clip(gap - GAP0, 0, None) / tau) if np.isfinite(tau) else 1.0
        return MU[dg] + (R[dg] - MU[dg]) * rho, V[dg] + prm["q"] * gap

    for rows in np.split(order, bounds):
        today = dn[rows[0]]
        if upto is not None and today >= upto:
            break
        dg = dog[rows]
        r_pre, v_pre = pre(dg, today)
        has = N[dg] > 0
        m_out[rows] = np.where(has, r_pre, np.nan)
        sd_out[rows] = np.sqrt(v_pre)
        n_out[rows] = N[dg]
        pull[rows] = np.where(has, r_pre - R[dg], np.nan)
        f = rows[full[rows]]
        if not len(f):
            continue
        u, inv = np.unique(race[f], return_inverse=True)
        cnt = np.bincount(inv)
        rp, vp = pre(dog[f], today)
        z = rp + te[f]
        mx = np.full(len(u), -np.inf)
        np.maximum.at(mx, inv, z)
        ez = np.exp(z - mx[inv])
        p = ez / np.bincount(inv, weights=ez)[inv]
        ce_sum += float(-(q[f] * np.log(np.clip(p, 1e-12, None))).sum())
        ce_n += len(u)
        y = np.log(q[f]) - te[f]                                # (a) the trap taken out
        mean_r = np.bincount(inv, weights=rp) / cnt
        x = y - (np.bincount(inv, weights=y) / cnt)[inv] + mean_r[inv]
        dgf = dog[f]
        if collect:
            h = N[dgf] > 0
            col["innov"].append((x - rp)[h])
            col["vpre"].append(vp[h])
            col["q"].append(q[f][h])
        # (b) each reading's own noise; a dog twice in a day: its readings combined by precision
        ud, inv2 = np.unique(dgf, return_inverse=True)
        prec = np.bincount(inv2, weights=1.0 / rb[f])
        xm = np.bincount(inv2, weights=x / rb[f]) / prec
        mu_new = np.bincount(inv2, weights=mean_r[inv]) / np.bincount(inv2)
        rpu, vpu = pre(ud, today)
        k = vpu / (vpu + 1.0 / prec)
        R[ud] = rpu + k * (xm - rpu)
        V[ud] = (1 - k) * vpu
        N[ud] += 1
        LAST[ud] = today
        MU[ud] = mu_new                                         # (d) the field it last met
    if collect:
        col = {k: np.concatenate(v) if v else np.array([]) for k, v in col.items()}
    return (pd.DataFrame({"m": m_out, "sd": sd_out, "n": n_out, "pull": pull}, index=df.index),
            (ce_sum / ce_n if ce_n else np.nan), col)


def error_memory(df: pd.DataFrame, p: np.ndarray, won: np.ndarray, prm: dict,
                 upto: int | None = None) -> tuple[pd.DataFrame, float]:
    """(c) The market's error on each dog: a Kalman level of the result surprise (won less the de-biased chance,
    noise p(1 - p)), drift ``eq`` a day, prior variance ``ev0``. The state before each row's day, and the surprises'
    mean predictive log-likelihood (the fit's score)."""
    dog = pd.factorize(df.dog_id)[0]
    dn = _figs._dayn(df)
    trial = df.is_trial.fillna(False).to_numpy(bool) if "is_trial" in df else np.zeros(len(df), bool)
    use = np.isfinite(p) & np.isfinite(won) & ~trial
    s = np.where(use, won - np.nan_to_num(p), 0.0)
    r = np.clip(np.nan_to_num(p) * (1 - np.nan_to_num(p)), 0.01, None)
    nd = dog.max() + 1 if len(dog) else 0
    S, V, LAST = np.zeros(nd), np.full(nd, prm["ev0"]), np.full(nd, -1)
    m_out, sd_out = np.full(len(df), np.nan), np.full(len(df), np.nan)
    order = np.argsort(dn, kind="stable")
    bounds = np.flatnonzero(np.diff(dn[order])) + 1
    ll, n = 0.0, 0
    for rows in np.split(order, bounds):
        today = dn[rows[0]]
        if upto is not None and today >= upto:
            break
        dg = dog[rows]
        vp = V[dg] + prm["eq"] * np.where(LAST[dg] >= 0, today - LAST[dg], 0)
        m_out[rows] = S[dg]
        sd_out[rows] = np.sqrt(vp)
        f = rows[use[rows]]
        if not len(f):
            continue
        tot = vp[use[rows]] + r[f]
        ll += float((-0.5 * (np.log(2 * np.pi * tot) + (s[f] - S[dog[f]]) ** 2 / tot)).sum())
        n += len(f)
        ud, inv = np.unique(dog[f], return_inverse=True)
        prec = np.bincount(inv, weights=1.0 / r[f])
        sm = np.bincount(inv, weights=s[f] / r[f]) / prec
        vpu = V[ud] + prm["eq"] * np.where(LAST[ud] >= 0, today - LAST[ud], 0)
        k = vpu / (vpu + 1.0 / prec)
        S[ud] = S[ud] + k * (sm - S[ud])
        V[ud] = (1 - k) * vpu
        LAST[ud] = today
    return pd.DataFrame({"m": m_out, "sd": sd_out}, index=df.index), (ll / n if n else np.nan)


def fit_rating(df: pd.DataFrame, q: np.ndarray, te: np.ndarray, cut: int | None) -> dict:
    """Stage 1: one noise level, no reversion (figs' grid); stage 2: the noise by share band from the early
    innovations; stage 3: the bands' scale and the reversion tau. All on the days before ``cut``."""
    if cut is None:
        return {**DEFAULT, "fitted": False}
    nb = len(BANDS) - 1
    best = None
    for r in GRID_R:
        for qq in GRID_Q:
            for v0 in GRID_V0:
                prm = {"r_band": [r] * nb, "q": qq, "v0": v0, "tau": np.inf}
                ce = rating(df, q, te, prm, upto=cut)[1]
                if np.isfinite(ce) and (best is None or ce < best[0]):
                    best = (ce, prm)
    if best is None:
        return {**DEFAULT, "fitted": False}
    stage1 = best
    col = rating(df, q, te, best[1], upto=cut, collect=True)[2]
    b = band_of(col["q"])
    rband, counts = [], []
    for i in range(nb):
        k = b == i
        counts.append(int(k.sum()))
        rband.append(float(max(np.var(col["innov"][k]) - col["vpre"][k].mean(), R_FLOOR)) if k.sum() >= MIN_BAND
                     else best[1]["r_band"][i])
    for sc in GRID_SCALE:
        for tau in GRID_TAU:
            prm = {**best[1], "r_band": [sc * x for x in rband], "tau": tau}
            ce = rating(df, q, te, prm, upto=cut)[1]
            if np.isfinite(ce) and ce < best[0]:
                best = (ce, prm)
    return {**best[1], "fitted": True, "ce": round(best[0], 6), "ce_stage1": round(stage1[0], 6),
            "band_counts": counts, "band_r_raw": [round(x, 4) for x in rband]}


def fit_error(df: pd.DataFrame, p: np.ndarray, won: np.ndarray, cut: int | None) -> dict:
    if cut is None:
        return {"eq": 1e-5, "ev0": 0.001, "fitted": False}
    best = None
    for eq in GRID_EQ:
        for ev0 in GRID_EV0:
            prm = {"eq": eq, "ev0": ev0}
            ll = error_memory(df, p, won, prm, upto=cut)[1]
            if np.isfinite(ll) and (best is None or ll > best[0]):
                best = (ll, prm)
    if best is None:
        return {"eq": 1e-5, "ev0": 0.001, "fitted": False}
    return {**best[1], "fitted": True, "ll": round(best[0], 6)}


def _columns(out: pd.DataFrame, name: str, mr: pd.DataFrame, te: np.ndarray, race: pd.Series) -> None:
    filled = mr.m.fillna(mr.m.groupby(race).transform("mean")).fillna(0.0)
    g = filled.groupby(race)
    out[f"{PREFIX}{name}"] = mr.m
    out[f"{PREFIX}{name}_sd"] = mr.sd
    out[f"{PREFIX}{name}_vs_field"] = (filled - (g.transform("sum") - filled) /
                                       (g.transform("size") - 1).replace(0, np.nan)).where(mr.m.notna())
    sh = _figs.race_softmax(filled + te, race, 1.0)
    out[f"{PREFIX}{name}_share"] = sh
    out[f"{PREFIX}{name}_lshare"] = np.log(sh.clip(1e-9))


def _clean(prm: dict) -> dict:
    return {k: ("inf" if isinstance(v, float) and not np.isfinite(v) else v) for k, v in prm.items()}


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = runs.reset_index(drop=True)
    race = df.race_id
    trial = df.is_trial.fillna(False).to_numpy(bool) if "is_trial" in df else np.zeros(len(df), bool)
    won = pd.to_numeric(df.won, errors="coerce").to_numpy(float)
    q = shin_probs(df.sp_decimal, df.race_id)
    use = np.isfinite(q) & (q > 0) & ~trial
    full = pd.Series(use).groupby(race.to_numpy()).transform("all").to_numpy() & (df.field.to_numpy(float) >= 2)
    dn = _figs._dayn(df)
    cut = _early_cut(dn, full)
    te = trap_effect(df, q, full)
    prm = fit_rating(df, q, te, cut)
    mr, ce, _ = rating(df, q, te, prm)
    out = pd.DataFrame({"race_id": race.to_numpy(), "dog_id": df.dog_id.to_numpy()})
    _columns(out, "mr", mr, te, race)
    out[f"{PREFIX}mr_n"] = mr.n
    out[f"{PREFIX}mr_pull"] = mr.pull
    out[f"{PREFIX}trap"] = te
    ep = fit_error(df, q, won, cut)
    em, ell = error_memory(df, q, won, ep)
    out[f"{PREFIX}err"] = em.m
    out[f"{PREFIX}err_sd"] = em.sd
    out[f"{PREFIX}err_z"] = em.m / em.sd
    notes = {"early_cut": str(pd.Timestamp("2000-01-01") + pd.Timedelta(days=cut)) if cut is not None else None,
             "rating": _clean(prm), "ce_all": round(ce, 6) if np.isfinite(ce) else None,
             "error": {**ep, "ll_all": round(ell, 6) if np.isfinite(ell) else None}}
    # the same rating on the Betfair SP where the race is wholly priced by it, the de-biased SP elsewhere (figs' bmr)
    prices = (ctx or {}).get("prices")
    bq = None
    if "bsp_p_norm" in df:
        bq = pd.to_numeric(df.bsp_p_norm, errors="coerce")
    elif prices is not None and len(prices):
        from greyhound.parity import attach_bsp
        b = attach_bsp(df[["race_date", "race_time", "track", "trap", "race_id"]], prices)
        bq = pd.Series(b.bsp_p_norm.to_numpy(), index=df.index)
    if bq is not None:
        whole = bq.notna().groupby(race).transform("all")
        qb = bq.where(whole, pd.Series(q, index=df.index)).to_numpy(float)
        fullb = pd.Series(np.isfinite(qb) & (qb > 0) & ~trial).groupby(race.to_numpy()).transform("all").to_numpy()
        teb = trap_effect(df, qb, fullb & (df.field.to_numpy(float) >= 2))
        br, bce, _ = rating(df, qb, teb, prm)
        _columns(out, "bmr", br, teb, race)
        notes["bsp_rating"] = {"races_on_bsp": int(whole.groupby(race).first().sum()),
                               "ce_all": round(bce, 6) if np.isfinite(bce) else None}
    else:
        for c in ("bmr", "bmr_sd", "bmr_vs_field", "bmr_share", "bmr_lshare"):
            out[f"{PREFIX}{c}"] = np.nan
    cols = [c for c in out.columns if c.startswith(PREFIX)]
    out[cols] = out[cols].astype(np.float32)
    return out, notes


COLUMNS = {
    f"{PREFIX}mr": "the market-implied rating: the log de-biased SP share less the track-trip-trap mean (earlier "
                   "days), race-relative, opponent-adjusted, Kalman-filtered with share-dependent noise and reversion "
                   "after layoffs",
    f"{PREFIX}mr_sd": "the rating's spread before the race",
    f"{PREFIX}mr_vs_field": "the rating against the mean of the race's other runners",
    f"{PREFIX}mr_share": "its forecast share: a softmax of the field's ratings plus each trap's mean",
    f"{PREFIX}mr_lshare": "the log of that share",
    f"{PREFIX}mr_n": "races behind the rating",
    f"{PREFIX}mr_pull": "how far the layoff reversion moved the rating toward the field it last met",
    f"{PREFIX}trap": "today's track-trip-trap mean of the race-centred log de-biased share (earlier days, shrunk)",
    f"{PREFIX}err": "the market's error memory: a Kalman level of won less the de-biased SP's chance (per run)",
    f"{PREFIX}err_sd": "its spread",
    f"{PREFIX}err_z": "the error memory over its spread",
    f"{PREFIX}bmr": "the rating read on the Betfair SP where the race is wholly priced by it, the de-biased SP "
                    "elsewhere (the same constants)",
    f"{PREFIX}bmr_sd": "its spread",
    f"{PREFIX}bmr_vs_field": "it against the mean of the race's other runners",
    f"{PREFIX}bmr_share": "its forecast share (softmax of ratings plus the trap's mean on that share)",
    f"{PREFIX}bmr_lshare": "the log of that share",
}
