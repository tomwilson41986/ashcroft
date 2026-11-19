"""Ability with the trials fed in (the feature review of 17 Nov 2026, B2 and the "ability" row; a research variant of
ability.py).

ability.py's filter has a trial noise that never fires: data.clean drops every trial (no off time), so the filter
never sees one. This table runs the same Kalman local level (ability.filter_dog_states, its q, r and debut prior
fitted by ability.estimate) with:

- **the trials as observations**: each trial's gsr, esr and fsr on the race scale (trials2.race_scale: the engine's
  standard, the walk-forward trial-to-race offset by track, trip and solo or company, less the session's going), with
  ability.py's 1.5 x run noise. A trial is an observation, never a row of the table.
- **noise per trip band** (``BANDS``): q and r a band, by moments of the one-step-ahead innovations on the early runs
  (e^2 = p + q x days + r x noise multiple, least squares a band, held within a factor of the pooled fit).
- **a damped level**: before a day's runs read it, the level is pulled towards the grade par (the race class's mean
  figure over earlier days, shrunk by ``K_PAR`` runs to the mean of all) by exp(-days off / tau): the longer the
  layoff the more it reverts.

Which of these helps is chosen on the early runs (ability.FIT_BEFORE, or the first fifth of the days) by the
one-step-ahead predictive likelihood of the races (bands on or off, tau from ``TAUS``); the meta records the grid.
Every column reads the state after the dog's EARLIER days; all are card-safe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import ability, core
from greyhound.pit import trials as tl
from greyhound.pit import trials2

PREFIX = "ka_"
SOURCE = ("GBGB results through the metrics engine's per-run figures (greyhound/metrics.py) and the trial rows "
          "data.clean drops (ctx['trial_runs']), the trials on the race scale (greyhound/pit/trials2.py)")
DEPENDS = ("ability", "trials", "trials2")
FIGURES = ability.FIGURES
BANDS = (400.0, 550.0, 700.0)          # sprint, standard, stayer, marathon
TAUS = (None, 90.0, 180.0, 365.0, 730.0)
K_PAR = 50.0
MIN_BAND_ROWS = 500
MAX_DOGS = 30_000


def grade_par(races: pd.DataFrame, fig: str) -> pd.Series:
    """Each race row's class par for ``fig``: the class's mean figure over earlier days, shrunk by K_PAR runs to the
    mean of every run of earlier days."""
    v = pd.to_numeric(races[fig], errors="coerce")
    d = pd.DataFrame({"race_date": races.race_date.to_numpy(), "race_class": races.race_class.astype(str).to_numpy(),
                      "_all": "all", "s": v.fillna(0.0).to_numpy(), "h": v.notna().to_numpy(float)}, index=races.index)
    c = core.prior_window(d, ["race_class"], ["s", "h"], None)
    a = core.prior_window(d, ["_all"], ["s", "h"], None)
    glob = (a.s / a.h.where(a.h > 0)).fillna(0.0)
    return ((c.s + K_PAR * glob) / (c.h + K_PAR)).where(a.h > 0)


def early_mask(frame: pd.DataFrame) -> np.ndarray:
    """ability.estimate's early slice: the rows before ability.FIT_BEFORE, or the first fifth when too few."""
    early = (frame.race_date < ability.FIT_BEFORE).to_numpy()
    if early.sum() < 50_000 and len(frame):
        cut = frame.race_date.sort_values().iloc[int(len(frame) * 0.2)]
        early = (frame.race_date < cut).to_numpy()
    return early


def _loglik(e: np.ndarray, s2: np.ndarray, use: np.ndarray) -> float:
    ok = use & np.isfinite(e) & (s2 > 0)
    return float(np.mean(-0.5 * (np.log(2 * np.pi * s2[ok]) + e[ok] ** 2 / s2[ok]))) if ok.any() else -np.inf


def fit(frame: pd.DataFrame, fig: str, prm: dict, seed: int = 0) -> dict:
    """Band q and r by moments, then the choice of bands and tau by the races' predictive likelihood (early rows)."""
    e_all = frame[early_mask(frame)]
    dogs = e_all.dog_id.unique()
    if len(dogs) > MAX_DOGS:
        e_all = e_all[e_all.dog_id.isin(np.random.default_rng(seed).choice(dogs, MAX_DOGS, replace=False))]
    if len(e_all) < 200:
        return {"bands": None, "tau": None, "fit": "default (too few early rows)"}
    dog, dayn = e_all.dog_id.to_numpy(), e_all.dn.to_numpy(np.int64)
    obs, trial = e_all[fig].to_numpy(float), e_all.trial.to_numpy(bool)
    band, par = e_all.band.to_numpy(int), e_all[f"par_{fig}"].to_numpy(float)
    q, r = prm["q"], prm["r"]
    mult = np.where(trial, 1.5, 1.0)
    m, v, _, since = ability.filter_dog_states(dog, dayn, obs, trial, prm)
    gap = np.nan_to_num(since)
    y = (obs - m) ** 2 - (v - q * gap)
    qb, rb = {}, {}
    for b in range(len(BANDS) + 1):
        ok = (band == b) & np.isfinite(y)
        if ok.sum() < MIN_BAND_ROWS:
            qb[b], rb[b] = q, r
            continue
        coef = np.linalg.lstsq(np.c_[gap[ok], mult[ok]], y[ok], rcond=None)[0]
        qb[b] = float(np.clip(coef[0], q / 5, q * 5))
        rb[b] = float(np.clip(coef[1], r / 3, r * 3))
    races = ~trial
    grid, best = [], None
    for use_bands in (False, True):
        q_row = np.array([qb[b] for b in range(len(BANDS) + 1)])[band] if use_bands else None
        r_row = np.array([rb[b] for b in range(len(BANDS) + 1)])[band] if use_bands else None
        for tau in TAUS:
            mm, vv, _, _ = ability.filter_dog_states(dog, dayn, obs, trial, prm, q_row=q_row, r_row=r_row,
                                                     par=par if tau else None, tau=tau)
            s2 = vv + (r_row if use_bands else r) * mult
            ll = _loglik(obs - mm, s2, races)
            grid.append((use_bands, tau, round(ll, 5)))
            if best is None or ll > best[0]:
                best = (ll, use_bands, tau)
    return {"bands": {"q": qb, "r": rb} if best[1] else None, "tau": best[2], "band_q": qb, "band_r": rb,
            "loglik": round(best[0], 5), "grid": grid, "rows": int(len(e_all))}


def cached_params(ctx: dict | None) -> dict | None:
    """The nightly build's fitted constants (``ctx["fitted"]["ability2"]``: the stored table's notes), with the
    band keys back to integers (JSON keeps them as strings); None when absent or incomplete."""
    c = ((ctx or {}).get("fitted") or {}).get("ability2")
    if not c or "params" not in c or "race_noise" not in c or not all(f in c["params"] for f in FIGURES):
        return None
    def num(x):                               # the meta is written with default=str: numbers may come back as text
        if isinstance(x, dict):
            return {k: num(v) for k, v in x.items()}
        if isinstance(x, list):
            return [num(v) for v in x]
        if isinstance(x, str):
            try:
                return float(x)
            except ValueError:
                return x
        return x

    params = {}
    for f in FIGURES:
        p = num(dict(c["params"][f]))
        if p.get("bands"):
            p["bands"] = {k: {int(b): float(v) for b, v in p["bands"][k].items()} for k in ("q", "r")}
        params[f] = p
    return {"params": params, "race_noise": float(num(c["race_noise"]))}


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = figs if figs is not None else core.per_run_figures(runs)
    in_runs = tl.is_trial_class(df.race_class).to_numpy()
    races = df[~in_runs]
    tr = trials2.trial_rows(df, ctx)
    notes = {"trial_rows": int(len(tr))}
    dn = lambda s: tl._dn(pd.Series(s)).astype(float)                    # noqa: E731
    # the frame the filter runs on: every row of df (a T-class row of df reads, never observes) and every trial
    rows = pd.DataFrame({"dog_id": df.dog_id.to_numpy(), "race_date": df.race_date.astype(str).to_numpy(),
                         "dn": dn(df.race_date.to_numpy()), "t": pd.to_datetime(df.t, errors="coerce").to_numpy(),
                         "trial": False, "dist": pd.to_numeric(df.distance_m, errors="coerce").to_numpy(),
                         "row": np.arange(len(df))})
    for f in FIGURES:
        rows[f] = np.where(in_runs, np.nan, pd.to_numeric(df[f], errors="coerce").to_numpy(float))
        par = np.full(len(df), np.nan)
        if (~in_runs).any():
            par[~in_runs] = grade_par(races, f).to_numpy(float)
        rows[f"par_{f}"] = par
    parts = [rows]
    if len(tr):
        tr, tfit = trials2.race_scale(tr, races, FIGURES)
        notes["trial_fit"] = tfit
        day = pd.to_datetime(tr.race_date, errors="coerce").dt.normalize()
        t = pd.DataFrame({"dog_id": tr.dog_id.to_numpy(), "race_date": tr.race_date.astype(str).to_numpy(),
                          "dn": tr.dn.to_numpy(float), "t": (day + pd.Timedelta(hours=23, minutes=59)).to_numpy(),
                          "trial": True, "dist": pd.to_numeric(tr.distance_m, errors="coerce").to_numpy(), "row": -1})
        for f in FIGURES:
            t[f] = tr[f"{f}_rs"].to_numpy(float)
            t[f"par_{f}"] = np.nan
        parts.append(t[np.isfinite(t.dn)])
    fr = pd.concat(parts, ignore_index=True)
    fr = fr[np.isfinite(fr.dn)]
    fr["dn"] = fr.dn.astype(np.int64)
    fr["band"] = np.digitize(np.nan_to_num(fr.dist.to_numpy(float), nan=500.0), BANDS)
    fr = fr.sort_values(["dog_id", "dn", "trial", "t"], kind="stable").reset_index(drop=True)
    dog, dayn, trial, band = fr.dog_id.to_numpy(), fr.dn.to_numpy(np.int64), fr.trial.to_numpy(bool), fr.band.to_numpy()
    est_frame = fr.assign(is_trial=fr.trial)
    res = {}
    params = {}
    # the constants are fitted on the early slice only (before ability.FIT_BEFORE), which a later day never changes:
    # the price run reads the nightly build's (``ctx["fitted"]["ability2"]``, the stored table's notes from this
    # code) instead of fitting them again (the owner, 18 Nov 2026: "yes, cache the constants")
    cached = cached_params(ctx)
    notes["constants"] = "cached" if cached else "fitted"
    for f in FIGURES:
        if cached:
            params[f] = cached["params"][f]
        else:
            prm = ability.estimate(est_frame, f)
            ext = fit(fr, f, prm)
            params[f] = {**{k: v for k, v in prm.items() if k != "grid"}, **ext}
        prm, ext = params[f], params[f]
        bands = ext.get("bands")
        q_row = np.array([bands["q"][b] for b in range(len(BANDS) + 1)])[band] if bands else None
        r_row = np.array([bands["r"][b] for b in range(len(BANDS) + 1)])[band] if bands else None
        tau = ext.get("tau")
        m, v, c, since = ability.filter_dog_states(dog, dayn, fr[f].to_numpy(float), trial, prm, q_row=q_row,
                                                   r_row=r_row, par=fr[f"par_{f}"].to_numpy(float) if tau else None,
                                                   tau=tau)
        res[f] = (m, v, c, since)
    keep = (fr.row >= 0).to_numpy()
    at = fr.row.to_numpy()[keep]
    out = pd.DataFrame({"race_id": df.race_id.to_numpy(), "dog_id": df.dog_id.to_numpy()}, index=df.index)
    for c in COLUMNS:
        out[c] = np.nan

    def put(col: str, vals: np.ndarray) -> None:
        a = np.full(len(df), np.nan)
        a[at] = vals[keep]
        out[col] = a

    for f in FIGURES:
        m, v, c, since = res[f]
        put(f"{PREFIX}{f}_mean", m)
        put(f"{PREFIX}{f}_sd", np.sqrt(v))
        if f == "gsr":
            put(f"{PREFIX}n", c)
            put(f"{PREFIX}days", since)
            put(f"{PREFIX}gsr_par", fr.par_gsr.to_numpy(float))
    # trial figures seen before the day: the dog's trial days with a gsr, before
    obs_t = trial & np.isfinite(fr.gsr.to_numpy(float))
    tk = np.unique(dog[obs_t].astype(np.int64) * trials2.BIG + dayn[obs_t])
    qd = df.dog_id.to_numpy(np.int64)
    qn = np.nan_to_num(tl._dn(df.race_date).astype(float), nan=0.0).astype(np.int64)
    out[f"{PREFIX}n_trials"] = (np.searchsorted(tk, qd * trials2.BIG + qn) -
                                np.searchsorted(tk, qd * trials2.BIG)).astype(float)
    out[f"{PREFIX}gsr_vs_par"] = out[f"{PREFIX}gsr_mean"] - out[f"{PREFIX}gsr_par"]
    g = out.groupby("race_id")[f"{PREFIX}gsr_mean"]
    out[f"{PREFIX}gsr_gap"] = g.transform("max") - out[f"{PREFIX}gsr_mean"]
    out[f"{PREFIX}gsr_rank"] = g.rank(ascending=False, method="average")
    r = params["gsr"]["r"]
    tmp = out[["race_id"]].assign(**{f"{ability.PREFIX}gsr_mean": out[f"{PREFIX}gsr_mean"],
                                     f"{ability.PREFIX}gsr_sd": out[f"{PREFIX}gsr_sd"]})
    k, kgrid = (cached["race_noise"], "cached") if cached else ability.race_noise(tmp, runs, r)
    out[f"{PREFIX}gsr_pbest"] = ability.p_best(out.race_id.to_numpy(), out[f"{PREFIX}gsr_mean"].to_numpy(float),
                                               out[f"{PREFIX}gsr_sd"].to_numpy(float), r * k)
    out[list(COLUMNS)] = out[list(COLUMNS)].astype(np.float32)
    notes.update({"params": params, "fit_before": ability.FIT_BEFORE, "race_noise": k, "race_noise_grid": kgrid,
                  "rows_with_a_trial_before": round(float((out[f"{PREFIX}n_trials"] > 0).mean()), 4)})
    return out.sort_index().reset_index(drop=True), notes


COLUMNS = {
    **{f"{PREFIX}{f}_mean": f"the dog's {f} level before the run, trials included (Kalman local level, damped to the "
                            f"grade par over layoffs when the early fit chose it; lengths)" for f in FIGURES},
    **{f"{PREFIX}{f}_sd": "the spread of that level" for f in FIGURES},
    f"{PREFIX}n": "figures the filter has seen, trials included",
    f"{PREFIX}n_trials": "trial days with a figure before today",
    f"{PREFIX}days": "days since the dog's last figure, trial or race",
    f"{PREFIX}gsr_par": "the race class's par gsr (earlier days, shrunk to the mean of all)",
    f"{PREFIX}gsr_vs_par": "the level less that par",
    f"{PREFIX}gsr_gap": "lengths behind the field's best level",
    f"{PREFIX}gsr_rank": "the level's rank in the field (1 the best)",
    f"{PREFIX}gsr_pbest": "chance this dog runs the field's best figure (ability.p_best, the race's noise fitted on "
                          "early races)",
}
