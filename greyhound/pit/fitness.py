"""Fitness and fatigue: an impulse response over the dog's whole load history, trials included (the owner, 11 Nov
2026: "identify NEW features that have not been discovered yet").

The idea. Sports science models performance as a baseline plus two responses to every past exertion: a slow positive
one (fitness) and a fast negative one (fatigue): the Banister impulse-response model. Racing models, and this
repository, read the load as days since the last run and a count of runs in a window. Nobody convolves the whole
history of outings with response kernels: two hard runs three and seven days back are not one run five days back,
and a dog with a long steady base is not a dog that has run as often for a fortnight only.

The mechanism. A sprint leaves a cost that fades in days (soreness, muscle damage, worst after a check or a bump)
and a benefit that fades in months (race fitness). Today's readiness is the sum of both over every outing, so it
depends on the dose and the spacing of all of them, the trials too (a trial is an outing no race feature sees:
data.clean drops it).

The construction.

- **Outings**: the race rows of the runs (not today's card rows, which have not happened) and the trial rows of
  ``ctx["trial_runs"]`` when given, one row a dog-day (a day's outings added).
- **Doses** of an outing: ``out`` = 1; ``m`` = distance_m / 480; ``hit`` = the number of interference tokens in its
  comment (``greyhound.pit.xtime.TROUBLE`` on the tokens of ``greyhound.pit.trouble.codes``); ``trial`` = 1 for a
  trial (a row of the trial channel, or a T class row in the runs).
- **States**: for tau in 4, 10, 21, 45, 90 and 180 days the state before today is the sum over the dog's EARLIER
  days of dose x exp(-(days ago) / tau) (``greyhound.pit.figs.decayed`` with a half-life of tau x ln 2). Thirteen
  of them: ``out`` at every tau, ``m`` at 10, 45, 180, ``hit`` at 10, 45, ``trial`` at 21, 90. A silence of
  ``ERA_GAPS`` time constants starts a state afresh (what is dropped is below exp(-60) of a dose; it keeps the
  running sums inside a float for a dog whose record spans many years).
- ``ff_tsb`` = ff_out_45 / 45 - ff_out_10 / 10: the classic fitness less fatigue, each as a daily rate.
  ``ff_mono`` = ff_out_10 / (ff_out_45 + 1): the acute load against the chronic.
- ``ff_pred``: a FITTED response, the sign not imposed. Every ``REFIT_MONTHS`` a ridge regression (``LAMBDA``, an
  intercept that is not shrunk) of the run's residual (``greyhound.pit.dyad.residuals``: its calculated time less
  the expected-time table's ``xt_tot_time``, in lengths, above 0 slower than expected) on the thirteen states,
  standardised by the fit rows' mean and sd, fitted on the graded runs of the periods BEFORE it only (at least
  ``MIN_FIT`` of them) and read on that period's rows: the lengths slower (above 0) or faster (below 0) than its
  expected time the dog's load says it will run. NaN before the first fit.
- **The burn-in.** A record starts in the middle of every dog's career, so for its first months the states are
  short of the outings before it: a run in the first ``BURN_DAYS`` days after the record's first race day (the
  longest time constant) joins no fit. It still gets its states and, once there is a fit, a response. Measured on
  the GBGB record of 1 Jul 2025 to 5 Nov 2026 cut to start on 1 Jan 2026 (the review of 11 Nov 2026): the mean
  ``ff_out_180`` of the cut record is 11% of the whole record's in its first 30 days, 49% in days 60 to 90, 76% in
  days 150 to 180 and 90% in days 240 to 300, and a fit that read those first months (rows before Jul 2026, read on
  the 84,469 runs after) had a correlation of 0.0557 with the residual against 0.0902 without them. A record
  shorter than the burn-in and one refit period has no fitted response at all (that record at 12 months: none).

Every column is read from earlier days only: an outing enters the states of later days, a run's residual enters the
fits of later periods (tests/test_greyhound_fitness.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core, figs as figs_mod

PREFIX = "ff_"
SOURCE = ("GBGB results (the days each dog raced, distance, comment: sources/gbgb.py), the trial rows data.clean "
          "drops, and the expected-time table (xt_tot_time) for the fitted response")
TAUS = (4, 10, 21, 45, 90, 180)
DOSES = {"out": TAUS, "m": (10, 45, 180), "hit": (10, 45), "trial": (21, 90)}
STATES = [f"{PREFIX}{d}_{tau}" for d in ("out", "m", "hit", "trial") for tau in DOSES[d]]
REF_M = 480.0                # a standard trip: the metres dose of one outing over it is 1
ERA_GAPS = 60                # a silence of this many time constants starts a state afresh
LAMBDA = 10.0
MIN_FIT = 50_000
REFIT_MONTHS = 12
BURN_DAYS = 180              # a run this soon after the record's first race day joins no fit (states still filling)
EPOCH = pd.Timestamp("2000-01-01")


def hits(comment: pd.Series) -> np.ndarray:
    """The number of interference tokens in each comment (crowded, bumped, baulked, hampered, checked ...)."""
    from greyhound.pit import trouble, xtime
    c = comment.fillna("").astype(str)
    n = {u: sum(1 for t in trouble.codes(u) if xtime.TROUBLE.search(t)) for u in pd.unique(c)}
    return c.map(n).to_numpy(float)


def outings(df: pd.DataFrame, trial_runs: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    """The frame the states are read from: every row of ``df`` first, in its order (a card row with no dose), then
    the trial rows that are not in ``df``. Columns: dog_id, race_date (a day), w (1 for an outing that happened),
    m, hit, trial."""
    from greyhound.pit import trials
    card = df.is_card.fillna(False).to_numpy(bool) if "is_card" in df else np.zeros(len(df), bool)
    t_class = trials.is_trial_class(df.race_class).to_numpy(bool) if "race_class" in df else np.zeros(len(df), bool)
    comment = df.comment if "comment" in df else pd.Series("", index=df.index)
    a = pd.DataFrame({"dog_id": df.dog_id.to_numpy(), "race_date": core.day(df).to_numpy(),
                      "w": (~card).astype(float),
                      "m": (pd.to_numeric(df.distance_m, errors="coerce") / REF_M).fillna(0.0).to_numpy(),
                      "hit": hits(comment), "trial": t_class.astype(float)})
    notes = {"race_outings": int((~card & ~t_class).sum()), "trial_rows_in_runs": int((~card & t_class).sum()),
             "card_rows": int(card.sum()), "trial_outings": 0}
    tr = trial_runs
    if tr is not None and len(tr):
        tr = tr[tr.dog_id.notna() & tr.race_date.notna()]
        if "race_id" in tr:
            tr = tr.drop_duplicates(["race_id", "dog_id"], keep="last")
            seen = pd.MultiIndex.from_arrays([df.race_id.to_numpy(), df.dog_id.to_numpy()])
            tr = tr[~pd.MultiIndex.from_arrays([tr.race_id.to_numpy(), tr.dog_id.to_numpy()]).isin(seen)]
        day = pd.to_datetime(tr.race_date, errors="coerce").dt.normalize()
        tr = tr[day.notna().to_numpy()]
        day = day[day.notna()]
        dist = pd.to_numeric(tr.distance_m, errors="coerce") if "distance_m" in tr else pd.Series(np.nan, index=tr.index)
        b = pd.DataFrame({"dog_id": tr.dog_id.astype(a.dog_id.dtype).to_numpy(), "race_date": day.to_numpy(), "w": 1.0,
                          "m": (dist / REF_M).fillna(0.0).to_numpy(),
                          "hit": hits(tr.comment) if "comment" in tr else 0.0, "trial": 1.0})
        notes["trial_outings"] = int(len(b))
        a = pd.concat([a, b], ignore_index=True)
    happened = a[a.w > 0]
    notes["outing_dog_days"] = int(len(happened.drop_duplicates(["dog_id", "race_date"])))
    notes["interference_tokens_an_outing"] = round(float(happened.hit.mean()), 4) if len(happened) else None
    return a, notes


def states(frame: pd.DataFrame) -> pd.DataFrame:
    """The thirteen states of every row of ``frame`` (``outings``): the dog's decayed dose sums over EARLIER days."""
    dog = pd.factorize(frame.dog_id)[0]
    dn = (frame.race_date - EPOCH).dt.days.to_numpy(np.int64)
    order = np.lexsort((dn, dog))
    first = np.r_[True, dog[order][1:] != dog[order][:-1]]
    gap = np.r_[0, np.diff(dn[order])]
    one = pd.Series(1.0, index=frame.index)
    out = {}
    for tau in TAUS:                                              # six constants; no loop over rows or dogs
        era = np.empty(len(frame), np.int64)
        era[order] = np.cumsum(first | (gap > ERA_GAPS * tau))
        key = pd.DataFrame({"dog_id": era, "race_date": frame.race_date.to_numpy()}, index=frame.index)
        extra = [d for d in ("m", "hit", "trial") if tau in DOSES[d]] or [None]
        for i, d in enumerate(extra):
            s = figs_mod.decayed(key, frame[d] if d else one, frame.w, tau * np.log(2.0))
            if i == 0:
                out[f"{PREFIX}out_{tau}"] = s.sw.to_numpy(float)
            if d:
                out[f"{PREFIX}{d}_{tau}"] = s.swx.to_numpy(float)
    return pd.DataFrame({c: out[c] for c in STATES}, index=frame.index)


def _label(period: int) -> str:
    month = int(period) * REFIT_MONTHS
    return str(month // 12) if REFIT_MONTHS == 12 else f"{month // 12}-{month % 12 + 1:02d}"


def fit_predict(X: np.ndarray, r: np.ndarray, period: np.ndarray) -> tuple[np.ndarray, dict]:
    """Each period's rows read by a ridge of ``r`` on the standardised columns of ``X`` fitted on the rows of the
    periods before it with a residual (from their running moments: the fit of period p never sees a row of p)."""
    k = X.shape[1]
    pred = np.full(len(r), np.nan)
    ok = np.isfinite(r) & np.isfinite(X).all(axis=1)
    n, sx, sxx, sy, sxy = 0, np.zeros(k), np.zeros((k, k)), 0.0, np.zeros(k)
    fits = {}
    for p in np.unique(period):                                    # a handful of refit periods
        rows = period == p
        if n >= MIN_FIT:
            mu, ybar = sx / n, sy / n
            sd = np.sqrt(np.clip(np.diag(sxx) / n - mu ** 2, 0.0, None))
            inv = np.divide(1.0, sd, out=np.zeros(k), where=sd > 1e-9)        # a column that never moved reads 0
            zz = (sxx - n * np.outer(mu, mu)) * np.outer(inv, inv)
            zy = (sxy - n * mu * ybar) * inv
            b = np.linalg.solve(zz + LAMBDA * np.eye(k), zy)
            pred[rows] = ybar + ((X[rows] - mu) * inv) @ b
            fits[_label(p)] = {"rows": int(n), "intercept": round(float(ybar), 4),
                               "lengths_a_sd": {c: round(float(v), 4) for c, v in zip(STATES, b)},
                               "sd": {c: round(float(v), 4) for c, v in zip(STATES, sd)}}
        f = rows & ok
        if f.any():
            xf, rf = X[f], r[f]
            n += int(f.sum())
            sx += xf.sum(axis=0)
            sxx += xf.T @ xf
            sy += float(rf.sum())
            sxy += xf.T @ rf
    return pred, fits


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    from greyhound.pit import dyad
    df = figs if figs is not None else core.per_run_figures(runs)
    xt = (ctx or {}).get("xtime")
    if xt is None:
        from greyhound.pit import xtime
        xt, _ = xtime.build(runs, df)
    frame, notes = outings(df, (ctx or {}).get("trial_runs"))
    st = states(frame).iloc[:len(df)]                              # the race and card rows back, in df's order
    X = st.to_numpy(float)
    out = pd.DataFrame({"race_id": df.race_id.to_numpy(), "dog_id": df.dog_id.to_numpy()}, index=df.index)
    for j, c in enumerate(STATES):
        out[c] = X[:, j]
    o10, o45 = out[f"{PREFIX}out_10"], out[f"{PREFIX}out_45"]
    out[f"{PREFIX}tsb"] = o45 / 45.0 - o10 / 10.0
    out[f"{PREFIX}mono"] = o10 / (o45 + 1.0)
    # the fitted response: the residual is the run's own result, so it is read by the fits of LATER periods only
    j = df[["race_id", "dog_id"]].merge(xt[["race_id", "dog_id", "xt_tot_time"]].drop_duplicates(["race_id", "dog_id"]),
                                        on=["race_id", "dog_id"], how="left")
    r = dyad.residuals(df, pd.Series(j.xt_tot_time.to_numpy(), index=df.index)).to_numpy(float)
    card = df.is_card.fillna(False).to_numpy(bool) if "is_card" in df else np.zeros(len(df), bool)
    r = np.where(card, np.nan, r)
    day = core.day(df)
    period = ((day.dt.year * 12 + day.dt.month - 1) // REFIT_MONTHS).fillna(-1).to_numpy(np.int64)
    # the burn-in: the record's first race day is a fact of the record, the same whatever any day's results are
    first = day[~card].min()
    burn = ((day - first).dt.days < BURN_DAYS).to_numpy(bool) if pd.notna(first) else np.zeros(len(df), bool)
    pred, fits = fit_predict(X, np.where(burn, np.nan, r), period)
    out[f"{PREFIX}pred"] = pred
    cols = [c for c in out.columns if c.startswith(PREFIX)]
    out[cols] = out[cols].astype(np.float32)
    year = day.dt.year.to_numpy()
    read = np.isfinite(pred) & np.isfinite(r)
    by_year = {}
    for y in sorted(set(year[read])):                              # out of sample by construction
        m = read & (year == y)
        by_year[int(y)] = {"runs": int(m.sum()),
                           "corr_with_the_residual": round(float(np.corrcoef(pred[m], r[m])[0, 1]), 4)
                           if m.sum() > 2 and pred[m].std() > 0 else None,
                           "rmse_lengths": round(float(np.sqrt((r[m] ** 2).mean())), 4),
                           "rmse_less_the_response": round(float(np.sqrt(((r[m] - pred[m]) ** 2).mean())), 4)}
    notes.update({"taus": list(TAUS), "ridge_lambda": LAMBDA, "min_fit": MIN_FIT, "refit_months": REFIT_MONTHS,
                  "burn_days": BURN_DAYS, "first_race_day": None if pd.isna(first) else str(first.date()),
                  "residual_rows": int(np.isfinite(r).sum()),
                  "residual_rows_in_the_burn_in": int((np.isfinite(r) & burn).sum()),
                  "rows_with_a_state_not_finite": int((~np.isfinite(X).all(axis=1)).sum()),
                  "rows_with_no_earlier_outing": int((X[:, STATES.index(f"{PREFIX}out_180")] == 0).sum()),
                  "rows_with_a_fitted_response": int(np.isfinite(pred).sum()),
                  "fits": fits, "read_by_year": by_year})
    return out.sort_index().reset_index(drop=True), notes


COLUMNS = {
    **{f"{PREFIX}out_{tau}": f"outings (races and trials) on earlier days, each weighted exp(-days ago / {tau})"
       for tau in DOSES["out"]},
    **{f"{PREFIX}m_{tau}": f"the same of each outing's distance over {REF_M:g} m, time constant {tau} days"
       for tau in DOSES["m"]},
    **{f"{PREFIX}hit_{tau}": f"the same of each outing's interference tokens (crowded, bumped, baulked, hampered, "
                             f"checked), time constant {tau} days" for tau in DOSES["hit"]},
    **{f"{PREFIX}trial_{tau}": f"the same of the trials alone, time constant {tau} days" for tau in DOSES["trial"]},
    f"{PREFIX}tsb": "fitness less fatigue as daily rates: ff_out_45 / 45 - ff_out_10 / 10",
    f"{PREFIX}mono": "the acute load against the chronic: ff_out_10 / (ff_out_45 + 1)",
    f"{PREFIX}pred": f"the fitted response: lengths slower (above 0) than its expected time, from a ridge of earlier "
                     f"periods' residuals on the thirteen states (refitted every {REFIT_MONTHS} months; the record's "
                     f"first {BURN_DAYS} days in no fit; none before the first fit)",
}
