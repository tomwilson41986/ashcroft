"""The dog's Irish form on the GB scale (the feature review of 17 Nov 2026, reports/greyhound_feature_review.md,
"irish ie_": "Irish figures not linked to the GB scale. Link offsets on dogs that move, plus a walk-forward transfer
of the last Irish figure to the first GB figure").

greyhound/pit/irish.py links each GB dog to its Irish runs (GreyhoundAPI, by name and by the API's GB rows) and
makes each Irish run's time and first-sectional figures: 100 x (the Irish track and trip's standard - ln time), the
standard the mean ln winning time there over the year before the day. A GB run's figure is made the same way here
(``gb_figures``: irish.run_figures on the GBGB runs, trials and cards left out), so the two differ by the Irish
track and trip's standard against the GB tracks': the link offset.

**The movers.** A dog with Irish runs whose first GB race day F comes within ``MOVE_DAYS`` of its last Irish day.
Its last Irish day's figures (read before F) against its figures on day F give one residual for the Irish track and
trip it last ran at (the key), for the time and for the first sectional.

**The walk-forward refits.** On 1 January of each year R (from the store's second year), from the movers with F
before R (``MIN_MOVERS`` of them at least, else that year's columns are empty):
- the link offsets: a key's offset is the movers' mean residual g plus its own mean's departure shrunk by
  n / (n + K), K = s2 / tau2 (s2 the residuals' variance within keys, tau2 the keys' variance between them, by
  moments, K in [``K_MIN``, ``K_MAX``]); the offset's variance s2 / (n + K) + s2 / N; a key with no mover g, with
  variance tau2 + s2 / N. The same for the sectional;
- the transfer: a ridge regression (``RIDGE``, standardised inputs) of the first GB figure on the dog's Irish form
  on the GB scale (level, last, best, sectional level, log runs, shrunk win rate, days since its last Irish run,
  class, a flag for each missing input).
A race on day D reads the refit of the last 1 January on or before D, and the dog's Irish days before D.

**The columns** (NaN for a dog with no Irish figure before the day, or before the first refit):
``iq_level`` the mean over its last three Irish days of (figure + its key's offset); ``iq_level_sd`` the square root
of the mean offset variance plus s2 / k (k the days read); ``iq_sec_level`` the same for the first sectional;
``iq_pred_gb`` the transfer's predicted first GB figure; ``iq_pred_vs_field`` and ``iq_level_vs_field`` those less
the mean GB level of the other runners in the race that have one (each runner's mean GB figure over its last three
GB days before the day).

Card-safe: every column (a card row reads Irish days and GB days before its day and the refit of a 1 January on or
before it, as its run would), given the same store history: the refits read every mover before R, so a card build
on a store cut later than the nightly build's would fit other offsets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core, irish

DEPENDS = ("irish",)
PREFIX = "iq_"
SOURCE = ("GreyhoundAPI Irish results linked to GBGB dogs as the irish table (greyhound/pit/irish.py), put on the "
          "GB scale by the dogs that move (their last Irish figures against their first GBGB figures), refitted "
          "each 1 January on the earlier movers")
MOVE_DAYS = 365
MIN_MOVERS = 30
K_MIN, K_MAX = 1.0, 500.0
RIDGE = 1.0
N_LAST = 3
COLUMNS = {
    "iq_level": "mean of its last three Irish time figures, each plus its Irish track and trip's link offset to the GB scale",
    "iq_level_sd": "the uncertainty of iq_level: sqrt(mean offset variance + the movers' residual variance / days read)",
    "iq_sec_level": "mean of its last three Irish first-sectional figures on the GB scale (the sectional's link offsets)",
    "iq_pred_gb": "the walk-forward transfer's predicted first GB time figure from its Irish form (ridge, refitted each 1 Jan)",
    "iq_pred_vs_field": "iq_pred_gb less the mean GB level (last three GB figures before the day) of the other runners that have one",
    "iq_level_vs_field": "iq_level less the mean GB level of the other runners that have one",
}
FEATS = ("level", "last", "best", "sec", "log_n", "win", "days", "cls", "sec_na", "cls_na")


def gb_figures(runs: pd.DataFrame) -> pd.DataFrame:
    """Each GBGB race run (no trial, no card) with its time and first-sectional figures, made as irish.run_figures
    makes the Irish ones; the index the runs' index."""
    keep = np.ones(len(runs), bool)
    for c in ("is_trial", "is_card"):
        if c in runs:
            keep &= ~runs[c].fillna(False).to_numpy(bool)
    r = runs[keep]
    api = pd.DataFrame({
        "dog_id": r.dog_id.to_numpy(), "track": r.track.astype(str).to_numpy(),
        "start_local": r.race_date.astype(str).str[:10].to_numpy(),
        "position": pd.to_numeric(r.position, errors="coerce").to_numpy(),
        "run_time_s": pd.to_numeric(r.run_time, errors="coerce").to_numpy() if "run_time" in r else np.nan,
        "adjusted_time_s": pd.to_numeric(r.adjusted_time, errors="coerce").to_numpy() if "adjusted_time" in r else np.nan,
        "sectional_s": pd.to_numeric(r.sectional, errors="coerce").to_numpy() if "sectional" in r else np.nan,
        "distance_m": pd.to_numeric(r.distance_m, errors="coerce").to_numpy(), "class_number": np.nan}, index=r.index)
    return irish.run_figures(api)


def _key(api: pd.DataFrame) -> pd.Series:
    """The Irish track and trip, as irish.run_figures keys its standards."""
    return api.track.astype(str) + "|" + pd.to_numeric(api.distance_m, errors="coerce").round().astype("Int64").astype(str)


def irish_days(irr: pd.DataFrame, kcode: np.ndarray) -> pd.DataFrame:
    """Per GB dog and Irish day: the running counts, the class, and the figures and keys of its last three days
    (``fig0`` the day itself)."""
    d = irr.assign(kc=kcode)
    per = d.groupby(["gb_dog", "race_date"], as_index=False).agg(
        n=("won", "size"), won=("won", "sum"), fig=("fig", "mean"), sec=("sec", "mean"), cls=("cls", "last"),
        kc=("kc", "last")).sort_values(["gb_dog", "race_date"])
    g = per.groupby("gb_dog")
    out = pd.DataFrame({"gb_dog": per.gb_dog.to_numpy(), "day": pd.to_datetime(per.race_date).to_numpy()})
    out["n"] = g.n.cumsum().to_numpy()
    out["won"] = g.won.cumsum().to_numpy()
    out["cls"] = per.cls.to_numpy(float)
    for j in range(N_LAST):
        out[f"fig{j}"] = g.fig.shift(j).to_numpy(float)
        out[f"sec{j}"] = g.sec.shift(j).to_numpy(float)
        out[f"kc{j}"] = g.kc.shift(j).fillna(-1).to_numpy(np.int64)
    out["last_day"] = out.day
    return out


def _asof(q: pd.DataFrame, h: pd.DataFrame, by: str) -> pd.DataFrame:
    """``h``'s last row of the same ``by`` on a day strictly before each ``q`` row's day; aligned to ``q``."""
    q = q.assign(_i=np.arange(len(q))).sort_values("day")
    m = pd.merge_asof(q, h.sort_values("day"), on="day", by=by, allow_exact_matches=False, direction="backward")
    return m.sort_values("_i").reset_index(drop=True)


def fit_offsets(res: np.ndarray, kc: np.ndarray, nk: int) -> tuple[np.ndarray, np.ndarray, float]:
    """The shrunk link offset of each key (0..nk-1) and its variance; the residual variance s2 (the docstring)."""
    ok = np.isfinite(res) & (kc >= 0)
    res, kc = res[ok], kc[ok]
    N = len(res)
    if N < 2:
        return np.full(nk, np.nan), np.full(nk, np.nan), np.nan
    g = float(res.mean())
    n = np.bincount(kc, minlength=nk).astype(float)
    mk = np.where(n > 0, np.bincount(kc, res, nk) / np.maximum(n, 1), g)
    s2 = float(((res - mk[kc]) ** 2).sum() / max(N - (n > 0).sum(), 1))
    s2 = s2 if s2 > 0 else float(res.var()) or 1e-6
    seen = n > 0
    tau2 = float(np.average((mk[seen] - g) ** 2, weights=n[seen]) - s2 * seen.sum() / N) if seen.any() else 0.0
    K = float(np.clip(s2 / tau2, K_MIN, K_MAX)) if tau2 > 0 else K_MAX
    off = g + n * (mk - g) / (n + K)
    var = np.where(seen, s2 / (n + K), max(tau2, s2 / K)) + s2 / N
    return off, var, s2


def _on_scale(m: pd.DataFrame, e: np.ndarray, OFF, VAR, OFS, VRS, S2, win_k: float) -> pd.DataFrame:
    """The Irish form of each row on the GB scale at refit ``e`` (a row's epoch; -1 none)."""
    n = len(m)
    E = len(OFF)
    ok_e = (e >= 0) & (e < E)
    ee = np.where(ok_e, e, 0)
    lv, lvar, sv, cnt, scnt = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    best = np.full(n, -np.inf)
    last = np.full(n, np.nan)
    for j in range(N_LAST):
        kc = m[f"kc{j}"].fillna(-1).to_numpy(np.int64)
        f = m[f"fig{j}"].to_numpy(float)
        s = m[f"sec{j}"].to_numpy(float)
        kk = np.maximum(kc, 0)
        o, v = OFF[ee, kk], VAR[ee, kk]
        os_, vs = OFS[ee, kk], VRS[ee, kk]
        x = f + o
        good = ok_e & (kc >= 0) & np.isfinite(x)
        lv += np.where(good, x, 0.0)
        lvar += np.where(good, v, 0.0)
        cnt += good
        best = np.where(good, np.maximum(best, x), best)
        if j == 0:
            last = np.where(good, x, np.nan)
        xs = s + os_
        gs = ok_e & (kc >= 0) & np.isfinite(xs)
        sv += np.where(gs, xs, 0.0)
        scnt += gs
    with np.errstate(invalid="ignore", divide="ignore"):
        level = np.where(cnt > 0, lv / cnt, np.nan)
        sd = np.where(cnt > 0, np.sqrt(lvar / cnt + S2[ee] / cnt), np.nan)
        sec = np.where(scnt > 0, sv / scnt, np.nan)
    nn = m.n.fillna(0).to_numpy(float)
    days = m.days.to_numpy(float) if "days" in m else np.full(n, np.nan)
    cls = m.cls.to_numpy(float)
    return pd.DataFrame({"level": level, "sd": sd, "last": last, "best": np.where(cnt > 0, best, np.nan), "sec": sec,
                         "log_n": np.log1p(nn), "win": (m.won.fillna(0).to_numpy(float) + win_k / 6.0) / (nn + win_k),
                         "days": np.clip(days, 0, MOVE_DAYS) / 100.0, "cls": cls,
                         "sec_na": (~np.isfinite(sec)).astype(float), "cls_na": (~np.isfinite(cls)).astype(float)})


def _design(F: pd.DataFrame, mu=None, sd=None):
    X = F[list(FEATS)].to_numpy(float)
    if mu is None:
        mu = np.nanmean(X, 0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
    X = np.where(np.isfinite(X), X, mu)
    if sd is None:
        sd = X.std(0)
        sd = np.where(sd > 1e-9, sd, 1.0)
    return (X - mu) / sd, mu, sd


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None):
    ctx = ctx or {}
    r = runs
    n = len(r)
    tab = pd.DataFrame({"race_id": r.race_id.to_numpy(), "dog_id": r.dog_id.to_numpy()}, index=r.index)
    for c in COLUMNS:
        tab[c] = np.float32(np.nan)
    ie, gb = ctx.get("api_ie"), ctx.get("api_gb")
    if ie is None or not len(ie):
        return tab, {"note": "no data found: no GreyhoundAPI Irish results in the store"}
    ie = ie.reset_index(drop=True)
    mp, mnotes = irish.dog_map(gb, r)
    nmp, first, nnotes = irish.name_map(ie, r)
    mnotes.update(nnotes)
    irr = irish.irish_runs(ie, {**nmp, **mp}, first)
    if not len(irr):
        return tab, {**mnotes, "note": "no Irish run of a mapped dog"}
    kcode, keys = pd.factorize(_key(ie).loc[irr.index])
    NK = max(len(keys), 1)
    h = irish_days(irr, np.asarray(kcode, np.int64))

    # the GB figures: per dog and day; each dog's GB level (last three GB days) and its first GB day
    gf = gb_figures(r)
    gf = gf.assign(day=pd.to_datetime(gf.race_date))
    gday = gf.groupby(["dog_id", "day"], as_index=False).agg(fig=("fig", "mean"), sec=("sec", "mean"))
    gday["dog_id"] = pd.to_numeric(gday.dog_id).astype("int64")
    gv = gday[gday.fig.notna()].sort_values(["dog_id", "day"]).copy()
    gv["lvl3"] = gv.groupby("dog_id").fig.rolling(N_LAST, min_periods=1).mean().reset_index(level=0, drop=True)
    firstgb = gday.sort_values("day").drop_duplicates("dog_id")            # the first race day (with or without a figure)

    # the movers: Irish form before F against the figures of day F
    mv = firstgb.rename(columns={"dog_id": "gb_dog"})[["gb_dog", "day", "fig", "sec"]]
    mv = mv.rename(columns={"fig": "y", "sec": "ys"})
    mm = _asof(mv, h, "gb_dog")
    mm["days"] = (mm.day - mm.last_day) / np.timedelta64(1, "D")
    mm = mm[mm.n.notna() & (mm.days <= MOVE_DAYS) & mm.fig0.notna() & (mm.kc0 >= 0)].reset_index(drop=True)
    mm["kc0"] = mm.kc0.astype(np.int64)

    # the refits: 1 January of each year after the store's first
    day = core.day(r)
    y0, y1 = int(day.dt.year.min()), int(day.dt.year.max())
    R = np.array([np.datetime64(f"{y}-01-01") for y in range(y0 + 1, y1 + 1)], dtype="datetime64[ns]")
    E = len(R)
    OFF, VAR = np.full((max(E, 1), NK), np.nan), np.full((max(E, 1), NK), np.nan)
    OFS, VRS = OFF.copy(), VAR.copy()
    S2 = np.full(max(E, 1), np.nan)
    coefs, fits = {}, []
    md = mm.day.to_numpy("datetime64[ns]")
    for k, Rk in enumerate(R):
        tr = mm[md < Rk]
        if len(tr) < MIN_MOVERS:
            fits.append({"refit": str(Rk)[:10], "movers": int(len(tr)), "fitted": False})
            continue
        kc = tr.kc0.to_numpy(np.int64)
        OFF[k], VAR[k], S2[k] = fit_offsets(tr.y.to_numpy(float) - tr.fig0.to_numpy(float), kc, NK)
        OFS[k], VRS[k], _ = fit_offsets(tr.ys.to_numpy(float) - tr.sec0.to_numpy(float), kc, NK)
        F = _on_scale(tr, np.full(len(tr), k), OFF, VAR, OFS, VRS, S2, irish.K)
        fy = np.isfinite(tr.y.to_numpy(float))
        F = F[fy]
        X, mu, sd = _design(F)
        yy = tr.y.to_numpy(float)[fy]
        b0 = float(yy.mean())
        beta = np.linalg.solve(X.T @ X + RIDGE * np.eye(X.shape[1]), X.T @ (yy - b0))
        resid = yy - b0 - X @ beta
        coefs[k] = (mu, sd, b0, beta)
        fits.append({"refit": str(Rk)[:10], "movers": int(len(tr)), "fitted": True,
                     "offset_mean": round(float(np.nanmean(tr.y - tr.fig0)), 3), "resid_var": round(float(S2[k]), 3),
                     "ridge_rmse_in_fit": round(float(np.sqrt((resid ** 2).mean())), 3),
                     "level_rmse_in_fit": round(float(np.sqrt(np.nanmean((yy - F.level.to_numpy()) ** 2))), 3)})

    # the rows: the dog's Irish days before the day, at the refit of the last 1 January on or before it
    q = pd.DataFrame({"gb_dog": pd.to_numeric(r.dog_id, errors="coerce").astype("int64").to_numpy(),
                      "day": day.to_numpy()})
    m = _asof(q, h, "gb_dog")
    m["days"] = (m.day - m.last_day) / np.timedelta64(1, "D")
    e = np.searchsorted(R, day.to_numpy("datetime64[ns]"), "right") - 1 if E else np.full(n, -1)
    e = np.where(np.isin(e, list(coefs)), e, -1)
    F = _on_scale(m, e, OFF, VAR, OFS, VRS, S2, irish.K)
    pred = np.full(n, np.nan)
    for k, (mu, sd, b0, beta) in coefs.items():
        sel = (e == k) & np.isfinite(F.level.to_numpy())
        if sel.any():
            X, _, _ = _design(F[sel], mu, sd)
            pred[sel] = b0 + X @ beta
    has = np.isfinite(F.level.to_numpy())
    tab["iq_level"] = F.level.to_numpy()
    tab["iq_level_sd"] = F.sd.to_numpy()
    tab["iq_sec_level"] = np.where(has, F.sec.to_numpy(), np.nan)
    tab["iq_pred_gb"] = pred

    # the field: the other runners' GB levels before the day
    g3 = _asof(pd.DataFrame({"dog_id": q.gb_dog.to_numpy(), "day": day.to_numpy()}), gv[["dog_id", "day", "lvl3"]],
               "dog_id").lvl3.to_numpy(float)
    race = r.race_id.to_numpy()
    s = pd.Series(g3)
    tot = s.groupby(race).transform("sum").to_numpy()
    cnt = s.notna().groupby(race).transform("sum").to_numpy()
    own = np.isfinite(g3)
    with np.errstate(invalid="ignore", divide="ignore"):
        oth_n = cnt - own
        oth = np.where(oth_n > 0, (tot - np.nan_to_num(g3)) / np.maximum(oth_n, 1), np.nan)
    tab["iq_pred_vs_field"] = pred - oth
    tab["iq_level_vs_field"] = tab.iq_level.to_numpy() - oth
    for c in COLUMNS:
        tab[c] = tab[c].astype(np.float32)
    notes = {**mnotes, "irish_runs_mapped": int(len(irr)), "irish_keys": int(len(keys)), "movers": int(len(mm)),
             "refits": fits, "rows_with_level": int(has.sum()), "rows_with_pred": int(np.isfinite(pred).sum()),
             "card_safe": list(COLUMNS)}
    return tab, notes
