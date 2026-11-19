"""The race simulator refitted (the feature review of 17 Nov 2026, reports/greyhound_feature_review.md, row "sim
block": "Hand-set constants (sim.py:24-29); uncorrected inputs; adjacent-trap collisions only; median for unknown
dogs"). A research variant of the fast-loop block greyhound/blocks/sim.py (``sim_``), whose output it leaves alone.
A metrics table, not a block: the fit reads earlier days' results (who led at the bend, who won), which a block on
the cached matrix does not carry.

Each race is run ``DRAWS`` times, in lengths, as in sim.py: the run to the first bend (early pace plus noise), the
checks there, the run-in (finishing speed plus noise scaled by the dog's own spread). What changes:

- **(a) fitted constants**: the bend and run-in noise, the weight of the dog's spread, the collision reach in lengths
  and the lengths lost, the lane weight, the draw-against-line penalty, the shrinkage, the unknown dog's extra noise,
  the debut offset and the trial offset are fitted by maximum likelihood of who led at the first bend (the race's
  unique fastest first sectional; where the sectionals are incomplete, the one runner whose comment says it led
  early) plus who won, by Nelder-Mead on common random numbers, starting from sim.py's hand-set values. The fit
  window is fixed and early: from ``WARMUP_DAYS`` after the first day of ``runs`` for ``FIT_FRAC`` of the days to
  ``FIT_END`` (1 Jun 2026, the test period's start). For the 2014-on build that is about mid-2014 to early 2017.
  Rows inside the window are in-sample; every row after it is out of sample. The log likelihood of the hand-set and
  the fitted constants on the races after the window and before ``FIT_END`` is in the notes;
- **(b) shrunk inputs**: the dog's early (``esr``) and run-home (``fsr``) figures over its last 5 earlier race days
  are pulled toward the track-and-grade par (the mean over the 365 earlier days of that track and class) by ``k``
  days of prior (fitted), its spread (``gsr``) toward the field's median spread;
- **(c) the lane**: the dog's habitual running line from its earlier comments (rails 1, middle 2, wide 3, shrunk to
  the middle by ``K_PATH`` days; a debutant's trial line as one observation), as a box (``2.5 x line - 1.5``). Its
  place across the track at the bend is its trap moved ``lane`` of the way to that box; two dogs can collide with a
  weight falling linearly to 0 at ``REACH`` boxes apart (traps 1 and 2 apart, and any pair whose lines converge),
  in place of sim.py's adjacent traps only; a dog drawn away from its line loses ``mis`` lengths a box to the bend;
- **(d) priors for unknown dogs**: a dog with no race sectional takes its last trial's sectional and run-home
  figures (the trials table, ``tl_``: on the race scale) as one observation plus ``trial_off``, and the par plus
  ``debut`` lengths, with noise widened by ``unk`` and narrowing as evidence comes (1 + (unk - 1) / (1 + days)), in
  place of the race's median.

The outputs: the chance of leading at the bend, of winning, of finishing in the first two, the expected bend
position, the chance of being checked (crowding), the mean lengths lost, and the inputs as the simulator read them.
Card-safe throughout: the inputs read earlier days only; the trap, track, class and field come from the card; the
constants are fixed by the early window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core

DEPENDS = ("trials",)
PREFIX = "sm2_"
SOURCE = ("GBGB results (sources/gbgb.py): sectional, trap, track, class, position, comment, through the engine's "
          "figures; the trials table's last trial (tl_)")
DRAWS = 300                    # simulations a race in the build
FIT_DRAWS = 200                # simulations a race in the fit
FIT_END = "2026-06-01"         # the test period starts here: nothing on or after it is fitted on
WARMUP_DAYS = 180              # the first days of runs give the dogs no form: the window starts after them
FIT_FRAC = 0.2                 # the share of the days from the warm-up's end to FIT_END that is fitted on
MAX_FIT_RACES = 4000
MIN_FIT_RACES = 300
MAX_VAL_RACES = 3000
MAX_EVALS = 600
REACH = 2.0                    # boxes: two dogs this far apart across the track at the bend cannot collide
K_PATH = 2.0                   # days of prior on the running line toward the middle
K_SD = 2.0                     # days of prior on the dog's spread toward the field's median
CHUNK = 2000
KEY = ["race_id", "dog_id"]
#: sim.py's hand-set constants (and neutral values for the new terms): the fit's start and the fallback
HAND = {"early_sd": 1.0, "run_sd": 1.5, "sd_w": 0.5, "close": 1.0, "check": 2.0, "lane": 0.2, "mis": 0.0,
        "k": 2.0, "unk": 1.5, "debut": 0.0, "trial_off": 0.0}
_LOG = ("early_sd", "run_sd", "sd_w", "close", "check", "k")
COLUMNS = {
    f"{PREFIX}p_lead": "the simulated chance of leading at the first bend (fitted constants)",
    f"{PREFIX}p_win": "the simulated chance of winning",
    f"{PREFIX}p_top2": "the simulated chance of finishing in the first two",
    f"{PREFIX}exp_bend": "the expected position at the first bend (1 leads) over the simulations",
    f"{PREFIX}p_crowd": "the chance of being checked at the first bend (losing ground to a dog ahead within the "
                        "fitted reach, across the track within REACH boxes)",
    f"{PREFIX}lost": "the mean lengths lost to checks at the first bend",
    f"{PREFIX}win_logit": "the log odds of sm2_p_win (clipped at half a simulation)",
    f"{PREFIX}win_vs_best": "sm2_p_win over the field's highest",
    f"{PREFIX}early": "the early-pace input in lengths: the dog's last 5 earlier days' esr shrunk to the "
                      "track-and-grade par (a trial and debut prior for a dog with none), less the draw-against-line "
                      "penalty",
    f"{PREFIX}run": "the run-home input in lengths: fsr shrunk the same way",
    f"{PREFIX}line_box": "the box its running line wants (rails 1, middle 3.5, wide 6), shrunk to the middle",
    f"{PREFIX}evid": "the evidence behind the early input: earlier race days with a sectional, plus 1 for a trial "
                     "prior",
    f"{PREFIX}prior": "where the early input came from: 0 its own races, 1 a trial prior, 2 the debut prior alone",
}


# ----- the inputs (earlier days only) -----

def _lag(df: pd.DataFrame, v: pd.Series, n: int | None = 5) -> tuple[pd.Series, pd.Series]:
    """The dog's mean of ``v`` over its last ``n`` earlier days with a value, and the days behind it."""
    m = core.dog_lagged(df.assign(_v=v), "_v", n)
    c = core.dog_lagged(df.assign(_c=v.notna().astype(float).where(v.notna())), "_c", n, "sum").fillna(0.0)
    return m, c


def _par(df: pd.DataFrame, v: pd.Series) -> pd.Series:
    """The track-and-class mean of ``v`` over the 365 days before the row's day (NaN with none)."""
    d = df[["track", "race_class"]].assign(_s=v.fillna(0.0), _n=v.notna().astype(float))
    w = core.prior_window(d.assign(race_date=df.race_date), ["track", "race_class"], ["_s", "_n"], 365)
    return (w._s / w._n).where(w._n > 0)


def inputs(df: pd.DataFrame, tl: pd.DataFrame | None) -> pd.DataFrame:
    """Each runner's pre-race inputs (earlier days only) and the labels (the race's own result: fit only)."""
    race = df.race_id
    trial = df.is_trial.fillna(False).astype(bool) if "is_trial" in df else pd.Series(False, index=df.index)
    num = lambda c: pd.to_numeric(df[c], errors="coerce").where(~trial) if c in df else \
        pd.Series(np.nan, index=df.index)  # noqa: E731
    x = pd.DataFrame({"race": race.to_numpy()}, index=df.index)
    x["trap"] = pd.to_numeric(df.trap, errors="coerce")
    for c in ("esr", "fsr"):
        v = num(c)
        x[f"{c}_m"], x[f"{c}_n"] = _lag(df, v)
        par = _par(df, v)
        fm = x[f"{c}_m"].groupby(race).transform("mean")
        x[f"{c}_par"] = par.fillna(fm).fillna(0.0)
    g = num("gsr")
    gm, gn = _lag(df, g)
    gsd = (core.dog_lagged(df.assign(_q=g ** 2), "_q", 5) - gm ** 2).clip(lower=0) ** 0.5
    fsd = gsd.groupby(race).transform("median").fillna(2.0)
    x["sd"] = ((gsd.fillna(0.0) * gn + K_SD * fsd) / (gn + K_SD)).clip(0.5, 8.0)
    path = num("path")
    pm, pn = _lag(df, path, 10)
    x["path_m"], x["path_n"] = pm, pn
    if tl is not None and len(tl):
        t = df[KEY].merge(tl.drop_duplicates(KEY), on=KEY, how="left")
        get = lambda c: pd.Series(pd.to_numeric(t[c], errors="coerce").to_numpy() if c in t else np.nan,  # noqa: E731
                                  index=df.index)
        x["t_sec"], x["t_run"], x["t_path"] = get("tl_sec_last"), get("tl_fig_last") - get("tl_sec_last"), \
            get("tl_path_last")
    else:
        x["t_sec"] = x["t_run"] = x["t_path"] = np.nan
    # the labels: the bend leader (the unique fastest first sectional; else the one runner the comment says led)
    sec = pd.to_numeric(df.sectional, errors="coerce") if "sectional" in df else pd.Series(np.nan, index=df.index)
    full = sec.notna().groupby(race).transform("all")
    first = sec.groupby(race).transform("min")
    by_sec = ((sec == first) & full).astype(float)
    led = pd.to_numeric(df["led"], errors="coerce").fillna(0.0) if "led" in df else pd.Series(0.0, index=df.index)
    x["lead"] = np.where(full & by_sec.groupby(race).transform("sum").eq(1), by_sec,
                         np.where(~full & led.groupby(race).transform("sum").eq(1), led, np.nan))
    x["won"] = pd.to_numeric(df.won, errors="coerce") if "won" in df else np.nan
    x["trial"] = trial.to_numpy()
    return x


def features(x: pd.DataFrame, p: dict) -> dict[str, np.ndarray]:
    """The simulator's per-runner inputs under constants ``p``: early, run, sd, noise multiplier, lateral place."""
    k = p["k"]
    en, rn = x.esr_n.to_numpy(float), x.fsr_n.to_numpy(float)
    ts, tr = x.t_sec.to_numpy(float), x.t_run.to_numpy(float)
    wt = ((en == 0) & np.isfinite(ts)).astype(float)
    debut = (en == 0).astype(float)
    e_par = x.esr_par.to_numpy(float) + p["debut"] * debut
    e = (np.nan_to_num(x.esr_m.to_numpy(float)) * en + wt * np.nan_to_num(ts + p["trial_off"]) + k * e_par) / \
        (en + wt + k)
    wr = wt * np.isfinite(tr)
    r = (np.nan_to_num(x.fsr_m.to_numpy(float)) * rn + wr * np.nan_to_num(tr) + k * x.fsr_par.to_numpy(float)) / \
        (rn + wr + k)
    pn, tp = x.path_n.to_numpy(float), x.t_path.to_numpy(float)
    wp = ((pn == 0) & np.isfinite(tp)).astype(float)
    path = (np.nan_to_num(x.path_m.to_numpy(float)) * pn + wp * np.nan_to_num(tp) + 2.0 * K_PATH) / (pn + wp + K_PATH)
    box = 2.5 * path - 1.5
    trap = x.trap.to_numpy(float)
    e = e - p["mis"] * np.abs(np.nan_to_num(trap - box))
    mult = 1.0 + (p["unk"] - 1.0) / (1.0 + en + wt)
    lat = trap + p["lane"] * (box - trap)
    prior = np.where(en > 0, 0.0, np.where(wt > 0, 1.0, 2.0))
    return {"early": e, "run": r, "sd": x.sd.to_numpy(float), "mult": mult, "lat": lat, "box": box,
            "evid": en + wt, "prior": prior}


# ----- the simulation -----

def _draws(k: int, n: int, stream: int) -> tuple[np.ndarray, np.ndarray]:
    """Standard normal draws [n, k] for the bend and the run-in, the same for every race of ``k`` runners."""
    rng = np.random.default_rng([7, stream, k])
    return rng.standard_normal((n, k)), rng.standard_normal((n, k))


def simulate(early, run, sd, mult, lat, p: dict, z) -> dict[str, np.ndarray]:
    """Arrays [races, k] (one size of field); ``z`` from ``_draws``. Each output [races, k]; lead and win counts too."""
    n, k = early.shape
    D = z[0].shape[0]
    m = mult[:, None, :]
    bend = early[:, None, :] + p["early_sd"] * z[0][None] * m
    lost = np.zeros((n, D, k))
    close, check = p["close"], p["check"]
    for j in range(k):
        for i in range(k):
            if i == j:
                continue
            w = np.clip(1.0 - np.abs(lat[:, i] - lat[:, j]) / REACH, 0.0, 1.0)        # [n]
            w = np.where(np.isfinite(w), w, 0.0)
            if not (w > 0).any():
                continue
            gap = bend[:, :, i] - bend[:, :, j]                                       # > 0: i ahead of j
            hit = (gap > 0) & (gap < close)
            lost[:, :, j] += np.where(hit, check * w[:, None] * (1.0 - gap / close), 0.0)
    noise = z[1][None] * np.sqrt(p["run_sd"] ** 2 + (p["sd_w"] * sd[:, None, :]) ** 2) * m
    final = bend - lost + run[:, None, :] + noise
    eye = np.arange(k)
    order = np.argsort(-final, axis=2)
    first, second = order[:, :, 0], order[:, :, 1]
    win_c = (first[:, :, None] == eye).sum(axis=1)
    top2 = ((first[:, :, None] == eye) | (second[:, :, None] == eye)).mean(axis=1)
    lead_c = (np.argmax(bend, axis=2)[:, :, None] == eye).sum(axis=1)
    pos = np.argsort(np.argsort(-bend, axis=2), axis=2) + 1.0
    return {"win_c": win_c, "lead_c": lead_c, "p_win": win_c / D, "p_lead": lead_c / D, "p_top2": top2,
            "exp_bend": pos.mean(axis=1), "p_crowd": (lost > 0).mean(axis=1), "lost": lost.mean(axis=1)}


def _groups(x: pd.DataFrame, rows: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """The races of ``rows`` (positions into x) by field size: [(k, positions [races, k] in trap order)]."""
    sub = pd.DataFrame({"race": x.race.to_numpy()[rows], "trap": x.trap.to_numpy()[rows], "pos": rows})
    sub = sub.sort_values(["race", "trap", "pos"], kind="stable")
    size = sub.groupby("race", sort=False).race.transform("size").to_numpy()
    out = []
    for k in np.unique(size):
        if k < 2:
            continue
        part = sub[size == k]
        out.append((int(k), part.pos.to_numpy().reshape(-1, int(k))))
    return out


def _pack(p: dict) -> np.ndarray:
    v = []
    for n in HAND:
        if n in _LOG:
            v.append(np.log(p[n]))
        elif n == "lane":
            q = min(max(p[n], 1e-4), 1 - 1e-4)
            v.append(np.log(q / (1 - q)))
        elif n == "unk":
            v.append(np.log(max(p[n] - 1.0, 1e-4)))
        else:
            v.append(p[n])
    return np.array(v, float)


def _unpack(v: np.ndarray) -> dict:
    p = {}
    for n, a in zip(HAND, v):
        a = float(np.clip(a, -12, 12))
        p[n] = np.exp(a) if n in _LOG else 1 / (1 + np.exp(-a)) if n == "lane" else 1 + np.exp(a) if n == "unk" else a
    return p


def loglik(x: pd.DataFrame, groups, p: dict, draws: int = FIT_DRAWS) -> float:
    """The mean over races of log P(the bend leader leads) + log P(the winner wins), counts smoothed by half a draw."""
    f = features(x, p)
    lead, won = x.lead.to_numpy(float), x.won.to_numpy(float)
    tot, n = 0.0, 0
    for k, idx in groups:
        z = _draws(k, draws, 1)
        r = simulate(*(f[c][idx] for c in ("early", "run", "sd", "mult", "lat")), p, z)
        pl = (r["lead_c"] + 0.5) / (draws + 0.5 * k)
        pw = (r["win_c"] + 0.5) / (draws + 0.5 * k)
        tot += (np.log(pl) * lead[idx]).sum() + (np.log(pw) * won[idx]).sum()
        n += len(idx)
    return tot / max(n, 1)


def _labelled(x: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    """The rows of complete races with one bend leader, one winner and 2 to 8 runners (no trial)."""
    s = x.iloc[rows]
    g = s.groupby("race")
    ok = (g.lead.transform("sum").eq(1) & g.lead.transform("count").eq(g.race.transform("size"))
          & g.won.transform("sum").eq(1) & g.won.transform("count").eq(g.race.transform("size"))
          & g.race.transform("size").between(2, 8) & ~s.trial & s.trap.notna()
          & g.trap.transform("nunique").eq(g.race.transform("size")))
    return rows[ok.to_numpy()]


def _sample(x: pd.DataFrame, rows: np.ndarray, cap: int, seed: int) -> np.ndarray:
    races = pd.unique(x.race.to_numpy()[rows])
    if len(races) > cap:
        races = np.random.default_rng(seed).choice(races, cap, replace=False)
    return rows[np.isin(x.race.to_numpy()[rows], races)]


def fit(x: pd.DataFrame, when: pd.Series) -> tuple[dict, dict]:
    """The constants fitted on the early window (``fit`` in the module docstring); the hand-set ones when it has too
    few races. Notes: the window, the races, the log likelihoods (fit and the later pre-test validation)."""
    from scipy.optimize import minimize
    end = pd.Timestamp(FIT_END)
    lo = when.min() + pd.Timedelta(days=WARMUP_DAYS)
    hi = lo + (end - lo) * FIT_FRAC if pd.notna(lo) and lo < end else lo
    notes = {"window": [str(lo.date()) if pd.notna(lo) else None, str(hi.date()) if pd.notna(hi) else None],
             "fit_end": FIT_END, "hand": dict(HAND)}
    w = when.to_numpy()
    rows_fit = _labelled(x, np.flatnonzero((w >= np.datetime64(lo)) & (w < np.datetime64(hi)))) if pd.notna(lo) \
        else np.array([], int)
    rows_fit = _sample(x, rows_fit, MAX_FIT_RACES, 0)
    n_fit = len(pd.unique(x.race.to_numpy()[rows_fit]))
    notes["fit_races"] = int(n_fit)
    if n_fit < MIN_FIT_RACES:
        notes["fitted"] = False
        return dict(HAND), notes
    xf = x.iloc[rows_fit].reset_index(drop=True)            # the fit reads its own rows only
    gf = _groups(xf, np.arange(len(xf)))
    f0 = loglik(xf, gf, HAND)
    res = minimize(lambda v: -loglik(xf, gf, _unpack(v)), _pack(HAND), method="Nelder-Mead",
                   options={"maxfev": MAX_EVALS, "xatol": 1e-3, "fatol": 1e-6,
                            "initial_simplex": _pack(HAND) + np.vstack([np.zeros(len(HAND)), 0.3 * np.eye(len(HAND))])})
    p = _unpack(res.x)
    notes.update({"fitted": True, "evals": int(res.nfev), "ll_fit_hand": round(f0, 5), "ll_fit": round(-res.fun, 5),
                  "constants": {n: round(float(v), 4) for n, v in p.items()}})
    rows_val = _labelled(x, np.flatnonzero((w >= np.datetime64(hi)) & (w < np.datetime64(end))))
    rows_val = _sample(x, rows_val, MAX_VAL_RACES, 1)
    if len(rows_val):
        xv = x.iloc[rows_val].reset_index(drop=True)
        gv = _groups(xv, np.arange(len(xv)))
        notes["val_races"] = int(len(pd.unique(xv.race)))
        notes["ll_val_hand"] = round(loglik(xv, gv, HAND), 5)
        notes["ll_val"] = round(loglik(xv, gv, p), 5)
    return p, notes


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    ctx = ctx or {}
    df0 = figs if figs is not None else core.per_run_figures(runs)
    tl = ctx.get("trials")
    if tl is None and (ctx.get("trial_runs") is not None or
                       ("is_trial" in df0 and df0.is_trial.fillna(False).astype(bool).any())):
        from greyhound.pit import trials
        tl = trials.build(runs, df0, ctx)[0]
    df = df0.sort_values(["t", "race_id", "trap"], kind="stable")
    x = inputs(df, tl)
    when = core.day(df)
    p, notes = fit(x, when)
    f = features(x, p)
    out = pd.DataFrame({"race_id": df.race_id.to_numpy(), "dog_id": df.dog_id.to_numpy()}, index=df.index)
    res = {c: np.full(len(df), np.nan) for c in ("p_lead", "p_win", "p_top2", "exp_bend", "p_crowd", "lost")}
    ok = np.flatnonzero(x.trap.notna().to_numpy())
    for k, idx in _groups(x, ok):
        z = _draws(k, DRAWS, 0)
        for s in range(0, len(idx), CHUNK):
            sl = idx[s:s + CHUNK]
            r = simulate(*(f[c][sl] for c in ("early", "run", "sd", "mult", "lat")), p, z)
            for c in res:
                res[c][sl.ravel()] = r[c].ravel()
    for c, v in res.items():
        out[f"{PREFIX}{c}"] = v
    win = pd.Series(res["p_win"], index=df.index)
    best = win.groupby(df.race_id).transform("max")
    out[f"{PREFIX}win_vs_best"] = (win / best.replace(0, np.nan)).to_numpy()
    q = win.clip(0.5 / DRAWS, 1 - 0.5 / DRAWS)
    out[f"{PREFIX}win_logit"] = np.log(q / (1 - q)).to_numpy()
    out[f"{PREFIX}early"], out[f"{PREFIX}run"] = f["early"], f["run"]
    out[f"{PREFIX}line_box"], out[f"{PREFIX}evid"], out[f"{PREFIX}prior"] = f["box"], f["evid"], f["prior"]
    notes["card_safe"] = list(COLUMNS)
    notes["prior_share"] = {str(int(k)): round(float(v), 4) for k, v in
                            pd.Series(f["prior"][~x.trial.to_numpy()]).value_counts(normalize=True).items()}
    cols = list(COLUMNS)
    out[cols] = out[cols].astype(np.float32)
    return out.sort_index().reset_index(drop=True)[["race_id", "dog_id"] + cols], notes
