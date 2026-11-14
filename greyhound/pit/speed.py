"""Speed figures for each run, and each dog's figures before its race (the owner, 14 Nov 2026: "Create speed figures
for each run", then "Continue with expected times and speed work. Implement"; the audit is reports/greyhound_model.md
section 60, research/greyhound/times_audit/).

**A run's figure** (percent of the time, higher is faster), built from what is known once its meeting is over:

1. **Its time.** GBGB times only the winner; a loser's time is the winner's calculated time (GBGB's going allowance
   applied) plus 0.08 s a length of its cumulative beaten distance, as GBGB rule 139 makes it, but with the distance
   capped at ``CAP`` lengths: a dog beaten a long way has eased, and its margin says little more (the audit: capped at
   6 lengths, the figure's prediction of the next winner improved by 0.006 in log-loss, z -9.1). Handicaps (a start
   given), hurdles' own standards, falls and void runs are not figured.
2. **The standard**: the mean log calculated time of the winners at the track, trip and hurdles over the ``STD_DAYS``
   before the day (at least ``MIN_STD`` of them; else every earlier winner there). The figure is
   ``100 x (standard - ln time)``, in percent, so it carries across trips (the audit: percent against lengths, -0.006).
3. **The meeting variant**: what GBGB's going allowance leaves (0.67 lengths a night in the audit). Each winner's
   figure less its grade's par at the track and trip (the mean figure of that grade's winners there over
   ``STD_DAYS``), averaged over the meeting and shrunk by ``n / (n + k)``, ``k`` the within- over the between-meeting
   variance measured on the meetings before ``K_BEFORE`` (else the first fifth of the days): a fixed constant, never
   fitted on the rows it prices.
4. **Linked standards.** Standards made of each track and trip's own winners are not comparable (the audit: half the
   track pairs off by more than 2 standard errors, 0.92 lengths on the median). Each month an offset a track and trip
   is fitted by ridge regression (``LAM``) on the dogs that ran at two of them within ``MOVE_DAYS``, over the
   ``LINK_MONTHS`` before the month (the mean change of a dog at the same track and trip taken out), so a dog's figure
   at one means the same at another. A run takes its month's offsets: fitted on runs before the month began.
5. **Anchored on the field**: half the time figure, half the field's strength before the race (the mean of its
   runners' ``sf_mean3``, earlier days) plus the dog's figure against the race's mean figure (the audit's largest gain:
   -0.021 in log-loss with the cap, z -13.0 against the live convention).

The first sectional and the run home (run time less sectional) get their own figures: against the mean log time of
every runner at the track, trip and trap (sectional) or track and trip (run home) over ``STD_DAYS``, with their own
meeting variant (GBGB's going is not applied to a sectional, though it correlates -0.24 with the night's sectionals).

**The columns** (a dog's EARLIER days only, a day's runs averaged; the race-relative ones from today's runners'
pre-race columns): ``sf_last``, ``sf_mean3``, ``sf_best3``, ``sf_ewm`` (half-life ``HALF`` runs), ``sf_n``, ``sf_vs``
(``sf_ewm`` less the mean of the other runners'), ``sf_rank`` (1 the race's best ``sf_ewm``), ``sf_time_mean3`` and
``sf_time_vs`` (the time figure alone), ``sf_sec_ewm``, ``sf_sec_vs``, ``sf_run_ewm``, ``sf_run_vs``, ``sf_var_last``
(the variant of the dog's last figured meeting: a run on a fast night).

Every column is read from earlier days (tests/test_greyhound_speed.py: the last day's results rewritten move nothing
on that day or before, a card row gets what its run had, a long beaten distance is capped).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core

PREFIX = "sf_"
SOURCE = ("GBGB results (calculated time with the going allowance, beaten distance, position, sectional, run time, "
          "class, track, trip, trap, meeting: sources/gbgb.py)")
CAP = 6.0                     # lengths: a loser's beaten distance is read no further (the audit, section 60)
STD_DAYS = 365
MIN_STD = 20                  # winners for a standard; 30 runners for a sectional or run-home par
MIN_PAR = 30
MIN_GRADE = 10                # winners for a grade's par
K_BEFORE = "2018-01-01"
LAM = 5.0
MOVE_DAYS = 28
LINK_MONTHS = 24
HALF = 2.0
BLEND = 0.5
COLUMNS = {
    "sf_last": "the dog's last speed figure (percent, higher faster): capped calculated time against the linked standard, less the meeting variant, half anchored on the field",
    "sf_mean3": "the mean of its last three speed figures (earlier days)",
    "sf_best3": "the best of its last three speed figures",
    "sf_ewm": "its speed figures weighted by recency, half-life two runs",
    "sf_n": "the number of earlier days with a speed figure",
    "sf_vs": "sf_ewm less the mean of the other runners' sf_ewm in today's race",
    "sf_rank": "the rank of sf_ewm in today's race (1 the best)",
    "sf_time_mean3": "the mean of its last three time figures (the speed figure before the field anchoring)",
    "sf_time_vs": "sf_time_mean3 less the mean of the other runners'",
    "sf_sec_ewm": "its first-sectional figures weighted by recency (percent against the track, trip and trap par, less the night's sectional variant)",
    "sf_sec_vs": "sf_sec_ewm less the mean of the other runners'",
    "sf_run_ewm": "its run-home figures (run time less sectional, against the track and trip par, less the night's variant) weighted by recency",
    "sf_run_vs": "sf_run_ewm less the mean of the other runners'",
    "sf_var_last": "the meeting variant of its last figured run (above 0: a fast night beyond GBGB's going)",
}


# ---- the figure of a run --------------------------------------------------------------------------------------------

def _num(s) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(float)


def capped_times(r: pd.DataFrame, ok: np.ndarray, cap: float = CAP) -> np.ndarray:
    """Each run's time: the race winner's calculated time plus 0.08 s a length of its cumulative beaten distance,
    capped at ``cap`` lengths; NaN where the winner's time or any distance in front of the dog is missing."""
    d = pd.DataFrame({"race": r.race_id.to_numpy(), "pos": _num(r.position), "bl": _num(r.beaten_lengths),
                      "calc": _num(r.adjusted_time), "ok": ok}, index=np.arange(len(r)))
    d = d[d.ok & np.isfinite(d.pos)].sort_values(["race", "pos"])
    first = d.groupby("race").pos.transform("min").eq(d.pos)
    gap = d.bl.where(~first, 0.0)
    miss = gap.isna().astype(int).groupby(d.race).cummax().astype(bool)
    cum = gap.fillna(0.0).groupby(d.race).cumsum()
    win = d.calc.where(first & d.pos.eq(1)).groupby(d.race).transform("min")
    t = (win + core.LENGTH_S * np.minimum(cum, cap)).where(~miss)
    out = np.full(len(r), np.nan)
    out[t.index.to_numpy()] = t.to_numpy()
    return out


def eb_variant(meet: np.ndarray, res: np.ndarray, use: np.ndarray, early: np.ndarray) -> tuple[np.ndarray, float]:
    """The meeting's mean residual over ``use`` rows, shrunk by n / (n + k); ``k`` = within over between variance on
    the ``early`` meetings. Returned a row (0 where the meeting has none) and k."""
    d = pd.DataFrame({"m": meet, "r": res, "e": early})[use & np.isfinite(res)]
    g = d.groupby("m")
    mean, n, var = g.r.mean(), g.r.size(), g.r.var()
    e = g.e.first().astype(bool)
    within = var[(n > 1) & e].mean()
    sel = (n >= 3) & e
    between = mean[sel].var() - within * (1.0 / n[sel]).mean()
    k = float(within / max(between, 1e-6)) if np.isfinite(within) and np.isfinite(between) else 10.0
    v = (mean * n / (n + k)).reindex(meet).to_numpy()
    return np.nan_to_num(v), k


def link_offsets(dog, dn, key, fig, month, ok, months, lam: float = LAM) -> dict:
    """Per month: an offset a track and trip (``key``) fitted on the moves of dogs between two keys within
    ``MOVE_DAYS`` whose later run fell in the ``LINK_MONTHS`` months before it. Returns {month: {key: offset}}."""
    d = pd.DataFrame({"dog": dog, "dn": dn, "key": key, "f": fig, "mo": month})[ok & np.isfinite(fig)]
    d = d.groupby(["dog", "dn", "key"], as_index=False).agg(f=("f", "mean"), mo=("mo", "first")).sort_values(["dog", "dn"])
    prev = d.groupby("dog")[["dn", "key", "f"]].shift(1)
    near = (d.dn - prev.dn <= MOVE_DAYS) & prev.f.notna()
    move, same = d[near & (d.key != prev.key)], d[near & (d.key == prev.key)]
    pk, pf = prev.key[move.index], prev.f[move.index]
    keys = sorted(set(d.key))
    ix = {k: i for i, k in enumerate(keys)}
    K = len(keys)
    a, b = np.array([ix[k] for k in pk], int), np.array([ix[k] for k in move.key], int)
    y = (move.f - pf).to_numpy()
    mo = move.mo.to_numpy()
    drift = (same.f - prev.f[same.index]).groupby(same.mo).agg(["sum", "size"])
    out = {}
    for m in months:
        w = (mo < m) & (mo >= m - LINK_MONTHS)
        if not w.any():
            out[m] = {}
            continue
        dr = drift[(drift.index < m) & (drift.index >= m - LINK_MONTHS)]
        mu = float(dr["sum"].sum() / dr["size"].sum()) if dr["size"].sum() > 0 else 0.0
        aa, bb, yy = a[w], b[w], y[w] - mu
        ata = np.zeros((K, K))
        np.add.at(ata, (aa, aa), 1.0)
        np.add.at(ata, (bb, bb), 1.0)
        np.add.at(ata, (aa, bb), -1.0)
        np.add.at(ata, (bb, aa), -1.0)
        aty = np.zeros(K)
        np.add.at(aty, bb, -yy)                       # (f_b + o_b) - (f_a + o_a) ~ 0  =>  o_b - o_a ~ -(f_b - f_a)
        np.add.at(aty, aa, yy)
        o = np.linalg.solve(ata + lam * np.eye(K), aty)
        seen = np.zeros(K, bool)
        seen[aa] = seen[bb] = True
        o = o - o[seen].mean()
        out[m] = {keys[i]: float(o[i]) for i in np.flatnonzero(seen)}
    return out


# ---- a dog's figures before its race -------------------------------------------------------------------------------

def earlier(dog, dn, val, how: str, n: int = 3, half: float = HALF) -> np.ndarray:
    """Per row: ``how`` over the dog's last ``n`` EARLIER days that have a value (a day's values averaged): mean,
    max, last, ewm (half-life ``half`` days with a value), count (every earlier day with a value)."""
    d = pd.DataFrame({"dog": dog, "dn": dn, "v": val})
    per = d[np.isfinite(d.v)].groupby(["dog", "dn"], as_index=False).v.mean().sort_values(["dog", "dn"])
    g = per.groupby("dog").v
    if how == "mean":
        s = g.rolling(n, min_periods=1).mean().reset_index(level=0, drop=True)
    elif how == "max":
        s = g.rolling(n, min_periods=1).max().reset_index(level=0, drop=True)
    elif how == "last":
        s = per.v
    elif how == "count":
        s = g.cumcount() + 1.0
    else:
        s = g.transform(lambda x: x.ewm(halflife=half).mean())
    per["s"] = s.reindex(per.index).to_numpy()
    q = pd.DataFrame({"dog": dog, "dn": dn, "i": np.arange(len(d))}).sort_values("dn")
    per = per.sort_values("dn")
    m = pd.merge_asof(q, per[["dog", "dn", "s"]], on="dn", by="dog", allow_exact_matches=False, direction="backward")
    out = np.full(len(d), np.nan)
    out[m.i.to_numpy()] = m.s.to_numpy()
    if how == "count":
        out = np.nan_to_num(out)
    return out


def _vs_field(race, x) -> np.ndarray:
    """``x`` less the mean of the other runners' (those with a value) in the race."""
    s = pd.Series(x)
    tot = s.groupby(race).transform("sum").to_numpy()
    cnt = s.notna().groupby(race).transform("sum").to_numpy()
    others = cnt - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(x) & (others > 0), x - (tot - x) / np.maximum(others, 1), np.nan)


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None):
    r = runs
    n = len(r)
    day = core.day(r)
    dn = ((day - pd.Timestamp("2000-01-01")).dt.days).to_numpy(float)
    month = (day.dt.year * 12 + day.dt.month).to_numpy(float)
    card = r.is_card.to_numpy(bool) if "is_card" in r else np.zeros(n, bool)
    hurdle = r.hurdles.fillna(False).astype(bool).to_numpy() if "hurdles" in r else np.zeros(n, bool)
    hcap = (r.handicap.fillna(False).astype(bool).to_numpy() if "handicap" in r else np.zeros(n, bool)) | \
        (_num(r.hcp_start) > 0 if "hcp_start" in r else np.zeros(n, bool))
    dnf = r.dnf.fillna(False).astype(bool).to_numpy() if "dnf" in r else np.zeros(n, bool)
    key = (r.track.astype(str) + "|" + pd.to_numeric(r.distance_m, errors="coerce").round().astype("Int64").astype(str)
           + "|" + hurdle.astype(int).astype(str)).to_numpy()
    grade = r.race_class.astype(str).to_numpy()
    meet = (r.meeting_id.astype(str) if "meeting_id" in r else r.track.astype(str) + day.astype(str)).to_numpy()
    trap = pd.to_numeric(r.trap, errors="coerce").fillna(0).astype(int).astype(str).to_numpy()
    dog = pd.to_numeric(r.dog_id, errors="coerce").to_numpy(float)
    race = r.race_id.to_numpy()
    ok = ~card & ~hcap & ~dnf & np.isfinite(_num(r.position))
    early = day < pd.Timestamp(K_BEFORE)
    if not early.any():
        early = dn <= np.nanquantile(dn, 0.2)
    early = np.asarray(early, bool)

    # 1-2. the time and the standard
    t = capped_times(r, ok)
    lt = np.log(t)
    win = ok & (_num(r.position) == 1) & np.isfinite(lt)
    base = pd.DataFrame({"race_date": r.race_date.to_numpy(), "_key": key, "_grade": grade,
                         "_lw": np.where(win, lt, 0.0), "_w": win.astype(float)})
    w365 = core.prior_window(base, ["_key"], ["_lw", "_w"], STD_DAYS)
    wall = core.prior_window(base, ["_key"], ["_lw", "_w"], None)
    std = np.where(w365._w >= MIN_STD, w365._lw / w365._w.replace(0, np.nan),
                   np.where(wall._w >= MIN_STD, wall._lw / wall._w.replace(0, np.nan), np.nan))
    P = 100.0 * (std - lt)

    # 3. the meeting variant: winners against their grade's par (earlier days)
    base["_pw"] = np.where(win & np.isfinite(P), P, 0.0)
    base["_gw"] = (win & np.isfinite(P)).astype(float)
    gp = core.prior_window(base, ["_key", "_grade"], ["_pw", "_gw"], STD_DAYS)
    par = np.where(gp._gw >= MIN_GRADE, gp._pw / gp._gw.replace(0, np.nan), 0.0)
    v, k_tot = eb_variant(meet, P - par, win & np.isfinite(P), early)
    PV = P - v

    # 4. linked standards: offsets a month, fitted before it
    months = np.unique(month[np.isfinite(month)])
    offs = link_offsets(dog, dn, key, PV, month, ok, months)
    off = np.array([offs.get(m, {}).get(k, 0.0) for m, k in zip(month, key)])
    F = PV + off

    # 5. anchored on the field
    X = earlier(dog, dn, F, "mean")
    xs = pd.Series(X)
    S = xs.groupby(race).transform("mean").to_numpy()
    fm = pd.Series(F).groupby(race).transform("mean").to_numpy()
    G = np.where(np.isfinite(S), BLEND * F + (1 - BLEND) * (S + F - fm), F)

    # the sectional and the run home
    sec = _num(r.sectional)
    sec_ok = ok & np.isfinite(sec) & (sec > 1) & (sec < 15)
    ls = np.where(sec_ok, np.log(np.where(sec_ok, sec, 1.0)), np.nan)
    runin = _num(r.run_time) - sec
    run_ok = sec_ok & np.isfinite(runin) & (runin > 1)
    lr = np.where(run_ok, np.log(np.where(run_ok, runin, 1.0)), np.nan)
    base2 = pd.DataFrame({"race_date": r.race_date.to_numpy(), "_key": key, "_trap": trap,
                          "_ls": np.nan_to_num(ls), "_sn": sec_ok.astype(float),
                          "_lr": np.nan_to_num(lr), "_rn": run_ok.astype(float)})
    pt = core.prior_window(base2, ["_key", "_trap"], ["_ls", "_sn"], STD_DAYS)
    pk = core.prior_window(base2, ["_key"], ["_ls", "_sn", "_lr", "_rn"], STD_DAYS)
    spar = np.where(pt._sn >= MIN_PAR, pt._ls / pt._sn.replace(0, np.nan),
                    np.where(pk._sn >= MIN_PAR, pk._ls / pk._sn.replace(0, np.nan), np.nan))
    rpar = np.where(pk._rn >= MIN_PAR, pk._lr / pk._rn.replace(0, np.nan), np.nan)
    SX = 100.0 * (spar - ls)
    RX = 100.0 * (rpar - lr)
    vs_, k_sec = eb_variant(meet, SX, sec_ok & np.isfinite(SX), early)
    vr_, k_run = eb_variant(meet, RX, run_ok & np.isfinite(RX), early)
    SX, RX = SX - vs_, RX - vr_

    # the columns: earlier days only
    out = {"sf_last": earlier(dog, dn, G, "last"), "sf_mean3": earlier(dog, dn, G, "mean"),
           "sf_best3": earlier(dog, dn, G, "max"), "sf_ewm": earlier(dog, dn, G, "ewm"),
           "sf_n": earlier(dog, dn, G, "count"), "sf_time_mean3": X,
           "sf_sec_ewm": earlier(dog, dn, SX, "ewm"), "sf_run_ewm": earlier(dog, dn, RX, "ewm"),
           "sf_var_last": earlier(dog, dn, np.where(np.isfinite(F), v, np.nan), "last")}
    out["sf_vs"] = _vs_field(race, out["sf_ewm"])
    out["sf_rank"] = pd.Series(out["sf_ewm"]).groupby(race).rank(ascending=False, method="average").to_numpy()
    out["sf_time_vs"] = _vs_field(race, X)
    out["sf_sec_vs"] = _vs_field(race, out["sf_sec_ewm"])
    out["sf_run_vs"] = _vs_field(race, out["sf_run_ewm"])
    tab = pd.DataFrame({"race_id": r.race_id.to_numpy(), "dog_id": r.dog_id.to_numpy(),
                        **{c: out[c].astype(np.float32) for c in COLUMNS}}, index=r.index)
    notes = {"cap_lengths": CAP, "k_total": round(k_tot, 3), "k_sectional": round(k_sec, 3), "k_run_home": round(k_run, 3),
             "runs_figured": int(np.isfinite(G).sum()), "offset_months": len(offs),
             "keys_linked_last_month": len(offs[months[-1]]) if len(months) else 0,
             "sectionals_figured": int(np.isfinite(SX).sum())}
    return tab, notes
