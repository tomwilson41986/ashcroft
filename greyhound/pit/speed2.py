"""Speed figures rebuilt on the review's lines (the feature review of 17 Nov 2026, reports/greyhound_feature_review.md,
rows "speed sf_": "One winners-only going; sectional not linked; fixed blend 0.5; half-life in runs not days", and
"Engine gsr / esr, fsr": "Percent of a rolling, linked standard ... esr from secfig's linked, grade-aware sectional;
fsr with its own run-home going"). A research variant of greyhound/pit/speed.py (``sf_``), whose output it leaves
alone.

The speed table's steps 1, 2 and 5 (capped times, the rolling standard in percent, the field anchoring) are kept;
what changes:

a. **The going**: the going table's method (greyhound/pit/going.py) in place of the winners-only variant. Each clean
   runner (no interference in its comment, finished, no handicap, not a card row) is read against its own level
   before the day (the mean of its last three speed-table figures, earlier days); the night's mean residual, then each
   trap third's (1-2, 3-4, 5-6) and each half of the meeting's deviation from it, each shrunk by n / (n + k)
   (speed.eb_variant, k from the early meetings). A run's allowance is the three summed.
b. **Linked after the new going**: the link offsets (speed.link_offsets) are fitted again on the figure less the new
   allowance, so the offsets and the going agree (the going table kept the speed table's offsets).
c. **The blend fitted**: ``G = F - (1 - w)(field mean of F - field strength)`` (speed.py step 5, ``w`` 0.5 there),
   ``w`` chosen on a grid by the winners' log likelihood of a one-feature conditional logit on the mean of the last
   three ``G`` (earlier days), on the races of the early slice only (before speed.K_BEFORE, else the first fifth of
   the days). The mean of three is linear in ``G``, so the grid costs one fit a point, no rebuild.
d. **Decay by days**: each recency-weighted level is a weighted mean of the dog's earlier days' figures, weight
   2^(-days ago / h), the half-life ``h`` (days) chosen on ``HALF_GRID`` the same way, one for each figure family.
   ``sv_fresh`` is the summed weight (how much recent evidence the level stands on).
e. **Engine stand-ins** (gsr, esr, fsr on linked percent figures, aggregated as the engine's consumers read them:
   last, mean of 3, max of 6, the decayed level): ``gsr`` the figure ``F`` (rolling standard, the new going, linked,
   before the field anchoring); ``esr`` secfig's first sectional (grade-aware sectional going, linked:
   greyhound/pit/secfig.py ``figures``); ``fsr`` the run home (run time less sectional) figured the same way against
   the track and trip's par, with its own grade-aware run-home going, linked (``secfig.figures(values=...,
   by_trap=False)``). Percent, higher faster, not the engine's lengths.

Every column is read from the dog's EARLIER days, or from today's runners' earlier-day columns (vs, rank): card-safe.
The fitted constants (w, the half-lives, the shrinkage k's) come from the early slice only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core, going, secfig, speed
from greyhound.pit.speed import _num, _vs_field, earlier

DEPENDS = ("speed", "going", "secfig")
PREFIX = "sv_"
SOURCE = ("GBGB results (calculated time, beaten distance, position, sectional, run time, comment, trap, off time, "
          "class, track, trip, meeting: sources/gbgb.py), with the speed, going and secfig tables' methods")
CLIP = going.CLIP             # percent: a clean runner's residual against its level is read no further
HALF_GRID = (7.0, 14.0, 30.0, 60.0, 90.0, 180.0, 365.0)     # days
BLEND_GRID = tuple(round(x, 2) for x in np.arange(0.0, 1.0001, 0.05))
MAX_FIT_RACES = 200_000       # early-slice races in a grid point's logit (a seeded sample beyond it)
MIN_FIT_RACES = 50            # fewer: the speed table's constants (blend 0.5, half-life 30 days) are kept
DEFAULT_HALF = 30.0

_FAM = {"gsr": "the linked percent time figure (rolling standard, clean-runner going by side and half, linked; no field anchoring)",
        "esr": "the linked first-sectional figure (secfig: track, trip and trap par, grade-aware sectional going, linked)",
        "fsr": "the linked run-home figure (run time less sectional against the track and trip par, its own grade-aware going, linked)"}
COLUMNS = {
    "sv_last": "the dog's last speed figure (percent, higher faster): capped time against the rolling standard, less the clean-runner going by side and half, linked, anchored on the field with the fitted blend",
    "sv_mean3": "the mean of its last three such figures (earlier days)",
    "sv_best3": "the best of its last three such figures",
    "sv_dk": "such figures decayed by days (weight 2^(-days ago / h), h fitted on the early slice)",
    "sv_n": "the number of earlier days with such a figure",
    "sv_fresh": "the summed decay weight behind sv_dk (about 1 a figure in the last half-life)",
    "sv_vs": "sv_dk less the mean of the other runners' in today's race",
    "sv_rank": "the rank of sv_dk in today's race (1 the best)",
    **{f"sv_{f}_{a}": f"{d}: {w}" for f, d in _FAM.items()
       for a, w in (("last", "the last"), ("mean3", "the mean of the last three"), ("max6", "the best of the last six"),
                    ("dk", "decayed by days, half-life fitted"))},
    **{f"sv_{f}_vs": f"sv_{f}_dk less the mean of the other runners' in today's race" for f in _FAM},
    "sv_var_last": "the night's clean-runner allowance at its last figured run (above 0: a fast night)",
    "sv_side_last": "its trap third's deviation from the night's allowance at its last figured run",
    "sv_half_last": "its half of the meeting's deviation from the night's allowance at its last figured run",
}
COLUMNS = {k: COLUMNS[k] for k in (
    "sv_last", "sv_mean3", "sv_best3", "sv_dk", "sv_n", "sv_fresh", "sv_vs", "sv_rank",
    *(f"sv_{f}_{a}" for f in _FAM for a in ("last", "mean3", "max6", "dk", "vs")),
    "sv_var_last", "sv_side_last", "sv_half_last")}


# ---- the pieces ----------------------------------------------------------------------------------------------------

def decayed(dog, dn, val, half: float) -> tuple[np.ndarray, np.ndarray]:
    """Per row: the weighted mean of the dog's EARLIER days' values (a day's values averaged), weight 2^(-days ago /
    ``half``), and the summed weight. The level is the same read on any later day (the weights' common factor
    cancels), so it is kept at each day and read back by merge_asof; the summed weight decays to the row's day."""
    d = pd.DataFrame({"dog": dog, "dn": dn, "v": val})
    per = d[np.isfinite(d.v)].groupby(["dog", "dn"], as_index=False).v.mean().sort_values(["dog", "dn"])
    g = per.dog.to_numpy()
    x = (per.dn - per.groupby("dog").dn.transform("min")).to_numpy(float) / half
    e = np.exp2(np.minimum(x, 1000.0))
    cs = pd.Series(e * per.v.to_numpy()).groupby(g).cumsum().to_numpy()
    cw = pd.Series(e).groupby(g).cumsum().to_numpy()
    per = per.assign(lvl=cs / cw, rel=cw / e, dl=per.dn.to_numpy(float)).sort_values("dn")
    q = pd.DataFrame({"dog": dog, "dn": dn, "i": np.arange(len(d))}).sort_values("dn")
    m = pd.merge_asof(q, per[["dog", "dn", "lvl", "rel", "dl"]], on="dn", by="dog", allow_exact_matches=False,
                      direction="backward")
    lvl, fresh = np.full(len(d), np.nan), np.zeros(len(d))
    i = m.i.to_numpy()
    lvl[i] = m.lvl.to_numpy()
    fresh[i] = np.nan_to_num((m.rel * np.exp2(-(m.dn - m.dl) / half)).to_numpy())
    return lvl, fresh


def allowance(meet, trapn, tt, res, clean, early) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """The going table's allowance (going.py build): the night's clean residual, then the trap third's and the
    meeting half's deviations, each shrunk (speed.eb_variant). Returns the three a row and their k's."""
    meet = np.asarray(meet).astype(str)
    v_night, k_night = speed.eb_variant(meet, res, clean, early)
    side = np.where(trapn <= 2, "i", np.where(trapn <= 4, "m", "o"))
    v_side, k_side = going._shrunk_dev(np.char.add(np.char.add(meet, "|"), side), res - v_night, clean, early)
    rank = pd.Series(np.asarray(tt)).groupby(meet).rank(method="dense").to_numpy()
    nraces = pd.Series(rank).groupby(meet).transform("max").to_numpy()
    half = np.where(rank <= nraces / 2.0, "e", "l")
    v_half, k_half = going._shrunk_dev(np.char.add(np.char.add(meet, "|"), half), res - v_night - v_side, clean, early)
    return v_night, v_side, v_half, {"k_night": round(k_night, 3), "k_side": round(k_side, 3),
                                     "k_half": round(k_half, 3)}


def fit_races(race, won, use, seed: int = 0) -> np.ndarray:
    """The rows of the races a constant is fitted on: every runner in ``use``, one winner, three runners or more;
    a seeded sample of ``MAX_FIT_RACES`` races beyond that."""
    d = pd.DataFrame({"r": race, "u": use, "w": won})
    g = d.groupby("r")
    ok = g.u.transform("all") & g.w.transform("sum").eq(1) & g.u.transform("size").ge(3)
    races = pd.unique(d.r[ok])
    if len(races) > MAX_FIT_RACES:
        races = np.random.default_rng(seed).choice(races, MAX_FIT_RACES, replace=False)
    return (ok & d.r.isin(races)).to_numpy()


def race_loglik(x, race, won, rows) -> tuple[float, int]:
    """The mean log chance of the winners under a one-feature conditional logit (core.clogit_fit) of ``x`` (scaled),
    on the races in ``rows`` whose runners all have ``x``. Returns it and the races used."""
    d = pd.DataFrame({"r": race, "x": x, "w": won})[rows]
    if len(d):
        g = d.groupby("r").x
        d = d[g.transform("count").to_numpy() == g.transform("size").to_numpy()]
    if d.empty or not np.isfinite(d.x.std()):
        return float("-inf"), 0
    gi = pd.factorize(d.r)[0]
    z = (d.x - d.x.mean()) / (d.x.std() or 1.0)
    X = z.to_numpy(float)[:, None]
    y = d.w.to_numpy(float)
    b = core.clogit_fit(X, gi, y, lam=1.0)
    p = core.group_softmax(X @ b, gi)
    ng = int(gi.max()) + 1
    return float(np.log(np.clip(p[y == 1], 1e-12, None)).sum() / ng), ng


def fit_blend(A, B, race, won, rows, grid=BLEND_GRID) -> tuple[float, dict]:
    """w on ``grid`` maximising the winners' log likelihood of A - (1 - w) B (the mean of three of G, written
    linearly: A the mean of three of F, B of the field term)."""
    sc = {w: race_loglik(A - (1.0 - w) * B, race, won, rows) for w in grid}
    if max(n for _, n in sc.values()) < MIN_FIT_RACES:
        return speed.BLEND, {"races": max(n for _, n in sc.values()), "kept": "speed.BLEND"}
    best = max(sc, key=lambda w: sc[w][0])
    return float(best), {"races": sc[best][1], "loglik": {str(w): round(v, 5) for w, (v, _) in sc.items()}}


def fit_half(dog, dn, val, race, won, rows, grid=HALF_GRID) -> tuple[float, dict]:
    """The half-life (days) on ``grid`` maximising the winners' log likelihood of the decayed level."""
    sc = {h: race_loglik(decayed(dog, dn, val, h)[0], race, won, rows) for h in grid}
    if max(n for _, n in sc.values()) < MIN_FIT_RACES:
        return DEFAULT_HALF, {"races": max(n for _, n in sc.values()), "kept": DEFAULT_HALF}
    best = max(sc, key=lambda h: sc[h][0])
    return float(best), {"races": sc[best][1], "loglik": {str(h): round(v, 5) for h, (v, _) in sc.items()}}


# ---- the table -----------------------------------------------------------------------------------------------------

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
    trapn = pd.to_numeric(r.trap, errors="coerce").to_numpy(float)
    dog = pd.to_numeric(r.dog_id, errors="coerce").to_numpy(float)
    race = r.race_id.to_numpy()
    pos = _num(r.position)
    ok = ~card & ~hcap & ~dnf & np.isfinite(pos)
    early = day < pd.Timestamp(speed.K_BEFORE)
    if not early.any():
        early = dn <= np.nanquantile(dn, 0.2)
    early = np.asarray(early, bool)

    # speed.py steps 1-4: the time, the rolling standard (percent), its winners-only variant and offsets (only to
    # have each dog's level before the day for the clean-runner going, as going.py does)
    t = speed.capped_times(r, ok)
    lt = np.log(t)
    win = ok & (pos == 1) & np.isfinite(lt)
    base = pd.DataFrame({"race_date": r.race_date.to_numpy(), "_key": key, "_grade": grade,
                         "_lw": np.where(win, lt, 0.0), "_w": win.astype(float)})
    w365 = core.prior_window(base, ["_key"], ["_lw", "_w"], speed.STD_DAYS)
    wall = core.prior_window(base, ["_key"], ["_lw", "_w"], None)
    std = np.where(w365._w >= speed.MIN_STD, w365._lw / w365._w.replace(0, np.nan),
                   np.where(wall._w >= speed.MIN_STD, wall._lw / wall._w.replace(0, np.nan), np.nan))
    P = 100.0 * (std - lt)
    base["_pw"] = np.where(win & np.isfinite(P), P, 0.0)
    base["_gw"] = (win & np.isfinite(P)).astype(float)
    gp = core.prior_window(base, ["_key", "_grade"], ["_pw", "_gw"], speed.STD_DAYS)
    par = np.where(gp._gw >= speed.MIN_GRADE, gp._pw / gp._gw.replace(0, np.nan), 0.0)
    v_old, _ = speed.eb_variant(meet, P - par, win & np.isfinite(P), early)
    months = np.unique(month[np.isfinite(month)])
    offs0 = speed.link_offsets(dog, dn, key, P - v_old, month, ok, months)
    off0 = np.array([offs0.get(m, {}).get(k, 0.0) for m, k in zip(month, key)])

    # a. the clean-runner going by side and half (going.py)
    E = earlier(dog, dn, P - v_old + off0, "mean")
    if "comment" in r:
        cm = r.comment.fillna("").astype(str)
        ci, uniq = pd.factorize(cm)
        from greyhound.pit.trouble import codes
        troubled = np.array([bool(codes(c)) for c in uniq], bool)[ci] if len(uniq) else np.zeros(n, bool)
    else:
        troubled = np.zeros(n, bool)
    res = np.clip(P + off0 - E, -CLIP, CLIP)
    clean = ok & ~troubled & np.isfinite(res)
    tt = r.t if "t" in r else pd.to_datetime(r.race_date.astype(str) + " " + r.race_time.astype(str), errors="coerce")
    v_night, v_side, v_half, k_go = allowance(meet, trapn, tt.to_numpy(), res, clean, early)
    v = v_night + v_side + v_half

    # b. linked again on the figure less the new going
    offs = speed.link_offsets(dog, dn, key, P - v, month, ok, months)
    off = np.array([offs.get(m, {}).get(k, 0.0) for m, k in zip(month, key)])
    F = P - v + off

    # c. the field anchoring, its blend fitted on the early slice
    won = ok & (pos == 1)
    rows = fit_races(race, won, ok & early)
    X = earlier(dog, dn, F, "mean")
    S = pd.Series(X).groupby(race).transform("mean").to_numpy()
    fm = pd.Series(F).groupby(race).transform("mean").to_numpy()
    D = np.where(np.isfinite(F), np.where(np.isfinite(S), fm - S, 0.0), np.nan)
    sub = dn <= (np.nanmax(dn[early]) if early.any() else -np.inf)      # the early days: all a fit reads
    w, w_notes = fit_blend(earlier(dog[sub], dn[sub], F[sub], "mean"), earlier(dog[sub], dn[sub], D[sub], "mean"),
                           race[sub], won[sub], rows[sub])
    G = F - (1.0 - w) * D

    # e. the sectional and the run home, linked (secfig)
    fe = secfig.figures(r)
    sec = _num(r.sectional)
    runin = np.where(np.isfinite(sec) & (sec > 1) & (sec < 15), _num(r.run_time) - sec, np.nan)
    ff = secfig.figures(r, values=runin, lo=1.0, hi=np.inf, by_trap=False)
    fam = {"gsr": F, "esr": fe["L"], "fsr": ff["L"]}

    # d. the half-lives, by days, fitted on the early slice
    halves, h_notes = {}, {}
    for name, val in (("sv", G), *fam.items()):
        halves[name], h_notes[name] = fit_half(dog[sub], dn[sub], val[sub], race[sub], won[sub], rows[sub])

    out = {"sv_last": earlier(dog, dn, G, "last"), "sv_mean3": earlier(dog, dn, G, "mean"),
           "sv_best3": earlier(dog, dn, G, "max"), "sv_n": earlier(dog, dn, G, "count")}
    out["sv_dk"], out["sv_fresh"] = decayed(dog, dn, G, halves["sv"])
    out["sv_vs"] = _vs_field(race, out["sv_dk"])
    out["sv_rank"] = pd.Series(out["sv_dk"]).groupby(race).rank(ascending=False, method="average").to_numpy()
    for name, val in fam.items():
        out[f"sv_{name}_last"] = earlier(dog, dn, val, "last")
        out[f"sv_{name}_mean3"] = earlier(dog, dn, val, "mean")
        out[f"sv_{name}_max6"] = earlier(dog, dn, val, "max", n=6)
        out[f"sv_{name}_dk"] = decayed(dog, dn, val, halves[name])[0]
        out[f"sv_{name}_vs"] = _vs_field(race, out[f"sv_{name}_dk"])
    fig_ok = np.isfinite(F)
    out["sv_var_last"] = earlier(dog, dn, np.where(fig_ok, v_night, np.nan), "last")
    out["sv_side_last"] = earlier(dog, dn, np.where(fig_ok, v_side, np.nan), "last")
    out["sv_half_last"] = earlier(dog, dn, np.where(fig_ok, v_half, np.nan), "last")
    tab = pd.DataFrame({"race_id": r.race_id.to_numpy(), "dog_id": r.dog_id.to_numpy(),
                        **{c: np.asarray(out[c], float).astype(np.float32) for c in COLUMNS}}, index=r.index)
    both = sorted(set(offs0.get(months[-1], {})) & set(offs.get(months[-1], {}))) if len(months) else []
    sd = lambda x: round(float(np.nanstd(x[fig_ok])), 4) if fig_ok.any() else None  # noqa: E731
    notes = {"blend": w, "blend_fit": w_notes, "half_life_days": halves, "half_life_fit": h_notes, **k_go,
             "clean_runners": int(clean.sum()), "troubled_left_out": int((ok & troubled).sum()),
             "sd_night": sd(v_night), "sd_side": sd(v_side), "sd_half": sd(v_half), "sd_old_variant": sd(v_old),
             "offset_shift_sd_last_month": round(float(np.std([offs[months[-1]][k] - offs0[months[-1]][k]
                                                               for k in both])), 4) if both else None,
             "runs_figured": int(np.isfinite(G).sum()), "esr_figured": int(np.isfinite(fe["L"]).sum()),
             "fsr_figured": int(np.isfinite(ff["L"]).sum()), "esr_fit": fe["notes"], "fsr_fit": ff["notes"],
             "fit_races": int(pd.unique(race[rows]).size),
             "card_safe": "every column: earlier days only, vs and rank from today's runners' earlier-day columns"}
    return tab, notes
