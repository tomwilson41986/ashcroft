"""Speed figures with a Timeform-style going allowance (the owner, 16 Nov 2026: "Rebuild the going allowance the
Timeform way, with inside/outside and early/late corrections"; ledger greyhound-going-batch43-1116).

Timeform's method (https://www.timeform.com/GREYHOUND-RACING/beginners-guide/betting/ratings-and-sectionals-explained-1382020):
each runner's time against what was expected of it, trouble priced first, the night's allowance from the whole field
in proportion to the trip, then corrected for inside/outside bias and for early or late in the meeting. Our speed
table (greyhound/pit/speed.py, ``sf_``) takes the variant from the WINNERS against their grade's par only. This table
keeps the speed table's steps 1, 2, 4 and 5 (capped times, the standard, linked standards, field anchoring) and
replaces step 3:

- **Expected**: each runner's linked figure before the variant against its own level before the day (the mean of
  its last three speed-table figures, earlier days only).
- **Clean runners only**: a run whose comment shows interference (core.INTERFERENCE), a non-finisher, a handicap, a
  card row are left out of the allowance (trouble first: their time says nothing about the going).
- **The night**: the clean residuals' mean over the meeting, shrunk by n / (n + k) (k the within- over the
  between-meeting variance on the meetings before ``K_BEFORE``: speed.eb_variant).
- **Inside or outside**: each third of the traps (1-2, 3-4, 5-6) its own deviation from the night's, shrunk the same
  way: a night when the rail or the outside was faster.
- **Early or late**: the first and the second half of the meeting's races (by off time) each its deviation, shrunk.

A run's allowance is the three summed; its figure is the linked figure less that. The columns read only earlier days.

Columns: ``gw_last``, ``gw_mean3``, ``gw_best3``, ``gw_ewm``, ``gw_n``, ``gw_vs``, ``gw_rank`` (as the speed table's,
on the new figure), ``gw_var_last`` (the night's allowance at its last figured run), ``gw_side_last`` (its trap
third's deviation that night), ``gw_half_last`` (its half of the meeting's deviation), ``gw_diff_mean3`` (the new
figure less the speed table's, mean of the last three: what the allowance changed for this dog).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core, speed
from greyhound.pit.speed import _num, earlier, _vs_field

PREFIX = "gw_"
SOURCE = ("GBGB results (calculated time, beaten distance, position, comment, trap, off time, class, track, trip, "
          "meeting: sources/gbgb.py), the speed table's figures (greyhound/pit/speed.py)")
CLIP = 6.0                    # percent: a clean runner's residual is read no further
COLUMNS = {
    "gw_last": "the dog's last speed figure with the Timeform-style going (clean field, trap third and meeting half corrected)",
    "gw_mean3": "the mean of its last three such figures (earlier days)",
    "gw_best3": "the best of its last three such figures",
    "gw_ewm": "such figures weighted by recency, half-life two runs",
    "gw_n": "the number of earlier days with such a figure",
    "gw_vs": "gw_ewm less the mean of the other runners' gw_ewm in today's race",
    "gw_rank": "the rank of gw_ewm in today's race (1 the best)",
    "gw_var_last": "the night's allowance (clean field against expectation) at its last figured run",
    "gw_side_last": "its trap third's deviation from the night's allowance at its last figured run",
    "gw_half_last": "its half of the meeting's deviation from the night's allowance at its last figured run",
    "gw_diff_mean3": "the new figure less the speed table's, mean of the last three (earlier days)",
}


def _shrunk_dev(group: np.ndarray, res: np.ndarray, use: np.ndarray, early: np.ndarray) -> tuple[np.ndarray, float]:
    return speed.eb_variant(group, res, use, early)


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
    ok = ~card & ~hcap & ~dnf & np.isfinite(_num(r.position))
    early = day < pd.Timestamp(speed.K_BEFORE)
    if not early.any():
        early = dn <= np.nanquantile(dn, 0.2)
    early = np.asarray(early, bool)

    # the speed table's steps 1-4 (as speed.build), to have its figure F and the linked, un-varied P + offset
    t = speed.capped_times(r, ok)
    lt = np.log(t)
    win = ok & (_num(r.position) == 1) & np.isfinite(lt)
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
    PV = P - v_old
    months = np.unique(month[np.isfinite(month)])
    offs = speed.link_offsets(dog, dn, key, PV, month, ok, months)
    off = np.array([offs.get(m, {}).get(k, 0.0) for m, k in zip(month, key)])
    L = P + off                                   # linked, before any variant
    F_old = PV + off                              # the speed table's figure before the field anchoring

    # the Timeform-style allowance: clean runners against their own level before the day
    E = earlier(dog, dn, F_old, "mean")
    from greyhound.pit.trouble import codes
    troubled = (np.array([bool(codes(c)) for c in r.comment.tolist()], bool) if "comment" in r
                else np.zeros(n, bool))           # the comment's interference codes, read as the trouble table reads them
    res = np.clip(L - E, -CLIP, CLIP)
    clean = ok & ~troubled & np.isfinite(res)
    v_night, k_night = speed.eb_variant(meet, res, clean, early)
    side = np.where(trapn <= 2, "i", np.where(trapn <= 4, "m", "o"))
    gside = np.char.add(np.char.add(meet.astype(str), "|"), side.astype(str))
    v_side, k_side = _shrunk_dev(gside, res - v_night, clean, early)
    tt = r.t if "t" in r else pd.to_datetime(r.race_date.astype(str) + " " + r.race_time.astype(str), errors="coerce")
    rank = pd.Series(tt.to_numpy()).groupby(meet).rank(method="dense").to_numpy()
    nraces = pd.Series(rank).groupby(meet).transform("max").to_numpy()
    half = np.where(rank <= nraces / 2.0, "e", "l")
    ghalf = np.char.add(np.char.add(meet.astype(str), "|"), half.astype(str))
    v_half, k_half = _shrunk_dev(ghalf, res - v_night - v_side, clean, early)
    v = v_night + v_side + v_half
    F = L - v

    # anchored on the field (speed.py step 5)
    X = earlier(dog, dn, F, "mean")
    S = pd.Series(X).groupby(race).transform("mean").to_numpy()
    fm = pd.Series(F).groupby(race).transform("mean").to_numpy()
    G = np.where(np.isfinite(S), speed.BLEND * F + (1 - speed.BLEND) * (S + F - fm), F)
    S_old = pd.Series(earlier(dog, dn, F_old, "mean")).groupby(race).transform("mean").to_numpy()
    fm_old = pd.Series(F_old).groupby(race).transform("mean").to_numpy()
    G_old = np.where(np.isfinite(S_old), speed.BLEND * F_old + (1 - speed.BLEND) * (S_old + F_old - fm_old), F_old)

    fig_ok = np.isfinite(F)
    out = {"gw_last": earlier(dog, dn, G, "last"), "gw_mean3": earlier(dog, dn, G, "mean"),
           "gw_best3": earlier(dog, dn, G, "max"), "gw_ewm": earlier(dog, dn, G, "ewm"),
           "gw_n": earlier(dog, dn, G, "count"),
           "gw_var_last": earlier(dog, dn, np.where(fig_ok, v_night, np.nan), "last"),
           "gw_side_last": earlier(dog, dn, np.where(fig_ok, v_side, np.nan), "last"),
           "gw_half_last": earlier(dog, dn, np.where(fig_ok, v_half, np.nan), "last"),
           "gw_diff_mean3": earlier(dog, dn, G - G_old, "mean")}
    out["gw_vs"] = _vs_field(race, out["gw_ewm"])
    out["gw_rank"] = pd.Series(out["gw_ewm"]).groupby(race).rank(ascending=False, method="average").to_numpy()
    tab = pd.DataFrame({"race_id": r.race_id.to_numpy(), "dog_id": r.dog_id.to_numpy(),
                        **{c: out[c].astype(np.float32) for c in COLUMNS}}, index=r.index)
    sd = lambda x: round(float(np.nanstd(x[fig_ok])), 4) if fig_ok.any() else None  # noqa: E731
    notes = {"k_night": round(k_night, 3), "k_side": round(k_side, 3), "k_half": round(k_half, 3),
             "clean_runners": int(clean.sum()), "troubled_left_out": int((ok & troubled).sum()),
             "sd_night": sd(v_night), "sd_side": sd(v_side), "sd_half": sd(v_half), "sd_old_variant": sd(v_old),
             "runs_figured": int(np.isfinite(G).sum())}
    return tab, notes
