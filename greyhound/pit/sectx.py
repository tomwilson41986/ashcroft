"""Sectionals and times taken apart (the owner, 17 Nov 2026: "can we derive a second sectional? And a Finishing Speed
%. pars per track for early / late sectionals / FSP, metrics vs. race pars for that race. go very deep on every
possible feature we can engineer around sectionals and times"; ledger greyhound-sectx-batch50-1117).

GBGB gives each dog its first sectional (the break to the line the first time) and times the winner; a loser's time
is the winner's run time plus 0.08 s a length beaten (GBGB rule 139, capped at ``speed.CAP`` lengths as the speed
table caps it). From those, each run's:

- **S1** the first sectional, **S2** the second (the run home: the dog's time less S1), **T** the whole time.
- **Pars** (earlier days only, the ``speed.STD_DAYS`` before the day, every finisher that ran clear of a fall or a
  hand-slip): the mean log S1 at the track, trip and trap (else track and trip), the mean log S2 and log T at the track
  and trip, the mean share of the time spent in each section; and **race pars**, the same by the race's grade.
- **Figures** in percent of the time (higher faster, so they carry across trips): ``100 x (par - ln time)`` for S1, S2
  and T.
- **Finishing Speed % (FSP)**: the run home's average speed over the whole run's average speed, against the track and
  trip's usual ratio: ``100 x (S2/T at par) / (S2/T)``. 100 is the track's usual finish; above 100 the dog finished
  stronger than it began, relative to a typical runner there. **Early Speed % (ESP)** the same for S1.
- **Going** a night and a section: each runner's figure less its race par, averaged over the meeting and shrunk
  (``speed.eb_variant``), taken off each figure (S1, S2, T and FSP each their own night).
- **Linked across tracks** by the dogs that move (``speed.link_offsets``): S1, S2 and T.
- **Against the race par** (the race's grade at the track and trip): S1, S2, T, FSP.
- **Against the field it met**: S1, S2, T, FSP less the mean of the race's other runners; the race's early pace (its
  runners' S1 against par); its rank at the first sectional and in the run home, the places it gained, the gap between
  its early and late ranks (a closer's sign).

**The columns** (a dog's EARLIER days only; ``sc_``): each run series' last, mean of 3, recency-weighted mean, and
for the main figures the best of 3, the spread over 5 and the trend (mean of 3 less mean of 10); the dog's S1 and FSP
at today's track and trip; and from today's runners' pre-race columns: the race's projected early pace and the dog
against it, its rank and gap to the fastest early, the dog's FSP and S2 against the field, and the closer's
interaction (its FSP against the field x the race's projected early pace against par), and the trip change x FSP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core, speed

PREFIX = "sc_"
SOURCE = "GBGB results (sources/gbgb.py): sectional, run time, beaten distance, position, trap, grade, track, trip, meeting"
LENGTH_S = 0.08

#: the per-run series and what the dog's columns read of each
SERIES = {
    "s1": "first-sectional figure (linked, less the night's S1 going)",
    "s2": "second-section (run home) figure (linked, less the night's S2 going)",
    "tt": "whole-time figure (linked, less the night's going)",
    "fsp": "Finishing Speed % (100 the track's usual finish, less the night's FSP going)",
    "esp": "Early Speed % (100 the track's usual early share)",
    "s1_rp": "S1 against its race par (the race's grade at the track and trip)",
    "s2_rp": "S2 against its race par",
    "tt_rp": "whole time against its race par",
    "fsp_rp": "FSP against its race par",
    "s1_vf": "S1 against the field it met",
    "s2_vf": "S2 against the field it met",
    "tt_vf": "whole time against the field it met",
    "fsp_vf": "FSP against the field it met",
    "k1": "its rank at the first sectional (1 fastest)",
    "k2": "its rank in the run home (1 fastest)",
    "gain": "places gained from the first sectional to the finish",
    "close": "first-sectional rank less run-home rank (above 0: faster home than away)",
    "pace": "the early pace of the race it ran in (its runners' mean S1 against their race par)",
}
MAIN = ("s1", "s2", "tt", "fsp")


def _cols() -> dict:
    c = {}
    for s, what in SERIES.items():
        c[f"sc_{s}_last"] = f"last run: {what}"
        c[f"sc_{s}_mean3"] = f"mean of its last three: {what}"
        c[f"sc_{s}_ewm"] = f"recency-weighted: {what}"
    for s in MAIN:
        c[f"sc_{s}_best3"] = f"best of its last three: {SERIES[s]}"
        c[f"sc_{s}_sd5"] = f"spread over its last five: {SERIES[s]}"
        c[f"sc_{s}_trend"] = f"mean of 3 less mean of 10: {SERIES[s]}"
    c.update({
        "sc_n": "its earlier days with a first sectional figured",
        "sc_s1_here": "its S1 figure (recency-weighted) at today's track and trip",
        "sc_fsp_here": "its FSP (recency-weighted) at today's track and trip",
        "sc_race_pace": "today's race's projected early pace: the mean of its runners' S1 against race par (recency-weighted)",
        "sc_s1_vs": "its recency-weighted S1 less the mean of today's other runners'",
        "sc_s1_rank": "its rank on recency-weighted S1 in today's race (1 the fastest early)",
        "sc_s1_gap": "its recency-weighted S1 less today's fastest",
        "sc_s2_vs": "its recency-weighted S2 less the mean of today's other runners'",
        "sc_fsp_vs": "its recency-weighted FSP less the mean of today's other runners'",
        "sc_tt_vs": "its recency-weighted whole-time figure less the mean of today's other runners'",
        "sc_closer_x_pace": "sc_fsp_vs x sc_race_pace (a closer in a fast-run race)",
        "sc_trip_x_fsp": "today's trip less its last trip (100 m) x its recency-weighted FSP less 100",
    })
    return c


COLUMNS = _cols()


def run_figures(r: pd.DataFrame) -> dict:
    """Every run's sections, figures, going, links and race-relative measures (from its result: read by later days
    only through ``speed.earlier``)."""
    n = len(r)
    num = speed._num
    day = core.day(r)
    dn = ((day - pd.Timestamp("2000-01-01")).dt.days).to_numpy(float)
    month = (day.dt.year * 12 + day.dt.month).to_numpy(float)
    card = r.is_card.to_numpy(bool) if "is_card" in r else np.zeros(n, bool)
    hurdle = r.hurdles.fillna(False).astype(bool).to_numpy() if "hurdles" in r else np.zeros(n, bool)
    hcap = (r.handicap.fillna(False).astype(bool).to_numpy() if "handicap" in r else np.zeros(n, bool)) | \
        (num(r.hcp_start) > 0 if "hcp_start" in r else np.zeros(n, bool))
    dnf = r.dnf.fillna(False).astype(bool).to_numpy() if "dnf" in r else np.zeros(n, bool)
    bad = r.comment.fillna("").astype(str).str.contains(r"\(Handslip\)|Fell|Fll|BrkLine", case=False).to_numpy() \
        if "comment" in r else np.zeros(n, bool)
    key = (r.track.astype(str) + "|" + pd.to_numeric(r.distance_m, errors="coerce").round().astype("Int64").astype(str)
           + "|" + hurdle.astype(int).astype(str)).to_numpy()
    grade = r.race_class.astype(str).to_numpy() if "race_class" in r else np.full(n, "")
    meet = (r.meeting_id.astype(str) if "meeting_id" in r else r.track.astype(str) + day.astype(str)).to_numpy()
    trap = pd.to_numeric(r.trap, errors="coerce").fillna(0).astype(int).astype(str).to_numpy()
    dog = pd.to_numeric(r.dog_id, errors="coerce").to_numpy(float)
    race = r.race_id.to_numpy()
    ok = ~card & ~hcap & ~dnf & ~bad & np.isfinite(num(r.position))
    early = np.asarray(day < pd.Timestamp(speed.K_BEFORE), bool)
    if not early.any():
        early = dn <= np.nanquantile(dn, 0.2)
    # the sections: the dog's time from the winner's run time and the lengths it was beaten
    rr = r.assign(adjusted_time=r.run_time) if "run_time" in r else r
    T = speed.capped_times(rr, ok)
    S1 = num(r.sectional)
    good = ok & np.isfinite(T) & np.isfinite(S1) & (S1 > 1) & (S1 < 15) & (S1 < T - 1)
    S1, T = np.where(good, S1, np.nan), np.where(good, T, np.nan)
    S2 = T - S1
    l1, l2, lt = np.log(S1), np.log(S2), np.log(T)
    r2, r1 = l2 - lt, l1 - lt                        # log shares of the time in each section
    base = pd.DataFrame({"race_date": r.race_date.to_numpy(), "_k": key, "_t": trap, "_g": grade})
    for c, v in (("l1", l1), ("l2", l2), ("lt", lt), ("r1", r1), ("r2", r2)):
        base[f"_{c}"] = np.nan_to_num(v)
    base["_n"] = good.astype(float)
    pk = core.prior_window(base, ["_k"], ["_l1", "_l2", "_lt", "_r1", "_r2", "_n"], speed.STD_DAYS)
    pt = core.prior_window(base, ["_k", "_t"], ["_l1", "_n"], speed.STD_DAYS)

    def par(w, c):
        return np.where(w._n >= speed.MIN_PAR, w[c] / w._n.replace(0, np.nan), np.nan)

    p1 = np.where(pt._n >= speed.MIN_PAR, pt._l1 / pt._n.replace(0, np.nan), par(pk, "_l1"))
    F = {"s1": 100.0 * (p1 - l1), "s2": 100.0 * (par(pk, "_l2") - l2), "tt": 100.0 * (par(pk, "_lt") - lt),
         "fsp": 100.0 * np.exp(par(pk, "_r2") - r2), "esp": 100.0 * np.exp(par(pk, "_r1") - r1)}
    # race pars (the race's grade at the track and trip, earlier days) and each section's going, a night
    for c in ("s1", "s2", "tt", "fsp"):
        base[f"_f{c}"] = np.where(good & np.isfinite(F[c]), F[c], 0.0)
    base["_fn"] = (good & np.isfinite(F["s1"]) & np.isfinite(F["s2"])).astype(float)
    gp = core.prior_window(base, ["_k", "_g"], ["_fs1", "_fs2", "_ftt", "_ffsp", "_fn"], speed.STD_DAYS)
    rp = {c: np.where(gp._fn >= speed.MIN_PAR, gp[f"_f{c}"] / gp._fn.replace(0, np.nan),
                      0.0 if c != "fsp" else 100.0) for c in ("s1", "s2", "tt", "fsp")}
    G, notes = {}, {}
    for c in ("s1", "s2", "tt", "fsp"):
        use = good & np.isfinite(F[c])
        v, k = speed.eb_variant(meet, F[c] - rp[c], use, early)
        G[c] = F[c] - v
        notes[f"k_going_{c}"] = round(k, 3)
        if c == "s1":
            Gv = v
    # linked across tracks (S1, S2, T)
    months = np.unique(month[np.isfinite(month)])
    L = {}
    for c in ("s1", "s2", "tt"):
        offs = speed.link_offsets(dog, dn, key, G[c], month, good & np.isfinite(G[c]), months)
        off = np.array([offs.get(m, {}).get(kk, 0.0) for m, kk in zip(month, key)])
        L[c] = G[c] + off
        notes[f"keys_linked_{c}"] = len(offs[months[-1]]) if len(months) else 0
    L["fsp"], L["esp"] = G["fsp"], F["esp"]
    out = dict(L)
    for c in ("s1", "s2", "tt", "fsp"):
        out[f"{c}_rp"] = G[c] - rp[c]
        out[f"{c}_vf"] = speed._vs_field(race, np.where(good, L[c], np.nan))
    s1s, s2s = pd.Series(np.where(good, S1, np.nan)), pd.Series(np.where(good, S2, np.nan))
    full = s1s.notna().groupby(race).transform("sum").to_numpy() == pd.Series(ok).groupby(race).transform("sum").to_numpy()
    out["k1"] = np.where(full, s1s.groupby(race).rank(method="min").to_numpy(), np.nan)
    out["k2"] = np.where(full, s2s.groupby(race).rank(method="min").to_numpy(), np.nan)
    out["gain"] = out["k1"] - num(r.position)
    out["close"] = out["k1"] - out["k2"]
    out["pace"] = pd.Series(np.where(good, out["s1_rp"], np.nan)).groupby(race).transform("mean").to_numpy()
    notes.update({"runs_with_both_sections": int(good.sum()), "s1_going_sd": round(float(np.nanstd(Gv[good])), 3)
                  if good.any() else None, "fsp_median": round(float(np.nanmedian(out["fsp"])), 2) if good.any() else None})
    return {"dog": dog, "dn": dn, "race": race, "key": key, "out": out, "notes": notes,
            "dist": pd.to_numeric(r.distance_m, errors="coerce").to_numpy(float)}


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None):
    r = runs
    f = run_figures(r)
    dog, dn, race, o = f["dog"], f["dn"], f["race"], f["out"]
    e = speed.earlier
    col = {}
    for s in SERIES:
        v = np.asarray(o[s], float)
        col[f"sc_{s}_last"] = e(dog, dn, v, "last")
        col[f"sc_{s}_mean3"] = e(dog, dn, v, "mean", n=3)
        col[f"sc_{s}_ewm"] = e(dog, dn, v, "ewm")
    for s in MAIN:
        v = np.asarray(o[s], float)
        col[f"sc_{s}_best3"] = e(dog, dn, v, "max", n=3)
        m5, q5 = e(dog, dn, v, "mean", n=5), e(dog, dn, v * v, "mean", n=5)
        col[f"sc_{s}_sd5"] = np.sqrt(np.maximum(q5 - m5 * m5, 0.0))
        col[f"sc_{s}_trend"] = col[f"sc_{s}_mean3"] - e(dog, dn, v, "mean", n=10)
    col["sc_n"] = e(dog, dn, np.asarray(o["s1"], float), "count")
    here = pd.factorize(pd.Series(dog).astype(str) + "|" + f["key"])[0].astype(float)
    col["sc_s1_here"] = e(here, dn, np.asarray(o["s1"], float), "ewm")
    col["sc_fsp_here"] = e(here, dn, np.asarray(o["fsp"], float), "ewm")
    rp_ewm = e(dog, dn, np.asarray(o["s1_rp"], float), "ewm")
    col["sc_race_pace"] = pd.Series(rp_ewm).groupby(race).transform("mean").to_numpy()
    s1 = col["sc_s1_ewm"]
    col["sc_s1_vs"] = speed._vs_field(race, s1)
    col["sc_s1_rank"] = pd.Series(s1).groupby(race).rank(ascending=False, method="average").to_numpy()
    col["sc_s1_gap"] = s1 - pd.Series(s1).groupby(race).transform("max").to_numpy()
    col["sc_s2_vs"] = speed._vs_field(race, col["sc_s2_ewm"])
    col["sc_fsp_vs"] = speed._vs_field(race, col["sc_fsp_ewm"])
    col["sc_tt_vs"] = speed._vs_field(race, col["sc_tt_ewm"])
    col["sc_closer_x_pace"] = col["sc_fsp_vs"] * col["sc_race_pace"]
    last_dist = e(dog, dn, f["dist"], "last")
    col["sc_trip_x_fsp"] = (f["dist"] - last_dist) / 100.0 * (col["sc_fsp_ewm"] - 100.0)
    tab = pd.DataFrame({"race_id": r.race_id.to_numpy(), "dog_id": r.dog_id.to_numpy(),
                        **{c: np.asarray(col[c], float).astype(np.float32) for c in COLUMNS}}, index=r.index)
    return tab, f["notes"]
