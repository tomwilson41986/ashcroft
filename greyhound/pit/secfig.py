"""Sectional figures on one scale across tracks (the owner, 17 Nov 2026: "Can we adjust sectional based on going
allowance? Vs field. We need to standardize / normalise to use consistently across tracks"; ledger
greyhound-secfig-batch49-1117).

The speed table (greyhound/pit/speed.py) figures the first sectional against the track, trip and trap's par with its
own meeting variant, but (a) the variant is read from every runner against the par, so a card of top grades reads as
a fast night (the time figure takes each winner against its grade's par; the sectional does not), and (b) the
sectional's par is not linked across tracks: a figure at one track does not mean the same at another (the time figure
is linked, step 4 there). This table builds the first sectional's figure with both, and reads it against the field:

1. **The figure**: ``100 x (par - ln sectional)``, the par the mean log sectional of every runner at the track, trip
   and trap over the ``speed.STD_DAYS`` before the day (at least ``speed.MIN_PAR``; else the track and trip's).
2. **Grade-aware going**: each runner's figure less its grade's sectional par at the track and trip (the mean figure
   of that grade's runners there over the year before), averaged over the meeting and shrunk by n / (n + k)
   (``speed.eb_variant``: k from the meetings before ``speed.K_BEFORE``). The night's sectional going.
3. **Linked across tracks**: a monthly offset a track and trip fitted by ridge regression on the dogs that ran at two
   of them within ``speed.MOVE_DAYS``, on the ``speed.LINK_MONTHS`` before the month (``speed.link_offsets``), so a
   sectional figure at one track means the same at another.
4. **Against the field**: each run's linked figure less the mean of the other runners' in that race (its early pace
   relative to the dogs it met, the same at every track).

**The columns** (a dog's EARLIER days only): ``sx_last``, ``sx_mean3``, ``sx_best3``, ``sx_ewm`` (half-life
``speed.HALF`` days), ``sx_n``, ``sx_rel_ewm`` (its race-relative sectionals weighted by recency), ``sx_rel_last``,
``sx_var_last`` (the sectional going of its last figured night); and from today's runners' pre-race columns
``sx_vs`` (``sx_ewm`` less the mean of the other runners') and ``sx_rank`` (1 the race's best ``sx_ewm``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core, speed

PREFIX = "sx_"
SOURCE = "GBGB results (sources/gbgb.py): sectional, trap, grade, track, trip, meeting"
COLUMNS = {
    "sx_last": "the dog's last first-sectional figure (percent, higher faster): against the track, trip and trap par, less the night's grade-aware sectional going, linked across tracks",
    "sx_mean3": "the mean of its last three linked sectional figures",
    "sx_best3": "the best of its last three linked sectional figures",
    "sx_ewm": "its linked sectional figures weighted by recency",
    "sx_n": "its earlier days with a sectional figure",
    "sx_rel_ewm": "its sectional against the field it met (the run's linked figure less the race's other runners' mean), weighted by recency",
    "sx_rel_last": "its last run's sectional against the field it met",
    "sx_var_last": "the grade-aware sectional going of its last figured night (above 0: a fast night)",
    "sx_vs": "sx_ewm less the mean of the other runners' today",
    "sx_rank": "the rank of sx_ewm in today's race (1 the fastest early)",
}


def figures(r: pd.DataFrame) -> dict:
    """Each run's sectional figure (raw, less the going, linked) and its race-relative figure, with the fit notes."""
    n = len(r)
    day = core.day(r)
    dn = ((day - pd.Timestamp("2000-01-01")).dt.days).to_numpy(float)
    month = (day.dt.year * 12 + day.dt.month).to_numpy(float)
    num = speed._num
    card = r.is_card.to_numpy(bool) if "is_card" in r else np.zeros(n, bool)
    hurdle = r.hurdles.fillna(False).astype(bool).to_numpy() if "hurdles" in r else np.zeros(n, bool)
    hcap = (r.handicap.fillna(False).astype(bool).to_numpy() if "handicap" in r else np.zeros(n, bool)) | \
        (num(r.hcp_start) > 0 if "hcp_start" in r else np.zeros(n, bool))
    dnf = r.dnf.fillna(False).astype(bool).to_numpy() if "dnf" in r else np.zeros(n, bool)
    key = (r.track.astype(str) + "|" + pd.to_numeric(r.distance_m, errors="coerce").round().astype("Int64").astype(str)
           + "|" + hurdle.astype(int).astype(str)).to_numpy()
    grade = r.race_class.astype(str).to_numpy() if "race_class" in r else np.full(n, "")
    meet = (r.meeting_id.astype(str) if "meeting_id" in r else r.track.astype(str) + day.astype(str)).to_numpy()
    trap = pd.to_numeric(r.trap, errors="coerce").fillna(0).astype(int).astype(str).to_numpy()
    dog = pd.to_numeric(r.dog_id, errors="coerce").to_numpy(float)
    race = r.race_id.to_numpy()
    ok = ~card & ~hcap & ~dnf & np.isfinite(num(r.position))
    early = np.asarray(day < pd.Timestamp(speed.K_BEFORE), bool)
    if not early.any():
        early = dn <= np.nanquantile(dn, 0.2)
    sec = num(r.sectional)
    sok = ok & np.isfinite(sec) & (sec > 1) & (sec < 15)
    ls = np.where(sok, np.log(np.where(sok, sec, 1.0)), np.nan)
    base = pd.DataFrame({"race_date": r.race_date.to_numpy(), "_key": key, "_trap": trap, "_grade": grade,
                         "_ls": np.nan_to_num(ls), "_sn": sok.astype(float)})
    pt = core.prior_window(base, ["_key", "_trap"], ["_ls", "_sn"], speed.STD_DAYS)
    pk = core.prior_window(base, ["_key"], ["_ls", "_sn"], speed.STD_DAYS)
    par = np.where(pt._sn >= speed.MIN_PAR, pt._ls / pt._sn.replace(0, np.nan),
                   np.where(pk._sn >= speed.MIN_PAR, pk._ls / pk._sn.replace(0, np.nan), np.nan))
    S = 100.0 * (par - ls)
    # 2. the night's sectional going, each runner against its grade's sectional par (earlier days)
    fin = sok & np.isfinite(S)
    base["_gs"] = np.where(fin, S, 0.0)
    base["_gn"] = fin.astype(float)
    gp = core.prior_window(base, ["_key", "_grade"], ["_gs", "_gn"], speed.STD_DAYS)
    gpar = np.where(gp._gn >= speed.MIN_PAR, gp._gs / gp._gn.replace(0, np.nan), 0.0)
    v, k = speed.eb_variant(meet, S - gpar, fin, early)
    SV = S - v
    # 3. linked across tracks
    months = np.unique(month[np.isfinite(month)])
    offs = speed.link_offsets(dog, dn, key, SV, month, sok, months)
    off = np.array([offs.get(m, {}).get(kk, 0.0) for m, kk in zip(month, key)])
    L = SV + off
    # 4. against the field it met
    rel = speed._vs_field(race, np.where(fin, L, np.nan))
    return {"dog": dog, "dn": dn, "race": race, "L": L, "rel": rel, "v": np.where(fin, v, np.nan),
            "notes": {"k_sectional_going": round(k, 3), "sectionals_figured": int(np.isfinite(L).sum()),
                      "offset_months": len(offs), "keys_linked_last_month": len(offs[months[-1]]) if len(months) else 0,
                      "offset_sd_last_month": round(float(np.std(list(offs[months[-1]].values()))), 3)
                      if len(months) and offs[months[-1]] else None,
                      "variant_sd": round(float(np.nanstd(v[fin])), 3) if fin.any() else None}}


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None):
    r = runs
    f = figures(r)
    dog, dn, race, L = f["dog"], f["dn"], f["race"], f["L"]
    e = speed.earlier
    out = {"sx_last": e(dog, dn, L, "last"), "sx_mean3": e(dog, dn, L, "mean"), "sx_best3": e(dog, dn, L, "max"),
           "sx_ewm": e(dog, dn, L, "ewm"), "sx_n": e(dog, dn, L, "count"),
           "sx_rel_ewm": e(dog, dn, f["rel"], "ewm"), "sx_rel_last": e(dog, dn, f["rel"], "last"),
           "sx_var_last": e(dog, dn, f["v"], "last")}
    out["sx_vs"] = speed._vs_field(race, out["sx_ewm"])
    out["sx_rank"] = pd.Series(out["sx_ewm"]).groupby(race).rank(ascending=False, method="average").to_numpy()
    tab = pd.DataFrame({"race_id": r.race_id.to_numpy(), "dog_id": r.dog_id.to_numpy(),
                        **{c: np.asarray(out[c], float).astype(np.float32) for c in COLUMNS}}, index=r.index)
    return tab, f["notes"]
