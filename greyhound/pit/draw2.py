"""The draw with a hierarchy across trips and the dogs' own lead chance taken out (the feature review of 17 Nov 2026,
reports/greyhound_feature_review.md, row "draw": "No hierarchy across trips; dr_lead mostly measures which dogs are
seeded in the trap"). A research variant of greyhound/pit/draw.py (``dr_``), whose output it leaves alone.

Two readings a trap a trip a day, as draw.py, each a Kalman local level fitted by the one-step predictive likelihood
on the early slice (before ``draw.FIT_BEFORE``, or the first 20% of reading-days when that holds under 5%):

- ``win``: the trap's winners less the de-biased SP chances (``core.debiased_probs``, Shin: review B5), not the
  proportionally normalised SP, so the favourite-longshot bias is not read as a trap effect;
- ``lead``: the trap's first-bend leads (the race's single fastest first sectional, every sectional known) less each
  dog's OWN expected lead chance before the race: a conditional logit, with no trap term, on its early pace against
  the field (the engine's ``esr`` over its last 5 earlier days) and its own lead rate (last 10 earlier days, shrunk to
  1 / field by ``K_LEAD`` days), fitted once on the early slice. A racing manager who seeds the quick breakers in
  trap 1 no longer makes trap 1 look biased: only leads beyond what the dogs drawn there were expected to take count.

The hierarchy: the trap's state at this trip (a Kalman level with a flat start, ``VFLAT``) is combined, by precision,
with the same trap's state at the track's OTHER trips (the pooled readings of every other distance at the track, a
level shrunk to no bias as draw.py's) widened by ``tau2``, the between-trip variance of a trap's bias at a track
(method of moments on the early slice). A trip with few races takes the track's trap bias; one with many, its own.
Every state is read before the row's day (the readings of the day itself never enter); card-safe throughout (the
trap, trip and field come from the card; the dog's inputs from earlier days).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core, draw

DEPENDS = ("draw",)
PREFIX = "dr2_"
SOURCE = "GBGB results (sources/gbgb.py): trap, distance, position, first sectional, field, SP (de-biased by Shin)"
VFLAT = 1.0                   # the trip level's starting variance: effectively no prior of its own
V0_OTHER = 0.01               # the other trips' level starts at no bias with draw.py's variance
K_LEAD = 3.0                  # earlier days of prior on a dog's own lead rate, toward 1 / field
MAX_LEAD_FIT = 100_000        # races in the own-lead logit's fit (a random sample of the early slice beyond it)
COLUMNS = {
    f"{PREFIX}win": "the trap's winners less the de-biased SP chances at this track and trip, combined with the same "
                    "trap at the track's other trips as the prior (a drifting state, earlier days)",
    f"{PREFIX}win_sd": "its spread",
    f"{PREFIX}win_other": "the prior: the trap's winners less the de-biased SP at the track's other trips",
    f"{PREFIX}lead": "the trap's first-bend leads less the dogs' own expected lead chances, this trip with the other "
                     "trips as the prior (a drifting state, earlier days)",
    f"{PREFIX}lead_sd": "its spread",
    f"{PREFIX}lead_other": "the prior: the same at the track's other trips",
    f"{PREFIX}p_lead_own": "the dog's own expected chance of leading at the first bend from its early pace and lead "
                           "rate, no trap term (a logit fitted on the early slice)",
    f"{PREFIX}pace_vs_field": "the dog's early pace (first-sectional figure, last 5 days) against its field (lengths)",
    f"{PREFIX}x_pace": "dr2_lead times dr2_pace_vs_field",
}


def kalman(rd: pd.DataFrame, col: str, q: float, r: float, v0: float, upto: int | None = None):
    """draw.kalman_days with the starting variance ``v0``: the state before each (key, day) and the mean one-step
    predictive log likelihood of the day means read before ``upto``."""
    keys = pd.factorize(rd.key)[0]
    nk = keys.max() + 1 if len(keys) else 0
    m, v, last = np.zeros(nk), np.full(nk, v0), np.full(nk, -1)
    pre_m, pre_v = np.full(len(rd), np.nan), np.full(len(rd), np.nan)
    s, n = rd[f"{col}_s"].to_numpy(float), rd[f"{col}_n"].to_numpy(float)
    dn = rd.dn.to_numpy()
    order = np.argsort(dn, kind="stable")
    bounds = np.flatnonzero(np.diff(dn[order])) + 1
    ll, cnt = 0.0, 0
    for rows in np.split(order, bounds) if len(rd) else []:
        if upto is not None and dn[rows[0]] >= upto:
            break
        k = keys[rows]
        vp = v[k] + q * np.where(last[k] >= 0, dn[rows] - last[k], 0)
        pre_m[rows], pre_v[rows] = m[k], vp
        ok = n[rows] > 0
        if not ok.any():
            continue
        k, vp, rr = k[ok], vp[ok], rows[ok]
        z = s[rr] / n[rr]
        rn = r / n[rr]
        e = z - m[k]
        ll += float((-0.5 * (np.log(2 * np.pi * (vp + rn)) + e * e / (vp + rn))).sum())
        cnt += len(rr)
        g = vp / (vp + rn)
        m[k] = m[k] + g * e
        v[k] = (1 - g) * vp
        last[k] = dn[rr]
    return pd.DataFrame({"m": pre_m, "v": pre_v}, index=rd.index), (ll / cnt if cnt else np.nan)


def _vs_field(v: pd.Series, race: pd.Series) -> pd.Series:
    g = v.groupby(race)
    return v - (g.transform("sum") - v.fillna(0)) / (g.transform("count") - v.notna()).replace(0, np.nan)


def own_lead(df: pd.DataFrame, lead: pd.Series, early: np.ndarray, field: pd.Series) -> tuple[np.ndarray, dict]:
    """Each runner's expected lead chance from its own earlier days, no trap: (chances, the fit's notes)."""
    race = df.race_id
    esr = core.dog_lagged(df, "esr", 5) if "esr" in df else pd.Series(np.nan, index=df.index)
    pace = _vs_field(esr, race)
    m = core.dog_lagged(df.assign(_l=lead), "_l", 10)
    n = core.dog_lagged(df.assign(_c=lead.notna().astype(float).where(lead.notna())), "_c", 10, "sum").fillna(0.0)
    base = 1.0 / field.clip(lower=2)
    rate = (m.fillna(0.0) * n + K_LEAD * base) / (n + K_LEAD)
    X = pd.DataFrame({"pace": pace, "rate": np.log(rate / (1 - rate).clip(lower=1e-6))}, index=df.index)
    lab = lead.notna() & lead.groupby(race).transform("sum").eq(1) & lead.groupby(race).transform("count").eq(field)
    fit = lab.to_numpy() & early
    mu, sd = X[fit].mean(), X[fit].std().replace(0, 1.0).fillna(1.0)
    Z = ((X - mu.fillna(0.0)) / sd).fillna(0.0).to_numpy()
    races = race[fit].unique()
    notes = {"fit_races": int(len(races))}
    if len(races) < 20:
        b = np.zeros(2)
    else:
        if len(races) > MAX_LEAD_FIT:
            races = np.random.default_rng(0).choice(races, MAX_LEAD_FIT, replace=False)
        tr = fit & race.isin(races).to_numpy()
        b = core.clogit_fit(Z[tr], pd.factorize(race[tr])[0], lead[tr].to_numpy(float), lam=1.0)
    notes["beta"] = dict(zip(("pace", "rate"), np.round(b, 4)))
    return core.group_softmax(Z @ b, pd.factorize(race)[0]), notes


def _other_trips(rd: pd.DataFrame) -> pd.DataFrame:
    """For each trip key and day of the track-and-trap's pooled readings: the readings of the OTHER trips that day."""
    sums = ["win_s", "win_n", "lead_s", "lead_n"]
    pool = rd.groupby(["key2", "dn"], sort=False)[sums].sum().reset_index()
    x = rd[["key", "key2"]].drop_duplicates().merge(pool, on="key2")
    own = x[["key", "dn"]].merge(rd[["key", "dn"] + sums], on=["key", "dn"], how="left").fillna(0.0)
    x[sums] = x[sums].to_numpy(float) - own[sums].to_numpy(float)
    return x.sort_values(["dn", "key"], kind="stable").reset_index(drop=True)


def _tau2(rd: pd.DataFrame, col: str, r: float, cut: int) -> float:
    """The between-trip variance of a trap's bias at a track, on the early slice (method of moments)."""
    e = rd[rd.dn < cut]
    k = e.groupby(["key2", "key"])[[f"{col}_s", f"{col}_n"]].sum().reset_index()
    k = k[k[f"{col}_n"] >= 30]
    if k.key2.duplicated(keep=False).sum() < 4:
        return 1e-4
    k = k[k.key2.duplicated(keep=False)]
    z = k[f"{col}_s"] / k[f"{col}_n"]
    dev = z - z.groupby(k.key2).transform("mean")
    t = float(np.mean(dev ** 2) - np.mean(r / k[f"{col}_n"]))
    return max(t, 1e-6)


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = figs if figs is not None else core.per_run_figures(runs)
    n = len(df)
    racing = ~df.is_trial.fillna(False).astype(bool) if "is_trial" in df else pd.Series(True, index=df.index)
    field = pd.to_numeric(df.field, errors="coerce") if "field" in df else df.groupby("race_id").dog_id.transform("size")
    field = field.fillna(df.groupby("race_id").dog_id.transform("size")).astype(float)
    dn = (core.day(df) - pd.Timestamp("2000-01-01")).dt.days
    dist = pd.to_numeric(df.distance_m, errors="coerce").round().astype("Int64").astype(str)
    key2 = df.track.astype(str) + "|" + df.trap.astype(str)
    key = df.track.astype(str) + "|" + dist + "|" + df.trap.astype(str)
    cut = int((pd.Timestamp(draw.FIT_BEFORE) - pd.Timestamp("2000-01-01")).days)
    days = np.sort(dn.dropna().unique())
    if len(days) and (days < cut).mean() < 0.05:
        cut = int(days[int(len(days) * 0.2)])
    early = (dn < cut).to_numpy()

    # the lead label and each dog's own expected lead chance (no trap)
    sec = pd.to_numeric(df.sectional, errors="coerce")
    first = sec.groupby(df.race_id).transform("min")
    known = sec.notna().groupby(df.race_id).transform("all")
    lead = (sec == first).astype(float).where(known & racing)
    p_own, lead_notes = own_lead(df, lead, early & racing.to_numpy(), field)
    single = lead.groupby(df.race_id).transform("sum").eq(1)
    lead_r = (lead - p_own).where(single)
    p_deb = core.debiased_probs(pd.to_numeric(df.sp_decimal, errors="coerce") if "sp_decimal" in df
                                else np.full(n, np.nan), df.race_id.to_numpy(), "shin")
    win_r = (pd.to_numeric(df.won, errors="coerce") - p_deb).where(racing)

    d = pd.DataFrame({"key": key, "key2": key2, "dn": dn, "win": win_r, "lead": lead_r}).dropna(subset=["dn"])
    g = d.groupby(["key", "key2", "dn"], sort=False)
    rd = pd.DataFrame({"win_s": g.win.sum(), "win_n": g.win.count(), "lead_s": g.lead.sum(),
                       "lead_n": g.lead.count()}).reset_index().sort_values(["dn", "key"], kind="stable")
    rd = rd.reset_index(drop=True)
    other = _other_trips(rd)
    notes = {"fit_before_day": str(pd.Timestamp("2000-01-01") + pd.Timedelta(days=cut))[:10], "keys": int(rd.key.nunique()),
             "own_lead": lead_notes, "rows": int(n)}
    for col in ("win", "lead"):
        e = rd[(rd.dn < cut) & (rd[f"{col}_n"] > 0)]
        r = float(np.nanvar(e[f"{col}_s"] / e[f"{col}_n"] * np.sqrt(e[f"{col}_n"]))) if len(e) else 0.1
        r = r if np.isfinite(r) and r > 0 else 0.1
        grid = {}
        for q in draw.QS:
            _, ll = kalman(rd, col, q, r, VFLAT, upto=cut)
            grid[q] = round(ll, 5) if np.isfinite(ll) else None
        ok = {k: v for k, v in grid.items() if v is not None}
        q = max(ok, key=ok.get) if ok else 1e-6
        tau2 = _tau2(rd, col, r, cut)
        st, _ = kalman(rd, col, q, r, VFLAT)
        so, _ = kalman(other, col, q, r, V0_OTHER)
        so = pd.concat([other[["key", "dn"]], so], axis=1)
        m = rd[["key", "dn"]].merge(so, on=["key", "dn"], how="left")
        mo, vo = m.m.fillna(0.0).to_numpy(), m.v.fillna(V0_OTHER).to_numpy()
        pt, po = 1.0 / st.v.to_numpy(), 1.0 / (vo + tau2)
        rd[f"{col}_m"] = (st.m.to_numpy() * pt + mo * po) / (pt + po)
        rd[f"{col}_sd"] = np.sqrt(1.0 / (pt + po))
        rd[f"{col}_o"] = m.m.to_numpy()
        notes[col] = {"q": q, "r": round(r, 5), "tau2": round(tau2, 7), "grid": grid}
    j = pd.DataFrame({"key": key.to_numpy(), "dn": dn.to_numpy()}).merge(
        rd[["key", "dn", "win_m", "win_sd", "win_o", "lead_m", "lead_sd", "lead_o"]], on=["key", "dn"], how="left")
    out = pd.DataFrame({"race_id": df.race_id.to_numpy(), "dog_id": df.dog_id.to_numpy()})
    out[f"{PREFIX}win"] = j.win_m.to_numpy()
    out[f"{PREFIX}win_sd"] = j.win_sd.to_numpy()
    out[f"{PREFIX}win_other"] = j.win_o.to_numpy()
    out[f"{PREFIX}lead"] = j.lead_m.to_numpy()
    out[f"{PREFIX}lead_sd"] = j.lead_sd.to_numpy()
    out[f"{PREFIX}lead_other"] = j.lead_o.to_numpy()
    out[f"{PREFIX}p_lead_own"] = p_own
    esr = core.dog_lagged(df, "esr", 5) if "esr" in df else pd.Series(np.nan, index=df.index)
    pace = _vs_field(esr, df.race_id).to_numpy(float)
    out[f"{PREFIX}pace_vs_field"] = pace
    out[f"{PREFIX}x_pace"] = out[f"{PREFIX}lead"].to_numpy() * pace
    cols = list(COLUMNS)
    out[cols] = out[cols].astype(np.float32)
    return out[["race_id", "dog_id"] + cols], notes
