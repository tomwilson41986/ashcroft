"""The draw that moves (the BFSP plan, reports/greyhound_bfsp_plan.md, step 2.3): each trap's bias at each track and
distance as a state that drifts from meeting to meeting, not one average over every year.

Two readings a trap a day, each a Kalman local level (the state's variance grows by ``q`` a day; a day's reading has
the noise of its races, ``r`` / races), fitted by the one-step predictive likelihood on the days before
``FIT_BEFORE``:
- ``win``: the trap's winners less the SP's chances (what the market has been missing about the trap);
- ``lead``: the trap's first-bend leads (the fastest first sectional) less its fair share, 1 / field.

Every column reads the state after the trap's EARLIER days: ``dr_win_vs_sp``, ``dr_lead`` and their spreads; and
``dr_x_pace``: the trap's lead bias times the dog's early pace against its field (its first-sectional figure over its
last 5 earlier days, from greyhound/metrics.py ``esr``), where a fast starter in a trap that leads gains most.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core

PREFIX = "dr_"
FIT_BEFORE = "2018-01-01"
# the first build (run 42842889301) chose 1e-6, the grid's edge, for both readings: a nearly static state
QS = (0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4)
SOURCE = "GBGB results (sources/gbgb.py): trap, position, sectional, field, SP"


def _readings(df: pd.DataFrame) -> pd.DataFrame:
    """A row a trap a track a distance a day: the sums and counts of the two readings over the day's races."""
    racing = ~df.is_trial.fillna(False).astype(bool) if "is_trial" in df else pd.Series(True, index=df.index)
    sec = pd.to_numeric(df.sectional, errors="coerce")
    first = sec.groupby(df.race_id).transform("min")
    known = sec.notna().groupby(df.race_id).transform("all")
    lead = ((sec == first).astype(float) - 1.0 / df.field).where(known & racing)
    win = (pd.to_numeric(df.won, errors="coerce") - pd.to_numeric(df.sp_p_norm, errors="coerce")).where(racing)
    key = df.track.astype(str) + "|" + pd.to_numeric(df.distance_m, errors="coerce").round().astype("Int64").astype(str) \
        + "|" + df.trap.astype(str)
    d = pd.DataFrame({"key": key, "dn": (core.day(df) - pd.Timestamp("2000-01-01")).dt.days.to_numpy(),
                      "win": win, "lead": lead})
    g = d.groupby(["key", "dn"])
    return pd.DataFrame({"win_s": g.win.sum(), "win_n": g.win.count(), "lead_s": g.lead.sum(),
                         "lead_n": g.lead.count()}).reset_index()


def kalman_days(rd: pd.DataFrame, col: str, q: float, r: float, upto: int | None = None) -> tuple[pd.DataFrame, float]:
    """The state before each (key, day) and the mean one-step predictive log likelihood of the day means read."""
    keys = pd.factorize(rd.key)[0]
    nk = keys.max() + 1
    m, v, last = np.zeros(nk), np.full(nk, 0.01), np.full(nk, -1)
    pre_m, pre_v = np.full(len(rd), np.nan), np.full(len(rd), np.nan)
    s, n = rd[f"{col}_s"].to_numpy(float), rd[f"{col}_n"].to_numpy(float)
    dn = rd.dn.to_numpy()
    order = np.argsort(dn, kind="stable")
    bounds = np.flatnonzero(np.diff(dn[order])) + 1
    ll, cnt = 0.0, 0
    for rows in np.split(order, bounds):
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
    return pd.DataFrame({"m": pre_m, "sd": np.sqrt(pre_v)}, index=rd.index), (ll / cnt if cnt else np.nan)


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = figs if figs is not None else core.per_run_figures(runs)
    rd = _readings(df)
    cut_day = pd.Timestamp(FIT_BEFORE)
    cut = int((cut_day - pd.Timestamp("2000-01-01")).days)
    if (rd.dn < cut).sum() < 0.05 * len(rd):
        cut = int(np.sort(rd.dn.to_numpy())[int(len(rd) * 0.2)]) if len(rd) else cut
    notes = {"fit_before": FIT_BEFORE, "keys": int(rd.key.nunique())}
    states = {}
    for col in ("win", "lead"):
        early = rd[rd.dn < cut]
        # a single race's reading noise: the spread of one race's residual (the day mean divides it by the races)
        r = float(np.nanvar(early[f"{col}_s"] / early[f"{col}_n"].replace(0, np.nan) *
                            np.sqrt(early[f"{col}_n"].replace(0, np.nan)))) if len(early) else 0.1
        r = r if np.isfinite(r) and r > 0 else 0.1
        grid = {}
        for q in QS:
            _, ll = kalman_days(rd, col, q, r, upto=cut)
            grid[q] = round(ll, 5) if np.isfinite(ll) else None
        ok = {k: v for k, v in grid.items() if v is not None}
        q = max(ok, key=ok.get) if ok else 1e-5
        st, _ = kalman_days(rd, col, q, r)
        states[col] = st
        notes[col] = {"q": q, "r": round(r, 5), "grid": grid}
    rd = rd.assign(win_m=states["win"].m, win_sd=states["win"].sd, lead_m=states["lead"].m, lead_sd=states["lead"].sd)
    key = df.track.astype(str) + "|" + pd.to_numeric(df.distance_m, errors="coerce").round().astype("Int64").astype(str) \
        + "|" + df.trap.astype(str)
    j = pd.DataFrame({"key": key.to_numpy(), "dn": (core.day(df) - pd.Timestamp("2000-01-01")).dt.days.to_numpy()}) \
        .merge(rd[["key", "dn", "win_m", "win_sd", "lead_m", "lead_sd"]], on=["key", "dn"], how="left")
    out = pd.DataFrame({"race_id": df.race_id.to_numpy(), "dog_id": df.dog_id.to_numpy()}, index=df.index)
    out[f"{PREFIX}win_vs_sp"] = j.win_m.to_numpy()
    out[f"{PREFIX}win_sd"] = j.win_sd.to_numpy()
    out[f"{PREFIX}lead"] = j.lead_m.to_numpy()
    out[f"{PREFIX}lead_sd"] = j.lead_sd.to_numpy()
    # the dog's early pace against its field: its first-sectional figure over its last 5 earlier days
    esr = core.dog_lagged(df, "esr", 5) if "esr" in df else pd.Series(np.nan, index=df.index)
    g = esr.groupby(df.race_id)
    others = (g.transform("sum") - esr.fillna(0)) / (g.transform("count") - esr.notna()).replace(0, np.nan)
    pace = (esr - others)
    out[f"{PREFIX}pace_vs_field"] = pace.to_numpy()
    out[f"{PREFIX}x_pace"] = (out[f"{PREFIX}lead"] * pace).to_numpy()
    cols = [c for c in out.columns if c.startswith(PREFIX)]
    out[cols] = out[cols].astype(np.float32)
    return out.reset_index(drop=True), notes


COLUMNS = {
    f"{PREFIX}win_vs_sp": "the trap's recent winners less the SP's chances at this track and distance (a drifting state)",
    f"{PREFIX}win_sd": "its spread",
    f"{PREFIX}lead": "the trap's recent first-bend leads less its fair share (a drifting state)",
    f"{PREFIX}lead_sd": "its spread",
    f"{PREFIX}pace_vs_field": "the dog's early pace (first-sectional figure, last 5 days) against its field (lengths)",
    f"{PREFIX}x_pace": "the trap's lead bias times the dog's early pace against its field",
}
