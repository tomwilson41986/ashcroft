"""The fear trace: conditioning to a place on the track, extinction, renewal and recovery (the owner, 11 Nov 2026:
"identify NEW features that have not been discovered yet").

A dog hit at a bend learns the bend. Learning theory says what follows a conditioning trial: the fear is tied to the
place, it is extinguished by CLEAN PASSES of the place and not by time, it is renewed when the context changes (a new
track), and it partly recovers by itself after a long break. The trouble and expected-time tables credit a troubled
run's time and move on; this table carries the conditioned state into the dog's later races.

**The events.** Each race comment is split on commas (lower case, spaces out: ``greyhound.pit.trouble.codes``). A
token is an interference event when ``greyhound.pit.xtime.TROUBLE`` matches it (crowded, bumped, baulked, checked,
hampered, impeded, stumbled). Its severity: 1 if it holds "slt" or "slight"; else 4 if it starts "vb" or holds
"verybadly"; else 3 if it starts "b" followed by crd, bmp, blk, baulk, bump or crowd, or holds "badly"; else 2 (the
words written out are the review's, 11 Nov 2026: 79 "slight..." and 61 "verybadly..." tokens in the runs of 1 Jul
2025 to 5 Nov 2026 were read as 2 and 3). Its places, read
from the token: "rnup", "runup", "stt", "start", "1/4" (or the quarter sign) and the digit 1 are EARLY; "1/2", "3/4"
(or their signs) and the digits 2, 3, 4 are MID; "rnin", "runin" and the digits 5, 6 are LATE (the fractions are
read first and taken out, so "crd1/2" is mid, not early); a token such as "bmp1&2" counts at both places; a token
with no place is mid.

**The state**, one row a dog and day it raced, in day order, for each place P:

- ``F_P <- F_P x RHO`` (``RHO`` = 0.7) on a day it ran with a comment and no interference at P (a clean pass), then
  ``F_P <- F_P + the severities of the day's events at P``. Time does nothing to F. A day with no comment (a void
  race, a card) is neither a pass nor an event;
- ``R_P``: the same sum without extinction, halved every ``HALF_LIFE`` (180) days (the memory extinction does not
  erase), carried to today's date when it is read.

A row reads the state after the dog's EARLIER days, never the day's own comment (two runs of a dog on one day read
the same state). The recursion F_k = F_(k-1) g_k + s_k is G_k x cumsum(s_j / G_j) with G = cumprod(g), done in logs
with one cumulative sum a dog.

**The columns**: the three traces; the raw early memory; the clean early passes since the last early event; the
severity of and days since the last event of any kind; the events so far; the renewal (a first race at this track:
what extinction had taken off the early trace comes back); the spontaneous recovery (the raw early memory times
1 - exp(-days off / 60)); and the dog's own resilience: its mean change in sectional residual
((sectional - the expected-time table's xt_sec_time) / 0.08, lengths, graded runs) from the day before an early
event to the day after it, shrunk by ``K`` (3) events toward none (above 0: its early pace drops after a hit). The
two days either side must be days it raced with no early event of their own and with a sectional residual; a triple
counts only once its day after is an earlier day than the row.

Trials (the T classes ``greyhound.data.clean`` drops) are not read: a trial's pass is not a pass in traffic.

Every column is read from earlier days and today's track and date only (tests/test_greyhound_fear.py).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from greyhound.pit import core, xtime

PREFIX = "fr_"
SOURCE = ("GBGB race comments (sources/gbgb.py) read with the expected-time table's interference pattern "
          "(greyhound.pit.xtime.TROUBLE); sectionals against the expected-time table (xt_sec_time)")
RHO = 0.7                    # what a clean pass leaves of the trace
HALF_LIFE = 180.0            # days: the raw memory's half life
TAU = 60.0                   # days: the time constant of spontaneous recovery
K = 3.0                      # events of prior toward no change in early pace after a hit
SEC_CLIP = 15.0              # lengths: a sectional further than this from its expected one is not read
MAX_PASSES = 1900            # exp(MAX_PASSES x -ln RHO) is finite in float64; no dog has this many clean passes
PLACES = ("early", "mid", "late")
_SLIGHT = re.compile(r"slt|slight")                 # "SltCrd1", and the same word written out ("SlightlyCrowded1")
_VERY = re.compile(r"^vb|verybadly")                # "VBBmp1", "VeryBadlyBumped1"
_SEV3 = re.compile(r"^b(crd|bmp|blk|baulk|bump|crowd)|badly")
_FRAC = re.compile(r"1/4|1/2|3/4|[¼½¾]")
_EARLY_WORD = re.compile(r"rnup|runup|stt|start")
_LATE_WORD = re.compile(r"rnin|runin")


def parse_token(tok: str) -> tuple[int, bool, bool, bool, bool]:
    """One comment token (lower case, no spaces): (severity, early, mid, late, a place was found); severity 0 when
    the token is not interference."""
    if not xtime.TROUBLE.search(tok):
        return 0, False, False, False, False
    sev = 1 if _SLIGHT.search(tok) else 4 if _VERY.search(tok) else 3 if _SEV3.search(tok) else 2
    fracs = _FRAC.findall(tok)
    rest = _FRAC.sub("", tok)
    early = bool(_EARLY_WORD.search(rest)) or "1" in rest or any(f in ("1/4", "¼") for f in fracs)
    mid = any(d in rest for d in "234") or any(f not in ("1/4", "¼") for f in fracs)
    late = bool(_LATE_WORD.search(rest)) or "5" in rest or "6" in rest
    found = early or mid or late
    return sev, early, mid or not found, late, found


def events(comment: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Each row's interference from its comment: the summed severity at each place (``early``, ``mid``, ``late``),
    the events (``n``), the worst severity (``worst``) and whether there is a comment at all (``said``). The loop is
    over the distinct tokens only."""
    c = comment.astype(object).where(comment.notna(), "").astype(str)
    code, uniq = pd.factorize(c)
    toks = pd.Series(uniq, dtype=object).str.lower().str.replace(r"\s+", "", regex=True).str.split(",").explode()
    toks = toks[toks.fillna("").str.len() > 0]
    which = toks.index.to_numpy()                                   # the distinct comment each token belongs to
    tcode, tuniq = pd.factorize(toks.to_numpy())
    parsed = np.array([parse_token(t) for t in tuniq], dtype=float).reshape(-1, 5)
    p = parsed[tcode] if len(tcode) else np.zeros((0, 5))
    nu = len(uniq)
    sev = p[:, 0]
    per = {"n": np.bincount(which, weights=(sev > 0).astype(float), minlength=nu),
           "said": np.bincount(which, minlength=nu) > 0}
    for j, name in enumerate(PLACES):
        per[name] = np.bincount(which, weights=sev * p[:, 1 + j], minlength=nu)
    worst = np.zeros(nu)
    np.maximum.at(worst, which, sev)
    per["worst"] = worst
    out = pd.DataFrame({k: v[code] for k, v in per.items()}, index=comment.index)
    notes = {"distinct_comments": int(nu), "distinct_tokens": int(len(tuniq)),
             "distinct_interference_tokens": int((parsed[:, 0] > 0).sum()) if len(parsed) else 0,
             "distinct_interference_tokens_with_no_place": int(((parsed[:, 0] > 0) & (parsed[:, 4] == 0)).sum())
             if len(parsed) else 0}
    return out, notes


def sec_residuals(df: pd.DataFrame, xt_sec_time: pd.Series) -> pd.Series:
    """Each graded run's sectional in lengths slower than its expected sectional (NaN: a trial, a fall, a card row,
    no clock, an outlier). The run's own result: it may only enter a feature of a later day."""
    bad = df.comment.fillna("").str.contains(r"\(Handslip\)|Fell|Fll|BrkLine", case=False) if "comment" in df else \
        pd.Series(False, index=df.index)
    trial = df.is_trial.fillna(False).astype(bool) if "is_trial" in df else pd.Series(False, index=df.index)
    r = (pd.to_numeric(df.sectional, errors="coerce") - pd.to_numeric(xt_sec_time, errors="coerce")) / core.LENGTH_S
    return r.where(~bad & ~trial & (r.abs() <= SEC_CLIP))


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = figs if figs is not None else core.per_run_figures(runs)
    n = len(df)
    xt = (ctx or {}).get("xtime")
    if xt is None:
        xt, _ = xtime.build(runs, df)
    j = df[["race_id", "dog_id"]].merge(xt[["race_id", "dog_id", "xt_sec_time"]].drop_duplicates(["race_id", "dog_id"]),
                                        on=["race_id", "dog_id"], how="left")
    sr = sec_residuals(df, pd.Series(j.xt_sec_time.to_numpy(), index=df.index)).to_numpy(float)
    card = df.is_card.fillna(False).to_numpy(bool) if "is_card" in df else np.zeros(n, bool)
    comment = df.comment if "comment" in df else pd.Series("", index=df.index)
    ev, notes = events(comment.where(~card, ""))
    dn = (core.day(df) - pd.Timestamp("2000-01-01")).dt.days.to_numpy().astype(np.int64)
    dog = pd.factorize(df.dog_id)[0].astype(np.int64)
    # ---- one row a dog and day, in dog then day order -------------------------------------------------------------
    span = int(dn.max() - dn.min()) + 2 if n else 2
    uk, row_dd = np.unique(dog * span + (dn - (dn.min() if n else 0)), return_inverse=True)
    m = len(uk)
    d_dog, d_dn = uk // span, uk % span + (dn.min() if n else 0)
    new = np.r_[True, d_dog[1:] != d_dog[:-1]] if m else np.zeros(0, bool)
    start = np.flatnonzero(new)
    grp = np.cumsum(new) - 1
    first = start[grp] if m else np.zeros(0, int)                    # the dog's first row
    idx = np.arange(m)
    has_prev = ~new                                                  # the row before is the same dog's earlier day

    def per_day(v: np.ndarray) -> np.ndarray:
        return np.bincount(row_dd, weights=v, minlength=m)

    def dog_cumsum(v: np.ndarray) -> np.ndarray:
        return pd.Series(v).groupby(d_dog).cumsum().to_numpy(float)

    def before(v: np.ndarray, none=np.nan) -> np.ndarray:
        """The state after the dog's previous day (``none`` on its first day)."""
        return np.where(has_prev, np.r_[none, v[:-1]] if m else v, none)

    said = per_day(ev.said.to_numpy(float)) > 0
    sev = {p: per_day(ev[p].to_numpy(float)) for p in PLACES}
    n_ev = per_day(ev.n.to_numpy(float))
    worst = np.zeros(m)
    np.maximum.at(worst, row_dd, ev.worst.to_numpy(float))
    rel = (d_dn - d_dn[first]).astype(float)                         # days since the dog's first day
    gap = np.where(has_prev, d_dn - np.r_[0, d_dn[:-1]], np.nan) if m else np.zeros(0)
    w = np.exp2(rel / HALF_LIFE)
    state = {}
    for p in PLACES:
        hit = sev[p] > 0
        clean = said & ~hit
        passes = np.minimum(dog_cumsum(clean.astype(float)), MAX_PASSES)       # ln G = passes x ln RHO
        f = np.exp(passes * np.log(RHO)) * dog_cumsum(sev[p] * np.exp(-passes * np.log(RHO)))
        state[p] = before(f, 0.0)
        if p == "early":
            raw = before(dog_cumsum(sev[p] * w), 0.0) / w            # carried to today's date
            last_hit = np.maximum.accumulate(np.where(hit, idx, -1))
            since = np.where(last_hit >= first, passes - passes[np.maximum(last_hit, 0)], np.nan)
            clean_early = before(since)
            hit_early = hit
    # the last interference of any kind
    any_ev = n_ev > 0
    last_any = np.maximum.accumulate(np.where(any_ev, idx, -1)) if m else idx
    ok_any = last_any >= first
    la = np.maximum(last_any, 0)
    last_sev = before(np.where(ok_any, worst[la], np.nan))
    last_day = before(np.where(ok_any, d_dn[la].astype(float), np.nan))
    # resilience: the sectional residual the day after an early hit less the day before it, counted on the day after
    s_sum, s_n = per_day(np.nan_to_num(sr)), per_day(np.isfinite(sr).astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        s_day = np.where(s_n > 0, s_sum / s_n, np.nan)
    change = np.full(m, np.nan)
    control = np.full(m, np.nan)
    if m >= 3:
        same = (d_dog[2:] == d_dog[:-2])                             # day before, the day, day after: one dog
        fair = same & said[:-2] & said[2:] & ~hit_early[:-2] & ~hit_early[2:]
        diff = s_day[2:] - s_day[:-2]
        change[2:] = np.where(fair & hit_early[1:-1], diff, np.nan)
        control[2:] = np.where(fair & said[1:-1] & ~hit_early[1:-1], diff, np.nan)
    got = np.isfinite(change)
    r_sum, r_n = before(dog_cumsum(np.nan_to_num(change)), 0.0), before(dog_cumsum(got.astype(float)), 0.0)
    resil = np.where(r_n > 0, r_sum / (r_n + K), np.nan)
    # ---- back to the rows ------------------------------------------------------------------------------------------
    track = pd.factorize(df.track.astype(str))[0].astype(np.int64)
    at = pd.DataFrame({"k": dog * (int(track.max()) + 2 if n else 1) + track, "dn": dn})
    first_here = (dn == at.groupby("k").dn.transform("min").to_numpy()) & has_prev[row_dd]   # raced before, never here
    out = pd.DataFrame({"race_id": df.race_id.to_numpy(), "dog_id": df.dog_id.to_numpy()}, index=df.index)
    for p in PLACES:
        out[f"{PREFIX}{p}"] = state[p][row_dd]
    out[f"{PREFIX}raw_early"] = raw[row_dd]
    out[f"{PREFIX}clean_early"] = clean_early[row_dd]
    out[f"{PREFIX}last_sev"] = last_sev[row_dd]
    out[f"{PREFIX}last_days"] = (d_dn - last_day)[row_dd]
    out[f"{PREFIX}n"] = before(dog_cumsum(n_ev), 0.0)[row_dd]
    extinguished = np.maximum(raw - state["early"], 0.0)
    out[f"{PREFIX}renew"] = np.where(first_here, extinguished[row_dd], 0.0)
    out[f"{PREFIX}recover"] = (raw * (1.0 - np.exp(-np.nan_to_num(gap) / TAU)))[row_dd]
    out[f"{PREFIX}resil"] = resil[row_dd]
    cols = [c for c in out.columns if c.startswith(PREFIX)]
    out[cols] = out[cols].astype(np.float32)
    evn = ev.n.to_numpy(float)
    notes.update({
        "rho": RHO, "half_life_days": HALF_LIFE, "tau_days": TAU, "k_prior": K, "rows": int(n), "dog_days": int(m),
        "dog_days_with_a_comment": int(said.sum()),
        "runs_with_interference": int((evn > 0).sum()), "interference_events": int(evn.sum()),
        "runs_with_interference_at": {p: int((ev[p].to_numpy() > 0).sum()) for p in PLACES},
        "runs_by_worst_severity": {str(int(s)): int((ev.worst.to_numpy() == s).sum()) for s in (1, 2, 3, 4)},
        "rows_with_an_earlier_event": int((out[f"{PREFIX}n"].to_numpy() > 0).sum()),
        "rows_first_at_a_track_with_a_renewal": int((out[f"{PREFIX}renew"].to_numpy() > 0).sum()),
        "early_hit_triples": int(got.sum()), "dogs_with_a_triple": int(len(np.unique(d_dog[got]))),
        "mean_sectional_change_after_an_early_hit": round(float(np.nanmean(change)), 4) if got.any() else None,
        "control_triples": int(np.isfinite(control).sum()),
        "mean_sectional_change_control": round(float(np.nanmean(control)), 4) if np.isfinite(control).any() else None,
    })
    return out.sort_index().reset_index(drop=True), notes


COLUMNS = {
    f"{PREFIX}early": f"the fear trace at the start, run-up and first bend: each earlier day's interference severity "
                      f"there (1 slight, 2, 3 badly, 4 very badly), times {RHO:g} for every clean pass since",
    f"{PREFIX}mid": "the same at bends 2 to 4 (and interference with no place given)",
    f"{PREFIX}late": "the same on the run-in and at bends 5 and 6",
    f"{PREFIX}raw_early": f"the early interference severities with no extinction, halved every {HALF_LIFE:g} days, "
                          "at today's date",
    f"{PREFIX}clean_early": "clean early passes since the last early interference (NaN: none yet)",
    f"{PREFIX}last_sev": "the severity of the dog's last interference of any kind (the worst of that day; NaN: none)",
    f"{PREFIX}last_days": "days since that interference",
    f"{PREFIX}n": "interference events on the dog's earlier days",
    f"{PREFIX}renew": "fear renewed in a new context: on the dog's first race at this track, the raw early memory "
                      "less the early trace (what clean passes elsewhere had taken off; never below 0), else 0",
    f"{PREFIX}recover": f"spontaneous recovery: the raw early memory x (1 - exp(-days since its last race / {TAU:g}))",
    f"{PREFIX}resil": "the dog's mean change in sectional residual (lengths against the expected sectional, above 0 "
                      f"slower) from the day before an early interference to the day after it, shrunk by {K:g} "
                      "events toward 0 (NaN: no such triple on earlier days)",
}
