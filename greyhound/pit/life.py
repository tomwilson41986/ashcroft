"""The dog's life and career: age, season, weight, trials and the path through the grades.

- age in days at the run (GBGB gives the birth month: the 15th is taken), sex;
- a bitch's season is NOT a column: GBGB serves the CURRENT season on past results (6 Nov 2026: 2,648 of 16,313
  dated seasons in a 2026 fetch fell after their race), so even a season dated before the race is biased (its absence
  says a later season came). It is counted in the meta only; the day's cards will carry it going forward;
- weight: the run's weigh-in against the dog's mean on earlier days and against its last run (card-unsafe: the
  weigh-in comes shortly before the off);
- trials: how many before, days since the last, the last trial's speed rating against the dog's level;
- the grades: grade change against the last race, races and wins at today's grade before, a run of drops;
- the first time at the track, at the distance; runs in the last 30 days.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from greyhound.pit import core

PREFIX = "lf_"
SOURCE = "GBGB results (born, sex, season, weight, grade, trials: sources/gbgb.py) through the engine's figures"
CARD_UNSAFE = ("lf_wt_vs_mean", "lf_wt_change")
MONTHS = {"ja": 1, "fe": 2, "fb": 2, "mr": 3, "ap": 4, "my": 5, "jn": 6, "jl": 7, "jy": 7, "au": 8, "se": 9, "sp": 9,
          "oc": 10, "no": 11, "nv": 11, "de": 12}


def season_date(s) -> pd.Timestamp | None:
    m = re.fullmatch(r"(\d{1,2})\.([A-Za-z]{2})\.(\d{2})", str(s or "").strip())
    if not m or m.group(2).lower() not in MONTHS:
        return None
    try:
        return pd.Timestamp(2000 + int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
    except ValueError:
        return None


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = figs if figs is not None else core.per_run_figures(runs)
    d = core.day(df)
    out = pd.DataFrame({"race_id": df.race_id, "dog_id": df.dog_id})
    born = pd.to_datetime("15-" + df.born.astype(str), format="%d-%b-%Y", errors="coerce")
    out[f"{PREFIX}age_days"] = (d - born).dt.days
    out[f"{PREFIX}bitch"] = (df.sex.astype(str).str.lower() == "b").astype(float).where(df.sex.notna())
    codes = df.season.astype(str).str.extract(r"^\d{1,2}\.([A-Za-z]{2})\.\d{2}$")[0].str.lower()
    unknown_codes = sorted(set(codes.dropna()) - set(MONTHS))
    uniq = {s: season_date(s) for s in pd.unique(df.season.dropna())}
    sd = pd.to_datetime(df.season.map(uniq), errors="coerce")
    after = sd > d
    # weight (the weigh-in, card-unsafe)
    out[f"{PREFIX}wt_vs_mean"] = df.weight_kg - core.dog_lagged(df, "weight_kg", None)
    out[f"{PREFIX}wt_change"] = df.weight_kg - core.dog_lagged(df, "weight_kg", 1)
    # trials
    trial = df.is_trial.fillna(False) if "is_trial" in df else pd.Series(False, index=df.index)
    t = df.assign(_tr=trial.astype(float), _trgsr=df.gsr.where(trial), _trday=d.where(trial))
    out[f"{PREFIX}trials"] = core.dog_lagged(t, "_tr", None, "sum").fillna(0.0)
    # the last trial on an earlier day, for every run of the dog: carry it forward in time
    tt = t[["dog_id", "t"]].assign(_trday=t._trday, _trgsr=t._trgsr).sort_values(["dog_id", "t"])
    g = tt.groupby("dog_id")
    tt["_ld"] = g._trday.ffill()
    tt["_lg"] = g._trgsr.ffill()
    tt["_ld"] = tt.groupby("dog_id")._ld.shift(1)
    tt["_lg"] = tt.groupby("dog_id")._lg.shift(1)
    tt = tt.reindex(df.index)
    same_day = tt._ld == d                                     # never read a trial from the run's own day
    out[f"{PREFIX}days_since_trial"] = (d - tt._ld).dt.days.where(~same_day)
    lvl = core.dog_lagged(df.assign(_g=df.gsr.where(~trial)), "_g", 5)
    out[f"{PREFIX}trial_vs_level"] = (tt._lg - lvl).where(~same_day)
    # grades and places (races only)
    races = df[~trial]
    prev = core.prev_day_values(races, "dog_id", ["grade", "distance_m", "track", "grade_family"]).reindex(df.index)
    out[f"{PREFIX}grade_change"] = (df.grade - prev.grade).where(prev.grade_family == df.grade_family)
    out[f"{PREFIX}dist_change"] = df.distance_m - prev.distance_m
    gw = core.prior_window(df.assign(_w=df.won.where(~trial).fillna(0.0), _r=(~trial).astype(float)),
                           ["dog_id", "race_class"], ["_w", "_r"], None)
    out[f"{PREFIX}runs_at_grade"] = gw._r
    out[f"{PREFIX}wins_at_grade"] = gw._w
    at_track = core.prior_window(df.assign(_r=(~trial).astype(float)), ["dog_id", "track"], ["_r"], None)
    out[f"{PREFIX}runs_at_track"] = at_track._r
    at_dist = core.prior_window(df.assign(_r=(~trial).astype(float)), ["dog_id", "track", "distance_m"], ["_r"], None)
    out[f"{PREFIX}runs_at_trackdist"] = at_dist._r
    r30 = core.prior_window(df.assign(_r=(~trial).astype(float)), ["dog_id"], ["_r"], 30)
    out[f"{PREFIX}runs_30d"] = r30._r
    drops = (out[f"{PREFIX}grade_change"] > 0).astype(float)
    out[f"{PREFIX}drop_streak"] = drops.groupby(df.dog_id).transform(
        lambda s: s.groupby((s == 0).cumsum()).cumsum())
    meta = {"season_after_run": int(after.sum()), "season_dated": int(sd.notna().sum()),
            "unknown_month_codes": unknown_codes, "born_known": int(born.notna().sum())}
    return out.reset_index(drop=True), meta


COLUMNS = {
    f"{PREFIX}age_days": "age at the run in days (birth month from GBGB, the 15th taken)",
    f"{PREFIX}bitch": "1 a bitch, 0 a dog",
    f"{PREFIX}wt_vs_mean": "the weigh-in less the dog's mean weight on earlier days (card-unsafe)",
    f"{PREFIX}wt_change": "the weigh-in less the last run's (card-unsafe)",
    f"{PREFIX}trials": "trials before the run",
    f"{PREFIX}days_since_trial": "days since the dog's last trial",
    f"{PREFIX}trial_vs_level": "the last trial's speed rating less the dog's last-5 race mean",
    f"{PREFIX}grade_change": "today's grade number less the last race's (+: down in grade), same grade family",
    f"{PREFIX}dist_change": "metres more than the last race",
    f"{PREFIX}runs_at_grade": "races at today's grade before",
    f"{PREFIX}wins_at_grade": "wins at today's grade before",
    f"{PREFIX}runs_at_track": "races at today's track before",
    f"{PREFIX}runs_at_trackdist": "races at today's track and distance before",
    f"{PREFIX}runs_30d": "races in the 30 days before",
    f"{PREFIX}drop_streak": "consecutive races down in grade, today's included",
}
