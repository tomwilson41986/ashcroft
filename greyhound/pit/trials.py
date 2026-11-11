"""Trial form: what a dog did in its trials before today (the audit of 11 Nov 2026, findings 1 to 3).

GBGB serves every trial (classes T, T1 to T4) with an empty off time, and ``greyhound/data.py clean()`` drops a row
with no time, so no trial has ever reached the model: 72,816 trial rows in the 16 months to 5 Nov 2026 alone, and
1,985 of the 1,992 dogs having their first race from 1 Jun 2026 had a timed trial before it. The trial is the form a
handicapper reads for a debutant, a dog back from a break and a dog changing track, and it is printed on the card.

This table brings the trials in on their own channel: the engine and the other tables are untouched (a trial's solo
"win" would flood every win rate), and the trials are read here only as times, dates, weights and running lines.

- **The trial rows** come from the raw GBGB frame before ``clean`` (``select``; ``greyhound/track.py runs_through``
  hands them over through ``trials_out``; ``ctx["trial_runs"]``), and from any T-class row already in ``runs``.
- **A trial's figures**: its calculated time and its sectional against the track and trip's standard on that day
  (the median winning calculated time, and the median sectional of every runner, of GRADED races over the 365 days
  before: the figs table's standard), in lengths, higher faster. The same figure is made for every race run, so a
  trial and a race are on one scale.
- **Before today** (earlier days only): how many trials, how long ago, whether the last outing was a trial, the races
  since; the last trial's figures, the best and the mean of the last three; whether it was at today's track and trip;
  how many ran in it; its weight against the last race weight; its running line and break; and the dog's level on
  every outing, trials and races together (decayed by ``HALF_LIFE`` days), against the same from races alone and
  against the field.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core, figs as figs_mod

PREFIX = "tl_"
SOURCE = "GBGB results, the trial rows (classes T, T1-T4: sources/gbgb.py) that data.clean drops, and the race rows"
TRIAL_RE = r"^T\d*$"
HALF_LIFE = 60.0
OUTLIER = 40.0
BIG = 1_000_000                       # day numbers are below this: a dog's block of the sorted (dog, day) key
KEEP = ["race_date", "race_time", "track", "distance_m", "race_id", "race_class", "trap", "dog_id", "dog_name",
        "sire", "dam", "born", "sex", "season", "trainer", "position", "sectional", "run_time", "adjusted_time",
        "weight_kg", "comment"]


def is_trial_class(race_class: pd.Series) -> pd.Series:
    return race_class.astype(str).str.strip().str.upper().str.match(TRIAL_RE)


def select(raw: pd.DataFrame) -> pd.DataFrame:
    """The trial rows of a GBGB frame: a T class, a dog, a finishing position of 1 or more and a clock."""
    if raw is None or not len(raw):
        return pd.DataFrame(columns=KEEP)
    num = lambda c: pd.to_numeric(raw[c], errors="coerce") if c in raw else pd.Series(np.nan, index=raw.index)  # noqa: E731
    ok = is_trial_class(raw.race_class) & raw.dog_id.notna() & (num("position") >= 1) & (num("run_time") > 0)
    t = raw.loc[ok, [c for c in KEEP if c in raw]].copy()
    t["dog_id"] = t.dog_id.astype("int64")
    return t.drop_duplicates(["race_id", "dog_id"], keep="last").reset_index(drop=True)


def _dn(dates: pd.Series) -> np.ndarray:
    return (pd.to_datetime(dates, errors="coerce").dt.normalize() - pd.Timestamp("2000-01-01")).dt.days.to_numpy()


def _key(df: pd.DataFrame) -> pd.Series:
    return figs_mod._key(df)


def standards(df: pd.DataFrame) -> pd.DataFrame:
    """Each race row's standard time and sectional at its track and trip (graded races of the 365 days before)."""
    bad = df.comment.fillna("").str.contains(r"\(Handslip\)|Fell|Fll|BrkLine", case=False) if "comment" in df else \
        pd.Series(False, index=df.index)
    trial = df.is_trial.fillna(False).astype(bool) if "is_trial" in df else pd.Series(False, index=df.index)
    fair = ~bad & ~trial
    t = pd.to_numeric(df.adjusted_time, errors="coerce")
    s = pd.to_numeric(df.sectional, errors="coerce")
    won = pd.to_numeric(df.won, errors="coerce") == 1
    return pd.DataFrame({"std_t": figs_mod.rolling_median(df, t.where(won & fair)),
                         "std_s": figs_mod.rolling_median(df, s.where(fair))}, index=df.index)


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    from greyhound.metrics import parse_comment
    df = figs if figs is not None else core.per_run_figures(runs)
    df = df.sort_values(["t", "race_id", "trap"], kind="stable")
    L = core.LENGTH_S
    in_runs = is_trial_class(df.race_class)
    tr = [select(df[in_runs])] if in_runs.any() else []
    extra = (ctx or {}).get("trial_runs")
    if extra is not None and len(extra):
        tr.append(select(extra))
    tr = pd.concat(tr, ignore_index=True).drop_duplicates(["race_id", "dog_id"], keep="last") if tr else \
        pd.DataFrame(columns=KEEP)
    races = df[~in_runs]
    std = standards(races)
    key_r, dn_r = _key(races), _dn(races.race_date)
    out = pd.DataFrame({"race_id": df.race_id.to_numpy(), "dog_id": df.dog_id.to_numpy()}, index=df.index)
    cols = list(COLUMNS)
    for c in cols:
        out[c] = np.nan
    notes = {"trial_rows": int(len(tr)), "trial_rows_in_runs": int(in_runs.sum()),
             "from": str(tr.race_date.min()) if len(tr) else None, "to": str(tr.race_date.max()) if len(tr) else None}
    if not len(tr):
        out[[f"{PREFIX}n_all", f"{PREFIX}n28", f"{PREFIX}n90"]] = 0.0
        out[cols] = out[cols].astype(np.float32)
        return out.sort_index().reset_index(drop=True), notes
    # each trial's standard: the track and trip's on the last race day at or before the trial (made of earlier days)
    look = pd.DataFrame({"key": key_r.to_numpy(), "dn": dn_r, "std_t": std.std_t.to_numpy(),
                         "std_s": std.std_s.to_numpy()}).dropna(subset=["dn"]).drop_duplicates(["key", "dn"]).sort_values("dn")
    tr = tr.assign(key=_key(tr).to_numpy(), dn=_dn(tr.race_date)).dropna(subset=["dn"]).sort_values("dn")
    tr["dn"] = tr.dn.astype(np.int64)
    look["dn"] = look.dn.astype(np.int64)
    tr = pd.merge_asof(tr, look, on="dn", by="key", direction="backward")
    tt, ts = pd.to_numeric(tr.adjusted_time, errors="coerce"), pd.to_numeric(tr.sectional, errors="coerce")
    bad_t = tr.comment.fillna("").str.contains(r"\(Handslip\)|Fell|Fll|BrkLine", case=False)
    f = ((tr.std_t - tt) / L).where(~bad_t)
    s = ((tr.std_s - ts) / L).where(~bad_t)
    tr["fig"], tr["sec"] = f.where(f.abs() <= OUTLIER), s.where(s.abs() <= OUTLIER)
    tr["solo"] = tr.groupby("race_id").dog_id.transform("size").astype(float)
    uniq = {c: parse_comment(c) for c in pd.unique(tr.comment.fillna(""))}
    tr["path"] = [uniq[c].get("path", np.nan) for c in tr.comment.fillna("")]
    tr["brk"] = [uniq[c].get("brk", np.nan) for c in tr.comment.fillna("")]
    tr["wt"] = pd.to_numeric(tr.weight_kg, errors="coerce").where(lambda w: (w >= 15) & (w <= 50))
    # one row a dog-day of trials (a dog trialled twice a day: the day's mean), in (dog, day) order
    dogs = pd.Index(pd.unique(np.concatenate([df.dog_id.to_numpy(), tr.dog_id.to_numpy()])))
    tr["dc"] = dogs.get_indexer(tr.dog_id)
    per = tr.groupby(["dc", "dn"], as_index=False).agg(fig=("fig", "mean"), sec=("sec", "mean"), solo=("solo", "min"),
                                                       path=("path", "mean"), brk=("brk", "mean"), wt=("wt", "mean"),
                                                       track=("track", "last"), distance_m=("distance_m", "last"))
    per = per.sort_values(["dc", "dn"]).reset_index(drop=True)
    g = per.groupby("dc")
    per["best3"] = g.fig.rolling(3, min_periods=1).max().reset_index(level=0, drop=True)
    per["mean3"] = g.fig.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    per["sbest3"] = g.sec.rolling(3, min_periods=1).max().reset_index(level=0, drop=True)
    tkey = per.dc.to_numpy(np.int64) * BIG + per.dn.to_numpy(np.int64)
    dc = dogs.get_indexer(df.dog_id).astype(np.int64)
    dn = _dn(df.race_date).astype(np.int64)
    block = np.searchsorted(tkey, dc * BIG)

    def before(day: np.ndarray) -> np.ndarray:
        return np.searchsorted(tkey, dc * BIG + day) - block          # the dog's trial days before ``day``

    n_all = before(dn)
    last = block + n_all - 1                                           # the last trial day before today
    has = n_all > 0
    li = np.where(has, last, 0)

    def take(col: str) -> np.ndarray:
        return np.where(has, per[col].to_numpy(float)[li], np.nan)

    days = np.where(has, dn - per.dn.to_numpy()[li], np.nan)
    out[f"{PREFIX}n_all"] = n_all.astype(float)
    out[f"{PREFIX}n28"] = (n_all - before(dn - 28)).astype(float)
    out[f"{PREFIX}n90"] = (n_all - before(dn - 90)).astype(float)
    out[f"{PREFIX}days"] = days
    # the races: each dog's race days, to count the races since the trial and the days since the last race
    rk = np.unique(dogs.get_indexer(races.dog_id).astype(np.int64) * BIG + dn_r.astype(np.int64))
    rblock = np.searchsorted(rk, dc * BIG)
    r_before = np.searchsorted(rk, dc * BIG + dn) - rblock
    last_race_dn = np.where(r_before > 0, rk[np.where(r_before > 0, rblock + r_before - 1, 0)] % BIG, -1)
    race_days = np.where(r_before > 0, dn - last_race_dn, np.nan)
    out[f"{PREFIX}out_days"] = np.fmin(days, race_days)
    out[f"{PREFIX}last_was_trial"] = np.where(has, (np.nan_to_num(race_days, nan=1e9) > days).astype(float), 0.0)
    after_trial = np.searchsorted(rk, dc * BIG + np.where(has, per.dn.to_numpy()[li] + 1, 0)) - rblock
    out[f"{PREFIX}races_since"] = np.where(has, r_before - after_trial, np.nan)
    out[f"{PREFIX}fig_last"], out[f"{PREFIX}fig_best3"], out[f"{PREFIX}fig_mean3"] = take("fig"), take("best3"), take("mean3")
    out[f"{PREFIX}sec_last"], out[f"{PREFIX}sec_best3"] = take("sec"), take("sbest3")
    out[f"{PREFIX}solo_last"] = take("solo")
    same_track = np.where(has, per.track.astype(str).to_numpy()[li] == df.track.astype(str).to_numpy(), False)
    same_trip = np.where(has, np.isclose(pd.to_numeric(per.distance_m, errors="coerce").to_numpy(float)[li],
                                         pd.to_numeric(df.distance_m, errors="coerce").to_numpy(float)), False)
    out[f"{PREFIX}same_track"] = np.where(has, same_track.astype(float), np.nan)
    out[f"{PREFIX}same_trip"] = np.where(has, (same_track & same_trip).astype(float), np.nan)
    out[f"{PREFIX}fig_here"] = np.where(same_track & same_trip, take("fig"), np.nan)
    out[f"{PREFIX}wt_last"] = take("wt")
    w_race = pd.to_numeric(races.weight_kg, errors="coerce").where(lambda w: (w >= 15) & (w <= 50))
    prev_w = core.prev_day_values(races.assign(_w=w_race), "dog_id", ["_w"])._w.reindex(df.index).to_numpy(float)
    out[f"{PREFIX}wt_vs_race"] = take("wt") - prev_w
    out[f"{PREFIX}path_last"], out[f"{PREFIX}brk_last"] = take("path"), take("brk")
    # the level on every outing, trials and races together, against the races alone
    fr = ((std.std_t - pd.to_numeric(races.adjusted_time, errors="coerce")) / L)
    fr = fr.where(fr.abs() <= OUTLIER)
    both = pd.concat([pd.DataFrame({"dog_id": races.dog_id.to_numpy(), "race_date": races.race_date.to_numpy(),
                                    "fig": fr.to_numpy(), "row": np.arange(len(races))}),
                      pd.DataFrame({"dog_id": tr.dog_id.to_numpy(), "race_date": tr.race_date.to_numpy(),
                                    "fig": tr.fig.to_numpy(), "row": -1})], ignore_index=True)
    one = pd.Series(1.0, index=both.index)
    d_all = figs_mod.decayed(both, both.fig, one, HALF_LIFE)
    d_race = figs_mod.decayed(both, both.fig.where(both.row >= 0), one, HALF_LIFE)
    is_race = (both.row >= 0).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        lvl = np.where(d_all.sw.to_numpy()[is_race] > 0, d_all.swx.to_numpy()[is_race] / d_all.sw.to_numpy()[is_race], np.nan)
        lvl_r = np.where(d_race.sw.to_numpy()[is_race] > 0,
                         d_race.swx.to_numpy()[is_race] / d_race.sw.to_numpy()[is_race], np.nan)
    lv = pd.Series(lvl, index=races.index).reindex(df.index)
    lr = pd.Series(lvl_r, index=races.index).reindex(df.index)
    out[f"{PREFIX}lvl"] = lv.to_numpy()
    out[f"{PREFIX}lvl_n"] = pd.Series(d_all.sw.to_numpy()[is_race], index=races.index).reindex(df.index).to_numpy()
    out[f"{PREFIX}lvl_gain"] = np.where(has, (lv - lr).to_numpy(), np.nan)
    gr = lv.groupby(df.race_id)
    out[f"{PREFIX}lvl_vs_field"] = (lv - (gr.transform("sum") - lv.fillna(0.0)) /
                                    (gr.transform("count") - lv.notna()).replace(0, np.nan)).to_numpy()
    out[cols] = out[cols].astype(np.float32)
    graded = ~in_runs.to_numpy() & ~(df.is_trial.fillna(False).to_numpy(bool) if "is_trial" in df else False)
    first_race = graded & (r_before == 0)
    notes.update({"trial_dog_days": int(len(per)), "dogs_with_a_trial": int(per.dc.nunique()),
                  "trial_figures": int(tr.fig.notna().sum()),
                  "race_rows_with_a_trial_before": round(float(has[graded].mean()), 4) if graded.any() else None,
                  "race_rows_last_outing_a_trial": round(float(np.nanmean(out[f"{PREFIX}last_was_trial"].to_numpy()[graded])), 4)
                  if graded.any() else None,
                  "race_rows_a_trial_in_28_days": round(float((out[f"{PREFIX}n28"].to_numpy()[graded] > 0).mean()), 4)
                  if graded.any() else None,
                  "first_races": int(first_race.sum()),
                  "first_races_with_a_trial": round(float(has[first_race].mean()), 4) if first_race.any() else None,
                  "first_races_with_a_trial_figure": round(float(np.isfinite(out[f"{PREFIX}fig_last"].to_numpy()[first_race]).mean()), 4)
                  if first_race.any() else None})
    return out.sort_index().reset_index(drop=True), notes


COLUMNS = {
    f"{PREFIX}n_all": "trials on the dog's record before today",
    f"{PREFIX}n28": "trials in the 28 days before today",
    f"{PREFIX}n90": "trials in the 90 days before today",
    f"{PREFIX}days": "days since its last trial",
    f"{PREFIX}out_days": "days since its last outing of any kind, trial or race",
    f"{PREFIX}last_was_trial": "1 when the last outing before today was a trial",
    f"{PREFIX}races_since": "races run since the last trial",
    f"{PREFIX}fig_last": "the last trial's calculated time against the track and trip's graded standard (lengths, "
                         "higher faster)",
    f"{PREFIX}fig_best3": "the best of the last three trial days' figures",
    f"{PREFIX}fig_mean3": "the mean of the last three",
    f"{PREFIX}sec_last": "the last trial's sectional against the graded standard sectional (lengths)",
    f"{PREFIX}sec_best3": "the best of the last three",
    f"{PREFIX}solo_last": "how many dogs ran in the last trial (1: alone)",
    f"{PREFIX}same_track": "the last trial was at today's track",
    f"{PREFIX}same_trip": "the last trial was at today's track and trip",
    f"{PREFIX}fig_here": "the last trial's figure when it was at today's track and trip",
    f"{PREFIX}wt_last": "the weight at the last trial (kg)",
    f"{PREFIX}wt_vs_race": "that weight less the weight at the last race before today",
    f"{PREFIX}path_last": "the running line in the last trial (1 rails, 2 middle, 3 wide)",
    f"{PREFIX}brk_last": "the break in the last trial (the engine's break score)",
    f"{PREFIX}lvl": f"the dog's level on every outing before today, trials and races on one scale, decayed by "
                    f"{int(HALF_LIFE)} days (lengths against the graded standard)",
    f"{PREFIX}lvl_n": "the decayed weight of outings behind it",
    f"{PREFIX}lvl_gain": "that level less the same from races alone (what the trials add; none for a dog with no "
                         "race yet: its level is the trials')",
    f"{PREFIX}lvl_vs_field": "the level against the mean of the race's other runners",
}
