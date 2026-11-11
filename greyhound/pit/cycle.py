"""A bitch's season cycle, read from her own absences (the owner, 11 Nov 2026: "identify NEW features").

A greyhound bitch comes into season every several months; she may not race for at least 21 days, is usually rested
8 to 10 weeks, and form students hold that she goes off before it and runs her best 14 to 16 weeks after
(reports/greyhound_model.md section 41.2). GBGB serves only her CURRENT season date on every past result, so the date
cannot be a feature of history (greyhound/pit/life.py). But the season leaves a mark that IS known at the time: a
gap in her record.

- **An inferred season**: an absence of ``MIN_GAP`` to ``MAX_GAP`` days between two days she ran (trials count as
  running). Its start is taken as the last day she ran before it.
- **Her own cycle**: the mean interval between her inferred seasons (intervals of ``MIN_CYCLE`` to ``MAX_CYCLE``
  days), shrunk by ``K0`` intervals toward the population's median interval, fitted on the seasons before
  ``FIT_BEFORE``.
- **Where she is today**: days since her last inferred season began, that as a fraction of her cycle (the phase),
  the days to her next expected season (below 0: overdue; a pre-season dip is the hypothesis), the days and runs
  since she came back, the length of the absence, how many seasons are on her record, and, for a bitch with none
  yet, her age in months (the first season is still ahead).

The same columns are made for dogs (``oc_d_``) as the control: a dog's long absence is an injury or a rest, not a
cycle, so whatever the bitches' columns carry beyond the dogs' is the season. Everything is read from the days she
ran before today and today's date; the GBGB season field is used only in the notes, to check the inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core, form, life

PREFIX = "oc_"
SOURCE = "GBGB results (the days each dog ran, sex, born: sources/gbgb.py); the season field for the check only"
MIN_GAP, MAX_GAP = 49, 200
MIN_CYCLE, MAX_CYCLE = 120, 500
K0 = 1.0
C0_DEFAULT = 240.0
FIT_BEFORE = "2018-01-01"
MIN_FIT = 200


def absences(dog: np.ndarray, dn: np.ndarray) -> pd.DataFrame:
    """One row a dog-day, in order: the inferred seasons so far and where today sits after the last."""
    per = pd.DataFrame({"dog": dog, "dn": dn}).drop_duplicates().sort_values(["dog", "dn"]).reset_index(drop=True)
    g = per.groupby("dog")
    prev = g.dn.shift()
    gap = per.dn - prev
    absent = gap.between(MIN_GAP, MAX_GAP)
    per["absent"] = absent
    per["start"] = prev.where(absent)                               # the last day run before the absence
    per["n"] = absent.groupby(per.dog).cumsum()
    per["last_start"] = per.start.groupby(per.dog).ffill()
    per["back_dn"] = per.dn.where(absent).groupby(per.dog).ffill()  # the day she came back
    per["gap"] = gap.where(absent).groupby(per.dog).ffill()
    per["runs_back"] = per.groupby(["dog", "n"]).cumcount().where(per.n > 0)
    a = per[absent]
    interval = a.start - a.groupby("dog").start.shift()
    ok = interval.between(MIN_CYCLE, MAX_CYCLE)
    per["interval"] = interval.where(ok).reindex(per.index)
    per["sum_i"] = per.interval.fillna(0.0).groupby(per.dog).cumsum()
    per["n_i"] = per.interval.notna().groupby(per.dog).cumsum()
    return per


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = figs if figs is not None else core.per_run_figures(runs)
    day = core.day(df)
    dn = (day - pd.Timestamp("2000-01-01")).dt.days.to_numpy()
    per = absences(df.dog_id.to_numpy(), dn)
    bitch_row = form._bitch(df).to_numpy() >= 0.5
    is_bitch = pd.Series(bitch_row).groupby(df.dog_id.to_numpy()).max()
    per_b = per.dog.map(is_bitch).fillna(False).to_numpy(bool)
    cut = (pd.Timestamp(FIT_BEFORE) - pd.Timestamp("2000-01-01")).days
    early = per.interval[per_b & (per.dn < cut)].dropna()
    c0 = float(early.median()) if len(early) >= MIN_FIT else C0_DEFAULT
    cycle = (per.sum_i + K0 * c0) / (per.n_i + K0)
    since = per.dn - per.last_start
    per["days"] = since
    per["cycle"] = cycle.where(per.n > 0)
    per["phase"] = since / cycle
    per["due"] = cycle - since
    per["back_days"] = per.dn - per.back_dn
    j = pd.DataFrame({"dog": df.dog_id.to_numpy(), "dn": dn}).merge(per, on=["dog", "dn"], how="left")
    age = form._age_months(df).to_numpy(float)
    out = pd.DataFrame({"race_id": df.race_id.to_numpy(), "dog_id": df.dog_id.to_numpy()}, index=df.index)
    for tag, mask in (("b", bitch_row), ("d", ~bitch_row)):
        for name in ("days", "back_days", "runs_back", "gap", "n") + (("cycle", "phase", "due") if tag == "b" else ()):
            out[f"{PREFIX}{tag}_{name}"] = np.where(mask, j[name].to_numpy(float), np.nan)
    out[f"{PREFIX}b_none_age"] = np.where(bitch_row & (j.n.to_numpy(float) == 0), age, np.nan)
    cols = [c for c in out.columns if c.startswith(PREFIX)]
    out[cols] = out[cols].astype(np.float32)
    notes = {"min_gap": MIN_GAP, "max_gap": MAX_GAP, "population_cycle_days": round(c0, 1),
             "early_intervals": int(len(early)),
             "cycle_quartiles_all_bitches": [round(float(q), 1) for q in per.interval[per_b].dropna().quantile(
                 [0.25, 0.5, 0.75])] if per.interval[per_b].notna().any() else None,
             "bitches": int(is_bitch.sum()), "bitches_with_a_season": int(per[per_b & (per.n > 0)].dog.nunique()),
             "absences_a_bitch_year": None, "absences_a_dog_year": None}
    for tag, m in (("bitch", per_b), ("dog", ~per_b)):
        span = per[m].groupby("dog").dn.agg(lambda s: s.max() - s.min()).sum() / 365.25
        notes[f"absences_a_{tag}_year"] = round(float(per[m].absent.sum() / span), 3) if span > 0 else None
    # the check: GBGB's current season date against the nearest inferred season (not a feature)
    try:
        if "season" in df:
            uniq = {s: life.season_date(s) for s in pd.unique(df.season.dropna())}
            sd = pd.to_datetime(df.season.map(uniq), errors="coerce")
            known = pd.DataFrame({"dog": df.dog_id.to_numpy(), "sd": sd.to_numpy()})[bitch_row].dropna().drop_duplicates("dog")
            known["sdn"] = (known.sd - pd.Timestamp("2000-01-01")).dt.days
            a = per[per.absent][["dog", "start", "dn"]].merge(known[["dog", "sdn"]], on="dog")
            a["off"] = a.sdn - a.start                               # the dated season, days after her last run
            near = a.loc[a.assign(k=a.off.abs()).groupby("dog").k.idxmin()]
            hit = near[near.off.between(-30, 90)]
            notes["check"] = {"bitches_with_a_dated_season": int(len(known)),
                              "with_an_inferred_season_within_-30_to_90_days": int(len(hit)),
                              "days_from_last_run_to_the_dated_season_quartiles":
                                  [round(float(q), 1) for q in hit.off.quantile([0.25, 0.5, 0.75])] if len(hit) else None,
                              "dated_inside_the_absence_share": round(float(((hit.off >= 0) & (hit.sdn <= hit.dn)).mean()), 3)
                              if len(hit) else None}
    except Exception as exc:                                         # the check never stops the table
        notes["check"] = {"error": str(exc)[:200]}
    return out.reset_index(drop=True), notes


COLUMNS = {
    **{f"{PREFIX}{t}_days": f"days since the {who}'s last absence of {MIN_GAP} to {MAX_GAP} days began ({what})"
       for t, who, what in (("b", "bitch", "her last inferred season"), ("d", "dog", "the control: a rest or an injury"))},
    **{f"{PREFIX}{t}_back_days": "days since the first run back from it" for t in "bd"},
    **{f"{PREFIX}{t}_runs_back": "days run since coming back (0: today is the first)" for t in "bd"},
    **{f"{PREFIX}{t}_gap": "the length of that absence in days" for t in "bd"},
    **{f"{PREFIX}{t}_n": "such absences on the record so far" for t in "bd"},
    f"{PREFIX}b_cycle": f"her own cycle in days: the mean interval between her inferred seasons, shrunk by {K0:g} "
                        "interval toward the population's median",
    f"{PREFIX}b_phase": "days since her last inferred season as a fraction of her cycle",
    f"{PREFIX}b_due": "days to her next expected season (below 0: overdue)",
    f"{PREFIX}b_none_age": "her age in months when no season is on her record yet (the first is still ahead)",
}
