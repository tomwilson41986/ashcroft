"""Neighbour externality, collision momentum and pressure-side laterality: what the dogs drawn beside a dog do to
its time (the owner, 11 Nov 2026: "identify NEW features that have not been discovered yet").

GBGB's comments name the victim of interference (Crd, Bmp, Blk) and never the culprit. Yet some dogs cost the dogs
drawn beside them lengths every time: they move off their line at the break, lean at the bend, check in front of a
rival. That is a property of the NEIGHBOUR, not of the pair (the dyad table, ``dy_``, knows only pairs that have
met) and not of the dog's own time (a rating says how fast it runs, not what it does to the next trap). The racing
manager redraws the traps every race, so each dog is seen beside many different dogs and its effect on them can be
read from THEIR times: a regularised plus-minus on the neighbours' residuals.

Construction, every sum over a dog's EARLIER days only (``dyad.earlier``):

- ``r`` = the run's lengths slower than its expected time before the race (``dyad.residuals`` on the expected-time
  table's ``xt_tot_time``; NaN for a trial, a fall, a card row, an outlier), race-centred: ``rc`` = r less the mean
  of r over the race's runners that have one (two or more needed). The night's going and the grade cancel.
- the traps 1 to 6 are the slots of a [races x 6] array; a dog's inside neighbour is the runner in trap - 1 and its
  outside neighbour the runner in trap + 1, when that trap is occupied (a row with a trap outside 1 to 6, or a
  second row in a trap, is read by nothing and gets no features).
- for each dog N: ``S_o``, ``n_o`` = the sum and count of ``rc`` of the dog drawn immediately OUTSIDE N; ``S_i``,
  ``n_i`` the same for the dog drawn immediately INSIDE N. Ratings shrunk by ``K`` neighbour-runs toward no
  effect: ``harm_out = S_o / (n_o + K)`` (lengths the dog on N's outside loses), ``harm_in = S_i / (n_i + K)``.
- today, for the row V: ``nx_in`` = harm_out of the dog in trap - 1 (what the dog on V's inside usually does to
  the dog outside it), ``nx_out`` = harm_in of the dog in trap + 1, ``nx_sum``, ``nx_n`` the evidence behind them;
  ``nx_own_out`` / ``nx_own_in`` V's own two ratings (the harm it does); ``nx_free_in`` / ``nx_free_out`` 1 when
  the trap beside it is vacant on the card (or there is none: trap 1, trap 6). A vacant trap costs nothing: 0.
- ``nx_victim``: V's own sensitivity to company, its shrunk mean ``rc`` on earlier days with both adjacent traps
  occupied less the same with at least one free (``K`` each; NaN until it has run both ways).
- collision momentum (a bump moves the lighter dog more): ``w`` = the weight at the dog's last earlier-day race
  (``MIN_KG`` to ``MAX_KG`` kg only, never today's weigh-in), ``c = exp(-|xt_sec V - xt_sec neighbour| / SCALE)``
  the chance the two reach the bend together, from the expected sectionals in lengths; ``nx_mass_in`` =
  log(w_V / w_inside) x c, ``nx_mass_out`` likewise (0 with the trap vacant, NaN with a weight or an expected
  sectional missing), ``nx_mass`` their sum, a neighbour with no earlier weight counted as 0 (NaN when V has no
  earlier weight or expected sectional, or no neighbour it has can be read). Above 0: V is the heavier party.
- pressure-side laterality (a rival in the left eye against one in the right): on V's earlier days, the shrunk
  mean ``rc`` of the runs where its OUTSIDE neighbour was expected ``PRESS`` lengths or more faster to the line
  (``xt_sec``, known before those races) less its shrunk mean ``rc`` over all its runs = ``nx_lat_out``; the same
  for the runs where its INSIDE neighbour was expected within ``PRESS`` lengths of it = ``nx_lat_in`` (NaN until
  it has had such a run); ``nx_press`` = (today's outside neighbour expected ``PRESS`` or more faster) x
  nx_lat_out + (today's inside neighbour within ``PRESS``) x nx_lat_in (a side with no record counted as 0).

The expected sectional exists only at a track and trip whose sectionals are timed (not the sprints: about a third
of the runs): there the mass and pressure columns are NaN, never a guess.

A run's residual is its own result: it enters only the sums of LATER days. Today's columns read the card (who is in
which trap), the expected sectionals (made before the race) and earlier days (tests/test_greyhound_ext.py: a middle
day's results rewritten row by row move nothing on that day or before).

Two limits of the construction, neither a read of the day's result (the review of 11 Nov 2026):

- the vacancy is read from the rows of the race. A result holds the field that ran (``data.clean`` drops a named dog
  with no result: it was not weighed in); a card holds the field as listed when it was read. A dog taken out after
  that is still a neighbour on the card and a vacant trap in the nightly build: ``nx_free_*`` and that side's
  ``nx_in`` / ``nx_out`` / ``nx_mass_*`` / ``nx_press`` then differ between the two.
- the race mean a neighbour's residual is centred on holds the rater's own residual, and a neighbour's rating
  holds the row's own earlier runs beside that neighbour: part of ``nx_in`` / ``nx_out`` / ``nx_own_*`` is the dogs'
  own earlier form against their expected times, not what one does to the other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core, dyad

PREFIX = "nx_"
SOURCE = ("GBGB results (calculated time, trap, weight: sources/gbgb.py) against the expected-time table "
          "(xt_tot_time, xt_sec)")
K = 5.0                      # neighbour-runs (or own runs) of prior toward no effect
SLOTS = 6                    # traps 1 to 6
MIN_KG, MAX_KG = 15.0, 50.0  # a weight outside is a keying error
SCALE = 1.0                  # lengths of expected sectional between two dogs at which a contact is 1/e as likely
PRESS = 0.5                  # lengths of expected sectional: the neighbour 'presses'


def last_weight(df: pd.DataFrame) -> np.ndarray:
    """The weight at each dog's last earlier-day race that has one (``df`` in time order): never the day's own."""
    w = pd.to_numeric(df.weight_kg, errors="coerce") if "weight_kg" in df else pd.Series(np.nan, index=df.index)
    w = w.where(w.between(MIN_KG, MAX_KG))
    held = w.groupby(df.dog_id.to_numpy()).ffill()                 # the last valid weight up to and with each run
    return core.prev_day_values(df[["dog_id", "t", "race_date"]].assign(_w=held), "dog_id", ["_w"])["_w"].to_numpy(float)


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = figs if figs is not None else core.per_run_figures(runs)
    df = df.sort_values(["t", "race_id", "trap"], kind="stable")
    xt = (ctx or {}).get("xtime")
    if xt is None:
        from greyhound.pit import xtime
        xt, _ = xtime.build(runs, df)
    j = df[["race_id", "dog_id"]].merge(xt[["race_id", "dog_id", "xt_tot_time", "xt_sec"]]
                                        .drop_duplicates(["race_id", "dog_id"]), on=["race_id", "dog_id"], how="left")
    n = len(df)
    r = dyad.residuals(df, pd.Series(j.xt_tot_time.to_numpy(), index=df.index)).to_numpy(float)
    xs = j.xt_sec.to_numpy(float)                                   # expected sectional, lengths, higher faster
    race = pd.factorize(df.race_id)[0]
    nr = int(race.max()) + 1 if n else 0
    dog = pd.factorize(df.dog_id)[0].astype(np.int64)
    dn = (core.day(df) - pd.Timestamp("2000-01-01")).dt.days.to_numpy()
    trap = pd.to_numeric(df.trap, errors="coerce").to_numpy(float)
    in_range = np.isfinite(trap) & (trap >= 1) & (trap <= SLOTS) & (trap == np.round(trap))
    slot = np.where(in_range, trap - 1, 0).astype(np.int64)
    cell = np.where(in_range, race * SLOTS + slot, -1 - np.arange(n))
    ok = in_range & ~pd.Series(cell).duplicated().to_numpy()        # one row a trap: a second is read by nothing
    # the residual against the race: the going, the grade and the night are common to the runners
    has_r = np.isfinite(r)
    cnt = np.bincount(race, weights=has_r, minlength=nr)
    tot = np.bincount(race, weights=np.where(has_r, r, 0.0), minlength=nr)
    with np.errstate(invalid="ignore", divide="ignore"):
        rcen = np.where(has_r & (cnt[race] >= 2), r - tot[race] / np.maximum(cnt[race], 1), np.nan)
    ro, so = race[ok], slot[ok] + 1                                 # a spare slot either side: nothing drawn there

    def beside(v: np.ndarray, fill: float = np.nan) -> tuple[np.ndarray, np.ndarray]:
        """Each row's inside (trap - 1) and outside (trap + 1) neighbour's value of ``v``; ``fill`` with none."""
        g = np.full((nr, SLOTS + 2), fill)
        g[ro, so] = v[ok]
        a, b = np.full(n, fill), np.full(n, fill)
        a[ok], b[ok] = g[ro, so - 1], g[ro, so + 1]
        return a, b

    occ_in, occ_out = (x > 0 for x in beside(np.ones(n), 0.0))
    rc_in, rc_out = beside(rcen)
    xs_in, xs_out = beside(xs)
    with np.errstate(invalid="ignore"):
        press_out = occ_out & ((xs_out - xs) >= PRESS)              # NaN compares False: no expected sectional, no press
        press_in = occ_in & (np.abs(xs_in - xs) <= PRESS)
    own = np.isfinite(rcen) & ok
    both = own & occ_in & occ_out
    free = own & ~(occ_in & occ_out)
    z = np.where(own, rcen, 0.0)
    f_out, f_in = np.isfinite(rc_out), np.isfinite(rc_in)
    acc = [np.where(f_out, rc_out, 0.0), f_out, np.where(f_in, rc_in, 0.0), f_in,          # the dogs beside N
           z, own, z * both, both, z * free, free,                                         # N itself, and in company
           z * press_out, own & press_out, z * press_in, own & press_in]                   # N pressed from each side
    S_o, n_o, S_i, n_i, S_a, n_a, S_b, n_b, S_f, n_f, S_po, n_po, S_pi, n_pi = dyad.earlier(dog, dn, acc)
    harm_out, harm_in = S_o / (n_o + K), S_i / (n_i + K)
    out = pd.DataFrame({"race_id": df.race_id.to_numpy(), "dog_id": df.dog_id.to_numpy()}, index=df.index)
    # today's neighbours: the inside one's effect on the dog outside it, the outside one's on the dog inside it
    nb_in, _ = beside(harm_out, 0.0)
    _, nb_out = beside(harm_in, 0.0)
    ev_in, _ = beside(n_o, 0.0)
    _, ev_out = beside(n_i, 0.0)
    out[f"{PREFIX}in"] = nb_in
    out[f"{PREFIX}out"] = nb_out
    out[f"{PREFIX}sum"] = nb_in + nb_out
    out[f"{PREFIX}n"] = ev_in + ev_out
    out[f"{PREFIX}own_out"] = harm_out
    out[f"{PREFIX}own_in"] = harm_in
    out[f"{PREFIX}free_in"] = (~occ_in).astype(float)
    out[f"{PREFIX}free_out"] = (~occ_out).astype(float)
    out[f"{PREFIX}victim"] = np.where((n_b > 0) & (n_f > 0), S_b / (n_b + K) - S_f / (n_f + K), np.nan)
    # collision momentum: log mass ratio against each neighbour, by the chance they reach the bend together
    w = last_weight(df)
    w_in, w_out = beside(w)
    with np.errstate(invalid="ignore", divide="ignore"):
        m_in = np.where(occ_in, np.log(w / w_in) * np.exp(-np.abs(xs - xs_in) / SCALE), 0.0)
        m_out = np.where(occ_out, np.log(w / w_out) * np.exp(-np.abs(xs - xs_out) / SCALE), 0.0)
    out[f"{PREFIX}mass_in"] = m_in
    out[f"{PREFIX}mass_out"] = m_out
    read = (occ_in & np.isfinite(m_in)) | (occ_out & np.isfinite(m_out)) | ~(occ_in | occ_out)
    out[f"{PREFIX}mass"] = np.where(np.isfinite(w) & np.isfinite(xs) & read,
                                    np.nan_to_num(m_in) + np.nan_to_num(m_out), np.nan)
    # laterality: the dog's own record when pressed from each side, against all its runs
    base = S_a / (n_a + K)
    lat_out = np.where(n_po > 0, S_po / (n_po + K) - base, np.nan)
    lat_in = np.where(n_pi > 0, S_pi / (n_pi + K) - base, np.nan)
    out[f"{PREFIX}lat_out"] = lat_out
    out[f"{PREFIX}lat_in"] = lat_in
    out[f"{PREFIX}press"] = np.where(np.isfinite(xs), press_out * np.nan_to_num(lat_out)
                                     + press_in * np.nan_to_num(lat_in), np.nan)
    cols = [c for c in out.columns if c.startswith(PREFIX)]
    out.loc[~ok, cols] = np.nan                                     # no trap 1 to 6: no neighbours to read
    out[cols] = out[cols].astype(np.float32)
    graded = ok & ~(df.is_trial.fillna(False).to_numpy(bool) if "is_trial" in df else np.zeros(n, bool))

    def share(mask: np.ndarray) -> float | None:
        return round(float(mask[graded].mean()), 4) if graded.any() else None

    met = graded & np.isfinite(rcen) & (out[f"{PREFIX}n"].to_numpy(float) >= 2 * K)
    notes = {"k_prior": K, "press_lengths": PRESS, "contact_scale_lengths": SCALE, "rows": int(n), "races": int(nr),
             "rows_without_a_trap_1_to_6": int((~in_range).sum()), "second_rows_in_a_trap": int((in_range & ~ok).sum()),
             "runs_with_a_race_centred_residual": int(np.isfinite(rcen).sum()),
             "neighbour_runs_read": int(f_out.sum() + f_in.sum()),
             "inside_trap_vacant_share": share(~occ_in & (slot > 0)),
             "outside_trap_vacant_share": share(~occ_out & (slot < SLOTS - 1)),
             "runs_with_a_rated_neighbour": share(out[f"{PREFIX}n"].to_numpy(float) > 0),
             "mean_neighbour_runs_behind_a_row": round(float(np.nanmean(out[f"{PREFIX}n"].to_numpy(float)[graded])), 2)
             if graded.any() else None,
             "runs_with_a_last_weight": share(np.isfinite(w)),
             "runs_with_an_expected_sectional": share(np.isfinite(xs)),
             "pressed_from_outside_share": share(press_out), "pressed_from_inside_share": share(press_in),
             # the idea's own check, out of sample by construction: does the neighbours' record tell the run's residual
             "rows_with_10_or_more_neighbour_runs_and_a_residual": int(met.sum()),
             "corr_nx_sum_with_the_runs_residual_on_those": round(float(np.corrcoef(
                 out[f"{PREFIX}sum"].to_numpy(float)[met], rcen[met])[0, 1]), 4) if met.sum() >= 50 else None}
    return out.sort_index().reset_index(drop=True), notes


COLUMNS = {
    f"{PREFIX}in": "lengths the dog drawn one trap inside has cost the dogs drawn immediately outside it on earlier "
                   f"days (their race-centred residuals' mean, shrunk by {K:g} neighbour-runs; above 0 it costs them; "
                   "0 with the trap vacant)",
    f"{PREFIX}out": "the same for the dog drawn one trap outside: what it has cost the dogs drawn immediately inside it",
    f"{PREFIX}sum": "the two together: lengths today's neighbours should cost the dog",
    f"{PREFIX}n": "the neighbour-runs behind them (the inside neighbour's runs with a dog outside it and the outside "
                  "neighbour's with a dog inside it)",
    f"{PREFIX}own_out": "the dog's own rating: lengths it has cost the dogs drawn immediately outside it (shrunk)",
    f"{PREFIX}own_in": "lengths it has cost the dogs drawn immediately inside it (shrunk)",
    f"{PREFIX}free_in": "1 when the trap immediately inside is vacant on the card, or the dog is in trap 1",
    f"{PREFIX}free_out": f"1 when the trap immediately outside is vacant on the card, or the dog is in trap {SLOTS}",
    f"{PREFIX}victim": "its sensitivity to company: its shrunk mean race-centred residual on earlier days with both "
                       f"adjacent traps occupied less the same with at least one free ({K:g} runs of prior each; above "
                       "0 it runs slower in company)",
    f"{PREFIX}mass_in": "collision momentum against the inside neighbour: log(its last earlier weight / the "
                        "neighbour's) x exp(-|difference of their expected sectionals in lengths|); above 0 it is the "
                        "heavier; 0 with the trap vacant",
    f"{PREFIX}mass_out": "the same against the outside neighbour",
    f"{PREFIX}mass": "the two together (a neighbour with no earlier weight counted as 0)",
    f"{PREFIX}lat_out": "pressed from the outside: its shrunk mean race-centred residual on the earlier runs where "
                        f"the outside neighbour was expected {PRESS:g} lengths or more faster to the line, less its "
                        "shrunk mean over all its runs (above 0 it runs slower when pressed there)",
    f"{PREFIX}lat_in": f"pressed from the inside: the same on the runs where the inside neighbour was expected within "
                       f"{PRESS:g} lengths of it",
    f"{PREFIX}press": "today's pressure by the dog's own record: nx_lat_out when today's outside neighbour is "
                      f"expected {PRESS:g} or more faster, plus nx_lat_in when today's inside neighbour is within "
                      f"{PRESS:g} (NaN where the track and trip has no expected sectional)",
}
