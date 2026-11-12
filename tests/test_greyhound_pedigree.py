"""greyhound/pit/pedigree.py: the sire -> dam -> litter -> dog posteriors. A fast sire's progeny get a high sire
posterior before they have run, and the dog's own runs are not in its family's; a first-crop sire's posterior is
the prior (mean 0, the prior's sd); the sd narrows as the progeny run; a sire whose progeny stay gets the trip-band
posterior and one whose young dogs improve the improvement posterior; the day's own results move nothing (the last
day and a middle day), nor do later days; a card row gets what its run had; one row a race and dog, in the order of
the runs. Planted data, used in these tests only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.data import clean
from greyhound.pit import pedigree
from tests.test_greyhound_pit import cardify_last_day, perturb_last_day

COLS = list(pedigree.COLUMNS)
N_SIRES, N_DOGS, DAYS = 8, 360, 170
SIRE_EFFECT = 0.12            # seconds between neighbouring sires (sire 0 the fastest)
STAY_SIRE = 3                 # this sire's progeny are faster over the long trip, slower over the short
IMP_SIRE = 5                  # this sire's young dogs improve with racing
NEW_SIRE = 7                  # this sire's progeny first run in the last fortnight


def synth(days=DAYS, seed=7):
    """Two tracks, a short and a long trip at each; 360 dogs by 8 sires out of 90 dams, litters of 4 (a dam is always
    mated to one sire). Dogs 0-179 run from the first day (born Jan-2023); dogs 180-359 are young (born Mar-2024)
    and debut from day 60; sire 7's dogs run only in the last fortnight."""
    rng = np.random.default_rng(seed)
    dogs = np.arange(N_DOGS)
    lit = dogs // 4
    dam = lit                                             # one litter a dam
    sire = dam % N_SIRES
    s_eff = (np.arange(N_SIRES) - 3.5) * SIRE_EFFECT      # seconds; above 0 slower
    s_eff[NEW_SIRE] = 0.0
    d_eff = rng.normal(0, 0.10, lit.max() + 1)
    own = rng.normal(0, 0.10, N_DOGS)
    base = s_eff[sire] + d_eff[dam] + own
    young = dogs >= 180
    nrun = np.zeros(N_DOGS, int)
    rows, rid = [], 7000
    for d in range(days):
        day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        avail = dogs[(~young | (d >= 60)) & ((sire != NEW_SIRE) | (d >= days - 14))]
        pick = rng.permutation(avail)[:72]
        for r in range(12):
            field = pick[6 * r:6 * r + 6]
            rid += 1
            track = "Hove" if r % 2 else "Romford"
            dist = 500.0 if r % 4 < 2 else 280.0
            stay = np.where(sire[field] == STAY_SIRE, -0.25 if dist == 500.0 else 0.25, 0.0)
            imp = np.where((sire[field] == IMP_SIRE) & young[field] & (nrun[field] >= 3), -0.30, 0.0)
            t = (29.5 if dist == 500.0 else 16.5) + base[field] + stay + imp + rng.normal(0, 0.08, 6)
            nrun[field] += 1
            order = np.argsort(t)
            pos = np.argsort(order) + 1
            for k, dog in enumerate(field):
                rows.append({"race_date": day, "race_time": f"{12 + r // 2}:{30 * (r % 2):02d}", "meeting_id": d * 10 + r % 2,
                             "track": track, "race_id": rid, "race_class": "A3", "distance_m": dist, "trap": k + 1,
                             "dog_id": int(dog) + 1, "dog_name": f"Dog {dog}", "sire": f"Sire {sire[dog]}",
                             "dam": f"Dam {dam[dog]}", "born": "Mar-2024" if young[dog] else "Jan-2023", "sex": "d",
                             "sp_decimal": 6.0, "position": int(pos[k]),
                             "beaten_lengths": float(round((t[k] - t.min()) / 0.08, 2)), "sectional": 4.5,
                             "run_time": float(round(t[k], 2)), "adjusted_time": float(round(t[k], 2)),
                             "weight_kg": 30.0, "comment": "QAw"})
    return clean(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def built():
    runs = synth()
    tab, notes = pedigree.build(runs)
    return runs, runs.merge(tab, on=["race_id", "dog_id"]), tab, notes


def _same(a, b, cols=COLS):
    a, b = a[cols].to_numpy(float), b[cols].to_numpy(float)
    assert (np.isnan(a) == np.isnan(b)).all()
    np.testing.assert_allclose(np.nan_to_num(a), np.nan_to_num(b), atol=1e-5)


def test_every_column_is_registered_and_keys_are_unique(built):
    runs, m, tab, notes = built
    assert list(tab.columns) == ["race_id", "dog_id"] + COLS and all(c.startswith(pedigree.PREFIX) for c in COLS)
    assert (tab[COLS].dtypes == np.float32).all()
    assert len(tab) == len(runs) == len(m) and not tab.duplicated(["race_id", "dog_id"]).any()
    assert (tab.race_id.to_numpy() == runs.race_id.to_numpy()).all() and (tab.dog_id.to_numpy() == runs.dog_id.to_numpy()).all()
    assert notes["rows"] == len(runs) and notes["months_with_a_fit"] == notes["months"] - 1
    first_month = m.race_date < "2025-02-01"                          # nothing before it to fit on
    assert m.loc[first_month, COLS].isna().all().all()
    late = m[m.race_date >= "2025-03-01"]
    assert late[[c for c in COLS if c not in ("pg_imp_fam_mu", "pg_imp_fam_sd")]].notna().mean().min() > 0.95
    assert m[m.race_date >= "2025-06-01"].pg_imp_fam_mu.notna().all()


def test_the_table_is_in_the_order_of_the_runs_however_they_are_sorted(built):
    runs, m, tab, _ = built
    sh = runs.sample(frac=1.0, random_state=3)
    t2, _ = pedigree.build(sh)
    assert (t2.race_id.to_numpy() == sh.race_id.to_numpy()).all() and (t2.dog_id.to_numpy() == sh.dog_id.to_numpy()).all()
    j = tab.merge(t2, on=["race_id", "dog_id"], suffixes=("", "_b"))
    _same(j, j[[c + "_b" for c in COLS]].set_axis(COLS, axis=1))


def test_a_planted_sire_effect_is_recovered_before_the_dog_has_run(built):
    runs, m, tab, _ = built
    m = m.assign(s=m.sire.str[5:].astype(int), nth=m.groupby("dog_id").cumcount())
    debut = m[(m.nth == 0) & (m.race_date >= "2025-03-02") & (m.s != NEW_SIRE)]      # the young dogs' first runs
    by = debut.groupby("s").pg_sire_mu.mean()
    assert len(debut) > 100 and list(by.sort_values(ascending=False).index) == list(range(N_SIRES - 1))
    assert by[0] > 0.5 and by[6] < -0.5                               # the fastest sire's progeny, the slowest's
    assert (debut.pg_ped_weight > 0.999).all()                        # no run of its own: the family is all there is
    assert debut.pg_post_shift.isna().all()
    np.testing.assert_allclose(debut.pg_fam_pull, debut.pg_fam_mu, atol=1e-5)
    # the family prior says which debutant is faster: against the time it then ran, within the race's trip
    z = debut.adjusted_time - debut.groupby("distance_m").adjusted_time.transform("mean")
    assert np.corrcoef(debut.pg_fam_mu, -z)[0, 1] > 0.5
    # an exposed dog leans on its own runs
    old = m[(m.nth >= 30)]
    assert old.pg_ped_weight.max() < 0.5 and old.pg_post_shift.notna().all()


def test_a_first_crop_sire_has_the_prior_and_the_sd_narrows_with_progeny(built):
    runs, m, tab, notes = built
    m = m.assign(s=m.sire.str[5:].astype(int))
    new = m[m.s == NEW_SIRE]
    first_day = new[new.race_date == new.race_date.min()]
    h = notes["last_fit"]["spd"]                                      # the last month's fit is that day's month's
    assert notes["last_fit"]["month"] == first_day.race_date.iloc[0][:7]
    np.testing.assert_allclose(first_day.pg_sire_mu, 0.0, atol=1e-7)
    np.testing.assert_allclose(first_day.pg_sire_sd, np.sqrt(h["ts"]), rtol=1e-5)
    later = new[new.race_date == new.race_date.max()]
    assert later.pg_sire_sd.max() < first_day.pg_sire_sd.min()
    old = m[(m.s == 0) & (m.race_date == m.race_date.max())]
    assert old.pg_sire_sd.max() < later.pg_sire_sd.min()             # 150 days of progeny against a fortnight


def test_the_dogs_own_runs_are_not_in_its_family(built):
    """A dog's family posteriors do not move when its own earlier results are rewritten (its own mean does)."""
    runs, m, tab, _ = built
    dog = int(runs[runs.race_date < "2025-05-01"].dog_id.value_counts().index[0])
    r = runs.copy()
    mine = (r.dog_id == dog) & (r.race_date >= "2025-05-01") & (r.race_date < "2025-06-10")   # within two months
    assert mine.sum() >= 3
    r.loc[mine, "adjusted_time"] -= 1.0
    t2, _ = pedigree.build(r)
    probe = ((runs.dog_id == dog) & (runs.race_date >= "2025-06-10") & (runs.race_date < "2025-07-01")).to_numpy()
    assert probe.sum() >= 1
    fam = ["pg_sire_mu", "pg_sire_sd", "pg_dam_mu", "pg_lit_mu", "pg_fam_mu", "pg_sband_mu", "pg_strack_mu"]
    # the month of June is fitted on rows before June: the rewritten May runs move the fit a little for everyone, so
    # compare the dog with a dog of another family that month and ask that the dog's own shift is what moves
    d = (t2.loc[probe, fam].to_numpy(float) - tab.loc[probe, fam].to_numpy(float))
    other = ((runs.dog_id != dog) & (runs.sire == runs.loc[runs.dog_id == dog, "sire"].iloc[0])
             & (runs.race_date >= "2025-06-10") & (runs.race_date < "2025-07-01")).to_numpy()
    d_other = (t2.loc[other, fam].to_numpy(float) - tab.loc[other, fam].to_numpy(float))
    assert np.abs(d).max() < 0.25 * np.abs(d_other[:, 0]).max() + 0.02          # its sire moves for its half-sibs
    assert np.abs(d_other[:, 0]).max() > 0.01
    # the dog now looks faster than its family says: the pedigree pulls it back by more (an exposed dog: a little)
    assert (t2.loc[probe, "pg_post_shift"].to_numpy() < tab.loc[probe, "pg_post_shift"].to_numpy() - 0.005).all()


def test_the_trip_band_and_the_improvement_posteriors_find_their_sires(built):
    runs, m, tab, _ = built
    m = m.assign(s=m.sire.str[5:].astype(int))
    late = m[(m.race_date >= "2025-05-01") & (m.s != NEW_SIRE)]
    long_, short = late[late.distance_m == 500.0], late[late.distance_m == 280.0]
    by_long, by_short = long_.groupby("s").pg_sband_mu.mean(), short.groupby("s").pg_sband_mu.mean()
    assert by_long.idxmax() == STAY_SIRE and by_short.idxmin() == STAY_SIRE
    assert by_long[STAY_SIRE] > 5 * by_long.drop(STAY_SIRE).abs().max()
    assert (late.pg_sband_sd > 0).all()
    imp = late.groupby("s").pg_imp_fam_mu.mean()
    assert imp.idxmax() == IMP_SIRE and imp[IMP_SIRE] > 2 * imp.drop(IMP_SIRE).abs().max()


def test_the_days_own_results_move_nothing(built):
    runs, m, tab, _ = built
    t2, _ = pedigree.build(perturb_last_day(runs))
    _same(tab, t2)
    # a middle day (not a month's last): that day and every day before it are unmoved; later days are not
    mid = "2025-04-12"
    r = runs.copy()
    on = r.race_date == mid
    r.loc[on, "adjusted_time"] += 0.6
    r.loc[on, "beaten_lengths"] = 9.0 - r.loc[on, "beaten_lengths"].clip(upper=9.0)
    r.loc[on, "position"] = 7 - r.loc[on, "position"]
    r["won"] = (r.position == 1).astype(float)
    t3, _ = pedigree.build(r)
    upto = (runs.race_date <= mid).to_numpy()
    _same(tab[upto], t3[upto])
    after = (runs.race_date > mid).to_numpy()
    assert np.nanmax(np.abs(tab.loc[after, "pg_sire_mu"].to_numpy() - t3.loc[after, "pg_sire_mu"].to_numpy())) > 1e-3


def test_later_days_move_nothing(built):
    runs, m, tab, _ = built
    for cut in ("2025-03-20", "2025-05-01", "2025-06-09"):
        early = runs[runs.race_date <= cut]
        t2, _ = pedigree.build(early)
        _same(tab[(runs.race_date <= cut).to_numpy()], t2)


def test_a_card_row_gets_the_features_its_run_had(built):
    runs, m, tab, _ = built
    card = cardify_last_day(runs)
    assert card.is_card.sum() > 0 and card.loc[card.is_card, "adjusted_time"].isna().all()
    t2, _ = pedigree.build(card)
    j = card[["race_id", "dog_id", "is_card"]].merge(t2, on=["race_id", "dog_id"]).merge(
        tab, on=["race_id", "dog_id"], suffixes=("", "_full"))
    assert len(j) == len(runs)
    _same(j, j[[c + "_full" for c in COLS]].set_axis(COLS, axis=1))
    last = j[j.is_card]
    assert last[["pg_sire_mu", "pg_fam_mu", "pg_ped_weight", "pg_sband_mu", "pg_strap_mu"]].notna().all().all()


def test_a_dog_with_no_pedigree_is_nan_and_two_runs_a_day_keep_their_keys():
    runs = synth(days=80)
    r = runs.copy()
    r.loc[r.dog_id == 5, ["sire", "dam"]] = None
    extra = r[(r.race_date == "2025-03-10")].head(6).copy()           # the same dogs out again that evening
    extra["race_id"] = 99999
    extra["race_time"] = "21:45"
    extra["t"] = extra.t + pd.Timedelta(hours=9)
    r = pd.concat([r, extra], ignore_index=True)
    tab, _ = pedigree.build(r)
    assert len(tab) == len(r) and not tab.duplicated(["race_id", "dog_id"]).any()
    assert tab.loc[(r.dog_id == 5).to_numpy(), COLS].isna().all().all()
    j = r[["race_id", "dog_id"]].merge(tab, on=["race_id", "dog_id"])
    a = j[j.race_id == 99999].sort_values("dog_id")
    b = j[j.race_id.isin(r.loc[(r.race_date == "2025-03-10") & (r.race_id != 99999), "race_id"])
          & j.dog_id.isin(a.dog_id)].sort_values("dog_id")
    keep = ~a.dog_id.isin([5]).to_numpy()
    same_day = ["pg_sire_mu", "pg_sire_sd", "pg_dam_mu", "pg_lit_mu", "pg_fam_mu", "pg_ped_weight", "pg_rc_fam_mu"]
    np.testing.assert_allclose(a.loc[keep, same_day].to_numpy(float), b.loc[keep, same_day].to_numpy(float), atol=1e-6)


def test_any_index_and_an_empty_frame(built):
    """The table is keyed by position, not by the frame's index: a frame concatenated without ``ignore_index`` (labels
    repeated) gives the same table; a frame with no rows gives an empty one with every column."""
    runs, m, tab, _ = built
    t2, _ = pedigree.build(runs.set_axis(np.arange(len(runs)) % 7))
    assert (t2.race_id.to_numpy() == runs.race_id.to_numpy()).all() and (t2.dog_id.to_numpy() == runs.dog_id.to_numpy()).all()
    _same(tab, t2)
    t0, notes = pedigree.build(runs.iloc[:0])
    assert len(t0) == 0 and list(t0.columns) == ["race_id", "dog_id"] + COLS and notes["rows"] == 0
