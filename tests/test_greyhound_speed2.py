"""greyhound/pit/speed2.py: speed figures on the review's lines (clean-runner going by side and half, relinked, fitted
blend, decay by days, linked gsr/esr/fsr stand-ins). Later results move nothing on or before their day; a card row
gets what its run had; the planted strong league has the higher figures; the going's side and half, the blend and the
half-life are recovered where planted; secfig.figures' new keywords leave its output bit-identical. Planted data
only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.pit import core, secfig, speed, speed2
from tests.test_greyhound_league import rewrite_results, synth
from tests.test_greyhound_pit import cardify_last_day
from tests.test_greyhound_secfig import _two_tracks

COLS = list(speed2.COLUMNS)
KEY = ["race_id", "dog_id"]


@pytest.fixture(scope="module")
def built():
    runs = synth()
    tab, notes = speed2.build(runs)
    return runs, tab, notes


def _on(runs, tab, mask):
    return runs.loc[mask, KEY].merge(tab, on=KEY, how="left")[COLS].to_numpy(float)


def test_the_contract(built):
    runs, tab, notes = built
    assert list(tab.columns) == KEY + COLS and all(c.startswith(speed2.PREFIX) for c in COLS)
    assert len(tab) == len(runs) and not tab.duplicated(KEY).any()
    assert (tab.race_id.to_numpy() == runs.race_id.to_numpy()).all()
    assert not any(c.startswith(speed2.PREFIX) for c in speed.COLUMNS)
    late = (runs.race_date >= "2025-03-01").to_numpy()
    for c in ("sv_mean3", "sv_dk", "sv_vs", "sv_gsr_max6", "sv_esr_dk", "sv_fsr_mean3", "sv_var_last"):
        assert tab.loc[late, c].notna().mean() > 0.9, c
    assert 0.0 <= notes["blend"] <= 1.0 and set(notes["half_life_days"]) == {"sv", "gsr", "esr", "fsr"}
    assert notes["clean_runners"] > 0 and notes["fit_races"] >= speed2.MIN_FIT_RACES


@pytest.mark.parametrize("cut", ["last", "middle"])
def test_later_results_move_nothing_on_or_before_their_day(built, cut):
    runs, tab, notes = built
    days = sorted(runs.race_date.unique())
    since = days[-1] if cut == "last" else days[len(days) // 2]
    tab2, notes2 = speed2.build(rewrite_results(runs, since))
    keep = (runs.race_date <= since).to_numpy()
    assert np.allclose(_on(runs, tab, keep), _on(runs, tab2, keep), equal_nan=True, atol=1e-5)
    assert notes2["blend"] == notes["blend"] and notes2["half_life_days"] == notes["half_life_days"]


def test_a_card_row_gets_what_its_run_had(built):
    runs, tab, _ = built
    card_tab, _ = speed2.build(cardify_last_day(runs))
    last = (runs.race_date == runs.race_date.max()).to_numpy()
    x = runs.loc[last, KEY].merge(tab, on=KEY)[COLS].to_numpy(float)
    y = runs.loc[last, KEY].merge(card_tab, on=KEY)[COLS].to_numpy(float)
    assert np.allclose(x, y, equal_nan=True, atol=1e-5)


def test_the_planted_strong_league_has_the_higher_figures(built):
    runs, tab, _ = built
    m = runs.merge(tab, on=KEY)
    late = m[m.race_date >= "2025-03-01"]
    for c in ("sv_dk", "sv_gsr_dk", "sv_gsr_mean3"):
        s = late.groupby(["track", "race_class"])[c].mean()
        assert s[("Hove", "A1")] > s[("Hove", "A5")], c


def test_esr_is_secfigs_linked_sectional_aggregated():
    r = _two_tracks()
    f = secfig.figures(r)
    dog, dn = f["dog"], f["dn"]
    late = (r.race_date >= "2025-01-01").to_numpy()
    fast = (r.dog_id <= 30).to_numpy()
    lvl, _ = speed2.decayed(dog, dn, f["L"], 30.0)
    assert np.nanmean(lvl[late & fast]) - np.nanmean(lvl[late & ~fast]) > 0.7    # the planted 1.0 percent, linked


# ---- the pieces ----------------------------------------------------------------------------------------------------

def test_decayed_is_the_day_weighted_mean_of_earlier_days():
    dog = np.array([1.0, 1, 1, 1, 2])
    dn = np.array([0.0, 10, 10, 40, 5])
    val = np.array([2.0, 4.0, 6.0, np.nan, 1.0])
    lvl, fresh = speed2.decayed(dog, dn, val, 10.0)
    assert np.isnan(lvl[0]) and fresh[0] == 0.0                    # nothing earlier
    assert np.isclose(lvl[1], 2.0) and np.isclose(fresh[1], 0.5)   # day 0's 2.0, ten days (one half-life) ago
    assert np.isclose(lvl[3], (2.0 * 0.0625 + 5.0 * 0.125) / 0.1875)  # day 10's two runs averaged, then weighted
    assert np.isclose(fresh[3], 0.0625 + 0.125) and np.isnan(lvl[4])


def test_the_allowance_recovers_a_planted_side_and_half():
    """400 meetings of 60 clean runners: a night effect, the outside third 1 percent quick on half the nights, the
    second half of the meeting 0.8 percent slow on a third of them, noise sd 1."""
    rng = np.random.default_rng(2)
    m = np.repeat(np.arange(400), 60)
    trap = np.tile(np.repeat(np.arange(1, 7), 10), 400).astype(float)
    tt = np.tile(np.arange(60) % 12, 400).astype(float)
    night = rng.normal(0, 0.5, 400)[m]
    outq = (rng.random(400) < 0.5)[m] & (trap >= 5)
    late_slow = (rng.random(400) < 0.33)[m] & (tt >= 6)
    res = night + 1.0 * outq - 0.8 * late_slow + rng.normal(0, 1.0, len(m))
    early = m < 200
    vn, vs, vh, ks = speed2.allowance(m.astype(str), trap, tt, res, np.ones(len(m), bool), early)
    assert np.corrcoef(vn, night)[0, 1] > 0.8
    dside = vs[outq].mean() - vs[(trap >= 5) & ~outq].mean()
    dhalf = vh[late_slow].mean() - vh[(tt >= 6) & ~late_slow].mean()
    assert 0.25 < dside < 1.0 and -0.8 < dhalf < -0.1, (dside, dhalf)      # shrunk towards 0, the right way


def _races(n_races=3000, field=6, seed=4):
    rng = np.random.default_rng(seed)
    race = np.repeat(np.arange(n_races), field)
    return rng, race


def test_the_blend_is_recovered():
    """Winners drawn from a softmax of 2 x (A - 0.3 B): the planted w = 0.7."""
    rng, race = _races()
    A, B = rng.normal(0, 1, len(race)), rng.normal(0, 1, len(race))
    u = 2.0 * (A - 0.3 * B)
    p = core.group_softmax(u, race)
    won = np.zeros(len(race), bool)
    cum = pd.Series(p).groupby(race).cumsum().to_numpy()
    draw = rng.random(race.max() + 1)[race]
    first = (cum >= draw) & ~pd.Series(cum >= draw).groupby(race).shift(1, fill_value=False).to_numpy()
    won[first] = True
    rows = speed2.fit_races(race, won, np.ones(len(race), bool))
    w, nt = speed2.fit_blend(A, B, race, won, rows)
    assert 0.55 <= w <= 0.85, nt


def test_the_half_life_is_recovered():
    """Each dog's ability a random walk; a figure each run is the ability plus noise; winners from the ability. The
    level decayed with the right day half-life beats one decayed far too fast or too slow."""
    rng = np.random.default_rng(7)
    ndog, ndays = 300, 400
    walk = np.cumsum(rng.normal(0, 0.03, (ndog, ndays)), axis=1) + rng.normal(0, 1.0, (ndog, 1))
    rows = []
    rid = 0
    for d in range(0, ndays, 2):
        pool = rng.permutation(ndog)[:240]
        for k in range(40):
            rid += 1
            for dg in pool[6 * k:6 * k + 6]:
                rows.append((rid, dg, d))
    race, dog, dn = (np.array(x) for x in zip(*rows))
    ab = walk[dog, dn]
    fig = ab + rng.normal(0, 1.0, len(dog))
    p = core.group_softmax(2.0 * ab, race - 1)
    cum = pd.Series(p).groupby(race).cumsum().to_numpy()
    draw = rng.random(race.max() + 1)[race]
    won = (cum >= draw) & ~pd.Series(cum >= draw).groupby(race).shift(1, fill_value=False).to_numpy()
    use = dn >= 60
    rows_ = speed2.fit_races(race, won, use)
    h, nt = speed2.fit_half(dog.astype(float), dn.astype(float), fig, race, won, rows_, grid=(2.0, 30.0, 2000.0))
    assert h == 30.0, nt


# ---- secfig's new keywords -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("frame", ["synth", "two_tracks"])
def test_secfig_defaults_are_bit_identical(frame):
    r = synth() if frame == "synth" else _two_tracks()
    if frame == "synth":
        r = r.assign(sectional=4.5 + np.random.default_rng(0).normal(0, 0.05, len(r)))
    a, b = _base_figures(r), secfig.figures(r)
    c = secfig.figures(r, values=speed._num(r.sectional), lo=1.0, hi=15.0, by_trap=True)
    for k in ("dog", "dn", "race", "L", "rel", "v"):
        assert np.array_equal(a[k], b[k], equal_nan=True) and np.array_equal(a[k], c[k], equal_nan=True), k
    assert a["notes"] == b["notes"] == c["notes"]
    t1, _ = secfig.build(r)
    assert set(t1.columns) == {"race_id", "dog_id", *secfig.COLUMNS}


def test_secfig_run_home_uses_the_track_par():
    r = _two_tracks()
    f = secfig.figures(r, values=speed._num(r.sectional) * 5.0, lo=1.0, hi=np.inf, by_trap=False)
    assert np.isfinite(f["L"]).sum() > 0.8 * len(r)


def _base_figures(r: pd.DataFrame) -> dict:
    """secfig.figures as at the base commit d2d7d80 (frozen here for the identity test)."""
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
