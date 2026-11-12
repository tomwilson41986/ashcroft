"""greyhound/pit/fear.py: the fear trace. A dog bumped at the first bend carries a trace that falls by 0.7 a clean
run and not with time; it is renewed on the dog's first race at a new track and partly recovers after a break; a dog
whose sectional is slower after every hit is told from one that shrugs it off; the comment tokens are read to a
severity and a place; the day's own results move nothing; a card row gets what its run had.

The review's tests (11 Nov 2026): the table against the recursion written as a plain loop from the specification; a
MIDDLE day's results move no row of that day or before (exactly), with the expected-time table held and rebuilt; the
frame cut after any day gives the same rows; a second run the same day reads what the first read; the order of the
rows and the traps do not matter."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from greyhound.data import clean
from greyhound.pit import core, fear, xtime
from tests.test_greyhound_pit import cardify_last_day, perturb_last_day

COLS = list(fear.COLUMNS)

HIT, AWAY, OFF = 10, 30, (50, 110)          # dog 1: bumped on day 10; race 1 at a new track on day 30; off 50-109
DROP = 3.0                                  # lengths dog 3's sectional is slower the run after a hit


def row(day, hhmm, meeting, track, rid, trap, dog, pos, sec, t, comment):
    return {"race_date": day, "race_time": hhmm, "meeting_id": meeting, "track": track, "race_id": rid,
            "race_class": "A3", "distance_m": 500.0, "trap": trap, "dog_id": dog, "dog_name": f"Dog {dog}",
            "born": "Jan-2023", "sex": "d", "sp_decimal": 6.0, "position": pos, "beaten_lengths": 0.5,
            "sectional": sec, "run_time": t, "adjusted_time": t, "weight_kg": 30.0, "comment": comment}


def synth(days=130, seed=4):
    """One track, 12 dogs, two races of six a day (dogs 1-6, dogs 7-12). Dog 1 is bumped at the first bend on day
    ``HIT`` and never again, and is off from day 50 to day 109 (a five-runner race meanwhile); dog 2 is never
    touched; dogs 3 and 4 are bumped every fourth day, and dog 3's sectional is ``DROP`` lengths slower the day
    after; the rest are crowded at random. On day ``AWAY`` race 1 is run at a track none of them has seen."""
    rng = np.random.default_rng(seed)
    rows, rid = [], 5000
    for d in range(days):
        day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        for r, field in enumerate((range(1, 7), range(7, 13))):
            rid += 1
            field = [g for g in field if not (g == 1 and OFF[0] <= d < OFF[1])]
            sec = 4.5 + rng.normal(0, 0.02, len(field))
            rest = 25.0 + rng.normal(0, 0.05, len(field))
            com = []
            for k, dog in enumerate(field):
                c = "QAw,Rls"
                if dog == 1 and d == HIT:
                    c = "Mid,Bmp1"
                elif dog in (3, 4) and d % 4 == 1:
                    c = "Bmp1,Mid" if dog == 3 else "Rls,Crd1"
                elif dog >= 5 and rng.random() < 0.2:
                    c = ("EP,Crd1", "Mid,Crd3", "SltCrdRnIn,Wide", "BCrd1&2")[int(rng.integers(4))]
                if dog == 3 and d % 4 == 2:
                    sec[k] += DROP * core.LENGTH_S
                com.append(c)
            t = sec + rest
            pos = np.argsort(np.argsort(t)) + 1
            track = "Romford" if (r == 0 and d == AWAY) else "Hove"
            for k, dog in enumerate(field):
                rows.append(row(day, f"{18 + r}:00", d * 10 + (track == "Romford"), track, rid, k + 1, dog,
                                int(pos[k]), float(sec[k]), float(t[k]), com[k]))
    return clean(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def built():
    runs = synth()
    tab, notes = fear.build(runs, core.per_run_figures(runs))
    m = runs.merge(tab, on=["race_id", "dog_id"])
    m["d"] = (pd.to_datetime(m.race_date) - pd.Timestamp("2025-01-01")).dt.days
    return runs, tab, m, notes


def test_every_column_is_registered_and_there_is_one_row_a_run(built):
    runs, tab, m, notes = built
    assert {c for c in tab.columns if c.startswith(fear.PREFIX)} == set(fear.COLUMNS)
    assert list(tab.columns) == ["race_id", "dog_id"] + list(fear.COLUMNS)
    assert len(tab) == len(runs) == len(m) and not tab.duplicated(["race_id", "dog_id"]).any()
    assert (tab.race_id.to_numpy() == runs.race_id.to_numpy()).all() and (tab.dog_id.to_numpy() == runs.dog_id.to_numpy()).all()
    assert all(str(tab[c].dtype) == "float32" for c in fear.COLUMNS)
    assert notes["rows"] == len(runs) and notes["runs_with_interference_at"]["early"] > 0
    assert notes["interference_events"] >= notes["runs_with_interference"] > 0


def test_the_tokens_are_read_to_a_severity_and_a_place():
    # (severity, early, mid, late, a place was found)
    assert fear.parse_token("bmp1") == (2, True, False, False, True)
    assert fear.parse_token("sltcrd1/2") == (1, False, True, False, True)          # a half is mid, not the digit 1
    assert fear.parse_token("crd1/4") == (2, True, False, False, True)
    assert fear.parse_token("crd¼") == (2, True, False, False, True)
    assert fear.parse_token("crd3/4") == (2, False, True, False, True)
    assert fear.parse_token("vbbmp1") == (4, True, False, False, True)
    assert fear.parse_token("bcrd3") == (3, False, True, False, True)
    assert fear.parse_token("badlybaulked1") == (3, True, False, False, True)
    assert fear.parse_token("verybadlybumped1") == (4, True, False, False, True)   # the words written out
    assert fear.parse_token("slightlycrowded3") == (1, False, True, False, True)
    assert fear.parse_token("vsaw&fcdtock1") == (2, True, False, False, True)      # very slow away is not very badly
    assert fear.parse_token("baulked2") == (2, False, True, False, True)           # 'b' + aulk is not 'badly'
    assert fear.parse_token("bmp1&2") == (2, True, True, False, True)              # counts at both
    assert fear.parse_token("crdrnup") == (2, True, False, False, True)
    assert fear.parse_token("stumbledstart") == (2, True, False, False, True)
    assert fear.parse_token("bmprunin") == (2, False, False, True, True)
    assert fear.parse_token("crd1&rnin") == (2, True, False, True, True)
    assert fear.parse_token("crowded5") == (2, False, False, True, True)
    assert fear.parse_token("crd") == (2, False, True, False, False)               # no place: mid
    assert fear.parse_token("qaw")[0] == 0 and fear.parse_token("ld1")[0] == 0 and fear.parse_token("mid")[0] == 0
    ev, notes = fear.events(pd.Series(["Mid, Bmp1 ,BCrd1&2", None, "", "QAw,Rls", "SltCrdRnIn"]))
    assert list(ev.early) == [5.0, 0.0, 0.0, 0.0, 0.0] and list(ev.mid) == [3.0, 0.0, 0.0, 0.0, 0.0]
    assert list(ev.late) == [0.0, 0.0, 0.0, 0.0, 1.0] and list(ev.n) == [2.0, 0.0, 0.0, 0.0, 1.0]
    assert list(ev.worst) == [3.0, 0.0, 0.0, 0.0, 1.0] and list(ev.said) == [True, False, False, True, True]
    assert notes["distinct_interference_tokens"] == 3


def test_the_trace_falls_by_seven_tenths_a_clean_run(built):
    _, _, m, _ = built
    one = m[m.dog_id == 1].set_index("d")
    assert (one.loc[:HIT, ["fr_early", "fr_mid", "fr_late", "fr_raw_early", "fr_n"]] == 0).all().all()   # incl. the day
    assert one.loc[:HIT, ["fr_clean_early", "fr_last_sev", "fr_last_days"]].isna().all().all()
    for d in range(HIT + 1, AWAY):
        r = one.loc[d]
        assert r.fr_early == pytest.approx(2.0 * 0.7 ** (d - HIT - 1), rel=1e-5)
        assert r.fr_raw_early == pytest.approx(2.0 * 0.5 ** ((d - HIT) / 180.0), rel=1e-5)
        assert r.fr_clean_early == d - HIT - 1 and r.fr_last_sev == 2 and r.fr_last_days == d - HIT and r.fr_n == 1
        assert r.fr_mid == 0 and r.fr_late == 0 and r.fr_renew == 0
    two = m[m.dog_id == 2]
    assert (two[["fr_early", "fr_mid", "fr_late", "fr_raw_early", "fr_n", "fr_renew", "fr_recover"]] == 0).all().all()
    assert two[["fr_clean_early", "fr_last_sev", "fr_last_days", "fr_resil"]].isna().all().all()


def test_time_off_does_not_extinguish_but_the_fear_recovers(built):
    _, _, m, _ = built
    one = m[m.dog_id == 1].set_index("d")
    assert OFF[0] - 1 in one.index and OFF[0] not in one.index and OFF[1] in one.index
    back, before = one.loc[OFF[1]], one.loc[OFF[0] - 1]
    gap = OFF[1] - (OFF[0] - 1)
    assert back.fr_early == pytest.approx(before.fr_early * 0.7, rel=1e-4)        # one pass, whatever the days
    assert back.fr_raw_early == pytest.approx(2.0 * 0.5 ** ((OFF[1] - HIT) / 180.0), rel=1e-5)
    assert back.fr_recover == pytest.approx(back.fr_raw_early * (1 - np.exp(-gap / 60.0)), rel=1e-4)
    assert back.fr_recover > 20 * before.fr_recover > 0                           # a day off recovers next to nothing
    assert m[(m.d == OFF[0]) & (m.race_id == m[(m.d == OFF[0]) & (m.dog_id == 2)].race_id.iloc[0])].shape[0] == 5


def test_the_fear_is_renewed_at_a_new_track(built):
    _, _, m, _ = built
    away = m[(m.d == AWAY) & (m.track == "Romford")].set_index("dog_id")
    assert len(away) == 6
    one = away.loc[1]
    want = 2.0 * 0.5 ** ((AWAY - HIT) / 180.0) - 2.0 * 0.7 ** (AWAY - HIT - 1)
    assert want > 1.8 and one.fr_renew == pytest.approx(want, rel=1e-4)           # nearly the whole of the hit is back
    assert one.fr_renew == pytest.approx(one.fr_raw_early - one.fr_early, rel=1e-4)
    assert away.loc[2].fr_renew == 0 and away.loc[3].fr_renew > 0                 # never hit: nothing to renew
    home = m[m.track == "Hove"]
    assert (home.fr_renew == 0).all() and (m.fr_renew >= 0).all()                 # the track they know; a debut
    assert (m[m.d == 0][list(fear.COLUMNS)].fillna(0) == 0).all().all()


def test_a_dog_slower_after_every_hit_is_told_from_one_that_is_not(built):
    _, _, m, notes = built
    late = m[m.d >= 80]
    three, four = late[late.dog_id == 3].fr_resil, late[late.dog_id == 4].fr_resil
    # DROP lengths from the day before to the day after, about 19 hits by day 80, shrunk by K = 3: about 2.6
    assert 1.8 < three.mean() < DROP + 0.3 and abs(four.mean()) < 0.6 and three.min() > four.max() + 1.0
    # day 0 has no expected sectional, so the first triple read is days 4, 5 (the hit), 6: the row of day 6, the
    # triple's own day after, does not read it; day 7 does
    assert m[(m.dog_id == 3) & (m.d <= 6)].fr_resil.isna().all()
    assert m[(m.dog_id == 3) & (m.d == 7)].fr_resil.notna().all()
    assert notes["early_hit_triples"] > 50 and notes["mean_sectional_change_after_an_early_hit"] > \
        notes["mean_sectional_change_control"]
    # hit every fourth day with three clean passes between: the trace settles at 2 / (1 - 0.7 ** 3) before a hit
    before_hit = late[(late.dog_id == 4) & (late.d % 4 == 1)].fr_early
    assert np.allclose(before_hit, 2.0 * 0.7 ** 3 / (1 - 0.7 ** 3), rtol=1e-3)
    assert (late[late.dog_id == 4].fr_n == late[late.dog_id == 4].d.map(lambda d: (d + 2) // 4)).all()


def edge_frame():
    """Dog 1 runs twice on day 1 and is hit both times; dog 2 is hit on day 1 and next runs 100 days later; dog 3
    has its first run on the last day; day 2 is a void race (no comment)."""
    rows = [row("2025-03-01", "18:00", 1, "Hove", 1, 1, 1, 1, 4.5, 29.5, "Bmp1"),
            row("2025-03-01", "18:00", 1, "Hove", 1, 2, 2, 2, 4.5, 29.6, "VBBmp1&2"),
            row("2025-03-01", "21:00", 1, "Hove", 2, 1, 1, 1, 4.5, 29.5, "BCrd1,Crd3"),
            row("2025-03-02", "18:00", 2, "Hove", 3, 1, 1, 1, 4.5, 29.5, None),
            row("2025-03-03", "18:00", 3, "Hove", 4, 1, 1, 1, 4.5, 29.5, "QAw"),
            row("2025-03-04", "18:00", 4, "Hove", 5, 1, 1, 1, 4.5, 29.5, "QAw"),
            row("2025-06-09", "18:00", 5, "Hove", 6, 1, 1, 1, 4.5, 29.5, "QAw"),
            row("2025-06-09", "18:00", 5, "Hove", 6, 2, 2, 2, 4.5, 29.6, "QAw"),
            row("2025-06-09", "18:00", 5, "Hove", 6, 3, 3, 3, 4.5, 29.7, "Crd1")]
    return clean(pd.DataFrame(rows))


def test_same_day_runs_a_void_day_time_off_and_a_dog_with_no_history():
    runs = edge_frame()
    xt = runs[["race_id", "dog_id"]].assign(xt_sec_time=4.5)
    tab, _ = fear.build(runs, core.per_run_figures(runs), {"xtime": xt})
    m = runs[["race_id", "dog_id", "race_date"]].merge(tab, on=["race_id", "dog_id"]).set_index(["race_id", "dog_id"])
    cols = list(fear.COLUMNS)
    # both runs of day 1 read nothing: the earlier run that day is not an earlier day
    assert (m.loc[(1, 1), cols].fillna(0) == 0).all() and (m.loc[(2, 1), cols].fillna(0) == 0).all()
    void = m.loc[(3, 1)]                                             # day 2 reads both runs of day 1
    assert void.fr_early == 5 and void.fr_mid == 2 and void.fr_n == 3 and void.fr_last_sev == 3
    assert void.fr_last_days == 1 and void.fr_clean_early == 0
    d3 = m.loc[(4, 1)]                                               # the void day was not a clean pass
    assert d3.fr_early == 5 and d3.fr_clean_early == 0 and d3.fr_last_days == 2
    d4 = m.loc[(5, 1)]
    assert d4.fr_early == pytest.approx(3.5) and d4.fr_mid == pytest.approx(1.4) and d4.fr_clean_early == 1
    far = m.loc[(6, 2)]                                              # 100 days and not one pass: the trace is whole
    assert far.fr_early == 4 and far.fr_mid == 4 and far.fr_clean_early == 0 and far.fr_last_sev == 4
    assert far.fr_last_days == 100 and far.fr_raw_early == pytest.approx(4 * 0.5 ** (100 / 180), rel=1e-5)
    assert far.fr_recover == pytest.approx(far.fr_raw_early * (1 - np.exp(-100 / 60)), rel=1e-5)
    assert far.fr_renew == 0                                         # time took the raw memory below the trace: not < 0
    new = m.loc[(6, 3)]                                              # no history, and its own 'Crd1' today is not read
    assert (new[["fr_early", "fr_mid", "fr_late", "fr_raw_early", "fr_n", "fr_renew", "fr_recover"]] == 0).all()
    assert new[["fr_clean_early", "fr_last_sev", "fr_last_days", "fr_resil"]].isna().all()
    assert len(tab) == len(runs)


def test_an_empty_frame_builds():
    runs = edge_frame()
    figs = core.per_run_figures(runs).iloc[:0]
    tab, notes = fear.build(runs.iloc[:0], figs, {"xtime": pd.DataFrame(columns=["race_id", "dog_id", "xt_sec_time"])})
    assert len(tab) == 0 and set(fear.COLUMNS) <= set(tab.columns) and notes["rows"] == 0


@pytest.mark.parametrize("change", [perturb_last_day, cardify_last_day])
def test_the_days_own_results_move_nothing(change):
    runs = synth(days=40)
    keys = runs.loc[runs.race_date == runs.race_date.max(), ["race_id", "dog_id"]]
    a = fear.build(runs, core.per_run_figures(runs))[0]
    r2 = change(runs)
    b = fear.build(r2, core.per_run_figures(r2))[0]
    cols = list(fear.COLUMNS)
    x = keys.merge(a, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    z = keys.merge(b, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    assert np.isfinite(x).mean() > 0.8 and np.allclose(x, z, atol=1e-4, equal_nan=True)


def test_the_expected_time_table_is_taken_from_the_context():
    runs = synth(days=30)
    figs = core.per_run_figures(runs)
    xt, _ = xtime.build(runs, figs)
    a = fear.build(runs, figs, {"xtime": xt})[0]
    b = fear.build(runs, figs)[0]
    assert np.allclose(a[list(fear.COLUMNS)].to_numpy(float), b[list(fear.COLUMNS)].to_numpy(float), equal_nan=True)


# ---- the review's tests (11 Nov 2026): a reference written from the specification, and stronger leak tests ---------

def ref_token(tok):
    """The specification's reading of one token, written apart from ``fear.parse_token``: (severity, places)."""
    if not xtime.TROUBLE.search(tok):
        return 0, set()
    if "slt" in tok or "slight" in tok:
        sev = 1
    elif tok.startswith("vb") or "verybadly" in tok:
        sev = 4
    elif any(tok.startswith("b" + c) for c in ("crd", "bmp", "blk", "baulk", "bump", "crowd")) or "badly" in tok:
        sev = 3
    else:
        sev = 2
    places = set()
    for frac, place in (("1/4", "early"), ("¼", "early"), ("1/2", "mid"), ("½", "mid"), ("3/4", "mid"), ("¾", "mid")):
        if frac in tok:
            places.add(place)
            tok = tok.replace(frac, "")
    if any(w in tok for w in ("rnup", "runup", "stt", "start", "1")):
        places.add("early")
    if any(d in tok for d in "234"):
        places.add("mid")
    if any(w in tok for w in ("rnin", "runin", "5", "6")):
        places.add("late")
    return sev, places or {"mid"}


def reference(figs, xt):
    """The table by the recursion as the specification writes it: a plain loop over each dog's days."""
    df = figs.merge(xt[["race_id", "dog_id", "xt_sec_time"]].drop_duplicates(["race_id", "dog_id"]),
                    on=["race_id", "dog_id"], how="left")
    df["_dn"] = (pd.to_datetime(df.race_date) - pd.Timestamp("2000-01-01")).dt.days
    card = df.is_card.fillna(False).astype(bool) if "is_card" in df else pd.Series(False, index=df.index)
    trial = df.is_trial.fillna(False).astype(bool) if "is_trial" in df else pd.Series(False, index=df.index)
    out = []
    for dog, g in df.groupby("dog_id", sort=False):
        F = dict.fromkeys(fear.PLACES, 0.0)
        hits, seen, n_ev, clean, last, prev = [], set(), 0.0, np.nan, None, None
        hist, tri_sum, tri_n = [], 0.0, 0
        for dn, day in sorted(g.groupby("_dn"), key=lambda kv: kv[0]):
            raw = sum(s * 0.5 ** ((dn - d0) / fear.HALF_LIFE) for d0, s in hits)
            for _, r in day.iterrows():
                first_here = prev is not None and r.track not in seen
                out.append({
                    "race_id": r.race_id, "dog_id": dog, "fr_early": F["early"], "fr_mid": F["mid"],
                    "fr_late": F["late"], "fr_raw_early": raw, "fr_clean_early": clean,
                    "fr_last_sev": last[1] if last else np.nan, "fr_last_days": dn - last[0] if last else np.nan,
                    "fr_n": n_ev, "fr_renew": max(raw - F["early"], 0.0) if first_here else 0.0,
                    "fr_recover": raw * (1 - np.exp(-(dn - prev) / fear.TAU)) if prev is not None else 0.0,
                    "fr_resil": tri_sum / (tri_n + fear.K) if tri_n else np.nan})
            sev, worst, events, said, res = dict.fromkeys(fear.PLACES, 0.0), 0, 0, False, []
            for i, r in day.iterrows():
                text = "" if card[i] or not isinstance(r.comment, str) else r.comment
                toks = [t for t in "".join(text.lower().split()).split(",") if t]
                said |= bool(toks)
                for t in toks:
                    s, places = ref_token(t)
                    if s:
                        events, worst = events + 1, max(worst, s)
                        for p in places:
                            sev[p] += s
                e = (r.sectional - r.xt_sec_time) / core.LENGTH_S
                fallen = bool(re.search(r"\(Handslip\)|Fell|Fll|BrkLine", text, flags=re.I))
                if np.isfinite(e) and abs(e) <= fear.SEC_CLIP and not trial[i] and not fallen:
                    res.append(e)
            for p in fear.PLACES:
                if said and not sev[p]:
                    F[p] *= fear.RHO
                F[p] += sev[p]
            if sev["early"]:
                hits.append((dn, sev["early"]))
                clean = 0.0
            elif said and not np.isnan(clean):
                clean += 1.0
            if events:
                last = (dn, worst)
            n_ev += events
            hist.append((said, sev["early"] > 0, np.mean(res) if res else np.nan))
            if len(hist) >= 3:
                (s0, h0, e0), (_, h1, _), (s2, h2, e2) = hist[-3:]
                if h1 and s0 and s2 and not h0 and not h2 and np.isfinite(e0) and np.isfinite(e2):
                    tri_sum, tri_n = tri_sum + (e2 - e0), tri_n + 1
            seen |= set(day.track)
            prev = dn
    return pd.DataFrame(out)


def against_reference(figs, xt):
    tab = fear.build(None, figs, {"xtime": xt})[0]
    ref = figs[["race_id", "dog_id"]].merge(reference(figs, xt), on=["race_id", "dog_id"], how="left")
    assert len(ref) == len(tab) and (ref.race_id.to_numpy() == tab.race_id.to_numpy()).all()
    return tab[COLS].to_numpy(float), ref[COLS].to_numpy(float)


def perturb_day(runs, day):
    """One day's results rewritten: positions, times, sectionals, comments (very badly bumped), SPs, weights."""
    r = runs.copy()
    m = r.race_date == day
    r.loc[m, "position"] = 7 - r.loc[m, "position"]
    r.loc[m, ["adjusted_time", "run_time"]] += 0.5
    r.loc[m, "sectional"] += 0.2
    r.loc[m, "comment"] = "SAw,VBBmp1&2,CrdRnIn"
    r.loc[m, "sp_decimal"] = 9.0
    r.loc[m, "weight_kg"] += 2.0
    r["won"] = (r.position == 1).astype(float)
    return r


def keyed(tab, keys):
    return keys.merge(tab, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[COLS].to_numpy(float)


def test_the_table_is_the_recursion_written_out_as_a_loop():
    runs = synth(days=70)
    figs = core.per_run_figures(runs)
    got, want = against_reference(figs, xtime.build(runs, figs)[0])
    assert np.allclose(got, want, rtol=1e-4, atol=1e-4, equal_nan=True)
    assert np.isfinite(want).mean() > 0.8 and (want[:, COLS.index("fr_renew")] > 0).any()
    assert np.isfinite(want[:, COLS.index("fr_resil")]).any()
    runs = edge_frame()
    got, want = against_reference(core.per_run_figures(runs), runs[["race_id", "dog_id"]].assign(xt_sec_time=4.5))
    assert np.allclose(got, want, rtol=1e-4, atol=1e-4, equal_nan=True)


@pytest.mark.parametrize("rebuild_the_expected_times", [False, True])
def test_a_middle_days_results_move_nothing_on_that_day_or_before(rebuild_the_expected_times):
    """Stronger than the last day: day D's results rewritten move no row of day D or of any earlier day, exactly,
    with the expected-time table held and with it rebuilt from the rewritten results (D is past the fifth of the
    rows that table fits its filter's constants on); and they do move later rows, so the test has teeth."""
    runs = synth(days=80)
    days = sorted(runs.race_date.unique())
    figs = core.per_run_figures(runs)
    xt = xtime.build(runs, figs)[0]
    a = fear.build(runs, figs, {"xtime": xt})[0]
    # day 46: dogs 3 and 4 were bumped on day 45, so their triple (44, 45, 46) is completed by day 46's own sectional
    for D in (days[45], days[HIT], days[AWAY]) if not rebuild_the_expected_times else (days[46], days[58]):
        r2 = perturb_day(runs, D)
        f2 = core.per_run_figures(r2)
        b = fear.build(r2, f2, {"xtime": xtime.build(r2, f2)[0] if rebuild_the_expected_times else xt})[0]
        upto = runs.loc[runs.race_date <= D, ["race_id", "dog_id"]]
        after = runs.loc[runs.race_date > D, ["race_id", "dog_id"]]
        assert np.array_equal(keyed(a, upto), keyed(b, upto), equal_nan=True), D
        moved = ~np.isclose(keyed(a, after), keyed(b, after), equal_nan=True)
        for c in ("fr_early", "fr_mid", "fr_late", "fr_raw_early", "fr_clean_early", "fr_last_sev", "fr_last_days",
                  "fr_n", "fr_recover", "fr_resil"):
            assert moved[:, COLS.index(c)].any(), (D, c)


def test_the_frame_cut_after_any_day_gives_the_same_rows():
    """No statistic of the whole frame reaches an earlier row (the first race at a track, the day numbering, the
    dog-day order): the build on the days up to D is the build on all of them, row for row."""
    runs = synth(days=60)
    days = sorted(runs.race_date.unique())
    figs = core.per_run_figures(runs)
    xt = xtime.build(runs, figs)[0]
    a = fear.build(runs, figs, {"xtime": xt})[0]
    for D in (days[0], days[HIT], days[AWAY], days[45]):
        cut = runs.race_date <= D
        b = fear.build(runs[cut], figs[cut.to_numpy()], {"xtime": xt})[0]
        keys = runs.loc[cut, ["race_id", "dog_id"]]
        assert len(b) == int(cut.sum()) and np.array_equal(keyed(a, keys), keyed(b, keys), equal_nan=True), D


def test_the_order_of_the_rows_and_the_traps_do_not_matter():
    runs = synth(days=40)
    figs = core.per_run_figures(runs)
    xt = xtime.build(runs, figs)[0]
    a = fear.build(runs, figs, {"xtime": xt})[0]
    perm = np.random.default_rng(1).permutation(len(figs))
    b = fear.build(runs.iloc[perm], figs.iloc[perm], {"xtime": xt})[0]
    assert (b.race_id.to_numpy() == a.race_id.to_numpy()).all() and (b.dog_id.to_numpy() == a.dog_id.to_numpy()).all()
    assert np.allclose(a[COLS].to_numpy(float), b[COLS].to_numpy(float), rtol=1e-6, atol=1e-6, equal_nan=True)
    wide = figs.assign(trap=figs.trap.where(figs.dog_id != 5, 8.0))        # a reserve's trap: no column reads it
    c = fear.build(runs, wide, {"xtime": xt})[0]
    assert np.array_equal(a[COLS].to_numpy(float), c[COLS].to_numpy(float), equal_nan=True)


def test_a_second_run_the_same_day_reads_what_the_first_read():
    """Dog 1, with a history, runs at 14:00 and at 21:00 on day 5: the evening run reads nothing of the afternoon
    (its comment, its sectional), whatever happened in it; the next day reads both."""
    def frame(afternoon, sec):
        rows = [row("2025-03-01", "18:00", 1, "Hove", 1, 1, 1, 1, 4.5, 29.5, "Bmp1"),
                row("2025-03-02", "18:00", 2, "Hove", 2, 1, 1, 1, 4.5, 29.5, "QAw"),
                row("2025-03-03", "18:00", 3, "Hove", 3, 1, 1, 1, 4.5, 29.5, "Crd1"),
                row("2025-03-04", "18:00", 4, "Hove", 4, 1, 1, 1, 4.5, 29.5, "QAw"),
                row("2025-03-05", "14:00", 5, "Romford", 5, 1, 1, 1, sec, 29.5, afternoon),
                row("2025-03-05", "21:00", 6, "Hove", 6, 1, 1, 1, 4.5, 29.5, "QAw"),
                row("2025-03-06", "18:00", 7, "Hove", 7, 1, 1, 1, 4.5, 29.5, "QAw")]
        runs = clean(pd.DataFrame(rows))
        tab = fear.build(runs, core.per_run_figures(runs), {"xtime": runs[["race_id", "dog_id"]].assign(xt_sec_time=4.5)})[0]
        return tab.set_index("race_id")[COLS]
    quiet, hit = frame("QAw", 4.5), frame("VBBmp1,CrdRnIn", 5.2)
    assert np.array_equal(quiet.loc[[5, 6]].to_numpy(float), hit.loc[[5, 6]].to_numpy(float), equal_nan=True)
    both = quiet.loc[[5, 6]].drop(columns="fr_renew")
    assert np.array_equal(both.iloc[0].to_numpy(float), both.iloc[1].to_numpy(float), equal_nan=True)
    assert quiet.loc[5, "fr_early"] == pytest.approx(2 * 0.7 * 0.7 + 2 * 0.7) and quiet.loc[5, "fr_clean_early"] == 1
    assert quiet.loc[5, "fr_renew"] > 0 and quiet.loc[6, "fr_renew"] == 0        # Romford is new that day, Hove is not
    assert quiet.loc[5, "fr_resil"] == pytest.approx(0.0)                          # days 2, 3 (hit), 4: no change
    assert hit.loc[7, "fr_early"] == pytest.approx(quiet.loc[5, "fr_early"] + 4)   # the next day reads the afternoon
    assert quiet.loc[7, "fr_early"] == pytest.approx(quiet.loc[5, "fr_early"] * 0.7)
    assert hit.loc[7, "fr_late"] == 2 and hit.loc[7, "fr_last_sev"] == 4 and hit.loc[7, "fr_last_days"] == 1
    assert hit.loc[7, "fr_n"] == quiet.loc[7, "fr_n"] + 2


def test_a_card_row_with_a_history_gets_features_and_its_blank_day_is_no_pass():
    runs = synth(days=30)
    card = cardify_last_day(runs)
    tab = fear.build(card, core.per_run_figures(card))[0]
    last = card.race_date == card.race_date.max()
    assert bool(card.is_card[last].all()) and len(tab) == len(card)
    assert tab.loc[last.to_numpy(), ["fr_early", "fr_mid", "fr_late", "fr_raw_early", "fr_n", "fr_renew",
                                     "fr_recover"]].notna().all().all()
    assert tab.loc[last.to_numpy(), "fr_last_days"].notna().mean() > 0.8
