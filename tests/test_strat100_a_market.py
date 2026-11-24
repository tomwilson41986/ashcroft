"""The a_market strategies (S069-S100) on a toy database with the data map's columns: every strategy runs and returns
a bets frame the core accepts; the loader drops the holdout; the fills, limits and stakes do what each specification
says; and no selection reads a column it could not have known when the order went in (look-ahead). Every figure here
comes from a seeded toy frame; no real data is read."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from research.strat100 import core
from research.strat100 import data_a_market as D
from research.strat100 import strategies_a_market as S

TRAINERS = [f"{s}, {i}" for s, i in (("Able", "A"), ("Baker", "B"), ("Cole", "C"), ("Dunn", "D"), ("Eve", "E"),
                                     ("Ford", "F"), ("Gray", "G"), ("Hall", "H"), ("Ives", "I"), ("Judd", "J"))]
JOCKEYS = [f"{s}, {i}" for s, i in (("Kent", "K"), ("Lane", "L"), ("Moss", "M"), ("Nash", "N"), ("Owen", "O"),
                                    ("Penn", "P"), ("Quin", "Q"), ("Reid", "R"), ("Shaw", "S"))]
TRAVELLER = "Moore, Ryan"                     # rides every Ascot race and one at Kempton
COMMENTS = {"lead": ["made all, ran on well", "led, headed final furlong, no extra"],
            "prom": ["chased leaders, kept on same pace", "prominent, ridden 2f out, one pace"],
            "mid": ["mid-division, never dangerous", "midfield, some headway final furlong"],
            "rear": ["held up in rear, never nearer", "held up, stayed on late"]}
#   (name, track, hh:mm, race_type, race_code, surface, furlongs, declared, handicap, pool, class)
RACES = [("6f Hcap", "Ascot", "13:30", "Flat", "Flat", "Turf", 6.0, 8, True, "flat", "Class 4"),
         ("2m4f Hcap Chs", "Ascot", "14:05", "Chase", "National Hunt", "Turf", 20.0, 6, True, "jump", "Class 3"),
         ("2m Nov Hrd", "Ascot", "14:40", "Hurdle", "National Hunt", "Turf", 16.0, 9, False, "jump", "Class 4"),
         ("1m2f Hcap", "Kempton", "18:00", "Flat", "All Weather", "AW", 10.0, 10, True, "aw", "Class 5")]
BIG = ("7f Hcap", "Kempton", "18:30", "Flat", "All Weather", "AW", 7.0, 16, True, "big", "Class 2")


def _horses(rng):
    hs, sid = [], 1000
    for pool, n, leaders, rears in (("flat", 30, 3, 6), ("jump", 40, 4, 8), ("aw", 30, 8, 10), ("big", 44, 6, 10)):
        for i in range(n):
            style = "lead" if i < leaders else "rear" if i < leaders + rears else ("prom" if i % 2 else "mid")
            a = rng.normal(0.0, 0.7)
            trait = ""
            if i % 11 == 5:
                a, trait = 0.9, "flatterer"            # trades short and loses
            elif i % 13 == 7:
                a = -4.0                                # a no-hoper: 50+ in a small field
            elif i % 9 == 4:
                a = 2.0                                 # a standout: an odds-on favourite
            hs.append({"name": f"{pool.title()} Horse {i}", "sid": sid, "pool": pool, "a": a, "style": style,
                       "trait": trait, "sloppy": pool == "jump" and i % 3 == 0, "slow": i % 7 == 3,
                       "trainer": TRAINERS[i % len(TRAINERS)], "headgear": "b" if i % 5 == 0 else "", "runs": 0,
                       "last": None})
            sid += 1
    return hs


def toy(seed: int = 11, until: str = "2026-04-20"):
    """(race_results, betfair_prices) as the database holds them, 2018-01-03 on, with a few days in the holdout."""
    rng = np.random.default_rng(seed)
    horses = _horses(rng)
    pools = {p: [h for h in horses if h["pool"] == p] for p in ("flat", "jump", "aw", "big")}
    rr, bp = [], []
    rid, bid, eid = 1, 1, 100000
    day, n_day = date(2018, 1, 3), 0
    end = date.fromisoformat(until)
    while day <= end:
        iso = day.isoformat()
        ran_today = set()
        card = RACES + ([BIG] if n_day % 6 == 0 else [])
        for r_i, (ename, track, hhmm, rtype, rcode, surf, furl, declared, hcap, pool, cls) in enumerate(card):
            cands = [h for h in pools[pool] if h["sid"] not in ran_today]
            field = [cands[i] for i in rng.choice(len(cands), size=declared, replace=False)]
            non_runner = field[-1] if (r_i == 0 and n_day % 7 == 3) else None
            runners = [h for h in field if h is not non_runner]
            n = len(runners)
            k = 2 if declared <= 7 else 4 if (declared >= 16 and hcap) else 3
            s = np.exp(np.array([h["a"] for h in runners]) + rng.normal(0, 0.3, n))
            p = s / s.sum()
            order = np.argsort(-(np.log(p) + rng.gumbel(size=n)))
            if runners[order[0]]["trait"] == "flatterer":                    # a flatterer never wins
                order[[0, 1]] = order[[1, 0]]
            pos = np.empty(n, int)
            pos[order] = np.arange(1, n + 1)
            bsp = np.clip(1.0 / p * np.exp(rng.normal(0, 0.03, n)), 1.05, 900.0)
            ppwap = np.clip(bsp * np.exp(rng.normal(0, 0.06, n)), 1.03, 950.0)
            pp = p ** 0.75
            pp = np.minimum(0.97, pp / pp.sum() * k)
            pbsp = 1.0 / (pp * np.exp(rng.normal(0, 0.25, n)))
            pbsp = np.maximum(1.02, pbsp * (1.0 / pbsp).sum() / (k * 1.01))
            hh, mm = hhmm.split(":")
            menu = f"{track} {day.day}th {day.strftime('%b')}"
            eid += 2
            jocks = list(rng.choice(JOCKEYS, size=n))
            if track == "Ascot" or r_i == 3:
                jocks[int(np.argmin(bsp))] = TRAVELLER                       # the traveller rides the favourite
            going = "Standard" if surf == "AW" else ["Good", "Soft", "Good To Firm"][n_day % 3]
            for j, h in enumerate(field):
                is_nr = h is non_runner
                i = runners.index(h) if not is_nr else -1
                gap = None if h["last"] is None else (day - h["last"]).days
                if not is_nr and rng.random() < 0.03:
                    h["trainer"] = TRAINERS[(TRAINERS.index(h["trainer"]) + 3) % len(TRAINERS)]
                if not is_nr and rng.random() < 0.08:
                    h["headgear"] = "" if h["headgear"] else "b"
                mu = (0.10 if h["trainer"] == TRAINERS[0] else -0.02) + (0.12 if gap is not None and gap <= 4 else 0.0)
                if is_nr:
                    am, pw, sp, psp, won, plc = 12.0, 11.0, np.nan, np.nan, 0.0, 0.0
                    lo = hi = np.nan
                else:
                    sp, pw, psp = bsp[i], ppwap[i], pbsp[i]
                    am = float(np.clip(pw * np.exp(rng.normal(mu, 0.18)), 1.03, 990.0))
                    won, plc = float(pos[i] == 1), float(pos[i] <= k)
                    if won:
                        lo, hi = 1.01, sp * (1.0 + rng.uniform(0, 1.2))
                    else:
                        hi = 1000.0 if rng.random() < 0.85 else sp * rng.uniform(1.0, 1.4)
                        u = rng.uniform(0.1, 0.6) if h["style"] == "lead" else rng.uniform(0.15, 1.0)
                        lo = 1.0 + (sp - 1.0) * u
                        if h["trait"] == "flatterer" and rng.random() < 0.7:
                            lo = rng.uniform(1.3, 1.9)
                        lo = max(1.02, lo)
                joined = (not is_nr) and rng.random() > 0.03
                common = {"country": "uk", "race_date": iso, "race_time": hhmm, "menu_hint": menu,
                          "event_dt": f"{day.day:02d}-{day.month:02d}-{day.year} {hhmm}", "track_bf": "",
                          "selection_id": h["sid"], "selection_name": h["name"],
                          "horse_norm": h["name"].lower().replace(" ", ""), "source_file": "toy", "loaded_at": "toy"}
                bp.append({**common, "id": bid, "market_type": "win", "event_id": eid, "event_name": ename,
                           "win_lose": won, "bsp": sp, "ppwap": pw, "morningwap": am,
                           "ppmax": np.nanmax([sp, pw, am]) * 1.1, "ppmin": np.nanmin([sp, pw, am]) * 0.9,
                           "ipmax": hi, "ipmin": lo, "morning_vol": float(rng.uniform(60, 4000)),
                           "pp_vol": float(rng.uniform(1500, 60000)), "ip_vol": float(rng.uniform(0, 20000)),
                           "race_results_id": rid if joined else np.nan})
                bp.append({**common, "id": bid + 1, "market_type": "place", "event_id": eid + 1,
                           "event_name": "To Be Placed", "win_lose": plc, "bsp": psp,
                           "ppwap": psp * 0.97 if not is_nr else 3.0, "morningwap": 0.0, "ppmax": 0.0, "ppmin": 0.0,
                           "ipmax": 0.0, "ipmin": 0.0, "morning_vol": 0.0, "pp_vol": float(rng.uniform(400, 9000)),
                           "ip_vol": 0.0, "race_results_id": np.nan})
                bid += 2
                if is_nr:
                    continue
                c = COMMENTS[h["style"]][int(rng.integers(2))]
                if h["slow"] and rng.random() < 0.5:
                    c = "slowly away, " + c
                if rtype == "Chase" and h["sloppy"] and rng.random() < 0.75:
                    c += ", mistake 4th" if rng.random() < 0.8 else ", fell 3 out"
                rr.append({"id": rid, "race_date": iso, "race_time": f"{int(hh) - 12}:{mm}", "track": track,
                           "race_name": f"Toy {'Handicap ' if hcap else 'Novices '}{rtype} ({cls})",
                           "race_class": cls, "race_type": rtype, "race_code": rcode, "surface_type": surf,
                           "dist_furlongs": furl, "going_description": going, "number_of_runners": n,
                           "placing_numerical": int(pos[i]), "horse_name": h["name"], "trainer": h["trainer"],
                           "jockey_name": jocks[i], "jockeys_claim": 0, "official_rating": round(70 + 10 * h["a"]),
                           "horse_age": 5, "bfsp": round(float(sp), 2), "bfsp_place": round(float(psp), 2),
                           "bf_plcs_paid": k, "comment": c, "stall": (j + 1) if rtype == "Flat" else 0,
                           "days_since_lr": gap, "career_runs": h["runs"], "headgear": h["headgear"]})
                rid += 1
                h["runs"] += 1
                h["last"] = day
                ran_today.add(h["sid"])
        day += timedelta(days=3 if n_day % 2 == 0 else 7)
        n_day += 1
    return pd.DataFrame(rr), pd.DataFrame(bp)


@pytest.fixture(scope="module")
def raw():
    return toy()


@pytest.fixture(scope="module")
def frames(raw):
    return D.build_frames(*raw)


@pytest.fixture(scope="module")
def built(frames):
    """Every strategy's bets on the toy frames, built once."""
    return {s.id: s.build(frames) for s in S.STRATEGIES}


BY_ID = {s.id: s for s in S.STRATEGIES}
SPEC = [f"S{n:03d}" for n in range(69, 101)]
BENCH = ["BASE-S072", "BASE-S073", "BASE-S074", "BASE-S075"]
MORNING = ["S077", "S084", "S086", "S087", "S090"]                       # entry at morningwap
LIMIT = ["S069", "S070", "S071", "S076", "S078", "S079"]                 # LIMIT_ON_CLOSE: the BSP only fills
SP_MOC = ["S072", "S073", "S074", "S075", "S080", "S081", "S082", "S083", "S085", "S088", "S089", "S091"] + BENCH
INRUN = [f"S{n:03d}" for n in range(92, 101)]
SAME_DAY = ["S069", "S070", "S071", "S072", "S073", "S074", "S075", "S076", "S079", "S080", "S084", "S085", "S086",
            "S089", "S091"] + BENCH                                      # a selection reads the day (and fits to 2024)
ORDER = ["race_date", "race_key", "selection", "side", "market", "entry_type", "entry_price", "stake"]
KEYS = ["race_date", "race_key", "selection", "side"]
D0, D1 = "2025-01-01", "2025-12-31"


def orders(bets: pd.DataFrame, cols, lo: str, hi: str | None = None) -> pd.DataFrame:
    b = core.check_bets(bets)
    b = b[(b["race_date"] >= lo) & (b["race_date"] <= (hi or lo))]
    return b.sort_values(KEYS, kind="stable").reset_index(drop=True)[cols]


def perturb(raw, d0: str, *, results=False, inplay=False, bsp=False, preplay=False, seed=5):
    """The raw tables with the named LATER columns of every day from ``d0`` changed. Which runners hold a BSP, how
    many places a market paid and the sum of 1 / place BSP (K9's check of k) are kept: those are facts of the
    market's terms, not of the result."""
    rr, bp = raw[0].copy(), raw[1].copy()
    rng = np.random.default_rng(seed)
    m = (bp["race_date"] >= d0).to_numpy()
    win, r = m & bp["market_type"].eq("win").to_numpy(), (rr["race_date"] >= d0).to_numpy()

    def shuffled(frame, mask, col, by):
        part = frame.loc[mask, by + [col]]
        return part.groupby(by, sort=False)[col].transform(lambda s: rng.permutation(s.to_numpy())).to_numpy()
    if inplay:
        n = int(win.sum())
        bp.loc[win, "ipmin"] = np.where(rng.random(n) < 0.5, 1.01, 1.0 + rng.uniform(0.0, 30.0, n))
        bp.loc[win, "ipmax"] = np.where(rng.random(n) < 0.5, 1000.0, 1.0 + rng.uniform(0.0, 30.0, n))
        bp.loc[win, "ip_vol"] = rng.uniform(0, 1e5, n)
    if results:
        bp.loc[m, "win_lose"] = shuffled(bp, m, "win_lose", ["event_id"])
        rr.loc[r, "placing_numerical"] = shuffled(rr, r, "placing_numerical", ["race_date", "track", "race_time"])
        rr.loc[r, "comment"] = shuffled(rr, r, "comment", ["race_date", "track", "race_time"])
    if bsp:
        n = int(win.sum())
        bp.loc[win, "bsp"] = np.maximum(1.02, bp.loc[win, "bsp"].to_numpy() * rng.uniform(0.5, 1.8, n))
        pl = m & bp["market_type"].eq("place").to_numpy() & bp["bsp"].notna().to_numpy()
        bp.loc[pl, "bsp"] = shuffled(bp, pl, "bsp", ["event_id"])          # the place SPs dealt again in the race
        rr.loc[r, "bfsp"] = rr.loc[r, "bfsp"].to_numpy() * rng.uniform(0.5, 1.8, int(r.sum()))
        rr.loc[r, "bfsp_place"] = rr.loc[r, "bfsp_place"].to_numpy() * rng.uniform(0.9, 1.1, int(r.sum()))
    if preplay:
        n = int(win.sum())
        bp.loc[win, "ppwap"] = np.maximum(1.02, bp.loc[win, "ppwap"].to_numpy() * rng.uniform(0.5, 1.8, n))
        bp.loc[win, "ppmax"] = bp.loc[win, "ppmax"].to_numpy() * rng.uniform(1.0, 2.0, n)
        bp.loc[win, "ppmin"] = np.maximum(1.01, bp.loc[win, "ppmin"].to_numpy() * rng.uniform(0.5, 1.0, n))
        bp.loc[m, "pp_vol"] = bp.loc[m, "pp_vol"].to_numpy() * rng.uniform(0.2, 5.0, int(m.sum()))
    return rr, bp


def cover(built: dict, ids: list, lo: str = "2025-03-01") -> dict:
    """{day: [ids]}: a few race days from ``lo`` on which, between them, every one of ``ids`` has an order."""
    days = {}
    for i in ids:
        b = core.check_bets(built[i])
        days[i] = set(b.loc[b["race_date"] >= lo, "race_date"])
    assert all(days.values()), f"no order from {lo}: {[i for i, d in days.items() if not d]}"
    left, out = set(ids), {}
    while left:
        best = max(sorted(set().union(*(days[i] for i in left))), key=lambda d: sum(d in days[i] for i in left))
        out[best] = sorted(i for i in left if best in days[i])
        left -= set(out[best])
    return out


def check_unmoved(raw, built, ids, cols, **what):
    """Rebuild the frames with the later columns of every day from d0 changed; the orders must not move. A rule
    that reads only the day is held on the whole of 2025; a rule that also reads earlier days on its first day."""
    same = [i for i in ids if i in SAME_DAY]
    hist = [i for i in ids if i not in SAME_DAY]
    if same:
        fr = D.build_frames(*perturb(raw, D0, **what), place_chance=any(i < "S077" for i in same), styles=False)
        for i in same:
            was, now = orders(built[i], cols, D0, D1), orders(BY_ID[i].build(fr), cols, D0, D1)
            assert len(was) > 0, f"{i}: the toy holds no order in 2025"
            pd.testing.assert_frame_equal(was, now, obj=i)
    for day, group in cover(built, hist).items():
        fr = D.build_frames(*perturb(raw, day, **what), place_chance=False, styles=any(i in INRUN for i in group))
        for i in group:
            was, now = orders(built[i], cols, day), orders(BY_ID[i].build(fr), cols, day)
            assert len(was) > 0
            pd.testing.assert_frame_equal(was, now, obj=f"{i} on {day}")


# ---- the registry and the frames -------------------------------------------------------------------------------

def test_the_registry_holds_the_32_strategies_and_their_benchmarks():
    ids = [s.id for s in S.STRATEGIES]
    assert len(ids) == len(set(ids)) and set(ids) == set(SPEC) | set(BENCH)
    assert all(s.dataset == "a_market" for s in S.STRATEGIES)
    assert {s.id for s in S.STRATEGIES if s.benchmark} == set(BENCH)
    assert {s.id: s.paired for s in S.STRATEGIES if s.paired} == {f"S07{n}": f"BASE-S07{n}" for n in (2, 3, 4, 5)}
    assert {s.id for s in S.STRATEGIES if not s.blind} == {"S069", "S076", "S081", "S082", "S083", "S084"}
    assert {s.id: s.select_from for s in S.STRATEGIES if s.select_from} == {
        "S071": "2021-01-01", "S077": "2022-01-01", "S078": "2022-01-01", "S086": "2022-01-01",
        "S092": "2021-01-01", "S094": "2021-01-01"}
    assert all(BY_ID[i].tested_before for i in SPEC)
    assert {s.id for s in S.STRATEGIES if s.entry == "wap"} == set(MORNING) and BY_ID["S100"].entry == "inplay"
    assert set(MORNING) | set(LIMIT) | set(SP_MOC) | set(INRUN) == set(ids)


def test_the_loader_drops_the_holdout_and_counts_the_races(raw, frames):
    rr, bp = raw
    assert (rr["race_date"] >= core.HOLDOUT_FROM).any() and (bp["race_date"] >= core.HOLDOUT_FROM).any()
    for name in ("win", "pairs", "runs"):
        assert len(frames[name]) and frames[name]["race_date"].max() < core.HOLDOUT_FROM
    assert frames["days"] == sorted(frames["days"]) and frames["days"][-1] < core.HOLDOUT_FROM
    w, p, r = frames["win"], frames["pairs"], frames["runs"]
    assert len(w) == int(((bp["market_type"] == "win") & (bp["race_date"] < core.HOLDOUT_FROM)).sum())
    assert len(r) == int((rr["race_date"] < core.HOLDOUT_FROM).sum()) and r["id"].is_unique
    nr = w[~w["race_whole"]]
    assert len(nr) and (nr["n_sel"] == nr["n_bsp"] + 1).all()              # the toy's one non-runner a race
    assert p["place_whole"].all() and set(p.loc[p["k_ok"], "k"]) == {2.0, 3.0, 4.0}
    ok = p["p_place"].notna()
    assert ok.sum() > 10000
    assert np.allclose(p[ok].groupby("event_id")["p_place"].sum(), p[ok].groupby("event_id")["k"].first(), atol=0.02)
    assert np.allclose(w[w["q_pp"].notna()].groupby("event_id")["q_pp"].sum(), 1.0)
    assert r["bp_id"].notna().mean() > 0.9 and r["p_lead"].notna().all()


def test_the_database_is_read_without_the_holdout_and_a_missing_one_says_no_data_found(raw, tmp_path, monkeypatch):
    import sqlite3
    db = tmp_path / "toy.db"
    con = sqlite3.connect(db)
    raw[0].to_sql("race_results", con, index=False)
    raw[1].to_sql("betfair_prices", con, index=False)
    con.close()
    rr, bp = D.read_tables(db)
    assert len(rr) == int((raw[0]["race_date"] < core.HOLDOUT_FROM).sum()) and rr["race_date"].max() < core.HOLDOUT_FROM
    assert len(bp) == int((raw[1]["race_date"] < core.HOLDOUT_FROM).sum()) and bp["race_date"].max() < core.HOLDOUT_FROM
    assert list(rr.columns) == D.RR_COLS and list(bp.columns) == D.BP_COLS
    monkeypatch.setenv("STRAT100_DB", str(db))
    assert D.db_path() == db
    monkeypatch.setattr(D, "build_frames", lambda a, b: {"rr": len(a), "bp": len(b)})
    from research.strat100 import data_a
    assert data_a.load_a_market() == {"rr": len(rr), "bp": len(bp)}
    monkeypatch.setenv("STRAT100_DB", str(tmp_path / "absent.db"))
    monkeypatch.setattr(D, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(D.NoDataFound, match="no data found"):
        D.db_path()
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    with pytest.raises(D.NoDataFound, match="no data found"):
        D.read_tables(empty)


# ---- every strategy runs and returns a bets frame the core accepts ------------------------------------------------

@pytest.mark.parametrize("sid", SPEC + BENCH)
def test_every_strategy_returns_a_valid_bets_frame(sid, frames, built):
    s, b = BY_ID[sid], built[sid]
    checked = core.check_bets(b, s)                                        # raises on a frame the core would refuse
    assert len(checked) > 0, f"{sid}: the toy frames give it nothing to bet"
    assert checked["unsettled"].sum() == 0 and (~checked["void"]).sum() > 0
    assert not checked.duplicated(["race_date", "race_key", "selection", "side", "market"]).any()
    r = core.score(s, b, frames["days"], draws=200)
    assert r["error"] is None and r["bets_outside_windows"] == 0 and r["all_bets"] == int((~checked["void"]).sum())
    assert set(checked["market"]) == {s.market} and set(checked["race_date"]) <= set(frames["days"])
    live = checked[~checked["void"]]
    assert set(live["side"]) == ({"back"} if sid in ("S092", "S093", "S096") or s.side == "back" else {"lay"})
    if s.select_from:
        assert live["race_date"].min() >= s.select_from
    assert r["flags"] == ("no cuts" if sid in MORNING else "upper bound" if sid in INRUN else "")


def test_no_strategy_changes_the_shared_frames(raw, frames, built):
    again = D.build_frames(*raw, place_chance=False, styles=True)
    for name in ("win", "runs"):
        pd.testing.assert_frame_equal(frames[name], again[name])
    cols = [c for c in frames["pairs"].columns if c != "p_place"]
    pd.testing.assert_frame_equal(frames["pairs"][cols], again["pairs"][cols])


# ---- look-ahead: what an order may not read ----------------------------------------------------------------------

def test_lookahead_the_days_result_and_in_running_prices_never_move_an_order(raw, built):
    """Every strategy: the day's win_lose, placing, comment, ipmin, ipmax and ip_vol changed, the same orders at the
    same prices and stakes (and, for all but the resting lay S100, the same fills)."""
    for group, cols in (([i for i in SPEC + BENCH if i != "S100"], ORDER + ["void"]), (["S100"], ORDER)):
        check_unmoved(raw, built, group, cols, results=True, inplay=True)
    fr = D.build_frames(*perturb(raw, D0, results=True, inplay=True), place_chance=False, styles=False)
    was, now = core.settle(core.check_bets(built["S080"])), core.settle(core.check_bets(S.s080(fr)))
    assert not np.isclose(was["pnl"].sum(), now["pnl"].sum())             # the change is real: the money moved


def test_lookahead_the_bsps_value_never_chooses_an_sp_order(raw, built):
    """Market-on-close and limit-on-close strategies: every BSP's value changed (which runners hold one is kept),
    the same orders; for a limit only the fill may differ."""
    check_unmoved(raw, built, SP_MOC + LIMIT, KEYS, bsp=True)
    fr = D.build_frames(*perturb(raw, D0, bsp=True), place_chance=False, styles=False)
    was, now = core.check_bets(built["S079"]), core.check_bets(S.s079(fr))
    assert (was["void"].to_numpy() != now["void"].to_numpy()).any()         # ... and the fills did move


def test_lookahead_a_morning_order_reads_nothing_after_the_morning(raw, built):
    """Morning entries: ppwap, ppmax, ppmin, pp_vol, the BSP's value, the in-play columns and the result changed,
    the same orders at the same morning price and stake."""
    check_unmoved(raw, built, MORNING, ORDER, preplay=True, bsp=True, inplay=True, results=True)


def test_lookahead_a_fitted_table_or_model_reads_only_the_years_before(raw, frames, built):
    """S086's cells and the steamer's threshold are fitted to the end of 2021, S071's stack on the three years
    before each year: every later column from 2022 on changed, the cells, the threshold and 2022's stack stand."""
    fr = D.build_frames(*perturb(raw, "2022-01-01", preplay=True, bsp=True, inplay=True, results=True),
                        place_chance=False, styles=False)
    pd.testing.assert_frame_equal(orders(built["S086"], ORDER, "2022-01-01", "2026-12-31"),
                                  orders(S.s086(fr), ORDER, "2022-01-01", "2026-12-31"))
    a, b = S._steamer(frames), S._steamer(fr)
    assert a["low"].iloc[0] == b["low"].iloc[0]
    runs = frames["runs"]
    day = runs.loc[runs["race_date"] >= "2022-01-01", "race_date"].min()             # 2022's first race day
    both = a[a["id"].isin(runs.loc[runs["race_date"] == day, "id"])].merge(b, on="id", suffixes=("", "_b"))
    assert len(both) > 20 and np.allclose(both["pred"], both["pred_b"])
    fr2 = D.build_frames(*perturb(raw, "2022-01-01", results=True, inplay=True), styles=False)
    y22 = ("2022-01-01", "2022-12-31")
    pd.testing.assert_frame_equal(orders(built["S071"], ORDER + ["void"], *y22),
                                  orders(S.s071(fr2), ORDER + ["void"], *y22))


# ---- the fills, limits and stakes -----------------------------------------------------------------------------------

def rows_of(bets, frame, key="selection_id"):
    """The checked bets beside the frame rows they came from (race key and selection)."""
    b = core.check_bets(bets)
    f = frame.assign(race_key=frame["rk"], selection=frame[key].astype("Int64").astype(str))
    out = b.merge(f, on=["race_key", "selection"], how="left", suffixes=("", "_f"))
    assert len(out) == len(b)
    return out


def test_k8_a_resting_order_fills_only_a_full_tick_through_its_price():
    assert S.lay_target(5.0, 0.5) == 3.0 and S.back_target(5.0, 1.5) == 7.0
    assert S.lay_filled([2.98, 3.0, 2.99, 1.0, np.nan, 0.0], 3.0).tolist() == [True, False, False, False, False, False]
    assert S.back_filled([7.2, 7.0, 7.1, np.nan, 1000.0], 7.0).tolist() == [True, False, False, False, True]
    assert S.lay_filled([1.99, 2.0, 1.01], 2.0).tolist() == [True, False, True]          # S100's 1 < ipmin <= 1.99
    assert not S.lay_filled([1.01], 1.01)[0] and not S.back_filled([1000.0], 1000.0)[0]   # nothing past the ladder
    x = pd.DataFrame({"bsp": [5.0, 5.0, 9.0], "ipmin": [1.01, 2.98, 1.9], "ipmax": [5.0, 1000.0, 1000.0],
                      "win_lose": [1.0, 0.0, 0.0]})
    f = S.inrun_flags(x)
    assert f["dob"].tolist() == [1, 1, 1] and f["d15"].tolist() == [0, 1, 1] and f["short2"].tolist() == [0, 0, 1]


def test_harville_is_exact_and_henery_matches_its_win_chances():
    rng = np.random.default_rng(1)
    for n, k in ((5, 2), (8, 3), (12, 3)):
        q = rng.dirichlet(np.ones(n))
        assert np.allclose(S.harville_place(q, k), core.harville_place(q, k), atol=1e-9)
        assert np.isclose(S.harville_place(q, k).sum(), k)
    q = rng.dirichlet(np.ones(16) * 2)
    assert np.isclose(S.harville_place(q, 4).sum(), 4.0)
    assert np.abs(S.harville_place(q, 4) - core.harville_place(q, 4)).max() < 0.02        # the core simulates k = 4
    q = np.array([0.5, 0.2, 0.12, 0.08, 0.05, 0.03, 0.015, 0.005])
    theta = S.henery_theta(q)
    z = np.random.default_rng(99).standard_normal((400000, len(q))) + theta
    assert np.abs(np.bincount(z.argmin(axis=1), minlength=len(q)) / len(z) - q).max() < 0.003
    hen, har = S.henery_place(q, 3), S.harville_place(q, 3)
    assert np.isclose(hen.sum(), 3.0) and np.abs(hen - core.henery_place(q, 3)).max() < 0.03
    assert hen[0] < har[0] and hen[-1] > har[-1]            # the favourite places less often, the outsider more
    assert S.henery_place(q[:3] / q[:3].sum(), 3).tolist() == [1.0, 1.0, 1.0]


def test_a_window_sum_reads_only_the_days_before():
    key, day = ["a", "a", "a", "b", "a", ""], [10, 12, 15, 12, 20, 12]
    got = S._window(key, day, {"n": np.ones(6), "v": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]},
                    ["a", "a", "a", "b", "c", "", "a"], [15, 16, 13, 12, 99, 99, 743], back=5)
    assert got["v"].tolist() == [3.0, 6.0, 3.0, 0.0, 0.0, 0.0, 0.0] and got["n"].tolist() == [2, 2, 2, 0, 0, 0, 0]
    assert S._window(key, day, {"v": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]}, ["a"], [743], back=730)["v"].tolist() == [20.0]


def test_the_previous_run_is_the_latest_row_of_an_earlier_day(frames):
    r, p = frames["runs"], S._prev(frames)
    assert p.index.equals(r.index)
    h = r[r["horse_name"] == r["horse_name"].iloc[500]]
    ph = p.loc[h.index]
    assert pd.isna(ph["prev_race_date"].iloc[0]) and (ph["prev_race_date"].iloc[1:] < h["race_date"].iloc[1:]).all()
    assert ph["prev_race_date"].iloc[1:].tolist() == h["race_date"].iloc[:-1].tolist()
    assert ph["prev_trainer"].iloc[1:].tolist() == h["trainer"].iloc[:-1].tolist()


def test_the_limit_on_close_fills_only_where_the_sp_reaches_the_limit(frames, built):
    x = rows_of(built["S069"], frames["pairs"])
    limit = 1.0 + (1.05 / x["p_place"] - 1.0) / 0.98
    assert (x["ppwap"] <= 5.0).all() and (x["p_pp_vol"] >= 1000).all() and x["k_ok"].all()
    assert ((x["p_bsp"] >= limit) == ~x["void"]).all() and x["void"].any() and (~x["void"]).any()
    live = x[~x["void"]]
    assert (live["entry_price"] == live["p_bsp"]).all() and (live["stake"] == 20.0).all()
    assert (live["won"] == live["p_win_lose"]).all() and (live["exit"] == "result").all()
    x = rows_of(built["S070"], frames["pairs"])
    assert x["ppwap"].between(5.0, 21.0, inclusive="right").all() and (x["stake"] <= 0.02 * x["p_pp_vol"] + 1e-9).all()
    assert ((x["p_bsp"] >= 1.0 + (1.10 / x["p_place"] - 1.0) / 0.98) == ~x["void"]).all()
    x = rows_of(built["S076"], frames["pairs"])
    m = np.minimum(10.0, 1.0 + ((1.0 - x["p_place"]) * 0.98 - 0.05) / x["p_place"])
    assert (x["ppwap"] >= 50.0).all() and ((x["p_bsp"] <= m) == ~x["void"]).all()
    live = core.settle(x[~x["void"]])
    assert np.allclose(live["risk"], 20.0) and (live["entry_price"] <= 10.0).all()        # liability GBP20 a fill
    x = rows_of(built["S079"], frames["win"])
    limit = np.asarray(core.tick_floor(0.8 * x["morningwap"].to_numpy()))
    assert (x["morningwap"] <= 8.0).all() and x["race_whole"].all() and (x["n_bsp"] == x["rr_runners"]).all()
    assert ((x["bsp"] <= limit) == ~x["void"]).all() and np.allclose(core.settle(x[~x["void"]])["risk"], 100.0)
    x = rows_of(built["S078"], frames["win"])
    assert ((x["bsp"] <= 8.0) == ~x["void"]).all() and np.allclose(core.settle(x[~x["void"]])["risk"], 100.0)


def test_the_stacked_chance_is_a_chance_and_sums_to_the_places(frames, built):
    st = S._stack(frames)
    assert st["year"].min() == 2021 and st["p_stack"].between(0, 1).all()
    assert np.allclose(st.groupby("event_id")["p_stack"].sum(), st.groupby("event_id")["k"].first(), atol=0.05)
    x = rows_of(built["S071"], st)
    assert ((x["p_bsp"] >= 1.0 + (1.05 / x["p_stack"] - 1.0) / 0.98) == ~x["void"]).all()
    assert (x["ppwap"] <= 21.0).all() and (x["p_pp_vol"] >= 500).all() and (x["stake"] <= 10.0).all()


def test_the_place_shape_rules_pick_what_they_say(frames, built):
    p = frames["pairs"]
    fav = p[p["rank_pp"] == 1].groupby("event_id")["ppwap"].min()
    x = rows_of(built["S072"], p)
    assert x["n_win_bsp"].between(8, 12).all() and (x["k"] == 3).all() and x["rank_pp"].isin([2, 3]).all()
    assert x["ppwap"].between(3.5, 10.0).all() and (x["event_id"].map(fav) <= 2.0).all() and (x["stake"] == 20).all()
    assert set(x["race_key"] + x["selection"]) < set((b := rows_of(built["BASE-S072"], p))["race_key"] + b["selection"])
    x, b = rows_of(built["S073"], p), rows_of(built["BASE-S073"], p)
    sat = pd.to_datetime(x["race_date"]).dt.day_name().eq("Saturday")
    assert sat.all() and not pd.to_datetime(b["race_date"]).dt.day_name().eq("Saturday").any()
    for y in (x, b):
        assert (y["n_win_bsp"] >= 16).all() and (y["k"] == 4).all() and y["ppwap"].between(6.0, 21.0).all()
    x, b = rows_of(built["S074"], p), rows_of(built["BASE-S074"], p)
    assert ((x["n_win_bsp"] == 7) & (x["k"] == 3)).all() and (x["rank_pp"] >= 3).all() and (x["ppwap"] <= 21).all()
    assert not ((b["n_win_bsp"] <= 7) & (b["k"] == 3)).any() and (b["rank_pp"] >= 3).all()
    x = rows_of(built["S075"], p)
    assert (np.log(x["morningwap"] / x["ppwap"]) >= 0.15).all() and x["ppwap"].between(2, 15).all()
    assert (x["morning_vol"] >= 200).all() and (x["p_pp_vol"] >= 1000).all() and len(built["BASE-S075"]) > len(x)
    for sid in ("S072", "S073", "S074", "S075"):
        live = core.check_bets(built[sid])
        assert (live["entry_type"] == "sp").all() and (live["market"] == "place").all() and not live["void"].any()


def test_a_morning_entry_is_at_the_morning_wap_within_a_tenth_of_its_volume_in_whole_races(frames, built):
    r = frames["runs"]
    for sid in MORNING:
        x = rows_of(built[sid], r[r["bp_id"].notna()])
        assert (x["entry_price"] == x["morningwap"]).all() and (x["entry_type"] == "wap").all(), sid
        assert x["race_whole"].eq(True).all() and (x["stake"] <= 0.10 * x["morning_vol"] + 1e-9).all(), sid
        live = x[~x["void"]]
        assert (live["stake"] >= 2.0).all() and ((live["exit"] == "result") == (sid == "S084")).all(), sid
        if sid != "S084":
            assert (live["exit_price"] == live["bsp"]).all(), sid
    x = core.settle(rows_of(built["S077"], r[r["bp_id"].notna()]))
    assert (x["morningwap"] <= 8.0).all() and (x["morning_vol"] >= 200).all() and (x["risk"] <= 100.0 + 1e-6).all()
    assert np.allclose(x["pnl"], x["stake"] * (1.0 - x["morningwap"] / x["bsp"]))        # a lay closed at the BSP
    for sid in ("S086", "S087", "S090"):
        x = rows_of(built[sid], r[r["bp_id"].notna()])
        live = x[~x["void"]]
        assert (live["stake"] <= 100.0).all() and (live["stake"] * (live["morningwap"] - 1.0) <= 50.0 + 1e-6).all()
    x = rows_of(built["S084"], r[r["bp_id"].notna()])
    low = r[(r["code"] == "chase") & r["hcap"]].groupby(["race_date", "track", "race_time"])["official_rating"].min()
    assert (x["code_f"] == "chase").all() and x["hcap"].all() and (x["morning_vol"] >= 50).all()
    assert (x["official_rating"] == pd.MultiIndex.from_frame(x[["race_date", "track", "race_time"]]).map(low)).all()


def test_the_sp_segments_pick_what_they_say(frames, built):
    r, p = frames["runs"], S._prev(frames)
    full = r.join(p)
    x = rows_of(built["S081"], full[full["bp_id"].notna()])
    live = core.settle(x)
    assert (x["code_f"] == "hurdle").all() and (x["prev_code"] == "chase").all() and (x["pp_vol"] >= 1000).all()
    assert live["risk"].between(10.0 - 1e-6, 50.0 + 1e-6).all() and (live["risk"] <= 0.02 * x["pp_vol"] + 1e-6).all()
    x = rows_of(built["S082"], full[full["bp_id"].notna()])
    assert (x["headgear"] == "").all() and (x["prev_headgear"] != "").all() and (x["stake"] <= 20).all()
    assert (x["stake"] <= 0.01 * x["pp_vol"] + 1e-9).all()
    x = rows_of(built["S083"], full[full["bp_id"].notna()])
    assert (x["trainer"] != x["prev_trainer"]).all() and x["prev_trainer"].notna().all()
    x = rows_of(built["S088"], full[full["bp_id"].notna()])
    gap = (x["date"] - x["prev_date"]).dt.days
    assert gap.between(1, 90).all() and (x["prev_code"] == x["code_f"]).all() and (x["prev_bsp"] >= 6.0).all()
    assert (x["prev_ipmin"] <= 2.0).all() and (x["prev_win_lose"] == 0).all()
    x = rows_of(built["S080"], frames["win"])
    assert (np.log(x["ppwap"] / x["morningwap"]) >= 0.18).all() and x["ppwap"].between(3, 20).all()
    assert (x["pp_vol"] >= 2000).all() and (x["stake"] == 10.0).all() and (x["entry_price"] == x["bsp"]).all()
    x = core.settle(rows_of(built["S091"], frames["win"]))
    assert (x["morningwap"] >= 80).all() and (x["pp_vol"] >= 200).all() and np.allclose(x["risk"], 10.0)
    assert S._surnames(pd.Series(["Henderson, N J", "Dan Skelton", "Charlie & Mark Johnston", None])) == [
        frozenset({"henderson"}), frozenset({"skelton"}), frozenset({"johnston"}), frozenset()]


def test_the_second_string_and_the_last_race_favourite(frames, built):
    r, w = frames["runs"], frames["win"]
    x = rows_of(built["S085"], r[r["bp_id"].notna()])
    race = ["race_date", "track", "race_time", "trainer"]
    mates = r.groupby(race)["morningwap"].agg(["min", "size", lambda s: s.nsmallest(2).iloc[-1]])
    got = mates.reindex(pd.MultiIndex.from_frame(x[race]))
    assert (got["size"].to_numpy() >= 2).all() and (got["min"].to_numpy() <= 6.0).all()
    assert np.allclose(got.iloc[:, 2].to_numpy(), x["morningwap"]) and (x["morningwap"] <= 26.0).all()
    assert (x["morningwap"] >= got["min"].to_numpy()).all() and not x.duplicated(race).any()
    x = rows_of(built["S089"], w)
    last = w.groupby(["country", "menu_hint", "race_date"])["event_ts"].max()
    assert (x["event_ts"] == pd.MultiIndex.from_frame(x[["country", "menu_hint", "race_date"]]).map(last)).all()
    assert (x["morningwap"] == x["event_id"].map(w.groupby("event_id")["morningwap"].min())).all()
    assert (x["morningwap"] <= 5.0).all() and (x["n_sel"] >= 5).all() and not x.duplicated("race_key").any()


def test_the_yard_gate_voids_the_strategy(frames, monkeypatch):
    y = S._yard(frames)
    corr, n = S._yard_persistence(y)
    assert n >= 30 and corr >= 0.2                          # the toy's first yard steams every year
    assert (y["yard_n"] >= 0).all() and y.loc[y["yard_n"] >= 50, "yard_score"].notna().all()
    monkeypatch.setattr(S, "YARD_MIN_CORR", 2.0)
    assert len(S.s087(frames)) == 0


def test_the_in_running_trades_close_at_the_target_or_stand_to_the_result(frames, built):
    w = frames["win"]
    for sid, f in (("S092", 0.5), ("S093", 0.5), ("S096", 0.6)):
        x = core.settle(rows_of(built[sid], w))
        target = S.lay_target(x["bsp"], f)
        hit = S.lay_filled(x["ipmin"], target)
        assert (x["side"] == "back").all() and (x["entry_price"] == x["bsp"]).all(), sid
        assert ((x["exit"] == "inplay") == hit).all() and hit.any() and (~hit).any(), sid
        assert (x["stake"] <= np.minimum(20.0, 0.005 * x["pp_vol"]) + 1e-9).all(), sid
        live = ~x["void"]
        assert np.allclose(x.loc[hit & live, "pnl"], (x["stake"] * (x["bsp"] / target - 1.0))[hit & live]), sid
        held = ~hit & live
        assert np.allclose(x.loc[held, "pnl"], np.where(x["win_lose"] == 1, x["stake"] * (x["bsp"] - 1), -x["stake"])[held])
    for sid in ("S094", "S095", "S097", "S098", "S099"):
        x = core.settle(rows_of(built[sid], w))
        target = S.back_target(x["bsp"], 1.5)
        hit = S.back_filled(x["ipmax"], target)
        assert (x["side"] == "lay").all() and ((x["exit"] == "inplay") == hit).all() and hit.any(), sid
        live = ~x["void"]
        assert (x.loc[live, "risk"] <= 40.0 + 1e-6).all() and (x["stake"] <= 0.005 * x["pp_vol"] + 1e-9).all(), sid
        assert np.allclose(x.loc[hit & live, "pnl"], (x["stake"] * (1.0 - x["bsp"] / target))[hit & live]), sid
    bands = {"S092": (2.5, 20), "S093": (3, 15), "S094": (2, 12), "S095": (2.5, 10), "S096": (2.5, 12),
             "S097": (2.5, 10), "S098": (3, 12), "S099": (2, 10), "S100": (4, 15)}
    for sid, (lo, hi) in bands.items():
        assert rows_of(built[sid], w)["bsp"].between(lo, hi).all(), sid


def test_the_habits_are_read_from_earlier_recorded_runs(frames, built):
    w = frames["win"].join(S._history(frames), on="bp_id")
    x = rows_of(built["S093"], w)
    assert (x["n8"] >= 6).all() and (x["dob8"] >= 0.625).all()
    x = rows_of(built["S095"], w)
    assert (x["n10"] >= 6).all() and (x["d15_10"] >= 0.8).all() and (x["frame_n10"] >= 3).all()
    assert (x["frame_d15"] / x["frame_n10"] >= 0.8).all()
    x = core.settle(rows_of(built["S100"], w))
    assert (x["n8"] >= 6).all() and (x["short2_n8"] >= 3).all() and (x["wins_n8"] <= 1).all()
    assert (x["entry_price"] == 2.0).all() and (((x["ipmin"] > 1) & (x["ipmin"] <= 1.99)) == ~x["void"]).all()
    live = x[~x["void"]]
    assert np.allclose(live["pnl"], np.where(live["win_lose"] == 1, -20.0, 20.0)) and np.allclose(live["risk"], 20.0)
    # one horse by hand: its last 8 recorded runs before a late race
    h = w[(w["horse_norm"] == w["horse_norm"].iloc[2000]) & w["bsp"].notna()].sort_values("event_ts")
    fl = S.inrun_flags(h)
    row, prev = h.iloc[-1], fl.iloc[-9:-1]
    gap = (pd.Timestamp(row["race_date"]) - pd.to_datetime(h["race_date"].iloc[-9:-1])).dt.days
    assert row["n8"] == int((gap <= 730).sum()) and np.isclose(row["dob8"], prev["dob"][(gap <= 730).to_numpy()].mean())
    assert np.isclose(row["h_dob_car"], fl["dob"].iloc[:-1].mean()) and row["h_n"] == len(h) - 1


def test_the_style_and_comment_rules(frames, built):
    r = frames["runs"]
    x = rows_of(built["S096"], r[r["bp_id"].notna()])
    assert x["code_f"].isin(["flat", "aw"]).all() and (x["dist_furlongs"] <= 8).all() and (x["shape_bin"] == 1).all()
    assert (x["lead_rank"] == 1).all() and (x["p_lead"] >= 0.45).all() and (x["style_n_eff"] >= 4).all()
    assert x["number_of_runners"].between(5, 12).all() and not x.duplicated("race_key").any()
    x = rows_of(built["S098"], r[r["bp_id"].notna()])
    assert (x["dist_furlongs"] >= 7).all() and (x["number_of_runners"] >= 10).all() and (x["shape_bin"] == 2).all()
    assert (x["p_rear"] >= 0.5).all() and (x["style_n_eff"] >= 4).all()
    cs = S._comment_shares(frames)
    x = rows_of(built["S099"], r.join(cs)[r["bp_id"].notna()])
    assert (x["code_f"] == "chase").all() and (x["chase_n"] >= 4).all() and (x["chase_err"] >= 0.5).all()
    import re
    hit = [bool(re.search(S.MISTAKE_RX, c)) for c in (
        "mistake 4th", "mistakes", "blundered 3 out", "not fluent", "less than fluent 2nd", "hit 5th", "pecked 2 out",
        "jumped left throughout", "jumped slowly 1st", "clouted last", "hit rail", "unmistakeable", "led, jumped well")]
    assert hit == [True] * 10 + [False] * 3
    # a sloppy jumper by hand: the share is over its EARLIER chases only
    h = r[(r["horse_name"] == x["horse_name"].iloc[-1]) & (r["code"] == "chase")]
    flag = h["comment"].str.contains(S.MISTAKE_RX, regex=True) | h["comment"].str.contains(r"\bfell\b")
    assert np.isclose(cs.loc[h.index[-1], "chase_err"], flag.iloc[-9:-1].mean()) and cs.loc[h.index[0], "chase_n"] == 0


def test_the_patient_riders_are_the_top_fifth_on_the_day(frames, built):
    r = frames["runs"]
    x = rows_of(built["S097"], r[r["bp_id"].notna()])
    pr = r[r["bp_id"].notna() & r["bsp"].notna()]
    fl = S.inrun_flags(pr)
    day = x["race_date"].iloc[-1]
    lo = (pd.Timestamp(day) - pd.Timedelta(days=730)).date().isoformat()
    win = pr[(fl["won"] == 1) & (pr["race_date"] < day) & (pr["race_date"] >= lo)]
    share = (fl.loc[win.index, "d15"]).groupby(S._name(win["jockey_name"])).agg(["mean", "size"])
    share = share[share["size"] >= 40]
    top = set(share.index[share["mean"] >= share["mean"].quantile(0.8)])
    assert set(S._name(x.loc[x["race_date"] == day, "jockey_name"])) <= top and len(top) >= 1
    assert set(S._name(pr.loc[(pr["race_date"] == day) & pr["bsp"].between(2.5, 10), "jockey_name"])) & top == set(
        S._name(x.loc[x["race_date"] == day, "jockey_name"]))


def test_no_data_gives_no_bets_not_a_crash(frames):
    empty = {k: (v.iloc[:0] if isinstance(v, pd.DataFrame) else []) for k, v in frames.items()}
    for s in S.STRATEGIES:
        b = s.build(empty)
        assert len(core.check_bets(b, s)) == 0, s.id
        assert core.score(s, b, [], draws=50)["verdict"] == "not read: too few", s.id
