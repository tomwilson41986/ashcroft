"""The a_model half of the 100-strategy harness on TOY data (no real figure is read or made here).

A toy database with the data map's schema (race_results, betfair_prices) and two toy out-of-sample files are built in
a temporary folder; data_a.load_a_model() reads them as it reads the real ones (scripts/clv_betfair.load, whole races,
the walk-forward closing model, the steamer fit), with rows dated in the locked holdout planted to prove they are
dropped. Then: every strategy returns a valid bets frame the core can score; the look-ahead tests (a later column
changed must not change a selection); the arithmetic of each rule on hand-made rows.
"""
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.strat100 import core, data_a, run
from research.strat100 import strategies_a_model as A

ROOT = Path(__file__).resolve().parents[1]
BY_ID = {s.id: s for s in A.STRATEGIES}
SPEC_IDS = [f"S{n:03d}" for n in range(53, 69)]
LATER = ["bsp", "ppwap", "ppmax", "ppmin", "pp_vol", "ipmax", "ipmin", "ip_vol", "win_lose", "won", "bfsp"]
SEL = ["race_date", "race_key", "selection", "side", "stake", "entry_price"]
EM_DASH = chr(8212)

RR_COLS = ["id", "race_date", "race_time", "track", "race_name", "race_restrictions_age", "race_class", "major",
           "race_distance", "prize_money", "going_description", "number_of_runners", "place", "distbt", "horse_name",
           "stall", "trainer", "horse_age", "jockey_name", "jockeys_claim", "pounds", "odds", "fav",
           "official_rating", "comptime", "comptime_numeric", "total_dst_bt", "median_or", "dist_furlongs",
           "placing_numerical", "race_code", "bfsp", "bfsp_place", "plcs_paid", "bf_plcs_paid", "yards", "rail_move",
           "race_type", "comment", "card_no", "stall_positioning", "track_direction", "headgear", "days_since_lr",
           "career_runs", "stallion", "surface_type", "horse_prizewin", "horse_sex", "dam", "dam_stallion",
           "max_or_in_race", "created_at"]
BP_COLS = ["id", "country", "market_type", "race_date", "race_time", "event_id", "menu_hint", "event_name",
           "event_dt", "track_bf", "selection_id", "selection_name", "horse_norm", "win_lose", "bsp", "ppwap",
           "morningwap", "ppmax", "ppmin", "ipmax", "ipmin", "morning_vol", "pp_vol", "ip_vol", "race_results_id",
           "source_file", "loaded_at"]
#: track, country, race_code, surface_type, race_type, race_class, race_name, time, field
MEETINGS = [("Kempton", "uk", "All Weather", "Polytrack", "Handicap", "Class 5", "Racing League Handicap", "2.30", 6),
            ("Cheltenham", "uk", "Chase", "Turf", "Handicap Chase", "Class 2", "Big Handicap Chase", "3.05", 6),
            ("Naas", "ire", "Hurdle", "Turf", "Maiden Hurdle", None, "Naas Maiden Hurdle", "3.40", 8),
            ("Ascot", "uk", "Flat", "Turf", "Novice Stakes", "Class 4", "EBF Novice Stakes", "4.15", 5),
            ("Warwick", "uk", "NH Flat", "Turf", "NH Flat", "Class 5", "Standard Open NH Flat Race", "4.50", 6)]
HIST_DAYS = ["2024-10-05", "2024-10-19", "2024-11-02", "2024-11-16"]
DEV_DAYS = ["2024-12-28", "2024-12-30"] + [f"{m}-{d:02d}" for m in (
    "2025-01", "2025-02", "2025-03", "2025-04", "2025-10", "2025-11", "2025-12", "2026-01", "2026-02")
    for d in (3, 9, 15, 21, 27)]
HOLD_DAYS = ["2026-03-28", "2026-04-02"]              # the locked holdout: planted, must never come out of the loader
NON_RUNNER_RACE = ("2025-02-09", "Kempton")           # a Betfair selection with no BSP: K6 drops the race's bets
BROKEN_RACE = ("2025-03-15", "Ascot")                 # a runner with no morning price: not a whole race


def softmax(u):
    e = np.exp(u - u.max())
    return e / e.sum()


def make_toy(folder: Path, seed: int = 11) -> dict:
    """The toy database and out-of-sample files. Prices are random draws; nothing here is a real figure."""
    rng = np.random.default_rng(seed)
    rr, bp, oos, oos_st = [], [], [], []
    rid, bid, eid = 0, 0, 100000
    months = sorted({d[:7] for d in DEV_DAYS + HOLD_DAYS})
    for day in HIST_DAYS + DEV_DAYS + HOLD_DAYS:
        pool = rng.permutation(150)
        used = 0
        dd, mm, yy = day[8:], day[5:7], day[:4]
        for track, country, code, surface, rtype, rclass, rname, rtime, n in MEETINGS:
            horses = pool[used:used + n]
            used += n
            z = rng.normal(0, 1, n)
            model, market = z + rng.normal(0, 0.45, n), z + rng.normal(0, 0.45, n)
            if track == "Ascot" and int(dd) % 2 == 1:                  # a favourite the model does not rate
                model[market.argmax()] = model.min() - 0.6
            close = 0.6 * market + 0.4 * model + rng.normal(0, 0.08, n)
            q_model, q_mkt, q_close = softmax(model), softmax(market), softmax(close)
            q_st = softmax(model + rng.normal(0, 0.10, n))
            order = rng.choice(n, size=n, replace=False, p=q_close)   # the finishing order
            pos = np.empty(n, int)
            pos[order] = np.arange(1, n + 1)
            k = 2 if n < 8 else 3
            p_pl = core.place_prob(q_close, k) * np.exp(rng.normal(0, 0.05, n))
            p_pl = np.clip(p_pl * k / p_pl.sum(), 0.02, 0.97)
            p_pl = p_pl * k / p_pl.sum()
            win_eid, place_eid = eid, eid + 1
            eid += 2
            hh, mi = rtime.split(".")
            event_dt = f"{dd}-{mm}-{yy} {int(hh) + 12}:{mi}"
            hint = f"{track} {int(dd)} {mm}"
            for i in range(n):
                rid += 1
                name = f"Horse {horses[i]:03d}"
                mw, bsp, pred = 1 / q_mkt[i], 1 / q_close[i], 1 / q_model[i]
                vol = float(np.exp(rng.normal(6.5, 1.0)))
                if track == "Ascot" and int(dd) % 2 == 1 and i == market.argmax():
                    vol = 2500.0
                if (day, track) == BROKEN_RACE and i == 0:
                    mw = 0.0
                won = int(pos[i] == 1)
                ppwap = float(np.sqrt(max(mw, 1.01) * bsp) * np.exp(rng.normal(0, 0.02)))
                ipmin = 1.01 if won else max(1.01, bsp * rng.uniform(0.3, 1.0))
                ipmax = bsp * rng.uniform(1.0, 1.5) if won else 1000.0
                row = dict.fromkeys(RR_COLS)
                row.update(id=rid, race_date=day, race_time=rtime, track=track, race_name=rname, race_class=rclass,
                           number_of_runners=n, horse_name=name, trainer=f"Trainer {horses[i] % 9}",
                           jockey_name=f"Jockey {horses[i] % 11}", jockeys_claim=0, horse_age=4 + horses[i] % 5,
                           official_rating=60 + int(horses[i] % 40), placing_numerical=int(pos[i]), race_code=code,
                           bfsp=bsp, bfsp_place=1 / p_pl[i], plcs_paid=k, bf_plcs_paid=k, race_type=rtype,
                           headgear=None, days_since_lr=int(rng.integers(5, 60)), career_runs=int(rng.integers(0, 30)),
                           surface_type=surface, odds=bsp, dist_furlongs=8.0, going_description="Good")
                rr.append(row)
                for mtype, e, b_, w_, wl in (("win", win_eid, bsp, ppwap, won),
                                             ("place", place_eid, 1 / p_pl[i], 1 / p_pl[i] * 1.01, int(pos[i] <= k))):
                    bid += 1
                    brow = dict.fromkeys(BP_COLS)
                    brow.update(id=bid, country=country, market_type=mtype, race_date=day, race_time=rtime, event_id=e,
                                menu_hint=hint, event_name=rname, event_dt=event_dt, track_bf=track.lower(),
                                selection_id=5000 + int(horses[i]), selection_name=name,
                                horse_norm=name.lower().replace(" ", ""), win_lose=float(wl), bsp=b_, ppwap=w_,
                                morningwap=mw if mtype == "win" else w_, ppmax=max(mw, bsp) * 1.1,
                                ppmin=max(1.01, min(max(mw, 1.01), bsp) * 0.9), ipmax=ipmax, ipmin=ipmin,
                                morning_vol=vol if mtype == "win" else vol / 5, pp_vol=vol * 10,
                                ip_vol=vol * 3, race_results_id=rid, source_file="toy.csv", loaded_at="toy")
                    bp.append(brow)
                if day not in HIST_DAYS:
                    o = {"race_date": day, "race_time": rtime, "track": track, "horse_name": name,
                         "predicted_bfsp": pred, "bfsp": bsp, "overlay_pct": 100 * (bsp / pred - 1),
                         "predicted_win_prob_norm": q_model[i], "placing_numerical": int(pos[i]), "won": won,
                         "fold_idx": months.index(day[:7]) // 3, "predicted_bfsp_raw": pred, "race_type": rtype,
                         "race_code": code, "surface_type": surface, "going_description": "Good",
                         "dist_furlongs": 8.0, "number_of_runners": n, "race_class": rclass}
                    oos.append(o)
                    oos_st.append({**o, "predicted_bfsp": 1 / q_st[i]})
            if (day, track) == NON_RUNNER_RACE:                           # withdrawn after the morning: no BSP
                bid += 1
                brow = dict.fromkeys(BP_COLS)
                brow.update(id=bid, country=country, market_type="win", race_date=day, race_time=rtime,
                            event_id=win_eid, menu_hint=hint, event_name=rname, event_dt=event_dt, selection_id=9999,
                            selection_name="Non Runner", win_lose=0.0, bsp=None, ppwap=8.0, morningwap=8.0,
                            ppmax=9.0, ppmin=7.0, ipmax=0.0, ipmin=0.0, morning_vol=300.0, pp_vol=900.0, ip_vol=0.0,
                            race_results_id=None, source_file="toy.csv", loaded_at="toy")
                bp.append(brow)
    db = folder / "toy.db"
    conn = sqlite3.connect(db)
    pd.DataFrame(rr, columns=RR_COLS).to_sql("race_results", conn, index=False)
    pd.DataFrame(bp, columns=BP_COLS).to_sql("betfair_prices", conn, index=False)
    conn.close()
    oos_dir = folder / "oos"
    oos_dir.mkdir()
    pd.DataFrame(oos).to_csv(oos_dir / "oos_gate_stn.csv", index=False)
    pd.DataFrame(oos_st).to_csv(oos_dir / "oos_gate_st.csv", index=False)
    return {"db": db, "oos_dir": oos_dir, "work": folder / "work"}


def _no_network(*a, **k):
    raise AssertionError("the tests never reach GitHub")


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    return make_toy(tmp_path_factory.mktemp("strat100_a_model"))


@pytest.fixture(scope="module")
def frames(toy):
    """load_a_model() itself, on the toy database and files (the steamer fitted on the toy history)."""
    mp = pytest.MonkeyPatch()
    mp.setenv("STRAT100_DB", str(toy["db"]))
    mp.setenv("STRAT100_OOS_DIR", str(toy["oos_dir"]))
    mp.setenv("STRAT100_WORK", str(toy["work"]))
    mp.delenv("STRAT100_STEAMER", raising=False)
    mp.setattr(data_a, "MIN_TRAIN_RACES", 5)
    mp.setattr(data_a, "_artifact_zips", _no_network)
    try:
        return data_a.load_a_model()
    finally:
        mp.undo()


def sel(bets: pd.DataFrame) -> pd.DataFrame:
    return bets[SEL].sort_values(["race_date", "race_key", "selection"]).reset_index(drop=True)


def build(sid: str, fr) -> tuple[pd.DataFrame, list]:
    out = BY_ID[sid].build(fr)
    assert isinstance(out, tuple) and len(out) == 2, "every a_model strategy returns (bets, its days)"
    return out


def with_later_changed(fr: dict, rows=None, seed: int = 3) -> dict:
    """The frames with every later column of ``rows`` (default all) replaced: another BSP, another pre-play record,
    another in-play range, the result turned over. What the morning knew is left as it was."""
    r = fr["runners"].copy()
    m = np.ones(len(r), bool) if rows is None else np.asarray(rows, bool)
    rng = np.random.default_rng(seed)
    n = int(m.sum())
    r.loc[m, "bsp"] = np.maximum(1.02, r.loc[m, "bsp"].to_numpy(float) * rng.uniform(0.4, 2.5, n))
    r.loc[m, "bfsp"] = r.loc[m, "bsp"]
    r.loc[m, "ppwap"] = np.maximum(1.02, r.loc[m, "ppwap"].to_numpy(float) * rng.uniform(0.4, 2.5, n))
    r.loc[m, "ppmax"], r.loc[m, "ppmin"], r.loc[m, "pp_vol"] = 999.0, 1.01, 7.0
    r.loc[m, "ipmax"], r.loc[m, "ipmin"], r.loc[m, "ip_vol"] = 1000.0, 1.01, 1.0
    r.loc[m, "win_lose"] = 1.0 - r.loc[m, "win_lose"].to_numpy(float)
    r.loc[m, "won"] = ~r.loc[m, "won"].astype(bool)
    return {**fr, "runners": r}


def mini(rows: list, **meta) -> dict:
    """Hand-made runners (one dict a runner; a race by default 'r1' on 2025-11-03) as the loader's frame."""
    base = {"race_date": "2025-11-03", "race": "r1", "track": "Toy", "race_time": "2.30", "horse_name": "a",
            "month": "2025-11", "n_field": 6, "predicted_bfsp": 4.0, "morningwap": 5.0, "morning_vol": 1000.0,
            "bsp": 4.0, "ev": 0.05, "exp_bsp": 5.0 / 1.05, "ev_st": 0.05, "predicted_bfsp_st": 4.0, "n_bsp": 5,
            "code": "aw", "kind": "handicap", "race_name": "Toy Handicap", "race_code": "All Weather",
            "race_class": "Class 5", "cls": 5.0, "surface_type": "Polytrack", "country": "uk", "menu_hint": "Toy 3 11",
            "event_dt": "03-11-2025 14:30", "selection_id": 1, "event_id": 1, "win_lose": 0.0, "won": False,
            "bfsp": 4.0, "fold_idx": 0, "ppwap": 4.5, "pp_vol": 20000.0, "ppmax": 6.0, "ppmin": 3.5, "ipmax": 1000.0,
            "ipmin": 3.0, "ip_vol": 100.0, "k6_ok": True, "steam_score": 0.05, "steam_field": 6.0,
            "steam_code": "aw"}
    r = pd.DataFrame([{**base, **x} for x in rows])
    return {"runners": r, "place": pd.DataFrame(), "days": sorted(r["race_date"].unique()),
            "meta": {"steam_p70": 0.0, "steam_p20": -0.05, **meta}}


def money(bets: pd.DataFrame) -> pd.DataFrame:
    return core.settle(core.check_bets(bets))


# ---- the loader ---------------------------------------------------------------------------------------------------

def test_the_loader_never_lets_a_holdout_row_through(toy, frames):
    planted = pd.read_csv(toy["oos_dir"] / "oos_gate_stn.csv")
    assert (planted["race_date"] >= core.HOLDOUT_FROM).sum() > 0             # the toy does hold holdout rows
    for name in ("runners", "place"):
        assert len(frames[name]) and frames[name]["race_date"].max() < core.HOLDOUT_FROM
        core.assert_no_holdout(frames[name])
    assert frames["days"][-1] < core.HOLDOUT_FROM and frames["days"][0] == "2024-12-28"
    for f in ("oos_gate_stn_dev.csv", "oos_gate_st_dev.csv"):                # the files written for clv_betfair.load
        assert pd.read_csv(toy["work"] / f)["race_date"].max() < core.HOLDOUT_FROM
    assert "STRAT100_OOS_DIR" in frames["meta"]["sources"]["gate_stn"]       # it says what it read
    assert frames["meta"]["problems"] == []


def test_the_runner_accepts_the_loaders_frames(toy, frames, monkeypatch):
    monkeypatch.setattr(data_a, "load_a_model", lambda: frames)
    _, days, tables = run.load("a_model")
    assert set(tables) == {"runners", "place"} and days == frames["days"]


def test_whole_races_k6_and_the_joins(frames):
    r = frames["runners"]
    assert not r.duplicated(data_a.KEY).any()
    assert not ((r["race_date"] == BROKEN_RACE[0]) & (r["track"] == BROKEN_RACE[1])).any()   # not a whole race
    nr = r[(r["race_date"] == NON_RUNNER_RACE[0]) & (r["track"] == NON_RUNNER_RACE[1])]
    assert len(nr) == 6 and not nr["k6_ok"].any()                             # kept for the fit, never bet (K6)
    assert r.loc[~r["race"].isin(nr["race"]), "k6_ok"].all()
    assert r["race_name"].notna().all() and r["country"].isin(["uk", "ire"]).all()
    assert set(r["code"]) == {"aw", "jumps", "flat turf"}
    assert (r.loc[r["track"] == "Warwick", "code"] == "jumps").all()          # NH Flat is jumps by harness.code_of
    assert r.loc[r["track"] == "Warwick", "steam_score"].isna().all()         # and not a main code for the steamer
    assert r.loc[r["track"] != "Warwick", "steam_score"].notna().all()
    assert (r.loc[r["track"] == "Naas", "cls"].isna()).all() and (r.loc[r["track"] == "Cheltenham", "cls"] == 2).all()
    assert r["ev"].isna().sum() == (r["month"] == "2024-12").sum() > 0        # the first month is only fitted on
    assert np.allclose(1 - r["morningwap"] / r["exp_bsp"], -r["ev"], equal_nan=True)
    assert r["ev_st"].notna().sum() == r["ev"].notna().sum()
    for c in ("pred_move", "net", "hold_morning", "hold_bsp", "overlay_pct"):  # the BSP-derived columns are left out
        assert c not in r.columns


def test_text_columns_are_coerced(frames, toy):
    conn = sqlite3.connect(toy["db"])
    win = pd.read_sql_query(data_a.WIN_SQL, conn, params=("2024-12-28", core.HOLDOUT_FROM))
    conn.close()
    w = frames["runners"][data_a.KEY + ["race", "predicted_bfsp", "morningwap", "morning_vol", "bsp", "won", "field",
                                        "race_code", "surface_type", "race_class"]]
    a = data_a._assemble_a_model(w, win, min_train_races=5)["runners"]
    as_text = win.copy()
    for c in data_a.WIN_NUM:
        as_text[c] = as_text[c].astype(str)                                   # '4.2', 'nan', 'None'
    b = data_a._assemble_a_model(w, as_text, min_train_races=5)["runners"]
    pd.testing.assert_frame_equal(a, b)
    assert a["k6_ok"].sum() == frames["runners"]["k6_ok"].sum()


def test_earlier_betfair_sps_are_counted_strictly_before_the_day():
    runners = pd.DataFrame({"horse_name": ["A", "A", "B", "C"],
                            "race_date": ["2025-03-01", "2025-01-10", "2025-03-01", "2025-03-01"]})
    hist = pd.DataFrame({"horse_name": ["A", "A", "A", "A", "B", "Z"],
                         "race_date": ["2024-12-01", "2025-01-10", "2025-03-01", "2025-06-01", "2025-03-01", "2020-01-01"]})
    assert data_a._n_bsp(runners, hist).tolist() == [2, 1, 0, 0]      # the day's own run and later runs never count


def test_n_bsp_in_the_loaded_frames_reads_only_earlier_days(frames, toy):
    conn = sqlite3.connect(toy["db"])
    rr = pd.read_sql_query("SELECT horse_name, race_date FROM race_results WHERE bfsp > 1", conn)
    conn.close()
    r = frames["runners"].sample(40, random_state=1)
    for _, x in r.iterrows():
        assert x["n_bsp"] == ((rr["horse_name"] == x["horse_name"]) & (rr["race_date"] < x["race_date"])).sum()


def test_place_rows_carry_the_places_paid(frames):
    p = frames["place"]
    assert p["k_ok"].all() and set(p["k_places"]) == {2.0, 3.0}
    bad = pd.DataFrame({"country": "uk", "race_date": "2025-11-03", "place_event_id": [1, 1, 1, 1, 2, 2, 2],
                        "menu_hint": "T", "event_dt": "x", "selection_id": [1, 2, 3, 4, 5, 6, 7],
                        "place_won": [1, 1, 0, 0, 1, 0, 0], "place_bsp": [1.5, 2.0, 2.5, 4.0, 1.5, 2.0, 2.5],
                        "place_ppwap": 2.0, "place_pp_vol": 100.0})
    out = data_a._place_frame(bad)
    assert out.loc[out["place_event_id"] == 1, "k_ok"].all()             # 2 placed, the book rounds to 2
    assert not out.loc[out["place_event_id"] == 2, "k_ok"].any()         # 1 placed: not a 2-4 place market


def test_kind_of_is_the_race_types_querys_own_rule():
    src = (ROOT / "research/queries/done/race_types_1120.py").read_text()
    own = (ROOT / "research/strat100/data_a.py").read_text()
    for pattern in (r"""hcap = s.str.contains(r"handicap|hcap|h'cap|nursery")""",
                    r"""green = s.str.contains(r"novice|nov\b|maiden|mdn|beginner|beg\b|juvenile|juv\b|nh flat|nhf|bumper|inhf")"""):
        assert pattern in src and pattern in own
    got = data_a.kind_of(pd.Series(["Big Handicap Chase", "Nursery", "Maiden Hurdle", "Group 1 Stakes", None, "H'cap"]))
    assert got.tolist() == ["handicap", "handicap", "novice/maiden/bumper", "other", "other", "handicap"]


def test_an_out_of_sample_file_keeps_its_times_and_loses_the_holdout(tmp_path, monkeypatch):
    f = tmp_path / "oos_gate_stn.csv"
    f.write_text("race_date,race_time,track,horse_name,predicted_bfsp,won\n"
                 "2026-03-27,2.30,Toy,A,4.0,1\n2026-03-28,2.30,Toy,B,4.0,0\n2026-11-01,2.30,Toy,C,4.0,0\n")
    monkeypatch.setenv("STRAT100_OOS_DIR", str(tmp_path))
    monkeypatch.setattr(data_a, "_artifact_zips", _no_network)
    o, src = data_a._oos_frame("oos_gate_stn.csv", [(1, "x")], {})
    assert o["horse_name"].tolist() == ["A"] and o["race_time"].tolist() == ["2.30"] and "local file" in src


def test_no_database_is_no_data_found_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("STRAT100_DB", str(tmp_path / "absent.db"))
    with pytest.raises(data_a.NoData, match="no data found"):
        data_a.load_a_model()
    monkeypatch.setattr(data_a, "load_a_model", lambda: (_ for _ in ()).throw(data_a.NoData("no data found: toy")))
    with pytest.raises(data_a.NoData):
        run.load("a_model")


# ---- the registry and every strategy on the loaded toy ---------------------------------------------------------------

def test_the_registry_holds_the_sixteen_and_their_benchmarks():
    ids = [s.id for s in A.STRATEGIES]
    assert len(ids) == len(set(ids)) and set(SPEC_IDS) <= set(ids)
    assert set(ids) - set(SPEC_IDS) == {"S053-MAIN", "S068-REST"}
    for s in A.STRATEGIES:
        assert s.dataset == "a_model" and s.tested_before and callable(s.build)
        assert s.paired is None or s.paired in ids
        assert EM_DASH not in s.name + s.note + s.tested_before               # hyphens, never em dashes
    assert BY_ID["S053"].benchmark and BY_ID["S053-MAIN"].benchmark and BY_ID["S068-REST"].benchmark
    assert BY_ID["S061"].paired == "S053-MAIN" and BY_ID["S068"].paired == "S068-REST"
    assert all(BY_ID[i].paired == "S053" for i in SPEC_IDS if i not in ("S053", "S061", "S066", "S067", "S068"))
    assert BY_ID["S068"].entry == "sp" and BY_ID["S066"].side == "lay" and BY_ID["S065"].side == "both"
    # K10: the blend was chosen on a window that holds CONFIRM, and each rule was read there before: none is blind
    assert not any(s.blind for s in A.STRATEGIES)
    for i in ("S066", "S067", "S068", "S068-REST"):
        assert "Not blind:" in BY_ID[i].note and "No a_model strategy is blind" in BY_ID[i].note
    found, _ = run.discover()
    assert set(ids) <= {s.id for s in found}                                  # the runner takes every one


@pytest.mark.parametrize("sid", [s.id for s in A.STRATEGIES])
def test_every_strategy_returns_a_valid_bets_frame(sid, frames):
    before = {k: v.copy() for k, v in frames.items() if isinstance(v, pd.DataFrame)}
    bets, days = build(sid, frames)
    for k, v in before.items():
        pd.testing.assert_frame_equal(frames[k], v)                           # the shared frames are not touched
    b = core.check_bets(bets, BY_ID[sid])
    assert len(b) > 0, f"{sid} finds nothing on the toy: the test would prove nothing"
    assert b["unsettled"].sum() == 0 and not b["void"].any()
    assert not b.duplicated(["race_date", "race_key", "selection", "market"]).any()      # one bet a runner
    assert days and days[-1] < core.HOLDOUT_FROM and set(b["race_date"]) <= set(days)
    r = core.score(BY_ID[sid], bets, days, n_trials=486, draws=200)
    assert r["error"] is None and r["rows"] == len(b) and r["bets_outside_windows"] == 0
    assert r["select_bets"] + r["confirm_bets"] == len(b)
    want = {"S065": "no cuts; upper bound", "S068": "", "S068-REST": ""}.get(sid, "no cuts")
    assert r["flags"] == want


def test_the_runner_reads_a_strategy_and_its_benchmark(frames):
    days = frames["days"]
    tables = {k: v for k, v in frames.items() if isinstance(v, pd.DataFrame)}
    got = [run.run_one(BY_ID[i], frames, days, tables, 200, 60) for i in ("S053", "S054", "S053-MAIN", "S061")]
    assert all(g.get("error") is None for g in got)
    run.finish(got, 486, 200)
    by = {g["id"]: g for g in got}
    assert by["S053"]["verdict"].startswith("benchmark (") and by["S054"]["pair_verdict"] is not None
    assert by["S061"]["pair_select_days"] > 0 and by["S061"]["pair_confirm_days"] > 0
    assert "CHANGED THE SHARED FRAMES" not in " ".join(str(g.get("note")) for g in got)


def test_the_subsets_are_subsets_of_s053(frames):
    base = sel(build("S053", frames)[0])
    key = set(zip(base["race_key"], base["selection"], base["stake"].round(6)))
    for sid in ("S054", "S055", "S056", "S057", "S058", "S059", "S060", "S053-MAIN"):
        s = sel(build(sid, frames)[0])
        assert set(zip(s["race_key"], s["selection"], s["stake"].round(6))) <= key, sid
    for sid in ("S062", "S063", "S065"):                                      # another exit, the same bets
        pd.testing.assert_frame_equal(sel(build(sid, frames)[0]), base)
    r = frames["runners"].set_index(["race", "horse_name"])
    jumps = build("S056", frames)[0]
    assert (r.loc[list(zip(jumps["race_key"], jumps["selection"])), "code"] == "jumps").all()
    kept = build("S058", frames)[0]
    assert not kept["race_key"].str.contains("Cheltenham").any()              # the toy's Class 2 chase
    assert base["race_key"].str.contains("Cheltenham").any()


# ---- look-ahead: a later column changed must not change a selection --------------------------------------------------

MORNING = ["S053", "S054", "S055", "S056", "S057", "S058", "S059", "S060", "S062", "S063", "S065", "S066", "S067",
           "S053-MAIN"]


@pytest.mark.parametrize("sid", MORNING)
def test_a_morning_selection_reads_nothing_later(sid, frames):
    """Every BSP, pre-play, in-play and result column replaced on every row: the bets picked, their stakes and their
    entry prices are the same (only the settlement may move)."""
    changed = with_later_changed(frames)
    assert not np.allclose(changed["runners"]["bsp"], frames["runners"]["bsp"])
    pd.testing.assert_frame_equal(sel(build(sid, changed)[0]), sel(build(sid, frames)[0]))


def test_the_stacked_weights_read_earlier_months_only(frames):
    """S061: the last month's later columns replaced: that month's bets are unchanged (its weights are fitted on the
    months before). The control: an earlier month's BSPs replaced do move the last month's forecast."""
    r = frames["runners"]
    last = r["month"] == r["month"].max()
    a = sel(build("S061", frames)[0])
    b = sel(build("S061", with_later_changed(frames, last))[0])
    pd.testing.assert_frame_equal(b, a)
    u0, w0 = A._stack(r)
    u1, w1 = A._stack(with_later_changed(frames, ~last)["runners"])
    first_scored = r.loc[r["ev"].notna(), "month"].min()
    assert first_scored == "2025-01" and w0["month"].min() > first_scored    # never the first scored month itself
    assert not np.allclose(w0["w_rule"], w1["w_rule"])
    m = u0["month"] == r["month"].max()
    assert not np.allclose(u0.loc[m, "combined"], u1.loc[m, "combined"])


def test_the_day_throttle_reads_the_day_before_only(frames):
    """S064: the last day's later columns replaced: the last day's stakes are unchanged. The control: the day before
    made a losing day halves them."""
    r = frames["runners"]
    base, days = build("S053", frames)
    last = base["race_date"].max()
    prev = max(d for d in days if d < last)
    a = sel(build("S064", frames)[0])
    b = sel(build("S064", with_later_changed(frames, r["race_date"] == last))[0])
    pd.testing.assert_frame_equal(b[b["race_date"] == last], a[a["race_date"] == last])
    lost = frames["runners"].copy()
    lost.loc[lost["race_date"] == prev, "bsp"] = lost.loc[lost["race_date"] == prev, "morningwap"] * 3.0   # all drift
    c = sel(build("S064", {**frames, "runners": lost})[0])
    full = sel(base)
    assert np.allclose(c.loc[c["race_date"] == last, "stake"], 0.5 * full.loc[full["race_date"] == last, "stake"])


def test_expected_clv_reads_no_close_of_its_own_month(frames):
    """The loader's walk-forward closing model: the last month's BSPs replaced leave that month's expected CLV as it
    was; an earlier month's BSPs replaced move it (the model is fitted on them); a morning price replaced moves it."""
    w = frames["runners"][data_a.KEY + ["race", "month", "predicted_bfsp", "morningwap", "morning_vol", "bsp"]]
    last = (w["month"] == w["month"].max()).to_numpy()
    ev = data_a._ev_walk_forward(w, 5)
    rng = np.random.default_rng(5)
    later = w.assign(bsp=np.where(last, np.maximum(1.02, w["bsp"] * rng.uniform(0.4, 2.5, len(w))), w["bsp"]))
    assert np.allclose(data_a._ev_walk_forward(later, 5)[last], ev[last])
    earlier = w.assign(bsp=np.where(~last, np.maximum(1.02, w["bsp"] * rng.uniform(0.4, 2.5, len(w))), w["bsp"]))
    assert not np.allclose(data_a._ev_walk_forward(earlier, 5)[last], ev[last])
    morning = w.assign(morningwap=np.where(last, w["morningwap"] * 1.3, w["morningwap"]))
    assert not np.allclose(data_a._ev_walk_forward(morning, 5)[last], ev[last])
    assert ev[w["month"] == "2024-12"].isna().all() and ev[last].notna().all()


def test_the_volume_is_read_as_none_by_the_closing_model(frames):
    w = frames["runners"][data_a.KEY + ["race", "month", "predicted_bfsp", "morningwap", "morning_vol", "bsp"]]
    a = data_a._ev_walk_forward(w, 5)
    b = data_a._ev_walk_forward(w.assign(morning_vol=w["morning_vol"] * 7 + 3), 5)
    assert np.allclose(a, b, equal_nan=True)


def test_a_place_limit_is_set_before_the_place_price_is_known(frames):
    """S068: the place BSP, the win BSP, the in-play record and the win result replaced: the limit orders (which
    runners, at what limit, picked or not) are the same. The place BSP then decides only the fill."""
    o = A._place_orders(frames)
    r = frames["runners"].copy()
    rng = np.random.default_rng(9)
    r["bsp"] = np.maximum(1.02, r["bsp"] * rng.uniform(0.4, 2.5, len(r)))
    r["ipmin"], r["ipmax"], r["win_lose"] = 1.01, 1000.0, 1.0 - r["win_lose"]
    pl = frames["place"].copy()
    pl["place_bsp"] = np.maximum(1.02, pl["place_bsp"] * rng.uniform(0.4, 2.5, len(pl)))
    pl["place_ppwap"] = 3.0
    o2 = A._place_orders({**frames, "runners": r, "place": pl})
    cols = ["race", "horse_name", "pick", "p_place", "limit"]
    pd.testing.assert_frame_equal(o2[cols].reset_index(drop=True), o[cols].reset_index(drop=True))
    assert len(o) and o["pick"].any() and (~o["pick"]).any()
    assert not np.allclose(o2["place_bsp"], o["place_bsp"])
    filled = A._place_bets(o)
    assert 0 < len(filled) < len(o) and (filled["entry_price"] > 1).all()


# ---- each rule's arithmetic on hand-made rows ---------------------------------------------------------------------

def test_s053_guards_and_the_base_stake():
    fr = mini([{"horse_name": "ok"},                                             # 400 / 4 = 100
               {"horse_name": "thin", "morning_vol": 400.0},                     # 10% of 400 = 40
               {"horse_name": "short", "morningwap": 2.0, "predicted_bfsp": 1.9, "morning_vol": 5000.0},   # cap 200
               {"horse_name": "long", "morningwap": 101.0},                      # 400 / 100 = 4 -> floor 15
               {"horse_name": "low_ev", "ev": 0.029},
               {"horse_name": "under_model", "morningwap": 3.9},
               {"horse_name": "over_200", "morningwap": 201.0},
               {"horse_name": "no_money", "morning_vol": 99.0},
               {"horse_name": "k6", "k6_ok": False},
               {"horse_name": "unscored", "ev": np.nan}])
    bets, _ = build("S053", fr)
    got = dict(zip(bets["selection"], bets["stake"]))
    assert got == pytest.approx({"ok": 100.0, "thin": 40.0, "short": 200.0, "long": 15.0})
    assert (bets["entry_price"] == [5.0, 5.0, 2.0, 101.0]).all() and (bets["exit"] == "bsp").all()
    s = money(bets)
    assert s["pnl"].iloc[0] == pytest.approx(100.0 * (5.0 / 4.0 - 1))            # closed at the BSP: the move


def test_the_race_type_rules_on_hand_made_rows():
    fr = mini([{"horse_name": "hcap_aw"},
               {"horse_name": "nov_jump", "race": "r2", "kind": "novice/maiden/bumper", "code": "jumps",
                "race_code": "Hurdle", "n_bsp": 2},
               {"horse_name": "c2_chase", "race": "r3", "code": "jumps", "race_code": "Handicap Chase", "cls": 2.0},
               {"horse_name": "c4_chase", "race": "r4", "code": "jumps", "race_code": "Chase", "cls": 4.0, "n_bsp": 0},
               {"horse_name": "irish_chase", "race": "r5", "code": "jumps", "race_code": "Chase", "cls": 1.0,
                "country": "ire", "kind": "other", "n_bsp": np.nan}])
    def names(sid):
        return sorted(build(sid, fr)[0]["selection"])
    assert names("S054") == ["c2_chase", "c4_chase", "hcap_aw"]
    assert names("S055") == ["c2_chase", "hcap_aw"]                              # under 3, or unknown: not backed
    assert names("S056") == ["c2_chase", "c4_chase", "irish_chase", "nov_jump"]
    assert names("S057") == ["hcap_aw"]
    assert names("S058") == ["c4_chase", "hcap_aw", "irish_chase", "nov_jump"]   # the Irish chase stays in


def test_both_blends_and_the_steamer_gate_on_hand_made_rows():
    fr = mini([{"horse_name": "both"},
               {"horse_name": "stn_only", "ev_st": 0.01},
               {"horse_name": "st_unscored", "ev_st": np.nan},
               {"horse_name": "drifter", "steam_score": -0.2},
               {"horse_name": "small_field", "steam_field": 4.0},
               {"horse_name": "bumper", "steam_score": np.nan, "steam_code": np.nan}])
    assert sorted(build("S059", fr)[0]["selection"]) == ["both", "bumper", "drifter", "small_field"]
    assert sorted(build("S060", fr)[0]["selection"]) == ["both", "st_unscored", "stn_only"]
    assert sorted(build("S053-MAIN", fr)[0]["selection"]) == ["both", "drifter", "st_unscored", "stn_only"]


def test_missing_inputs_are_no_data_found(frames):
    r = frames["runners"]
    for sid, col in (("S059", "ev_st"), ("S060", "steam_score"), ("S061", "steam_score"), ("S055", "n_bsp"),
                     ("S053-MAIN", "steam_score")):
        with pytest.raises(data_a.NoData, match="no data found"):
            BY_ID[sid].build({**frames, "runners": r.assign(**{col: np.nan})})
    with pytest.raises(data_a.NoData, match="no data found"):
        BY_ID["S060"].build({**frames, "meta": {}})
    with pytest.raises(data_a.NoData, match="no data found"):
        BY_ID["S068"].build({**frames, "place": frames["place"].iloc[:0]})
    with pytest.raises(data_a.NoData, match="no data found"):
        BY_ID["S053"].build({"runners": r.iloc[:0], "days": [], "meta": {}})
    tables = {"runners": r}
    out = run.run_one(BY_ID["S060"], {**frames, "meta": {}}, frames["days"], tables, 100, 30)
    assert out["verdict"] == "error" and "no data found" in out["error"]        # recorded, the run goes on


def test_the_half_hedge_keeps_half_the_winnings():
    fr = mini([{"horse_name": "w", "ev": 0.12, "win_lose": 1.0},
               {"horse_name": "l", "ev": 0.12, "win_lose": 0.0, "race": "r2"},
               {"horse_name": "full", "ev": 0.05, "win_lose": 1.0, "race": "r3"},
               {"horse_name": "irish", "ev": 0.05, "win_lose": 1.0, "race": "r4", "country": "ire"}])
    s = money(build("S062", fr)[0]).set_index("selection")
    stake, p, b = 100.0, 5.0, 4.0
    assert s.loc["w", "pnl"] == pytest.approx(0.5 * stake * (p - 1))                       # 200
    assert s.loc["l", "pnl"] == pytest.approx(-stake + 0.5 * stake * (p - 1) / (b - 1))    # -100 + 66.67
    assert s.loc["full", "pnl"] == pytest.approx(stake * (p / b - 1))
    assert s.loc["irish", "pnl"] == pytest.approx(stake * (p / b - 1))                     # under 10%: closed in full
    s = money(build("S063", fr)[0]).set_index("selection")
    assert s.loc["irish", "pnl"] == pytest.approx(0.5 * stake * (p - 1))                   # Irish: half held
    assert s.loc["w", "pnl"] == pytest.approx(stake * (p / b - 1))                         # UK: closed in full
    assert s["unsettled"].sum() == 0


def test_the_day_multiplier():
    pnl = pd.Series([10.0, -5.0, 0.0, -1.0, -2.0, 3.0], index=[f"2025-11-0{i}" for i in range(1, 7)])
    assert A._day_multiplier(pnl).tolist() == [1.0, 1.0, 0.5, 1.0, 0.5, 0.5]     # a no-bet day is not a losing day
    fr = mini([{"horse_name": "d1", "race_date": "2025-11-03", "race": "a", "bsp": 6.0},       # drifts: a losing day
               {"horse_name": "d2", "race_date": "2025-11-04", "race": "b", "bsp": 4.0},       # after a loss: half
               {"horse_name": "d3", "race_date": "2025-11-05", "race": "c", "bsp": 4.0}])      # after a win: full
    bets, _ = build("S064", fr)
    assert bets["stake"].tolist() == pytest.approx([100.0, 50.0, 100.0])


def test_the_in_running_lay_fills_a_tick_through_the_target_only():
    bsp = 5.0
    target = 1 + (bsp - 1) * 0.5                                                 # 3.0
    under = core.tick_under(target)                                              # 2.98
    fr = mini([{"horse_name": "at_target", "ipmin": target, "bsp": bsp, "morningwap": 6.0, "win_lose": 0.0},
               {"horse_name": "tick_under", "ipmin": under, "bsp": bsp, "morningwap": 6.0, "win_lose": 0.0, "race": "r2"},
               {"horse_name": "won_unfilled", "ipmin": np.nan, "bsp": bsp, "morningwap": 6.0, "win_lose": 1.0,
                "race": "r3"},
               {"horse_name": "no_trade", "ipmin": 0.0, "bsp": bsp, "morningwap": 6.0, "win_lose": 0.0, "race": "r4"},
               {"horse_name": "no_trade_1", "ipmin": 1.0, "bsp": bsp, "morningwap": 6.0, "win_lose": 0.0,
                "race": "r5"}])
    bets, _ = build("S065", fr)
    s = money(bets).set_index("selection")
    stake = 400.0 / 5.0
    assert under == pytest.approx(2.98)
    assert bets.set_index("selection")["exit"].to_dict() == {"at_target": "result", "tick_under": "inplay",
                                                             "won_unfilled": "result", "no_trade": "result",
                                                             "no_trade_1": "result"}
    assert s.loc["no_trade", "pnl"] == pytest.approx(-stake)     # the files' 0 = no in-running trade: never a fill
    assert s.loc["no_trade_1", "pnl"] == pytest.approx(-stake)
    assert s.loc["at_target", "pnl"] == pytest.approx(-stake)                    # touched, not traded through: lost
    assert s.loc["tick_under", "pnl"] == pytest.approx(stake * (6.0 / target - 1))     # stake x morningwap / L at L
    assert s.loc["won_unfilled", "pnl"] == pytest.approx(stake * 5.0)
    assert s["unsettled"].sum() == 0


def test_the_lay_book_on_hand_made_rows():
    lay = {"ev": -0.05, "exp_bsp": 5.0 / 0.95, "predicted_bfsp": 6.0, "bsp": 6.0}
    fr = mini([{"horse_name": "lay", **lay},                                     # 400 / 4 = 100
               {"horse_name": "thin", **lay, "morning_vol": 199.0},
               {"horse_name": "over_10", **lay, "morningwap": 10.5, "predicted_bfsp": 12.0},
               {"horse_name": "edge_small", **lay, "ev": -0.029, "exp_bsp": 5.0 / 0.971},
               {"horse_name": "model_shorter", **lay, "predicted_bfsp": 4.9},
               {"horse_name": "at_ten", **lay, "morningwap": 10.0, "exp_bsp": 10.0 / 0.95, "predicted_bfsp": 12.0,
                "morning_vol": 300.0}])                                          # 400 / 9 = 44.4 -> 10% of 300 = 30
    bets, _ = build("S066", fr)
    assert dict(zip(bets["selection"], bets["stake"])) == pytest.approx({"lay": 100.0, "at_ten": 30.0})
    assert (bets["side"] == "lay").all()
    s = money(bets).set_index("selection")
    assert s.loc["lay", "pnl"] == pytest.approx(100.0 * (1 - 5.0 / 6.0))         # closed by a back at the BSP
    assert s.loc["lay", "risk"] == pytest.approx(400.0)                          # GBP400 at risk


def test_the_false_favourite_on_hand_made_rows():
    def race(name, fav_price=3.0, fav_model=4.0, other_model=3.5, n=5, **fav):
        rows = [{"race": name, "horse_name": f"{name}_fav", "morningwap": fav_price, "predicted_bfsp": fav_model,
                 "morning_vol": 1000.0, "n_field": n, "bsp": 3.6, **fav}]
        rows += [{"race": name, "horse_name": f"{name}_{i}", "morningwap": 8.0 + i, "predicted_bfsp": other_model + i,
                  "n_field": n} for i in range(n - 1)]
        return rows
    rows = (race("good") + race("rated", fav_model=3.0, other_model=5.0)         # the model's shortest too: no bet
            + race("close", fav_model=3.7)                                       # 3.7 < 1.25 x 3.0
            + race("small", n=4) + race("thin", morning_vol=299.0) + race("odds_on", fav_price=1.7, fav_model=3.0)
            + race("joint", fav_model=3.5)                                       # joint-shortest on the model: no bet
            + race("k6", k6_ok=False))
    tie = race("tie")
    tie[1]["morningwap"] = 3.0                                                   # two at the lowest price: skipped
    fr = mini(rows + tie)
    bets, days = build("S067", fr)
    assert bets["selection"].tolist() == ["good_fav"] and bets["side"].tolist() == ["lay"]
    assert bets["stake"].iloc[0] == pytest.approx(100.0)                         # 200 at risk / (3.0 - 1)
    assert money(bets)["pnl"].iloc[0] == pytest.approx(100.0 * (1 - 3.0 / 3.6))
    assert days == fr["days"]                                                    # every race day: no closing model


def test_the_place_limit_on_hand_made_rows():
    q = np.array([0.4, 0.25, 0.15, 0.1, 0.06, 0.04])
    rows = [{"horse_name": f"h{i}", "selection_id": i, "ppwap": 1 / q[i], "morningwap": 1 / q[i],
             "predicted_bfsp": 1 / q[i], "ev": 0.05 if i == 1 else -0.2, "pp_vol": 20000.0 if i != 2 else 500.0}
            for i in range(6)]
    fr = mini(rows)
    p = core.place_prob(q, 2)
    limit = 1 + (1 / p - 1) / 0.98
    place = pd.DataFrame({"race_date": "2025-11-03", "country": "uk", "menu_hint": "Toy 3 11",
                          "event_dt": "03-11-2025 14:30", "selection_id": range(6), "place_event_id": 2,
                          "place_bsp": [limit[0] * 0.99, limit[1] + 0.01, limit[2] * 1.2, limit[3], limit[4] * 0.5,
                                        limit[5] * 2], "place_ppwap": 2.0, "place_pp_vol": 100.0,
                          "place_won": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0], "k_places": 2.0, "k_ok": True})
    fr["place"] = place
    o = A._place_orders(fr).set_index("horse_name")
    assert np.allclose(o["limit"].to_numpy(), limit[:5]) and "h5" not in o.index      # win ppwap 25 is over 21.0
    assert o["pick"].to_dict() == {"h0": False, "h1": True, "h2": False, "h3": False, "h4": False}
    mine, _ = build("S068", fr)
    rest, _ = build("S068-REST", fr)
    assert mine["selection"].tolist() == ["h1"] and mine["stake"].tolist() == [10.0]
    assert mine["entry_price"].iloc[0] == pytest.approx(limit[1] + 0.01) and mine["market"].iloc[0] == "place"
    own = A._place_bets(o.reset_index()[~o["pick"].to_numpy()])
    assert dict(zip(own["selection"], own["stake"])) == pytest.approx({"h2": 5.0, "h3": 10.0})     # 1% of 500; at L
    # the benchmark is stake-matched by day: the rest's 15 scaled to S068's 10
    assert dict(zip(rest["selection"], rest["stake"])) == pytest.approx({"h2": 10 / 3, "h3": 20 / 3})
    assert rest["stake"].sum() == pytest.approx(mine["stake"].sum())
    s = money(mine)
    assert s["pnl"].iloc[0] == pytest.approx(10.0 * (limit[1] + 0.01 - 1))        # placed: held to the result
    assert money(rest)["pnl"].tolist() == pytest.approx([-10 / 3, -20 / 3])
    fr["place"] = place.assign(k_ok=False)                                        # the places paid not confirmed
    assert len(A._place_orders(fr)) == 0
    fr["place"] = place[place["selection_id"] != 4]                               # a runner with no place row
    assert len(A._place_orders(fr)) == 0


def test_s068_is_not_better_than_a_benchmark_that_loses_less_a_pound(monkeypatch):
    """S068 one GBP10 place back a day losing about 26% a pound; the rest ten a day losing about 11% a pound. In GBP
    a day at their own stakes the rest loses more, so the unmatched difference is positive every day; stake-matched,
    S068 is the worse bet and the pair says so."""
    days = [f"2025-{m:02d}-{d:02d}" for m in (4, 5) for d in range(1, 21)] + \
           [f"2025-{m:02d}-{d:02d}" for m in (10, 11) for d in range(1, 21)]
    rows = []
    for i, day in enumerate(days):
        rows.append({"race_date": day, "race": f"{day}|p", "horse_name": "pick", "pick": True,
                     "place_won": float(i % 8 < 3)})
        for j in range(10):
            rows.append({"race_date": day, "race": f"{day}|r{j}", "horse_name": f"rest{j}", "pick": False,
                         "place_won": float(j < 4 + i % 2)})
    o = pd.DataFrame(rows).assign(place_bsp=2.0, limit=1.9, pp_vol=50000.0)
    monkeypatch.setattr(A, "_place_orders", lambda frames: o)
    fr = mini([{"race_date": d, "race": f"{d}|p"} for d in days])
    mine, d1 = A.s068(fr)
    rest, d2 = A.s068_rest(fr)
    assert d1 == d2 == days and len(mine) == len(days) and len(rest) == 10 * len(days)
    assert np.allclose(rest.groupby("race_date")["stake"].sum(), 10.0)           # matched to S068's GBP10 a day
    a = core.score(BY_ID["S068"], mine, days, n_trials=486, draws=400)
    b = core.score(BY_ID["S068-REST"], rest, days, n_trials=486, draws=400)
    assert a["confirm_ret"] < b["confirm_ret"] < 0                               # S068 is the worse bet a pound
    core.pair(a, b, 486, draws=400)
    assert a["pair_confirm_mean"] < 0 and a["pair_verdict"] == "not better than its benchmark"
    unmatched = core.score(BY_ID["S068-REST"], A._place_bets(o[~o["pick"]]), days, n_trials=486, draws=400)
    assert unmatched["confirm_pnl"] < a["confirm_pnl"]       # the size trap the matching removes: more pounds lost
    assert unmatched["confirm_ret"] == pytest.approx(b["confirm_ret"])           # the return a pound is the same


def _toy_w_and_win(frames, toy):
    conn = sqlite3.connect(toy["db"])
    win = pd.read_sql_query(data_a.WIN_SQL, conn, params=("2024-12-28", core.HOLDOUT_FROM))
    conn.close()
    w = frames["runners"][data_a.KEY + ["race", "predicted_bfsp", "morningwap", "morning_vol", "bsp", "won", "field",
                                        "race_code", "surface_type", "race_class"]]
    return w, win


def test_a_betfair_runner_the_forecast_file_lacks_is_not_a_whole_race(frames, toy):
    """A selection with a BSP in Betfair's market that the forecast file does not hold: the race is not bet (its
    book, its favourite and its place chance would be read on a short field). Every other race is as it was."""
    w, win = _toy_w_and_win(frames, toy)
    race = frames["runners"].loc[frames["runners"]["k6_ok"], "race"].iloc[0]
    one = frames["runners"][frames["runners"]["race"] == race].iloc[0]
    src = win[(win["event_id"] == one["event_id"]) & (win["country"] == one["country"])].iloc[0]
    extra = src.copy()
    extra["bp_id"], extra["selection_id"], extra["bf_bsp"], extra["bf_morningwap"] = 10 ** 7, 424242, 2.0, 2.0
    for c in ("race_results_id", "race_date", "track", "race_time", "horse_name", "race_name"):
        extra[c] = None                                                       # no race_results row: not in the frame
    out = data_a._assemble_a_model(w, pd.concat([win, extra.to_frame().T], ignore_index=True), min_train_races=5)
    r = out["runners"]
    assert not r.loc[r["race"] == race, "k6_ok"].any()
    assert r.loc[r["race"] == race, "market_rows"].eq(r.loc[r["race"] == race, "n_field"] + 1).all()
    assert out["meta"]["races_dropped_field_mismatch"] == 1
    base = frames["runners"].set_index(data_a.KEY)["k6_ok"]
    other = r[r["race"] != race].set_index(data_a.KEY)["k6_ok"]
    assert (other == base.reindex(other.index)).all()
    assert frames["meta"]["races_dropped_field_mismatch"] == 0


def test_two_betfair_rows_for_one_runner_is_not_a_bet(frames, toy):
    w, win = _toy_w_and_win(frames, toy)
    one = frames["runners"][frames["runners"]["k6_ok"]].iloc[0]
    src = win[win["race_results_id"] == one["race_results_id"]].iloc[0].copy()
    src["bp_id"] = 10 ** 7
    r = data_a._assemble_a_model(w, pd.concat([win, src.to_frame().T], ignore_index=True), min_train_races=5)["runners"]
    assert not r.loc[r["race"] == one["race"], "k6_ok"].any()                 # neither row is taken: a guess


def test_a_forecast_row_given_twice_is_not_two_bets(frames, toy):
    """A runner given twice in the out-of-sample file (clv_betfair.load de-duplicates the price side only): its race
    is dropped and said so, never bet twice."""
    w, win = _toy_w_and_win(frames, toy)
    race = frames["runners"].loc[frames["runners"]["k6_ok"], "race"].iloc[0]
    twice = pd.concat([w, w[w["race"] == race].iloc[[0]]], ignore_index=True)
    out = data_a._assemble_a_model(twice, win, w_st=twice, min_train_races=5)
    r = out["runners"]
    assert not r.duplicated(data_a.KEY).any() and race not in set(r["race"])
    assert r["race"].nunique() == w["race"].nunique() - 1
    assert sum("a forecast row given twice" in p for p in out["meta"]["problems"]) == 2      # gate_stn and gate_st
    kept, n = data_a._drop_twice(twice)
    assert n == 1 and len(kept) == len(w) - (w["race"] == race).sum()


def test_a_price_at_or_under_one_is_no_price(frames, toy):
    """Betfair's files write 0, not blank, where nothing traded: the loader never hands it on as a price."""
    w, win = _toy_w_and_win(frames, toy)
    one = frames["runners"][frames["runners"]["k6_ok"]].iloc[0]
    hit = win["race_results_id"] == one["race_results_id"]
    zero = win.copy()
    zero.loc[hit, ["ipmin", "ipmax", "ppmin"]] = [0.0, 0.0, 1.0]
    r = data_a._assemble_a_model(w, zero, min_train_races=5)["runners"]
    x = r[r["race_results_id"] == one["race_results_id"]].iloc[0]
    assert np.isnan(x["ipmin"]) and np.isnan(x["ipmax"]) and np.isnan(x["ppmin"]) and x["ppwap"] > 1
    assert bool(x["k6_ok"])                                                   # the race is still whole
    for c in data_a.NOT_A_PRICE:
        assert not (r[c] <= 1.0).any()


def test_the_steamer_reads_card_facts_and_earlier_days_only(toy):
    """The CLEAN features of the last toy day are the same when that day's results and prices are blanked
    (morning_features.card_view): the steamer's score is a morning fact."""
    from research.paper import morning_features as MF
    df = core.drop_holdout(MF.L.load(str(toy["db"])))
    assert df["race_date"].max() < core.HOLDOUT_FROM
    day = df["race_date"].max()
    a = MF.build(df)
    b = MF.build(MF.card_view(df, day))
    key = ["race_date", "track", "race_time", "horse_name"]
    a = a[a["race_date"] == day].sort_values(key).reset_index(drop=True)
    b = b[b["race_date"] == day].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[key + MF.CLEAN], b[key + MF.CLEAN])
    assert len(a) > 0


def test_no_em_dash_and_no_order_code_in_the_new_files():
    for f in ("research/strat100/strategies_a_model.py", "research/strat100/data_a.py",
              "tests/test_strat100_a_model.py"):
        text = (ROOT / f).read_text()
        assert EM_DASH not in text, f
        if f.startswith("research"):
            assert not re.search(r"(?<![_\w])place_orders?\(|placeOrders|betfair_client|boto3|auto_trade|from trading", text), f
