"""research/greyhound/close_lab.py: the checkpoints rebuilt from the stream with the place market joined and the SP
as the target; each spec fitted on earlier days only; the baselines; the champion only on fresh days."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import pandas as pd

from research.greyhound import close_lab as cl

OFF = datetime(2026, 11, 5, 14, 0, tzinfo=timezone.utc)


def line(t, mc):
    return json.dumps({"rt": t.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                       "m": json.dumps({"op": "mcm", "pt": int(t.timestamp() * 1000), "mc": [mc]})})


def defn(mtype, status="OPEN", bsp=None):
    rs = [{"id": i, "status": "ACTIVE", **({"bsp": bsp[i]} if bsp else {})} for i in (1, 2)]
    return {"marketTime": OFF.strftime("%Y-%m-%dT%H:%M:%S.000Z"), "inPlay": False, "status": status,
            "marketType": mtype, "eventId": "e1", "venue": "Romford", "runners": rs}


def test_snapshots_and_the_frame():
    t0 = OFF - timedelta(seconds=400)
    lines = [line(t0, {"id": "1.1", "img": True, "marketDefinition": defn("WIN"), "rc": [
                 {"id": 1, "atb": [[2.0, 50], [1.99, 20]], "atl": [[2.02, 40]], "spn": 2.0, "tv": 100, "ltp": 2.0,
                  "spb": [[1.01, 30]], "spl": [[1000, 12]], "trd": [[2.0, 60], [2.02, 40]]},
                 {"id": 2, "atb": [[3.0, 30]], "atl": [[3.1, 30]], "spn": 3.1, "tv": 50, "ltp": 3.05}]}),
             line(t0, {"id": "1.2", "img": True, "marketDefinition": defn("PLACE"), "rc": [
                 {"id": 1, "atb": [[1.3, 10]], "atl": [[1.32, 10]]}, {"id": 2, "atb": [[1.6, 10]], "atl": [[1.65, 10]]}]}),
             line(OFF - timedelta(seconds=100), {"id": "1.1", "rc": [{"id": 1, "tv": 160}]}),
             line(OFF + timedelta(minutes=2), {"id": "1.1", "marketDefinition": defn("WIN", "CLOSED",
                                                                                      bsp={1: 1.9, 2: 3.3})})]
    snap = cl.snapshots(lines)
    # the market closed two minutes after its scheduled off: the late checkpoints run until then, and no further
    assert set(snap.check) == set(cl.CHECKS) | set(range(0, -121, -10)) and set(snap.mtype) == {"WIN", "PLACE"}
    f = cl.frame(snap)
    assert set(f.sid) == {1, 2} and np.isclose(f.y.iloc[0], np.log(f.bsp.iloc[0]))
    assert (pd.to_datetime(f.t_closed, utc=True) - pd.to_datetime(f["off"], utc=True)).dt.total_seconds().eq(120).all()
    one = f[(f.sid == 1) & (f.check == 60)].iloc[0]
    # the SP pool: GBP30 of SP backs need 30 x (mid - 1) of liability, against GBP12 of SP lays
    assert np.isclose(one.sp_back_liab, np.log1p(30 * (one["mid"] - 1))) and np.isclose(one.sp_lay_liab, np.log1p(12))
    assert one.sp_imb > 0 and np.isclose(one.vwap_vs_mid, np.log((2.0 * 60 + 2.02 * 40) / 100) - one.l_mid)
    assert set(cl.FEATURES["race"] + cl.FEATURES["path"] + cl.FEATURES["sppool"]) <= set(f.columns)
    r = f[(f.sid == 1) & (f.check == 60)].iloc[0]
    assert r.p_mid > 1 and r.tv == 160 and r.tv_d60 == 60                     # 100 at 120 s out, 160 at 60 s out
    assert np.isclose(r.spn_vs_mid, np.log(2.0) - r.l_mid)


def synth(day_seed, n=600, sd=0.1):
    """A day of 6-runner races, each read at one checkpoint, where log(SP) = log(mid) + 0.5 x spn_vs_mid + noise and
    the winner is drawn from the SP's chances: a model reading spn_vs_mid should beat the mid on both targets."""
    rng = np.random.default_rng(day_seed)
    mid = np.exp(rng.uniform(0.3, 2.5, n))
    gap = rng.normal(0, sd, n)
    races = n // 6
    secs = np.repeat(rng.choice([10, 60, 120, 200], races), 6)
    d = pd.DataFrame({"market": [f"m{day_seed}_{i // 6}" for i in range(n)], "sid": np.arange(n) % 6,
                      "secs": secs, "check": secs, "l_mid": np.log(mid), "l_ltp": np.log(mid),
                      "l_spn": np.log(mid) + gap, "spn_vs_mid": gap})
    for fam in cl.FEATURES.values():
        for c in fam:
            if c not in d:
                d[c] = 0.0
    d["y"] = d.l_mid + 0.5 * gap + rng.normal(0, 0.02, n)
    p = np.exp(-d.y.to_numpy()).reshape(races, 6)
    p = p / p.sum(1, keepdims=True)
    win = np.array([rng.choice(6, p=r) for r in p])
    d["won"] = (np.tile(np.arange(6), races) == np.repeat(win, 6)).astype(int)
    return d


def test_specs_are_fitted_on_earlier_days_and_the_champion_needs_fresh_days():
    days = {"2026-11-05": synth(1), "2026-11-06": synth(2), "2026-11-07": synth(3)}
    specs = [{"name": "ridge_sp", "registered": "2026-11-05", "features": ["sp"], "model": {"kind": "ridge", "lam": 1.0}},
             {"name": "late", "registered": "2026-11-07", "features": ["sp"], "model": {"kind": "ridge", "lam": 1.0}}]
    res = cl.evaluate(days, specs)
    first = res[res.day == "2026-11-05"]
    assert set(first.spec) == set(cl.BASELINES) | {cl.ORACLE}               # nothing to train on yet
    later = res[res.day == "2026-11-07"].set_index("spec").rmse
    assert later["ridge_sp"] < later["mid"]
    ch = cl.champion(res, specs, days, incumbent=None, bar=(3.0, 1, 100))    # one day it could be fitted for
    assert ch["challenger"] == "ridge_sp" and ch["champion"] == "ridge_sp"   # 'late' has no fresh day yet
    assert ch["paired"]["z"] > 3 and ch["paired"]["races"] == 100
    assert cl.champion(res, specs, days, incumbent=None)["champion"] in cl.BASELINES   # one fresh day: under the bar
    only_first = cl.champion(first, specs, days, incumbent="spn")            # a stored baseline is re-picked
    sc = first[first.spec.isin(cl.BASELINES)].set_index("spec").rmse
    assert only_first["champion"] == only_first["incumbent"] and sc[only_first["champion"]] == sc.min()


def test_the_registered_specs_load_with_a_hash():
    specs = cl.load_specs()
    assert len({s["name"] for s in specs}) == len(specs) and all(len(s["hash"]) == 10 for s in specs)
    feats = {f for fam in cl.FEATURES.values() for f in fam}
    assert all(set(s["features"]) <= set(cl.FEATURES) for s in specs) and feats


def test_both_targets_are_scored_and_each_has_its_champion():
    days = {f"2026-11-0{i}": synth(10 + i, n=3000, sd=0.6) for i in (5, 6, 7, 8)}
    specs = [{"name": "ridge_sp", "registered": "2026-11-05", "features": ["sp"], "model": {"kind": "ridge", "lam": 1.0}},
             {"name": "win_sp", "registered": "2026-11-05", "target": "win", "features": ["sp"],
              "model": {"kind": "gbm", "rounds": 60, "min_leaf": 50}}]
    res = cl.evaluate(days, specs)
    assert res.win_ll.notna().all() and (res.win_races == 500).all()          # every race has one winner
    last = res[res.day == "2026-11-08"].set_index("spec")
    assert last.loc["ridge_sp", "win_ll"] < last.loc["mid", "win_ll"]          # the SP forecast also picks winners
    assert last.loc["win_sp", "win_ll"] < last.loc["mid", "win_ll"]            # the result learnt from the market on
    assert np.isfinite(last.loc["win_sp", "rmse"])                              # a win spec is scored on the SP too
    q = cl.win_chances(days["2026-11-08"], days["2026-11-08"].l_mid.to_numpy())
    assert np.allclose(pd.Series(q).groupby(days["2026-11-08"].market.to_numpy()).sum(), 1.0)
    chw = cl.champion(res, specs, days, incumbent=None, target="win")
    assert chw["metric"] == "win_ll" and chw["challenger"] in ("ridge_sp", "win_sp")
    assert chw["paired"] is not None and chw["paired"]["races"] >= 20
    reg = {s["name"]: s for s in cl.load_specs()}
    assert reg["gbm_win_book"]["target"] == "win" and reg["gbm_win_book"]["hash"] != reg["gbm_book"]["hash"]


def test_the_ridge_survives_an_infinite_feature():
    days = {"2026-11-05": synth(1), "2026-11-06": synth(2)}
    days["2026-11-05"].loc[0, "spn_vs_mid"] = np.inf
    spec = {"name": "r", "registered": "2026-11-05", "features": ["sp"], "model": {"kind": "ridge", "lam": 1.0}}
    f = cl.fit_predict(days["2026-11-05"], days["2026-11-06"], spec)
    assert np.isfinite(f).all()


def test_the_trade_rules_read_the_forecast_and_count_only_fresh_days():
    d = pd.DataFrame({"market": ["m1", "m1", "m2"], "secs": [60, 60, 30], "y": np.log([5.0, 2.0, 4.0]),
                      "b1": [6.0, 2.0, 4.0], "bs1": [10.0, 10.0, 10.0], "l1": [6.2, 2.02, 3.0], "ls1": [10.0, 10.0, 10.0],
                      "won": [0, 1, 1]})
    d["e_gbm_all_small"] = 0.0                                           # forecast = the SP itself
    tr = cl.trade_rules({"2026-11-09": d}, comm=0.0)
    back = tr[tr.rule == "back_60_0.03"]                                 # 6.0 offered against a forecast of 5.0
    assert len(back) == 1 and back.pnl_sp.iloc[0] == pytest.approx(6.0 / 5.0 - 1) and back.pnl_win.iloc[0] == -1.0
    lay = tr[tr.rule == "lay_30_0.1"]                                    # 3.0 to lay against a forecast of 4.0
    assert len(lay) == 1 and lay.pnl_sp.iloc[0] == pytest.approx(1 - 3.0 / 4.0) and lay.pnl_win.iloc[0] == -2.0
    assert cl.trade_scores(tr.assign(day="2026-11-08")).empty                 # the registration day itself never counts
    assert set(cl.trade_scores(tr).rule) >= {"back_60_0.03", "lay_30_0.1"}


def test_a_crown_is_lost_when_it_no_longer_clears_the_bar():
    days = {f"2026-11-0{i}": synth(20 + i) for i in (5, 6, 7, 8, 9)}
    for d in days.values():
        d["wom1"] = np.random.default_rng(0).normal(size=len(d))            # noise: a spec reading it has no edge
    specs = [{"name": "noise", "registered": "2026-11-05", "features": ["sp"], "model": {"kind": "ridge", "lam": 1.0}}]
    specs[0]["features"] = ["flow"]                                         # constant columns: it forecasts the mid
    res = cl.evaluate(days, specs)
    ch = cl.champion(res, specs, days, incumbent="noise", bar=(3.0, 2, 100))
    assert ch["incumbent"] in cl.BASELINES and "crown returns" in ch["note"]


def test_the_headline_leaves_out_the_late_checkpoints_and_an_old_spec_never_learns_from_them():
    def with_late(seed):
        d = synth(seed)
        late = d.assign(secs=-30, check=-30, y=d.y + 5.0)                   # after the scheduled off: a different world
        return pd.concat([d, late], ignore_index=True)
    days = {"2026-11-05": with_late(1), "2026-11-06": with_late(2)}
    spec = {"name": "old", "registered": "2026-11-05", "features": ["sp"], "model": {"kind": "ridge", "lam": 1.0}}
    new = dict(spec, name="new", late=True)
    res = cl.evaluate(dict(days), [spec, new]).set_index(["day", "spec"])
    r = res.loc[("2026-11-06", "old")]
    assert r.runners == 600 and r.rmse < 0.1 and r.rmse_late > 4                # the late rows: their own window
    assert res.loc[("2026-11-06", "new"), "rmse"] > r.rmse                     # 'new' learnt from them, 'old' did not
    plain = cl.fit_predict(synth(1), days["2026-11-06"], spec)
    assert np.allclose(plain, cl.fit_predict(days["2026-11-05"], days["2026-11-06"], spec))


def test_the_model_options_fit_and_the_book_is_normalised():
    days = {f"2026-11-0{i}": synth(30 + i, n=3000, sd=0.3) for i in (5, 6)}
    tr, te = days["2026-11-05"], days["2026-11-06"]
    mid_err = float(np.sqrt(np.mean((te.y - te.l_mid) ** 2)))
    for opt in ({"by_window": True}, {"loss": "huber"}, {"norm": True}):
        spec = {"name": "g", "late": True, "features": ["sp"],
                "model": {"kind": "gbm", "leaves": 7, "min_leaf": 50, "rounds": 80, **opt}}
        f = cl.fit_predict(tr, te, spec)
        assert np.isfinite(f).all() and float(np.sqrt(np.mean((te.y - f) ** 2))) < mid_err, opt
    key = [te.market, te.check]
    book = pd.Series(np.exp(-f), index=te.index).groupby(key).sum()             # the last option: norm
    want = (te.check.map(cl.book_ratio(tr)) * np.exp(-te.l_mid).groupby(key).transform("sum")).groupby(key).first()
    assert np.allclose(book, want)


def test_the_conditional_logit_learns_the_result_and_the_second_stage_reads_it():
    days = {f"2026-11-0{i}": synth(40 + i, n=6000, sd=0.6) for i in (5, 6)}
    tr, te = days["2026-11-05"], days["2026-11-06"]
    spec = {"name": "c", "late": True, "target": "win", "features": ["sp"], "model": {"kind": "clogit", "lam": 1.0}}
    f = cl.fit_predict(tr, te, spec)
    w = cl.won(te) == 1
    ll = lambda fc: float(-np.log(cl.win_chances(te, fc)[w]).mean())
    assert ll(f) < ll(te.l_mid.to_numpy())                                      # it beats the mid on the result
    assert np.allclose(pd.Series(np.exp(-f)).groupby([te.market, te.check]).sum(), 1.0)
    ss = cl.second_stage(te, te.y.to_numpy())                                   # the SP itself beside the mid
    assert ss["races"] == 1000 and ss["b_forecast"] > 0.5 and ss["lr_p"] < 0.01 and ss["gain"] > 0
    flat = cl.second_stage(te, te.l_mid.to_numpy() + np.random.default_rng(0).normal(0, 0.3, len(te)))
    assert flat["lr_p"] > 0.01                                                  # noise beside the mid adds nothing


def test_the_family_of_tests_is_read_as_one():
    from research.greyhound import multitest as mt
    assert np.allclose(mt.bh([0.01, 0.02, 0.03, 0.5, np.nan])[:4], [0.04, 0.04, 0.04, 0.5])
    assert np.isclose(mt.p_one_sided([0.0])[0], 0.5) and mt.p_one_sided([3.0])[0] < 0.0014
    assert mt.z_by_race(np.r_[np.ones(20), -np.ones(20)]) is None            # too few races to mean anything
    assert mt.z_by_race(np.full(200, 0.98)) is None                          # every lay won: no spread, no evidence
    assert mt.z_by_race(np.r_[np.ones(50), -np.ones(50)]) == 0.0
