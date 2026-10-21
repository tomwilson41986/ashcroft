"""The owner's ask (21 Oct): "replay the combinations" of move-model-1021 through the trader's own code (read-only).

done/blend_ladder_replay_1020.py's replay (the trader's session a minute at a time against the recorded books, fills
against each level's own money, GBP5,000 in the account, GBP4,000 a day, Betfair's settled prices) on the live config
for 22 Oct (the ladder, the guard, nothing over 200, at most 5 top-ups), 14-20 Oct as the price files allow. For each
replayed day the move models are fitted on the trader's books of the days before (from 7 Oct, every 10th read, the
CLV of backing at the best back now as the target, move_model_1021's 42 features) and predict every read of the day;
when the session plans a race, each runner takes the prediction of the latest read at or before that minute. Arms:

- now: the served closing model's expected CLV, as live;
- LightGBM veto: no back where LightGBM expects the CLV under 0;
- ridge veto: the same with ridge;
- ridge average: the expected CLV is the served model's and ridge's average (and the ladder reads its levels by it).

Read the CLV x stake (the hedged backs' expected profit before commission) and the days ahead.
"""
import os
import sys

path = os.path.join(os.getcwd(), "research", "queries", "done", "blend_ladder_replay_1020.py")
src = open(path, encoding="utf-8").read()
src = src.replace("REPLAY_DAYS = [date(2026, 10, d) for d in range(12, 20)]",
                  "REPLAY_DAYS = [date(2026, 10, d) for d in range(14, 21)]")
old = "    if key not in _cache:\n"
assert old in src
src = src.replace(old, "    if MODE.get('ml') or key not in _cache:\n", 1)
ML = r'''
# ============================================================== the move models (combo_replay_1021)
import io as _io
import re as _re
import bisect as _bisect
import lightgbm as _lgb
from sklearn.linear_model import Ridge as _Ridge
from sklearn.preprocessing import StandardScaler as _Scaler
from betfair_prices import normalise_horse as _nh, normalise_track as _nt, race_time_to_24h as _rt, utc_iso_to_uk_hhmm as _uk

MODE, CONTEXT, PRED = {"ml": None}, {}, {}
_AW = {"kempton", "wolverhampton", "lingfield", "chelmsford", "chelmsfordcity", "southwell", "newcastle", "dundalk"}
_MLDAYS = [(date(2026, 10, 7) + timedelta(n)).isoformat() for n in range(14)]
_text = {c: str for c in ("date", "race_date", "venue", "track", "race_time", "runner_name", "horse_name")}
_pf = []
for _d in _MLDAYS:
    _raw = get(f"predictions/{_d}.csv")
    if _raw is not None:
        _x = pd.read_csv(_io.BytesIO(_raw), dtype=_text, low_memory=False)
        _x["file_day"] = _d
        _pf.append(_x)
_p = pd.concat(_pf, ignore_index=True)
for _a, _b in (("race_date", "date"), ("track", "venue"), ("horse_name", "runner_name")):
    if _a not in _p.columns and _b in _p.columns:
        _p[_a] = _p[_b]
_p["race_date"] = _p["race_date"].fillna(_p["file_day"]).astype(str).str.slice(0, 10)
_p["predicted_bfsp"] = pd.to_numeric(_p["predicted_bfsp"], errors="coerce")
_p["_k"] = (_p.race_date + "|" + _p.track.map(_nt) + "|" + _p.race_time.map(_rt).astype(str) + "|" + _p.horse_name.map(_nh))
_p = _p.drop_duplicates("_k", keep="last")[["_k", "predicted_bfsp"]]
_mk = pd.read_sql_query(f"""SELECT * FROM betfair_live_markets WHERE race_date >= '{_MLDAYS[0]}' AND race_date <= '{_MLDAYS[-1]}'
                            AND UPPER(COALESCE(market_type, 'WIN')) = 'WIN'""", conn)
_mk["horse"] = _mk.runner_name.map(_nh)
_mk["_k"] = (_mk.race_date + "|" + _mk.venue.map(_nt) + "|" + _mk.market_start_utc.map(_uk).astype(str) + "|" + _mk.horse)
_mk = _mk.merge(_p, on="_k", how="left")
_mk["selection_id"] = pd.to_numeric(_mk.selection_id, errors="coerce")
_fin = pd.read_sql_query(f"""SELECT market_id, selection_id, runner_status AS final_status, bsp FROM betfair_live_marks
                             WHERE mark = 'final' AND race_date >= '{_MLDAYS[0]}' AND race_date <= '{_MLDAYS[-1]}'""", conn)
_fin["selection_id"] = pd.to_numeric(_fin.selection_id, errors="coerce")
_mk = _mk.merge(_fin, on=["market_id", "selection_id"], how="left")
_mk["ran"] = _mk.final_status.isin(["WINNER", "LOSER"])
_num = lambda c: pd.to_numeric(_mk.get(c), errors="coerce")  # noqa: E731
_mk["fc_price"] = 1 + _num("forecast_num") / _num("forecast_den")
_mk["or_"], _mk["wt"], _mk["age_"], _mk["draw"], _mk["dslr"] = (_num(c) for c in (
    "official_rating", "weight_value", "age", "stall_draw", "days_since_last_run"))
_mk["headgear"] = _mk.wearing.fillna("").astype(str).str.len().gt(0).astype(float)
_form = _mk.form.fillna("").astype(str)
_mk["form_last"] = _form.str[-1:].map(lambda s: float(s) if s.isdigit() else (10.0 if s else np.nan)).replace(0, 10.0)
_mk["form_wins"], _mk["form_len"] = _form.str.count("1"), _form.str.len()
_nm, _rty = _mk.market_name.fillna("").str.lower(), _mk.race_type.fillna("").str.lower()
_mk["jumps"] = (_rty.str.contains("hurdle|chase|bumper|nh") | _nm.str.contains(r"hrd|chs|nhf|hurdle|chase")).astype(float)
_mk["chase"] = (_nm.str.contains("chs|chase") | _rty.str.contains("chase")).astype(float)
_mk["hcap"] = _nm.str.contains("hcap|handicap|nursery").astype(float)
_mk["beginners"] = _nm.str.contains("beg").astype(float)
_mk["maiden_novice"] = _nm.str.contains("mdn|maiden|nov").astype(float)
_mk["ie"] = _mk.country.eq("IE").astype(float)
_mk["aw"] = (~_mk.jumps.astype(bool) & _mk.venue.fillna("").str.lower().str.replace(r"[^a-z]", "", regex=True).isin(_AW)).astype(float)


def _furlongs(name):
    m = _re.search(r"(\d+)m(?:\s*(\d+)f)?|(\d+)f", name)
    if not m:
        return np.nan
    if m.group(3):
        return float(m.group(3))
    return 8.0 * float(m.group(1)) + (float(m.group(2)) if m.group(2) else 0.0)


_mk["furlongs"] = _nm.map(_furlongs)
_mk["field"] = _mk.groupby("market_id").selection_id.transform("size")
_runs = _mk[_mk.ran & (_mk.race_date >= "2026-10-14")].groupby("horse").race_date.apply(sorted).to_dict()
_mk["stale"] = [float(any(x < day for x in _runs.get(h, []))) for h, day in zip(_mk.horse, _mk.race_date)]
_mk["start"] = pd.to_datetime(_mk.market_start_utc, utc=True, errors="coerce")
_CAT = ["fc_price", "or_", "wt", "age_", "draw", "dslr", "headgear", "form_last", "form_wins", "form_len", "jumps",
        "chase", "hcap", "beginners", "maiden_novice", "ie", "aw", "furlongs", "field", "stale"]
_BOOK = ["polled_utc", "market_id", "selection_id", "status", "inplay", "runner_status", "back1", "back1_size", "back2",
         "back2_size", "back3", "back3_size", "lay1", "lay1_size", "lay2", "lay2_size", "lay3", "lay3_size",
         "market_total_matched", "runner_total_matched", "last_traded"]
_F = ["lq_now", "lq_model", "overlay", "book", "l_back", "spread", "thin", "wom", "l_back1_size", "ltp_gap",
      "l_runner_matched", "l_market_matched", "runner_share", "minutes_to_off", "mv_open", "mv_10", "mv_30", "vol_10",
      "traded_10", "fc_vs_now", "model_rank", "market_rank"] + _CAT


def _features(day):
    """Every read of the day's trader books with the move models' features (and the target where the BSP is known)."""
    raw = get(f"betfair_live/{day}/books_trader.csv.gz")
    if raw is None:
        return None
    bk = pd.read_csv(_io.BytesIO(raw), compression="gzip", usecols=lambda c: c in _BOOK, dtype={"market_id": str},
                     low_memory=False)
    for c in _BOOK[6:] + ["selection_id"]:
        if c in bk.columns:
            bk[c] = pd.to_numeric(bk[c], errors="coerce")
    ip = bk.inplay.astype(str).str.lower().isin(["1", "true", "1.0"])
    bk = bk[(bk.status == "OPEN") & ~ip & bk.runner_status.fillna("ACTIVE").eq("ACTIVE") & (bk.back1 > 1)].copy()
    bk["t"] = pd.to_datetime(bk.polled_utc, utc=True, errors="coerce")
    bk = bk.sort_values(["market_id", "selection_id", "t"])
    gr = bk.groupby(["market_id", "selection_id"])
    lb = np.log(bk.back1)
    bk["mv_open"] = lb - np.log(gr.back1.transform("first"))
    bk["mv_10"] = lb - np.log(gr.back1.shift(10))
    bk["mv_30"] = lb - np.log(gr.back1.shift(30))
    bk["vol_10"] = lb.groupby([bk.market_id, bk.selection_id]).transform(lambda s: s.diff().rolling(10, min_periods=3).std())
    bk["traded_10"] = np.log1p(bk.runner_total_matched.fillna(0)) - np.log1p(gr.runner_total_matched.shift(10).fillna(0))
    bk = bk.merge(_mk[["market_id", "selection_id", "predicted_bfsp", "bsp", "final_status", "start"] + _CAT],
                  on=["market_id", "selection_id"], how="left")
    bk["snap"] = bk.market_id + "|" + bk.polled_utc.astype(str)
    bk = bk[bk.predicted_bfsp > 1].copy()
    g_ = rb.race_codes(bk.snap)
    bk["now"] = rb.market_now(bk.back1.to_numpy(float), bk.lay1.to_numpy(float))
    bk["lq_now"] = np.log(rb.normalise_within(1 / bk.now.to_numpy(float), g_))
    bk["lq_model"] = np.log(rb.normalise_within(1 / bk.predicted_bfsp.to_numpy(float), g_))
    bk["overlay"] = np.log(bk.back1 / bk.predicted_bfsp)
    bk["book"] = np.bincount(g_, weights=1 / bk.now.to_numpy(float))[g_]
    bk["l_back"] = np.log(bk.back1)
    bk["spread"] = np.clip(np.log(np.where(bk.lay1 > 1, bk.lay1, np.e * bk.back1) / bk.back1), 0, 1)
    bk["thin"] = np.clip(np.log(bk.back1 / np.where(bk.back2 > 1, bk.back2, bk.back1 / np.e)), 0, 1)
    ob = bk[["back1_size", "back2_size", "back3_size"]].fillna(0).sum(axis=1)
    ol = bk[["lay1_size", "lay2_size", "lay3_size"]].fillna(0).sum(axis=1)
    bk["wom"] = np.clip(np.log((ol + 1) / (ob + 1)), -5, 5)
    bk["l_back1_size"] = np.log1p(bk.back1_size.fillna(0))
    bk["ltp_gap"] = np.clip(np.log(bk.back1 / np.where(bk.last_traded > 1, bk.last_traded, bk.back1)), -1, 1)
    bk["l_runner_matched"] = np.log1p(bk.runner_total_matched.fillna(0))
    bk["l_market_matched"] = np.log1p(bk.market_total_matched.fillna(0))
    bk["runner_share"] = bk.runner_total_matched.fillna(0) / bk.market_total_matched.where(bk.market_total_matched > 0)
    bk["minutes_to_off"] = (bk.start - bk.t).dt.total_seconds() / 60
    bk["fc_vs_now"] = np.log(bk.fc_price / bk.back1)
    bk["model_rank"] = bk.groupby("snap").predicted_bfsp.rank()
    bk["market_rank"] = bk.groupby("snap").back1.rank()
    bk["z"] = np.clip(bk.back1 / bk.bsp - 1, -0.9, 3.0)
    bk["day"] = day
    return bk


_FEAT = {}
for _d in _MLDAYS:
    _x = _features(_d)
    if _x is not None:
        _FEAT[_d] = _x
        print(f"  move-model features {_d}: {len(_x):,} runner-reads", flush=True)


def _train(day):
    """The training rows of the days before `day`: every 10th read of each market, whole reads with a BSP."""
    parts = []
    for d, x in _FEAT.items():
        if d >= day:
            continue
        reads = x[["market_id", "polled_utc"]].drop_duplicates().sort_values(["market_id", "polled_utc"])
        reads = reads[reads.groupby("market_id").cumcount() % 10 == 0]
        y = x.merge(reads, on=["market_id", "polled_utc"])
        okr = (y.bsp > 1) & y.final_status.isin(["WINNER", "LOSER"])
        y = y[okr.groupby(y.snap).transform("all")]
        parts.append(y)
    return pd.concat(parts, ignore_index=True)


def ml_prepare(day):
    """Fit on the days before `day`, predict every read of `day`; PRED: selection -> (times, LightGBM, ridge)."""
    PRED.clear()
    tr = _train(day)
    te = _FEAT.get(day)
    if te is None or tr.empty:
        print(f"  {day}: no move-model predictions")
        return
    med = tr[_F].median()
    sc = _Scaler().fit(tr[_F].fillna(med))
    ridge = _Ridge(alpha=10.0).fit(sc.transform(tr[_F].fillna(med)), tr.z)
    gbm = _lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.7,
                             subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1).fit(tr[_F], tr.z)
    te = te.copy()
    te["p_ridge"] = ridge.predict(sc.transform(te[_F].fillna(med)))
    te["p_gbm"] = gbm.predict(te[_F])
    te = te.sort_values("t")
    for sid, x in te.groupby("selection_id"):
        PRED[int(sid)] = (list(x.t), x.p_gbm.to_numpy(float), x.p_ridge.to_numpy(float))
    print(f"  {day}: move models fitted on {len(tr):,} runner-reads, predicting {len(te):,}; LightGBM under 0 on "
          f"{(te.p_gbm < 0).mean():.0%} of reads, ridge {(te.p_ridge < 0).mean():.0%}", flush=True)


def _lookup(sid, now):
    got = PRED.get(int(sid))
    if not got or now is None:
        return np.nan, np.nan
    times, pg, pr = got
    i = _bisect.bisect_right(times, pd.Timestamp(now)) - 1
    if i < 0 or pd.Timestamp(now) - times[i] > pd.Timedelta("10min"):
        return np.nan, np.nan
    return pg[i], pr[i]


_served_cache = {}
_before_ml = rb.race_expected_clv


def ml_rec(back, lay, model_price, matched, model, n_draws=50000, rng=None, market=None):
    key = (tuple(np.round(np.asarray(back, float), 4)), tuple(np.round(np.asarray(lay, float), 4)),
           tuple(np.round(np.asarray(model_price, float), 4)))
    if key not in _served_cache:
        _served_cache[key] = _before_ml(back, lay, model_price, matched, model, n_draws=n_draws, rng=rng, market=market)
    ev, exp_bsp = (np.array(a, float) for a in _served_cache[key])
    mode = MODE.get("ml")
    sids = CONTEXT.get("sids") or []
    if not mode or len(sids) != len(ev):
        return ev, exp_bsp
    preds = [_lookup(s, CONTEXT.get("now")) for s in sids]
    pg = np.array([a for a, _ in preds], float)
    pr = np.array([b for _, b in preds], float)
    back = np.asarray(back, float)
    if mode == "veto_gbm":
        ev = np.where(np.isfinite(pg) & (pg < 0), -1.0, ev)
    elif mode == "veto_ridge":
        ev = np.where(np.isfinite(pr) & (pr < 0), -1.0, ev)
    elif mode == "avg_ridge":
        new = np.where(np.isfinite(pr), (ev + pr) / 2, ev)
        with np.errstate(invalid="ignore", divide="ignore"):
            exp_bsp = np.where(np.isfinite(new) & (new > -1), back / (1 + new), exp_bsp)
        ev = new
    return ev, exp_bsp


rb.race_expected_clv = ml_rec


class MLSession(LadderSession):
    def plan(self, m, book):
        CONTEXT["now"] = self.clock()
        if book is not None:
            CONTEXT["sids"] = [sid for sid, q in book.runners.items() if q.status == "ACTIVE"]
        return super().plan(m, book)


g["PolicySession"] = MLSession
'''
a, b = src.index("ARMS = ["), src.index("]\nrows = []")
src = src[:a] + ML + '''
ARMS = [  # name, blend, policy
    ("now (the rule of 22 Oct)", False, {"ml": None}),
    ("LightGBM veto", False, {"ml": "veto_gbm"}),
    ("ridge veto", False, {"ml": "veto_ridge"}),
    ("ridge average", False, {"ml": "avg_ridge"}),
''' + src[b:]
old = "    fit = fit_blend(L[L.day < day.isoformat()])\n"
assert old in src
src = src.replace(old, "    fit = None\n    ml_prepare(day.isoformat())\n")
src = src.replace("""    print(f"  the spread blend for {day} (fitted on 1 Oct - the day before): weights {np.round(fit['beta'], 3)}, "
          f"book {fit['book']:.4f}")
""", "")
old = """        BLEND["fit"], BLEND["tag"] = (fit if blend else None), (f"blend{day}" if blend else "served")\n"""
assert old in src
src = src.replace(old, old + """        MODE["ml"] = pol.get("ml")
        BLEND["tag"] = f"served_{day}_{MODE['ml']}"
""")
src = src.replace('"out/blend_ladder_replay"', '"out/combo_replay"')
src += '''
print("\\n== the spread of the days (result after commission)")
print(res.groupby("arm", sort=False).agg(mean_day=("result_after_commission", "mean"),
                                         sd_day=("result_after_commission", "std"),
                                         worst_day=("result_after_commission", "min")).round(2).to_string())
cx = res.pivot_table(index="day", columns="arm", values="clv_x_stake", sort=False)
base = cx["now (the rule of 22 Oct)"]
print("\\n== days ahead of now (CLV x stake):", {c: int((cx[c] > base).sum()) for c in cx.columns})
'''
sys.argv = ["combo_replay_1021.py"]
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
