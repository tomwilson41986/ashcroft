"""The owner's ask (21 Oct): an updated model from the live trading data and what we know: "Let's prototype improving
the model accuracy at predicting BFSP and which horses will shorten", with statistics, machine learning, deep learning
(read-only research; nothing served).

The BFSP model (106 iterations on horseracebase's history) is level on every recent test: its features are the form
book. What it never sees is the market's own information, which the live record holds for 1-20 Oct: the book at every
read (three levels each side and their money, the spread, the last traded price, the runner's and market's matched
money, the price's path so far) and Betfair's catalogue (its forecast price, the rating, weight, draw, age, days since
the last run, headgear, the race's type, handicap, field, trip, country). Two prototypes, walk-forward by day (each day
scored by fits on the days before it, from the eighth day), against what the trader uses now:

1. The move: which horses will shorten. Target: the CLV of backing at the best back now, back / BSP - 1 (clipped),
   at every 10th read of each market. Models: the served closing model (0.77 market + 0.26 ours, its expected CLV),
   a ridge regression, LightGBM and a neural net (an MLP) on the same features. Scored by the correlation with the
   CLV that came, the calibration, the AUC of "shortens 5% or more", and the owner's rule on each (the first read at
   which a runner clears 3%, over our price, at most 200; to win GBP400, GBP15-200): its CLV and money; and at the same
   number of horses a day as the served model picks (quality at equal volume).
2. The 06:00 price: the model's predicted BFSP corrected by the catalogue and the race (one row a runner, no market
   price): |log(price / BSP)| for the model as it is, a ridge and LightGBM correction.
"""
import io
import os
import re
import sqlite3
import sys
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from betfair_prices import normalise_horse, normalise_track, race_time_to_24h, utc_iso_to_uk_hhmm  # noqa: E402
from model import race_book as rb  # noqa: E402

import lightgbm as lgb  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
DAYS = [(date(2026, 10, 1) + timedelta(n)).isoformat() for n in range(20)]
EVERY, BAR = 10, 0.03
AW = {"kempton", "wolverhampton", "lingfield", "chelmsford", "chelmsfordcity", "southwell", "newcastle", "dundalk"}


def get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ the 06:00 prices and the catalogue
text = {c: str for c in ("date", "race_date", "venue", "track", "race_time", "runner_name", "horse_name")}
pf = []
for day in DAYS:
    raw = get(f"predictions/{day}.csv")
    if raw is not None:
        x = pd.read_csv(io.BytesIO(raw), dtype=text, low_memory=False)
        x["file_day"] = day
        pf.append(x)
p = pd.concat(pf, ignore_index=True)
for a, b in (("race_date", "date"), ("track", "venue"), ("horse_name", "runner_name")):
    if a not in p.columns and b in p.columns:
        p[a] = p[b]
p["race_date"] = p["race_date"].fillna(p["file_day"]).astype(str).str.slice(0, 10)
p["predicted_bfsp"] = pd.to_numeric(p["predicted_bfsp"], errors="coerce")
p["_k"] = (p.race_date + "|" + p.track.map(normalise_track) + "|" + p.race_time.map(race_time_to_24h).astype(str) + "|"
           + p.horse_name.map(normalise_horse))
p = p.drop_duplicates("_k", keep="last")[["_k", "predicted_bfsp"]]
mk = pd.read_sql_query(f"""SELECT * FROM betfair_live_markets WHERE race_date >= '{DAYS[0]}' AND race_date <= '{DAYS[-1]}'
                           AND UPPER(COALESCE(market_type, 'WIN')) = 'WIN'""", conn)
mk["horse"] = mk.runner_name.map(normalise_horse)
mk["_k"] = (mk.race_date + "|" + mk.venue.map(normalise_track) + "|" + mk.market_start_utc.map(utc_iso_to_uk_hhmm).astype(str)
            + "|" + mk.horse)
mk = mk.merge(p, on="_k", how="left")
mk["selection_id"] = pd.to_numeric(mk.selection_id, errors="coerce")
fin = pd.read_sql_query(f"""SELECT market_id, selection_id, runner_status AS final_status, bsp FROM betfair_live_marks
                            WHERE mark = 'final' AND race_date >= '{DAYS[0]}' AND race_date <= '{DAYS[-1]}'""", conn)
fin["selection_id"] = pd.to_numeric(fin.selection_id, errors="coerce")
mk = mk.merge(fin, on=["market_id", "selection_id"], how="left")
mk["ran"] = mk.final_status.isin(["WINNER", "LOSER"])
# the catalogue's features
num = lambda c: pd.to_numeric(mk.get(c), errors="coerce")  # noqa: E731
mk["fc_price"] = 1 + num("forecast_num") / num("forecast_den")
mk["or_"] = num("official_rating")
mk["wt"] = num("weight_value")
mk["age_"] = num("age")
mk["draw"] = num("stall_draw")
mk["dslr"] = num("days_since_last_run")
mk["headgear"] = mk.wearing.fillna("").astype(str).str.len().gt(0).astype(float)
form = mk.form.fillna("").astype(str)
mk["form_last"] = form.str[-1:].map(lambda s: float(s) if s.isdigit() else (10.0 if s else np.nan)).replace(0, 10.0)
mk["form_wins"] = form.str.count("1")
mk["form_len"] = form.str.len()
nm = mk.market_name.fillna("").str.lower()
rtype = mk.race_type.fillna("").str.lower()
mk["jumps"] = (rtype.str.contains("hurdle|chase|bumper|nh") | nm.str.contains(r"hrd|chs|nhf|hurdle|chase")).astype(float)
mk["chase"] = (nm.str.contains("chs|chase") | rtype.str.contains("chase")).astype(float)
mk["hcap"] = nm.str.contains("hcap|handicap|nursery").astype(float)
mk["beginners"] = nm.str.contains("beg").astype(float)
mk["maiden_novice"] = nm.str.contains("mdn|maiden|nov").astype(float)
mk["ie"] = mk.country.eq("IE").astype(float)
mk["aw"] = (~mk.jumps.astype(bool) & mk.venue.fillna("").str.lower().str.replace(r"[^a-z]", "", regex=True).isin(AW)).astype(float)


def furlongs(name):
    m = re.search(r"(\d+)m(?:\s*(\d+)f)?|(\d+)f", name)
    if not m:
        return np.nan
    if m.group(3):
        return float(m.group(3))
    return 8.0 * float(m.group(1)) + (float(m.group(2)) if m.group(2) else 0.0)


mk["furlongs"] = nm.map(furlongs)
mk["field"] = mk.groupby("market_id").selection_id.transform("size")
# stale form: ran in one of our markets since 14 Oct and before the day (horseracebase's pause)
runs = mk[mk.ran & (mk.race_date >= "2026-10-14")].groupby("horse").race_date.apply(sorted).to_dict()
mk["stale"] = [float(any(d < day for d in runs.get(h, []))) for h, day in zip(mk.horse, mk.race_date)]
mk["start"] = pd.to_datetime(mk.market_start_utc, utc=True, errors="coerce")
CAT = ["fc_price", "or_", "wt", "age_", "draw", "dslr", "headgear", "form_last", "form_wins", "form_len", "jumps",
       "chase", "hcap", "beginners", "maiden_novice", "ie", "aw", "furlongs", "field", "stale"]
print(f"catalogue 1-20 Oct: {mk.market_id.nunique():,} markets, {len(mk):,} runners; priced at 06:00 "
      f"{mk.predicted_bfsp.notna().mean():.1%}; ran with a BSP {(mk.ran & (mk.bsp > 1)).mean():.1%}; Betfair's forecast "
      f"price on {mk.fc_price.notna().mean():.1%}")

# ------------------------------------------------------------------ 2. the 06:00 price corrected (one row a runner)
print("\n== 2. the 06:00 price corrected by the catalogue (no market price): |log(price / BSP)| on the test days")
r = mk[mk.ran & (mk.bsp > 1) & (mk.predicted_bfsp > 1)].copy()
g = rb.race_codes(r.market_id)
r["lq_model"] = np.log(rb.normalise_within(1 / r.predicted_bfsp.to_numpy(float), g))
r["l_fc"] = np.log(r.fc_price)
r["fc_vs_model"] = np.log(r.fc_price / r.predicted_bfsp)
r["model_rank"] = r.groupby("market_id").predicted_bfsp.rank()
r["y"] = np.log(r.bsp / r.predicted_bfsp)
F2 = ["lq_model", "l_fc", "fc_vs_model", "model_rank"] + CAT
days = sorted(r.race_date.unique())
out = []
for i, d in enumerate(days):
    if i < 7:
        continue
    tr, te = r[r.race_date < d], r[r.race_date == d].copy()
    med = tr[F2].median()
    sc = StandardScaler().fit(tr[F2].fillna(med))
    ridge = Ridge(alpha=10.0).fit(sc.transform(tr[F2].fillna(med)), tr.y)
    te["ridge"] = ridge.predict(sc.transform(te[F2].fillna(med)))
    gbm = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8,
                            subsample_freq=1, colsample_bytree=0.8, verbose=-1).fit(tr[F2], tr.y)
    te["gbm"] = gbm.predict(te[F2])
    te["shift"] = tr.y.mean()                                            # the model's average bias alone
    out.append(te)
T = pd.concat(out, ignore_index=True)
rows = {}
for name, adj in (("the model as served", 0.0), ("its bias removed", T["shift"]), ("ridge correction", T.ridge),
                  ("LightGBM correction", T.gbm)):
    e = T.y - adj
    rows[name] = {"runners": len(T), "mae_log": float(np.abs(e).mean()), "rmse_log": float(np.sqrt((e ** 2).mean())),
                  "spearman_price": float(pd.Series(np.log(T.predicted_bfsp) + adj).corr(np.log(T.bsp), method="spearman"))}
fc = T[T.fc_price > 1]
rows["Betfair's forecast price alone"] = {"runners": len(fc), "mae_log": float(np.abs(np.log(fc.bsp / fc.fc_price)).mean()),
                                          "rmse_log": float(np.sqrt((np.log(fc.bsp / fc.fc_price) ** 2).mean())),
                                          "spearman_price": float(fc.fc_price.corr(fc.bsp, method="spearman"))}
print(pd.DataFrame(rows).T.round(4).to_string())
imp = pd.Series(gbm.feature_importances_, index=F2).sort_values(ascending=False)
print("   the LightGBM correction's splits (last fit):", ", ".join(f"{k} {v}" for k, v in imp.head(10).items()))

# ------------------------------------------------------------------ 1. the move from the book now to the BSP
BOOK = ["polled_utc", "market_id", "selection_id", "status", "inplay", "runner_status", "back1", "back1_size", "back2",
        "back2_size", "back3", "back3_size", "lay1", "lay1_size", "lay2", "lay2_size", "lay3", "lay3_size",
        "market_total_matched", "runner_total_matched", "last_traded"]
parts = []
for day in DAYS:
    raw = get(f"betfair_live/{day}/books_trader.csv.gz")
    if raw is None:
        continue
    bk = pd.read_csv(io.BytesIO(raw), compression="gzip", usecols=lambda c: c in BOOK, dtype={"market_id": str},
                     low_memory=False)
    for c in BOOK[6:] + ["selection_id"]:
        if c in bk.columns:
            bk[c] = pd.to_numeric(bk[c], errors="coerce")
    ip = bk.inplay.astype(str).str.lower().isin(["1", "true", "1.0"])
    bk = bk[(bk.status == "OPEN") & ~ip & bk.runner_status.fillna("ACTIVE").eq("ACTIVE") & (bk.back1 > 1)].copy()
    bk["t"] = pd.to_datetime(bk.polled_utc, utc=True, errors="coerce")
    bk = bk.sort_values(["market_id", "selection_id", "t"])
    gr = bk.groupby(["market_id", "selection_id"])
    lb = np.log(bk.back1)
    bk["mv_open"] = lb - gr.back1.transform("first").pipe(np.log)        # the move since the trader's first read
    bk["mv_10"] = lb - np.log(gr.back1.shift(10))                        # since about 10 reads ago
    bk["mv_30"] = lb - np.log(gr.back1.shift(30))
    bk["vol_10"] = lb.groupby([bk.market_id, bk.selection_id]).transform(lambda s: s.diff().rolling(10, min_periods=3).std())
    bk["traded_10"] = np.log1p(bk.runner_total_matched.fillna(0)) - np.log1p(gr.runner_total_matched.shift(10).fillna(0))
    reads = bk[["market_id", "polled_utc"]].drop_duplicates().sort_values(["market_id", "polled_utc"])
    reads = reads[reads.groupby("market_id").cumcount() % EVERY == 0]
    bk = bk.merge(reads, on=["market_id", "polled_utc"])
    bk["day"] = day
    parts.append(bk)
L = pd.concat(parts, ignore_index=True)
cat_cols = ["market_id", "selection_id", "predicted_bfsp", "bsp", "final_status", "start"] + CAT
L = L.merge(mk[cat_cols], on=["market_id", "selection_id"], how="left")
L["snap"] = L.market_id + "|" + L.polled_utc.astype(str)
ok = (L.predicted_bfsp > 1) & (L.bsp > 1) & L.final_status.isin(["WINNER", "LOSER"])
L = L[ok.groupby(L.snap).transform("all") & (L.groupby("snap").snap.transform("size") >= 2)].copy()
L = L.sort_values(["day", "snap"]).reset_index(drop=True)
g = rb.race_codes(L.snap)
L["now"] = rb.market_now(L.back1.to_numpy(float), L.lay1.to_numpy(float))
L["lq_now"] = np.log(rb.normalise_within(1 / L.now.to_numpy(float), g))
L["lq_model"] = np.log(rb.normalise_within(1 / L.predicted_bfsp.to_numpy(float), g))
L["overlay"] = np.log(L.back1 / L.predicted_bfsp)
L["book"] = np.bincount(g, weights=1 / L.now.to_numpy(float))[g]
L["l_back"] = np.log(L.back1)
L["spread"] = np.clip(np.log(np.where(L.lay1 > 1, L.lay1, np.e * L.back1) / L.back1), 0, 1)
L["thin"] = np.clip(np.log(L.back1 / np.where(L.back2 > 1, L.back2, L.back1 / np.e)), 0, 1)
offb = L[["back1_size", "back2_size", "back3_size"]].fillna(0).sum(axis=1)
offl = L[["lay1_size", "lay2_size", "lay3_size"]].fillna(0).sum(axis=1)
L["wom"] = np.clip(np.log((offl + 1) / (offb + 1)), -5, 5)
L["l_back1_size"] = np.log1p(L.back1_size.fillna(0))
L["ltp_gap"] = np.clip(np.log(L.back1 / np.where(L.last_traded > 1, L.last_traded, L.back1)), -1, 1)
L["l_runner_matched"] = np.log1p(L.runner_total_matched.fillna(0))
L["l_market_matched"] = np.log1p(L.market_total_matched.fillna(0))
L["runner_share"] = L.runner_total_matched.fillna(0) / L.market_total_matched.where(L.market_total_matched > 0)
L["minutes_to_off"] = (L.start - pd.to_datetime(L.polled_utc, utc=True)).dt.total_seconds() / 60
L["fc_vs_now"] = np.log(L.fc_price / L.back1)
L["model_rank"] = L.groupby("snap").predicted_bfsp.rank()
L["market_rank"] = L.groupby("snap").back1.rank()
L["z"] = np.clip(L.back1 / L.bsp - 1, -0.9, 3.0)                          # the CLV of backing at the best back now
L["shortened"] = (L.bsp < L.back1 / 1.05).astype(int)
F1 = ["lq_now", "lq_model", "overlay", "book", "l_back", "spread", "thin", "wom", "l_back1_size", "ltp_gap",
      "l_runner_matched", "l_market_matched", "runner_share", "minutes_to_off", "mv_open", "mv_10", "mv_30", "vol_10",
      "traded_10", "fc_vs_now", "model_rank", "market_rank"] + CAT
print(f"\n== 1. the move: {L.snap.nunique():,} whole reads of {L.market_id.nunique():,} markets, {len(L):,} runner-reads, "
      f"{L.day.min()} .. {L.day.max()}; shortened 5% or more by the off: {L.shortened.mean():.1%}")
served = rb.load_closing_model("data/models/closing_model_novol.json")
days = sorted(L.day.unique())
out, imps = [], []
for i, d in enumerate(days):
    if i < 7:
        continue
    tr, te = L[L.day < d], L[L.day == d].copy()
    gte = rb.race_codes(te.snap)
    inp = rb.closing_inputs(te.assign(race=te.snap, morningwap=te.now, morning_vol=0.0), race_col="race")
    te["p_served"] = te.back1 * served.book * served.predict(inp, gte) - 1
    med = tr[F1].median()
    sc = StandardScaler().fit(tr[F1].fillna(med))
    Xtr, Xte = sc.transform(tr[F1].fillna(med)), sc.transform(te[F1].fillna(med))
    te["p_ridge"] = Ridge(alpha=10.0).fit(Xtr, tr.z).predict(Xte)
    gbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.7,
                            subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1).fit(tr[F1], tr.z)
    te["p_gbm"] = gbm.predict(te[F1])
    imps.append(pd.Series(gbm.booster_.feature_importance("gain"), index=F1))
    take = np.random.default_rng(0).choice(len(tr), size=min(len(tr), 60000), replace=False)
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, max_iter=60,
                       early_stopping=True, random_state=0).fit(Xtr[take], tr.z.to_numpy()[take])
    te["p_mlp"] = mlp.predict(Xte)
    te["p_blend"] = (te.p_gbm + te.p_mlp) / 2
    out.append(te)
    print(f"   {d}: trained on {len(tr):,} runner-reads, scored {len(te):,}", flush=True)
S = pd.concat(out, ignore_index=True)
S["guard"] = S.back1 >= S.predicted_bfsp - 1e-9
MODELS = ["p_served", "p_ridge", "p_gbm", "p_mlp", "p_blend"]
rows = {}
for m in MODELS:
    pr = S[m]
    try:
        auc = roc_auc_score(S.shortened, pr)
    except ValueError:
        auc = np.nan
    sel = S[(pr >= BAR) & S.guard & (S.back1 <= 200)].sort_values("polled_utc").drop_duplicates(
        ["day", "market_id", "selection_id"])
    w = np.clip(400 / (sel.back1 - 1), 15, 200)
    rows[m] = {"corr_with_clv": float(pr.corr(S.z)), "spearman": float(pr.corr(S.z, method="spearman")),
               "auc_shorten_5pct": auc, "mean_pred": float(pr.mean()), "share_over_bar": float((pr >= BAR).mean()),
               "rule_horses": len(sel), "rule_staked": float(w.sum()),
               "rule_clv": float((sel.z * w).sum() / w.sum()) if len(sel) else np.nan,
               "rule_clv_x_stake": float((sel.z * w).sum()), "rule_mean_pred": float(sel[m].mean()) if len(sel) else np.nan}
print(pd.DataFrame(rows).T.round(4).to_string())

print("\n   at equal volume: each model's best N horses a day (N = the served model's horses that day), the first read "
      "at which each clears the served model's day-cut")
served_n = S[(S.p_served >= BAR) & S.guard & (S.back1 <= 200)].drop_duplicates(["day", "market_id", "selection_id"]) \
    .groupby("day").size()
eq = {}
for m in MODELS:
    picks = []
    for d, g_ in S[S.guard & (S.back1 <= 200)].groupby("day"):
        best = g_.sort_values(m, ascending=False).drop_duplicates(["market_id", "selection_id"]).head(int(served_n.get(d, 0)))
        picks.append(best)
    sel = pd.concat(picks)
    w = np.clip(400 / (sel.back1 - 1), 15, 200)
    eq[m] = {"horses": len(sel), "staked": float(w.sum()), "clv": float((sel.z * w).sum() / w.sum()),
             "clv_x_stake": float((sel.z * w).sum())}
print(pd.DataFrame(eq).T.round(4).to_string())

print("\n   calibration: the CLV that came (mean back / BSP - 1) by each model's prediction")
cal = {}
for m in MODELS:
    e = pd.cut(S[m], [-1, -0.10, -0.03, 0, 0.03, 0.06, 0.10, 0.20, 100])
    cal[m] = S.groupby(e, observed=True).z.mean()
    cal[f"n_{m}"] = S.groupby(e, observed=True).size()
print(pd.DataFrame(cal).round(4).to_string())

imp = pd.concat(imps, axis=1).mean(axis=1)
imp = (imp / imp.sum()).sort_values(ascending=False)
print("\n   what LightGBM reads to predict the move (share of split gain, averaged over the fits):")
print(imp.round(4).head(25).to_string())
print("\n   by day: the rule's CLV for the served model and LightGBM")
print(S.assign(day_=S.day).groupby("day_").apply(lambda g_: pd.Series({m: float((lambda s: (s.z * np.clip(400 / (s.back1 - 1), 15, 200)).sum() / np.clip(400 / (s.back1 - 1), 15, 200).sum() if len(s) else np.nan)(
    g_[(g_[m] >= BAR) & g_.guard & (g_.back1 <= 200)].sort_values("polled_utc").drop_duplicates(["market_id", "selection_id"])))
    for m in ("p_served", "p_gbm", "p_blend")})).round(4).to_string())
