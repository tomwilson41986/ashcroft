"""The owner (22 Oct): "which type of races our model performs poorly in and whether we can do race type adjustments
for them" (read-only).

The sample: the served blend's walk-forward predictions (iteration 110's gate_step: the three members as served,
27 Sep 2025 - 31 Mar 2026, 53,910 runners; research/data/oos_served_iter110.csv.gz), joined to Betfair's price files
(morning WAP, its matched money, BSP). The 06:00 files of August and September are not the sample: they were older
models (an error of about 0.9 in log against 0.4 today). The locked holdout (from 1 Apr 2026) is not read.

1. Where the model prices worst: by code, kind, code x kind, country, class, field size and price band, our price's
   error against the BSP (within-race mean absolute log error, and its bias: above zero, our price longer than the BSP), the
   morning WAP's error, and the model's skill against it (1 - our error / the market's); and the blend the closing
   model would fit in that segment alone (the weights on the market's log chance and ours).
2. Race-type adjustments, walk-forward by month (each month scored by fits on the months before, from the third):
   - one blend for every race (the closing model's form refitted: market and ours);
   - the blend with the weights by race type (each segment's own weights on the market and ours, shrunk to the
     common blend by L2);
   - the one blend, races of a type skipped where the months before closed below zero on the rule.
   Each scored by nearness to the BSP book out of sample (cross-entropy per race against the one blend) and by the
   owner's rule on it: backed at the morning WAP where the expected CLV is at least 3%, the WAP at or over our price
   and GBP100 matched on the horse, staked to win GBP400 (GBP15-200), CLV at the BSP. The price files' WAP is not
   cut for a later non-runner; every arm alike.
"""
import io
import os
import sqlite3
import sys
from datetime import timedelta

import boto3
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, ".")
from betfair_prices import normalise_horse, normalise_track, race_time_to_24h  # noqa: E402
from model import race_book as rb  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
BAR = 0.03
IRE = {"curragh", "leopardstown", "fairyhouse", "punchestown", "navan", "naas", "gowranpark", "cork", "galway",
       "killarney", "limerick", "listowel", "tipperary", "tramore", "wexford", "downroyal", "downpatrick", "dundalk",
       "bellewstown", "ballinrobe", "clonmel", "kilbeggan", "laytown", "roscommon", "sligo", "thurles"}


def get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001
        return None


def preds(d0, d1):
    text = {c: str for c in ("date", "race_date", "venue", "track", "race_time", "runner_name", "horse_name")}
    out, d = [], d0
    while d <= d1:
        raw = get(f"predictions/{d.isoformat()}.csv")
        if raw is not None:
            x = pd.read_csv(io.BytesIO(raw), dtype=text, low_memory=False)
            x["file_day"] = d.isoformat()
            out.append(x)
        d += timedelta(days=1)
    p = pd.concat(out, ignore_index=True)
    for a, b in (("race_date", "date"), ("track", "venue"), ("horse_name", "runner_name")):
        if a not in p.columns and b in p.columns:
            p[a] = p[b]
    p["race_date"] = p["race_date"].fillna(p["file_day"]).astype(str).str.slice(0, 10)
    p["predicted_bfsp"] = pd.to_numeric(p["predicted_bfsp"], errors="coerce")
    p["_k"] = (p.race_date + "|" + p.track.map(normalise_track) + "|" + p.race_time.map(race_time_to_24h).astype(str)
               + "|" + p.horse_name.map(normalise_horse))
    return p.drop_duplicates("_k", keep="last")[["_k", "predicted_bfsp"]]


def lq(price, g):
    return np.log(rb.normalise_within(1.0 / np.asarray(price, float), g))


def fit(X, q, g, l2=1e-4, prior=None):
    """Conditional logit by cross-entropy against the BSP book q; L2 toward `prior` (zero by default)."""
    nr = g.max() + 1
    p0 = np.zeros(X.shape[1]) if prior is None else np.asarray(prior, float)

    def loss(b):
        lqh = rb.group_log_softmax(X @ b, g)
        return (-float(np.sum(q * lqh)) / nr + 0.5 * l2 * float((b - p0) @ (b - p0)),
                -((q - np.exp(lqh))[:, None] * X).sum(axis=0) / nr + l2 * (b - p0))

    b0 = p0.copy()
    if prior is None:
        b0[:2] = 0.5
    return minimize(loss, b0, jac=True, method="L-BFGS-B").x


def xent(X, b, q, g):
    return -float(np.sum(q * rb.group_log_softmax(X @ b, g))) / (g.max() + 1)


# ------------------------------------------------------------------ the served blend's walk-forward predictions
o = pd.read_csv("research/data/oos_served_iter110.csv.gz", dtype={"race_time": str})
o["_k"] = (o.race_date.astype(str).str.slice(0, 10) + "|" + o.track.map(normalise_track) + "|"
           + o.race_time.map(race_time_to_24h).astype(str) + "|" + o.horse_name.map(normalise_horse))
bf = pd.read_sql_query("""SELECT r.race_date, r.track, r.race_time, r.horse_name, b.event_id, b.bsp, b.morningwap,
                                 b.morning_vol
                          FROM betfair_prices b JOIN race_results r ON r.id = b.race_results_id
                          WHERE LOWER(b.market_type) = 'win' AND r.race_date >= '2025-09-27'
                          AND r.race_date < '2026-04-01'""", conn)
bf["_k"] = (bf.race_date.astype(str).str.slice(0, 10) + "|" + bf.track.map(normalise_track) + "|"
            + bf.race_time.map(race_time_to_24h).astype(str) + "|" + bf.horse_name.map(normalise_horse))
h = o.merge(bf.drop_duplicates("_k", keep=False)[["_k", "event_id", "bsp", "morningwap", "morning_vol"]], on="_k")
print(f"the served blend's predictions: {len(o):,} runners; joined to Betfair's price files {len(h):,}")
h["day"] = h.race_date.astype(str).str.slice(0, 10)
h["race"] = h.day + "|" + h.event_id.astype(str)
ok = (h.bsp > 1) & (h.predicted_bfsp > 1) & (h.morningwap > 1)
field = h.groupby("race").race.transform("size")
h = h[ok.groupby(h.race).transform("all") & (field >= 2)].sort_values(["day", "race"]).reset_index(drop=True)
h["field"] = h.groupby("race").race.transform("size")
rt = h.race_type.fillna("").str.lower()
rc = h.race_code.fillna("").str.lower()
h["code"] = np.select([rt.str.contains("nh flat"), rt.str.contains("chase"), rt.str.contains("hurdle"),
                       rc.str.contains("all weather")], ["NH flat", "chase", "hurdle", "all-weather"], "flat turf")
h["kind"] = np.select([rt.str.contains("hunters"), rt.str.contains("beginners"), rt.str.contains("seller|claimer"),
                       rt.str.contains("handicap") & rt.str.contains("novices"), rt.str.contains("handicap"),
                       rt.str.contains("maiden"), rt.str.contains("novices"), rt.str.contains("nh flat")],
                      ["hunters", "beginners", "claimer/seller", "novice handicap", "handicap", "maiden", "novice",
                       "bumper"], "other non-handicap")
h["country"] = np.where(h.race_class.eq("Irish"), "Ireland", "GB")
h["class"] = h.race_class.fillna("?")
h["fieldband"] = pd.cut(h.field, [0, 6, 9, 12, 16, 100], labels=["2-6", "7-9", "10-12", "13-16", "17+"])
h["band"] = pd.cut(h.bsp, [1, 3, 6, 11, 21, 51, 1001], right=False, labels=["<3", "3-6", "6-11", "11-21", "21-51", "51+"])
h["seg"] = h.code + " | " + np.where(h.kind.isin(["handicap", "novice handicap"]), "handicap",
                                     np.where(h.kind.isin(["maiden", "novice", "beginners", "bumper"]),
                                              "maiden/novice/beg/bumper", "other"))
g = rb.race_codes(h.race)
h["g"] = g
h["e_model"] = np.log(h.predicted_bfsp / h.bsp)
h["e_mkt"] = np.log(h.morningwap / h.bsp)
h["lq_m"], h["lq_f"], h["q"] = lq(h.morningwap, g), lq(h.predicted_bfsp, g), rb.normalise_within(1 / h.bsp.to_numpy(float), g)
# the model's error once each race's level is taken out (what a within-race forecast is judged on)
h["e_model_w"] = h.e_model - h.groupby("race").e_model.transform("mean")
h["e_mkt_w"] = h.e_mkt - h.groupby("race").e_mkt.transform("mean")
print(f"{h.race.nunique():,} races, {len(h):,} runners, {h.day.min()} .. {h.day.max()}")


def seg_table(col):
    def f(x):
        gg = rb.race_codes(x.race)
        b = fit(x[["lq_m", "lq_f"]].to_numpy(float), x.q.to_numpy(float), gg) if x.race.nunique() >= 40 else [np.nan] * 2
        return pd.Series({"races": x.race.nunique(), "runners": len(x),
                          "model_mae": x.e_model_w.abs().mean(), "model_bias": x.e_model.mean(),
                          "market_mae": x.e_mkt_w.abs().mean(), "skill_vs_market": 1 - x.e_model_w.abs().mean() / x.e_mkt_w.abs().mean(),
                          "w_market": b[0], "w_model": b[1]})
    return h.groupby(col, observed=True).apply(f).sort_values("model_mae", ascending=False)


print("\n== 1. where the model prices worst (within-race mean absolute log error against the BSP; bias above 0 = ours "
      "longer than the BSP; the blend the closing model fits on the segment alone)")
for col in ("code", "kind", "seg", "country", "class", "fieldband", "band"):
    print(f"\n   by {col}")
    print(seg_table(col).round(3).to_string())

# ------------------------------------------------------------------ 2. adjustments, walk-forward by week
h["week"] = h.day.str.slice(0, 7)
weeks = sorted(h.week.unique())
segs = sorted(h.seg.unique())
S = (h.seg.to_numpy()[:, None] == np.array(segs)[None, :]).astype(float)
base_X = h[["lq_m", "lq_f"]].to_numpy(float)
seg_X = np.hstack([base_X, S * h.lq_m.to_numpy()[:, None], S * h.lq_f.to_numpy()[:, None]])
book = float(np.median(np.bincount(g, weights=1.0 / h.bsp.to_numpy(float))))


def rule(te, ev):
    """The owner's rule at the morning WAP: expected CLV >= 3%, the WAP at or over our price, GBP100 matched."""
    pick = (ev >= BAR) & (te.morningwap >= te.predicted_bfsp) & (te.morning_vol >= 100)
    x = te[pick]
    w = np.clip(400 / (x.morningwap - 1), 15, 200)
    clv = x.morningwap / x.bsp - 1
    return x.assign(w=w, clv=clv)


out, picks = [], []
for i, wk in enumerate(weeks):
    if i < 2:
        continue
    tr, te = h.week < wk, h.week == wk
    gtr, gte = rb.race_codes(h.race[tr]), rb.race_codes(h.race[te])
    qtr, qte = h.q[tr].to_numpy(float), h.q[te].to_numpy(float)
    T = h[te].copy()
    # one blend
    b1 = fit(base_X[tr.to_numpy()], qtr, gtr)
    # by race type: each segment's weights shrunk to the one blend
    prior = np.r_[b1, np.zeros(2 * len(segs))]
    b2 = fit(seg_X[tr.to_numpy()], qtr, gtr, l2=2e-3, prior=prior)
    arms = {"one blend": (base_X, b1), "weights by race type": (seg_X, b2)}
    for name, (X, b) in arms.items():
        Xte = X[te.to_numpy()]
        ev = T.morningwap.to_numpy() * book * np.exp(rb.group_log_softmax(Xte @ b, gte)) - 1
        r = rule(T, ev)
        out.append({"week": wk, "arm": name, "xent": xent(Xte, b, qte, gte), "bets": len(r), "staked": r.w.sum(),
                    "clv_x_stake": (r.w * r.clv).sum()})
        picks.append(r.assign(arm=name, week=wk))
    # the one blend, race types skipped where the months before closed below zero on the rule
    prev = pd.concat([x for x in picks if x.arm.iloc[0] == "one blend" and x.week.iloc[0] < wk], ignore_index=True) \
        if any(x.arm.iloc[0] == "one blend" and x.week.iloc[0] < wk for x in picks if len(x)) else pd.DataFrame()
    bad = set()
    if len(prev):
        s = prev.groupby("seg").apply(lambda x: (x.w * x.clv).sum())
        n = prev.groupby("seg").size()
        bad = set(s[(s < 0) & (n >= 20)].index)
    ev = T.morningwap.to_numpy() * book * np.exp(rb.group_log_softmax(base_X[te.to_numpy()] @ b1, gte)) - 1
    r = rule(T, ev)
    r = r[~r.seg.isin(bad)]
    out.append({"week": wk, "arm": "one blend, losing types skipped", "xent": np.nan, "bets": len(r), "staked": r.w.sum(),
                "clv_x_stake": (r.w * r.clv).sum(), "skipped": ", ".join(sorted(bad))})
    picks.append(r.assign(arm="one blend, losing types skipped", week=wk))

R = pd.DataFrame(out)
print("\n== 2. race-type adjustments, walk-forward by month (cross-entropy to the BSP book: lower is nearer)")
print(R.pivot_table(index="week", columns="arm", values="xent").round(5).to_string())
print(R.pivot_table(index="week", columns="arm", values="clv_x_stake").round(0).to_string())
tot = R.groupby("arm", sort=False).agg(weeks=("week", "nunique"), xent=("xent", "mean"), bets=("bets", "sum"),
                                       staked=("staked", "sum"), clv_x_stake=("clv_x_stake", "sum"))
tot["clv"] = tot.clv_x_stake / tot.staked
print("\n   together (the rule's CLV at the BSP, at the morning WAP, before any non-runner cut)")
print(tot.round(4).to_string())
print("\n   race types skipped each week:", R[R.arm.str.contains("skipped")][["week", "skipped"]].to_string(index=False))
P = pd.concat([x for x in picks if len(x)], ignore_index=True)
print("\n   the one blend's picks by race type (all weeks scored): staked and CLV at the BSP")
q1 = P[P.arm == "one blend"].groupby("seg").apply(lambda x: pd.Series({"bets": len(x), "staked": x.w.sum(),
                                                                     "clv": (x.w * x.clv).sum() / x.w.sum()}))
print(q1.sort_values("clv").round(4).to_string())
