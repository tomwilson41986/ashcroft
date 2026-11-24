"""Has the model lost ground to the market? (the owner, 24 Nov 2026: "i am concerned about the performance of our
model"; read-only, descriptive: ledger audit-1124-model-vs-market.)

Every UK/IE win market of the live record (1 Oct - 23 Nov 2026), each runner with the served 06:00 price
(S3 predictions/<day>.csv, joined by day, track, time and horse as benter_blend_1020.py), the book's mid at 08:00,
09:00, 10:00 UK and 60 and 5 minutes before the off, and the BSP (the final mark). By period, by day and by code:

  1. the error against the BSP (mean |log(price / BSP)|) of the model and of the market at each mark, the model's
     bias, and how many runners each covers;
  2. the weight the model earns beside the 08:00 market: a conditional logit of the BSP book on [log q_market_0800,
     log q_model] fitted within each period (in sample: a description of the period, not a forecast);
  3. the opportunity set, model-free of the trader's own rule: runners whose 08:00 best back is at least 3% over the
     closing model's forecast close (weights 0.774 market, 0.263 model, renormalised in the race): how many a day,
     and their mean move log(back1_0800 / BSP), with a 90% interval by day; the same for every runner (the market's
     own drift) so the picks' excess is read against it;
  4. the same picks by the gap between the model and the market (bands), by price band and by code.

Periods as the weekly table: A 10 Oct-8 Nov (old blend, autumn card), B 9-13 Nov (old blend, winter card),
C 14 Nov (market-rating blend), D 15-21 Nov (stack blend), E 22-23 Nov (gate_stn); 1-9 Oct shown as "pre".
"""
import io
import os
import sqlite3
import sys
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, ".")
from betfair_prices import normalise_horse, normalise_track, race_time_to_24h, utc_iso_to_uk_hhmm  # noqa: E402
from model import race_book as rb  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 300)
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
D0, D1 = date(2026, 10, 1), date(2026, 11, 23)
MARKS = ["clock_08:00", "clock_09:00", "clock_10:00", "off_60", "off_5"]
W_MKT, W_MOD, BAR = 0.774, 0.263, 0.03
rng = np.random.default_rng(20261124)
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)


def period(d):
    if d < "2026-10-10":
        return "pre 1-9 Oct"
    if d <= "2026-11-08":
        return "A 10 Oct-8 Nov"
    if d <= "2026-11-13":
        return "B 9-13 Nov"
    if d == "2026-11-14":
        return "C 14 Nov"
    if d <= "2026-11-21":
        return "D 15-21 Nov"
    return "E 22-23 Nov"


def get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001
        return None


def preds():
    text = {c: str for c in ("date", "race_date", "venue", "track", "race_time", "runner_name", "horse_name")}
    out, d = [], D0
    while d <= D1:
        raw = get(f"predictions/{d.isoformat()}.csv")
        if raw is None:
            print("no predictions file", d)
        else:
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


def ci(by_day_num, by_day_den):
    """A ratio of sums over days with a 90% interval by resampling days."""
    num, den = np.asarray(by_day_num, float), np.asarray(by_day_den, float)
    if len(num) < 3 or den.sum() <= 0:
        return (num.sum() / den.sum() if den.sum() > 0 else np.nan), np.nan, np.nan
    idx = rng.integers(0, len(num), size=(2000, len(num)))
    r = num[idx].sum(1) / den[idx].sum(1)
    return num.sum() / den.sum(), float(np.quantile(r, 0.05)), float(np.quantile(r, 0.95))


def fit(X, q, g, l2=1e-4):
    nr = g.max() + 1

    def loss(b):
        lqh = rb.group_log_softmax(X @ b, g)
        return (-float(np.sum(q * lqh)) / nr + 0.5 * l2 * float(b @ b),
                -((q - np.exp(lqh))[:, None] * X).sum(axis=0) / nr + l2 * b)

    return minimize(loss, np.full(X.shape[1], 0.5), jac=True, method="L-BFGS-B").x


mk = pd.read_sql_query(f"""SELECT race_date, market_id, selection_id, venue, market_start_utc, runner_name, race_type
                           FROM betfair_live_markets WHERE race_date >= '{D0}' AND race_date <= '{D1}'
                           AND UPPER(COALESCE(market_type, 'WIN')) = 'WIN'""", conn)
mk["selection_id"] = pd.to_numeric(mk.selection_id, errors="coerce")
mk = mk.drop_duplicates(["market_id", "selection_id"])
mk["_k"] = (mk.race_date + "|" + mk.venue.map(normalise_track) + "|" + mk.market_start_utc.map(utc_iso_to_uk_hhmm).astype(str)
            + "|" + mk.runner_name.map(normalise_horse))
mk = mk.merge(preds(), on="_k", how="left")
marks = pd.read_sql_query(f"""SELECT market_id, selection_id, mark, back1, back1_size, lay1, bsp, runner_status
                              FROM betfair_live_marks WHERE race_date >= '{D0}' AND race_date <= '{D1}'
                              AND mark IN ('final', {", ".join(repr(m) for m in MARKS)})""", conn)
marks["selection_id"] = pd.to_numeric(marks.selection_id, errors="coerce")
fin = marks[marks.mark == "final"].drop_duplicates(["market_id", "selection_id"])[
    ["market_id", "selection_id", "bsp", "runner_status"]]
mk = mk.merge(fin, on=["market_id", "selection_id"], how="left")
for m in MARKS:
    x = marks[marks.mark == m].drop_duplicates(["market_id", "selection_id"])
    x = x.assign(**{f"mid_{m}": np.sqrt(x.back1 * x.lay1), f"back_{m}": x.back1, f"size_{m}": x.back1_size})
    x.loc[~((x.back1 > 1) & (x.lay1 > 1) & (x.lay1 / x.back1 < 1.5)), f"mid_{m}"] = np.nan
    mk = mk.merge(x[["market_id", "selection_id", f"mid_{m}", f"back_{m}", f"size_{m}"]],
                  on=["market_id", "selection_id"], how="left")
mk["period"] = mk.race_date.map(period)
print(f"win markets {mk.market_id.nunique():,}, runners {len(mk):,}, {mk.race_date.min()} .. {mk.race_date.max()}; "
      f"priced by the 06:00 file {mk.predicted_bfsp.notna().mean():.1%}; with a BSP {(mk.bsp > 1).mean():.1%}")
print("race_type values:", mk.race_type.value_counts(dropna=False).to_dict())
ok = mk[(mk.bsp > 1) & mk.predicted_bfsp.gt(1)].copy()
print(f"runners with a model price and a BSP: {len(ok):,} in {ok.market_id.nunique():,} markets")
ok["e_model"] = np.abs(np.log(ok.predicted_bfsp / ok.bsp))
ok["b_model"] = np.log(ok.predicted_bfsp / ok.bsp)
for m in MARKS:
    ok[f"e_{m}"] = np.abs(np.log(ok[f"mid_{m}"] / ok.bsp))


def errtab(by):
    g = ok.groupby(by)
    t = pd.DataFrame({"runners": g.size(), "markets": g.market_id.nunique(), "model": g.e_model.mean(),
                      "model_bias": g.b_model.mean()})
    for m in MARKS:
        both = ok[ok[f"e_{m}"].notna()]
        gg = both.groupby(by)
        t[f"mkt_{m[-5:]}"] = gg[f"e_{m}"].mean()
        t[f"model_same_{m[-5:]}"] = gg.e_model.mean()
    return t.round(4)


print("\n== 1. error against the BSP, mean |log(price / BSP)|: the model and the market's mid at each mark")
print("   (model_same_X: the model on the runners the market's mid at X covers, so each pair is like for like)")
print(errtab("period").to_string())
print("\n   by code")
print(errtab("race_type").to_string())
print("\n   by period and code")
print(errtab(["period", "race_type"]).to_string())
print("\n   by day")
print(errtab("race_date").to_string())

print("\n== 2. the weight the model earns beside the 08:00 market (conditional logit of the BSP book, in sample a period)")
w = ok[ok["mid_clock_08:00"].notna()].copy()
n_in = w.groupby("market_id").size()
full = mk[(mk.bsp > 1)].groupby("market_id").size()
keep = n_in[n_in == full.reindex(n_in.index)].index   # whole races only: every runner with a BSP has both prices
w = w[w.market_id.isin(keep)]
print(f"   whole races with every runner priced by both: {w.market_id.nunique():,} markets, {len(w):,} runners")


def weights(d):
    g = pd.factorize(d.market_id)[0]
    q = rb.normalise_within(1.0 / d.bsp.to_numpy(float), g)
    X = np.column_stack([np.log(rb.normalise_within(1.0 / d["mid_clock_08:00"].to_numpy(float), g)),
                         np.log(rb.normalise_within(1.0 / d.predicted_bfsp.to_numpy(float), g))])
    b = fit(X, q, g)
    ll = lambda bb: float(np.sum(q * rb.group_log_softmax(X @ bb, g))) / (g.max() + 1)  # noqa: E731
    return b, ll(b), ll(fit(X[:, :1], q, g).tolist() + [0.0]), g.max() + 1


rows = []
for name, d in list(w.groupby("period")) + [("all", w)] + [(f"code {k}", v) for k, v in w.groupby("race_type")]:
    if d.market_id.nunique() < 30:
        continue
    b, l2, l1, nr = weights(d)
    rows.append({"group": name, "races": nr, "w_market": round(b[0], 3), "w_model": round(b[1], 3),
                 "gain_per_race_x1000": round(1000 * (l2 - l1), 3)})
print(pd.DataFrame(rows).to_string(index=False))
print("   (gain: the cross-entropy against the BSP book the model adds beside the market, per race, x1000)")

print("\n== 3. the opportunity set at 08:00: best back at least 3% over the forecast close (0.774 market, 0.263 model)")
o = mk[mk["mid_clock_08:00"].notna() & mk.predicted_bfsp.gt(1) & mk["back_clock_08:00"].gt(1)].copy()
g = pd.factorize(o.market_id)[0]
lq = W_MKT * np.log(rb.normalise_within(1.0 / o["mid_clock_08:00"].to_numpy(float), g)) + W_MOD * np.log(
    rb.normalise_within(1.0 / o.predicted_bfsp.to_numpy(float), g))
o["q_close"] = np.exp(rb.group_log_softmax(lq, g))
o["exp_clv"] = o["back_clock_08:00"] * o.q_close - 1.0
o["gap"] = np.log(o["mid_clock_08:00"] / o.predicted_bfsp)       # > 0: the model has the horse shorter than the market
o["has_bsp"] = o.bsp > 1
o["move"] = np.log(o["back_clock_08:00"] / o.bsp)                 # the CLV of a back at 08:00 closed at the BSP (log)
o["pick"] = (o.exp_clv >= BAR) & (o["back_clock_08:00"] >= o.predicted_bfsp) & (o["back_clock_08:00"] <= 200)
print(f"   runners with both prices at 08:00: {len(o):,}; picks {int(o.pick.sum()):,}; picks with a BSP "
      f"{int((o.pick & o.has_bsp).sum()):,} (a pick with no BSP was a non-runner: void)")


def movetab(d, by):
    out = []
    for k, x in d.groupby(by):
        days = x.groupby("race_date").agg(n=("move", "size"), s=("move", "sum"))
        m, lo, hi = ci(days.s, days.n)
        out.append({"group": k, "days": len(days), "runners": len(x), "a_day": round(len(x) / max(len(days), 1), 1),
                    "mean_move": round(m, 4), "lo": round(lo, 4) if lo == lo else np.nan,
                    "hi": round(hi, 4) if hi == hi else np.nan,
                    "mean_exp_clv": round(float(x.exp_clv.mean()), 4), "median_price": round(float(x["back_clock_08:00"].median()), 2)})
    return pd.DataFrame(out)


ob = o[o.has_bsp]
print("\n   every runner (the market's own drift from the 08:00 best back to the BSP, log)")
print(movetab(ob, "period").to_string(index=False))
print("\n   the picks")
print(movetab(ob[ob.pick], "period").to_string(index=False))
print("\n   the picks by code")
print(movetab(ob[ob.pick], "race_type").to_string(index=False))
print("\n   the picks by period and code")
print(movetab(ob[ob.pick], ["period", "race_type"]).to_string(index=False))
ob = ob.assign(gap_band=pd.cut(ob.gap, [-9, -0.3, -0.1, 0, 0.1, 0.2, 0.3, 0.5, 9]),
               price_band=pd.cut(ob["back_clock_08:00"], [1, 3, 6, 12, 25, 50, 200, 1001]),
               clv_band=pd.cut(ob.exp_clv, [-9, 0, 0.03, 0.05, 0.08, 0.12, 0.2, 9]))
print("\n== 4. every runner at 08:00 by the gap log(market / model) (the model shorter than the market when positive)")
print(movetab(ob, "gap_band").to_string(index=False))
print("\n   by the expected CLV band (is the forecast close calibrated? mean_move should rise with mean_exp_clv)")
print(movetab(ob, "clv_band").to_string(index=False))
print("\n   the picks by price band")
print(movetab(ob[ob.pick], "price_band").to_string(index=False))
late = ob[ob.race_date >= "2026-11-09"]
print("\n   from 9 Nov (the winter card): by the expected CLV band")
print(movetab(late, "clv_band").to_string(index=False))
print("\n   from 9 Nov: the picks by price band")
print(movetab(late[late.pick], "price_band").to_string(index=False))
print("\n   from 9 Nov: by the gap band")
print(movetab(late, "gap_band").to_string(index=False))
