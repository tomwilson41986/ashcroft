"""The owner's QA (21 Oct): "Look at the performance last 7 days. Did we change something that moved away from our earlier
profitable approach?" Betfair's statement for 14-20 Oct (-GBP209.86) matches the ledger to the penny; 10-13 Oct made
+GBP1,665. The live rule changed on 17 Oct (staking) and 20 Oct (guard, ladder, cap, live from 21 Oct), after the turn.
What did change on 14 Oct: horseracebase's download pause (hrb-pause-1015), so every day's predictions since are made
without any result after 13 Oct; and audit-1016 saw the morning price's margin over the BSP halve. Read-only, from the
live record (betfair_live_markets, betfair_live_marks, the 06:00 files, the ledgers), none of it from horseracebase:

1. The model against the BSP by day, 1-20 Oct: |log(model / BSP)|, the bias, the rank correlation; the market's own
   price at 08:00 UK against the BSP the same way (the control).
2. The model's weight beside the market: the closing blend (conditional logit of the BSP book on the 08:00 price and the
   model) fitted on 1-6, 7-13 and 14-20 Oct.
3. Stale form: from 15 Oct, the runners that ran (in a market of ours, settled) since 14 Oct and before the day, so their
   last run is missing from the model's inputs; the model's error and the trades' CLV for them against the rest.
4. The entry: by day, the first backs' price against the BSP (median log), their expected CLV, and the money.
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
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
D0, D1 = date(2026, 10, 1), date(2026, 10, 20)
DAYS = [(D0 + timedelta(n)).isoformat() for n in range((D1 - D0).days + 1)]


def get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001
        return None


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

mk = pd.read_sql_query(f"""SELECT race_date, market_id, selection_id, venue, market_start_utc, runner_name
                           FROM betfair_live_markets WHERE race_date >= '{DAYS[0]}' AND race_date <= '{DAYS[-1]}'
                           AND UPPER(COALESCE(market_type, 'WIN')) = 'WIN'""", conn)
mk["horse"] = mk.runner_name.map(normalise_horse)
mk["_k"] = (mk.race_date + "|" + mk.venue.map(normalise_track) + "|" + mk.market_start_utc.map(utc_iso_to_uk_hhmm).astype(str)
            + "|" + mk.horse)
mk = mk.merge(p, on="_k", how="left")
marks = pd.read_sql_query(f"""SELECT market_id, selection_id, mark, back1, lay1, bsp, runner_status FROM betfair_live_marks
                              WHERE race_date >= '{DAYS[0]}' AND race_date <= '{DAYS[-1]}'
                              AND mark IN ('final', 'clock_08:00')""", conn)
fin = marks[marks["mark"] == "final"][["market_id", "selection_id", "bsp", "runner_status"]].rename(
    columns={"runner_status": "final_status"})
m8 = marks[marks["mark"] == "clock_08:00"][["market_id", "selection_id", "back1", "lay1"]]
mk = mk.merge(fin, on=["market_id", "selection_id"], how="left").merge(m8, on=["market_id", "selection_id"], how="left")
mk["ran"] = mk.final_status.isin(["WINNER", "LOSER"])
print(f"live catalogue {DAYS[0]} .. {DAYS[-1]}: {mk.market_id.nunique():,} markets, {len(mk):,} runners; priced by the "
      f"06:00 file {mk.predicted_bfsp.notna().mean():.1%}; ran with a BSP {(mk.ran & (mk.bsp > 1)).mean():.1%}")

# stale form: ran (settled) in one of our markets from 14 Oct and before the day
runs = mk[mk.ran & (mk.race_date >= "2026-10-14")][["horse", "race_date"]].drop_duplicates()
first_run = {}
for h, g in runs.groupby("horse"):
    first_run[h] = sorted(g.race_date)
mk["stale"] = [any(d < day for d in first_run.get(h, [])) for h, day in zip(mk.horse, mk.race_date)]

v = mk[mk.ran & (mk.bsp > 1) & (mk.predicted_bfsp > 1)].copy()
v["err"] = np.log(v.predicted_bfsp / v.bsp)
v["mid8"] = np.sqrt(v.back1 * np.where(v.lay1 > 1, v.lay1, v.back1))
v["err8"] = np.log(v.mid8 / v.bsp)
v["period"] = np.select([v.race_date <= "2026-10-06", v.race_date <= "2026-10-13"], ["a 1-6 Oct", "b 7-13 Oct"],
                        "c 14-20 Oct")


def stats(g):
    h8 = g[g.mid8 > 1]
    return pd.Series({"runners": len(g), "model_mae_log": g.err.abs().mean(), "model_bias": g.err.median(),
                      "model_spearman": g.predicted_bfsp.corr(g.bsp, method="spearman"),
                      "m08_runners": len(h8), "m08_mae_log": h8.err8.abs().mean(),
                      "model_mae_same": h8.err.abs().mean(),
                      "m08_spearman": h8.mid8.corr(h8.bsp, method="spearman"), "stale_share": g.stale.mean()})


print("\n== 1. the model and the 08:00 market against the BSP, by day")
print(v.groupby("race_date").apply(stats).round(4).to_string())
print(v.groupby("period").apply(stats).round(4).to_string())

print("\n== 2. the closing blend by period: the BSP book on the 08:00 price and the model (weights, nearness)")
w = v[v.mid8 > 1].copy()
w["race"] = w.market_id
for per, g in w.groupby("period"):
    g = g.copy()
    gg = rb.race_codes(g.race)
    X = np.column_stack([np.log(rb.normalise_within(1 / g.mid8.to_numpy(float), gg)),
                         np.log(rb.normalise_within(1 / g.predicted_bfsp.to_numpy(float), gg))])
    q = rb.normalise_within(1 / g.bsp.to_numpy(float), gg)
    nr = gg.max() + 1

    def loss(b):
        lqh = rb.group_log_softmax(X @ b, gg)
        return -float(np.sum(q * lqh)) / nr, -((q - np.exp(lqh))[:, None] * X).sum(axis=0) / nr

    b = minimize(loss, np.array([0.5, 0.5]), jac=True, method="L-BFGS-B").x
    lq = rb.group_log_softmax(X @ b, gg)
    near = float(np.mean(np.abs(np.log(q) - lq)))
    near_m = float(np.mean(np.abs(np.log(q) - X[:, 0])))
    near_f = float(np.mean(np.abs(np.log(q) - X[:, 1])))
    print(f"   {per}: {nr} races, weights market {b[0]:.3f} model {b[1]:.3f}; nearness to the BSP book: blend {near:.4f}, "
          f"market alone {near_m:.4f}, model alone {near_f:.4f}")

print("\n== 3. stale form (15-20 Oct): the runners that ran since 14 Oct before the day")
s = v[v.race_date >= "2026-10-15"]
print(s.groupby("stale").apply(stats).round(4).to_string())

leds = []
for day in DAYS:
    raw = get(f"trading/live/{day}/ledger.csv")
    if raw is not None:
        d = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
        d["day"] = day
        leds.append(d)
led = pd.concat(leds, ignore_index=True)
for c in ("matched", "avg_price", "bsp", "selection_id", "edge", "pnl", "best_back"):
    led[c] = pd.to_numeric(led[c], errors="coerce")
led["t"] = pd.to_datetime(led.ts, utc=True, errors="coerce")
K = ["day", "market_id", "selection_id"]
b = led[(led.event == "back") & (led.matched > 0)].sort_values("t").copy()
b["sxp"] = b.matched * b.avg_price
pos = b.groupby(K).agg(S=("matched", "sum"), sxp=("sxp", "sum"), first_price=("avg_price", "first"),
                       first_edge=("edge", "first"), first_t=("t", "first")).reset_index()
pos["Pm"] = pos.sxp / pos.S
st = led[led.event == "settle"].sort_values("t").drop_duplicates(K, keep="last")[K + ["result", "bsp", "avg_price", "pnl"]] \
    .rename(columns={"avg_price": "P", "bsp": "B"})
pos = pos.merge(st, on=K, how="left")
pos = pos[pos.result.isin(["WINNER", "LOSER"]) & (pos.B > 1) & (pos.P > 1)].copy()
pos["clv"] = pos.P / pos.B - 1
pos["selection_id"] = pos.selection_id.astype("int64")
mk["selection_id"] = pd.to_numeric(mk.selection_id, errors="coerce")
pos = pos.merge(mk.dropna(subset=["selection_id"]).assign(selection_id=lambda x: x.selection_id.astype("int64"))[
    ["market_id", "selection_id", "stale", "predicted_bfsp"]], on=["market_id", "selection_id"], how="left")
t = pos[pos.day >= "2026-10-15"]
print("   our positions 15-20 Oct by stale form:")
print(t.groupby("stale", dropna=False).apply(lambda g: pd.Series({
    "horses": len(g), "staked": g.S.sum(), "clv": (g.clv * g.S).sum() / g.S.sum(), "result": g.pnl.sum()})).round(4).to_string())

print("\n== 4. the entry by day: first backs' price against the BSP, their expected CLV, the money")
pos["period"] = np.select([pos.day <= "2026-10-09", pos.day <= "2026-10-13"], ["a to 9 Oct", "b 10-13 Oct"], "c 14-20 Oct")
e = pos.groupby("day").apply(lambda g: pd.Series({
    "horses": len(g), "staked": g.S.sum(), "first_vs_bsp_median_log": np.log(g.first_price / g.B).median(),
    "first_edge_mean": g.first_edge.mean(), "first_edge_median": g.first_edge.median(),
    "model_vs_bsp_median_log": np.log(g.predicted_bfsp / g.B).median(),
    "clv": (g.clv * g.S).sum() / g.S.sum()}))
print(e.round(4).to_string())
print(pos.groupby("period").apply(lambda g: pd.Series({
    "horses": len(g), "first_vs_bsp_median_log": np.log(g.first_price / g.B).median(),
    "first_edge_mean": g.first_edge.mean(), "share_first_edge_under_6pct": (g.first_edge < 0.06).mean(),
    "clv": (g.clv * g.S).sum() / g.S.sum()})).round(4).to_string())
