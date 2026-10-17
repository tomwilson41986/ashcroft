"""The full review the owner asked for on 17 Oct (read-only), on every live position from 30 Sep (17 Oct as far as settled):

1. Hold or lay? Each back is closed with a lay at BSP whose liability is the back's winnings, so a winner nets
   nothing and the profit comes from the losers: S(P - B)/(B - 1). Held to the result instead, a back makes S(P - 1)
   or loses S. Under BSP-fair chances both have the same expectation, S(P/B - 1); holding adds only the variance and
   the commission, unless the horses we back win more often than the BSP says. So: the counterfactual P&L of holding
   every horse, of holding only the horses that shortened, and of half hedging; the backed horses' winners against
   the BSP's (A/E, and a z-score of holding's excess over the BSP-fair value); and on every runner of every 06:00 file,
   whether the model's price adds to the BSP in a conditional logit (if not, holding is a coin we have no edge on).
2. Lay late at the exchange's price, greened (the same profit whatever wins), against the lay at BSP: the best lay
   price and size at the last book at least 30 seconds (and 2 minutes) before the market turned in play, filled down
   the three best lays, the rest at BSP.
3. Staking: the realized CLV (at BSP, the greened profit per GBP) by price band, by stake, by expected edge at entry;
   each day's positions reweighted at the same total stake (level, to win uncapped, in proportion to the expected
   edge, edge over the variance of the price's move) - the same horses at the same prices, so it ignores whether the
   bigger stakes would have filled.
4. The owner's question of 17 Oct: do the long-priced horses drift a lot, and should there be a price cut-off in the
   morning? Each band's drift (share closing longer, median and spread of log(BSP / price)), CLV, P&L, and the days'
   result without the bands above each cut-off.
"""
import io
import os
import sqlite3
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
K = ["day", "market_id", "selection_id"]
C = 0.02
DAYS = [(date(2026, 9, 30) + timedelta(n)).isoformat() for n in range(18)]
BANDS = [1.0, 3.0, 5.0, 8.0, 13.0, 21.0, 1001.0]
BAND_NAMES = ["<3", "3-5", "5-8", "8-13", "13-21", "21+"]


def get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001
        return None


def num(d, cols):
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


# ---------------------------------------------------------------- the ledgers and the books
WANT = {"polled_utc", "source", "market_id", "status", "inplay", "selection_id", "runner_status", "back1",
        "back1_size", "lay1", "lay1_size", "lay2", "lay2_size", "lay3", "lay3_size", "sp_actual"}
leds, lates, finals = [], [], []
for day in DAYS:
    raw = get(f"trading/live/{day}/ledger.csv")
    if raw is None:
        print("no ledger", day)
        continue
    d = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
    d["day"] = day
    leds.append(d)
    parts = []
    for n in ("books_trader", "books"):
        r = get(f"betfair_live/{day}/{n}.csv.gz")
        if r is not None:
            parts.append(pd.read_csv(io.BytesIO(r), compression="gzip", usecols=lambda c: c in WANT,
                                     dtype={"market_id": str}, low_memory=False))
    if not parts:
        print("no books", day)
        continue
    bk = num(pd.concat(parts, ignore_index=True), ["selection_id", "back1", "back1_size", "lay1", "lay1_size", "lay2",
                                                   "lay2_size", "lay3", "lay3_size", "sp_actual"])
    bk = bk.dropna(subset=["selection_id"])
    bk["selection_id"] = bk.selection_id.astype("int64")
    bk["t"] = pd.to_datetime(bk.polled_utc, utc=True, errors="coerce")
    bk["ip"] = bk.inplay.astype(str).str.lower().isin(["1", "true", "1.0"])
    first_ip = bk[bk.ip].groupby("market_id").t.min().rename("first_ip")
    bk = bk.merge(first_ip, on="market_id", how="left")
    pre = bk[(bk.status == "OPEN") & ~bk.ip & bk.first_ip.notna() & (bk.t < bk.first_ip)
             & bk.runner_status.eq("ACTIVE")].copy()
    pre["secs"] = (pre.first_ip - pre.t).dt.total_seconds()
    for tag, gap in (("l30", 30), ("l120", 120)):
        x = pre[pre.secs >= gap].sort_values("t").drop_duplicates(["market_id", "selection_id"], keep="last")
        x = x[["market_id", "selection_id", "secs", "back1", "lay1", "lay1_size", "lay2", "lay2_size", "lay3",
               "lay3_size"]].rename(columns=lambda c: c if c in ("market_id", "selection_id") else f"{tag}_{c}")
        x["day"] = day
        lates.append((tag, x))
    # every runner's result and BSP, for the model-against-BSP test
    fin = bk[bk.source.eq("final") & bk.status.eq("CLOSED")].sort_values("t").drop_duplicates(
        ["market_id", "selection_id"], keep="last")[["market_id", "selection_id", "runner_status"]]
    sp = bk[bk.sp_actual > 1].sort_values("t").groupby(["market_id", "selection_id"]).sp_actual.last().rename("bsp")
    fin = fin.merge(sp, on=["market_id", "selection_id"], how="left")
    fin["day"] = day
    finals.append(fin)

led = num(pd.concat(leds, ignore_index=True), ["matched", "avg_price", "bsp", "clv", "selection_id", "edge", "pnl",
                                                "commission", "minutes_to_off", "back_size"])
led = led.dropna(subset=["selection_id"])
led["selection_id"] = led.selection_id.astype("int64")
led["t"] = pd.to_datetime(led.ts, utc=True, errors="coerce")
b = led[(led.event == "back") & (led.matched > 0)].sort_values("t").copy()
b["sxp"], b["exs"] = b.matched * b.avg_price, b.edge.fillna(0) * b.matched
pos = b.groupby(K).agg(S=("matched", "sum"), sxp=("sxp", "sum"), exs=("exs", "sum"), backs=("matched", "size"),
                       first_t=("t", "min"), first_price=("avg_price", "first"), first_stake=("matched", "first"),
                       runner=("runner", "first"), venue=("venue", "first")).reset_index()
pos["Pm"], pos["edge"] = pos.sxp / pos.S, pos.exs / pos.S
st = led[led.event == "settle"].sort_values("t").drop_duplicates(K, keep="last")[
    K + ["result", "bsp", "avg_price", "pnl"]].rename(columns={"avg_price": "P", "pnl": "pnl_live"})
pos = pos.merge(st, on=K, how="left")
for tag in ("l30", "l120"):
    x = pd.concat([f for t_, f in lates if t_ == tag], ignore_index=True)
    pos = pos.merge(x, on=K, how="left")
comm = led[led.event == "commission"].groupby(["day", "market_id"]).commission.sum()
day_live = (led[led.event == "settle"].sort_values("t").drop_duplicates(K, keep="last").groupby("day").pnl.sum()
            - comm.groupby("day").sum().reindex(led.day.unique()).fillna(0)).dropna()
print("== the live result by day as the ledger settled it (P&L less commission):")
print(day_live.round(2).to_string(), f"\n   total GBP{day_live.sum():,.2f}")

v = pos[pos.result.isin(["WINNER", "LOSER"]) & (pos.bsp > 1) & (pos.P > 1) & (pos.S > 0)].copy()
print(f"\npositions: {len(pos):,}; settled with a BSP {len(v):,} (non-runners and unsettled left out), "
      f"GBP{v.S.sum():,.2f} staked, {v.day.nunique()} days")
S, P, B, win = v.S.to_numpy(), v.P.to_numpy(), v.bsp.to_numpy(), v.result.eq("WINNER").to_numpy()
v["hold"] = np.where(win, S * (P - 1), -S)
v["live_calc"] = np.where(win, 0.0, S * (P - 1) / (B - 1) - S)
v["green_bsp"] = S * (P / B - 1)
print(f"check: the lay at BSP as computed against the ledger's P&L: GBP{v.live_calc.sum():,.2f} against "
      f"GBP{v.pnl_live.sum():,.2f} (median absolute gap GBP{(v.live_calc - v.pnl_live).abs().median():.2f})")


def green_late(v, tag):
    """Greened at the late book: lay down the three best lays until S*P is laid (the green condition), the rest at
    BSP; profit -S + total lay stake, the same whatever wins."""
    out, filled = np.full(len(v), np.nan), np.zeros(len(v))
    for i, r in enumerate(v.itertuples()):
        need = r.S * r.P
        if not (getattr(r, f"{tag}_lay1") > 1):
            continue
        acc, ls = 0.0, 0.0
        for lv in (1, 2, 3):
            pr, sz = getattr(r, f"{tag}_lay{lv}"), getattr(r, f"{tag}_lay{lv}_size")
            if not (pr > 1) or not (sz > 0) or acc >= need:
                continue
            take = min(sz, (need - acc) / pr)
            acc, ls = acc + take * pr, ls + take
        filled[i] = acc / need
        ls += max(0.0, need - acc) / r.bsp
        out[i] = -r.S + ls
    return out, filled


for tag in ("l30", "l120"):
    v[f"green_{tag}"], v[f"fill_{tag}"] = green_late(v, tag)
    v[f"green_{tag}"] = v[f"green_{tag}"].fillna(v.live_calc)
v["short30"] = v.l30_back1.notna() & (v.l30_back1 < v.Pm)  # the price shortened by the late book
v["drift30"] = v.l30_back1.notna() & (v.l30_back1 >= v.Pm)
v["short30_10"] = v.l30_back1.notna() & (v.l30_back1 < v.Pm * 0.9)
v["hold_short_lay_bsp"] = np.where(v.short30, v.hold, v.live_calc)
v["hold_short10_lay_bsp"] = np.where(v.short30_10, v.hold, v.live_calc)
v["hold_short_green_late"] = np.where(v.short30, v.hold, v.green_l30)
v["hold_drift_lay_bsp"] = np.where(v.drift30, v.hold, v.live_calc)
v["half_hedge"] = 0.5 * v.hold + 0.5 * v.live_calc
POL = ["live_calc", "green_bsp", "green_l30", "green_l120", "hold", "hold_short_lay_bsp", "hold_short10_lay_bsp",
       "hold_short_green_late", "hold_drift_lay_bsp", "half_hedge"]


def after_commission(v, col):
    m = v.groupby(["day", "market_id"])[col].sum()
    return (m - C * m.clip(lower=0)).groupby("day").sum()


def summary(v, cols):
    rows = {}
    for c in cols:
        d = after_commission(v, c)
        cum = d.cumsum()
        rows[c] = {"total": d.sum(), "mean_day": d.mean(), "sd_day": d.std(), "worst_day": d.min(),
                   "best_day": d.max(), "losing_days": int((d < 0).sum()), "max_drawdown": (cum.cummax() - cum).max(),
                   "per_100_staked": 100 * d.sum() / v.S.sum()}
    return pd.DataFrame(rows).T


print("\n== 1/2. each way of closing the same positions, after 2% commission on each market's net winnings")
print("   live_calc: the lay at BSP as live (liability = the winnings); green_bsp: greened at BSP; green_l30 / l120: greened"
      " at the exchange's best lays 30 s / 2 min before the market turned in play (the rest at BSP); hold: no lay;"
      " hold_short_*: hold the horses whose price had shortened by the late book, close the rest; half_hedge: half"
      " held")
print(summary(v, POL).round(2).to_string())
print(f"   late book found for {v.l30_lay1.gt(1).mean():.0%} of positions (30 s), median {v.l30_secs.median():.0f} s "
      f"before in play; the three best lays filled the whole lay for {(v.fill_l30 >= 0.999).mean():.0%} "
      f"(the rest at BSP)")
print("\n   by day, after commission:")
print(pd.DataFrame({c: after_commission(v, c) for c in ["live_calc", "green_l30", "hold", "hold_short_lay_bsp"]})
      .round(2).to_string())
g = v.assign(late_minus_bsp=np.log(v.l30_lay1 / v.bsp)).dropna(subset=["late_minus_bsp"])
print(f"\n   the best lay 30 s before the off against the BSP: median log(lay / BSP) {g.late_minus_bsp.median():+.4f}, "
      f"mean {g.late_minus_bsp.mean():+.4f}, the lay shorter than the BSP (better for a layer) "
      f"{(g.l30_lay1 < g.bsp).mean():.0%}")

print("\n== 1. do the horses we back win more often than the BSP says?")
q = 1.0 / B
D = S * P * (win - q)
var = (S * P) ** 2 * q * (1 - q)
v["D"], v["varD"], v["q"] = D, var, q
v["band"] = pd.cut(v.first_price, BANDS, labels=BAND_NAMES, right=False)


def ae(g):
    return pd.Series({"horses": len(g), "winners": int(g.result.eq("WINNER").sum()), "bsp_expects": g.q.sum(),
                      "A/E": g.result.eq("WINNER").sum() / g.q.sum(), "hold_excess_gbp": g.D.sum(),
                      "z": g.D.sum() / np.sqrt(g.varD.sum())})


print(ae(v).round(3).to_string())
print("   by whether the price had shortened by the late book:")
print(v.groupby(np.select([v.short30, v.drift30], ["shortened", "not shortened"], "no late book")).apply(ae).round(3).to_string())
print("   by the first back's price:")
print(v.groupby("band", observed=True).apply(ae).round(3).to_string())

def clogit(df, race, cols, y):
    """Conditional logit (one winner a race) by Newton: the weights of the features and their standard errors."""
    df = df.copy()
    df["_r"] = df.groupby(race).ngroup()
    ok = df.groupby("_r")[y].transform("sum") == 1
    df = df[ok].sort_values("_r")
    if df.empty:
        return np.full(len(cols), np.nan), np.full(len(cols), np.nan), np.nan, 0
    r = df._r.to_numpy()
    r = pd.factorize(r)[0]
    X, yy = df[cols].to_numpy(float), df[y].to_numpy(bool)
    th = np.ones(len(cols)) / len(cols)
    nr = r.max() + 1
    for _ in range(50):
        u = X @ th
        mx = np.full(nr, -np.inf)
        np.maximum.at(mx, r, u)
        e = np.exp(u - mx[r])
        den = np.zeros(nr)
        np.add.at(den, r, e)
        p = e / den[r]
        ex = np.zeros((nr, len(cols)))
        np.add.at(ex, r, p[:, None] * X)
        grad = X[yy].sum(0) - ex.sum(0)
        H = -((p[:, None, None] * X[:, :, None] * X[:, None, :]).sum(0) - (ex[:, :, None] * ex[:, None, :]).sum(0))
        step = np.linalg.solve(H, grad)
        th = th - step
        if np.abs(step).max() < 1e-9:
            break
    ll = (np.log(p)[yy]).sum()
    se = np.sqrt(np.diag(np.linalg.inv(-H)))
    return th, se, ll, nr



# the model against the BSP on every runner of the 06:00 files, joined to Betfair's price files by track, time and
# horse (the files' selection ids are not filled), through the results horseracebase gives (to 13 Oct)
import sys  # noqa: E402

sys.path.insert(0, ".")
from betfair_prices import normalise_horse, normalise_track, race_time_to_24h  # noqa: E402

text = {c: str for c in ("date", "race_date", "venue", "track", "race_time", "runner_name", "horse_name")}
fr = []
for dd in pd.date_range("2026-08-01", "2026-10-17"):
    raw = get(f"predictions/{dd:%Y-%m-%d}.csv")
    if raw is None:
        continue
    x = pd.read_csv(io.BytesIO(raw), dtype=text, low_memory=False)
    x["file_day"] = f"{dd:%Y-%m-%d}"
    fr.append(x)
pq = pd.concat(fr, ignore_index=True)
for a_, b_ in (("race_date", "date"), ("track", "venue"), ("horse_name", "runner_name")):
    if a_ not in pq.columns and b_ in pq.columns:
        pq[a_] = pq[b_]
pq["race_date"] = pq["race_date"].fillna(pq["file_day"])
pq["predicted_bfsp"] = pd.to_numeric(pq["predicted_bfsp"], errors="coerce")
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
bfh = pd.read_sql_query("""SELECT r.race_date, r.track, r.race_time, r.horse_name, b.win_lose, b.bsp, b.event_id
                           FROM betfair_prices b JOIN race_results r ON r.id = b.race_results_id
                           WHERE LOWER(b.market_type) = 'win' AND r.race_date >= '2026-08-01'""", conn)


def key(x):
    return (x["race_date"].astype(str).str.slice(0, 10) + "|" + x["track"].map(normalise_track) + "|"
            + x["race_time"].map(race_time_to_24h).astype(str) + "|" + x["horse_name"].map(normalise_horse))


pq["_k"], bfh["_k"] = key(pq), key(bfh)
h = pq.drop_duplicates("_k", keep="last")[["_k", "predicted_bfsp"]].merge(bfh.drop_duplicates("_k"), on="_k")
h = h[(h.bsp > 1) & (h.predicted_bfsp > 1)].copy()
h["day"] = h.race_date.astype(str).str.slice(0, 10)
h["race"], h["win"] = h.day + "|" + h.event_id.astype(str), h.win_lose.eq(1)
for c, src_ in (("x_bsp", "bsp"), ("x_model", "predicted_bfsp")):
    inv = 1.0 / h[src_]
    h[c] = np.log(inv / inv.groupby(h.race).transform("sum"))
print(f"\n   the model against the BSP: every runner of the 06:00 files joined to Betfair's price files "
      f"({len(h):,} runners, {h.day.min()} .. {h.day.max()}); conditional logit of the winner on log chances")
for label, part in (("1 Aug - 28 Sep", h[h.day < "2026-09-29"]), ("29 Sep on (the live model)", h[h.day >= "2026-09-29"]),
                    ("all", h)):
    print(f"   {label}:")
    for cols in (["x_bsp"], ["x_model"], ["x_bsp", "x_model"]):
        th, se, ll, nr = clogit(part, "race", cols, "win")
        print(f"     {' + '.join(cols):18s} weights {np.round(th, 3)} se {np.round(se, 3)} log-lik {ll:,.1f} ({nr} races)")

print("\n== 3/4. by the first back's price: drift, CLV, P&L (the owner's question: do long prices drift a lot?)")
v["lmove"] = np.log(v.bsp / v.Pm)
v["c"] = v.P / v.bsp - 1


def band(g):
    return pd.Series({"horses": len(g), "staked": g.S.sum(), "share_of_stake": g.S.sum() / v.S.sum(),
                      "closed_longer": (g.bsp > g.Pm).mean(), "median_log_move": g.lmove.median(),
                      "sd_log_move": g.lmove.std(), "clv": (g.c * g.S).sum() / g.S.sum(),
                      "expected_edge": (g.edge * g.S).sum() / g.S.sum(), "live_pnl": g.live_calc.sum(),
                      "hold_pnl": g.hold.sum()})


print(v.groupby("band", observed=True).apply(band).round(3).to_string())
v["window"] = np.where(v.first_t.dt.tz_convert("Europe/London").dt.hour < 8, "before 08:00", "08:00 on")
print("   by band and the window of the first back (CLV, staked):")
print(v.groupby(["band", "window"], observed=True).apply(lambda g: pd.Series(
    {"horses": len(g), "staked": g.S.sum(), "clv": (g.c * g.S).sum() / g.S.sum()})).round(3).to_string())
print("   a morning price cut-off: the live result (before commission) of the positions left, all days")
for cut in (5.0, 8.0, 13.0, 21.0, 1001.0):
    k = v[v.first_price < cut]
    print(f"     first price under {cut:>6}: {len(k):,} horses, GBP{k.S.sum():,.0f} staked, live P&L "
          f"GBP{k.live_calc.sum():,.2f}, CLV {(k.c * k.S).sum() / k.S.sum():+.2%}")

print("\n== 3. staking: realized CLV (greened at BSP) by stake and by the expected edge at entry")
v["stake_band"] = pd.cut(v.S, [0, 10, 25, 50, 100, 200, 301], right=True)
print(v.groupby("stake_band", observed=True).apply(lambda g: pd.Series(
    {"horses": len(g), "staked": g.S.sum(), "clv": (g.c * g.S).sum() / g.S.sum(),
     "median_price": g.Pm.median()})).round(3).to_string())
v["edge_dec"] = pd.qcut(v.edge.rank(method="first"), 5, labels=["q1 low", "q2", "q3", "q4", "q5 high"])
print(v.groupby("edge_dec", observed=True).apply(lambda g: pd.Series(
    {"horses": len(g), "edge_from": g.edge.min(), "edge_to": g.edge.max(), "expected": (g.edge * g.S).sum() / g.S.sum(),
     "realized_clv": (g.c * g.S).sum() / g.S.sum()})).round(3).to_string())
vb = v.groupby("band", observed=True).c.var()
v["var_band"] = v.band.map(vb).astype(float)
w = {"actual (to win 400, cap 300)": v.S, "level": pd.Series(1.0, index=v.index),
     "to win, uncapped": 1.0 / (v.first_price - 1), "in proportion to the edge": v.edge.clip(lower=0),
     "edge / variance of the move": v.edge.clip(lower=0) / v.var_band}
rows = {}
for name, wt in w.items():
    wt = wt.fillna(0)
    scale = v.S.groupby(v.day).transform("sum") / wt.groupby(v.day).transform("sum")
    st_ = wt * scale
    pnl = st_ * v.c
    d = pnl.groupby(v.day).sum()
    rows[name] = {"total_gbp": d.sum(), "sd_day": d.std(), "worst_day": d.min(), "clv": pnl.sum() / st_.sum(),
                  "largest_stake": st_.max(), "share_over_5": st_[v.Pm >= 5].sum() / st_.sum()}
print("   the same positions, each day's stake shared out another way (greened at BSP, before commission):")
print(pd.DataFrame(rows).T.round(3).to_string())


print("\n== robustness: the CLV of each band with each horse's P/B - 1 capped at +100% (a few huge shorteners), and by day")
v["c_cap"] = v.c.clip(upper=1.0)
print(v.groupby("band", observed=True).apply(lambda g: pd.Series(
    {"clv": (g.c * g.S).sum() / g.S.sum(), "clv_capped": (g.c_cap * g.S).sum() / g.S.sum(),
     "level_clv": g.c.mean(), "level_clv_capped": g.c_cap.mean(), "share_over_+100%": (g.c > 1).mean(),
     "days_positive": int((g.groupby("day").apply(lambda d: (d.c * d.S).sum()) > 0).sum()),
     "days": g.day.nunique()})).round(3).to_string())
mx = b.groupby(K).back_size.max().rename("max_back_size")
v = v.join(mx, on=K)
rows = {}
for name, wt in w.items():
    wt = pd.Series(np.asarray(wt, dtype=float), index=v.index).fillna(0)
    scale = v.S.groupby(v.day).transform("sum") / wt.groupby(v.day).transform("sum")
    st_ = wt * scale
    cap_ = np.minimum(st_, np.maximum(v.S, v.max_back_size.fillna(0)))
    for lab, stake in (("as shared", st_), ("capped at the size offered", cap_)):
        pnl = stake * v.c
        pnl_c = stake * v.c_cap
        d = pnl.groupby(v.day).sum()
        rows[(name, lab)] = {"staked": stake.sum(), "total_gbp": d.sum(), "total_capped_moves": pnl_c.sum(),
                             "sd_day": d.std(), "days_better_than_actual": int((d > (v.S * v.c).groupby(v.day).sum()).sum())}
print("   the reweights again: capped at the best back's size the trader saw (no more than was offered), and with each "
      "horse's move capped at +100%")
print(pd.DataFrame(rows).T.round(2).to_string())
