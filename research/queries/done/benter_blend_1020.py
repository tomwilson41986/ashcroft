"""The owner's question (20 Oct): would blending the market's prices with ours, a la Benter, help? Where can we get an
edge on probabilities, read through price, liquidity and depth (the owner: "time isn't a dimension")? Read-only.

The live trader already blends: its closing model (data/models/closing_model_novol.json) is a conditional logit of
the BSP book on the market's price now and ours, 0.77 x market + 0.26 x model, fitted on January-March 2026 morning
WAPs. Two things it is not: fitted on the books the trader actually reads (the best back and lay, their money, the
ladder), or fitted to the winners (Benter's own stage two). So:

A. The live books, 1-19 Oct (the trader's reads, every 10th read of each market; every runner priced by the 06:00
   file, joined by track, time and horse; the BSP and winner from betfair_live_marks). Walk-forward by day (each day
   scored by a fit on the days before, from the sixth day): the served model; the blend refitted on these books
   (market price now and ours); with the market price read as the best back or the mid; with the blend's weights
   moving with the spread and the market's money (Benter's weights by liquidity); and with the book's own signals
   added (spread, weight of money on the three levels, a thin top price, the last traded price, the runner's money).
   Each scored by nearness to the BSP book, by calibration (the realised CLV at the BSP of every runner-read by the
   expected CLV each model gives it), and by the owner's rule on it (first read at expected CLV >= 3% and at or over
   our price; to win GBP400, GBP15-200; CLV at the BSP).
B. Benter to the result: does the blend (market now + ours) or the book's signals tell us about the winner beyond the
   BSP? Conditional logits of the winner on the live books' first read and last read before the off, and on the
   history (Betfair's price files, morning WAP and pre-play WAP, with the 06:00 files from 1 Aug, results to 13 Oct):
   weights, log-likelihood, pseudo R2. And the hold rule a Benter blend would choose (p_blend x morning WAP, after
   2% on winnings, >= 1.03), walk-forward by week, held to the result and laid at the BSP, against the closing rule.
"""
import io
import os
import sqlite3
import sys
from datetime import date, timedelta

import boto3
import botocore
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
BAR, C = 0.03, 0.02
LIVE_DAYS = [(date(2026, 10, 1) + timedelta(n)).isoformat() for n in range(19)]
EVERY = 10
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)


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


def fit(X, q, g, l2=1e-4, start=None):
    """Conditional logit by cross-entropy against q (soft labels: the BSP book; hard: the winner)."""
    nr = g.max() + 1

    def loss(b):
        lqh = rb.group_log_softmax(X @ b, g)
        return (-float(np.sum(q * lqh)) / nr + 0.5 * l2 * float(b @ b),
                -((q - np.exp(lqh))[:, None] * X).sum(axis=0) / nr + l2 * b)

    b0 = np.zeros(X.shape[1]) if start is None else np.asarray(start, float)
    if start is None:
        b0[: min(2, X.shape[1])] = 0.5
    return minimize(loss, b0, jac=True, method="L-BFGS-B").x


def clogit_se(X, y, g, b):
    """Standard errors of a hard-label conditional logit at b, and its log-likelihood."""
    lp = rb.group_log_softmax(X @ b, g)
    p = np.exp(lp)
    nr = g.max() + 1
    ex = np.zeros((nr, X.shape[1]))
    np.add.at(ex, g, p[:, None] * X)
    H = (p[:, None, None] * X[:, :, None] * X[:, None, :]).sum(0) - (ex[:, :, None] * ex[:, None, :]).sum(0)
    try:
        se = np.sqrt(np.diag(np.linalg.inv(H)))
    except np.linalg.LinAlgError:
        se = np.full(len(b), np.nan)
    return se, float(lp[y.astype(bool)].sum())


def lq(price, g):
    return np.log(rb.normalise_within(1.0 / np.asarray(price, float), g))


def z(x, ref):
    sd = np.std(ref)
    return (x - np.mean(ref)) / (sd if sd > 1e-9 else 1.0)


# ===================================================================================== A. the live books
p_live = preds(date(2026, 10, 1), date(2026, 10, 19))
mk = pd.read_sql_query(f"""SELECT race_date, market_id, selection_id, venue, market_start_utc, runner_name
                           FROM betfair_live_markets WHERE race_date >= '{LIVE_DAYS[0]}' AND race_date <= '{LIVE_DAYS[-1]}'
                           AND UPPER(COALESCE(market_type, 'WIN')) = 'WIN'""", conn)
mk["_k"] = (mk.race_date + "|" + mk.venue.map(normalise_track) + "|" + mk.market_start_utc.map(utc_iso_to_uk_hhmm).astype(str)
            + "|" + mk.runner_name.map(normalise_horse))
mk = mk.merge(p_live, on="_k", how="left")
mk["selection_id"] = pd.to_numeric(mk.selection_id, errors="coerce")
fin = pd.read_sql_query(f"""SELECT market_id, selection_id, runner_status AS final_status, bsp FROM betfair_live_marks
                            WHERE mark = 'final' AND race_date >= '{LIVE_DAYS[0]}' AND race_date <= '{LIVE_DAYS[-1]}'""",
                        conn)
fin["selection_id"] = pd.to_numeric(fin.selection_id, errors="coerce")
mk = mk.merge(fin, on=["market_id", "selection_id"], how="left")
print(f"live catalogue 1-19 Oct: {mk.market_id.nunique():,} win markets, {len(mk):,} runners; priced by the 06:00 file "
      f"{mk.predicted_bfsp.notna().mean():.1%}; with a BSP {(mk.bsp > 1).mean():.1%}; winners "
      f"{mk.final_status.eq('WINNER').sum():,}")

BOOK = ["polled_utc", "market_id", "selection_id", "status", "inplay", "runner_status", "back1", "back1_size", "back2",
        "back2_size", "back3", "back3_size", "lay1", "lay1_size", "lay2_size", "lay3_size", "market_total_matched",
        "runner_total_matched", "last_traded"]
snaps = []
for day in LIVE_DAYS:
    raw = get(f"betfair_live/{day}/books_trader.csv.gz")
    if raw is None:
        print("no trader books", day)
        continue
    bk = pd.read_csv(io.BytesIO(raw), compression="gzip", usecols=lambda c: c in BOOK, dtype={"market_id": str},
                     low_memory=False)
    for c in BOOK[6:] + ["selection_id"]:
        if c in bk.columns:
            bk[c] = pd.to_numeric(bk[c], errors="coerce")
    ip = bk.inplay.astype(str).str.lower().isin(["1", "true", "1.0"])
    bk = bk[(bk.status == "OPEN") & ~ip & bk.runner_status.fillna("ACTIVE").eq("ACTIVE")].copy()
    reads = bk[["market_id", "polled_utc"]].drop_duplicates().sort_values(["market_id", "polled_utc"])
    reads["n"] = reads.groupby("market_id").cumcount()
    last = reads.groupby("market_id").n.transform("max")
    reads = reads[(reads.n % EVERY == 0) | (reads.n == last)]
    reads["first"], reads["last"] = reads.n.eq(0), reads.n.eq(last)
    bk = bk.merge(reads[["market_id", "polled_utc", "first", "last"]], on=["market_id", "polled_utc"])
    bk["day"] = day
    snaps.append(bk)
    print(f"  {day}: {len(reads):,} reads kept, {len(bk):,} runner-reads")
L = pd.concat(snaps, ignore_index=True)
L = L.merge(mk[["market_id", "selection_id", "predicted_bfsp", "bsp", "final_status"]], on=["market_id", "selection_id"],
            how="left")
L["snap"] = L.market_id + "|" + L.polled_utc.astype(str)
ok = (L.back1 > 1) & (L.predicted_bfsp > 1) & (L.bsp > 1) & L.final_status.isin(["WINNER", "LOSER"])
whole = ok.groupby(L.snap).transform("all") & (L.groupby("snap").snap.transform("size") >= 2)
L = L[whole].copy()
L["win"] = L.final_status.eq("WINNER").astype(float)
L = L[L.groupby("snap").win.transform("sum") == 1].copy()
L = L.sort_values(["day", "snap"]).reset_index(drop=True)
print(f"whole reads (every active runner priced, with a BSP and one winner): {L.snap.nunique():,} reads of "
      f"{L.market_id.nunique():,} markets, {len(L):,} runner-reads")

g_all = rb.race_codes(L.snap)
L["now"] = rb.market_now(L.back1.to_numpy(float), L.lay1.to_numpy(float))
L["mid"] = np.sqrt(L.back1 * np.where(L.lay1 > 1, L.lay1, L.back1))
L["lq_now"], L["lq_back"], L["lq_mid"] = lq(L.now, g_all), lq(L.back1, g_all), lq(L.mid, g_all)
L["lq_model"], L["q_bsp"] = lq(L.predicted_bfsp, g_all), np.exp(lq(L.bsp, g_all))
L["spread"] = np.clip(np.log(np.where(L.lay1 > 1, L.lay1, np.e * L.back1) / L.back1), 0, 1)
off_b = L[["back1_size", "back2_size", "back3_size"]].fillna(0).sum(axis=1)
off_l = L[["lay1_size", "lay2_size", "lay3_size"]].fillna(0).sum(axis=1)
L["wom"] = np.clip(np.log((off_l + 1) / (off_b + 1)), -5, 5)       # backers' money waiting over layers' money
L["thin"] = np.clip(np.log(L.back1 / np.where(L.back2 > 1, L.back2, L.back1 / np.e)), 0, 1)
L["ltp"] = np.clip(np.log(L.back1 / np.where(L.last_traded > 1, L.last_traded, L.back1)), -1, 1)
L["lvr"] = np.log1p(L.runner_total_matched.fillna(0).clip(lower=0))
L["lvm"] = np.log1p(L.market_total_matched.fillna(0).clip(lower=0))
L["book_bsp"] = np.bincount(g_all, weights=1.0 / L.bsp)[g_all]
served = rb.load_closing_model("data/models/closing_model_novol.json")


def design(d, ref, kind):
    if kind == "now":
        return np.column_stack([d.lq_now, d.lq_model])
    if kind == "back":
        return np.column_stack([d.lq_back, d.lq_model])
    if kind == "mid":
        return np.column_stack([d.lq_mid, d.lq_model])
    sz, mz = z(d.spread, ref.spread), z(d.lvm, ref.lvm)
    base = [d.lq_now, d.lq_model, d.lq_now * sz, d.lq_model * sz, d.lq_now * mz, d.lq_model * mz]
    if kind == "liquidity":
        return np.column_stack(base)
    extra = [z(d[c], ref[c]) for c in ("spread", "wom", "thin", "ltp", "lvr")]
    return np.column_stack(base + extra)


KINDS = ["now", "back", "mid", "liquidity", "book"]
days = sorted(L.day.unique())
scored = []
for i, d in enumerate(days):
    if i < 5:
        continue
    tr, te = L[L.day < d], L[L.day == d].copy()
    gtr, gte = rb.race_codes(tr.snap), rb.race_codes(te.snap)
    book = float(np.median(np.bincount(gtr, weights=1.0 / tr.bsp)))
    inp = rb.closing_inputs(te.assign(race=te.snap, morningwap=te.now, morning_vol=0.0), race_col="race")
    te["q_served"] = served.predict(inp, gte)
    te["ev_served"] = te.back1 * served.book * te.q_served - 1
    for kind in KINDS:
        b = fit(design(tr, tr, kind), tr.q_bsp.to_numpy(), gtr)
        te[f"q_{kind}"] = np.exp(rb.group_log_softmax(design(te, tr, kind) @ b, gte))
        te[f"ev_{kind}"] = te.back1 * book * te[f"q_{kind}"] - 1
        if d == days[-1]:
            print(f"   the {kind} blend fitted on 1-{days[-2][-2:]} Oct, weights {np.round(b, 3)}")
    scored.append(te)
S = pd.concat(scored, ignore_index=True)
S["clv"] = S.back1 / S.bsp - 1
S["guard"] = S.back1 >= S.predicted_bfsp - 1e-9
print(f"\n== A. the live books, scored {S.day.min()} .. {S.day.max()} ({S.snap.nunique():,} reads, {len(S):,} runner-reads)")
print("   the served closing model's weights (market, model): "
      f"{np.round(served.beta[:2], 3)}; refitted blends above")
rows = []
for m in ["served"] + KINDS:
    q = S[f"q_{m}"]
    ev = S[f"ev_{m}"]
    sel = S[(ev >= BAR) & S.guard & (S.back1 <= 1000)].sort_values("polled_utc").drop_duplicates(
        ["day", "market_id", "selection_id"])
    w = np.clip(400 / (sel.back1 - 1), 15, 200)
    rows.append(pd.Series({
        "nearness_abs_log": float(np.mean(np.abs(np.log(S.q_bsp) - np.log(q)))),
        "xent_per_read": float(-(S.q_bsp * np.log(q)).sum() / S.snap.nunique()),
        "share_ev_over_bar": float((ev >= BAR).mean()),
        "rule_horses": len(sel), "rule_staked": float(w.sum()), "rule_clv": float((sel.clv * w).sum() / w.sum()),
        "rule_clv_x_stake": float((sel.clv * w).sum()), "rule_mean_ev": float(sel[f"ev_{m}"].mean())}, name=m))
print(pd.DataFrame(rows).round(4).to_string())
print("\n   calibration: realised CLV at the BSP (mean back / BSP - 1) of every runner-read, by each model's expected CLV")
cal = {}
for m in ["served"] + KINDS:
    e = pd.cut(S[f"ev_{m}"], [-1, -0.10, -0.03, 0, 0.03, 0.06, 0.10, 0.20, 100])
    cal[m] = S.groupby(e, observed=True).clv.mean()
    cal[f"n_{m}"] = S.groupby(e, observed=True).size()
print(pd.DataFrame(cal).round(4).to_string())
print("\n   the rule by the market's money and the spread (served against the book blend: stake-weighted CLV)")
for m in ("served", "book"):
    sel = S[(S[f"ev_{m}"] >= BAR) & S.guard].sort_values("polled_utc").drop_duplicates(["day", "market_id", "selection_id"])
    sel = sel.assign(w=np.clip(400 / (sel.back1 - 1), 15, 200))
    for col, bins in (("market_total_matched", [-1, 1000, 2500, 5000, 20000, 1e12]), ("spread", [-0.01, 0.02, 0.05, 0.10, 0.25, 1.01])):
        t = sel.groupby(pd.cut(sel[col], bins), observed=True).apply(lambda g: pd.Series({
            "horses": len(g), "staked": g.w.sum(), "clv": (g.clv * g.w).sum() / g.w.sum()}))
        print(f"   {m}, by {col}:\n" + t.round(4).to_string())

print("\n== B. Benter to the result, the live books (conditional logit of the winner)")
for tag, part in (("first read of each market", L[L["first"]]), ("last read before the off", L[L["last"]])):
    part = part.copy()
    g = rb.race_codes(part.snap)
    part["lq_bsp"] = lq(part.bsp, g)
    ll0 = -np.log(np.bincount(g)).sum()
    print(f"   {tag}: {g.max() + 1:,} races")
    for cols in (["lq_now"], ["lq_model"], ["lq_now", "lq_model"], ["lq_bsp"], ["lq_bsp", "lq_model"],
                 ["lq_bsp", "lq_now", "lq_model"], ["lq_bsp", "wom", "spread", "thin", "ltp"]):
        X = np.column_stack([part[c] if c.startswith("lq") else z(part[c], part[c]) for c in cols])
        b = fit(X, part.win.to_numpy(), g, l2=1e-6)
        se, ll = clogit_se(X, part.win.to_numpy(), g, b)
        print(f"     {' + '.join(cols):34s} weights {np.round(b, 3)} se {np.round(se, 3)} log-lik {ll:,.1f} "
              f"pseudo-R2 {1 - ll / ll0:.4f}")

# ===================================================================================== B. the history
print("\n== B. Benter to the result, the history (the 06:00 files from 1 Aug, Betfair's price files, results to 13 Oct)")
p_hist = preds(date(2026, 8, 1), date(2026, 10, 19))
bf = pd.read_sql_query("""SELECT r.race_date, r.track, r.race_time, r.horse_name, b.event_id, b.win_lose, b.bsp,
                                 b.morningwap, b.ppwap, b.morning_vol
                          FROM betfair_prices b JOIN race_results r ON r.id = b.race_results_id
                          WHERE LOWER(b.market_type) = 'win' AND r.race_date >= '2026-08-01'""", conn)
bf["_k"] = (bf.race_date.astype(str).str.slice(0, 10) + "|" + bf.track.map(normalise_track) + "|"
            + bf.race_time.map(race_time_to_24h).astype(str) + "|" + bf.horse_name.map(normalise_horse))
h = bf.drop_duplicates("_k").merge(p_hist, on="_k")
h["day"] = h.race_date.astype(str).str.slice(0, 10)
h["race"] = h.day + "|" + h.event_id.astype(str)
okh = (h.bsp > 1) & (h.predicted_bfsp > 1) & (h.morningwap > 1) & (h.ppwap > 1) & h.win_lose.notna()
field = h.groupby("race").race.transform("size")
h = h[okh.groupby(h.race).transform("all") & (field >= 2)].copy()
h["win"] = (pd.to_numeric(h.win_lose, errors="coerce") == 1).astype(float)
h = h[h.groupby("race").win.transform("sum") == 1].sort_values(["day", "race"]).reset_index(drop=True)
g = rb.race_codes(h.race)
for c, src in (("lq_morning", "morningwap"), ("lq_pp", "ppwap"), ("lq_bsp", "bsp"), ("lq_model", "predicted_bfsp")):
    h[c] = lq(h[src], g)
ll0 = -np.log(np.bincount(g)).sum()
print(f"   {g.max() + 1:,} races, {len(h):,} runners, {h.day.min()} .. {h.day.max()}")
for cols in (["lq_morning"], ["lq_model"], ["lq_morning", "lq_model"], ["lq_pp"], ["lq_pp", "lq_model"], ["lq_bsp"],
             ["lq_bsp", "lq_model"], ["lq_bsp", "lq_morning", "lq_model"]):
    X = h[cols].to_numpy(float)
    b = fit(X, h.win.to_numpy(), g, l2=1e-6)
    se, ll = clogit_se(X, h.win.to_numpy(), g, b)
    print(f"     {' + '.join(cols):34s} weights {np.round(b, 3)} se {np.round(se, 3)} log-lik {ll:,.1f} "
          f"pseudo-R2 {1 - ll / ll0:.4f}")

print("\n   the rules walk-forward by week (fit on the weeks before, from the fourth week): backed at the morning WAP")
h["week"] = pd.to_datetime(h.day).dt.to_period("W").astype(str)
weeks = sorted(h.week.unique())
out = []
for i, wk in enumerate(weeks):
    if i < 3:
        continue
    tr, te = h[h.week < wk], h[h.week == wk].copy()
    gtr, gte = rb.race_codes(tr.race), rb.race_codes(te.race)
    Xtr, Xte = tr[["lq_morning", "lq_model"]].to_numpy(float), te[["lq_morning", "lq_model"]].to_numpy(float)
    q_close = np.exp(lq(tr.bsp, gtr))
    b_close = fit(Xtr, q_close, gtr)                       # the closing blend: to the BSP book
    b_win = fit(Xtr, tr.win.to_numpy(), gtr, l2=1e-6)      # Benter's blend: to the winner
    book = float(np.median(np.bincount(gtr, weights=1.0 / tr.bsp)))
    te["ev_close"] = te.morningwap * book * np.exp(rb.group_log_softmax(Xte @ b_close, gte)) - 1
    p_win = np.exp(rb.group_log_softmax(Xte @ b_win, gte))
    te["ev_win"] = p_win * (1 + (te.morningwap - 1) * (1 - C)) - 1
    te["b_close"], te["b_win"] = str(np.round(b_close, 2)), str(np.round(b_win, 2))
    out.append(te)
W = pd.concat(out, ignore_index=True)
W = W[W.morning_vol >= 100]
W["w"] = np.clip(400 / (W.morningwap - 1), 15, 200)
W["clv"] = W.morningwap / W.bsp - 1
W["hold"] = np.where(W.win == 1, (W.morningwap - 1) * (1 - C), -1.0)
print("   the last fits' weights (market, model): to the BSP", W.b_close.iloc[-1], "; to the winner", W.b_win.iloc[-1])


def rule(sel, name):
    w = sel.w
    r = sel.race.unique()
    rng = np.random.default_rng(0)
    by = sel.groupby("race")
    agg = pd.DataFrame({"w": by.w.sum(), "c": by.apply(lambda x: (x.clv * x.w).sum()),
                        "hd": by.apply(lambda x: (x.hold * x.w).sum())})
    boots = [agg.iloc[rng.integers(0, len(agg), len(agg))].sum() for _ in range(300)]
    bh = np.array([b_.hd / b_.w for b_ in boots])
    print(f"   {name:44s} horses {len(sel):5d} staked GBP{w.sum():10,.0f}  CLV at the BSP "
          f"{(sel.clv * w).sum() / w.sum():+.2%}  held to the result {(sel.hold * w).sum() / w.sum():+.2%} "
          f"(90% {np.quantile(bh, 0.05):+.1%} to {np.quantile(bh, 0.95):+.1%}) over {len(r)} races")


rule(W[W.ev_close >= BAR], "closing blend: expected CLV >= 3%")
rule(W[W.ev_win >= BAR], "Benter blend: expected return >= 3% (hold)")
rule(W[(W.ev_win >= BAR) & (W.ev_close >= BAR)], "both")
rule(W[(W.ev_win >= BAR) & (W.ev_close < BAR)], "Benter only")
rule(W[(W.ev_close >= BAR) & (W.ev_win < BAR)], "closing only")
for lo, hi in ((0.03, 0.06), (0.06, 0.10), (0.10, 0.20), (0.20, 100)):
    rule(W[(W.ev_close >= lo) & (W.ev_close < hi)], f"closing blend {lo:.0%}-{hi:.0%}")
