"""The Professor, bookmaker prices only, with a profit at BFSP (the owner, 25 Oct). Read-only. Fixed before the run.

professor_bsp_1025 (run 40178868298) found that at the bookmakers' real margin the 10%-over-the-close rule picks long
prices that drift (the bookmakers' price a median 11-21% under the BSP, strike about 10%), and that the model's
candidates as a whole win as the BSP says (0.989 of the BSP's expected winners), except the shortest: p_win over 0.2
won 1.03-1.10 of expectation (789 horses, Aug-Oct). The hypothesis: the model's short-priced horses beat the BSP, so
a bookmaker-only rule restricted to them can show a profit at BFSP.

Grid: the bookmakers' price = the exchange's / margin m in {1.14, 1.18}; min_p_win in {0.15, 0.20, 0.25, 0.30}; the
overlay of the bookmakers' price over the expected BSP at least bar in {-0.10, -0.05, 0, 0.05, 0.10}; one a race, by
p_win, 5 a day. Scored held to the result at the BSP after 2% (the profit at BFSP) and at the bookmakers' price.
The decision (set now): a cell is viable only if its ROI at the BSP and at the bookmakers' price are both above zero
on Aug-Sep, on Oct and on the live record (1-22 Oct, the trader's first price), with at least one tip a day on each.
"""
import io
import itertools
import os
import sqlite3
import sys
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from betfair_prices import normalise_horse, normalise_track, race_time_to_24h  # noqa: E402
from model import race_book as rb  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 500)
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)

BAR_LIVE, C, MAX_PRICE, FREE = 0.03, 0.02, 51.0, 2
LIVE_DAYS = [(date(2026, 10, 1) + timedelta(n)).isoformat() for n in range(22)]
CUT_UK = "09:55"
GRID = dict(overlay_bar=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25], min_p_win=[0.0, 0.08, 0.12, 0.16], max_per_race=[1],
            max_tips=[2, 5, 10])
HAIRCUTS = [0.05, 0.10, 0.15]
K = ["day", "market_id", "selection_id"]
KEYS = ["overlay_bar", "min_p_win", "per_race", "max_tips"]
COLS = KEYS + ["tips", "a_day", "strike", "exp_strike", "avg_price", "pts", "roi", "roi_lo", "roi_hi", "clv", "luck",
               "clv_hc5", "clv_hc10", "clv_hc15", "roi_bog", "roi_sp", "roi_bsp", "worst_run", "drawdown",
               "free_tips", "free_strike", "free_exp_strike", "free_roi", "free_clv"]


def get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001
        return None


# ===================================================================================== the scoring
def score(sel: pd.DataFrame, price_col: str) -> dict:
    """Held to the result at 1 point level at `price_col`; the BSP leg after commission; runs and drawdown in time."""
    if sel.empty:
        return {"tips": 0}
    s = sel.sort_values(["day", "race_order"])
    p = s[price_col].to_numpy(float)
    won = s.won.to_numpy(bool)
    pnl = np.where(won, p - 1.0, -1.0)
    sp = s.sp.to_numpy(float)
    has_sp = np.isfinite(sp) & (sp > 1)
    bog = np.where(has_sp, np.maximum(p, sp), p)
    pnl_bog = np.where(won, bog - 1.0, -1.0)
    pnl_sp = np.where(has_sp, np.where(won, sp - 1.0, -1.0), np.nan)
    pnl_bsp = np.where(won, (s.bsp.to_numpy(float) - 1.0) * (1 - C), -1.0)
    cum = np.concatenate([[0.0], np.cumsum(pnl)])
    dd = float((np.maximum.accumulate(cum) - cum).max())
    runs, run = 0, 0
    for w in won:
        run = 0 if w else run + 1
        runs = max(runs, run)
    lo, hi = rb.ratio_interval(pnl, np.ones(len(pnl)), n=1000)
    by_day = pd.Series(pnl).groupby(s.day.to_numpy()).sum()
    bsp = s.bsp.to_numpy(float)
    clv = float(np.mean(p / bsp - 1.0))
    return {"tips": len(s), "a_day": len(s) / max(s.day.nunique(), 1), "strike": float(won.mean()),
            "exp_strike": float(np.mean(1.0 / bsp)), "avg_price": float(np.mean(p)), "pts": float(pnl.sum()),
            "roi": float(pnl.mean()), "roi_lo": lo, "roi_hi": hi, "clv": clv, "luck": float(pnl.mean()) - clv,
            "clv_hc5": float(np.mean((1 + (p - 1) * 0.95) / bsp - 1.0)),
            "clv_hc10": float(np.mean((1 + (p - 1) * 0.90) / bsp - 1.0)),
            "clv_hc15": float(np.mean((1 + (p - 1) * 0.85) / bsp - 1.0)),
            "roi_bog": float(pnl_bog.mean()), "roi_sp": float(np.nanmean(pnl_sp)) if has_sp.any() else np.nan,
            "roi_bsp": float(pnl_bsp.mean()), "worst_run": runs, "drawdown": dd, "days_up": float((by_day > 0).mean())}


def pick(d: pd.DataFrame, price_col: str, overlay_bar: float, min_p_win: float, max_per_race: int, max_tips: int,
         rank_by: str = "p_win") -> pd.DataFrame:
    """The rule on one population: qualifiers, one or all a race, ranked within the day, capped."""
    x = d[(d[price_col] > 1) & (d[price_col] <= MAX_PRICE)].copy()
    x["overlay"] = x[price_col] / x.exp_bsp - 1.0
    x = x[(x.overlay >= overlay_bar) & (x.p_win >= min_p_win)]
    key = {"p_win": ["p_win", "overlay"], "overlay": ["overlay", "p_win"], "edge": ["edge", "p_win"]}[rank_by]
    x = x.sort_values(["day"] + key, ascending=[True, False, False])
    if max_per_race == 1:
        x = x.drop_duplicates(["day", "race"], keep="first")
    x["rank"] = x.groupby("day").cumcount() + 1
    return x[x["rank"] <= max_tips]


def cell(d: pd.DataFrame, price_col: str, ob, mp, pr, mt, rank_by="p_win") -> dict:
    sel = pick(d, price_col, ob, mp, pr, mt, rank_by)
    r = {"overlay_bar": ob, "min_p_win": mp, "per_race": pr, "max_tips": mt}
    r.update(score(sel, price_col))
    f = score(sel[sel["rank"] <= FREE], price_col)
    r.update({"free_tips": f.get("tips", 0), "free_strike": f.get("strike", np.nan),
              "free_exp_strike": f.get("exp_strike", np.nan), "free_roi": f.get("roi", np.nan),
              "free_clv": f.get("clv", np.nan)})
    return r


def grid(d: pd.DataFrame, price_col: str, label: str, full: bool = True) -> pd.DataFrame:
    g = pd.DataFrame([cell(d, price_col, *c) for c in itertools.product(*GRID.values())])
    print(f"\n== grid: {label} ({d.day.nunique()} days, {len(d):,} candidates), held at {price_col}"
          + ("" if full else ": the 12 best by ROI"))
    show = g if full else g.sort_values("roi", ascending=False).head(12)
    print(show[[c for c in COLS if c in show.columns]].round(3).to_string(index=False))
    return g


def choose(train: pd.DataFrame, hold: pd.DataFrame, live: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Two picks on the grid tables. The pre-registered rule (the first run): the highest realised ROI on Aug-Sep with
    the holdout's BSP ROI positive and the free strike at least 20% on both. The revised rule: the highest BSP-implied
    ROI (the CLV) on Aug-Sep among the cells with at least 4 tips a day, with the holdout's and the live record's
    implied ROI positive and the free tips' expected strike at least 20% on Aug-Sep; the simpler cell at a tie."""
    m = train.merge(hold, on=KEYS, suffixes=("", "_hold")).merge(live, on=KEYS, suffixes=("", "_live"))
    ok = m[(m.roi_bsp_hold > 0) & (m.free_strike >= 0.20) & (m.free_strike_hold >= 0.20) & (m.tips >= 100)]
    pre = (ok if len(ok) else m).sort_values("roi", ascending=False).iloc[0]
    ok = m[(m.a_day >= 4) & (m.clv_hold > 0) & (m.clv_live > 0) & (m.free_exp_strike >= 0.20)]
    if ok.empty:
        print("   no cell meets the revised rule; the best implied ROI with 4 a day instead")
        ok = m[m.a_day >= 4]
    best = ok.clv.max()
    near = ok[ok.clv >= best - 0.005].sort_values(["min_p_win", "overlay_bar", "per_race", "max_tips"],
                                                 ascending=[True, True, True, False])
    return pre, near.iloc[0]


def by_rank(sel: pd.DataFrame, price_col: str, label: str):
    r = pd.cut(sel["rank"], [0, 1, 2, 3, 5, 10], labels=["1", "2", "3", "4-5", "6-10"])
    t = sel.groupby(r, observed=True).apply(lambda x: pd.Series({
        "tips": len(x), "strike": x.won.mean(), "exp_strike": (1 / x.bsp).mean(), "avg_price": x[price_col].mean(),
        "roi": np.where(x.won, x[price_col] - 1, -1.0).mean(), "clv": (x[price_col] / x.bsp - 1).mean(),
        "p_win": x.p_win.mean(), "overlay": (x[price_col] / x.exp_bsp - 1).mean()}))
    print(f"   {label}, by rank within the day:\n" + t.round(3).to_string())


def bands(sel: pd.DataFrame, price_col: str, label: str):
    b = pd.cut(sel[price_col], [1, 4, 8, 16, 51], right=False)
    t = sel.groupby(b, observed=True).apply(lambda x: pd.Series({
        "tips": len(x), "strike": x.won.mean(), "roi": np.where(x.won, x[price_col] - 1, -1.0).mean(),
        "roi_bsp": np.where(x.won, (x.bsp - 1) * (1 - C), -1.0).mean(), "clv": (x[price_col] / x.bsp - 1).mean()}))
    print(f"   {label}, by price band:\n" + t.round(3).to_string())


def line(d: dict) -> str:
    return ", ".join(f"{k} {v:+.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in d.items())


# ===================================================================================== A. the live rule's own horses
print("== A. the live rule's own horses, 1-22 Oct (the ledger)")
NUM = ["matched", "avg_price", "bsp", "clv", "selection_id", "edge", "pnl", "minutes_to_off", "best_back", "p_model",
       "p_pool", "p_market", "move"]
leds = []
for day in LIVE_DAYS:
    raw = get(f"trading/live/{day}/ledger.csv")
    if raw is None:
        print("   no ledger", day)
        continue
    d = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
    d["day"] = day
    leds.append(d)
led = pd.concat(leds, ignore_index=True)
for c in NUM:
    if c in led.columns:
        led[c] = pd.to_numeric(led[c], errors="coerce")
led["t"] = pd.to_datetime(led.ts, utc=True, errors="coerce")
b = led[(led.event == "back") & (led.matched > 0)].sort_values("t")
pos = b.groupby(K).agg(runner=("runner", "first"), venue=("venue", "first"), off=("off", "first"),
                       first_price=("avg_price", "first"), p_pool=("p_pool", "first"), edge=("edge", "first"),
                       move=("move", "first"), best_back=("best_back", "first"), first_t=("t", "first"),
                       stake=("matched", "sum")).reset_index()
pos["first_uk"] = pos.first_t.dt.tz_convert("Europe/London").dt.strftime("%H:%M")
pos["race_order"] = pd.to_datetime(pos.off, utc=True, errors="coerce")
pos["race"] = pos.day + "|" + pos.market_id
pos["exp_bsp"] = 1.0 / pos.p_pool
pos["p_win"] = pos.p_pool
st = led[led.event == "settle"].sort_values("t").drop_duplicates(K, keep="last")[K + ["result", "bsp"]]
pos = pos.merge(st, on=K, how="left")
mk = pd.read_sql_query(f"""SELECT m.race_date AS day, m.market_id, m.selection_id, r.odds AS sp, r.placing_numerical
                           FROM betfair_live_markets m LEFT JOIN race_results r ON r.id = m.race_results_id
                           WHERE m.race_date >= '{LIVE_DAYS[0]}' AND m.race_date <= '{LIVE_DAYS[-1]}'""", conn)
mk["selection_id"] = pd.to_numeric(mk.selection_id, errors="coerce")
mk["market_id"] = mk.market_id.astype(str)
pos = pos.merge(mk.drop_duplicates(K), on=K, how="left")
marks = pd.read_sql_query(f"""SELECT race_date AS day, market_id, selection_id, mark, back1
                              FROM betfair_live_marks WHERE mark IN ('clock_09:00', 'clock_10:00')
                              AND race_date >= '{LIVE_DAYS[0]}' AND race_date <= '{LIVE_DAYS[-1]}'""", conn)
marks["selection_id"] = pd.to_numeric(marks.selection_id, errors="coerce")
marks["market_id"] = marks.market_id.astype(str)
marks["back1"] = pd.to_numeric(marks.back1, errors="coerce")
for mark, col in (("clock_10:00", "back_10"), ("clock_09:00", "back_09")):
    m = marks[marks.mark == mark].drop_duplicates(K)[K + ["back1"]].rename(columns={"back1": col})
    pos = pos.merge(m, on=K, how="left")
pos["sp"] = pd.to_numeric(pos.sp, errors="coerce")
pos["won"] = pos.result.eq("WINNER")
A = pos[pos.result.isin(["WINNER", "LOSER"]) & (pos.bsp > 1) & (pos.first_price > 1) & (pos.exp_bsp > 1)].copy()
A["mark_price"] = A.back_10.where(A.back_10 > 1, A.back_09)
print(f"   positions {len(pos):,} on {pos.day.nunique()} days; settled with a BSP {len(A):,}; with a 10:00 mark "
      f"{(A.back_10 > 1).mean():.1%} (09:00 or 10:00 {(A.mark_price > 1).mean():.1%}); with an industry SP "
      f"{(A.sp > 1).mean():.1%}; first backed by {CUT_UK} UK {(A.first_uk <= CUT_UK).mean():.1%}")
print(f"   winners {int(A.won.sum())} of {len(A)} ({A.won.mean():.1%}); p_win summed {A.p_win.sum():.1f}; "
      f"first price <= 51: {(A.first_price <= MAX_PRICE).mean():.1%}")
print("   every position held at its first price (no selection, <= 51): "
      + line(score(A[A.first_price <= MAX_PRICE], "first_price")))
A_cut = A[A.first_uk <= CUT_UK].copy()
print(f"   the positions held by {CUT_UK} UK: {len(A_cut):,} on {A_cut.day.nunique()} days; held at the first price: "
      + line(score(A_cut[A_cut.first_price <= MAX_PRICE], "first_price")))
gA = None
A_mark = A_cut[A_cut.mark_price > 1].copy()
gA_mark = None

# ===================================================================================== B. the history
print("\n== B. the history, 1 Aug - 22 Oct (the 06:00 files, Betfair's price files, race_results)")


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
    for a, bb in (("race_date", "date"), ("track", "venue"), ("horse_name", "runner_name")):
        if a not in p.columns and bb in p.columns:
            p[a] = p[bb]
    p["race_date"] = p["race_date"].fillna(p["file_day"]).astype(str).str.slice(0, 10)
    p["predicted_bfsp"] = pd.to_numeric(p["predicted_bfsp"], errors="coerce")
    p["_k"] = (p.race_date + "|" + p.track.map(normalise_track) + "|" + p.race_time.map(race_time_to_24h).astype(str)
               + "|" + p.horse_name.map(normalise_horse))
    return p.drop_duplicates("_k", keep="last")[["_k", "predicted_bfsp"]]


p_hist = preds(date(2026, 8, 1), date(2026, 10, 22))
bf = pd.read_sql_query("""SELECT r.race_date, r.track, r.race_time, r.horse_name, r.odds AS sp, r.placing_numerical,
                                 b.event_id, b.win_lose, b.bsp, b.morningwap, b.morning_vol
                          FROM betfair_prices b JOIN race_results r ON r.id = b.race_results_id
                          WHERE LOWER(b.market_type) = 'win' AND r.race_date >= '2026-08-01' AND r.race_date <= '2026-10-22'""",
                       conn)
bf["_k"] = (bf.race_date.astype(str).str.slice(0, 10) + "|" + bf.track.map(normalise_track) + "|"
            + bf.race_time.map(race_time_to_24h).astype(str) + "|" + bf.horse_name.map(normalise_horse))
h = bf.drop_duplicates("_k").merge(p_hist, on="_k")
h["day"] = h.race_date.astype(str).str.slice(0, 10)
h["race"] = h.day + "|" + h.event_id.astype(str)
h["race_order"] = h.race_time.map(race_time_to_24h).astype(str)
for c in ("sp", "bsp", "morningwap", "morning_vol"):
    h[c] = pd.to_numeric(h[c], errors="coerce")
okh = (h.bsp > 1) & (h.predicted_bfsp > 1) & (h.morningwap > 1) & h.win_lose.notna()
field = h.groupby("race").race.transform("size")
h = h[okh.groupby(h.race).transform("all") & (field >= 2)].copy()
h["won"] = pd.to_numeric(h.win_lose, errors="coerce") == 1
h = h[h.groupby("race").won.transform("sum") == 1].sort_values(["day", "race"]).reset_index(drop=True)
print(f"   {h.race.nunique():,} whole races, {len(h):,} runners, {h.day.min()} .. {h.day.max()}; with an industry SP "
      f"{(h.sp > 1).mean():.1%}")

served = rb.load_closing_model("data/models/closing_model_novol.json")
g = rb.race_codes(h.race)
inputs = rb.closing_inputs(h.assign(morning_vol=0.0), race_col="race")
q_hat, sig = served.predict(inputs, g), served.sigma(inputs)
ev, inv = np.full(len(h), np.nan), np.full(len(h), np.nan)
rng = np.random.default_rng(1023)
starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
ends = np.r_[starts[1:], len(g)]
mw = h.morningwap.to_numpy(float)
for s0, s1 in zip(starts, ends):
    draws = rb.closing_draws(q_hat[s0:s1], sig[s0:s1], book=served.book, n_draws=2000, rng=rng)
    inv[s0:s1] = draws.mean(axis=0)
    ev[s0:s1] = mw[s0:s1] * inv[s0:s1] - 1.0
h["edge"], h["exp_bsp"], h["p_win"] = ev, 1.0 / inv, inv
h["over_model"] = h.morningwap >= h.predicted_bfsp - 1e-9
B = h[(h.edge >= BAR_LIVE) & h.over_model & (h.morningwap <= 200)].copy()
B["price"] = B.morningwap
print(f"   the live rule's candidates (expected CLV >= 3% at the morning WAP, at or over our price, <= 200): "
      f"{len(B):,} horses on {B.day.nunique()} days ({len(B) / B.day.nunique():.0f} a day), winners {int(B.won.sum())} "
      f"({B.won.mean():.1%}), p_win summed {B.p_win.sum():.0f}; at <= 51: {int((B.price <= MAX_PRICE).sum()):,}")
print("   every candidate held at the morning WAP (no selection, <= 51): "
      + line(score(B[B.price <= MAX_PRICE], "price")))
print("   the calibration of p_win on the candidates (winners against p_win summed) by p_win band:")
pb = pd.cut(B.p_win, [0, 0.05, 0.08, 0.12, 0.2, 0.3, 1.0])
print(B.groupby(pb, observed=True).apply(lambda x: pd.Series({
    "horses": len(x), "winners": int(x.won.sum()), "p_win_sum": x.p_win.sum(),
    "won_over_expected": x.won.sum() / x.p_win.sum(), "mean_price": x.price.mean(),
    "mean_exp_bsp": x.exp_bsp.mean(), "mean_bsp": x.bsp.mean()})).round(3).to_string())

train, hold = B[B.day < "2026-10-01"].copy(), B[B.day >= "2026-10-01"].copy()
gT = None
gH = None



def cell_score(d, col, m, mp, bar):
    dd = d.copy()
    dd["books"] = dd[col] / m
    sel = pick(dd, "books", bar, mp, 1, 5)
    if sel.empty:
        return {"tips": 0}
    won = sel.won.to_numpy(bool)
    bsp = sel.bsp.to_numpy(float)
    pb = np.where(won, (bsp - 1) * (1 - C), -1.0)
    pa = np.where(won, sel.books.to_numpy(float) - 1, -1.0)
    lo, hi = rb.ratio_interval(pb, np.ones(len(pb)), n=1000)
    return {"tips": len(sel), "a_day": len(sel) / max(dd.day.nunique(), 1), "strike": won.mean(),
            "avg_books": sel.books.mean(), "roi_bsp": pb.mean(), "roi_bsp_lo": lo, "roi_bsp_hi": hi,
            "roi_books": pa.mean(), "won_over_bsp": won.sum() / (1 / bsp).sum()}


rows = []
for m in (1.14, 1.18):
    for mp in (0.15, 0.20, 0.25, 0.30):
        for bar in (-0.10, -0.05, 0.0, 0.05, 0.10):
            r = {"margin": m, "min_p_win": mp, "bar": bar}
            for tag, d, col in (("aug_sep", train, "price"), ("oct", hold, "price"), ("live", A_cut, "first_price")):
                s = cell_score(d, col, m, mp, bar)
                for k, v in s.items():
                    r[f"{tag}_{k}"] = v
            rows.append(r)
G = pd.DataFrame(rows)
cols = ["margin", "min_p_win", "bar"] + [f"{t}_{k}" for t in ("aug_sep", "oct", "live") for k in
                                          ("tips", "a_day", "strike", "roi_bsp", "roi_books", "won_over_bsp")]
print("\n== the short-priced bookmaker-only rule: ROI at the BSP (after 2%) and at the bookmakers' price, three samples")
print(G[cols].round(3).to_string(index=False))
ok = G[(G.aug_sep_roi_bsp > 0) & (G.oct_roi_bsp > 0) & (G.live_roi_bsp > 0) & (G.aug_sep_roi_books > 0)
       & (G.oct_roi_books > 0) & (G.live_roi_books > 0) & (G.aug_sep_a_day >= 1) & (G.oct_a_day >= 1) & (G.live_a_day >= 1)]
print(f"\n== viable cells by the rule set before the run: {len(ok)}")
if len(ok):
    print(ok[["margin", "min_p_win", "bar", "aug_sep_roi_bsp", "aug_sep_roi_bsp_lo", "aug_sep_roi_bsp_hi", "oct_roi_bsp",
              "oct_roi_bsp_lo", "oct_roi_bsp_hi", "live_roi_bsp", "live_roi_bsp_lo", "live_roi_bsp_hi"]].round(3).to_string(index=False))
