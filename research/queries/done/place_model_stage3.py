"""Stage 3 for the place market: a place model from our win forecasts, scored against the place SP (read-only).

The place SP misprices against the win SP (scan 2, research/queries/done/edge_bsp_bias.py), but only at the off:
decided at pre-play prices the rules lose their edge (research/queries/done/place_rules_preplay.py, run 37072650112).
What is left is our own information. The served blend's walk-forward forecasts (iteration 105's oos_gate_step.csv,
run 36407030755) price every runner at 06:00, and the closing model turns them and the morning market into the
expected BSP book. Each win book becomes a place chance by scan 2's discounted Harville (ordering exponents fitted on
2021-22 finishing orders) for the places the place market paid. Betfair's win and place rows come from the database
(the nightly load); place rows the database lacks are read from the archived day files. Development window only:
nothing on or after 1 April 2026 is read. Every fit is walk-forward by month (fitted on the months before the month it
scores). 2% commission.

1. Forecasting "placed" (the place market's settled result): log loss and Brier of the place chance from the place SP
   (normalised to the places paid), the place pre-play WAP, the win SP (scan 2's reference), the win pre-play WAP, the
   win morning WAP, our forecast and the closing model's book.
2. Benter's test against the place SP: placed ~ logit(q_place_sp) + logit(q_x), each month fitted on the months before
   it; the test months' log loss against the place SP alone (recalibrated the same way), with a race-bootstrap
   interval, and the weight on q_x. A gain resolvably below zero is information the place SP does not hold.
3. Value at the place SP: back (and lay) at the place SP where the expected value clears 0/3/5/10%, decided
   - in the morning: the chance from our forecast and the morning market, the place SP forecast from the same;
   - near the off: the chance from the place and win pre-play WAPs and our forecast, the place SP forecast from them
     (the pre-play WAP stands in for a price some minutes before the off);
   - at the SP (not tradable; how much there is to find): the chance from the place SP, the win SP and our forecast.
   Settled at the place SP. Backs also at the owner's staking (to win GBP250 at the forecast place SP).
4. The win rule's horses in the place market: for the horses the live rule would back (expected CLV >= 3% under the
   closing model, at least GBP100 matched in the morning), the place pre-play WAP against the place SP, beside the win
   market's morning and pre-play WAPs against the win SP; every runner's drift is the baseline. The recorder's books
   (from 1 Oct) hold the place market's morning prices; this is the files' first look.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research", "queries", "done"))
from model import race_book as rb  # noqa: E402
from model.ordering import place_probabilities  # noqa: E402

t0 = time.time()
pd.set_option("display.width", 240)
pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 40)
UNTIL = "2026-04-01"
COMM = 0.02
TO_WIN = 250.0
MIN_VOL = 100.0
BARS = (0.0, 0.03, 0.05, 0.10)
BANDS = ((2, 7), (8, 11), (12, 15), (16, 40))
FIT = {(2, 7): (0.734, 0.557), (8, 11): (0.804, 0.640), (12, 15): (0.839, 0.697), (16, 40): (0.829, 0.735)}  # scan 2
KEY = ["race_date", "track", "race_time", "horse_name"]
PRICE_BANDS = [1, 2, 3, 5, 8, 13, 21, 1e9]
PRICE_LABELS = ["1-2", "2-3", "3-5", "5-8", "8-13", "13-21", "21+"]


def log(msg: str) -> None:
    print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)


def lg(q) -> np.ndarray:
    q = np.clip(np.asarray(q, float), 1e-4, 1 - 1e-4)
    return np.log(q / (1 - q))


def band_of(n: int) -> tuple[int, int]:
    return next((b for b in BANDS if b[0] <= n <= b[1]), BANDS[-1])


# ---------------------------------------------------------------------------
# the data
# ---------------------------------------------------------------------------

def load_forecasts() -> pd.DataFrame:
    from race_book_backtest import fetch_forecasts
    o = fetch_forecasts()
    o["race_date"] = o["race_date"].astype(str)
    log(f"forecasts: {len(o):,} rows, {o.race_date.min()} to {o.race_date.max()}")
    o = o[o.race_date < UNTIL].drop_duplicates(KEY).copy()
    o["race"] = o.race_date + "|" + o.track.astype(str) + "|" + o.race_time.astype(str)
    o["field"] = o.groupby("race").horse_name.transform("size")
    return o


def place_rows_from_archive(days: list[str]) -> pd.DataFrame:
    """The place rows of the given racing days from the archived day files (a file is dated the morning after)."""
    import boto3
    import betfair_prices as bp
    bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
    s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
    want = {date.fromisoformat(x) + timedelta(days=k) for x in days for k in (0, 1)}
    names = [n for n in bp.archive_index(s3, bucket) if (m := bp.parse_file_name(n)) and m[1] == "place" and m[2] in want]

    def fetch(name):
        raw = s3.get_object(Bucket=bucket, Key=f"{bp.S3_PREFIX}/{name}")["Body"].read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        try:
            return bp.parse_file_text(text, name)
        except Exception as exc:  # a malformed file is reported and left out
            print(f"  {name}: {type(exc).__name__}")
            return None

    with ThreadPoolExecutor(32) as ex:
        parts = [p for p in ex.map(fetch, names) if p is not None]
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    d = d[(d.market_type == "place") & d.race_date.isin(days)]
    log(f"  {len(names)} archived place files read for {len(days)} racing days: {len(d):,} rows")
    return d.rename(columns={"event_id": "pl_event", "event_dt": "pl_dt", "bsp": "pl_bsp", "ppwap": "pl_ppwap",
                             "pp_vol": "pl_vol", "win_lose": "placed"})[
        ["race_date", "selection_id", "pl_event", "pl_dt", "pl_bsp", "pl_ppwap", "pl_vol", "placed"]]


def load() -> pd.DataFrame:
    o = load_forecasts()
    lo = o.race_date.min()
    conn = sqlite3.connect("horse_racing.db")
    w = pd.read_sql_query(
        """SELECT r.race_date, r.track, r.race_time, r.horse_name, b.selection_id, b.event_id, b.event_dt,
                  b.morningwap, b.morning_vol, b.ppwap, b.bsp, b.win_lose
           FROM betfair_prices b JOIN race_results r ON r.id = b.race_results_id
           WHERE b.market_type = 'win' AND r.race_date >= ? AND r.race_date < ?""", conn, params=(lo, UNTIL))
    pl = pd.read_sql_query(
        """SELECT race_date, selection_id, event_id AS pl_event, event_dt AS pl_dt, bsp AS pl_bsp, ppwap AS pl_ppwap,
                  pp_vol AS pl_vol, win_lose AS placed
           FROM betfair_prices WHERE market_type = 'place' AND race_date >= ? AND race_date < ?""",
        conn, params=(lo, UNTIL))
    nwin = pd.read_sql_query(
        """SELECT event_id, COUNT(*) AS n_win FROM betfair_prices
           WHERE market_type = 'win' AND bsp > 1 AND race_date >= ? AND race_date < ? GROUP BY event_id""",
        conn, params=(lo, UNTIL))
    conn.close()
    w["race_date"], pl["race_date"] = w.race_date.astype(str), pl.race_date.astype(str)
    cover = pd.DataFrame({"win rows matched": w.groupby(w.race_date.str[:7]).size(),
                          "place rows": pl.groupby(pl.race_date.str[:7]).size()}).fillna(0).astype(int)
    log("the database's Betfair rows in the window, by month:\n" + cover.to_string())
    missing = sorted(set(w.race_date) - set(pl.race_date))
    if missing:
        log(f"{len(missing)} racing days with win rows and no place rows in the database: reading the archive")
        extra = place_rows_from_archive(missing)
        if len(extra):
            pl = pd.concat([pl, extra], ignore_index=True)
    w = w[w.selection_id.notna()].drop_duplicates(KEY)
    pl = pl[pl.selection_id.notna()].drop_duplicates(["race_date", "selection_id", "pl_dt"])
    w["selection_id"], pl["selection_id"] = w.selection_id.astype("int64"), pl.selection_id.astype("int64")
    d = o.merge(w, on=KEY, how="inner")
    d = d.merge(pl, on=["race_date", "selection_id"], how="inner")
    d = d[d.event_dt.astype(str) == d.pl_dt.astype(str)]                  # the same race in both markets
    d = d.merge(nwin, on="event_id", how="left")
    ok = ((d.morningwap > 1) & (d.ppwap > 1) & (d.bsp > 1) & (d.pl_bsp > 1) & (d.pl_ppwap > 1)
          & (d.predicted_bfsp > 1) & d.placed.notna())
    d = d[ok].copy()
    # a race is read only whole: every runner the forecasts carry, every runner of the win market, one place market
    d["got"] = d.groupby("race").horse_name.transform("size")
    d["pl_events"] = d.groupby("race").pl_event.transform("nunique")
    d = d[(d.got == d.field) & (d.field == d.n_win) & (d.pl_events == 1)].copy()
    d["placed"] = (d.placed >= 0.5).astype(float)
    d["k"] = d.groupby("race").placed.transform("sum").round().astype(int)
    d = d[(d.k >= 2) & (d.k <= 4) & (d.k < d.field)].copy()
    d["month"] = d.race_date.str[:7]
    log(f"whole races in both markets: {d.race.nunique():,} ({len(d):,} runners); places paid "
        f"{d.groupby('race').k.first().value_counts().sort_index().to_dict()}; months "
        f"{d.groupby('month').race.nunique().to_dict()}")
    return d.sort_values(["race_date", "race", "predicted_bfsp"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# place chances
# ---------------------------------------------------------------------------

def harville_place(d: pd.DataFrame, win_prob: np.ndarray, rng) -> np.ndarray:
    """Each runner's chance to finish in the places paid, from a win book (normalised in the race)."""
    out = np.full(len(d), np.nan)
    for _, ix in d.groupby("race", sort=False).indices.items():
        p = win_prob[ix] / win_prob[ix].sum()
        gm, dl = FIT[band_of(len(ix))]
        out[ix] = place_probabilities(p, int(d.k.iat[ix[0]]), gm, dl, n_sims=4000, rng=rng)
    return np.clip(out, 1e-4, 1 - 1e-4)


def own_place(d: pd.DataFrame, price: str) -> np.ndarray:
    """The place market's own chance: 1/price scaled so the race sums to the places paid."""
    inv = 1.0 / d[price].to_numpy(float)
    tot = d.assign(inv=inv).groupby("race").inv.transform("sum").to_numpy()
    return np.clip(d.k.to_numpy() * inv / tot, 1e-4, 1 - 1e-4)


def closing_books(d: pd.DataFrame) -> np.ndarray:
    """The closing model's expected BSP book for each runner, fitted on the months before the runner's own."""
    q = np.full(len(d), np.nan)
    mon = d.month.to_numpy()
    for m in sorted(set(mon))[1:]:
        tr, te = d[mon < m], d[mon == m]
        if tr.race.nunique() < 500:
            continue
        cm = rb.fit_closing_model(rb.closing_inputs(tr), tr["bsp"], rb.race_codes(tr["race"]))
        q[mon == m] = cm.predict(rb.closing_inputs(te), rb.race_codes(te["race"]))
    return q


# ---------------------------------------------------------------------------
# fits
# ---------------------------------------------------------------------------

def fit_logit(X: np.ndarray, y: np.ndarray, l2: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    X1 = np.column_stack([np.ones(len(y)), X])

    def f(b):
        z = X1 @ b
        return (np.logaddexp(0, z) - y * z).sum() + l2 * (b @ b), X1.T @ (expit(z) - y) + 2 * l2 * b

    b = minimize(f, np.zeros(X1.shape[1]), jac=True, method="L-BFGS-B").x
    p = expit(X1 @ b)
    cov = np.linalg.pinv((X1 * (p * (1 - p))[:, None]).T @ X1)
    return b, np.sqrt(np.clip(np.diag(cov), 0, None))


def pred_logit(b: np.ndarray, X: np.ndarray) -> np.ndarray:
    return expit(np.column_stack([np.ones(len(X)), X]) @ b)


def design(f: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.column_stack([lg(f[c]) if c.startswith("q_") else f[c].to_numpy(float) for c in cols])


def walk_logit(d: pd.DataFrame, cols: list[str], months: list[str]) -> np.ndarray:
    """P(placed) for each scored month from a logistic fitted on the months before it."""
    out = np.full(len(d), np.nan)
    mon = d.month.to_numpy()
    for m in months:
        tr, te = mon < m, mon == m
        if tr.any() and te.any():
            b, _ = fit_logit(design(d[tr], cols), d.placed.to_numpy(float)[tr])
            out[te] = pred_logit(b, design(d[te], cols))
    return out


def walk_ols(d: pd.DataFrame, y: str, cols: list[str], months: list[str]) -> np.ndarray:
    """E[place SP] for each scored month: log place SP regressed on decision-time readings of the months before."""
    out = np.full(len(d), np.nan)
    mon = d.month.to_numpy()
    for m in months:
        tr, te = mon < m, mon == m
        if not (tr.any() and te.any()):
            continue
        Xtr = np.column_stack([np.ones(tr.sum()), design(d[tr], cols)])
        ytr = np.log(d[y].to_numpy(float)[tr])
        beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
        s2 = float(np.var(ytr - Xtr @ beta))
        out[te] = np.exp(np.column_stack([np.ones(te.sum()), design(d[te], cols)]) @ beta + s2 / 2)
    return out


def logloss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def race_mean_interval(v: np.ndarray, race: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """The per-runner mean of v with a 90% interval from resampling races."""
    g = pd.DataFrame({"v": v, "r": race}).groupby("r").v.agg(["sum", "size"])
    s, c = g["sum"].to_numpy(), g["size"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), (n, len(s)))
    r = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.percentile(r, 5)), float(np.percentile(r, 95))


def bet_summary(f: pd.DataFrame, pnl: np.ndarray, stake: np.ndarray, days: int) -> dict:
    n = len(f)
    if n == 0:
        return {"bets": 0}
    by = pd.DataFrame({"p": pnl, "s": stake, "r": f.race.to_numpy()}).groupby("r")[["p", "s"]].sum()
    lo, hi = rb.ratio_interval(by.p.to_numpy(), by.s.to_numpy()) if len(by) >= 20 else (np.nan, np.nan)
    return {"bets": n, "a_day": round(n / max(days, 1), 1), "staked": round(float(stake.sum()), 0),
            "pnl": round(float(pnl.sum()), 0), "roi": round(float(pnl.sum() / stake.sum()), 4),
            "lo90": round(lo, 4), "hi90": round(hi, 4)}


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

d = load()
if d.race.nunique() < 1000:
    raise SystemExit("too few whole races to read")
rng = np.random.default_rng(0)
d["q_psp"] = own_place(d, "pl_bsp")
d["q_ppp"] = own_place(d, "pl_ppwap")
d["q_winsp"] = harville_place(d, 1 / d.bsp.to_numpy(float), rng)
d["q_wpp"] = harville_place(d, 1 / d.ppwap.to_numpy(float), rng)
d["q_morning"] = harville_place(d, 1 / d.morningwap.to_numpy(float), rng)
d["q_model"] = harville_place(d, 1 / d.predicted_bfsp.to_numpy(float), rng)
qc = closing_books(d)
d["has_close"] = np.isfinite(qc)
d["q_close"] = np.nan
if d.has_close.any():
    has = d.has_close.to_numpy()
    d.loc[has, "q_close"] = harville_place(d[has].reset_index(drop=True), qc[has], rng)
d["log_field"] = np.log(d.field)
d["k2"], d["k4"] = (d.k == 2).astype(float), (d.k == 4).astype(float)
d["log_pl_ppwap"] = np.log(d.pl_ppwap)
d["win_band"] = pd.cut(d.morningwap, PRICE_BANDS, right=False, labels=PRICE_LABELS)        # ordered by price
log("place chances done")
OUT = Path("out/place_model_stage3")
OUT.mkdir(parents=True, exist_ok=True)

months = sorted(d.month.unique())
scored = [m for m in months[1:] if d[d.month < m].race.nunique() >= 500]
S = d[d.month.isin(scored)].copy()
y = S.placed.to_numpy(float)
days = S.race_date.nunique()
print(f"\nscored months {scored[0]} to {scored[-1]}: {S.race.nunique():,} races, {len(S):,} runners, {days} days; "
      f"placed {y.mean():.3f} a runner")

# 1. forecasting "placed"
print("\n== 1. forecasting placed: log loss and Brier a runner on the scored months (lower is better)")
rows = []
for name, col, when in (("place SP", "q_psp", "at the off"), ("place pre-play WAP", "q_ppp", "before the off"),
                        ("win SP (Harville)", "q_winsp", "at the off"), ("win pre-play WAP", "q_wpp", "before the off"),
                        ("win morning WAP", "q_morning", "morning"), ("our forecast", "q_model", "06:00"),
                        ("closing model's book", "q_close", "morning")):
    f = S[S[col].notna()]
    rows.append({"place chance from": name, "when": when, "runners": len(f),
                 "log loss": logloss(f.placed.to_numpy(float), f[col].to_numpy(float)).mean(),
                 "brier": float(((f[col] - f.placed) ** 2).mean()), "mean q": f[col].mean()})
print(pd.DataFrame(rows).round(5).to_string(index=False))

# 2. Benter's test against the place SP
print("\n== 2. Benter's test: placed ~ logit(q_place_sp) + logit(q_x), each month fitted on the months before it")
base = walk_logit(d, ["q_psp"], scored)
ll_base = logloss(d.placed.to_numpy(float), base)
rows = []
for name, cols in (("+ our forecast", ["q_psp", "q_model"]), ("+ closing model's book", ["q_psp", "q_close"]),
                   ("+ win morning WAP", ["q_psp", "q_morning"]), ("+ our forecast + morning", ["q_psp", "q_model", "q_morning"]),
                   ("+ win SP (scan 2, at the off)", ["q_psp", "q_winsp"]),
                   ("+ win SP + our forecast", ["q_psp", "q_winsp", "q_model"])):
    if "q_close" in cols:      # the closing book exists on the scored months only: fitted on those before each
        has = d.has_close.to_numpy()
        sub = d[has]
        use = [m for m in scored if sub[sub.month < m].race.nunique() >= 500]
        p = np.full(len(d), np.nan)
        p[has] = walk_logit(sub, cols, use)
    else:
        p = walk_logit(d, cols, scored)
    m = np.isfinite(p) & np.isfinite(base)
    diff = logloss(d.placed.to_numpy(float)[m], p[m]) - ll_base[m]
    est, lo, hi = race_mean_interval(diff, d.race.to_numpy()[m])
    full = d[m]
    b, se = fit_logit(design(full, cols), full.placed.to_numpy(float))
    rows.append({"model": name, "runners": int(m.sum()), "log loss vs place SP alone": est, "lo90": lo, "hi90": hi,
                 "weights (intercept, then each)": " ".join(f"{v:+.3f}({s:.3f})" for v, s in zip(b, se))})
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:+.5f}"))
for name, cols in (("+ our forecast", ["q_psp", "q_model"]), ("+ win SP", ["q_psp", "q_winsp"])):
    p = walk_logit(d, cols, scored)
    t = pd.DataFrame({"month": d.month, "diff": logloss(d.placed.to_numpy(float), p) - ll_base}).dropna()
    print(f"  {name} by month: " + "  ".join(f"{k} {v:+.5f}" for k, v in t.groupby("month")["diff"].mean().items()))

# 3. value at the place SP
print("\n== 3. value at the place SP (per GBP1 backed, per GBP1 laid; the lay's liability beside it)")
placed = d.placed.to_numpy(float) == 1
back_pnl = np.where(placed, (d.pl_bsp - 1) * (1 - COMM), -1.0)
lay_pnl = np.where(placed, -(d.pl_bsp - 1), 1 - COMM)
DEC = {
    "morning": (["q_model", "q_morning"], ["q_model", "q_morning", "log_field", "k2", "k4"]),
    "near the off": (["q_ppp", "q_wpp", "q_model"], ["log_pl_ppwap", "q_wpp", "q_model", "log_field", "k2", "k4"]),
    "at the SP (not tradable)": (["q_psp", "q_winsp", "q_model"], None),
}
res, picks = [], {}
for how, (pcols, scols) in DEC.items():
    q = walk_logit(d, pcols, scored)
    price = walk_ols(d, "pl_bsp", scols, scored) if scols else d.pl_bsp.to_numpy(float)
    ev_b = q * (price - 1) * (1 - COMM) - (1 - q)
    ev_l = (1 - q) * (1 - COMM) - q * (price - 1)
    if scols:
        f = np.isfinite(price)
        err = np.log(d.pl_bsp.to_numpy(float)[f] / price[f])
        print(f"  {how}: the place SP forecast's miss, median |log| {np.median(np.abs(err)):.3f}")
    for side, ev, pnl in (("back", ev_b, back_pnl), ("lay", ev_l, lay_pnl)):
        for bar in BARS:
            m = np.isfinite(ev) & (ev > bar)
            f = d[m]
            stake = np.ones(m.sum())
            r = bet_summary(f, pnl[m], stake, days)
            if side == "lay" and m.any():
                r["per_liability"] = round(float(pnl[m].sum() / (f.pl_bsp - 1).sum()), 4)
            if side == "back" and m.any() and scols:
                tw = TO_WIN / np.clip(price[m] - 1, 0.05, None)
                r["to_win_staked"] = round(float(tw.sum()), 0)
                r["to_win_pnl"] = round(float((tw * pnl[m]).sum()), 0)
            res.append({"decided": how, "side": side, "ev >": bar, **r})
            picks[(how, side, bar)] = m
print(pd.DataFrame(res).to_string(index=False))

for how in ("morning", "near the off"):
    for side in ("back", "lay"):
        m = picks[(how, side, 0.05)]
        if m.sum() < 30:
            continue
        f = d[m].assign(pnl=(back_pnl if side == "back" else lay_pnl)[m])
        print(f"\n  {how}, {side} EV > 5%: by month and by win morning price band (per GBP1)")
        print("   " + "  ".join(f"{k} {v.pnl.mean():+.3f} ({len(v)})" for k, v in f.groupby("month")))
        print("   " + "  ".join(f"{k} {v.pnl.mean():+.3f} ({len(v)})" for k, v in f.groupby("win_band", observed=True)))

# 4. the win rule's horses in the place market
print("\n== 4. the win rule's horses (expected CLV >= 3%, GBP100 matched in the morning) in the place market")
ev = rb.expected_clv_walk_forward(d[["race", "race_date", "morningwap", "morning_vol", "predicted_bfsp", "bsp"]].copy())
d["ev"] = np.nan
d.loc[ev.index, "ev"] = ev["ev"].to_numpy()
E = d[d.ev.notna()].copy()
E["sel"] = (E.ev >= 0.03) & (E.morning_vol.fillna(0) >= MIN_VOL)
E["stake"] = TO_WIN / (E.morningwap - 1)
rows = []
for who, f in (("every runner", E), ("the rule's horses", E[E.sel])):
    for band, g in [("all", f)] + [(b, g) for b, g in f.groupby("win_band", observed=True)]:
        if len(g) < 30:
            continue
        rows.append({"who": who, "win band": band, "runners": len(g),
                     "win morning/SP": float(np.exp(np.log(g.morningwap / g.bsp).mean())),
                     "win pre-play/SP": float(np.exp(np.log(g.ppwap / g.bsp).mean())),
                     "place pre-play/SP": float(np.exp(np.log(g.pl_ppwap / g.pl_bsp).mean())),
                     "win CLV at to-win stakes": float((g.stake * (g.morningwap / g.bsp - 1)).sum() / g.stake.sum()),
                     "place pre-play CLV (level)": float((g.pl_ppwap / g.pl_bsp - 1).mean()),
                     "place pp vol median": float(g.pl_vol.median())})
print(pd.DataFrame(rows).round(4).to_string(index=False))
keep = ["race", "race_date", "month", "track", "race_time", "horse_name", "field", "k", "predicted_bfsp", "morningwap",
        "morning_vol", "ppwap", "bsp", "pl_bsp", "pl_ppwap", "pl_vol", "placed", "q_psp", "q_ppp", "q_winsp", "q_wpp",
        "q_morning", "q_model", "q_close", "ev"]
d[keep].to_csv(OUT / "runners.csv.gz", index=False)
log(f"done; {len(d):,} runners written to {OUT / 'runners.csv.gz'}")
