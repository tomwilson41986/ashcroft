"""The top-up order-flow guard (ledger topup-flow-guard-1115; read-only).

The owner, 15 Nov 2026: "Work on all these" (item 3 of the paper list: order-flow imbalance, Cont, Kukanov, Stoikov
2014; flow toxicity, Easley, Lopez de Prado, O'Hara 2012). Do the trader's top-ups bought while the book runs against
the horse lose, so that a guard on order flow would save money?

Every top-up fill (a back after the horse's first) of the live record 10 Oct - 14 Nov, its CLV at the price Betfair
settled the horse at (the order's price scaled as research/queries/done/live_day_1114.py does) against the BSP, and from
the trader's own books (betfair_live/<day>/books_trader.csv.gz, else the recorder's books.csv.gz) the last book at or
before the fill and the last at or before 10 minutes earlier:

  drift10    the log change of the mid (back1, lay1) over the 10 minutes (positive = the price lengthened);
  flow10     the runner's matched money over the 10 minutes, signed by the move of its last traded price
             (the price up, drifting = negative; unchanged = 0);
  imbalance  (lay1_size - back1_size) / (lay1_size + back1_size) at the fill (negative = layers' money dominates).

Four rules, each skipping a top-up when: R1 drift10 > +2%; R2 drift10 > +5%; R3 imbalance < -0.5;
R4 flow10 < 0 and drift10 > 0. A skipped top-up is counted as not placed (no re-entry). Per rule: the skipped fills'
stake, CLV and locked pounds, the kept top-ups' CLV against all top-ups', 90% intervals by bootstrap of racing days,
and the deflated Sharpe of the daily saving (minus the skipped fills' locked pounds) with n_trials = the horse
ledger's trial count plus 4. Pass: the skipped fills' locked pounds below zero with the interval below zero, and DSR
at least 0.95. Nothing here places a bet or changes the trader.
"""
import io
import os
import sys
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from desk.trials import LEDGER, count_trials, parse  # noqa: E402
from research.overfit import dsr_check  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
K = ["market_id", "selection_id"]
DAYS = [str(date(2026, 10, 10) + timedelta(days=i)) for i in range((date(2026, 11, 14) - date(2026, 10, 10)).days + 1)]
BOOK_COLS = {"polled_utc", "market_id", "selection_id", "runner_status", "back1", "back1_size", "lay1", "lay1_size",
             "last_traded", "runner_total_matched"}
NEAR = pd.Timedelta(minutes=5)       # a book older than this at the fill (or at the fill less 10 minutes) is not used
RULES = {
    "R1 drift10 > +2%": lambda f: f.drift10 > np.log(1.02),
    "R2 drift10 > +5%": lambda f: f.drift10 > np.log(1.05),
    "R3 imbalance < -0.5": lambda f: f.imbalance < -0.5,
    "R4 flow10 < 0 and drift10 > 0": lambda f: (f.flow10 < 0) & (f.drift10 > 0),
}


def get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001
        return None


def fills(day):
    raw = get(f"trading/live/{day}/ledger.csv")
    if raw is None:
        return None
    d = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
    for c in ["matched", "avg_price", "bsp", "clv", "selection_id"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["t"] = pd.to_datetime(d["ts"], utc=True, errors="coerce")
    b = d[(d.event == "back") & (d.matched > 0)].sort_values("t").copy()
    if b.empty:
        return None
    b["sxp"] = b.matched * b.avg_price
    hm = b.groupby(K).agg(m=("matched", "sum"), sxp=("sxp", "sum")).reset_index()
    hm["matched_price"] = hm.sxp / hm.m
    st = d[d.event == "settle"].drop_duplicates(K, keep="last")[K + ["result", "bsp", "avg_price", "clv"]] \
        .rename(columns={"avg_price": "settled_price", "clv": "horse_clv"})
    b = b.drop(columns=["result", "bsp", "clv"], errors="ignore").merge(hm[K + ["matched_price"]], on=K) \
        .merge(st, on=K, how="left")
    ok = b.result.isin(["WINNER", "LOSER"]) & (b.bsp > 1) & b.horse_clv.notna()
    b["price_s"] = b.avg_price * b.settled_price / b.matched_price
    b["first"] = ~b.duplicated(K)
    b["clv"] = np.where(ok, b.price_s / b.bsp - 1, np.nan)
    b["locked"] = b.matched * b.clv
    b["day"] = day
    return b[ok].copy()


def books(day):
    for name in ("books_trader", "books"):
        raw = get(f"betfair_live/{day}/{name}.csv.gz")
        if raw is None:
            continue
        x = pd.read_csv(io.BytesIO(raw), compression="gzip", low_memory=False, dtype={"market_id": str},
                        usecols=lambda c: c in BOOK_COLS)
        for c in ("back1", "back1_size", "lay1", "lay1_size", "last_traded", "runner_total_matched", "selection_id"):
            x[c] = pd.to_numeric(x.get(c), errors="coerce")
        x["t"] = pd.to_datetime(x.polled_utc, utc=True, errors="coerce")
        x = x[x.runner_status.fillna("ACTIVE").eq("ACTIVE") & x.t.notna()].dropna(subset=["selection_id"])
        two = (x.back1 > 1) & (x.lay1 > 1)
        x["mid"] = np.where(two, (x.back1 + x.lay1) / 2, x.back1)
        return x, name
    return None, None


def asof(f, bk, at, suffix):
    left = f[K + ["fill_id"]].assign(t=at).sort_values("t")
    right = bk[K + ["t", "mid", "back1_size", "lay1_size", "last_traded", "runner_total_matched"]].sort_values("t")
    right = right.rename(columns={"t": "t_book"})
    m = pd.merge_asof(left, right, left_on="t", right_on="t_book", by=K, direction="backward")
    m.loc[(m.t - m.t_book) > NEAR, ["mid", "back1_size", "lay1_size", "last_traded", "runner_total_matched"]] = np.nan
    return m.drop(columns=K + ["t", "t_book"]).add_suffix(suffix).rename(columns={f"fill_id{suffix}": "fill_id"})


rows, sources = [], {}
for day in DAYS:
    f = fills(day)
    if f is None:
        continue
    f = f.reset_index(drop=True)
    f["fill_id"] = f.index
    f["selection_id"] = f.selection_id.astype("int64")
    bk, src = books(day)
    sources[day] = src or "none"
    if bk is None:
        rows.append(f.assign(drift10=np.nan, flow10=np.nan, imbalance=np.nan))
        continue
    bk["selection_id"] = bk.selection_id.astype("int64")
    now = asof(f, bk, f.t, "_0")
    then = asof(f, bk, f.t - pd.Timedelta(minutes=10), "_10")
    g = f.merge(now, on="fill_id").merge(then, on="fill_id")
    g["drift10"] = np.log(g.mid_0 / g.mid_10)
    move = np.sign(g.last_traded_0 - g.last_traded_10)
    g["flow10"] = (g.runner_total_matched_0 - g.runner_total_matched_10).clip(lower=0) * -move
    g["imbalance"] = (g.lay1_size_0 - g.back1_size_0) / (g.lay1_size_0 + g.back1_size_0)
    rows.append(g)

a = pd.concat(rows, ignore_index=True)
print("books read:", pd.Series(sources).value_counts().to_dict())
print(f"fills settled {len(a):,} on {a.day.nunique()} days, GBP{a.matched.sum():,.2f}; CLV {a.locked.sum() / a.matched.sum():+.2%}")
top = a[~a["first"]].copy()
first = a[a["first"]].copy()
for name, x in (("top-ups", top), ("first backs", first)):
    cov = x.drift10.notna()
    print(f"{name}: {len(x):,} fills, GBP{x.matched.sum():,.2f}, CLV {x.locked.sum() / x.matched.sum():+.2%}; "
          f"with a book now and 10 min before: {cov.sum():,} ({cov.mean():.0%})")

print("\n== CLV by drift10 (the mid's log move over the 10 minutes before the fill), top-ups and first backs")
for name, x in (("top-ups", top), ("first backs", first)):
    y = x[x.drift10.notna()].copy()
    y["band"] = pd.cut(y.drift10, [-np.inf, np.log(0.95), np.log(0.98), -1e-9, 1e-9, np.log(1.02), np.log(1.05), np.inf],
                       labels=["< -5%", "-5..-2%", "-2..0%", "0", "0..+2%", "+2..+5%", "> +5%"])
    t = y.groupby("band", observed=False).agg(fills=("matched", "size"), staked=("matched", "sum"),
                                              locked=("locked", "sum"))
    t["clv"] = t.locked / t.staked
    print(f"-- {name}\n{t.round(3).to_string()}")
y = top[top.imbalance.notna()].copy()
y["band"] = pd.cut(y.imbalance, [-1.01, -0.5, -0.1, 0.1, 0.5, 1.01])
t = y.groupby("band", observed=False).agg(fills=("matched", "size"), staked=("matched", "sum"), locked=("locked", "sum"))
t["clv"] = t.locked / t.staked
print(f"-- top-ups by imbalance at the best\n{t.round(3).to_string()}")

lines, _ = parse(LEDGER.read_text())
n_trials = count_trials(lines, "horse_racing") + 4
days = sorted(top.day.unique())
rng = np.random.default_rng(1115)
B = 4000
idx = rng.integers(0, len(days), size=(B, len(days)))
results, daily = {}, {}
for name, rule in RULES.items():
    skip = rule(top).fillna(False).to_numpy()
    s, k = top[skip], top[~skip]
    by_day = top.assign(skip=skip).groupby("day")
    sav = (-(by_day.apply(lambda g: g.locked[g.skip].sum()))).reindex(days, fill_value=0.0)
    daily[name] = sav
    # bootstrap of racing days: the total saving and the kept top-ups' CLV less all top-ups'
    per = top.assign(skip=skip).groupby("day").apply(
        lambda g: pd.Series({"st_all": g.matched.sum(), "lk_all": g.locked.sum(),
                             "st_keep": g.matched[~g.skip].sum(), "lk_keep": g.locked[~g.skip].sum(),
                             "sav": -g.locked[g.skip].sum()})).reindex(days, fill_value=0.0).to_numpy()
    boot = per[idx].sum(axis=1)
    tot_sav = boot[:, 4]
    dclv = boot[:, 3] / boot[:, 2] - boot[:, 1] / boot[:, 0]
    results[name] = {
        "skipped": len(s), "skipped_stake": s.matched.sum(), "skipped_clv": s.locked.sum() / s.matched.sum() if len(s) else np.nan,
        "skipped_locked": s.locked.sum(), "saving": -s.locked.sum(),
        "saving_lo": np.quantile(tot_sav, 0.05), "saving_hi": np.quantile(tot_sav, 0.95),
        "kept_clv": k.locked.sum() / k.matched.sum(), "all_clv": top.locked.sum() / top.matched.sum(),
        "d_clv_lo": np.quantile(dclv, 0.05), "d_clv_hi": np.quantile(dclv, 0.95)}
sh = [float(np.mean(v) / np.std(v, ddof=1)) if np.std(v, ddof=1) > 0 else np.nan for v in daily.values()]
print(f"\n== the four rules (top-ups only; n_trials for the deflated Sharpe: {n_trials}; {len(days)} racing days)")
for name in RULES:
    r = results[name]
    d = dsr_check(daily[name].to_numpy(), n_trials=n_trials, trial_sharpes=sh)
    passed = r["saving_lo"] > 0 and d["passed"]
    print(f"{name}: skip {r['skipped']:,} fills, GBP{r['skipped_stake']:,.2f}, their CLV {r['skipped_clv']:+.2%}, "
          f"locked GBP{r['skipped_locked']:+,.2f}; saving GBP{r['saving']:+,.2f} (90% {r['saving_lo']:+,.2f} to "
          f"{r['saving_hi']:+,.2f}); kept top-ups' CLV {r['kept_clv']:+.2%} against all {r['all_clv']:+.2%} "
          f"(difference 90% {r['d_clv_lo']:+.2%} to {r['d_clv_hi']:+.2%}); daily Sharpe {d['sharpe']}, DSR {d['dsr']} "
          f"({d['reason'] or 'clears 0.95'}) -> {'PASS' if passed else 'fail'}")
