"""The owner, 20 Nov 2026: "Results seemed to have dropped this week. Is it due to a model change?" Read-only.

The served model changed twice: the market-rating blend priced 14 Nov's card (ledger serve-market-rating-1113) and
the stack blend every card from 15 Nov (serve-stack-1114). The live record is split at those dates and at 9 Nov,
where the turf flat ended and the card shrank (ledger fall-diagnosis-1118):

    A  10 Oct - 8 Nov    the old (step) blend, the autumn card
    B  9 - 13 Nov        the old blend, the winter card
    C  14 Nov            the market-rating blend
    D  15-21 Nov         the stack blend
    E  22 Nov on         gate_stn (serve-stn-1121)

1. By period, from live_orders (mode live): days, horses, money and settled result a day; the stake-weighted CLV at
   Betfair's settled price (after the non-runners' reductions) with a 90% interval by race day; the share of horses
   reduced; D against B and D against A (days resampled within each period).
2. The trader's own forecast against what it got: the expected CLV it wrote at each back (the ledger's edge) against
   the realised CLV of that fill at the horse's settled price, by period. A closing model that no longer fits the
   served prices shows here as a wider gap in D.
3. How the picks changed: horses backed a race, the model's price against the market's at the back (log p_model /
   p_market), by period; and the same split by code (jumps, all-weather, flat turf).
4. The results gap: race_results stops at HRB's pause (ledger hrb-rating-sets-1118), so a horse that has run since
   has its latest run missing from every form feature. Betfair's own card gives each runner's days since its last
   run: the runners on each card since the last result day whose last run is after it, and the money on them.

Descriptive: no rule is fitted or chosen here, and nothing is counted as a trial.
"""
from __future__ import annotations

import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
rng = np.random.default_rng(20261120)
PERIODS = [("A 10 Oct-8 Nov  old blend, autumn card", "2026-10-10", "2026-11-08"),
           ("B 9-13 Nov      old blend, winter card", "2026-11-09", "2026-11-13"),
           ("C 14 Nov        market-rating blend", "2026-11-14", "2026-11-14"),
           ("D 15-21 Nov     stack blend", "2026-11-15", "2026-11-21"),
           ("E 22 Nov on     gate_stn", "2026-11-22", "2026-12-31")]
AW = ("southwell", "wolverhampton", "kempton", "lingfield", "newcastle", "chelmsford", "dundalk")


def say(*a):
    print(*a, flush=True)


def period(day: pd.Series) -> pd.Series:
    out = pd.Series("", index=day.index)
    for name, lo, hi in PERIODS:
        out[(day >= lo) & (day <= hi)] = name
    return out


def boot(num, den, day, draws=4000):
    """The ratio of sums and its draws, race days resampled."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    days, inv = np.unique(np.asarray(day), return_inverse=True)
    sn, sd = np.bincount(inv, num, len(days)), np.bincount(inv, den, len(days))
    idx = rng.integers(0, len(days), size=(draws, len(days)))
    return sn.sum() / max(sd.sum(), 1e-12), sn[idx].sum(1) / np.maximum(sd[idx].sum(1), 1e-12), len(days)


def fmt(est, bs, n):
    if n < 3:
        return f"{100 * est:+.2f}% ({n} day{'s' if n > 1 else ''}: no interval)"
    return f"{100 * est:+.2f}% ({100 * np.quantile(bs, 0.05):+.2f} to {100 * np.quantile(bs, 0.95):+.2f})"


con = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
lo = pd.read_sql_query("SELECT * FROM live_orders WHERE mode = 'live' AND race_date >= '2026-10-10'", con)
for c in ("matched", "avg_price", "bsp", "best_back", "minutes_to_off", "selection_id", "edge", "p_model", "p_market",
          "pnl", "commission"):
    lo[c] = pd.to_numeric(lo[c], errors="coerce")
lo["market_id"] = lo["market_id"].astype(str)
say(f"live_orders (live): {len(lo):,} rows, {lo['race_date'].min()} to {lo['race_date'].max()}")
card = pd.read_sql_query("""SELECT race_date, market_id, selection_id, venue, race_type, days_since_last_run
                            FROM betfair_live_markets WHERE market_type = 'WIN' AND country IN ('GB', 'IE')
                            AND race_date >= '2026-10-10'""", con)
card["market_id"] = card["market_id"].astype(str)
card["selection_id"] = pd.to_numeric(card["selection_id"], errors="coerce")
card["code"] = np.where(card["race_type"].fillna("").str.lower() == "flat", "flat turf", "jumps")
card.loc[(card["code"] == "flat turf") & card["venue"].fillna("").str.lower().str.contains("|".join(AW)), "code"] = "aw"
last_result = pd.read_sql_query("SELECT MAX(race_date) AS d FROM race_results", con)["d"].iloc[0][:10]
con.close()
say(f"race_results ends {last_result}; Betfair's card rows {len(card):,}, to {card['race_date'].max()}")

backs = lo[(lo["event"] == "back") & (lo["matched"] > 0)].copy()
sett = lo[(lo["event"] == "settle") & (lo["matched"] > 0)].copy()
comm = lo[lo["event"] == "commission"].groupby("race_date")["commission"].sum()
K = ["race_date", "market_id", "selection_id"]
backs["ts"] = pd.to_datetime(backs["ts"], errors="coerce")
backs = backs.sort_values("ts")
backs["n"] = backs.groupby(K).cumcount()
backs["cost"] = backs["matched"] * backs["avg_price"]
h = backs.groupby(K).agg(matched=("matched", "sum"), cost=("cost", "sum"), fills=("matched", "size")).reset_index()
h["matched_price"] = h["cost"] / h["matched"]
h = h.merge(sett[K + ["avg_price", "bsp", "pnl"]].rename(columns={"avg_price": "settled_price"}), on=K, how="inner")
h = h.merge(card[["market_id", "code"]].drop_duplicates("market_id"), on="market_id", how="left")
h["code"] = h["code"].fillna("unknown")
h["period"] = period(h["race_date"])
all_h = len(h)
h = h[(h["bsp"] > 1) & (h["settled_price"] > 1) & (h["period"] != "")].copy()
say(f"horses settled {all_h:,}; with a BSP and in a period {len(h):,}")
h["v_set"] = h["matched"] * (h["settled_price"] / h["bsp"] - 1.0)
h["v_mat"] = h["matched"] * (h["matched_price"] / h["bsp"] - 1.0)
h["reduced"] = h["settled_price"] < h["matched_price"] * 0.999

say("\n== 1. the live record by period (CLV at Betfair's settled price, stake-weighted, 90% interval by race day)")
draws = {}
for name, _, _ in PERIODS:
    g = h[h["period"] == name]
    if g.empty:
        say(f"   {name}: no data found")
        continue
    days = g["race_date"].nunique()
    pnl = g["pnl"].sum() - comm.reindex(sorted(g["race_date"].unique())).fillna(0.0).sum()
    est, bs, n = boot(g["v_set"], g["matched"], g["race_date"])
    draws[name] = (est, bs, n)
    em, bm, _ = boot(g["v_mat"], g["matched"], g["race_date"])
    say(f"   {name}: {days} days, {len(g) / days:.0f} horses and GBP{g['matched'].sum() / days:,.0f} a day, "
        f"settled GBP{pnl / days:+,.2f} a day after commission (GBP{pnl:+,.2f}); CLV settled {fmt(est, bs, n)}; "
        f"at the matched price {fmt(em, bm, n)}; reduced {100 * g['reduced'].mean():.1f}% of horses")
names = [p[0] for p in PERIODS]
for a, b in ((names[4], names[3]), (names[4], names[0]), (names[3], names[1]), (names[3], names[0]), (names[1], names[0])):
    if a in draws and b in draws and draws[a][2] >= 3 and draws[b][2] >= 3:
        d = draws[a][1] - draws[b][1]
        say(f"   {a[:1]} less {b[:1]}: {100 * (draws[a][0] - draws[b][0]):+.2f} points "
            f"({100 * np.quantile(d, 0.05):+.2f} to {100 * np.quantile(d, 0.95):+.2f})")
# the Monday table (the weekly edge table, 23 Nov): the last 7 days against the previous 7 and the 30 autumn days
WINDOWS = [("last 7 (17-23 Nov)", "2026-11-17", "2026-11-23"), ("previous 7 (10-16 Nov)", "2026-11-10", "2026-11-16"),
           ("autumn 30 (10 Oct-8 Nov)", "2026-10-10", "2026-11-08")]
say("\n   the weekly table: horses and money a day, settled a day after commission, CLV settled (90% interval)")
wd = {}
for wname, lo, hi in WINDOWS:
    g = h[(h["race_date"] >= lo) & (h["race_date"] <= hi)]
    if g.empty:
        say(f"   {wname}: no data found")
        continue
    days = g["race_date"].nunique()
    pnl = g["pnl"].sum() - sum(float(comm.get(d, 0.0)) for d in g["race_date"].unique())
    est, bs, n = boot(g["v_set"], g["matched"], g["race_date"])
    wd[wname] = (est, bs, n)
    say(f"   {wname}: {days} days, {len(g) / days:.0f} horses and GBP{g['matched'].sum() / days:,.0f} a day, settled "
        f"GBP{pnl / days:+,.2f} a day (GBP{pnl:+,.2f}); CLV settled {fmt(est, bs, n)}")
wn = [w[0] for w in WINDOWS]
for a, b in ((wn[0], wn[1]), (wn[0], wn[2])):
    if a in wd and b in wd:
        d = wd[a][1] - wd[b][1]
        say(f"   {a} less {b}: {100 * (wd[a][0] - wd[b][0]):+.2f} points ({100 * np.quantile(d, 0.05):+.2f} to "
            f"{100 * np.quantile(d, 0.95):+.2f})")
say("\n   by day from 1 Nov: horses, matched, CLV settled, result after commission")
for day, g in h[h["race_date"] >= "2026-11-01"].groupby("race_date"):
    say(f"   {day}  {len(g):4d}  GBP{g['matched'].sum():8,.2f}  {100 * g['v_set'].sum() / g['matched'].sum():+6.2f}%  "
        f"GBP{g['pnl'].sum() - float(comm.get(day, 0.0)):+8.2f}")

say("\n== 2. the trader's expected CLV at each back (the ledger's edge) against the realised, at the horse's settled price")
f = backs.drop(columns=["bsp"], errors="ignore").merge(
    h[K + ["bsp", "settled_price", "matched_price", "period", "code"]], on=K, how="inner")
f = f[(f["avg_price"] > 1) & f["edge"].notna()].copy()
f["real"] = f["avg_price"] * (f["settled_price"] / f["matched_price"]) / f["bsp"] - 1.0
f["kind"] = np.where(f["n"] == 0, "first back", "top-up")
for name, _, _ in PERIODS:
    for kind in ("first back", "top-up"):
        g = f[(f["period"] == name) & (f["kind"] == kind)]
        if len(g) < 20:
            continue
        ee, be, n = boot(g["matched"] * g["edge"], g["matched"], g["race_date"])
        er, br, _ = boot(g["matched"] * g["real"], g["matched"], g["race_date"])
        gap = br - be
        tail = "" if n < 3 else f" ({100 * np.quantile(gap, 0.05):+.2f} to {100 * np.quantile(gap, 0.95):+.2f})"
        say(f"   {name} | {kind:10s}: expected {100 * ee:+.2f}%, realised {fmt(er, br, n)}, realised less expected "
            f"{100 * (er - ee):+.2f} points{tail}; GBP{g['matched'].sum():,.0f}, {len(g):,} fills")

say("\n== 3. the picks: horses backed a race, the model's price against the market's at the first back")
first = f[f["kind"] == "first back"].copy()
with np.errstate(divide="ignore", invalid="ignore"):
    first["gap"] = np.log(first["p_model"] / first["p_market"])
for name, _, _ in PERIODS:
    g = first[first["period"] == name]
    if g.empty:
        continue
    per_race = g.groupby(["race_date", "market_id"]).size()
    say(f"   {name}: {per_race.mean():.2f} horses a race traded ({len(per_race)} races); log(p_model / p_market) at "
        f"the first back median {g['gap'].median():+.3f}, mean {g['gap'].mean():+.3f}; first price median "
        f"{g['avg_price'].median():.2f}; expected CLV mean {100 * g['edge'].mean():+.2f}%")
say("\n   CLV settled by code and period (stake-weighted; matched GBP, horses)")
for (name, code), g in h.groupby(["period", "code"]):
    if len(g) < 20:
        continue
    est, bs, n = boot(g["v_set"], g["matched"], g["race_date"])
    say(f"   {name} | {code:9s}: {fmt(est, bs, n):>32s}  GBP{g['matched'].sum():,.0f}, {len(g)}; reduced "
        f"{100 * g['reduced'].mean():.1f}%")

say(f"\n== 4. the results gap: race_results ends {last_result}; card runners whose last run (Betfair's days since the "
    f"last run) is after it")
card["dslr"] = pd.to_numeric(card["days_since_last_run"], errors="coerce")
gap_days = (pd.to_datetime(card["race_date"]) - pd.Timestamp(last_result)).dt.days
card["missing_run"] = card["dslr"].notna() & (card["dslr"] < gap_days) & (gap_days > 0)
late = card[card["race_date"] > last_result]
if late.empty:
    say("   no card after the last result day")
for day, g in late.groupby("race_date"):
    hm = h[(h["race_date"] == day)].merge(g[["market_id", "selection_id", "missing_run"]], on=["market_id", "selection_id"],
                                           how="left")
    mm = hm[hm["missing_run"].fillna(False).astype(bool)]
    tail = ""
    if len(hm):
        clv = f"{100 * mm['v_set'].sum() / mm['matched'].sum():+.2f}%" if len(mm) and mm["matched"].sum() > 0 else "no data"
        tail = (f"; the trader backed {len(mm)} of them (GBP{mm['matched'].sum():,.2f} of GBP{hm['matched'].sum():,.2f}, "
                f"CLV settled {clv})")
    say(f"   {day}: {int(g['missing_run'].sum())} of {len(g)} runners ({100 * g['missing_run'].mean():.1f}%) have run "
        f"since {last_result}; days since the last run known for {100 * g['dslr'].notna().mean():.0f}%{tail}")
norm = card[(card["race_date"] <= last_result) & card["dslr"].notna()]
say("   for scale, 10 Oct to the last result day: card runners whose last run was within k days: "
    + ", ".join(f"{k}d {100 * (norm['dslr'] <= k).mean():.1f}%" for k in (3, 5, 7, 9, 14)))
