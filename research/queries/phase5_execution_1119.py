"""Phase 5, step 1 of the owner's model brief (19 Nov 2026: "carry on with Phase 5"): why does live trading realise
about half the research CLV (+3.55% against +6.83%, ledger phase4-segments-1119)? Read-only.

Every live horse the trader backed and Betfair settled (live_orders, mode live, from 30 Sep 2026), joined to Betfair's
price file for its day (betfair_prices, WIN, by race date and selection id): the morning WAP the research takes, the
BSP. Stake-weighted by the money matched, 90% intervals by race day. The CLV is split:

    at the morning WAP     the live trader's own picks priced as the research prices them (WAP / BSP - 1)
    matched                the price the trader matched (its fills' average) against the BSP
    settled                the price Betfair settled at (after a non-runner's reduction) against the BSP
    so: matched - WAP is the price taken; settled - matched is the reductions.

And by the fill: first back against top-ups, the UK hour of the fill, a ladder fill (below the best back) against one
at the best back, the minutes to the off. Stakes to win (the owner's rule) weight the long prices less, as the
research does. Before commission throughout.
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
rng = np.random.default_rng(20261119)


def say(*a):
    print(*a, flush=True)


def ci(num, den, day, draws=2000):
    num, den = np.asarray(num, float), np.asarray(den, float)
    days, inv = np.unique(np.asarray(day), return_inverse=True)
    sn, sd = np.bincount(inv, num, len(days)), np.bincount(inv, den, len(days))
    idx = rng.integers(0, len(days), size=(draws, len(days)))
    bs = sn[idx].sum(1) / np.maximum(sd[idx].sum(1), 1e-12)
    return f"{100 * sn.sum() / sd.sum():+.2f}% ({100 * np.quantile(bs, 0.05):+.2f} to {100 * np.quantile(bs, 0.95):+.2f})"


con = sqlite3.connect("horse_racing.db")
lo = pd.read_sql_query("SELECT * FROM live_orders WHERE mode = 'live'", con)
for c in ("matched", "avg_price", "bsp", "best_back", "minutes_to_off", "selection_id"):
    lo[c] = pd.to_numeric(lo[c], errors="coerce")
backs = lo[(lo["event"] == "back") & (lo["matched"] > 0)].copy()
sett = lo[(lo["event"] == "settle") & (lo["matched"] > 0)].copy()
say(f"live_orders: {len(backs):,} matched backs, {len(sett):,} settled horses, "
    f"{sett['race_date'].min()} to {sett['race_date'].max()}")

bp = pd.read_sql_query("SELECT race_date, selection_id, morningwap, morning_vol, bsp AS bsp_file FROM betfair_prices "
                       "WHERE market_type = 'win' AND race_date >= '2026-09-30'", con)
con.close()
bp = bp.drop_duplicates(["race_date", "selection_id"])
say(f"betfair_prices WIN rows from 30 Sep: {len(bp):,}, to {bp['race_date'].max()}")

# ---- per horse
backs["ts"] = pd.to_datetime(backs["ts"], errors="coerce")
backs = backs.sort_values("ts")
backs["n"] = backs.groupby(["race_date", "market_id", "selection_id"]).cumcount()
backs["cost"] = backs["matched"] * backs["avg_price"]
h = backs.groupby(["race_date", "market_id", "selection_id"]).agg(
    matched=("matched", "sum"), cost=("cost", "sum"), fills=("matched", "size"), venue=("venue", "first")).reset_index()
h["matched_price"] = h["cost"] / h["matched"]
s = sett[["race_date", "market_id", "selection_id", "avg_price", "bsp"]].rename(columns={"avg_price": "settled_price"})
h = h.merge(s, on=["race_date", "market_id", "selection_id"], how="inner")
h = h.merge(bp, on=["race_date", "selection_id"], how="left")
h["bsp"] = h["bsp"].where(h["bsp"] > 1, h["bsp_file"])
ok = (h["bsp"] > 1) & (h["settled_price"] > 1)
h = h[ok].copy()
have_wap = h["morningwap"] > 1
say(f"horses settled with a BSP: {len(h):,}; with a morning WAP in the price files: {int(have_wap.sum()):,} "
    f"(the rest: days not yet loaded or no morning trade)")

w = h[have_wap].copy()
for name, price in (("at the morning WAP (the research's entry)", "morningwap"),
                    ("matched (the trader's fills)", "matched_price"),
                    ("settled (after the non-runners' reductions)", "settled_price")):
    w[f"v_{price}"] = w["matched"] * (w[price] / w["bsp"] - 1.0)
    say(f"   CLV {name}: {ci(w[f'v_{price}'], w['matched'], w['race_date'])}")
say(f"   so: the price taken (matched - WAP) "
    f"{ci(w['v_matched_price'] - w['v_morningwap'], w['matched'], w['race_date'])}; the reductions (settled - matched) "
    f"{ci(w['v_settled_price'] - w['v_matched_price'], w['matched'], w['race_date'])}")
w["vol_band"] = pd.cut(w["morning_vol"], [0, 100, 1000, 5000, 1e12], labels=["<100", "100-1k", "1k-5k", "5k+"])
w["price_band"] = pd.cut(w["morningwap"], [1, 3, 6, 12, 25, 50, 1e9], labels=["<3", "3-6", "6-12", "12-25", "25-50", "50+"])
w["week"] = pd.to_datetime(w["race_date"]).dt.to_period("W").astype(str)
for seg in ("price_band", "vol_band", "week"):
    say(f"\n-- by {seg}: CLV at the WAP | matched | settled (matched GBP, horses)")
    for v, g in w.groupby(seg, observed=True):
        if len(g) < 20:
            continue
        say(f"   {str(v):22s} {ci(g['v_morningwap'], g['matched'], g['race_date']):>26s} | "
            f"{ci(g['v_matched_price'], g['matched'], g['race_date']):>26s} | "
            f"{ci(g['v_settled_price'], g['matched'], g['race_date']):>26s}  GBP{g['matched'].sum():,.0f}, {len(g)}")

# ---- per fill
f = backs.merge(h[["race_date", "market_id", "selection_id", "bsp", "morningwap"]],
                on=["race_date", "market_id", "selection_id"], how="inner")
f = f[f["avg_price"] > 1].copy()
f["v"] = f["matched"] * (f["avg_price"] / f["bsp"] - 1.0)
f["kind"] = np.where(f["n"] == 0, "first back", "top-up")
uk = f["ts"].dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT") if f["ts"].dt.tz is None else f["ts"]
f["uk_hour"] = uk.dt.tz_convert("Europe/London").dt.hour
f["ladder"] = np.where(f["best_back"] > f["avg_price"] + 1e-9, "below the best back", "at the best back")
f["mins_band"] = pd.cut(f["minutes_to_off"], [0, 60, 180, 360, 600, 1e6],
                        labels=["<1h", "1-3h", "3-6h", "6-10h", "10h+"])
say(f"\nfills joined: {len(f):,}; matched GBP{f['matched'].sum():,.0f}; CLV at the fill (before reductions) "
    f"{ci(f['v'], f['matched'], f['race_date'])}")
for seg in ("kind", "uk_hour", "ladder", "mins_band"):
    say(f"\n-- fills by {seg}: CLV at the fill price (matched GBP, fills)")
    for v, g in f.groupby(seg, observed=True):
        if len(g) < 20:
            continue
        say(f"   {str(v):22s} {ci(g['v'], g['matched'], g['race_date']):>26s}  GBP{g['matched'].sum():,.0f}, {len(g)}")
