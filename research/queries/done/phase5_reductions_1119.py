"""Phase 5, step 2 of the owner's model brief (19 Nov 2026): the non-runners' reductions, measured and put into the
research CLV; top-ups against first backs after the reductions. Read-only.

1. The reduction a race's later withdrawals make, from Betfair's price files (betfair_prices, WIN): R = the summed
   1 / morning WAP of the runners with a morning price and no BSP (withdrawn after the morning trade), capped at 0.9.
   Checked against the live record: for each live horse, the settled price over the matched price beside 1 - R
   (Betfair cuts a matched price for a later non-runner; the test is how close the price-file estimate comes).
2. The research CLV with the reductions (iteration 122's oos_gate_st, the served stack blend, 2025-02-01 to
   2026-03-31; the locked holdout not read): the owner's staking at the 0% and 3% bars, each back's morning WAP
   reduced to 1 + (WAP - 1) x (1 - R) and to WAP x (1 - R) (two readings of the cut, reported side by side), against
   the plain figure.
3. Live: first backs against top-ups at the settled price, each fill's price cut by its horse's settled/matched
   ratio (an approximation: a top-up after a withdrawal is not cut, so the top-ups' figure is a floor).
"""
from __future__ import annotations

import io
import os
import sqlite3
import sys
import zipfile

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.getcwd())
from model import race_book as rb  # noqa: E402
from scripts import clv_betfair as cb  # noqa: E402

RUN, ART, FILE = 43428279939, "research-iter122-stack-served-blend-158", "oos_gate_st.csv"
UNTIL = "2026-04-01"
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
bp = pd.read_sql_query("SELECT race_date, event_id, selection_id, morningwap, morning_vol, bsp, race_results_id "
                       "FROM betfair_prices WHERE market_type = 'win'", con)
say(f"price files, WIN: {len(bp):,} rows, {bp['race_date'].min()} to {bp['race_date'].max()}")
bp["withdrawn"] = (bp["morningwap"] > 1) & ~(bp["bsp"] > 1)
R = (bp.loc[bp["withdrawn"]].assign(q=lambda x: 1.0 / x["morningwap"]).groupby(["race_date", "event_id"])["q"].sum()
     .clip(upper=0.9).rename("R"))
bp = bp.merge(R, left_on=["race_date", "event_id"], right_index=True, how="left")
bp["R"] = bp["R"].fillna(0.0)
races = bp.groupby(["race_date", "event_id"])["R"].first()
say(f"   races {len(races):,}; with a runner withdrawn after the morning trade {int((races > 0).sum()):,} "
    f"({100 * (races > 0).mean():.1f}%); R median where any {races[races > 0].median():.3f}")

# ---- 1. the check against the live record
lo = pd.read_sql_query("SELECT * FROM live_orders WHERE mode = 'live'", con)
con.close()
for c in ("matched", "avg_price", "bsp", "selection_id"):
    lo[c] = pd.to_numeric(lo[c], errors="coerce")
backs = lo[(lo["event"] == "back") & (lo["matched"] > 0)].copy()
backs["cost"] = backs["matched"] * backs["avg_price"]
backs["ts"] = pd.to_datetime(backs["ts"], errors="coerce")
backs = backs.sort_values("ts")
backs["n"] = backs.groupby(["race_date", "market_id", "selection_id"]).cumcount()
h = backs.groupby(["race_date", "market_id", "selection_id"]).agg(matched=("matched", "sum"), cost=("cost", "sum")).reset_index()
h["matched_price"] = h["cost"] / h["matched"]
sett = lo[(lo["event"] == "settle") & (lo["matched"] > 0)][["race_date", "market_id", "selection_id", "avg_price", "bsp"]]
h = h.merge(sett.rename(columns={"avg_price": "settled_price"}), on=["race_date", "market_id", "selection_id"])
h = h.merge(bp[["race_date", "selection_id", "R", "bsp"]].rename(columns={"bsp": "bsp_file"}).drop_duplicates(
    ["race_date", "selection_id"]), on=["race_date", "selection_id"], how="inner")
h["bsp"] = h["bsp"].where(h["bsp"] > 1, h["bsp_file"])
h = h[(h["settled_price"] > 1) & (h["matched_price"] > 1) & (h["bsp"] > 1)].copy()
h["ratio"] = h["settled_price"] / h["matched_price"]
h["ratio_m1"] = (h["settled_price"] - 1) / (h["matched_price"] - 1)
say(f"\n1. live horses with the price file's R: {len(h):,}; with R > 0: {int((h['R'] > 0).sum()):,}")
for lo_r, hi_r in ((0, 1e-9), (1e-9, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.9)):
    g = h[(h["R"] >= lo_r) & (h["R"] < hi_r)] if lo_r else h[h["R"] == 0]
    if len(g) < 10:
        continue
    say(f"   R {lo_r:.2f}-{hi_r:.2f}: {len(g):5d} horses, mean R {g['R'].mean():.3f}; settled/matched median "
        f"{g['ratio'].median():.3f}, (settled-1)/(matched-1) median {g['ratio_m1'].median():.3f}; 1 - R mean {1 - g['R'].mean():.3f}")

# ---- 2. the research CLV with the reductions
token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY", "tomwilson41986/ashcroft")
hd = {"Authorization": f"Bearer {token}"} if token else {}
arts = requests.get(f"https://api.github.com/repos/{repo}/actions/runs/{RUN}/artifacts?per_page=100", headers=hd,
                    timeout=60).json().get("artifacts", [])
art = [a for a in arts if a["name"] == ART][0]
z = requests.get(art["archive_download_url"], headers=hd, timeout=900)
os.makedirs("out", exist_ok=True)
with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
    open("out/oos_gate_st_122.csv", "wb").write(zf.read(next(n for n in zf.namelist() if n.endswith(FILE))))
d = cb.load("out/oos_gate_st_122.csv", "horse_racing.db", None, until=UNTIL)
w = rb.whole_races(d, field=d.groupby("race")["field"].first())
w = w.sort_values(["race_date", "race", "predicted_bfsp"]).reset_index(drop=True)
s = rb.expected_clv_walk_forward(w)
s = s[s["morning_vol"].fillna(0) >= rb.MIN_VOL].copy()
con = sqlite3.connect("horse_racing.db")
rid = pd.read_sql_query("SELECT r.race_date, r.track, r.race_time, r.horse_name, b.event_id FROM betfair_prices b "
                        "JOIN race_results r ON r.id = b.race_results_id WHERE b.market_type = 'win' AND r.race_date < ?",
                        con, params=(UNTIL,))
con.close()
s = s.merge(rid.drop_duplicates(cb.KEY), on=cb.KEY, how="left").merge(
    races.rename("R").reset_index(), on=["race_date", "event_id"], how="left")
s["R"] = s["R"].fillna(0.0)
say(f"\n2. research bets scored: {len(s):,} runners; in a race with a later withdrawal {100 * (s['R'] > 0).mean():.1f}%")
m, bsp = s["morningwap"].to_numpy(float), s["bsp"].to_numpy(float)
s["stake"] = rb.TO_WIN / (m - 1.0)
for bar in (0.0, 0.03):
    b = s[s["ev"] >= bar]
    mm, bb, r = b["morningwap"].to_numpy(float), b["bsp"].to_numpy(float), b["R"].to_numpy(float)
    plain = b["stake"] * (mm / bb - 1)
    cut_m1 = b["stake"] * ((1 + (mm - 1) * (1 - r)) / bb - 1)
    cut_all = b["stake"] * (np.maximum(mm * (1 - r), 1.01) / bb - 1)
    say(f"   bar {bar:.0%}: {len(b):,} bets; CLV plain {ci(plain, b['stake'], b['race_date'])}; "
        f"cut 1+(p-1)(1-R) {ci(cut_m1, b['stake'], b['race_date'])}; cut p(1-R) {ci(cut_all, b['stake'], b['race_date'])}")

# ---- 3. live: first backs and top-ups after the reductions
f = backs.drop(columns=["bsp"], errors="ignore").merge(h[["race_date", "market_id", "selection_id", "bsp", "ratio"]],
                                                       on=["race_date", "market_id", "selection_id"])
f = f[f["avg_price"] > 1].copy()
f["kind"] = np.where(f["n"] == 0, "first back", "top-up")
f["v_fill"] = f["matched"] * (f["avg_price"] / f["bsp"] - 1)
f["v_settled"] = f["matched"] * (f["avg_price"] * f["ratio"] / f["bsp"] - 1)
say("\n3. live fills: CLV at the fill | after the horse's reduction (matched GBP, fills)")
for k, g in f.groupby("kind"):
    say(f"   {k:11s} {ci(g['v_fill'], g['matched'], g['race_date']):>26s} | "
        f"{ci(g['v_settled'], g['matched'], g['race_date']):>26s}  GBP{g['matched'].sum():,.0f}, {len(g):,}")
f["topup_n"] = f["n"].clip(upper=6)
say("   by the fill's number on the horse (0 = the first back):")
for k, g in f.groupby("topup_n"):
    say(f"   {int(k)}{'+' if k == 6 else ' '}         {ci(g['v_settled'], g['matched'], g['race_date']):>26s}  "
        f"GBP{g['matched'].sum():,.0f}, {len(g):,}")
