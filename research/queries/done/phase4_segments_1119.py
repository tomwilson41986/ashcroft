"""Phase 4, step 1 of the owner's model brief (19 Nov 2026: "carry on with Phase 4"): where the edge is, by segment.
Read-only; descriptive (rules are chosen in step 2, on earlier months, and scored on later ones).

A. Research: the served stack blend's out-of-sample prices (iteration 122, run 43428279939, artifact
   research-iter122-stack-served-blend-158, oos_gate_st.csv; 2024-12-28 to 2026-03-31; the locked holdout not read)
   joined to Betfair's morning prices (scripts/clv_betfair.load). Each runner's expected CLV from the closing model
   fitted on the months before its own (model/race_book.expected_clv_walk_forward); the owner's staking at the 0% bar
   (back to win GBP250 at the morning WAP, at least GBP100 matched in the morning). Per bet: the stake, the CLV closed
   at the BSP (morning / BSP - 1, before commission) and the result held (before commission).
   By segment: code, course, class, field size, morning price band, distance, the off's hour, the race's order in
   the day, the expected-CLV band, month. Each: bets, turnover, the stake-weighted CLV with a 90% race-day interval,
   the held return, and the CLV shrunk to the overall by empirical Bayes (each family of segments its own spread).
B. Live: the trader's own settled horses since 1 Oct (live_orders, settle rows, mode live): the stake-weighted CLV
   at the settled price by course, price band and weekday, with 90% day intervals.
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
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
rng = np.random.default_rng(20261119)


def say(*a):
    print(*a, flush=True)


def fetch(path):
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY", "tomwilson41986/ashcroft")
    hd = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(f"https://api.github.com/repos/{repo}/actions/runs/{RUN}/artifacts?per_page=100", headers=hd,
                     timeout=60)
    art = [a for a in r.json().get("artifacts", []) if a["name"] == ART]
    if not art:
        raise SystemExit(f"artifact {ART} not found ({r.status_code}): no data found")
    z = requests.get(art[0]["archive_download_url"], headers=hd, timeout=900)
    with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(FILE))
        open(path, "wb").write(zf.read(name))


def ratio_ci(num, den, day, level=0.90, draws=2000):
    num, den = np.asarray(num, float), np.asarray(den, float)
    days, inv = np.unique(np.asarray(day), return_inverse=True)
    sn, sd = np.bincount(inv, num, len(days)), np.bincount(inv, den, len(days))
    idx = rng.integers(0, len(days), size=(draws, len(days)))
    bs = sn[idx].sum(1) / np.maximum(sd[idx].sum(1), 1e-12)
    a = (1 - level) / 2
    return sn.sum() / sd.sum(), float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a)), float(bs.std())


def table(b: pd.DataFrame, seg: str, num="clv_v", den="stake", held="res_v", min_bets=30) -> pd.DataFrame:
    rows = []
    for v, g in b.groupby(seg, observed=True):
        if len(g) < min_bets:
            continue
        est, lo, hi, se = ratio_ci(g[num], g[den], g["race_date"])
        rows.append({seg: v, "bets": len(g), "turnover": g[den].sum(), "clv": est, "lo": lo, "hi": hi, "se": se,
                     "held": g[held].sum() / g[den].sum() if held else np.nan})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    overall = b[num].sum() / b[den].sum()
    tau2 = max(0.0, float(np.var(t["clv"], ddof=1)) - float(np.mean(t["se"] ** 2))) if len(t) > 1 else 0.0
    w = tau2 / (tau2 + t["se"] ** 2) if tau2 > 0 else 0.0
    t["clv_shrunk"] = overall + w * (t["clv"] - overall)
    t["above_zero"] = t["lo"] > 0
    return t.sort_values("turnover", ascending=False)


def show(t, seg, n=25):
    if t.empty:
        say(f"\n-- by {seg}: no segment with 30 bets")
        return
    say(f"\n-- by {seg} ({len(t)} segments with 30+ bets; the {min(n, len(t))} largest by turnover)")
    out = t.head(n).copy()
    for c in ("clv", "lo", "hi", "held", "clv_shrunk"):
        out[c] = (100 * out[c]).round(2)
    say(out.drop(columns="se").round({"turnover": 0}).to_string(index=False))


# ---- A. research
os.makedirs("out", exist_ok=True)
pred = "out/oos_gate_st_122.csv"
fetch(pred)
d = cb.load(pred, "horse_racing.db", None, until=UNTIL)
say(f"A. {pred}: {d['race'].nunique():,} races, {len(d):,} runners joined to Betfair's morning prices "
    f"({d['race_date'].min()} to {d['race_date'].max()})")
w = rb.whole_races(d, field=d.groupby("race")["field"].first())
w = w.sort_values(["race_date", "race", "predicted_bfsp"]).reset_index(drop=True)
scored = rb.expected_clv_walk_forward(w)
say(f"   scored (closing model fitted on earlier months): {scored['race'].nunique():,} whole races, "
    f"{scored['race_date'].min()} to {scored['race_date'].max()}")
b = scored[(scored["ev"] >= 0.0) & (scored["morning_vol"].fillna(0) >= rb.MIN_VOL)].copy()
m, bsp = b["morningwap"].to_numpy(float), b["bsp"].to_numpy(float)
b["stake"] = rb.TO_WIN / (m - 1.0)
b["clv_v"] = b["stake"] * (m / bsp - 1.0)
b["res_v"] = np.where(b["won"].astype(bool), b["stake"] * (m - 1.0), -b["stake"])
est, lo, hi, _ = ratio_ci(b["clv_v"], b["stake"], b["race_date"])
say(f"   the owner's staking at the 0% bar: {len(b):,} bets, turnover GBP{b['stake'].sum():,.0f}, CLV {100 * est:+.2f}% "
    f"(90% {100 * lo:+.2f} to {100 * hi:+.2f}), held {100 * b['res_v'].sum() / b['stake'].sum():+.2f}% "
    f"(both before commission)")

jumps = b["race_code"].astype(str).str.contains("Hurdle|Chase|NH|National Hunt", case=False)
aw = b["surface_type"].astype(str).str.contains("Polytrack|Tapeta|Fibresand|All", case=False)
b["code"] = np.where(jumps, "jumps", np.where(aw, "aw", "flat turf"))
b["course"] = b["track"]
b["class"] = b.get("race_class", pd.Series("?", index=b.index)).astype(str)
b["field_size"] = pd.cut(b["number_of_runners"], [0, 7, 11, 15, 99], labels=["2-7", "8-11", "12-15", "16+"])
b["price_band"] = pd.cut(b["morningwap"], [1, 3, 6, 12, 25, 50, 1e9], labels=["<3", "3-6", "6-12", "12-25", "25-50", "50+"])
b["distance"] = pd.cut(pd.to_numeric(b.get("dist_furlongs"), errors="coerce"), [0, 6.5, 9, 13, 20, 99],
                       labels=["5-6f", "7-8f", "9-12f", "13-20f", "20f+"])
b["off_hour"] = (b["mins"] // 60).astype("Int64")
b["day_order"] = pd.cut(b["day_order"], [0, 5, 15, 30, 99], labels=["races 1-5", "6-15", "16-30", "31+"])
b["ev_band"] = pd.cut(b["ev"], [0, 0.03, 0.05, 0.10, 0.20, 9], labels=["0-3%", "3-5%", "5-10%", "10-20%", "20%+"],
                      include_lowest=True)
b["month"] = b["race_date"].astype(str).str[:7]
for seg in ("code", "price_band", "field_size", "class", "distance", "off_hour", "day_order", "ev_band", "month",
            "course"):
    show(table(b, seg), seg, n=40 if seg == "course" else 25)

# ---- B. live
say("\n\nB. live: the trader's settled horses (live_orders, settle rows, mode live)")
con = sqlite3.connect("horse_racing.db")
cols = [r[1] for r in con.execute("PRAGMA table_info(live_orders)")]
need = {"event", "mode", "matched", "avg_price", "bsp", "venue", "race_date"}
if not need <= set(cols):
    say(f"   no data found: live_orders lacks {sorted(need - set(cols))}")
else:
    lv = pd.read_sql_query("SELECT * FROM live_orders WHERE event = 'settle' AND mode = 'live'", con)
    lv["matched"] = pd.to_numeric(lv["matched"], errors="coerce")
    lv["avg_price"] = pd.to_numeric(lv["avg_price"], errors="coerce")
    lv["bsp"] = pd.to_numeric(lv["bsp"], errors="coerce")
    lv = lv[(lv["matched"] > 0) & (lv["bsp"] > 1) & (lv["avg_price"] > 1)].copy()
    lv["clv_v"] = lv["matched"] * (lv["avg_price"] / lv["bsp"] - 1.0)
    est, lo, hi, _ = ratio_ci(lv["clv_v"], lv["matched"], lv["race_date"])
    say(f"   {len(lv):,} settled horses on {lv['race_date'].nunique()} days ({lv['race_date'].min()} to "
        f"{lv['race_date'].max()}), matched GBP{lv['matched'].sum():,.0f}; CLV at the settled price "
        f"{100 * est:+.2f}% (90% {100 * lo:+.2f} to {100 * hi:+.2f})")
    lv["price_band"] = pd.cut(lv["avg_price"], [1, 3, 6, 12, 25, 50, 1e9],
                              labels=["<3", "3-6", "6-12", "12-25", "25-50", "50+"])
    lv["weekday"] = pd.to_datetime(lv["race_date"]).dt.day_name()
    for seg in ("price_band", "weekday", "venue"):
        show(table(lv, seg, num="clv_v", den="matched", held=None, min_bets=20), seg, n=30)
con.close()
