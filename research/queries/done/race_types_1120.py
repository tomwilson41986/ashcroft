"""The owner, 20 Nov 2026: "Are there any types of races we should drop?" Read-only; descriptive.

A. Research: the served stack blend's out-of-sample prices (iteration 130, run 44473514515, oos_gate_st.csv;
   2024-12-28 to 2026-03-31; the locked holdout not read) joined to Betfair's morning prices, each runner's expected
   CLV from the closing model fitted on the months before its own, the owner's staking (GBP250 to win at the morning
   WAP, at least GBP100 matched in the morning) at the 3% bar the live rule uses, the 0% bar beside it. By race
   type: the code (flat, all-weather, hurdle, chase, bumper), the kind (handicap, novice/maiden/beginners, other),
   the two together, and HRB's own race type. Each: bets, turnover, the stake-weighted CLV with a 90% race-day
   interval, the CLV shrunk toward the overall (empirical Bayes), and the return held to the result.
B. Live: the trader's settled horses (live_orders, mode live, 10 Oct on) by Betfair's race type and market name:
   CLV at the settled price and the settled result after commission.

The reading, fixed before the run: a type is a candidate to drop only if its research CLV at the 3% bar has its
90% interval below zero, or its live CLV has its interval below zero and research does not have its interval
above zero. Anything else is noise at these sample sizes (about 20 segments are read; one in ten intervals misses
zero by chance). A candidate is put to the owner; nothing is dropped here. Not a trial: no rule is fitted.
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
from model.race_shape import race_code  # noqa: E402
from scripts import clv_betfair as cb  # noqa: E402

RUN, PREFIX, FILE = 44473514515, "research-iter130", "oos_gate_st.csv"
UNTIL = "2026-04-01"
AW = ("southwell", "wolverhampton", "kempton", "lingfield", "newcastle", "chelmsford", "dundalk")
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
rng = np.random.default_rng(20261120)


def say(*a):
    print(*a, flush=True)


def fetch(path):
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY", "tomwilson41986/ashcroft")
    hd = {"Authorization": f"Bearer {token}"} if token else {}
    arts, page = [], 1
    while True:
        j = requests.get(f"https://api.github.com/repos/{repo}/actions/runs/{RUN}/artifacts?per_page=100&page={page}",
                         headers=hd, timeout=60).json()
        arts += j.get("artifacts", [])
        if len(j.get("artifacts", [])) < 100:
            break
        page += 1
    for a in [a for a in arts if a["name"].startswith(PREFIX)]:
        z = requests.get(a["archive_download_url"], headers=hd, timeout=900)
        with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
            hit = [n for n in zf.namelist() if n.endswith(FILE)]
            if hit:
                open(path, "wb").write(zf.read(hit[0]))
                return a["name"]
    raise SystemExit(f"no data found: {FILE} not in run {RUN}'s {PREFIX}* artifacts")


def ratio_ci(num, den, day, draws=2000):
    num, den = np.asarray(num, float), np.asarray(den, float)
    days, inv = np.unique(np.asarray(day), return_inverse=True)
    sn, sd = np.bincount(inv, num, len(days)), np.bincount(inv, den, len(days))
    idx = rng.integers(0, len(days), size=(draws, len(days)))
    bs = sn[idx].sum(1) / np.maximum(sd[idx].sum(1), 1e-12)
    return sn.sum() / sd.sum(), float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95)), float(bs.std())


def table(b, seg, num, den, extra=None, min_bets=30):
    rows = []
    for v, g in b.groupby(seg, observed=True):
        if len(g) < min_bets:
            continue
        est, lo, hi, se = ratio_ci(g[num], g[den], g["race_date"])
        row = {seg: v, "bets": len(g), "days": g["race_date"].nunique(), "turnover": round(g[den].sum()),
               "clv%": est, "lo%": lo, "hi%": hi, "se": se}
        for name, col in (extra or {}).items():
            row[name] = g[col].sum() / g[den].sum() if name.endswith("%") else g[col].sum()
        rows.append(row)
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    overall = b[num].sum() / b[den].sum()
    tau2 = max(0.0, float(np.var(t["clv%"], ddof=1)) - float(np.mean(t["se"] ** 2))) if len(t) > 1 else 0.0
    w = tau2 / (tau2 + t["se"] ** 2) if tau2 > 0 else 0.0
    t["shrunk%"] = overall + w * (t["clv%"] - overall)
    t["reading"] = np.where(t["hi%"] < 0, "BELOW ZERO", np.where(t["lo%"] > 0, "above zero", "not resolved"))
    for c in [c for c in t.columns if c.endswith("%")]:
        t[c] = (100 * t[c]).round(2)
    return t.drop(columns="se").sort_values("turnover", ascending=False)


def show(t, title, n=30):
    say(f"\n-- {title}")
    say("   no segment with enough bets" if t.empty else t.head(n).to_string(index=False))


def kind_of(text: pd.Series) -> np.ndarray:
    s = text.fillna("").astype(str).str.lower()
    hcap = s.str.contains(r"handicap|hcap|h'cap|nursery")
    green = s.str.contains(r"novice|nov\b|maiden|mdn|beginner|beg\b|juvenile|juv\b|nh flat|nhf|bumper|inhf")
    return np.where(hcap, "handicap", np.where(green, "novice/maiden/bumper", "other"))


# ---- A. research
os.makedirs("out", exist_ok=True)
pred = "out/oos_gate_st_130.csv"
art = fetch(pred)
d = cb.load(pred, "horse_racing.db", None, until=UNTIL)
say(f"A. {art} {FILE}: {d['race'].nunique():,} races, {len(d):,} runners joined to Betfair's morning prices "
    f"({d['race_date'].min()} to {d['race_date'].max()})")
w = rb.whole_races(d, field=d.groupby("race")["field"].first())
w = w.sort_values(["race_date", "race", "predicted_bfsp"]).reset_index(drop=True)
scored = rb.expected_clv_walk_forward(w)
scored["code"] = race_code(scored).to_numpy()
scored["kind"] = kind_of(scored["race_type"])
scored["code x kind"] = scored["code"] + " | " + scored["kind"]
scored["race_type"] = scored["race_type"].astype(str)
for bar in (0.03, 0.0):
    b = scored[(scored["ev"] >= bar) & (scored["morning_vol"].fillna(0) >= rb.MIN_VOL)].copy()
    m, bsp = b["morningwap"].to_numpy(float), b["bsp"].to_numpy(float)
    b["stake"] = rb.TO_WIN / (m - 1.0)
    b["clv_v"] = b["stake"] * (m / bsp - 1.0)
    b["res_v"] = np.where(b["won"].astype(bool), b["stake"] * (m - 1.0), -b["stake"])
    est, lo, hi, _ = ratio_ci(b["clv_v"], b["stake"], b["race_date"])
    say(f"\n=== research, expected CLV >= {100 * bar:.0f}%: {len(b):,} bets in {b['race'].nunique():,} races, turnover "
        f"GBP{b['stake'].sum():,.0f}, CLV {100 * est:+.2f}% ({100 * lo:+.2f} to {100 * hi:+.2f}), held "
        f"{100 * b['res_v'].sum() / b['stake'].sum():+.2f}% (before commission)")
    for seg in ("code", "kind", "code x kind", "race_type"):
        show(table(b, seg, "clv_v", "stake", {"held%": "res_v"}), f"research at the {100 * bar:.0f}% bar by {seg}")

# ---- B. live
say("\n\n=== B. live: the trader's settled horses (live_orders, mode live, 10 Oct on)")
con = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
lo_ = pd.read_sql_query("SELECT * FROM live_orders WHERE mode = 'live' AND race_date >= '2026-10-10' "
                        "AND event IN ('settle', 'commission')", con)
card = pd.read_sql_query("""SELECT DISTINCT market_id, venue, country, race_type, market_name FROM betfair_live_markets
                            WHERE market_type = 'WIN' AND race_date >= '2026-10-10'""", con)
con.close()
for c in ("matched", "avg_price", "bsp", "pnl", "commission"):
    lo_[c] = pd.to_numeric(lo_[c], errors="coerce")
lo_["market_id"] = lo_["market_id"].astype(str)
card["market_id"] = card["market_id"].astype(str)
card = card.drop_duplicates("market_id")
rt = card["race_type"].fillna("").str.lower()
code = np.where(rt.str.contains("chase"), "chase", np.where(rt.str.contains("hurdle"), "hurdle",
                np.where(rt.str.contains("flat") & rt.str.contains("nh|n\\.h"), "nhflat",
                         np.where(rt == "flat", "flat", "other: " + rt))))
aw = card["venue"].fillna("").str.lower().str.contains("|".join(AW))
card["code"] = np.where((code == "flat") & aw, "aw", code)
card["kind"] = kind_of(card["market_name"])
card["code x kind"] = card["code"] + " | " + card["kind"]
say(f"   Betfair's race types on the card: {card['race_type'].value_counts(dropna=False).to_dict()}")
lv = lo_[(lo_["event"] == "settle") & (lo_["matched"] > 0) & (lo_["bsp"] > 1) & (lo_["avg_price"] > 1)].copy()
lv = lv.merge(card[["market_id", "code", "kind", "code x kind", "country"]], on="market_id", how="left")
lv[["code", "kind", "code x kind", "country"]] = lv[["code", "kind", "code x kind", "country"]].fillna("unknown")
comm = lo_[lo_["event"] == "commission"].groupby("market_id")["commission"].sum()
first = ~lv.duplicated("market_id")
lv["net"] = lv["pnl"].fillna(0.0) - np.where(first, lv["market_id"].map(comm).fillna(0.0), 0.0)
lv["clv_v"] = lv["matched"] * (lv["avg_price"] / lv["bsp"] - 1.0)
est, lo, hi, _ = ratio_ci(lv["clv_v"], lv["matched"], lv["race_date"])
say(f"   {len(lv):,} settled horses on {lv['race_date'].nunique()} days ({lv['race_date'].min()} to "
    f"{lv['race_date'].max()}), matched GBP{lv['matched'].sum():,.0f}; CLV at the settled price {100 * est:+.2f}% "
    f"({100 * lo:+.2f} to {100 * hi:+.2f}); settled after commission GBP{lv['net'].sum():+,.2f}")
lv["price_band"] = pd.cut(lv["avg_price"], [1, 3, 6, 12, 25, 50, 1e9], labels=["<3", "3-6", "6-12", "12-25", "25-50", "50+"])
for seg in ("code", "kind", "code x kind", "country", "price_band"):
    show(table(lv, seg, "clv_v", "matched", {"settled GBP": "net"}, min_bets=20), f"live by {seg}")
