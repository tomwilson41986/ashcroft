"""Iteration 133's registered reading (ledger iter133-or-clean-at-weights, 21 Nov 2026). Read-only.

Run 44880970254 (research-loop, iteration 130's folds 2024-12-28 to 2026-03-31; the locked holdout not read). The main
(Huber) member: stn (reference), stn_or (the old OR features withheld, model/blocks/or_clean.py added), stn_or_atw
(stn_or with model/blocks/at_weights.py). Paired by runner with scripts/compare_oos_runs.py:

    (1) stn_or against stn        (2) stn_or_atw against stn_or

The reading (the owner's rule of 18 Nov): paired error against the BFSP at least 0.0007 lower with the 90% interval
excluding zero, on the flat and over jumps, Brier skill and concordance not resolved worse. Beside, not gates: each
arm's error in chases against the morning WAP (the chase gap), and the 3% rule's CLV (GBP250 to win at the morning
WAP, the closing model walk-forward, 2% commission on a winning CLV).
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.getcwd())
from model import race_book as rb  # noqa: E402
from model.race_shape import race_code  # noqa: E402
from scripts import clv_betfair as cb  # noqa: E402

RUN, PREFIX = 44880970254, "research-iter133"
ARMS = ["stn", "stn_or", "stn_or_atw"]
UNTIL = "2026-04-01"
COMMISSION = 0.02
pd.set_option("display.width", 250)
rng = np.random.default_rng(20261121)


def say(*a):
    print(*a, flush=True)


def fetch(into):
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY", "tomwilson41986/ashcroft")
    hd = {"Authorization": f"Bearer {token}"} if token else {}
    j = requests.get(f"https://api.github.com/repos/{repo}/actions/runs/{RUN}/artifacts?per_page=100", headers=hd,
                     timeout=60).json()
    art = [a for a in j.get("artifacts", []) if a["name"].startswith(PREFIX)]
    if not art:
        raise SystemExit(f"no data found: no {PREFIX}* artifact in run {RUN}")
    z = requests.get(art[0]["archive_download_url"], headers=hd, timeout=900)
    os.makedirs(into, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
        for a in ARMS:
            hit = [n for n in zf.namelist() if os.path.basename(n) == f"oos_{a}.csv"]
            if not hit:
                raise SystemExit(f"no data found: oos_{a}.csv not in {art[0]['name']}")
            open(f"{into}/oos_{a}.csv", "wb").write(zf.read(hit[0]))
    return art[0]["name"]


def compare(base, variant, title):
    say(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    r = subprocess.run([sys.executable, "scripts/compare_oos_runs.py", "--base", f"out/i133/oos_{base}.csv",
                        "--variant", f"out/i133/oos_{variant}.csv"], capture_output=True, text=True)
    say(r.stdout.strip() or r.stderr[-2000:])


def day_ci(x, day, draws=4000):
    x = np.asarray(x, float)
    days, inv = np.unique(np.asarray(day), return_inverse=True)
    sn, sd = np.bincount(inv, x, len(days)), np.bincount(inv, np.ones_like(x), len(days))
    idx = rng.integers(0, len(days), size=(draws, len(days)))
    bs = sn[idx].sum(1) / np.maximum(sd[idx].sum(1), 1e-12)
    return x.mean(), float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))


def ratio_ci(f, num, den="stake", draws=4000):
    days, inv = np.unique(f["race_date"].to_numpy(), return_inverse=True)
    sn, sd = np.bincount(inv, f[num].to_numpy(float), len(days)), np.bincount(inv, f[den].to_numpy(float), len(days))
    idx = rng.integers(0, len(days), size=(draws, len(days)))
    bs = sn[idx].sum(1) / np.maximum(sd[idx].sum(1), 1e-12)
    return sn.sum() / sd.sum(), float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))


say(f"{fetch('out/i133')} from run {RUN}")
for a in ARMS:
    d = pd.read_csv(f"out/i133/oos_{a}.csv", dtype={"race_time": str})
    say(f"   {a}: {len(d):,} runners, {d['race_date'].min()} to {d['race_date'].max()}, folds {sorted(d['fold_idx'].unique())}")
    assert d["race_date"].max() < UNTIL, a

compare("stn", "stn_or", "(1) stn_or (OR 0 read as unrated) against stn")
compare("stn_or", "stn_or_atw", "(2) stn_or_atw (the rating at the weights) against stn_or")
compare("stn", "stn_or_atw", "BESIDE: stn_or_atw against stn")

say(f"\n{'=' * 100}\nBESIDE: the chase gap (model error less the morning WAP's, against the BSP) and the 3% rule's CLV\n{'=' * 100}")
loaded = {a: cb.load(f"out/i133/oos_{a}.csv", "horse_racing.db", None, until=UNTIL) for a in ARMS}
common = set.intersection(*[set(zip(d["race"], d["horse_name"])) for d in loaded.values()])
for a, d in loaded.items():
    d = d[[k in common for k in zip(d["race"], d["horse_name"])]].copy()
    d["gap"] = np.abs(np.log(d["predicted_bfsp"] / d["bsp"])) - np.abs(np.log(d["morningwap"] / d["bsp"]))
    d["code"] = race_code(d).to_numpy()
    parts = []
    for c in ("chase", "hurdle", "flat", "aw"):
        g = d[d["code"] == c]
        if len(g):
            m, lo, hi = day_ci(g["gap"], g["race_date"])
            parts.append(f"{c} {m:+.4f} ({lo:+.4f} to {hi:+.4f}, {len(g):,})")
    say(f"   {a:11s} gap by code: " + "; ".join(parts))
for a in ARMS:
    d = loaded[a]
    w = rb.whole_races(d, field=d.groupby("race")["field"].first())
    w = w.sort_values(["race_date", "race", "predicted_bfsp"]).reset_index(drop=True)
    s = rb.expected_clv_walk_forward(w)
    b = s[(s["ev"] >= 0.03) & (s["morning_vol"].fillna(0) >= rb.MIN_VOL)].copy()
    m, bsp = b["morningwap"].to_numpy(float), b["bsp"].to_numpy(float)
    b["stake"] = rb.TO_WIN / (m - 1.0)
    clv = m / bsp - 1.0
    b["net_v"] = b["stake"] * np.where(clv > 0, clv * (1 - COMMISSION), clv)
    n, lo, hi = ratio_ci(b, "net_v")
    say(f"   {a:11s} 3% rule: {len(b):,} bets on {b['race_date'].nunique()} days, net CLV {100 * n:+.2f}% "
        f"({100 * lo:+.2f} to {100 * hi:+.2f}), GBP {b['net_v'].sum():+,.0f}")
