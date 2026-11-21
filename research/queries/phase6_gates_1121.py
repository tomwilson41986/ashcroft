"""Phase 6 of the owner's model brief: the holdout read's five gates (ledger HOLDOUT LOOK iter132-final, announced
2026-11-19T15:10Z; amendments of 20 and 21 Nov). Read once.

The candidate, fixed by iteration 131's development result: gate_stn (the served gated blend with collateral,
future_form and nbl_rating in all three members). Against gate_st, the blend served since 15 Nov. The locked folds:
28 Mar - 26 Jun and 27 Jun - 25 Sep 2026 (run RUN_A) and 26 Sep - 16 Nov (run RUN_B, the fold no research run had
scored). Each ensemble's two files are joined; nothing else is done to them.

    gate 1  paired error against the BFSP at least 0.0007 below gate_st, the 90% interval below zero
    gate 2  Brier skill and concordance against the market not resolved worse
            (both from scripts/compare_oos_runs.py, on the three folds and on the unseen fold alone)
    gate 3  the owner's staking at the live 3% bar (GBP250 to win at the morning WAP, the closing model fitted on the
            months before, 2% commission on a winning CLV): net CLV positive with its 90% interval above zero and not
            lower than gate_st's
    gate 4  the deflated Sharpe of that rule's daily net CLV at least 0.95, every trial in the ledger counted
    gate 5  the point-in-time tests (run beside this query, in the repository)

Beside, not gates: gate_stn_fig (iteration 131 failed its reading), the members, the Benter test.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.getcwd())
from desk import trials  # noqa: E402
from model import race_book as rb  # noqa: E402
from research import harness, overfit  # noqa: E402
from scripts import clv_betfair as cb  # noqa: E402

RUN_A, RUN_B = 44752561907, int(os.environ.get("PHASE6_RUN_B") or "0") or None
RUN_B = RUN_B or 44875953861
PREFIX = "research-iter132-final"
ARMS = ["gate_st", "gate_stn", "gate_stn_fig", "st", "dml_st", "rx16_st", "stn", "dml_stn", "rx16_stn"]
UNSEEN = "2026-09-26"
KEY = ["race_date", "track", "race_time", "horse_name"]
pd.set_option("display.width", 250)
rng = np.random.default_rng(20261121)


def say(*a):
    print(*a, flush=True)


def fetch(run, into):
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY", "tomwilson41986/ashcroft")
    hd = {"Authorization": f"Bearer {token}"} if token else {}
    j = requests.get(f"https://api.github.com/repos/{repo}/actions/runs/{run}/artifacts?per_page=100", headers=hd,
                     timeout=60).json()
    art = [a for a in j.get("artifacts", []) if a["name"].startswith(PREFIX)]
    if not art:
        raise SystemExit(f"no data found: no {PREFIX}* artifact in run {run}")
    z = requests.get(art[0]["archive_download_url"], headers=hd, timeout=900)
    os.makedirs(into, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
        for a in ARMS:
            hit = [n for n in zf.namelist() if os.path.basename(n) == f"oos_{a}.csv"]
            if not hit:
                raise SystemExit(f"no data found: oos_{a}.csv not in {art[0]['name']}")
            open(f"{into}/oos_{a}.csv", "wb").write(zf.read(hit[0]))
    return art[0]["name"]


os.makedirs("out/p6", exist_ok=True)
na, nb = fetch(RUN_A, "out/p6/a"), fetch(RUN_B, "out/p6/b")
say(f"folds 1-2: run {RUN_A} ({na}); the unseen fold: run {RUN_B} ({nb})")
for a in ARMS:
    x = pd.read_csv(f"out/p6/a/oos_{a}.csv", dtype={"race_time": str})
    y = pd.read_csv(f"out/p6/b/oos_{a}.csv", dtype={"race_time": str})
    y["fold_idx"] = y["fold_idx"] + x["fold_idx"].max() + 1
    assert x["race_date"].max() < UNSEEN <= y["race_date"].min(), (a, x["race_date"].max(), y["race_date"].min())
    both = pd.concat([x, y], ignore_index=True)
    assert not both.duplicated(KEY).any(), a
    both.to_csv(f"out/p6/all_{a}.csv", index=False)
    y.to_csv(f"out/p6/unseen_{a}.csv", index=False)
    if a in ("gate_st", "gate_stn"):
        say(f"   {a}: {len(x):,} runners {x['race_date'].min()} to {x['race_date'].max()}; unseen fold {len(y):,} "
            f"runners {y['race_date'].min()} to {y['race_date'].max()}")


def compare(base, variant, title):
    say(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    r = subprocess.run([sys.executable, "scripts/compare_oos_runs.py", "--base", base, "--variant", variant],
                       capture_output=True, text=True)
    say(r.stdout.strip() or r.stderr[-2000:])


# ---- gates 1 and 2
compare("out/p6/all_gate_st.csv", "out/p6/all_gate_stn.csv", "GATES 1-2, the three locked folds: gate_stn (candidate) against gate_st (served)")
compare("out/p6/unseen_gate_st.csv", "out/p6/unseen_gate_stn.csv", "GATES 1-2, the unseen fold alone (26 Sep - 16 Nov)")


# ---- gates 3 and 4: the owner's staking at the 3% bar, net of 2% commission on a winning CLV
def bets(arm, bar=0.03):
    d = cb.load(f"out/p6/all_{arm}.csv", "horse_racing.db", None)
    w = rb.whole_races(d, field=d.groupby("race")["field"].first())
    w = w.sort_values(["race_date", "race", "predicted_bfsp"]).reset_index(drop=True)
    s = rb.expected_clv_walk_forward(w)
    b = s[(s["ev"] >= bar) & (s["morning_vol"].fillna(0) >= rb.MIN_VOL)].copy()
    m, bsp = b["morningwap"].to_numpy(float), b["bsp"].to_numpy(float)
    b["stake"] = rb.TO_WIN / (m - 1.0)
    clv = m / bsp - 1.0
    b["gross_v"] = b["stake"] * clv
    b["net_v"] = b["stake"] * np.where(clv > 0, clv * (1 - cb.COMMISSION), clv)
    b["held_v"] = np.where(b["won"].astype(bool), b["stake"] * (m - 1.0) * (1 - cb.COMMISSION), -b["stake"])
    return b, s


def ratio_ci(f, num, den="stake", draws=4000):
    days, inv = np.unique(f["race_date"].to_numpy(), return_inverse=True)
    sn, sd = np.bincount(inv, f[num].to_numpy(float), len(days)), np.bincount(inv, f[den].to_numpy(float), len(days))
    idx = rng.integers(0, len(days), size=(draws, len(days)))
    bs = sn[idx].sum(1) / np.maximum(sd[idx].sum(1), 1e-12)
    return sn.sum() / sd.sum(), float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))


def diff_ci(a, b, num, den="stake", draws=4000):
    days = np.unique(np.concatenate([a["race_date"].to_numpy(), b["race_date"].to_numpy()]))
    pos = {d: i for i, d in enumerate(days)}

    def sums(f):
        inv = f["race_date"].map(pos).to_numpy()
        return (np.bincount(inv, f[num].to_numpy(float), len(days)), np.bincount(inv, f[den].to_numpy(float), len(days)))
    an, ad = sums(a)
    bn, bd = sums(b)
    idx = rng.integers(0, len(days), size=(draws, len(days)))
    bs = an[idx].sum(1) / np.maximum(ad[idx].sum(1), 1e-12) - bn[idx].sum(1) / np.maximum(bd[idx].sum(1), 1e-12)
    return an.sum() / ad.sum() - bn.sum() / bd.sum(), float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))


say(f"\n{'=' * 100}\nGATES 3-4: the owner's staking at the 3% bar, net of {100 * cb.COMMISSION:.0f}% commission on a winning CLV\n{'=' * 100}")
ledger_lines, _ = trials.parse(trials.LEDGER.read_text())
n_trials = trials.count_trials(ledger_lines)
say(f"trials counted in the ledger (every sport): {n_trials:,}; horse racing alone: "
    f"{trials.count_trials(ledger_lines, 'horse_racing'):,}")
book = {}
for arm in ("gate_st", "gate_stn", "gate_stn_fig"):
    b, s = bets(arm)
    book[arm] = b
    say(f"\n-- {arm}: scored {s['race'].nunique():,} whole races {s['race_date'].min()} to {s['race_date'].max()} "
        f"(the closing model needs earlier months inside the window, so the first weeks are fitted on, not bet)")
    for label, f in (("the three folds", b), ("the unseen fold", b[b["race_date"] >= UNSEEN])):
        if f.empty:
            say(f"   {label}: no bets")
            continue
        g, glo, ghi = ratio_ci(f, "gross_v")
        n, nlo, nhi = ratio_ci(f, "net_v")
        h, hlo, hhi = ratio_ci(f, "held_v")
        day = f.groupby("race_date").agg(v=("net_v", "sum"), s=("stake", "sum"))
        dsr = overfit.dsr_check((day["v"] / day["s"]).to_numpy(), n_trials=n_trials)
        say(f"   {label}: {len(f):,} bets on {f['race_date'].nunique()} days, turnover GBP{f['stake'].sum():,.0f}; CLV "
            f"gross {100 * g:+.2f}% ({100 * glo:+.2f} to {100 * ghi:+.2f}), NET {100 * n:+.2f}% ({100 * nlo:+.2f} to "
            f"{100 * nhi:+.2f}); held to the result net {100 * h:+.2f}% ({100 * hlo:+.2f} to {100 * hhi:+.2f}); "
            f"deflated Sharpe of the daily net CLV {dsr.get('dsr')} (Sharpe {dsr.get('sharpe')}, SR* {dsr.get('sr_star')}, "
            f"{dsr.get('n_obs')} days, {n_trials:,} trials): {'pass' if dsr.get('passed') else 'FAIL ' + str(dsr.get('reason'))}")
for label, cut in (("the three folds", "0000"), ("the unseen fold", UNSEEN)):
    a, b = book["gate_stn"], book["gate_st"]
    a, b = a[a["race_date"] >= cut], b[b["race_date"] >= cut]
    if len(a) and len(b):
        d, lo, hi = diff_ci(a, b, "net_v")
        say(f"\n   gate_stn less gate_st, net CLV, {label}: {100 * d:+.2f} points ({100 * lo:+.2f} to {100 * hi:+.2f}); "
            f"net CLV GBP {a['net_v'].sum():+,.0f} against {b['net_v'].sum():+,.0f}")

# ---- beside, not gates
compare("out/p6/all_gate_st.csv", "out/p6/all_gate_stn_fig.csv", "BESIDE (not a gate): gate_stn_fig against gate_st, the three folds")
say(f"\n{'=' * 100}\nBESIDE (not gates): the members' error against the BFSP, and the Benter test\n{'=' * 100}")
for a in ARMS:
    d = pd.read_csv(f"out/p6/all_{a}.csv", dtype={"race_time": str})
    ok = (d["bfsp"] > 1) & (d["predicted_bfsp"] > 1)
    e = np.abs(np.log(d.loc[ok, "predicted_bfsp"] / d.loc[ok, "bfsp"]))
    u = d.loc[ok, "race_date"] >= UNSEEN
    say(f"   {a:13s} three folds {e.mean():.4f}   unseen fold {e[u].mean():.4f}   ({int(ok.sum()):,} and {int(u.sum()):,} runners)")
for a in ("gate_st", "gate_stn"):
    d = pd.read_csv(f"out/p6/all_{a}.csv", dtype={"race_time": str})
    bt = harness.benter(d)
    w = "; ".join(f"fold {x['fold']}: model {x['beta_model']:+.3f}, market {x['beta_market']:.3f}" for x in bt["weights"])
    gp = bt.get("gain_per_race")
    say(f"   Benter, {a}: {w}; log-likelihood over the market alone, per 1,000 races "
        + (f"{1000 * gp[0]:+.2f} ({1000 * gp[1]:+.2f} to {1000 * gp[2]:+.2f}), {bt['races']:,} races" if gp else "no data"))
