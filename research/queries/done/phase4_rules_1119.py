"""Phase 4, step 2 of the owner's model brief (19 Nov 2026): selection rules chosen on earlier months, scored on the
next. Read-only; nothing here changes the live rule (the owner's decision).

The bets are the served stack blend's (iteration 122, oos_gate_st.csv, 2024-12-28 to 2026-03-31; the locked
holdout not read) at Betfair's morning WAP, each runner's expected CLV from the closing model fitted on earlier
months (model/race_book.expected_clv_walk_forward), backed to win a fixed amount (GBP250 at the morning price), at
least GBP100 matched in the morning. Each bet's CLV closed at the BSP less 2% commission on a positive one (the live
rate, trading/config_live.json; per bet, an approximation to Betfair's per-market charge).

Rules (40, each a trial): expected-CLV bar {0, 2, 3, 5, 8%} x minimum morning price {1, 3} x maximum {50, 200} x
minimum field {2, 6}. Walk-forward: from the fourth scored month, each month takes the rule with the most net CLV (GBP)
over all earlier months. Compared, day by day, with the live rule's settings (bar 3%, prices to 200, any field) on
the same days: the difference in net CLV GBP per day, 90% day-bootstrap interval. And across the 40 rules: the
deflated Sharpe of the walk-forward choice's daily returns with 40 trials (research/overfit.dsr_check) and the
probability of backtest overfitting (pbo_cscv, 16 blocks).

Pass (registered in the ledger before the run): the walk-forward choice beats the live rule's settings with the 90%
interval above zero AND its deflated Sharpe is at least 0.95 with every trial counted. Then a proposal to the owner.
"""
from __future__ import annotations

import io
import itertools
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.getcwd())
from model import race_book as rb  # noqa: E402
from research.overfit import dsr_check, pbo_cscv  # noqa: E402
from scripts import clv_betfair as cb  # noqa: E402

RUN, ART, FILE = 43428279939, "research-iter122-stack-served-blend-158", "oos_gate_st.csv"
UNTIL, COMMISSION = "2026-04-01", 0.02
BARS, MINP, MAXP, MINF = (0.0, 0.02, 0.03, 0.05, 0.08), (1.0, 3.0), (50.0, 200.0), (2, 6)
LIVE = (0.03, 1.0, 200.0, 2)
WARMUP = 3
rng = np.random.default_rng(20261119)


def say(*a):
    print(*a, flush=True)


def fetch(path):
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY", "tomwilson41986/ashcroft")
    hd = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(f"https://api.github.com/repos/{repo}/actions/runs/{RUN}/artifacts?per_page=100", headers=hd,
                     timeout=60)
    art = [a for a in r.json().get("artifacts", []) if a["name"] == ART]
    z = requests.get(art[0]["archive_download_url"], headers=hd, timeout=900)
    with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
        open(path, "wb").write(zf.read(next(n for n in zf.namelist() if n.endswith(FILE))))


os.makedirs("out", exist_ok=True)
pred = "out/oos_gate_st_122.csv"
fetch(pred)
d = cb.load(pred, "horse_racing.db", None, until=UNTIL)
w = rb.whole_races(d, field=d.groupby("race")["field"].first())
w = w.sort_values(["race_date", "race", "predicted_bfsp"]).reset_index(drop=True)
s = rb.expected_clv_walk_forward(w)
s = s[s["morning_vol"].fillna(0) >= rb.MIN_VOL].copy()
m, bsp = s["morningwap"].to_numpy(float), s["bsp"].to_numpy(float)
s["stake"] = rb.TO_WIN / (m - 1.0)
clv = s["stake"] * (m / bsp - 1.0)
s["net"] = clv - COMMISSION * np.clip(clv, 0, None)
s["fs"] = pd.to_numeric(s["number_of_runners"], errors="coerce").fillna(s["field"])
s["month"] = s["race_date"].astype(str).str[:7]
say(f"runners scored: {len(s):,} in {s['race'].nunique():,} races, {s['race_date'].min()} to {s['race_date'].max()}")

rules = list(itertools.product(BARS, MINP, MAXP, MINF))
assert LIVE in rules
days = sorted(s["race_date"].astype(str).unique())
months = sorted(s["month"].unique())
daily = {}
for r in rules:
    bar, lo, hi, mf = r
    sel = (s["ev"] >= bar) & (s["morningwap"] >= lo) & (s["morningwap"] <= hi) & (s["fs"] >= mf)
    daily[r] = s.loc[sel].groupby(s["race_date"].astype(str))["net"].sum().reindex(days, fill_value=0.0)
D = pd.DataFrame(daily)                                    # days x rules, net CLV GBP
monthly = D.groupby([d[:7] for d in D.index]).sum()

chosen, wf = {}, []
for i, mo in enumerate(months):
    if i < WARMUP:
        continue
    past = monthly.loc[months[:i]].sum()
    best = past.idxmax()
    chosen[mo] = best
    wf.append(D.loc[[d for d in days if d[:7] == mo], best].rename("wf"))
wf = pd.concat(wf)
live = D.loc[wf.index, LIVE]
diff = (wf - live).to_numpy()
idx = rng.integers(0, len(diff), size=(5000, len(diff)))
bs = diff[idx].mean(1)
lo90, hi90 = np.quantile(bs, [0.05, 0.95])
say(f"\nwalk-forward from {months[WARMUP]} ({len(wf)} days): the rule chosen each month on the months before it")
for mo, r in chosen.items():
    say(f"   {mo}: bar {r[0]:.0%}, price {r[1]:g}-{r[2]:g}, field >= {r[3]}")
say(f"   net CLV GBP: walk-forward {wf.sum():+,.0f}, the live rule's settings {live.sum():+,.0f}; "
    f"per day {diff.mean():+.2f} (90% {lo90:+.2f} to {hi90:+.2f})")

# overfitting: the walk-forward choice's daily returns against the 40 trials
stake_d = {}
for r in rules:
    bar, lo, hi, mf = r
    sel = (s["ev"] >= bar) & (s["morningwap"] >= lo) & (s["morningwap"] <= hi) & (s["fs"] >= mf)
    stake_d[r] = s.loc[sel].groupby(s["race_date"].astype(str))["stake"].sum().reindex(days, fill_value=0.0)
R = D / pd.DataFrame(stake_d).replace(0, np.nan)            # daily return on the money staked
R = R.fillna(0.0)
wf_stake = pd.concat([pd.Series(stake_d[chosen[mo]].loc[[d for d in days if d[:7] == mo]].to_numpy(),
                                index=[d for d in days if d[:7] == mo]) for mo in chosen])
wf_ret = (wf / wf_stake.replace(0, np.nan)).fillna(0.0)
sharpes = (R.mean() / R.std(ddof=1)).to_numpy()
try:
    dsr = dsr_check(wf_ret.to_numpy(), n_trials=len(rules), trial_sharpes=sharpes)
except Exception as e:  # noqa: BLE001 - say what failed rather than stop
    dsr = {"error": repr(e)}
pbo = pbo_cscv(R.to_numpy(), n_splits=16)
say(f"\n   deflated Sharpe of the walk-forward choice (40 trials): {dsr}")
say(f"   probability of backtest overfitting across the 40 rules: {pbo['pbo']} over {pbo['splits']} splits")

# every rule over the walk-forward months, for the record
tab = pd.DataFrame({"rule": [f"bar {r[0]:.0%}, {r[1]:g}-{r[2]:g}, field>={r[3]}" for r in rules],
                    "net_gbp": [D.loc[wf.index, r].sum() for r in rules],
                    "turnover": [stake_d[r].loc[wf.index].sum() for r in rules]})
tab["net_pct"] = 100 * tab["net_gbp"] / tab["turnover"]
say("\n   every rule over the same days (net CLV, 2% commission):")
say(tab.sort_values("net_gbp", ascending=False).round(2).to_string(index=False))
