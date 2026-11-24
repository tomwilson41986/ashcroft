"""M2 of the review of 20 Nov 2026 (ledger closing-refit-stn-1121): the trader's closing model refitted on the new
blend's own prices against the one it serves now. Read-only; nothing is traded or served.

The trader's expected CLV comes from data/models/closing_model_novol.json: a conditional logit of the Betfair SP book
on the market's price now (weight 0.774) and ours (0.263), fitted on January-March 2026 morning prices against an
older blend, with no traded volume (the delayed feed). The blend being served (gate_stn) prices differently; this asks
whether the rule picks better bets with the closing model refitted on gate_stn's own out-of-sample prices.

Iteration 131's re-run oos_gate_stn.csv (run 44715810929; 2024-12-28 to 2026-03-31; the locked holdout not read),
joined to Betfair's morning prices (scripts/clv_betfair.load), whole races. Volume set to nothing in the closing model's
inputs for both arms, as the live feed gives it (the 100 GBP morning-money filter still reads the real volume).
    served   the served closing model, fixed, on every month scored
    refit    the closing model fitted on the months before each month, on gate_stn's prices (as research scores)
Scored months: from the first with 500 earlier races to 2025-12 (the served model was fitted on January-March 2026,
so those months are left out for both). The owner's rule at the 3% bar (GBP250 to win at the morning WAP, 2% on a
winning CLV). Per day: net CLV x stake, refit less served.
"""
from __future__ import annotations

import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.getcwd())
from desk import trials  # noqa: E402
from model import race_book as rb  # noqa: E402
from research import overfit  # noqa: E402
from scripts import clv_betfair as cb  # noqa: E402

RUN, PREFIX, FILE = 44715810929, "research-iter131", "oos_gate_stn.csv"
UNTIL = "2026-01-01"
BAR = 0.03
COMMISSION = 0.02          # the account's rate (trading/config_live.json); scripts/clv_betfair.COMMISSION is 5%
rng = np.random.default_rng(20261121)


def fetch(path):
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY", "tomwilson41986/ashcroft")
    hd = {"Authorization": f"Bearer {token}"} if token else {}
    j = requests.get(f"https://api.github.com/repos/{repo}/actions/runs/{RUN}/artifacts?per_page=100", headers=hd,
                     timeout=60).json()
    for a in [a for a in j.get("artifacts", []) if a["name"].startswith(PREFIX)]:
        z = requests.get(a["archive_download_url"], headers=hd, timeout=900)
        with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
            hit = [n for n in zf.namelist() if os.path.basename(n) == FILE]
            if hit:
                open(path, "wb").write(zf.read(hit[0]))
                return a["name"]
    raise SystemExit(f"no data found: {FILE} not in run {RUN}")


def score(d: pd.DataFrame, model) -> pd.DataFrame:
    """Each runner's expected CLV at the morning WAP under one fixed closing model (volume read as none)."""
    out = []
    for month, m in d.groupby("month", sort=True):
        r_ = np.random.default_rng(int(month.replace("-", "")))
        for _, r in m.groupby("race", sort=False):
            inputs = rb.closing_inputs(r.assign(morning_vol=0.0))
            g = np.zeros(len(r), int)
            draws = rb.closing_draws(model.predict(inputs, g), model.sigma(inputs), book=model.book, n_draws=300, rng=r_)
            out.append(r.assign(ev=(r["morningwap"].to_numpy(float) * draws).mean(axis=0) - 1.0))
    return pd.concat(out)


def walk_forward(d: pd.DataFrame) -> pd.DataFrame:
    """As rb.expected_clv_walk_forward, with the volume read as none in the fit and the scoring."""
    x = d.assign(morning_vol_real=d["morning_vol"], morning_vol=0.0)
    s = rb.expected_clv_walk_forward(x)
    return s.assign(morning_vol=s["morning_vol_real"])


def bets(s: pd.DataFrame) -> pd.DataFrame:
    b = s[(s["ev"] >= BAR) & (s["morning_vol"].fillna(0) >= rb.MIN_VOL)].copy()
    m, bsp = b["morningwap"].to_numpy(float), b["bsp"].to_numpy(float)
    b["stake"] = rb.TO_WIN / (m - 1.0)
    clv = m / bsp - 1.0
    b["net_v"] = b["stake"] * np.where(clv > 0, clv * (1 - COMMISSION), clv)
    return b


os.makedirs("out", exist_ok=True)
art = fetch("out/oos_gate_stn_131.csv")
d = cb.load("out/oos_gate_stn_131.csv", "horse_racing.db", None, until=UNTIL)
w = rb.whole_races(d, field=d.groupby("race")["field"].first())
w = w.sort_values(["race_date", "race", "predicted_bfsp"]).reset_index(drop=True)
w["month"] = w["race_date"].astype(str).str[:7]
print(f"{art} {FILE}: {w['race'].nunique():,} whole races, {len(w):,} runners, {w['race_date'].min()} to "
      f"{w['race_date'].max()}")
refit = walk_forward(w)
months = sorted(refit["month"].unique())
served_model = rb.load_closing_model("data/models/closing_model_novol.json")
served = score(w[w["month"].isin(months)], served_model)
print(f"scored months {months[0]} to {months[-1]} ({len(months)}), {refit['race'].nunique():,} races")
a, b = bets(refit), bets(served)
for name, x in (("served", b), ("refit", a)):
    print(f"   {name:7s}: {len(x):,} bets on {x['race_date'].nunique()} days, turnover GBP{x['stake'].sum():,.0f}, net CLV "
          f"GBP{x['net_v'].sum():+,.0f} ({100 * x['net_v'].sum() / x['stake'].sum():+.2f}%), expected CLV mean "
          f"{100 * x['ev'].mean():+.2f}%")
days = sorted(set(a["race_date"]) | set(b["race_date"]))
da = a.groupby("race_date")["net_v"].sum().reindex(days, fill_value=0.0)
db = b.groupby("race_date")["net_v"].sum().reindex(days, fill_value=0.0)
diff = (da - db).to_numpy()
idx = rng.integers(0, len(diff), size=(4000, len(diff)))
bs = diff[idx].mean(axis=1)
n_trials = trials.count_trials(trials.parse(trials.LEDGER.read_text())[0])
dsr = overfit.dsr_check(diff, n_trials=n_trials)
print(f"\nrefit less served, net CLV x stake a day: GBP{diff.mean():+.2f} (90% {np.quantile(bs, 0.05):+.2f} to "
      f"{np.quantile(bs, 0.95):+.2f}), sum GBP{diff.sum():+,.0f} over {len(diff)} days, ahead on {(diff > 0).sum()}; "
      f"deflated Sharpe {dsr.get('dsr')} ({n_trials:,} trials): {'pass' if dsr.get('passed') else 'fail'}")
ok = diff.sum() > 0 and np.quantile(bs, 0.05) > 0 and dsr.get("passed")
print(f"verdict: {'PASS' if ok else 'FAIL'} (positive sum, the 90% day interval above zero, deflated Sharpe >= 0.95)")
