"""The owner, 20 Nov 2026: "test" (the agreement idea, from a reading of a ratings writer's method: a conclusion is
trusted when independent ratings agree). Read-only; a screen, not a rule.

Does the disagreement of the served blend's three members about a horse say how reliable our price is? Iteration
130's out-of-sample files (run 44473514515: oos_st, oos_dml_st, oos_rx16_st and the gated blend oos_gate_st;
2024-12-28 to 2026-03-31; the locked holdout not read). Per runner the disagreement is the standard deviation of the
three members' log prices. Longer prices disagree more by construction, so every reading is within the blend's price
band: each runner is put in the low, middle or high third of disagreement among the runners of its own fold and band.

1. The error: the blend's |log(price / BSP)| by third within band; high less low with a 90% race-day interval.
2. Against the morning: the blend's error less the morning WAP's, by third (is the model's lead over the morning
   price where the members agree?).
3. The bets: the owner's staking (GBP250 to win at the morning WAP, closing model fitted on earlier months) at the
   3% bar and the 0% bar; the stake-weighted CLV by third, low less high with a 90% interval (days resampled, both
   thirds on the same draws); the return held.

The reading, fixed before the run. The idea earns a registered rule test (a bar that moves with disagreement, chosen
on earlier folds, deflated Sharpe with every trial counted) only if both hold: (a) the error is higher in the high
third than the low third with the interval above zero, pooled and in at least four of the six bands; (b) at the 3%
bar the low third's CLV is above the high third's with the interval of the difference above zero. Otherwise it is
closed. One trial is counted for this screen.
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
from model import race_book as rb  # noqa: E402
from scripts import clv_betfair as cb  # noqa: E402

RUN, PREFIX = 44473514515, "research-iter130"
FILES = {"blend": "oos_gate_st.csv", "st": "oos_st.csv", "dml": "oos_dml_st.csv", "rx": "oos_rx16_st.csv"}
UNTIL = "2026-04-01"
KEY = ["race_date", "track", "race_time", "horse_name"]
BANDS, LABELS = [1, 3, 6, 12, 25, 50, 1e9], ["<3", "3-6", "6-12", "12-25", "25-50", "50+"]
pd.set_option("display.width", 250)
rng = np.random.default_rng(20261120)


def say(*a):
    print(*a, flush=True)


def fetch():
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
    want = dict(FILES)
    for a in [a for a in arts if a["name"].startswith(PREFIX)]:
        z = requests.get(a["archive_download_url"], headers=hd, timeout=900)
        with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
            for k, f in list(want.items()):
                hit = [n for n in zf.namelist() if os.path.basename(n) == f]
                if hit:
                    open(f"out/{f}", "wb").write(zf.read(hit[0]))
                    want.pop(k)
    if want:
        raise SystemExit(f"no data found: {sorted(want.values())} not in run {RUN}'s {PREFIX}* artifacts")


def boot(num, den, day, draws=4000, seed_idx=None):
    num, den = np.asarray(num, float), np.asarray(den, float)
    days, inv = np.unique(np.asarray(day), return_inverse=True)
    return np.bincount(inv, num, len(days)), np.bincount(inv, den, len(days)), days


def ci(num, den, day):
    sn, sd, days = boot(num, den, day)
    idx = rng.integers(0, len(days), size=(4000, len(days)))
    bs = sn[idx].sum(1) / np.maximum(sd[idx].sum(1), 1e-12)
    return sn.sum() / sd.sum(), float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))


def diff_ci(a, b, num, den="one"):
    """mean(or ratio) of frame a less frame b, days resampled together."""
    days = np.unique(np.concatenate([a["race_date"].to_numpy(), b["race_date"].to_numpy()]))
    pos = {d: i for i, d in enumerate(days)}

    def sums(f):
        inv = f["race_date"].map(pos).to_numpy()
        den_v = np.ones(len(f)) if den == "one" else f[den].to_numpy(float)
        return np.bincount(inv, f[num].to_numpy(float), len(days)), np.bincount(inv, den_v, len(days))
    an, ad = sums(a)
    bn, bd = sums(b)
    idx = rng.integers(0, len(days), size=(4000, len(days)))
    bs = an[idx].sum(1) / np.maximum(ad[idx].sum(1), 1e-12) - bn[idx].sum(1) / np.maximum(bd[idx].sum(1), 1e-12)
    est = an.sum() / ad.sum() - bn.sum() / bd.sum()
    return est, float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))


os.makedirs("out", exist_ok=True)
fetch()
d = cb.load(f"out/{FILES['blend']}", "horse_racing.db", None, until=UNTIL)
for k in ("st", "dml", "rx"):
    m = pd.read_csv(f"out/{FILES[k]}", dtype={"race_time": str})[KEY + ["predicted_bfsp"]]
    d = d.merge(m.rename(columns={"predicted_bfsp": f"p_{k}"}).drop_duplicates(KEY), on=KEY, how="left")
have = d[["p_st", "p_dml", "p_rx"]].gt(1).all(axis=1)
say(f"runners with a morning WAP, a BSP and the blend's price: {len(d):,}; with all three members' prices {int(have.sum()):,}")
d = d[have].copy()
logs = np.log(d[["p_st", "p_dml", "p_rx"]].to_numpy(float))
d["dis"] = logs.std(axis=1)
d["e_model"] = np.abs(np.log(d["predicted_bfsp"] / d["bsp"]))
d["e_morn"] = np.abs(np.log(d["morningwap"] / d["bsp"]))
d["gap"] = d["e_model"] - d["e_morn"]
d["band"] = pd.cut(d["predicted_bfsp"], BANDS, labels=LABELS)
if "fold_idx" not in d.columns:
    d["fold_idx"] = 0
d["third"] = d.groupby(["fold_idx", "band"], observed=True)["dis"].transform(
    lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "middle", "high"])).astype(str)
say(f"disagreement (sd of the three members' log prices): median {d['dis'].median():.4f}, 90th pct "
    f"{d['dis'].quantile(0.9):.4f}; the members' pairwise correlations of log price: st-dml "
    f"{np.corrcoef(logs[:, 0], logs[:, 1])[0, 1]:.4f}, st-rx {np.corrcoef(logs[:, 0], logs[:, 2])[0, 1]:.4f}, dml-rx "
    f"{np.corrcoef(logs[:, 1], logs[:, 2])[0, 1]:.4f}")
say("   median disagreement by the blend's price band: "
    + ", ".join(f"{b} {v:.4f}" for b, v in d.groupby("band", observed=True)["dis"].median().items()))

say("\n== 1. the blend's error against the BSP by third of disagreement, within its price band")
rows, held = [], 0
for b, g in d.groupby("band", observed=True):
    lo_, hi_ = g[g["third"] == "low"], g[g["third"] == "high"]
    est, lo, hi = diff_ci(hi_, lo_, "e_model")
    held += int(lo > 0)
    rows.append({"band": b, "runners": len(g), "low": lo_["e_model"].mean(), "middle": g[g["third"] == "middle"]["e_model"].mean(),
                 "high": hi_["e_model"].mean(), "high-low": est, "lo": lo, "hi": hi,
                 "spearman": g["dis"].corr(g["e_model"], method="spearman")})
say(pd.DataFrame(rows).round(4).to_string(index=False))
lo_, hi_ = d[d["third"] == "low"], d[d["third"] == "high"]
est, lo, hi = diff_ci(hi_, lo_, "e_model")
say(f"   pooled (thirds within band): low {lo_['e_model'].mean():.4f}, middle {d[d['third'] == 'middle']['e_model'].mean():.4f}, "
    f"high {hi_['e_model'].mean():.4f}; high less low {est:+.4f} ({lo:+.4f} to {hi:+.4f}); bands with the interval above "
    f"zero: {held} of {len(rows)}")
reading_a = lo > 0 and held >= 4

say("\n== 2. the blend's error less the morning WAP's, by third (negative: the model is the closer)")
for t in ("low", "middle", "high"):
    g = d[d["third"] == t]
    e, a, b_ = ci(g["gap"], np.ones(len(g)), g["race_date"])
    say(f"   {t:6s}: model {g['e_model'].mean():.4f}, morning {g['e_morn'].mean():.4f}, gap {e:+.4f} ({a:+.4f} to {b_:+.4f})")

say("\n== 3. the bets: the owner's staking, the closing model fitted on earlier months")
w = rb.whole_races(d, field=d.groupby("race")["field"].first())
w = w.sort_values(["race_date", "race", "predicted_bfsp"]).reset_index(drop=True)
scored = rb.expected_clv_walk_forward(w)
reading_b = False
for bar in (0.03, 0.0):
    b = scored[(scored["ev"] >= bar) & (scored["morning_vol"].fillna(0) >= rb.MIN_VOL)].copy()
    mw, bsp = b["morningwap"].to_numpy(float), b["bsp"].to_numpy(float)
    b["stake"] = rb.TO_WIN / (mw - 1.0)
    b["clv_v"] = b["stake"] * (mw / bsp - 1.0)
    b["res_v"] = np.where(b["won"].astype(bool), b["stake"] * (mw - 1.0), -b["stake"])
    e, lo, hi = ci(b["clv_v"], b["stake"], b["race_date"])
    say(f"\n   expected CLV >= {100 * bar:.0f}%: {len(b):,} bets, turnover GBP{b['stake'].sum():,.0f}, CLV {100 * e:+.2f}% "
        f"({100 * lo:+.2f} to {100 * hi:+.2f}); bets by third: {b['third'].value_counts().to_dict()}")
    for t in ("low", "middle", "high"):
        g = b[b["third"] == t]
        e, lo, hi = ci(g["clv_v"], g["stake"], g["race_date"])
        say(f"     {t:6s}: {len(g):6,} bets, GBP{g['stake'].sum():10,.0f}, CLV {100 * e:+.2f}% ({100 * lo:+.2f} to "
            f"{100 * hi:+.2f}), held {100 * g['res_v'].sum() / g['stake'].sum():+.2f}%, expected CLV mean {100 * g['ev'].mean():+.2f}%")
    e, lo, hi = diff_ci(b[b["third"] == "low"], b[b["third"] == "high"], "clv_v", "stake")
    say(f"     low less high: {100 * e:+.2f} points ({100 * lo:+.2f} to {100 * hi:+.2f})")
    if bar == 0.03:
        reading_b = lo > 0
    for band, g in b.groupby("band", observed=True):
        lo_g, hi_g = g[g["third"] == "low"], g[g["third"] == "high"]
        if len(lo_g) < 100 or len(hi_g) < 100:
            continue
        e, lo, hi = diff_ci(lo_g, hi_g, "clv_v", "stake")
        say(f"       band {band:6s} low less high: {100 * e:+.2f} points ({100 * lo:+.2f} to {100 * hi:+.2f}); "
            f"{len(lo_g):,} and {len(hi_g):,} bets")

say(f"\nreading: (a) the error rises with disagreement: {'yes' if reading_a else 'no'}; (b) at the 3% bar the low third's "
    f"CLV is above the high third's: {'yes' if reading_b else 'no'} -> "
    f"{'a rule test is earned' if (reading_a and reading_b) else 'closed as registered'}")
