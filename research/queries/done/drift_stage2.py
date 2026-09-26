"""Stage 2 on the early-price trade: does the forecast's word on the BSP depend on the market's state?

The rule (scripts/clv_betfair.py, pre-registered in reports/preregistration_clv_forward.md) backs at
Betfair's morning price wherever ln(morning / forecast) >= 0.2 with at least 100 matched, and trades
out at BSP. It reads the forecast's disagreement with the morning price the same way in every market.
But the morning market is thin: where little has been matched the morning price is a poorer guide to
the BSP and the forecast should count for more; where the price is long, the move it takes to pay is
larger. This fits the realised drift ln(morning / BSP) on the forecast's move, alone and with its
interactions with the morning volume, the price, the field size and the race type, and asks whether
the bets a stage 2 would choose beat the rule's out of time: each month scored with a fit on the
months before it only (January's fit scores February; January and February's score March).

Inputs: the 958's out-of-sample forecasts (iteration 68's cw arm, research loop run 36233732671),
joined to Betfair's win prices before the locked holdout (2026-04-01) by scripts/clv_betfair.load.
"""

from __future__ import annotations

import io
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.clv_betfair import load  # noqa: E402

RUN, ARTIFACT, ARM = 36233732671, "research-iter68-served-944-connection-windows-84", "cw"
UNTIL = "2026-04-01"
MIN_VOL = 100.0
RULE = 0.2
T0 = time.time()


def say(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


def fetch_predictions() -> str:
    import requests
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        raise SystemExit("no GITHUB_TOKEN: cannot read the iteration's artifact")
    auth = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    r = requests.get(f"https://api.github.com/repos/{repo}/actions/runs/{RUN}/artifacts",
                     headers=auth, params={"name": ARTIFACT}, timeout=60)
    arts = r.json().get("artifacts", []) if r.ok else []
    if not arts:
        raise SystemExit(f"artifact {ARTIFACT} of run {RUN} not found ({r.status_code})")
    z = requests.get(arts[0]["archive_download_url"], headers=auth, timeout=600)
    z.raise_for_status()
    dest = Path("candidates/it68")
    zipfile.ZipFile(io.BytesIO(z.content)).extractall(dest)
    path = dest / f"oos_{ARM}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not in the artifact")
    return str(path)


def interval(net: np.ndarray, race: np.ndarray, n: int = 2000, seed: int = 0) -> str:
    if len(net) < 30:
        return f"too few ({len(net)})"
    g = pd.DataFrame({"r": race, "v": net}).groupby("r")["v"].agg(["sum", "size"])
    s, c = g["sum"].to_numpy(), g["size"].to_numpy()
    rng = np.random.default_rng(seed)
    b = [s[i].sum() / c[i].sum() for i in (rng.integers(0, len(g), len(g)) for _ in range(n))]
    return (f"{100 * net.mean():+6.2f}% ({100 * np.percentile(b, 5):+.2f} to {100 * np.percentile(b, 95):+.2f}), "
            f"n={len(net):,}, {net.sum():+.1f} units")


def design(d: pd.DataFrame, full: bool) -> tuple[np.ndarray, list[str]]:
    x = d["pred_move"].to_numpy()
    lm = np.log(d["morningwap"].to_numpy())
    cols = {"1": np.ones(len(d)), "move": x, "ln_m": lm, "ln_m2": lm ** 2}
    if full:
        lv = np.log(d["morning_vol"].to_numpy() + 1.0) - np.log(1000.0)
        nr = d["number_of_runners"].to_numpy(float) - 10.0
        jumps = (d["race_code"].astype(str) == "National Hunt").to_numpy(float)
        hcap = d["race_type"].astype(str).str.contains("Handicap", case=False).to_numpy(float)
        cols.update({"ln_vol": lv, "ln_vol*ln_m": lv * lm, "runners": nr, "jumps": jumps, "handicap": hcap,
                     "move*ln_vol": x * lv, "move*ln_m": x * lm, "move*runners": x * nr, "move*jumps": x * jumps,
                     "move*handicap": x * hcap})
    return np.column_stack(list(cols.values())), list(cols)


def main() -> None:
    d = load(fetch_predictions(), "horse_racing.db", None, UNTIL)
    d = d[d["morning_vol"] >= MIN_VOL].reset_index(drop=True)
    d["drift"] = np.log(d["morningwap"] / d["bsp"])
    d["month"] = d["race_date"].str[:7]
    d["rank1"] = d.groupby("race")["predicted_bfsp"].rank(method="first") == 1
    say(f"{len(d):,} runners with >= {MIN_VOL:.0f} matched in the morning, {d['race'].nunique():,} races, "
        f"{d['race_date'].min()} to {d['race_date'].max()}; months {sorted(d['month'].unique())}")
    y, race = d["drift"].to_numpy(), d["race"].to_numpy()

    print("\n## The realised drift on the forecast's move, all months (race-bootstrap SEs)")
    for full in (False, True):
        X, names = design(d, full)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        uniq = np.unique(race)
        rows = pd.Series(np.arange(len(d))).groupby(race).apply(np.asarray).to_dict()
        rng = np.random.default_rng(1)
        boot = np.array([np.linalg.lstsq(X[p], y[p], rcond=None)[0]
                         for p in (np.concatenate([rows[r] for r in rng.choice(uniq, len(uniq))]) for _ in range(200))])
        se = boot.std(axis=0)
        resid = y - X @ beta
        print(f"\n{'with the interactions' if full else 'the move and the price alone'}: "
              f"R2 {1 - resid.var() / y.var():.4f}")
        for nm, b, s in zip(names, beta, se):
            print(f"  {nm:16s} {b:+.4f}  (SE {s:.4f}, t {b / s if s else float('nan'):+.1f})")

    print("\n## Where the rule's edge is: net per unit by the move and the morning volume")
    d["vol_band"] = pd.cut(d["morning_vol"], [MIN_VOL, 300, 1000, 3000, np.inf], right=False,
                           labels=["100-300", "300-1k", "1k-3k", "3k+"])
    d["move_band"] = pd.cut(d["pred_move"], [-np.inf, 0.0, 0.1, 0.2, 0.35, np.inf], right=False,
                            labels=["<0", "0-0.1", "0.1-0.2", "0.2-0.35", "0.35+"])
    tab = d.groupby(["vol_band", "move_band"], observed=True)["net"].agg(["mean", "size"])
    tab["mean"] = (100 * tab["mean"]).round(2)
    print(tab.unstack("move_band").to_string())

    print("\n## Out of time: each month's bets chosen by a fit on the months before it")
    months = sorted(m for m in d["month"].unique() if (d["month"] == m).sum() >= 1000)
    for m in months[1:]:
        tr, te = (d["month"] < m).to_numpy(), (d["month"] == m).to_numpy()
        t = d[te]
        rule = (t["pred_move"] >= RULE).to_numpy()
        print(f"\n{m}: trained on {tr.sum():,} runners before it; tested on {te.sum():,}")
        print(f"  the rule (move >= {RULE})              {interval(t['net'].to_numpy()[rule], t['race'].to_numpy()[rule])}")
        for full in (False, True):
            X, _ = design(d, full)
            beta = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
            yhat_tr, yhat = X[tr] @ beta, X[te] @ beta
            # the rule's count this month, the stage 2's top picks
            k = int(rule.sum())
            top = np.zeros(te.sum(), bool)
            top[np.argsort(-yhat)[:k]] = True
            lab = "stage 2 + interactions" if full else "stage 2, move and price"
            print(f"  {lab:24s} at the rule's count {interval(t['net'].to_numpy()[top], t['race'].to_numpy()[top])}")
            # a threshold fixed on the training months: the one that made the most there
            net_tr = d["net"].to_numpy()[tr]
            grid = np.quantile(yhat_tr, np.linspace(0.5, 0.98, 49))
            tau = grid[int(np.argmax([net_tr[yhat_tr >= g].sum() for g in grid]))]
            pick = yhat >= tau
            print(f"  {lab:24s} at its own threshold ({tau:+.3f}) "
                  f"{interval(t['net'].to_numpy()[pick], t['race'].to_numpy()[pick])}")
            # the owner's measure: the model's rank 1, where stage 2 expects the price to shorten
            r1 = t["rank1"].to_numpy()
            print(f"  {'':24s} rank 1 where it expects a drift > 0   "
                  f"{interval(t['net'].to_numpy()[r1 & (yhat > 0)], t['race'].to_numpy()[r1 & (yhat > 0)])}")
        r1 = t["rank1"].to_numpy()
        print(f"  {'the model rank 1, every race':38s} {interval(t['net'].to_numpy()[r1], t['race'].to_numpy()[r1])}")
    say("done")


if __name__ == "__main__":
    main()
