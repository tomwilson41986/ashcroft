"""Rank-gated blends of race_xent with the pairs: their early-price trade, paired against the candidate pair.

race_xent (the within-race cross-entropy on the Betfair SP book) prices each race's leaders better than any
model here and its long shots worse (iteration 103 on complete careers: ranks 1-3 -0.0110 to -0.0170 each
against the main, ranks 8 and beyond +0.045), because it counts a runner for its share of the race. An equal
average gives it a third of every runner. A gate gives it the leaders and leaves the long shots to the pair:
the blend takes race_xent's log price with weight w and the pair's with 1 - w, then renormalises each race to a
book of 1. The weight depends only on the pair's own forecast, known at 06:00:

  step: 1 for the pair's ranks 1-3, 0.5 for 4-7, 0 from 8
  ramp: 1 for ranks 1-3, then 0.8, 0.6, 0.4, 0.2, and 0 from 8
  prob: 1 where the pair gives 15% or more, 0 at 5% or less, linear between

On the price (scripts/compare_oos_runs.py, offline from the iterations' files), the complete-careers pair gated
with the capped race_xent: step 0.3926, ramp 0.3922, prob 0.3923 against the candidate pair's 0.3966 (-0.0040
to -0.0044, each interval clear of zero) and against the complete-careers pair's 0.3945 (-0.0019 to -0.0023),
Brier skill against the market resolved better (+0.0031; +0.0026 on the complete-careers pair), concordance
level or better. The gate was chosen after seeing this window's ranks, so it is checked where it could fail: in
each half (fold 0, Sep-Dec: -0.0023; folds 1-2, Dec-Mar: -0.0015, both resolved), with the 12,000-round
race_xent (-0.0036), and on cut careers, other fits entirely (the candidate pair gated with iteration 101's
race_xent: -0.0019, -0.0026 to -0.0013).

This settles the fixed early-price rule and the rank-1 pick (scripts/clv_betfair.py, holdout excluded) for the
gated blends and pairs each against the candidate pair, and the step gate against the complete-careers pair
and against race_xent alone. Read-only.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from research_loop import average_arms  # noqa: E402

IT96 = (36346975763, "research-iter96-served-968s5xh-kalman-handicap-travel-116")
IT97 = (36358454411, "research-iter97-served-968s5xh-travel-partners-117")
IT101 = 36371702086
IT102 = (36381991232, "research-iter102-served-pair-complete-careers-123")
IT103 = (36384069890, "research-iter103-served-race-xent-complete-careers-124")
KEY = ["race_date", "race_time", "track", "horse_name"]


def folds(run: int, number: int, arm: str) -> list[tuple[int, str, str]]:
    """An arm's out-of-sample predictions as its three fit artifacts hold them."""
    return [(run, f"fit-{number}-{arm}-{k}", f"oos_{arm}_f{k}.csv") for k in range(3)]


SOURCES = {
    "xthub_all3": [(*IT96, "oos_cw_dm_xthub_kr_hca_tv.csv")],        # iteration 96's main, 0.3984
    "dml_all3": [(*IT97, "oos_cw_dm_xt_dml_kr_hca_tv.csv")],         # iteration 97's partner, 0.4013
    "rx_all3": folds(IT101, 122, "cw_dm_xt_rx_kr_hca_tv"),           # race_xent on cut careers, 0.4092
    "main_h18": [(*IT102, "oos_base.csv")],                          # the complete-careers main, 0.3968
    "dml_h18": [(*IT102, "oos_dml_h18_t21.csv")],                    # the complete-careers partner, 0.3987
    "rx_h18": [(*IT103, "oos_rx_h18_t21.csv")],                      # race_xent, 12,000 rounds, 0.4072
    "rx16_h18": [(*IT103, "oos_rx_h18_t21_r16k.csv")],               # race_xent, 16,000-round cap, 0.4059
}
AVERAGES = {
    "pair97": ["xthub_all3", "dml_all3"],                            # control: the candidate pair (0.3966)
    "pair_h18": ["main_h18", "dml_h18"],                             # 0.3945
}
GATES = {                                                            # (the pair, race_xent, gate)
    "gate_step_h18": ("pair_h18", "rx16_h18", "step"),               # 0.3926
    "gate_ramp_h18": ("pair_h18", "rx16_h18", "ramp"),               # 0.3922
    "gate_prob_h18": ("pair_h18", "rx16_h18", "prob"),               # 0.3923
    "gate_step_rx12": ("pair_h18", "rx_h18", "step"),                # 0.3930
    "gate_step_cut": ("pair97", "rx_all3", "step"),                  # 0.3946
}
OUT = Path("out/gate_clv")


def gate(pair: Path, rx: Path, mode: str, out: Path) -> None:
    """race_xent's log price for the pair's leaders, the pair's for its long shots, a book of 1 per race."""
    import numpy as np
    import pandas as pd
    a = pd.read_csv(pair, dtype={"race_time": str})
    b = pd.read_csv(rx, dtype={"race_time": str}).set_index(KEY).reindex(a.set_index(KEY).index)
    if b["predicted_bfsp"].isna().any():
        raise SystemExit(f"{rx}: lacks runners of {pair}")
    race = (a["race_date"].astype(str) + "|" + a["track"].astype(str) + "|" + a["race_time"].astype(str)).to_numpy()
    la, lb = np.log(a["predicted_bfsp"].to_numpy(float)), np.log(b["predicted_bfsp"].to_numpy(float))
    rank = pd.Series(la).groupby(race).rank(method="first").to_numpy()
    if mode == "step":
        w = np.where(rank <= 3, 1.0, np.where(rank <= 7, 0.5, 0.0))
    elif mode == "ramp":
        w = np.clip((8 - rank) / 5, 0, 1)
    elif mode == "prob":
        w = np.clip((np.exp(-la) - 0.05) / 0.10, 0, 1)
    else:
        raise SystemExit(f"unknown gate {mode}")
    p = np.exp(-(w * lb + (1 - w) * la))
    pn = p / pd.Series(p).groupby(race).transform("sum").to_numpy()
    o = a.copy()
    o["predicted_win_prob_norm"] = pn
    o["predicted_bfsp"] = 1.0 / pn
    if "overlay_pct" in o and "bfsp" in o:
        o["overlay_pct"] = (o["bfsp"] / o["predicted_bfsp"] - 1) * 100
    o.to_csv(out, index=False)


def _zip(run: int, artifact: str, cache: dict) -> zipfile.ZipFile:
    import requests
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    auth = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    if (run, artifact) not in cache:
        r = requests.get(f"https://api.github.com/repos/{repo}/actions/runs/{run}/artifacts",
                         headers=auth, params={"name": artifact}, timeout=60)
        arts = r.json().get("artifacts", []) if r.ok else []
        if not arts:
            raise SystemExit(f"artifact {artifact} of run {run} not found ({r.status_code})")
        z = requests.get(arts[0]["archive_download_url"], headers=auth, timeout=600)
        z.raise_for_status()
        cache[(run, artifact)] = zipfile.ZipFile(io.BytesIO(z.content))
    return cache[(run, artifact)]


def fetch(name: str, parts: list[tuple[int, str, str]], cache: dict) -> Path:
    """One arm's predictions, its parts (one file, or a file per fold) concatenated."""
    import pandas as pd
    frames = []
    for run, artifact, member in parts:
        zf = _zip(run, artifact, cache)
        inner = next(n for n in zf.namelist() if n.split("/")[-1] == member)
        frames.append(pd.read_csv(io.BytesIO(zf.read(inner)), dtype={"race_time": str}))
    df = pd.concat(frames, ignore_index=True)
    path = OUT / f"oos_{name}.csv"
    df.to_csv(path, index=False)
    print(f"{name}: {len(df):,} runners from {len(parts)} file(s)")
    return path


def clv(item: tuple[str, Path]) -> tuple[str, str]:
    arm, path = item
    res = subprocess.run([sys.executable, "scripts/clv_betfair.py", "--predictions", str(path),
                          "--db", "horse_racing.db", "--until", "2026-04-01"], capture_output=True, text=True)
    return arm, res.stdout + (res.stderr[-2000:] if res.returncode else "")


PAIRED = [                                                           # (variant, control)
    ("gate_step_h18", "pair97"), ("gate_ramp_h18", "pair97"), ("gate_prob_h18", "pair97"),
    ("gate_step_rx12", "pair97"), ("gate_step_cut", "pair97"),
    ("gate_step_h18", "pair_h18"), ("gate_step_h18", "rx16_h18"),
]


def bets(path: Path):
    """The rule's bets and the top picks, each as net per unit by race (clv_betfair's own definitions)."""
    from clv_betfair import load
    d = load(str(path), "horse_racing.db", None, "2026-04-01")
    base = d[d["morning_vol"] >= 100.0]
    rule = base[base["pred_move"] >= 0.2]
    r = d.groupby("race")["predicted_bfsp"].rank(method="first")
    top = base[(r.reindex(base.index) == 1).to_numpy()]
    return {k: g.groupby("race")["net"].agg(["sum", "size"]) for k, g in (("rule", rule), ("top pick", top))}


def paired(variant: dict, control: dict, n: int = 2000, seed: int = 0) -> list[str]:
    """The variant's return less the control's, races resampled together (90% interval), and units per race."""
    import numpy as np
    out = []
    for k in ("rule", "top pick"):
        v, c = variant[k], control[k]
        races = v.index.union(c.index)
        vs, vn = (v.reindex(races)[col].fillna(0).to_numpy() for col in ("sum", "size"))
        cs, cn = (c.reindex(races)[col].fillna(0).to_numpy() for col in ("sum", "size"))
        rng = np.random.default_rng(seed)
        w = np.stack([np.bincount(rng.integers(0, len(races), len(races)), minlength=len(races)) for _ in range(n)])
        diff = (w @ vs) / (w @ vn) - (w @ cs) / (w @ cn)
        units = (w @ vs - w @ cs) / len(races) * 1000
        out.append(f"  {k:8s} {100 * vs.sum() / vn.sum():+6.2f}% (n {int(vn.sum()):,}) against "
                   f"{100 * cs.sum() / cn.sum():+6.2f}% (n {int(cn.sum()):,}): "
                   f"{100 * (vs.sum() / vn.sum() - cs.sum() / cn.sum()):+.2f} points "
                   f"({100 * np.percentile(diff, 5):+.2f} to {100 * np.percentile(diff, 95):+.2f}); "
                   f"units won per 1,000 of these races {1000 * (vs.sum() - cs.sum()) / len(races):+.1f} "
                   f"({np.percentile(units, 5):+.1f} to {np.percentile(units, 95):+.1f})")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    files = {arm: fetch(arm, parts, cache) for arm, parts in SOURCES.items()}
    for name, arms in AVERAGES.items():
        files[name] = OUT / f"oos_{name}.csv"
        average_arms([str(files[a]) for a in arms], str(files[name]))
    for name, (pair, rx, mode) in GATES.items():
        files[name] = OUT / f"oos_{name}.csv"
        gate(files[pair], files[rx], mode, files[name])
    order = ["gate_step_h18", "gate_ramp_h18", "gate_prob_h18", "gate_step_cut"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for arm, body in pool.map(clv, [(a, files[a]) for a in order]):
            print(f"\n===== {arm} =====")
            print(body)
    print("\n===== paired: each variant's bets against its control's, races resampled together =====")
    need = sorted({a for pair in PAIRED for a in pair})
    got = dict(zip(need, map(bets, (files[a] for a in need))))
    for v, c in PAIRED:
        print(f"{v} against {c}")
        print("\n".join(paired(got[v], got[c])))


if __name__ == "__main__":
    main()
