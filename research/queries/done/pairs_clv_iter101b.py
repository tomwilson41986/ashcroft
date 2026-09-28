"""The early-price trade for more averages with iteration 101's race_xent partner (follow-up to pairs_clv_iter101).

The first query found race_xent trades better in every average that holds it, its edge in the 1-16 morning-price
bands. This scores the averages it did not: the three members with race_xent at learning rate 0.02 (level on the
price, 0.3969), two race_xent fits together (0.4078), the demeaned-log partner beside race_xent (0.4001), and the
main beside both race_xent fits (0.4003), each paired against the candidate pair.

Original notes:

Iteration 101 fitted the within-race extra-trees partner with the three blocks on race_xent (the market's
within-race probabilities by their cross-entropy, so a runner counts for its share of the race) at learning
rates 0.02 and 0.05. On the price it is worse than the partner it would replace (lr 0.05: 0.4108 alone against
0.4013, beside the main 0.3986 against 0.3966) while its Brier skill against the market is resolved better, so
the question left is whether the better probabilities reach the bets. Iteration 100 fitted the main on the
history from 2018 (rows from 2021, float32: 0.3968 against 0.3984); beside iteration 97's partner it prices
0.3946 against the candidate pair's 0.3966.

This runs the fixed early-price rule and the rank-1 settlement (scripts/clv_betfair.py, holdout excluded) on
those pairs, each fit's folds read from its own fit artifact (so the query does not wait on a compare job), with
the candidate pair (iteration 96's main beside iteration 97's partner, 0.3966: the rule +10.70%, the top pick
traded out +2.01%) recomputed as the control. Read-only.
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
IT100 = 36366379549
IT101 = 36371702086


def folds(run: int, number: int, arm: str) -> list[tuple[int, str, str]]:
    """An arm's out-of-sample predictions as its three fit artifacts hold them."""
    return [(run, f"fit-{number}-{arm}-{k}", f"oos_{arm}_f{k}.csv") for k in range(3)]


SOURCES = {
    "xthub_all3": [(*IT96, "oos_cw_dm_xthub_kr_hca_tv.csv")],        # iteration 96's main, 0.3984
    "dml_all3": [(*IT97, "oos_cw_dm_xt_dml_kr_hca_tv.csv")],         # iteration 97's partner, 0.4013
    "rx_all3": folds(IT101, 122, "cw_dm_xt_rx_kr_hca_tv"),           # race_xent, lr 0.02
    "rx05_all3": folds(IT101, 122, "cw_dm_xt_rx05_kr_hca_tv"),       # race_xent, lr 0.05
}
AVERAGES = {
    "pair_all3_dmlall3": ["xthub_all3", "dml_all3"],                 # control: the candidate pair (0.3966)
    "three_all3_dml_rx": ["xthub_all3", "dml_all3", "rx_all3"],      # 0.3969
    "pair_rx_rx05": ["rx_all3", "rx05_all3"],                        # 0.4078
    "pair_dml_rx": ["dml_all3", "rx_all3"],                          # 0.4001
    "three_all3_rx_rx05": ["xthub_all3", "rx_all3", "rx05_all3"],    # 0.4003
}
OUT = Path("out/pairs_clv_iter101b")


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
        inner = next(n for n in zf.namelist() if n.endswith(member))
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
    ("three_all3_dml_rx", "pair_all3_dmlall3"), ("pair_rx_rx05", "pair_all3_dmlall3"),
    ("pair_dml_rx", "pair_all3_dmlall3"), ("three_all3_rx_rx05", "pair_all3_dmlall3"),
    ("rx_all3", "pair_all3_dmlall3"), ("pair_rx_rx05", "rx_all3"),
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
    order = list(AVERAGES) + ["rx_all3"]
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
