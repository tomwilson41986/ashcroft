"""The early-price trade for the complete-careers averages (iterations 102 and 103), paired against the candidate pair.

Iteration 102 fitted the within-race partner on the history from 2018 (rows from 2021, float32) beside the
complete-careers main: the pair prices 0.3945 against the candidate pair's 0.3966 (-0.0020, -0.0028 to -0.0013),
the rule +10.99% and the top pick traded out +1.83% (the candidate pair: +10.70%, +2.01%). Iteration 103 fitted
race_xent on the same history, at 12,000 rounds (0.4072 alone) and with a 16,000-round cap it stops short of in
every fold (0.4059); alone they trade +12.30% and +12.60%. Averaged offline with iteration 102's two fits, the
three members with the capped race_xent price 0.3946 (-0.0020, -0.0028 to -0.0012) with Brier skill against
the market resolved better (+0.0021) and concordance level, so the question is whether the three trade as
race_xent does while pricing as the complete-careers pair does.

Iteration 103's base is the same main refitted on another processor (fold 0 stopped at 10,653 rounds instead
of 11,989); it prices +0.0004 (+0.0001 to +0.0008) against iteration 102's, so its bets against iteration 102's
are the refit noise of the bets themselves.

The fixed early-price rule and the rank-1 settlement (scripts/clv_betfair.py, holdout excluded), each fit read
from its iteration's artifact, with the candidate pair (iteration 96's main beside iteration 97's partner)
recomputed as the control. Read-only.
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
IT102 = (36381991232, "research-iter102-served-pair-complete-careers-123")
IT103 = (36384069890, "research-iter103-served-race-xent-complete-careers-124")

SOURCES = {
    "xthub_all3": [(*IT96, "oos_cw_dm_xthub_kr_hca_tv.csv")],        # iteration 96's main, 0.3984
    "dml_all3": [(*IT97, "oos_cw_dm_xt_dml_kr_hca_tv.csv")],         # iteration 97's partner, 0.4013
    "main_h18": [(*IT102, "oos_base.csv")],                          # the complete-careers main, 0.3968
    "dml_h18": [(*IT102, "oos_dml_h18_t21.csv")],                    # the complete-careers partner, 0.3987
    "rx_h18": [(*IT103, "oos_rx_h18_t21.csv")],                      # race_xent, 12,000 rounds, 0.4072
    "rx16_h18": [(*IT103, "oos_rx_h18_t21_r16k.csv")],               # race_xent, 16,000-round cap, 0.4059
    "main_h18_b": [(*IT103, "oos_base.csv")],                        # the main refitted on another processor
}
AVERAGES = {
    "pair97": ["xthub_all3", "dml_all3"],                            # control: the candidate pair (0.3966)
    "pair_h18": ["main_h18", "dml_h18"],                             # 0.3945
    "three16_h18": ["main_h18", "dml_h18", "rx16_h18"],              # 0.3946
    "three_h18": ["main_h18", "dml_h18", "rx_h18"],                  # 0.3950
    "pair_rx16_h18": ["main_h18", "rx16_h18"],
}
OUT = Path("out/pairs_clv_iter103")


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
    ("pair_h18", "pair97"), ("three16_h18", "pair97"), ("three_h18", "pair97"),
    ("pair_rx16_h18", "pair97"), ("rx16_h18", "pair97"),
    ("three16_h18", "pair_h18"), ("rx16_h18", "rx_h18"), ("main_h18_b", "main_h18"),
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
    order = ["pair97", "three16_h18", "three_h18", "pair_rx16_h18"]
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
