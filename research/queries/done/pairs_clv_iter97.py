"""The early-price trade for the pair iteration 97 found: the 968s5xh with the three blocks beside the partner with them.

Iteration 97 fitted the within-race partner with travel, the Kalman rating and handicap angles: alone 0.4013 against
the trained partner's 0.4027 (concordance resolved better). Beside iteration 96's 968s5xh with the three blocks (seed
42, 0.3984) it prices the window at 0.3966 against the pair's 0.3981; beside the old partner 0.3970; the partner with
travel alone 0.3967. The in-run pair at seed 7 (0.3969) kept the bets (the rule +10.73%, the top pick traded out
+1.88%, the weight 0.464). This runs the fixed early-price rule and the rank-1 settlement (scripts/clv_betfair.py,
holdout excluded) on the seed-42 pairs that would serve, with the current pair recomputed as the control (iteration
94's in-run line: +10.77%, +1.72%). Read-only.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from research_loop import average_arms  # noqa: E402

IT94 = (36331536686, "research-iter94-served-968-slow-dml-partners-114")
IT96 = (36346975763, "research-iter96-served-968s5xh-kalman-handicap-travel-116")
IT97 = (36358454411, "research-iter97-served-968s5xh-travel-partners-117")
SOURCES = {
    "xthub": (*IT94, "oos_cw_dm_lr02_mcs500_xthub_rf.csv"),
    "xt_dml": (*IT94, "oos_cw_dm_lr02_mcs500_xt_dml_rf.csv"),
    "xthub_all3": (*IT96, "oos_cw_dm_xthub_kr_hca_tv.csv"),
    "dml_all3": (*IT97, "oos_cw_dm_xt_dml_kr_hca_tv.csv"),
    "dml_tv": (*IT97, "oos_cw_dm_xt_dml_tv.csv"),
}
AVERAGES = {
    "pair_xthub_xtdml": ["xthub", "xt_dml"],                  # control: iteration 94's in-run pair (0.3981)
    "pair_all3_dmlall3": ["xthub_all3", "dml_all3"],           # 0.3966 offline: the candidate
    "pair_all3_dmltv": ["xthub_all3", "dml_tv"],               # 0.3967
    "pair_all3_xtdml": ["xthub_all3", "xt_dml"],               # 0.3970 (iteration 96's line: +10.73%, +1.91%)
}
OUT = Path("out/pairs_clv_iter97")


def fetch(run: int, artifact: str, member: str, dest: Path, cache: dict) -> Path:
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
    zf = cache[(run, artifact)]
    name = next(n for n in zf.namelist() if n.endswith(member))
    path = dest / member
    path.write_bytes(zf.read(name))
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    files = {arm: fetch(run, art, member, OUT, cache) for arm, (run, art, member) in SOURCES.items()}
    for name, arms in AVERAGES.items():
        files[name] = OUT / f"oos_{name}.csv"
        average_arms([str(files[a]) for a in arms], str(files[name]))
    for arm, path in files.items():
        res = subprocess.run([sys.executable, "scripts/clv_betfair.py", "--predictions", str(path),
                              "--db", "horse_racing.db", "--until", "2026-04-01"], capture_output=True, text=True)
        print(f"\n===== {arm} =====")
        print(res.stdout)
        if res.returncode:
            print(res.stderr[-2000:])


if __name__ == "__main__":
    main()
