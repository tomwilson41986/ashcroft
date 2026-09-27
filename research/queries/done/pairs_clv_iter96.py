"""The early-price trade for the pairs iteration 96 made possible: does travel's price gain reach the bets?

The 968s5xh with travel prices the development window at 0.3989 and with travel, the Kalman
rating and handicap angles at 0.3984 (against 0.4001); averaged with the within-race partner
(which reads none of them) the pairs price 0.3975 and 0.3970 against the pair's 0.3981. The
single fits' own lines are in their run (the rule +10.05% and +10.16%, the top pick traded out
+1.56% and +1.57%, against the 968s5xh's +10.16% and +1.78%). This runs the fixed early-price
rule and the rank-1 settlement (scripts/clv_betfair.py, holdout excluded) on the offline pairs,
with the current pair recomputed as the control (iteration 94's in-run line: +10.77%, +1.72%).
Read-only.
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
SOURCES = {
    "xthub": (*IT94, "oos_cw_dm_lr02_mcs500_xthub_rf.csv"),
    "xt_dml": (*IT94, "oos_cw_dm_lr02_mcs500_xt_dml_rf.csv"),
    "xthub_tv": (*IT96, "oos_cw_dm_xthub_tv.csv"),
    "xthub_all3": (*IT96, "oos_cw_dm_xthub_kr_hca_tv.csv"),
}
AVERAGES = {
    "pair_xthub_xtdml": ["xthub", "xt_dml"],                  # control: iteration 94's in-run pair
    "pair_xthubtv_xtdml": ["xthub_tv", "xt_dml"],             # 0.3975 offline
    "pair_xthuball3_xtdml": ["xthub_all3", "xt_dml"],         # 0.3970
}
OUT = Path("out/pairs_clv_iter96")


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
