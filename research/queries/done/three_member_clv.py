"""The early-price trade for averages of three and four members: does the third member's price gain reach the bets?

The pair the serving note puts next (the extra-trees Huber fit, the 968s5xh, with the
within-race extra-trees fit) prices the development window at 0.3981 and its early-price
rule reads +10.77% in iteration 94's own run. Offline, adding the Huber fit at leaves of
500 (iteration 89's hub500) takes the price to 0.3974 (-0.0007, -0.0010 to -0.0004), and a
fourth member (the within-race Huber fit, iteration 94) to 0.3973. A third booster is
another 40 MB to store and serve, so the owner's choice needs its bets as well as its
price. This runs the fixed early-price rule and the rank-1 settlement (scripts/clv_betfair.py,
holdout excluded) on the offline averages, with the pair recomputed from the same files as
the control (it must reproduce iteration 94's in-run line). Read-only.
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
IT89 = (36295445927, "research-iter89-served-968-slow-leaves-109")
SOURCES = {
    "xthub": (*IT94, "oos_cw_dm_lr02_mcs500_xthub_rf.csv"),
    "xt_dml": (*IT94, "oos_cw_dm_lr02_mcs500_xt_dml_rf.csv"),
    "xthub_dml": (*IT94, "oos_cw_dm_lr02_mcs500_xthub_dml.csv"),
    "hub500": (*IT89, "oos_cw_dm_lr02_mcs500_hub.csv"),
}
AVERAGES = {
    "pair_xthub_xtdml": ["xthub", "xt_dml"],                               # control: iteration 94's in-run pair
    "three_xthub_xtdml_hub500": ["xthub", "xt_dml", "hub500"],             # 0.3974 offline
    "three_xthub_xthubdml_hub500": ["xthub", "xthub_dml", "hub500"],       # 0.3973
    "four_xthub_xtdml_hub500_xthubdml": ["xthub", "xt_dml", "hub500", "xthub_dml"],   # 0.3973
}
OUT = Path("out/three_member_clv")


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
