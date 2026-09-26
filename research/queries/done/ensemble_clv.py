"""The early-price trade for the 958 averaged with its Huber twin: does a two-loss average carry into the bets?

Offline, on the development window's walk-forward forecasts (every model fitted at the served
recipe on the same folds), the geometric mean of the 958 under squared error (iteration 68's cw
arm) and the 958 under the Huber loss (iteration 76's cw_huber) prices the BSP better than
either: -0.0026 (-0.0030 to -0.0022) against the 958, rank 1 -0.0011, Brier skill against the
market +0.0006 and concordance +0.0016, both resolved better. With the Huber model that also
reads the debut market block (iteration 80's cw_dm_huber) the average reads -0.0045. The price
error is not the bet: this runs the fixed early-price rule and the rank-1 settlement
(scripts/clv_betfair.py, holdout excluded) on each member and each average. Read-only.
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

SOURCES = {
    "cw": (36233732671, "research-iter68-served-944-connection-windows-84", "oos_cw.csv"),
    "cw_huber": (36248082126, "research-iter76-served-958-huber-93", "oos_cw_huber.csv"),
    "cw_dm_huber": (36253220334, "research-iter80-served-958h-debut-market-97", "oos_cw_dm_huber.csv"),
}
AVERAGES = {"avg_l2_huber": ["cw", "cw_huber"], "avg_l2_dmhuber": ["cw", "cw_dm_huber"]}
OUT = Path("out/ensemble_clv")


def fetch(run: int, artifact: str, member: str, dest: Path) -> Path:
    import requests
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    auth = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    r = requests.get(f"https://api.github.com/repos/{repo}/actions/runs/{run}/artifacts",
                     headers=auth, params={"name": artifact}, timeout=60)
    arts = r.json().get("artifacts", []) if r.ok else []
    if not arts:
        raise SystemExit(f"artifact {artifact} of run {run} not found ({r.status_code})")
    z = requests.get(arts[0]["archive_download_url"], headers=auth, timeout=600)
    z.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(z.content))
    name = next(n for n in zf.namelist() if n.endswith(member))
    path = dest / f"oos_{Path(member).stem.removeprefix('oos_')}.csv"
    path.write_bytes(zf.read(name))
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = {arm: fetch(run, art, member, OUT) for arm, (run, art, member) in SOURCES.items()}
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
