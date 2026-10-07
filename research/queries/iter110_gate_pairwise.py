"""Iteration 110, the blends against each other (read-only): does the in-running block beat the served blend refitted?

The research loop scores every variant against the base main, never one variant against another, so iteration 110's
table puts gate_ir (the block in all three members) at -0.0047 and gate_step (the served blend refitted) at -0.0044
without the paired difference between them. This fetches the iteration's artifact (research loop run 37532787409)
and runs scripts/compare_oos_runs.py with gate_step as the base and each blend with the block as the variant, on the
same 53,910 runners. The morning check's rule: the block goes into training only if the blend with it resolves better
and the seed-7 gain agrees (main_ir7 - base7 = -0.0005 against iteration 109's -0.0008).
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import requests

RUN, NAME = 37532787409, "research-iter110-inrunning-served-blend-134"
token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
if not token or not repo:
    raise SystemExit("no GITHUB_TOKEN: the artifact cannot be read")
auth = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
r = requests.get(f"https://api.github.com/repos/{repo}/actions/runs/{RUN}/artifacts", headers=auth,
                 params={"name": NAME}, timeout=60)
arts = r.json().get("artifacts", []) if r.ok else []
if not arts:
    raise SystemExit(f"artifact {NAME} of run {RUN} not found ({r.status_code})")
z = requests.get(arts[0]["archive_download_url"], headers=auth, timeout=900)
z.raise_for_status()
dest = Path("iter110")
zipfile.ZipFile(io.BytesIO(z.content)).extractall(dest)
files = {p.stem.removeprefix("oos_"): p for p in dest.rglob("oos_*.csv")}
print("OOS files:", sorted(files))
out = Path("out/iter110_gate_pairwise")
out.mkdir(parents=True, exist_ok=True)
for base, variant in (("gate_step", "gate_ir"), ("gate_step", "gate_rx_ir"), ("gate_step", "gate_pair_ir"),
                      ("base7", "main_ir7")):
    if base not in files or variant not in files:
        print(f"\n{variant} against {base}: missing")
        continue
    print(f"\n#### {variant} against {base}", flush=True)
    res = subprocess.run([sys.executable, "scripts/compare_oos_runs.py", "--base", str(files[base]),
                          "--variant", str(files[variant]), "--commission", "0.02",
                          "--out", str(out / f"{variant}_vs_{base}.md")], capture_output=True, text=True)
    md = out / f"{variant}_vs_{base}.md"
    print(md.read_text() if md.exists() else res.stdout[-6000:])
    if res.returncode:
        print(res.stderr[-3000:])
