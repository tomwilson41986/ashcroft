"""The owner, 20 Nov 2026: "improve so we stop losing these fits". A check of the fix on the real matrix. Read-only.

The fits lost with no log die 47-48 minutes into the job whatever they fit (a 9-minute figure fit or a 4-hour
member): "The hosted runner lost communication with the server". A kept fit's memory log shows why (run 44473514515,
fit st fold 5): ten minutes in, reading the cached matrix, RAM is full (15.8 of 16 GB) and 9 GB is in swap, because
pd.read_parquet holds the Arrow table and the frame at once and every block feature is written in double precision.
A runner whose disk is slow stops answering there and GitHub gives it up about 40 minutes later.

The fix (model/feature_cache._read_parquet, blocks.attach single_unless, evaluate_oos.float32_plan): the matrix is
read a row group at a time into its final arrays; with --float32 a feature no block reads is held in single
precision from the start. The values must not change. Checked here on the loop's own cached matrix:

1. the new reader's frame against the fingerprints stored when the matrix was built (one hash a column): equal?
2. a sample of columns read as float32 against the same columns read as float64 and cast: equal?
3. the figure member's fold 0 fitted by the new code (scripts/research_loop.py fit --arm fig --fold 0, the branch's
   loop.json) against iteration 131's own fold 0 (run 44549603086, the old code): the predictions, and the job's
   memory peak (RAM used plus swap, sampled every 5 seconds) against the old code's 22-25 GB.

No model is chosen and nothing is scored against a result: not a trial.
"""
from __future__ import annotations

import io
import json
import os
import re
import resource
import subprocess
import sys
import threading
import time
import zipfile

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.getcwd())
from model import feature_cache as fc  # noqa: E402

RUN = 44549603086
KEY = ["race_date", "race_time", "track", "horse_name"]


def say(*a):
    print(*a, flush=True)


def free_mb():
    out = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout.splitlines()
    mem = [int(x) for x in out[1].split()[1:]]
    swap = [int(x) for x in out[2].split()[1:]]
    return mem[1], swap[1], mem[0], swap[0]                 # used, swap used, total, swap total


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.peak_used, self.peak_swap, self.stop = 0, 0, 0, False

    def run(self):
        while not self.stop:
            used, swap, _, _ = free_mb()
            if used + swap > self.peak:
                self.peak, self.peak_used, self.peak_swap = used + swap, used, swap
            time.sleep(5)


used0, swap0, total, swap_total = free_mb()
say(f"runner: {total:,} MB RAM, {swap_total:,} MB swap; in use at the start {used0:,} MB, swap {swap0:,} MB")
key = fc.cache_key("horse_racing.db", "2018-01-01")
data, meta = fc._paths(".feature_cache", key)
if not (data.exists() and meta.exists()):
    raise SystemExit(f"no data found: the loop's matrix (key {key}) is not in this job's cache")
info = json.loads(meta.read_text())
say(f"matrix {data.name}: {info.get('rows'):,} rows x {info.get('columns'):,} columns, built {info.get('built_at')}, "
    f"{data.stat().st_size / 1e9:.2f} GB on disk")

# ---- 1. the new reader against the stored fingerprints
s = Sampler(); s.start()
t0 = time.time()
df = fc._read_parquet(data)
s.stop = True; s.join()
say(f"\n1. read by row group in {time.time() - t0:.0f}s: {len(df):,} x {len(df.columns):,}; peak RAM + swap "
    f"{s.peak:,} MB (RAM {s.peak_used:,}, swap {s.peak_swap:,}); the frame {df.memory_usage(deep=False).sum() / 1e9:.1f} GB; "
    f"dtypes {df.dtypes.astype(str).value_counts().to_dict()}")
stored = info.get("column_fingerprints") or {}
got = fc.column_fingerprints(df)
floats = [c for c in df.columns if df[c].dtype == np.float64]
# the floating-point columns are the ones the new reader fills itself; every other column is still pandas's own read
differ = [c for c in stored if c in got and got[c] != stored[c] and str(df[c].dtype).startswith("float")]
other = [c for c in stored if c in got and got[c] != stored[c] and not str(df[c].dtype).startswith("float")]
absent = [c for c in stored if c not in got]
say(f"   fingerprints: {len(stored):,} stored, {len(got):,} read; floating-point columns differing {len(differ)}, "
    f"absent {len(absent)}" + (f"; first differing {differ[:10]}" if differ else "")
    + f"; other columns whose hash moved in the file's round trip (read by pandas as before): {len(other)} {other[:8]}")
rng = np.random.default_rng(20261120)
sample = sorted(rng.choice(floats, size=min(40, len(floats)), replace=False).tolist())
want = {c: df[c].to_numpy().astype(np.float32) for c in sample}
del df
fc.release_memory()

# ---- 2. columns read as float32 against float64 cast once
single = fc._read_parquet(data, columns=sample, float32=sample)
bad = [c for c in sample if single[c].dtype != np.float32 or not np.array_equal(single[c].to_numpy(), want[c], equal_nan=True)]
say(f"\n2. {len(sample)} float64 columns read as float32 against the same read as float64 and cast: differing {len(bad)}"
    + (f" {bad[:10]}" if bad else ""))
del single, want
fc.release_memory()

# ---- 3. the figure member's fold 0 by the new code, and its memory
used1, swap1, _, _ = free_mb()
s = Sampler(); s.start()
t0 = time.time()
r = subprocess.run([sys.executable, "scripts/research_loop.py", "fit", "--arm", "fig", "--fold", "0"],
                   capture_output=True, text=True)
s.stop = True; s.join()
tail = [ln for ln in (r.stdout + r.stderr).splitlines() if "PerformanceWarning" not in ln]
say(f"\n3. fit fig fold 0 by the new code: exit {r.returncode} in {(time.time() - t0) / 60:.1f} min; peak RAM + swap "
    f"{s.peak:,} MB (RAM {s.peak_used:,}, swap {s.peak_swap:,}; before the fit {used1:,} and {swap1:,}); the child's "
    f"largest resident size {resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6:.1f} GB")
for ln in tail:
    if re.search(r"float32|Feature cache hit|block \w+ built|features, |Fold \d|rows x", ln):
        say("   " + ln[-200:])
if r.returncode != 0:
    say("\n".join(tail[-30:]))
    raise SystemExit("the fit failed")
new = pd.read_csv("reports/fit/oos_fig_f0.csv", dtype={"race_time": str})

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
hit = [a for a in arts if re.fullmatch(r"fit-\d+-fig-0", a["name"]) and not a.get("expired")]
if not hit:
    raise SystemExit(f"no data found: iteration 131's fig fold 0 artifact is not among {len(arts)} artifacts")
z = requests.get(hit[0]["archive_download_url"], headers=hd, timeout=600)
with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
    name = next(n for n in zf.namelist() if re.search(r"oos_.*\.csv$", n))
    old = pd.read_csv(io.BytesIO(zf.read(name)), dtype={"race_time": str})
say(f"   iteration 131's fold 0 ({hit[0]['name']}, {name}): {len(old):,} rows; the new code's: {len(new):,} rows")
m = old.merge(new, on=KEY, suffixes=("_old", "_new"), how="outer", indicator=True)
both = m[m["_merge"] == "both"]
d = (both["predicted_figure_new"] - both["predicted_figure_old"]).abs()
say(f"   paired {len(both):,} (only old {int((m['_merge'] == 'left_only').sum())}, only new "
    f"{int((m['_merge'] == 'right_only').sum())}); predicted figure identical on {int((d == 0).sum()):,}, "
    f"largest difference {d.max():.3g}, mean {d.mean():.3g} (the figure's sd {both['predicted_figure_old'].std():.3f})")
ok = not differ and not absent and not bad and len(both) == len(old) == len(new)
say(f"\nverdict: the matrix {'is' if ok else 'IS NOT'} the same column for column; peak memory of a fit "
    f"{s.peak / 1000:.1f} GB against 22-25 GB before")
