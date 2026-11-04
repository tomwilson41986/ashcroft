"""Rules at the SP with a price limit (research/greyhound/sp_limit.py; read-only): our out-of-sample prices from
1 Jan 2024 (fast-loop run 41888253783, research/greyhound/experiments/edge_oos.json: the live recipe on log(BSP) and
the softmax win model, each fitted once on 2021-2023), backs and lays at the win and place SP where the SP clears our
fair price by a margin; chosen on 2024, read on 2025-26 with 90% intervals by day."""
import io
import json
import os
import sys
import zipfile

import pandas as pd
import requests

sys.path.insert(0, os.getcwd())
from sources.common import Store                      # noqa: E402
from research.greyhound import sp_limit as sl        # noqa: E402

RUN = 41888253783
tok = os.environ.get("GITHUB_TOKEN")
if not tok:
    raise SystemExit("no data found: no GITHUB_TOKEN to read the artifacts")
H = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
arts = requests.get(f"https://api.github.com/repos/tomwilson41986/ashcroft/actions/runs/{RUN}/artifacts?per_page=50",
                    headers=H, timeout=60).json().get("artifacts", [])
print("artifacts:", [a["name"] for a in arts])
store = Store()
for name in ("edge_logbsp", "edge_softmax"):
    a = next((x for x in arts if x["name"] == f"fastloop-{name}"), None)
    if a is None:
        print(f"\n==== {name}: no data found (no artifact)")
        continue
    z = zipfile.ZipFile(io.BytesIO(requests.get(a["archive_download_url"], headers=H, timeout=300).content))
    f = next((n for n in z.namelist() if n.endswith(f"pred_{name}.parquet")), None)
    if f is None:
        print(f"\n==== {name}: no data found (no predictions in the artifact)")
        continue
    pred = pd.read_parquet(io.BytesIO(z.read(f)))
    print(f"\n==== {name}: {len(pred)} runners, {pred.race_date.min()} to {pred.race_date.max()}")
    rep = sl.run(pred, store)
    print(sl.summary(rep))
    os.makedirs("out", exist_ok=True)
    open(f"out/sp_limit_{name}.json", "w").write(json.dumps(rep, indent=1, default=str))
    print("PASSING:", json.dumps(rep["passing"], indent=1))
