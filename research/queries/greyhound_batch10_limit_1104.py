"""Batch 10's limit-order study per group of tracks (the owner, 4 Nov: "evaluate if we can cover other tracks with our
model or just leave them alone"; read-only). Fast-loop run 41845416756's wpred_groups artifact: each group's report
(the eight tracks closest to the BSP, the eight furthest), its header and its best 15 rules, as the study ranks them."""
import io
import os
import zipfile

import requests

RUN_ARTIFACT = 13336457353                                 # fastloop-wpred_groups, run 41845416756
tok = os.environ.get("GITHUB_TOKEN")
if not tok:
    raise SystemExit("no data found: no GITHUB_TOKEN to read the artifact")
r = requests.get(f"https://api.github.com/repos/tomwilson41986/ashcroft/actions/artifacts/{RUN_ARTIFACT}/zip",
                 headers={"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}, timeout=120)
r.raise_for_status()
z = zipfile.ZipFile(io.BytesIO(r.content))
names = sorted(n for n in z.namelist() if "limit_" in n and n.endswith(".md"))
print("limit reports in the artifact:", names or "no data found")
for n in names:
    text = z.read(n).decode("utf-8", "replace").splitlines()
    print(f"\n==== {n} ({len(text)} lines)")
    table_at = next((i for i, line in enumerate(text) if line.startswith("|")), None)
    head = text[:table_at] if table_at is not None else text[:40]
    print("\n".join(head[:40]))
    if table_at is not None:
        print("\n".join(text[table_at:table_at + 17]))           # the header, the rule and the best 15
