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
import re
for n in names:
    text = z.read(n).decode("utf-8", "replace").splitlines()
    print(f"\n==== {n}")
    print(next((line for line in text if line.startswith("Our price")), ""))
    at = next((i for i, line in enumerate(text) if line.startswith("## 2. Holdout")), None)
    end = next((i for i, line in enumerate(text) if line.startswith("## 2b") or line.startswith("## 3.")), len(text))
    if at is None:
        print("no data found: no holdout section")
        continue
    sec = text[at:end]
    print("\n".join(line for line in sec if "rules read" in line))
    rows = [line for line in sec if line.startswith("|")]
    print(rows[0] if rows else "")
    above = [line for line in rows[2:] if (m := re.search(r"\[(-?[\d.]+), (-?[\d.]+)\]", line))
             and float(m.group(1)) > 0]
    print(f"holdout rules with the 90% lower bound above zero: {len(above)}")
    print("\n".join(above) if above else "(none)")
    early = [line for line in text if line.startswith("| first_traded | bid | lay | 0.1 | all |")
             or line.startswith("| first_traded | bid | lay | 0.2 | all |")]
    eh = next((line for line in text if line.startswith("| mark | order | side | param | gate | fill | dev_sp")), "")
    print("\nthe early lay (first traded, lay at the forecast less the margin, every runner):")
    print(eh)
    print("\n".join(early))
