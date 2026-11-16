"""The candidate (the three members retrained with the market rating: train-bfsp runs 60, 61, 62) priced beside the
served model each morning from 12 Nov 2026 (the BFSP plan, step A: a week of shadow pricing; read-only).

Each morning the watchdog dispatches predict-now.yml with the candidate (an artifact only: nothing is served). Here
every such run's predictions (the 'predictions-now' artifact, read with the job's token) are set beside the served
06:00 file (S3 predictions/<day>.csv) on the runners both priced, against the BSP (betfair_live_marks' final mark):
the error of log(price / BSP) with each race centred (what the closing model reads), the paired change by race with
its se, the rank correlation, by day and by code, and whether the candidate's moves from the served price went the
BSP's way. Copy to research/queries/ to run; days without results yet are left out.
"""
import io
import os
import sqlite3
import sys
import zipfile

import boto3
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, ".")
from betfair_prices import normalise_horse  # noqa: E402

pd.set_option("display.width", 220)
FROM = "2026-11-12"
CAND_RUN = "42980241280"                       # the candidate's main member: a shadow run names it in its inputs
token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
if not token or not repo:
    raise SystemExit("no GITHUB_TOKEN: cannot read the shadow runs' artifacts")
H = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
api = f"https://api.github.com/repos/{repo}"
runs = requests.get(f"{api}/actions/workflows/predict-now.yml/runs", headers=H,
                    params={"created": f">={FROM}", "per_page": 100}, timeout=60).json().get("workflow_runs", [])
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
frames = []
for r in sorted(runs, key=lambda x: x["created_at"]):
    if r.get("conclusion") != "success":
        print(f"  run {r['id']} {r['created_at']}: {r.get('conclusion') or r.get('status')}, left out")
        continue
    arts = requests.get(f"{api}/actions/runs/{r['id']}/artifacts", headers=H, timeout=60).json().get("artifacts", [])
    a = next((x for x in arts if x["name"] == "predictions-now"), None)
    if a is None:
        continue
    z = zipfile.ZipFile(io.BytesIO(requests.get(a["archive_download_url"], headers=H, timeout=120).content))
    log = z.read("predict.log").decode("utf-8", "replace") if "predict.log" in z.namelist() else ""
    if "predictions.csv" not in z.namelist() or "1017 features" not in log:
        print(f"  run {r['id']} {r['created_at']}: not a candidate run (no 1017-feature model in its log), left out")
        continue
    c = pd.read_csv(z.open("predictions.csv"))
    day = str(c.race_date.iloc[0])[:10]
    try:
        am = pd.read_csv(io.BytesIO(s3.get_object(Bucket=bucket, Key=f"predictions/{day}.csv")["Body"].read()))
    except Exception as e:  # noqa: BLE001
        print(f"  {day}: no served file ({type(e).__name__})")
        continue
    hcol = "horse_name" if "horse_name" in am.columns else "runner_name"
    am["h"] = am[hcol].map(normalise_horse)
    c["h"] = c.horse_name.map(normalise_horse)
    m = c[["h", "track", "race_time", "race_name", "predicted_bfsp"]].rename(columns={"predicted_bfsp": "cand"}).merge(
        am.drop_duplicates("h")[["h", "predicted_bfsp"]].rename(columns={"predicted_bfsp": "served"}), on="h")
    bsp = pd.read_sql_query("""SELECT k.horse_norm AS h, MAX(f.bsp) AS bsp FROM betfair_live_markets k
                               JOIN betfair_live_marks f ON f.market_id = k.market_id AND f.selection_id = k.selection_id
                               WHERE k.race_date = ? AND f.mark = 'final' AND f.bsp > 1 GROUP BY k.horse_norm""",
                            conn, params=[day])
    m = m.merge(bsp, on="h")
    m = m[(m.cand > 1) & (m.served > 1)]
    m["day"] = day
    print(f"  {day} (run {r['id']}): candidate {len(c)}, served {len(am)}, both with a BSP {len(m)}")
    frames.append(m)
if not frames:
    raise SystemExit("no shadow day with results yet")
m = pd.concat(frames).drop_duplicates(["day", "h"], keep="last")
m["race"] = m.day + "|" + m.track + "|" + m.race_time.astype(str)
m["code"] = np.where(m.race_name.astype(str).str.contains("Hurdle|Chase|Bumper|NH Flat|National Hunt|Hunters", case=False),
                     "jumps", "flat/aw")
lb = np.log(m.bsp)
for c in ("cand", "served"):
    e = np.log(m[c]) - lb
    m[f"ec_{c}"] = (e - e.groupby(m.race).transform("mean")).abs()
m["d"] = m.ec_cand - m.ec_served


def row(g):
    per = g.groupby("race").d.mean()
    return pd.Series({"runners": len(g), "races": len(per), "served": g.ec_served.mean(), "candidate": g.ec_cand.mean(),
                      "change": per.mean(), "se": per.std(ddof=1) / np.sqrt(len(per)) if len(per) > 1 else np.nan,
                      "rank_served": np.log(g.served).corr(lb[g.index], method="spearman"),
                      "rank_cand": np.log(g.cand).corr(lb[g.index], method="spearman")})


m = m.reset_index(drop=True)
lb = np.log(m.bsp)
print(f"\n{m.day.nunique()} days, {m.race.nunique()} races, {len(m)} runners; change < 0: the candidate closer to the BSP")
print(pd.concat({"all": row(m), **{k: row(g) for k, g in m.groupby("code")}}, axis=1).T.round(4).to_string())
print("\nby day:")
print(pd.concat({k: row(g) for k, g in m.groupby("day")}, axis=1).T.round(4).to_string())
mv = np.log(m.cand / m.served)
toward = np.sign(mv) == np.sign(np.log(m.bsp / m.served))
for lo in (0.0, 0.05, 0.1):
    s = mv.abs() > lo
    print(f"moves over {lo:.0%}: {int(s.sum())} runners, toward the BSP {toward[s].mean():.1%}")
