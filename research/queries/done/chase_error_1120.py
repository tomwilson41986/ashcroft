"""The owner, 20 Nov 2026: "try and improve the chase error". Step 1, the diagnosis. Read-only; descriptive.

Live, the model's price is furthest from the Betfair SP, against what the morning market already knows, in chases:
0.360 against the morning price's 0.249 (ledger model-qa-1113). Where in chases is that gap? The served stack
blend's out-of-sample prices (iteration 130, run 44473514515, oos_gate_st.csv; 2024-12-28 to 2026-03-31; the locked
holdout not read) joined to Betfair's morning WAP and SP. Per runner: the model's error |log(price / BSP)|, the
morning price's error |log(WAP / BSP)|, their difference (the gap: what the market knows at breakfast that the
model does not), and the model's signed error (its bias). By code first; then, chases only, by the things a model
could be missing:

    the race        handicap or not, novice/beginners, field size, trip, going, class
    the price       the BSP band; the morning's matched money
    the horse       chase runs before today (0, 1-2, 3-5, 6+); how its last run ended (finished, fell or unseated,
                    pulled up, no earlier run); days since it; whether its last run was over hurdles or on the
                    Flat (a switch of code); runs this season; its age
    the race's mix  the share of the field with under 3 chase runs

Each: runners, both errors, the gap with a 90% race-day interval, the bias, and the segment's share of the chase
gap (gap x runners), so the largest contributors are on top. Hurdles are printed beside for the horse segments.

Descriptive: it says where to look. A fix (a block, a member) is registered before it is fitted. Not a trial.
"""
from __future__ import annotations

import io
import os
import sqlite3
import sys
import zipfile

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.getcwd())
from model.race_shape import race_code  # noqa: E402
from scripts import clv_betfair as cb  # noqa: E402

RUN, PREFIX, FILE = 44473514515, "research-iter130", "oos_gate_st.csv"
UNTIL = "2026-04-01"
KEY = ["race_date", "track", "race_time", "horse_name"]
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
rng = np.random.default_rng(20261120)


def say(*a):
    print(*a, flush=True)


def fetch(path):
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
    for a in [a for a in arts if a["name"].startswith(PREFIX)]:
        z = requests.get(a["archive_download_url"], headers=hd, timeout=900)
        with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
            hit = [n for n in zf.namelist() if n.endswith(FILE)]
            if hit:
                open(path, "wb").write(zf.read(hit[0]))
                return a["name"]
    raise SystemExit(f"no data found: {FILE} not in run {RUN}'s {PREFIX}* artifacts")


def mean_ci(x, day, draws=2000):
    x = np.asarray(x, float)
    days, inv = np.unique(np.asarray(day), return_inverse=True)
    sn, sd = np.bincount(inv, x, len(days)), np.bincount(inv, np.ones_like(x), len(days))
    idx = rng.integers(0, len(days), size=(draws, len(days)))
    bs = sn[idx].sum(1) / np.maximum(sd[idx].sum(1), 1e-12)
    return x.mean(), float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))


def table(d, seg, total_gap=None, min_n=150):
    rows = []
    for v, g in d.groupby(seg, observed=True):
        if len(g) < min_n:
            continue
        gap, lo, hi = mean_ci(g["gap"], g["race_date"])
        rows.append({seg: v, "runners": len(g), "model": g["e_model"].mean(), "morning": g["e_morn"].mean(),
                     "gap": gap, "lo": lo, "hi": hi, "bias": g["signed"].mean(),
                     "share%": 100 * g["gap"].sum() / total_gap if total_gap else np.nan})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    return t.sort_values("share%" if total_gap else "runners", ascending=False).round(
        {"model": 4, "morning": 4, "gap": 4, "lo": 4, "hi": 4, "bias": 3, "share%": 1})


def show(t, title):
    say(f"\n-- {title}")
    say("   no segment with enough runners" if t.empty else t.to_string(index=False))


# ---- the predictions and Betfair's prices
os.makedirs("out", exist_ok=True)
pred = "out/oos_gate_st_130.csv"
art = fetch(pred)
d = cb.load(pred, "horse_racing.db", None, until=UNTIL)
d["e_model"] = np.abs(np.log(d["predicted_bfsp"] / d["bsp"]))
d["e_morn"] = np.abs(np.log(d["morningwap"] / d["bsp"]))
d["gap"] = d["e_model"] - d["e_morn"]
d["signed"] = np.log(d["predicted_bfsp"] / d["bsp"])
d["code"] = race_code(d).to_numpy()
say(f"{art} {FILE}: {len(d):,} runners in {d['race'].nunique():,} races with a morning WAP and a BSP "
    f"({d['race_date'].min()} to {d['race_date'].max()})")
show(table(d, "code"), "by code: the model's error, the morning price's, the gap (model less morning), the model's bias")

# ---- each horse's history from the results (every run, earlier days only)
con = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
cols = [r[1] for r in con.execute("PRAGMA table_info(race_results)")]
text = next((c for c in ("placing", "finish_position", "position", "pos") if c in cols), None)
want = ["race_date", "race_time", "track", "horse_name", "race_type", "surface_type", "placing_numerical"] + \
       ([text] if text else []) + [c for c in ("horse_age",) if c in cols]
h = pd.read_sql_query(f"SELECT {', '.join(want)} FROM race_results WHERE race_date < ? AND race_date >= '2015-01-01'",
                      con, params=(UNTIL,))
con.close()
say(f"\nrace_results: {len(h):,} runs from 2015 for the histories; the placing code column: {text or 'none found'}")
h["race_date"] = h["race_date"].astype(str).str[:10]
h["code"] = race_code(h).to_numpy()
h = h.sort_values(["horse_name", "race_date", "race_time"]).reset_index(drop=True)
g = h.groupby("horse_name", sort=False)
h["is_chase"] = (h["code"] == "chase").astype(int)
h["chase_before"] = g["is_chase"].cumsum() - h["is_chase"]
h["runs_before"] = g.cumcount()
h["last_code"] = g["code"].shift(1)
h["last_date"] = g["race_date"].shift(1)
fin = pd.to_numeric(h["placing_numerical"], errors="coerce")
if text:
    code_txt = h[text].fillna("").astype(str).str.upper().str.strip()
    how = np.where(fin > 0, "finished", np.where(code_txt.str.contains(r"^(?:F|UR|U|BD|B)$"), "fell/unseated",
                   np.where(code_txt.str.contains(r"^(?:PU|P|RO|R|RR|CO|SU|S)$"), "pulled up/other", "did not finish")))
else:
    how = np.where(fin > 0, "finished", "did not finish")
h["how"] = how
h["last_how"] = g["how"].shift(1)
h["dsl"] = (pd.to_datetime(h["race_date"]) - pd.to_datetime(h["last_date"])).dt.days
season = pd.to_datetime(h["race_date"])
h["season"] = np.where(season.dt.month >= 5, season.dt.year, season.dt.year - 1)       # a jumps season from May
h["season_runs_before"] = h.groupby(["horse_name", "season"], sort=False).cumcount()
h["green"] = (h["chase_before"] < 3).astype(float)
keep = KEY + ["chase_before", "runs_before", "last_code", "last_how", "dsl", "season_runs_before", "green"] + \
       [c for c in ("horse_age",) if c in h.columns]
d["race_date"] = d["race_date"].astype(str).str[:10]
d = d.merge(h[keep].drop_duplicates(KEY), on=KEY, how="left")
say(f"joined to a history row: {int(d['runs_before'].notna().sum()):,} of {len(d):,}")

d["chase runs before"] = pd.cut(d["chase_before"], [-1, 0, 2, 5, 999], labels=["0", "1-2", "3-5", "6+"])
d["last run ended"] = d["last_how"].fillna("no earlier run")
d["days since"] = pd.cut(d["dsl"], [-1, 14, 30, 60, 150, 365, 99999],
                         labels=["0-14", "15-30", "31-60", "61-150", "151-365", "366+"]).astype(object).where(
                             d["dsl"].notna(), "no earlier run")
d["last run's code"] = d["last_code"].fillna("no earlier run")
d["season runs before"] = pd.cut(d["season_runs_before"], [-1, 0, 1, 3, 99], labels=["0 (seasonal debut)", "1", "2-3", "4+"])
rt = d["race_type"].fillna("").astype(str).str.lower()
d["kind"] = np.where(rt.str.contains("handicap"), np.where(rt.str.contains("novice|beginner"), "novice handicap", "handicap"),
                     np.where(rt.str.contains("novice|beginner|maiden"), "novice/beginners", "other non-handicap"))
d["field"] = pd.cut(d["number_of_runners"], [0, 4, 7, 11, 99], labels=["2-4", "5-7", "8-11", "12+"])
d["bsp band"] = pd.cut(d["bsp"], [1, 3, 6, 12, 25, 50, 1e9], labels=["<3", "3-6", "6-12", "12-25", "25-50", "50+"])
d["morning money"] = pd.cut(d["morning_vol"].fillna(0), [-1, 100, 500, 2000, 1e12], labels=["<100", "100-500", "500-2k", "2k+"])
d["trip"] = pd.cut(pd.to_numeric(d.get("dist_furlongs"), errors="coerce"), [0, 17.5, 21.5, 25.5, 99],
                   labels=["to 2m1f", "2m2f-2m5f", "2m6f-3m1f", "3m2f+"])
d["going"] = d.get("going_description", pd.Series("?", index=d.index)).fillna("?").astype(str)
d["class"] = d.get("race_class", pd.Series("?", index=d.index)).astype(str)
d["green share"] = pd.cut(d.groupby("race")["green"].transform("mean"), [-0.01, 0.25, 0.5, 0.75, 1.0],
                          labels=["0-25%", "25-50%", "50-75%", "75-100%"])
if "horse_age" in d.columns:
    d["age"] = pd.cut(pd.to_numeric(d["horse_age"], errors="coerce"), [0, 5, 6, 7, 9, 99], labels=["to 5", "6", "7", "8-9", "10+"])

ch = d[d["code"] == "chase"].copy()
total = ch["gap"].sum()
gap, lo, hi = mean_ci(ch["gap"], ch["race_date"])
say(f"\n=== chases: {len(ch):,} runners in {ch['race'].nunique():,} races; the model {ch['e_model'].mean():.4f}, the "
    f"morning price {ch['e_morn'].mean():.4f}, the gap {gap:.4f} ({lo:.4f} to {hi:.4f}); the model's bias "
    f"{ch['signed'].mean():+.3f} (negative: our price shorter than the BSP)")
segs = ["kind", "field", "bsp band", "morning money", "trip", "going", "class", "chase runs before", "last run ended",
        "days since", "last run's code", "season runs before", "green share"] + (["age"] if "age" in ch.columns else [])
for seg in segs:
    show(table(ch, seg, total_gap=total), f"chases by {seg} (share% = the segment's part of the whole chase gap)")

hu = d[d["code"] == "hurdle"].copy()
say(f"\n=== hurdles beside them: {len(hu):,} runners; the model {hu['e_model'].mean():.4f}, the morning price "
    f"{hu['e_morn'].mean():.4f}, the gap {hu['gap'].mean():.4f}")
for seg in ("kind", "last run ended", "days since", "season runs before"):
    show(table(hu, seg, total_gap=hu["gap"].sum()), f"hurdles by {seg}")

# ---- the direction of the miss in chases: does the BSP move toward or away from our price after the morning?
ch["toward"] = np.sign(np.log(ch["morningwap"] / ch["predicted_bfsp"])) == np.sign(np.log(ch["morningwap"] / ch["bsp"]))
moved = ch[np.abs(np.log(ch["morningwap"] / ch["bsp"])) > 0.05]
say(f"\nchases, runners whose price moved over 5% from the morning to the BSP: {len(moved):,}; moved toward our "
    f"price on {100 * moved['toward'].mean():.1f}% (hurdles "
    f"{100 * (np.sign(np.log(hu['morningwap'] / hu['predicted_bfsp'])) == np.sign(np.log(hu['morningwap'] / hu['bsp'])))[np.abs(np.log(hu['morningwap'] / hu['bsp'])) > 0.05].mean():.1f}%)")
