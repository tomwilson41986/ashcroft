"""Race names: what the conditions say that the model does not read, and whether its price misses by them.

The served model reads the race's type (race_type_cat: Maiden, Novices, Handicap, Claimer, Seller...),
its class and its top rating, but not its name, where the conditions that shape the market for
unexposed horses are written: auction and median-auction races (cheaper stock), sales races, EBF and
Plus 10 eligibility, apprentice, amateur and conditional riders, fillies' and mares' races, black type.
This counts each condition over the matrix years and, on the development window, reads the 968
recipe's out-of-sample forecast (iteration 81's cw_dm arm) against the BSP by condition: the error
ln(forecast / BSP) less its race's mean (a race-level condition cannot move a forecast normalised in
its race except through how it prices the runners against each other), by career stage, within race
type, with a race-bootstrap interval. A condition on which that error is biased is information the
name carries and the features do not. Read-only; nothing after 2026-03-31 is read.
"""

from __future__ import annotations

import io
import os
import re
import sqlite3
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

RUN, ARTIFACT, ARM = 36258512493, "research-iter81-served-958-seeds-99", "cw_dm"
START, END = "2021-01-01", "2026-04-01"
KEY = ["race_date", "race_time", "track", "horse_name"]

CONDITIONS = {
    "median": r"median",
    "auction": r"\bauction\b|\bauct\b",
    "sales": r"\bsales?\b",
    "ebf": r"\bebf\b|e\.b\.f|european breeders|\bbss\b|british stallion studs",
    "plus10": r"plus ?10|\+ ?10\b",
    "bonus": r"bonus|\bgbb\b",
    "apprentice": r"apprentice|\bapp\b",
    "amateur": r"amateur|gentlemen|lady riders|\bladies\b|fegentri|\bqr\b",
    "conditional": r"conditional",
    "hands_heels": r"hands (?:and|&) heels",
    "fillies": r"fillies|\bfilly\b",
    "mares": r"\bmares?'?\b",
    "colts_geldings": r"\bcolts?\b|\bgeldings?\b|c ?& ?g\b",
    "black_type": r"\b(?:group|grade)\s*[123]\b|\bg[123]\b|\blisted\b",
    "conditions": r"\bconditions\b",
    "classified": r"classified",
    "rated": r"\brated\b",
    "qualifier": r"qualif",
    "final": r"\bfinal\b",
    "series": r"series|championship|challenge",
    "novice": r"novice",
    "maiden": r"maiden",
    "nursery": r"nursery",
    "juvenile": r"juvenile|\b2yo\b|two.year.old|2-y-o",
    "beginners": r"beginners",
    "introductory": r"introductory",
    "restricted": r"restricted",
    "premier": r"premier",
    "veterans": r"veteran",
    "hunters": r"hunters?'?\s+chase|point.to.point",
    "handicap": r"handicap|\bh'cap\b|\bhcap\b",
    "selling": r"selling|seller",
    "claiming": r"claiming|claimer",
}


def fetch_predictions() -> pd.DataFrame:
    import requests
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    auth = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    r = requests.get(f"https://api.github.com/repos/{repo}/actions/runs/{RUN}/artifacts",
                     headers=auth, params={"name": ARTIFACT}, timeout=60)
    arts = r.json().get("artifacts", []) if r.ok else []
    if not arts:
        raise SystemExit(f"artifact {ARTIFACT} of run {RUN} not found ({r.status_code})")
    z = requests.get(arts[0]["archive_download_url"], headers=auth, timeout=600)
    z.raise_for_status()
    dest = Path("candidates/it81")
    zipfile.ZipFile(io.BytesIO(z.content)).extractall(dest)
    return pd.read_csv(dest / f"oos_{ARM}.csv", dtype={"race_time": str})


def flags(names: pd.Series) -> pd.DataFrame:
    t = names.fillna("").astype(str).str.lower()
    return pd.DataFrame({c: t.str.contains(rx, regex=True).to_numpy() for c, rx in CONDITIONS.items()},
                        index=names.index)


def boot_ci(values: np.ndarray, groups: np.ndarray, reps: int = 300, seed: int = 1) -> tuple[float, float]:
    """90% interval of the mean, resampling races."""
    if len(values) == 0:
        return (np.nan, np.nan)
    codes, uniq = pd.factorize(groups)
    sums = np.bincount(codes, weights=values)
    counts = np.bincount(codes)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(reps):
        pick = rng.integers(0, len(uniq), len(uniq))
        means.append(sums[pick].sum() / max(counts[pick].sum(), 1))
    return tuple(np.percentile(means, [5, 95]))


def main() -> None:
    conn = sqlite3.connect("horse_racing.db")
    rr = pd.read_sql_query(
        "SELECT race_date, race_time, track, horse_name, race_name, race_type, race_class, "
        "race_restrictions_age, major, career_runs FROM race_results WHERE race_date >= ? AND race_date < ?",
        conn, params=(START, END))
    print(f"race_results {START} to {END}: {len(rr):,} runs")
    races = rr.drop_duplicates(["race_date", "race_time", "track"]).copy()
    print(f"{len(races):,} races\n")

    print("== 60 race names from 2025, at random")
    s = races[races["race_date"].str.startswith("2025")]["race_name"].dropna()
    for n in s.sample(min(60, len(s)), random_state=1):
        print("  ", n)

    print("\n== the 150 commonest words (distinct races)")
    words = Counter()
    for n in races["race_name"].fillna("").str.lower():
        words.update(set(re.findall(r"[a-z0-9+']+", n)))
    print(", ".join(f"{w} {c}" for w, c in words.most_common(150)))

    print("\n== race_restrictions_age, the 25 commonest")
    print(races["race_restrictions_age"].value_counts().head(25).to_string())
    print("\n== major, the 25 commonest")
    print(races["major"].value_counts().head(25).to_string())

    f = flags(races["race_name"])
    year = races["race_date"].str[:4]
    print("\n== each condition: races, and their share of the year's races")
    tab = pd.DataFrame({c: f[c].groupby(year).mean() for c in f.columns}).T
    tab.insert(0, "races", f.sum())
    print((tab.round(4)).to_string())
    print("\n== each condition against race_type (races, the 8 commonest types)")
    rtypes = races["race_type"].value_counts().head(8).index
    print(pd.DataFrame({c: races[f[c]]["race_type"].value_counts().reindex(rtypes).fillna(0).astype(int)
                        for c in f.columns}).T.to_string())

    # the development window: the forecast's error, within its race, by condition and stage
    o = fetch_predictions()
    o = o[o["race_date"] < END]
    o = o[(o["bfsp"] > 1) & (o["predicted_bfsp"] > 1)].copy()
    rr["race_date"] = rr["race_date"].astype(str)
    o["race_date"] = o["race_date"].astype(str)
    o = o.merge(rr[KEY + ["race_name", "career_runs", "race_restrictions_age"]], on=KEY, how="left")
    print(f"\n== development window: {len(o):,} runners; race name found for {o['race_name'].notna().mean():.1%}")
    o["race"] = o["race_date"] + "|" + o["track"] + "|" + o["race_time"]
    e = np.log(o["predicted_bfsp"] / o["bfsp"])
    o["e_w"] = e - e.groupby(o["race"]).transform("mean")
    cr = pd.to_numeric(o["career_runs"], errors="coerce")
    o["stage"] = np.select([cr == 0, (cr >= 1) & (cr <= 3), cr >= 4], ["debut", "1-3 runs", "4+ runs"], "unknown")
    of = flags(o["race_name"])
    rt = o["race_type"].fillna("?")
    rows = []
    for c in of.columns:
        for stage in ["debut", "1-3 runs", "4+ runs"]:
            m = of[c].to_numpy() & (o["stage"] == stage).to_numpy()
            if m.sum() < 150:
                continue
            # against the same stage in the same race types without the condition
            base = (~of[c].to_numpy()) & (o["stage"] == stage).to_numpy()
            ref = o.loc[base].groupby(rt[base])["e_w"].mean()
            excess = o.loc[m, "e_w"].to_numpy() - rt[m].map(ref).to_numpy()
            ok = np.isfinite(excess)
            lo, hi = boot_ci(excess[ok], o.loc[m, "race"].to_numpy()[ok])
            rows.append({"condition": c, "stage": stage, "runners": int(m.sum()),
                         "races": int(o.loc[m, "race"].nunique()),
                         "mean_e_within": o.loc[m, "e_w"].mean(), "excess_vs_type": np.nanmean(excess),
                         "lo90": lo, "hi90": hi, "abs_err": float(np.abs(e[m]).mean())})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n== the forecast's error within its race (ln forecast/BSP less the race mean), by condition and stage;")
    print("   excess_vs_type: against the same stage in the same race types without the condition")
    print("   (positive: the forecast is too long on them against their fields, negative: too short)")
    print(out.round(4).to_string(index=False))
    print("\n== resolved (the 90% interval clear of zero):")
    print(out[(out["lo90"] > 0) | (out["hi90"] < 0)].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
