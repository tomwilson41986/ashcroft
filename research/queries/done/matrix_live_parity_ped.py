"""The debutants' pedigree from the card's horse tooltip: does it close the gap between the 06:00 path and training?

The 944's parity query (run 36230342697) found the 06:00 card, filled from
history, differs from training most for debutants: their sire, damsire and sex
are missing at 06:00 (no earlier rows to fill from), and on 28 and 25 March that
moved the 944's price for them by a mean 0.257 in log terms (54 of 745 runners,
56% of the whole gap). The HTML card carries the sire, dam and sex in each horse
cell's tooltip (scripts/probe_card_html.py, 26 Sep: all 568 runners, all 31
debutants); daily_predictions.parse_horse_title now reads it and
model/card_enrich.py fills a horse with no history from it (the sire as history
spells it, the damsire from the dam's other offspring, the sex from Female/Male).

The same two days, the card simulated two ways from the day's rows:

  filled_old  cut to what the HTML card's cells carry, then the fill (06:00 until now)
  filled      the same plus the tooltip's sire, dam and Female/Male, then the fill

each against training's matrix, for the served 615, the 944 and the 958, with
the prices split into debutants and the rest. The code is the earlier query's
(research/queries/done/matrix_live_parity_cw.py) with these variants. Read-only.
"""
import gc
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

BLOCKS = ["race_relative", "time_figure", "race_relative_wide", "exposure", "head_to_head", "bookings",
          "form_lines", "form_variants", "connection_windows", "collateral"]
CANDIDATES = [("944", 36225615528, "bfsp-model-37"), ("958", 36238560513, "bfsp-model-39")]
CARD_COLS = ["race_date", "race_time", "track", "going_description", "race_class", "race_distance", "prize_money",
             "race_name", "stall", "horse_name", "horse_age", "pounds", "jockey_name", "trainer", "official_rating",
             "headgear", "days_since_lr", "number_of_runners"]
OUTCOME_COLS = ["place", "distbt", "odds", "fav", "comptime", "comptime_numeric", "total_dst_bt", "placing_numerical",
                "bfsp", "bfsp_place", "comment", "horse_prizewin"]
KEY = ["track", "race_time", "horse_name"]
RACE = ["track", "race_time"]
OUT = Path("out/matrix_live_parity_ped")
VARIANTS = ("filled_old", "filled")
T0 = time.time()


def say(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def fetch_candidate(label, run_id, name):
    """A train run's model artifact, through the Actions API (actions: read)."""
    import requests
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        say(f"{label}: no GITHUB_TOKEN, so no candidate prices")
        return None
    auth = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    r = requests.get(f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/artifacts",
                     headers=auth, params={"name": name}, timeout=60)
    arts = r.json().get("artifacts", []) if r.ok else []
    if not arts:
        say(f"{label}: artifact {name} of run {run_id} not found ({r.status_code})")
        return None
    z = requests.get(arts[0]["archive_download_url"], headers=auth, timeout=600)   # redirected; requests drops auth
    if not z.ok:
        say(f"{label}: download failed ({z.status_code})")
        return None
    dest = Path("candidates") / label
    zipfile.ZipFile(io.BytesIO(z.content)).extractall(dest)
    found = sorted(dest.rglob("bfsp_model.lgb"))
    return str(found[0].parent) if found else None


def feature_list():
    """Every feature compared: the served model's, the candidates', and the named blocks'."""
    from model import blocks
    spec = json.loads((OUT / "models.json").read_text())
    cols = list(spec["served_cols"])
    for f in spec["candidates"].values():
        cols += f["cols"]
    cols += [c for b in BLOCKS for c in blocks.features(b)]
    return list(dict.fromkeys(cols))


def _numeric(frame, cols):
    out = frame[KEY].copy()
    for c in cols:
        if c in frame.columns:
            out[c] = pd.to_numeric(frame[c], errors="coerce").astype(float)
    return out


def stage_matrix():
    """Training's frame for the two days: the cached matrix, the blocks built on it as training builds them."""
    from model import blocks, feature_cache
    cols = feature_list()
    key = feature_cache.cache_key("horse_racing.db", "2021-01-01")
    hit = feature_cache.load(".feature_cache", key)
    if hit is None:
        print(f"no cached matrix for key {key} (this code and today's database): nothing to compare")
        return 1
    mat, _ = hit
    say(f"matrix: {len(mat):,} rows x {mat.shape[1]:,} columns")
    march = mat[(mat.race_date >= "2026-03-01") & (mat.race_date < "2026-04-01")]
    n_races = march.groupby(march.race_date.dt.date).apply(lambda g: g.groupby(RACE).ngroups)
    busy = n_races[n_races >= 12]
    days = [max(d for d in busy.index if d.weekday() == 5), max(d for d in busy.index if d.weekday() == 2)]
    say("days: " + ", ".join(f"{d} ({n_races[d]} races)" for d in days))
    del march
    mat, _ = blocks.attach_as_trained(mat, BLOCKS)
    absent = [c for c in cols if c not in mat.columns]
    if absent:
        say(f"{len(absent)} features not in the matrix: {absent[:5]}")
    for d in days:
        rows = mat[mat.race_date == pd.Timestamp(d)]
        _numeric(rows, cols).to_parquet(OUT / f"{d}_matrix.parquet", index=False)
        say(f"matrix path, {d}: {len(rows)} runners")
    (OUT / "days.json").write_text(json.dumps([str(d) for d in days]))
    return 0


class _Zero:
    """Stands in for a booster: prepare_and_predict builds every feature in `cols`, which no one model reads."""
    serving_target = "log_bfsp"
    serving_offset = 0.0

    def predict(self, X, num_iteration=None):
        return np.zeros(len(X))


def stage_serve(day, variant):
    """prepare_and_predict as the 06:00 job calls it, for one day and one variant."""
    import model.card_enrich as card_enrich
    from predict_bfsp_today import load_bfsp_model, load_historical, prepare_and_predict
    cols = feature_list()
    _, _, vocab = load_bfsp_model("data/models")
    hist = load_historical("horse_racing.db", start_date="2021-01-01")
    truth = hist[hist.race_date == pd.Timestamp(day)].copy()
    before = hist[hist.race_date < pd.Timestamp(day)]
    del hist
    gc.collect()
    if variant == "blind":
        runners = truth.copy()
        for c in OUTCOME_COLS:
            if c in runners.columns:
                runners[c] = np.nan
    else:
        runners = truth[[c for c in CARD_COLS if c in truth.columns]].copy()
        runners["official_rating"] = pd.to_numeric(runners.official_rating, errors="coerce").replace(0, np.nan)
        if variant == "filled":            # the horse tooltip: 'Bay, Male, Stallion - X, Dam - Y'
            runners["card_stallion"] = truth["stallion"]
            runners["card_dam"] = truth["dam"]
            sex = truth["horse_sex"]
            runners["card_sex"] = np.where(sex.isin(["Filly", "Mare"]), "Female", np.where(sex.notna(), "Male", None))
    real = card_enrich.enrich_card
    if variant == "blind":
        card_enrich.enrich_card = lambda card, history: card
    try:
        out = prepare_and_predict(before, runners, _Zero(), cols, pd.Timestamp(day).date(), vocab)
    finally:
        card_enrich.enrich_card = real
    _numeric(out, cols).to_parquet(OUT / f"{day}_{variant}.parquet", index=False)
    say(f"{day} {variant}: {len(out)} runners")
    return 0


def same(x, y):
    return ((np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= 1e-9 + 1e-6 * np.abs(x))).astype(float)


def compare():
    from model import blocks
    from model.bfsp_model import predict_prices
    from predict_bfsp_today import load_bfsp_model
    spec = json.loads((OUT / "models.json").read_text())
    served, served_cols, _ = load_bfsp_model("data/models")
    models = {"615": (served, served_cols)}
    for label, c in spec["candidates"].items():
        booster, fcols, _ = load_bfsp_model(c["path"])
        models[label] = (booster, fcols)
    gain = pd.read_csv("data/models/bfsp_feature_importance.csv").set_index("feature")["importance"]
    gain = gain / gain.sum()
    block_of = {}
    for b in blocks.build_order(BLOCKS):
        for c in blocks.features(b):
            block_of.setdefault(c, b)
    cols = feature_list()
    days = json.loads((OUT / "days.json").read_text())

    def prices(booster, fcols, frame, suffix):
        X = pd.DataFrame({c: frame[c + suffix] if c + suffix in frame.columns else np.nan for c in fcols},
                         index=frame.index)
        X["raceid"] = frame["rid"].to_numpy()
        return predict_prices(booster, X, fcols, race_col="raceid")["predicted_log_bfsp"].to_numpy()

    print("\nRunners in both, and races whose runners are the same in both (within-race readings compare only there)")
    for variant in VARIANTS:
        rows = []
        for d in days:
            r = pd.read_parquet(OUT / f"{d}_matrix.parquet")
            lv = pd.read_parquet(OUT / f"{d}_{variant}.parquet")
            m = r.merge(lv, on=KEY, suffixes=("_m", "_l"))
            n_r, n_l, n_m = r.groupby(RACE).size(), lv.groupby(RACE).size(), m.groupby(RACE).size()
            whole = n_m[(n_m == n_r.reindex(n_m.index)) & (n_m == n_l.reindex(n_m.index))].index
            m["whole"] = pd.MultiIndex.from_frame(m[RACE]).isin(whole)
            m["rid"] = d + "|" + m["track"].astype(str) + "|" + m["race_time"].astype(str)
            rows.append(m)
            print(f"  {variant:6s} {d}: matrix {len(r)}, live {len(lv)}, both {len(m)}; races {n_r.size}, "
                  f"with the same runners {len(whole)}")
        m = pd.concat(rows, ignore_index=True)

        print(f"\n{variant}: features that differ, matrix -> live (share of runners, races with the same runners)")
        w = m[m["whole"]]
        diff = []
        for c in cols:
            if c + "_m" not in w.columns or c + "_l" not in w.columns:
                continue
            x, y = w[c + "_m"].to_numpy(dtype=float), w[c + "_l"].to_numpy(dtype=float)
            s = same(x, y)
            a = np.abs(x - y)
            diff.append((c, block_of.get(c, "served"), 1 - s.mean(), float((np.isnan(x) != np.isnan(y)).mean()),
                         float(np.nanmax(a)) if np.isfinite(a).any() else 0.0, gain.get(c, np.nan)))
        diff = pd.DataFrame(diff, columns=["feature", "group", "share", "nan_mismatch", "max_abs", "gain615"])
        g = diff.groupby("group").agg(n=("feature", "size"), differ=("share", lambda s: int((s > 0).sum())),
                                      mean_share=("share", "mean"), worst=("share", "max"))
        print(g.to_string())
        sd = diff[diff.group == "served"]
        print(f"  served features, gain-weighted share differing: {(sd.share * sd.gain615.fillna(0)).sum():.4f}")
        for r in diff[diff.share > 0].sort_values("share", ascending=False).head(40).itertuples():
            print(f"  {r.feature:36s} {r.group:18s} differs {r.share:6.3f} (missing on one side {r.nan_mismatch:5.3f})"
                  f"  max |d| {r.max_abs:.4g}  gain615 {r.gain615:.4f}")
        diff.to_csv(OUT / f"features_{variant}.csv", index=False)

        print(f"\n{variant}: prices, each model on the matrix's features and on the live path's, the same runners")
        print(f"  {'model':6s} {'races':6s} {'runners':>8s} {'|dlog| mean':>11s} {'median':>7s} {'p90':>6s} "
              f"{'max':>6s} {'corr':>7s} {'top pick':>9s}")
        for label, (booster, fcols) in models.items():
            for scope, sub in (("same", m[m["whole"]]), ("all", m)):
                if not len(sub):
                    continue
                sub = sub.copy()
                sub["lm"], sub["ll"] = prices(booster, fcols, sub, "_m"), prices(booster, fcols, sub, "_l")
                dd = np.abs(sub["lm"] - sub["ll"]).to_numpy()
                top_m = sub.loc[sub.groupby("rid")["lm"].idxmin(), ["rid", "horse_name"]].set_index("rid")["horse_name"]
                top_l = sub.loc[sub.groupby("rid")["ll"].idxmin(), ["rid", "horse_name"]].set_index("rid")["horse_name"]
                agree = (top_m == top_l.reindex(top_m.index)).mean()
                print(f"  {label:6s} {scope:6s} {len(sub):8,d} {dd.mean():11.4f} {np.median(dd):7.4f} "
                      f"{np.quantile(dd, .9):6.3f} {dd.max():6.3f} {np.corrcoef(sub.lm, sub.ll)[0, 1]:7.4f} "
                      f"{agree:9.3f}")
                if scope == "all" and "career_runs_num_m" in sub.columns:
                    deb = sub["career_runs_num_m"].to_numpy() == 0
                    print(f"  {'':6s} {'':6s} debutants {int(deb.sum())}: |dlog| {dd[deb].mean():.4f}; "
                          f"the rest {int((~deb).sum())}: {dd[~deb].mean():.4f}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--matrix":
        return stage_matrix()
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        return stage_serve(sys.argv[2], sys.argv[3])

    from predict_bfsp_today import load_bfsp_model
    OUT.mkdir(parents=True, exist_ok=True)
    _, served_cols, vocab = load_bfsp_model("data/models")
    spec = {"served_cols": served_cols, "candidates": {}}
    for label, run_id, name in CANDIDATES:
        path = fetch_candidate(label, run_id, name)
        if path:
            _, fcols, cvocab = load_bfsp_model(path)
            spec["candidates"][label] = {"path": path, "cols": fcols}
            say(f"{label}: {len(fcols)} features from {path}; its categorical vocabulary "
                f"{'is' if cvocab == vocab else 'is NOT'} the served model's")
    (OUT / "models.json").write_text(json.dumps(spec))
    say(f"features compared: {len(feature_list())}")

    me = [sys.executable, __file__]
    if subprocess.run(me + ["--matrix"]).returncode != 0 or not (OUT / "days.json").exists():
        say("the matrix stage did not finish: nothing to compare")
        return 0
    for d in json.loads((OUT / "days.json").read_text()):
        for variant in VARIANTS:
            rc = subprocess.run(me + ["--serve", d, variant]).returncode
            if rc != 0:
                say(f"{d} {variant}: the live build failed ({rc})")
                return 0
    compare()
    say("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
