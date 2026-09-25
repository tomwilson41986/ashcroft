"""Does the 06:00 path serve what the model was trained on? The matrix against the card, drop-in blocks included.

A model is trained on the cached feature matrix (evaluate_oos.build_feature_frame:
the engine over the whole history, the runs with a usable price kept) with its
drop-in blocks built on that matrix by blocks.attach_as_trained. The 06:00 job
builds the same features another way: prepare_and_predict runs the engine on
the history before the day with the card appended, then attach_as_trained on
the history's priced runs plus the card. live_parity_check.py (run 35928189383)
compared live paths with each other; this compares the live path with training,
which is what a model trained on the matrix is actually served.

On two development days (the busiest Saturday and Wednesday of March 2026, as
before) every runner is served by prepare_and_predict two ways, each with the
history before the day:

  blind    the day's rows with their outcomes blanked, no fill: the live code
           with a perfect card
  filled   the rows cut to what the HTML card carries, then the fill: 06:00 as
           it runs

and compared with the same runners' rows in the matrix. Features compared: the
served 615's, and the blocks of the 853 and the 859 (within-race readings, time
figure, wide within-race readings with the form variants they read, exposure,
collateral form, bookings). Within-race readings see the whole card live but
only the priced runners in the matrix, so races whose runners differ between
the two are reported apart. Prices: the served 615, and the 853 from its train
run's artifact, each on both feature sets over the same runners.

Read-only. The matrix is the one research-query.yml restores; on a miss this
reports that and stops.
"""
import gc
import io
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, ".")
import model.card_enrich as card_enrich  # noqa: E402
from model import blocks, feature_cache  # noqa: E402
from model.bfsp_model import predict_prices  # noqa: E402
from predict_bfsp_today import load_bfsp_model, load_historical, prepare_and_predict  # noqa: E402

BLOCKS = ["race_relative", "time_figure", "race_relative_wide", "exposure", "head_to_head", "bookings"]
CANDIDATES = [("853", 36142969939, "bfsp-model-34")]      # (label, train-bfsp run id, its model artifact)
CARD_COLS = ["race_date", "race_time", "track", "going_description", "race_class", "race_distance", "prize_money",
             "race_name", "stall", "horse_name", "horse_age", "pounds", "jockey_name", "trainer", "official_rating",
             "headgear", "days_since_lr", "number_of_runners"]
OUTCOME_COLS = ["place", "distbt", "odds", "fav", "comptime", "comptime_numeric", "total_dst_bt", "placing_numerical",
                "bfsp", "bfsp_place", "comment", "horse_prizewin"]
KEY = ["track", "race_time", "horse_name"]
RACE = ["track", "race_time"]
T0 = time.time()


def say(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def fetch_candidate(label, run_id, name):
    """A train run's model artifact, through the Actions API (actions: read)."""
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


# ---- the models: the served one, and the candidates ----
served, served_cols, vocab = load_bfsp_model("data/models")
gain = pd.read_csv("data/models/bfsp_feature_importance.csv").set_index("feature")["importance"]
gain = gain / gain.sum()
models = {"615": (served, served_cols)}
for label, run_id, name in CANDIDATES:
    path = fetch_candidate(label, run_id, name)
    if path:
        booster, fcols, cvocab = load_bfsp_model(path)
        models[label] = (booster, fcols)
        same = cvocab == vocab
        say(f"{label}: {len(fcols)} features from {path}; its categorical vocabulary "
            f"{'is' if same else 'is NOT'} the served model's")
block_of = {}
for b in blocks.build_order(BLOCKS):
    for c in blocks.features(b):
        block_of.setdefault(c, b)
cols = list(dict.fromkeys(served_cols + [c for _, (_, f) in models.items() for c in f]
                          + [c for b in BLOCKS for c in blocks.features(b)]))
say(f"features compared: {len(cols)} ({len(served_cols)} served, {len(cols) - len(served_cols)} from blocks)")

# ---- the matrix path: training's frame, the blocks built on it as training builds them ----
key = feature_cache.cache_key("horse_racing.db", "2021-01-01")
hit = feature_cache.load(".feature_cache", key)
if hit is None:
    print(f"no cached matrix for key {key} (this code and today's database): nothing to compare")
    sys.exit(0)
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
keep = list(dict.fromkeys(KEY + ["race_date", "raceid", "bfsp"] + [c for c in cols if c in mat.columns]))
ref = {d: mat.loc[mat.race_date == pd.Timestamp(d), keep].reset_index(drop=True) for d in days}
say("matrix path built: " + ", ".join(f"{d} {len(ref[d])} runners" for d in days))
del mat
gc.collect()


# ---- the live path: prepare_and_predict as the 06:00 job calls it ----
class _Zero:
    """Stands in for a booster: prepare_and_predict builds every feature in `cols`, which no one model reads."""
    serving_target = "log_bfsp"
    serving_offset = 0.0

    def predict(self, X, num_iteration=None):
        return np.zeros(len(X))


hist = load_historical("horse_racing.db", start_date="2021-01-01")
say(f"history: {len(hist):,} rows")


def serve(day, runners, fill):
    real = card_enrich.enrich_card
    if not fill:
        card_enrich.enrich_card = lambda card, history: card
    try:
        before = hist[hist.race_date < pd.Timestamp(day)]
        out = prepare_and_predict(before, runners.copy(), _Zero(), cols, day, vocab)
    finally:
        card_enrich.enrich_card = real
    out = out[list(dict.fromkeys(KEY + [c for c in cols if c in out.columns]))].copy()
    gc.collect()
    return out


live = {}
for day in days:
    truth = hist[hist.race_date == pd.Timestamp(day)].copy()
    blind = truth.copy()
    for c in OUTCOME_COLS:
        if c in blind.columns:
            blind[c] = np.nan
    card = truth[[c for c in CARD_COLS if c in truth.columns]].copy()
    card["official_rating"] = pd.to_numeric(card.official_rating, errors="coerce").replace(0, np.nan)
    live[(day, "blind")] = serve(day, blind, fill=False)
    say(f"{day} blind: {len(live[(day, 'blind')])} runners")
    live[(day, "filled")] = serve(day, card, fill=True)
    say(f"{day} filled: {len(live[(day, 'filled')])} runners")
os.makedirs("out/matrix_live_parity", exist_ok=True)
for d in days:
    ref[d].to_csv(f"out/matrix_live_parity/{d}_matrix.csv.gz", index=False)
for (d, v), f in live.items():
    f.to_csv(f"out/matrix_live_parity/{d}_{v}.csv.gz", index=False)


# ---- compare ----
def same(x, y):
    x, y = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
    return ((x.isna() & y.isna()) | ((x - y).abs() <= 1e-9 + 1e-6 * x.abs())).to_numpy()


def prices(booster, fcols, frame, suffix):
    X = frame[[c + suffix for c in fcols]].copy()
    X.columns = fcols
    X["raceid"] = frame["rid"].to_numpy()
    return predict_prices(booster, X, fcols, race_col="raceid")["predicted_log_bfsp"].to_numpy()


print("\nRunners in both, and races whose runners are the same in both (within-race readings compare only there)")
for variant in ("blind", "filled"):
    rows = []
    for d in days:
        r, lv = ref[d], live[(d, variant)]
        m = r.merge(lv, on=KEY, suffixes=("_m", "_l"))
        n_r = r.groupby(RACE).size()
        n_l = lv.groupby(RACE).size()
        n_m = m.groupby(RACE).size()
        whole = n_m[(n_m == n_r.reindex(n_m.index)) & (n_m == n_l.reindex(n_m.index))].index
        m["whole"] = pd.MultiIndex.from_frame(m[RACE]).isin(whole)
        m["rid"] = m["track"].astype(str) + "|" + m["race_time"].astype(str)
        rows.append(m)
        print(f"  {variant:6s} {d}: matrix {len(r)}, live {len(lv)}, both {len(m)}; races {n_r.size}, "
              f"with the same runners {len(whole)}")
    m = pd.concat(rows, ignore_index=True)

    print(f"\n{variant}: features that differ, matrix -> live (share of runners; races with the same runners)")
    diff = []
    w = m[m["whole"]]
    for c in cols:
        if c + "_m" not in w.columns or c + "_l" not in w.columns:
            continue
        s = same(w[c + "_m"], w[c + "_l"])
        a = (pd.to_numeric(w[c + "_m"], errors="coerce") - pd.to_numeric(w[c + "_l"], errors="coerce")).abs()
        diff.append((c, block_of.get(c, "served"), 1 - s.mean(), float(a.max()) if a.notna().any() else 0.0,
                     gain.get(c, np.nan)))
    diff = pd.DataFrame(diff, columns=["feature", "group", "share", "max_abs", "gain615"])
    g = diff.groupby("group").agg(n=("feature", "size"), differ=("share", lambda s: int((s > 0).sum())),
                                  mean_share=("share", "mean"), worst=("share", "max"))
    print(g.to_string())
    served_d = diff[diff.group == "served"]
    print(f"  served features, gain-weighted share differing: {(served_d.share * served_d.gain615.fillna(0)).sum():.4f}")
    top = diff[diff.share > 0].sort_values("share", ascending=False).head(40)
    for r in top.itertuples():
        print(f"  {r.feature:36s} {r.group:18s} differs {r.share:6.3f}  max |d| {r.max_abs:.4g}  gain615 {r.gain615:.4f}")
    diff.to_csv(f"out/matrix_live_parity/features_{variant}.csv", index=False)

    print(f"\n{variant}: prices, each model on the matrix's features and on the live path's, the same runners")
    print(f"  {'model':6s} {'races':6s} {'runners':>8s} {'|dlog| mean':>11s} {'median':>7s} {'p90':>6s} {'max':>6s} "
          f"{'corr':>7s} {'top pick':>9s}")
    for label, (booster, fcols) in models.items():
        for scope, sub in (("same", m[m["whole"]]), ("all", m)):
            if not len(sub):
                continue
            sub = sub.copy()
            lm, ll = prices(booster, fcols, sub, "_m"), prices(booster, fcols, sub, "_l")
            d = np.abs(lm - ll)
            sub["lm"], sub["ll"] = lm, ll
            top_m = sub.loc[sub.groupby("rid")["lm"].idxmin(), ["rid", "horse_name"]].set_index("rid")["horse_name"]
            top_l = sub.loc[sub.groupby("rid")["ll"].idxmin(), ["rid", "horse_name"]].set_index("rid")["horse_name"]
            agree = (top_m == top_l.reindex(top_m.index)).mean()
            print(f"  {label:6s} {scope:6s} {len(sub):8,d} {d.mean():11.4f} {np.median(d):7.4f} "
                  f"{np.quantile(d, .9):6.3f} {d.max():6.3f} {np.corrcoef(lm, ll)[0, 1]:7.4f} {agree:9.3f}")
say("done")
