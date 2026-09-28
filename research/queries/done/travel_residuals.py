"""How far the yard sends a horse, and four blocks never tested on the price, against what the best model still gets wrong.

Travel is an intent signal no block reads. A trainer who sends a horse three hundred
miles, or across the Irish Sea, to a course the yard rarely visits has chosen the race;
the market hears about it through the day, and a forecast of the BSP made at 06:00 could
anticipate it. Everything here is on the 06:00 card or from earlier days:

    tv_km             great-circle km from today's course to the yard's base: the centre
                      of its runners' courses on earlier days, decayed (half-life 365 days),
                      once it has five runners' worth
    tv_usual          the yard's usual trip: its earlier runners' tv_km, decayed the same way
    tv_km_vs_usual    tv_km less tv_usual (km)
    tv_km_ratio       log((tv_km + 10) / (tv_usual + 10))
    tv_raid           1 when today's course is in the other country (Britain or Ireland)
                      from the one most of the yard's runners ran in
    tv_track_share    the yard's share of its runners at today's course (decayed counts)
    tv_n_meeting      the yard's runners at today's meeting (the card)
    tv_km_single      tv_km when the horse is the yard's only runner at the meeting, else 0
    tv_horse_km       km from the horse's last course (an earlier day) to today's

The four research blocks that were only ever tested against the result (nothing beyond
BSP there), and freshness, intent and form windows all failed that test yet sharpened the
BSP forecast (reports/feature_inventory.xlsx, N3):

    kf_*    a Kalman rating on the performance figure (model/state_space.py): the prior
            mean before the run, its sd, runs, within-race z / rank / gap to the best,
            and the rating less today's official rating
    hc_*    handicap angles (model/handicap_features.py): under a penalty, wrong at the
            weights, today's mark against the last winning mark
    ae_*    wins less the BSP's chances by cell, the ones no served block reads (trainer at
            the course, course/trip/draw third, trainer with jockey, sire): model/ae_features.py
    mj_*    black-type experience: runs, mean finishing position and win rate in Group,
            Grade, Listed and Class 1 races before today, and those against today's race

The best model so far (the 968s5xh averaged with the within-race extra-trees fit,
iteration 94's in-run pair: 0.3981 on the development window) priced every runner from
27 Sep 2025 to 31 Mar 2026 out of sample. As future_form_residuals did: for each feature a
quadratic with a missing flag, demeaned within race, fitted to the model's log error on
five of the six months and applied to the sixth; the change in the mean absolute log
error (x 10^4) with a 90% race-bootstrap interval, then each block jointly (ridge 100),
and the travel block by segment. Two engine features the model reads heavily are the
controls (what "already in the model" reads as), with a placebo. Read-only.
"""
import io
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

#: iteration 94's research artifact (run 36331536686): r/oos_xthub_xtdml.csv, the in-run pair
ARTIFACT_ID = 10939044871
OOS_NAME = "oos_xthub_xtdml.csv"
KEY = ["race_date", "race_time", "track", "horse_name"]
WANT = KEY + ["raceid", "trainer", "jockey_name", "stallion", "official_rating", "pounds", "placing_numerical",
              "dist_furlongs", "total_dst_bt", "median_or", "race_type", "race_name", "race_class", "major", "bfsp",
              "stall", "number_of_runners", "rPMW3", "rPMW5"]
OUT = Path("out/travel_residuals")
N_BOOT = 1000
HL = 365.0
T0 = time.time()

#: Racecourses (normalised name: lower case, parenthetical and non-letters removed) -> (lat, lon, Ireland?).
#: Approximate centres, to a few km: travel is read in tens and hundreds of km.
COURSES = {
    # Great Britain
    "aintree": (53.477, -2.951, 0), "ascot": (51.412, -0.673, 0), "ayr": (55.460, -4.614, 0),
    "bangor": (52.998, -2.915, 0), "bath": (51.417, -2.409, 0), "beverley": (53.846, -0.459, 0),
    "brighton": (50.832, -0.114, 0), "carlisle": (54.869, -2.917, 0), "cartmel": (54.199, -2.953, 0),
    "catterick": (54.381, -1.636, 0), "chelmsford": (51.764, 0.449, 0), "cheltenham": (51.921, -2.061, 0),
    "chepstow": (51.652, -2.683, 0), "chester": (53.187, -2.899, 0), "doncaster": (53.518, -1.106, 0),
    "epsom": (51.316, -0.257, 0), "exeter": (50.628, -3.599, 0), "fakenham": (52.826, 0.826, 0),
    "ffoslas": (51.749, -4.230, 0), "fontwell": (50.858, -0.635, 0), "goodwood": (50.893, -0.757, 0),
    "hamilton": (55.782, -4.033, 0), "haydock": (53.476, -2.628, 0), "hereford": (52.070, -2.730, 0),
    "hexham": (54.962, -2.113, 0), "huntingdon": (52.335, -0.174, 0), "kelso": (55.605, -2.425, 0),
    "kempton": (51.418, -0.402, 0), "leicester": (52.600, -1.098, 0), "lingfield": (51.176, -0.017, 0),
    "ludlow": (52.382, -2.724, 0), "marketrasen": (53.384, -0.321, 0), "musselburgh": (55.946, -3.050, 0),
    "newbury": (51.397, -1.302, 0), "newcastle": (55.012, -1.668, 0), "newmarket": (52.246, 0.378, 0),
    "newtonabbot": (50.535, -3.595, 0), "nottingham": (52.946, -1.106, 0), "perth": (56.421, -3.432, 0),
    "plumpton": (50.933, -0.059, 0), "pontefract": (53.694, -1.334, 0), "redcar": (54.612, -1.071, 0),
    "ripon": (54.124, -1.506, 0), "salisbury": (51.049, -1.847, 0), "sandown": (51.375, -0.359, 0),
    "sedgefield": (54.648, -1.462, 0), "southwell": (53.066, -0.919, 0), "stratford": (52.183, -1.723, 0),
    "taunton": (51.000, -3.083, 0), "thirsk": (54.227, -1.329, 0), "towcester": (52.128, -0.989, 0),
    "uttoxeter": (52.899, -1.856, 0), "warwick": (52.285, -1.594, 0), "wetherby": (53.930, -1.370, 0),
    "wincanton": (51.052, -2.407, 0), "windsor": (51.486, -0.628, 0), "wolverhampton": (52.603, -2.133, 0),
    "worcester": (52.198, -2.228, 0), "yarmouth": (52.617, 1.726, 0), "york": (53.943, -1.089, 0),
    # Ireland (Down Royal and Downpatrick race under the Irish authority)
    "ballinrobe": (53.630, -9.237, 1), "bellewstown": (53.690, -6.352, 1), "clonmel": (52.358, -7.690, 1),
    "cork": (52.130, -8.640, 1), "curragh": (53.163, -6.824, 1), "downroyal": (54.504, -6.143, 1),
    "downpatrick": (54.320, -5.719, 1), "dundalk": (54.024, -6.428, 1), "fairyhouse": (53.490, -6.533, 1),
    "galway": (53.298, -8.992, 1), "gowranpark": (52.627, -7.061, 1), "kilbeggan": (53.365, -7.494, 1),
    "killarney": (52.045, -9.505, 1), "laytown": (53.683, -6.237, 1), "leopardstown": (53.268, -6.198, 1),
    "limerick": (52.576, -8.766, 1), "listowel": (52.444, -9.487, 1), "naas": (53.214, -6.660, 1),
    "navan": (53.638, -6.702, 1), "punchestown": (53.197, -6.625, 1), "roscommon": (53.628, -8.178, 1),
    "sligo": (54.269, -8.466, 1), "thurles": (52.684, -7.817, 1), "tipperary": (52.475, -8.149, 1),
    "tramore": (52.158, -7.146, 1), "wexford": (52.339, -6.483, 1),
}
ALIASES = {"chelmsfordcity": "chelmsford", "bangorondee": "bangor", "kemptonpark": "kempton",
           "sandownpark": "sandown", "haydockpark": "haydock", "epsomdowns": "epsom", "lingfieldpark": "lingfield",
           "greatyarmouth": "yarmouth", "stratfordonavon": "stratford", "fontwellpark": "fontwell",
           "hamiltonpark": "hamilton", "catterickbridge": "catterick", "thecurragh": "curragh", "mallow": "cork",
           "royalwindsor": "windsor", "ffoslasracecourse": "ffoslas"}


def say(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def norm_track(s: pd.Series) -> pd.Series:
    t = (s.fillna("").astype(str).str.lower().str.replace(r"\(.*?\)", "", regex=True)
         .str.replace(r"[^a-z]", "", regex=True))
    return t.replace(ALIASES)


def km(lat1, lon1, lat2, lon2):
    r = np.pi / 180.0
    a = (np.sin((lat2 - lat1) * r / 2) ** 2
         + np.cos(lat1 * r) * np.cos(lat2 * r) * np.sin((lon2 - lon1) * r / 2) ** 2)
    return 12742.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _name_codes(s: pd.Series) -> np.ndarray:
    t = s.fillna("").astype(str).str.strip().str.lower()
    bad = t.isin(["", "nan", "none"]).to_numpy()
    return np.where(bad, -1, pd.factorize(t, sort=True)[0]).astype(np.int64)


def fetch_oos():
    import requests
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        say("no GITHUB_TOKEN: no forecasts to read")
        return None
    auth = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    z = requests.get(f"https://api.github.com/repos/{repo}/actions/artifacts/{ARTIFACT_ID}/zip", headers=auth,
                     timeout=600)
    if not z.ok:
        say(f"artifact {ARTIFACT_ID}: download failed ({z.status_code})")
        return None
    zf = zipfile.ZipFile(io.BytesIO(z.content))
    name = next((n for n in zf.namelist() if n.endswith("/" + OOS_NAME) or n == OOS_NAME), None)
    if name is None:
        say(f"artifact {ARTIFACT_ID}: no {OOS_NAME} in {zf.namelist()[:10]}")
        return None
    return pd.read_csv(zf.open(name))


def load_matrix():
    import pyarrow.parquet as pq
    from model import feature_cache
    key = feature_cache.cache_key("horse_racing.db", "2021-01-01")
    data, _ = feature_cache._paths(".feature_cache", key)
    if not data.exists():
        say(f"no cached matrix for key {key}: nothing to read")
        return None
    have = set(pq.read_schema(data).names)
    cols = [c for c in WANT if c in have]
    say(f"matrix columns missing: {[c for c in WANT if c not in have]}")
    mat = pd.read_parquet(data, columns=cols)
    say(f"matrix: {len(mat):,} rows")
    return mat


# ----------------------------------------------------------------------------- blocks

def travel_block(m: pd.DataFrame) -> pd.DataFrame:
    from model.race_shape import _codes, asof_decayed_mean, day_index
    n = len(m)
    day = day_index(m)
    t = norm_track(m["track"])
    table = pd.DataFrame.from_dict(COURSES, orient="index", columns=["lat", "lon", "ire"])
    lat = t.map(table["lat"]).to_numpy(float)
    lon = t.map(table["lon"]).to_numpy(float)
    ire = t.map(table["ire"]).to_numpy(float)
    miss = t[np.isnan(lat)].value_counts()
    print(f"\nCourses matched: {np.isfinite(lat).mean():.4f} of rows; unmatched (rows): "
          + (", ".join(f"{k or '(blank)'} {v}" for k, v in miss.head(30).items()) or "none"))
    tr = _name_codes(m["trainer"])
    blat, n_b = asof_decayed_mean(tr, day, lat, tr, day, HL)
    blon, _ = asof_decayed_mean(tr, day, lon, tr, day, HL)
    tv_km = np.where(n_b >= 5, km(lat, lon, blat, blon), np.nan)
    usual, n_u = asof_decayed_mean(tr, day, tv_km, tr, day, HL)
    tv_usual = np.where(n_u >= 5, usual, np.nan)
    ire_share, n_i = asof_decayed_mean(tr, day, ire, tr, day, HL)
    raid = np.where((n_i >= 5) & np.isfinite(ire), ((ire_share > 0.5) != (ire > 0.5)).astype(float), np.nan)
    tt = np.where((tr >= 0) & np.isfinite(lat), _codes(tr, t.to_numpy()), -1).astype(np.int64)
    ones = np.ones(n)
    _, n_tt = asof_decayed_mean(tt, day, ones, tt, day, HL)
    _, n_t = asof_decayed_mean(tr, day, ones, tr, day, HL)
    share = np.where((n_t > 0) & (tt >= 0), n_tt / (n_t + 1.0), np.nan)
    meet = m["race_date"].astype(str) + "|" + t
    n_meet = pd.Series(ones).groupby([meet.to_numpy(), tr]).transform("sum").to_numpy()
    n_meet = np.where(tr >= 0, n_meet, np.nan)
    # the horse's last course on an earlier day
    h = _name_codes(m["horse_name"])
    o = np.lexsort((day, h))
    hs, ds = h[o], day[o]
    earlier = np.r_[False, (hs[1:] == hs[:-1]) & (ds[1:] > ds[:-1])] & (hs >= 0)
    plat = np.full(n, np.nan)
    plon = np.full(n, np.nan)
    plat[o] = np.where(earlier, np.r_[np.nan, lat[o][:-1]], np.nan)
    plon[o] = np.where(earlier, np.r_[np.nan, lon[o][:-1]], np.nan)
    out = pd.DataFrame(index=m.index)
    out["tv_km"] = tv_km
    out["tv_usual"] = tv_usual
    out["tv_km_vs_usual"] = tv_km - tv_usual
    out["tv_km_ratio"] = np.log((tv_km + 10.0) / (tv_usual + 10.0))
    out["tv_raid"] = raid
    out["tv_track_share"] = share
    out["tv_n_meeting"] = n_meet
    out["tv_km_single"] = np.where(np.isfinite(tv_km), np.where(n_meet == 1, tv_km, 0.0), np.nan)
    out["tv_horse_km"] = km(lat, lon, plat, plon)
    return out


def kalman_block(m: pd.DataFrame) -> pd.DataFrame:
    from model.perf_figures import performance_figure_lbs
    from model.state_space import add_kalman_features
    cols = [c for c in ("race_date", "race_time", "track", "horse_name", "raceid", "official_rating", "median_or",
                        "total_dst_bt", "placing_numerical", "dist_furlongs") if c in m.columns]
    sub = m[cols].copy()
    sub["perf_lbs"] = performance_figure_lbs(sub).to_numpy()
    # the filter's starting mean from the matrix's first half-year only: add_kalman_features' default (the mean
    # of every figure) would carry each day's own results into every debutant's prior
    d = pd.to_datetime(sub["race_date"], errors="coerce")
    early = d < d.min() + pd.Timedelta(days=182)
    init = float(np.nanmean(sub.loc[early, "perf_lbs"]))
    k = add_kalman_features(sub, "perf_lbs", prefix="kf", init_mean=init)
    out = pd.DataFrame(index=m.index)
    for c in ("kf_rating", "kf_sd", "kf_n", "kf_z", "kf_rank", "kf_vs_max"):
        out[c] = k[c].to_numpy()
    orr = pd.to_numeric(m["official_rating"], errors="coerce").replace(0, np.nan).to_numpy(float)
    out["kf_vs_or"] = out["kf_rating"].to_numpy() - orr
    return out


def handicap_block(m: pd.DataFrame) -> pd.DataFrame:
    from model.handicap_features import add_handicap_features
    res, names = add_handicap_features(m)
    return res[[c for c in names if c != "hc_is_handicap"]]       # race-level: nothing within a race


def ae_block(m: pd.DataFrame) -> pd.DataFrame:
    from model.ae_features import ENTITIES, add_ae_features
    ents = {k: ENTITIES[k] for k in ("trainer_track", "track_draw", "trainer_jockey", "sire")}
    res, names = add_ae_features(m, entities=ents)
    return res[names]


def major_block(m: pd.DataFrame) -> pd.DataFrame:
    from model.connections import is_black_type
    from model.race_shape import asof_decayed_mean, day_index
    day = day_index(m)
    h = _name_codes(m["horse_name"])
    bt = is_black_type(m).to_numpy()
    pos = pd.to_numeric(m["placing_numerical"], errors="coerce").to_numpy(float)
    nr = pd.to_numeric(m["number_of_runners"], errors="coerce").to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        nfp = np.where((pos > 0) & (nr > 1), (nr - pos) / (nr - 1), np.nan)
    won = np.where(pos > 0, (pos == 1).astype(float), np.nan)
    mj_nfp, mj_n = asof_decayed_mean(h, day, np.where(bt, nfp, np.nan), h, day)
    mj_wr, _ = asof_decayed_mean(h, day, np.where(bt, won, np.nan), h, day)
    all_nfp, _ = asof_decayed_mean(h, day, nfp, h, day)
    out = pd.DataFrame(index=m.index)
    out["mj_n"] = mj_n
    out["mj_nfp"] = mj_nfp
    out["mj_wr"] = mj_wr
    out["mj_nfp_vs_all"] = mj_nfp - all_nfp
    out["mj_n_today_bt"] = np.where(bt, mj_n, np.nan)
    out["mj_nfp_today_bt"] = np.where(bt, mj_nfp, np.nan)
    return out


# ----------------------------------------------------------------------------- the screen (future_form_residuals)

def _norm_keys(df):
    out = df.copy()
    out["race_date"] = pd.to_datetime(out["race_date"]).dt.normalize()
    for c in ("race_time", "track", "horse_name"):
        out[c] = out[c].astype(str).str.strip()
    return out


def demean(a, race):
    means = pd.DataFrame(a).groupby(race).transform("mean").to_numpy()
    return a - means


def design(x):
    miss = ~np.isfinite(x)
    ok = ~miss
    mu = np.nanmean(x[ok]) if ok.any() else 0.0
    sd = np.nanstd(x[ok]) if ok.any() else 1.0
    z = np.where(miss, 0.0, (x - mu) / (sd if sd > 0 else 1.0))
    z = np.clip(z, -5, 5)
    cols = [z, z * z]
    if miss.any() and not miss.all():
        cols.append(miss.astype(float))
    return np.column_stack(cols)


def oof_adjustment(X, e_dm, race, month, ridge=0.0):
    Xd = demean(X, race)
    adj = np.zeros(len(e_dm))
    for mo in np.unique(month):
        te = month == mo
        tr = ~te
        A = Xd[tr]
        G = A.T @ A + ridge * np.eye(A.shape[1])
        try:
            beta = np.linalg.solve(G, A.T @ e_dm[tr])
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(A, e_dm[tr], rcond=None)[0]
        adj[te] = Xd[te] @ beta
    return adj


def delta_with_ci(e0, adj, race_codes, rng, rows=None):
    d = np.abs(e0 - adj) - np.abs(e0)
    rc = race_codes
    if rows is not None:
        d, rc = d[rows], pd.factorize(race_codes[rows])[0]
    sums = np.bincount(rc, weights=d)
    counts = np.bincount(rc).astype(float)
    n_r = len(sums)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.integers(0, n_r, n_r)
        boots[i] = sums[pick].sum() / counts[pick].sum()
    return d.mean(), np.quantile(boots, 0.05), np.quantile(boots, 0.95)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    oos = fetch_oos()
    if oos is None:
        return 0
    say(f"forecasts: {len(oos):,} runners")
    mat = load_matrix()
    if mat is None:
        return 0
    blocks = {}
    for name, fn in (("travel", travel_block), ("kalman", kalman_block), ("handicap", handicap_block),
                     ("ae cells", ae_block), ("black type", major_block)):
        try:
            blocks[name] = fn(mat)
            say(f"{name}: {blocks[name].shape[1]} features")
        except Exception as exc:                       # one block failing must not cost the rest
            say(f"{name}: FAILED {type(exc).__name__}: {exc}")
    feats = pd.concat([mat[KEY]] + list(blocks.values()), axis=1)
    for c in ("rPMW3", "rPMW5"):
        if c in mat.columns:
            feats[c] = mat[c].to_numpy()
    del mat
    fcols = [c for c in feats.columns if c not in KEY]
    left = _norm_keys(oos[KEY + ["predicted_bfsp", "bfsp", "race_type"]]).drop_duplicates(KEY, keep=False)
    right = _norm_keys(feats).drop_duplicates(KEY, keep=False)
    m = left.merge(right, on=KEY, how="inner")
    m = m[(m["predicted_bfsp"] > 1) & (m["bfsp"] > 1)].reset_index(drop=True)
    say(f"joined: {len(m):,} runners of {len(oos):,}")
    if len(m) < 1000:
        return 0
    e0 = np.log(m["predicted_bfsp"].to_numpy(float)) - np.log(m["bfsp"].to_numpy(float))
    race = pd.factorize(m["race_date"].dt.strftime("%Y-%m-%d") + "|" + m["track"] + "|" + m["race_time"])[0]
    month = m["race_date"].dt.to_period("M").astype(str).replace({"2025-09": "2025-10"}).to_numpy()
    e_dm = demean(e0[:, None], race)[:, 0]
    print(f"\nThe model's mean |log error| on these runners: {np.abs(e0).mean():.4f}; "
          f"months {', '.join(sorted(set(month)))}")
    rng = np.random.default_rng(0)
    m["placebo"] = np.random.default_rng(1).normal(size=len(m))
    rows = []
    for c in fcols + ["placebo"]:
        x = pd.to_numeric(m[c], errors="coerce").to_numpy(float)
        if np.isfinite(x).sum() < 100 or np.nanstd(x) == 0:
            rows.append({"feature": c, "read": float(np.isfinite(x).mean()), "delta": np.nan, "lo": np.nan,
                         "hi": np.nan})
            continue
        adj = oof_adjustment(design(x), e_dm, race, month)
        d, lo, hi = delta_with_ci(e0, adj, race, rng)
        rows.append({"feature": c, "read": float(np.isfinite(x).mean()), "delta": d, "lo": lo, "hi": hi})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "per_feature.csv", index=False)
    print("\nEach feature alone: the change in the model's mean |log error| (x 10^4; - = it explains some of the"
          " miss), 90% race-bootstrap interval; * = the interval below zero")
    for r in res.sort_values("delta").itertuples():
        flag = " *" if r.hi < 0 else ""
        print(f"  {r.feature:18s} read {r.read:5.3f}  {1e4 * r.delta:+7.2f} ({1e4 * r.lo:+7.2f} to "
              f"{1e4 * r.hi:+7.2f}){flag}")
    for c in ("tv_km", "tv_usual", "tv_km_vs_usual", "tv_track_share", "tv_n_meeting", "tv_horse_km"):
        if c in m.columns:
            q = m[c].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).round(2).to_dict()
            print(f"  {c:16s} quantiles {q}")
    if "tv_raid" in m.columns:
        print(f"  tv_raid: {int((m['tv_raid'] == 1).sum()):,} raiders among the window's runners")

    print("\nEach block jointly (ridge 100, every feature's z, z^2 and missing flag):")
    groups = {name: [c for c in b.columns] for name, b in blocks.items()}
    groups["travel less tv_n_meeting (the stable block's)"] = [c for c in groups.get("travel", [])
                                                                if c != "tv_n_meeting"]
    groups["controls (rPMW3, rPMW5: in the model)"] = [c for c in ("rPMW3", "rPMW5") if c in m.columns]
    groups["all five blocks"] = [c for b in blocks.values() for c in b.columns]
    joint = {}
    for label, cols in groups.items():
        if not cols:
            continue
        X = np.column_stack([design(pd.to_numeric(m[c], errors="coerce").to_numpy(float)) for c in cols])
        adj = oof_adjustment(X, e_dm, race, month, ridge=100.0)
        joint[label] = adj
        dd, lo, hi = delta_with_ci(e0, adj, race, rng)
        print(f"  {label:48s} {len(cols):3d} features  {1e4 * dd:+7.2f} ({1e4 * lo:+7.2f} to {1e4 * hi:+7.2f})")

    if "travel" in joint:
        print("\nThe travel block jointly, by segment (the adjustment fitted on every runner):")
        rt = m["race_type"].fillna("").astype(str).str.lower()
        ire = norm_track(m["track"]).map({k: v[2] for k, v in COURSES.items()}).fillna(-1).to_numpy()
        segs = {
            "maidens, novices, bumpers": rt.str.contains("maiden|novice|bumper|nh flat|national hunt flat").to_numpy(),
            "handicaps": rt.str.contains("handicap").to_numpy(),
            "Irish courses": ire == 1,
            "British courses": ire == 0,
            "raiders (tv_raid 1)": (m.get("tv_raid", pd.Series(np.nan, index=m.index)) == 1).to_numpy(),
            "travelled 200 km or more": (m.get("tv_km", pd.Series(np.nan, index=m.index)) >= 200).to_numpy(),
        }
        for label, rows_ in segs.items():
            if rows_.sum() < 200:
                print(f"  {label:32s} {int(rows_.sum()):6,d} runners: too few")
                continue
            dd, lo, hi = delta_with_ci(e0, joint["travel"], race, rng, rows=rows_)
            print(f"  {label:32s} {int(rows_.sum()):6,d} runners  {1e4 * dd:+7.2f} ({1e4 * lo:+7.2f} to "
                  f"{1e4 * hi:+7.2f})")
    say("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
