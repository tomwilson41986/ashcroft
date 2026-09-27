"""Seven more readings no block has, against what the best model still gets wrong.

travel_residuals found the yard's trip reads only where the error is largest. The rest of the
not-yet-developed list (reports/feature_inventory.xlsx) is screened the same way, every reading
from the 06:00 card or from earlier days:

    jk_km             km from today's course to the jockey's base (the decayed centre of his or her
                      rides' courses on earlier days), once there are five rides' worth
    jk_km_vs_usual    jk_km less the jockey's usual trip
    jk_rides_meeting  the jockey's rides at today's meeting (the card)
    jk_rides_day      the jockey's rides anywhere today
    jk_single_far     jk_km when it is the jockey's only ride at the meeting, else 0
    wt_vs_last        weight carried today less at the horse's last run (lb)
    wt_vs_mean3       weight carried today less the mean of its last three runs
    cls_vs_median     today's class less the median class of the horse's earlier runs (+ = lower)
    cls_vs_best       today's class less the best (lowest) class it has run in
    or_vs_peak        today's official rating less its highest before today
    or_slope3         the change in its official rating over its last three runs, per run
    or_pct_field      today's official rating as a percentile of the field's
    hg_items          headgear items worn today; hg_removed: wore some last time, none today;
                      hg_blnk, hg_vsor, hg_ckpc, hg_hood, hg_tt, hg_eye, hg_h_b: each item worn today
    lpp_last          how strung out its last race was: the last finisher's lengths behind per place
    lb_vs_lpp_last    its lengths behind in that race against the race's spread

The first-time headgear, the class change and the claim change against the last run are the
intent block's, already in the model. The screen is future_form_residuals' and
travel_residuals': each feature's within-race quadratic fitted out of month to the pair's
out-of-sample log error (iteration 94's in-run pair, 0.3981), the change in mean |log error|
x 10^4 with a 90% race-bootstrap interval, then groups jointly (ridge 100). Read-only.
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "research/queries/done")
import travel_residuals as T  # noqa: E402  (the screen's helpers and the course table)

WANT = T.KEY + ["raceid", "trainer", "jockey_name", "official_rating", "pounds", "placing_numerical",
                "number_of_runners", "total_dst_bt", "race_class", "headgear", "dist_furlongs", "rPMW3", "rPMW5"]
#: horseracebase's headgear tokens (model/intent_features.HEADGEAR_ITEMS)
ITEMS = {"blnk": "blinkers", "vsor": "visor", "ckpc": "cheekpieces", "hood": "hood", "tt": "tongue tie",
         "eye": "eyeshield", "h+b": "hood and blinkers"}
T0 = time.time()


def say(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def load():
    """The matrix's rows: from the feature cache when one is keyed to today's database, else the same rows (runs from
    2021 with a usable price) straight from race_results, which is all these readings need (the controls, engine
    columns, are then absent)."""
    import sqlite3
    import pyarrow.parquet as pq
    from model import feature_cache
    key = feature_cache.cache_key("horse_racing.db", "2021-01-01")
    data, _ = feature_cache._paths(".feature_cache", key)
    if data.exists():
        have = set(pq.read_schema(data).names)
        say(f"matrix columns missing: {[c for c in WANT if c not in have]}")
        return pd.read_parquet(data, columns=[c for c in WANT if c in have])
    say(f"no cached matrix for key {key}: reading race_results")
    cols = [c for c in WANT if c not in ("raceid", "rPMW3", "rPMW5")]
    with sqlite3.connect("horse_racing.db") as conn:
        have = {r[1] for r in conn.execute("PRAGMA table_info(race_results)")}
        use = [c for c in cols if c in have]
        df = pd.read_sql(f"SELECT {', '.join(use)}, bfsp FROM race_results WHERE race_date >= '2021-01-01'", conn)
    df = df[pd.to_numeric(df["bfsp"], errors="coerce") > 1.0].drop(columns="bfsp").reset_index(drop=True)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["raceid"] = df["race_date"].dt.strftime("%Y-%m-%d") + "|" + df["race_time"].astype(str) + "|" + df["track"].astype(str)
    say(f"race_results: {len(df):,} priced runs from 2021; columns missing: {[c for c in cols if c not in have]}")
    return df


def prev_rows(h, day, race):
    """Sorted order, and for each sorted position the position of the horse's last run on an earlier day (-1)."""
    n = len(h)
    o = np.lexsort((race, day, h))
    hs, ds = h[o], day[o]
    idx = np.arange(n)
    new_h = np.r_[True, hs[1:] != hs[:-1]]
    new_day = new_h | np.r_[True, ds[1:] != ds[:-1]]
    start = np.maximum.accumulate(np.where(new_day, idx, 0))
    p = start - 1
    ok = (p >= 0) & (hs >= 0) & np.where(p >= 0, hs[np.maximum(p, 0)] == hs, False)
    return o, np.where(ok, p, -1)


def readings(m):
    from model.race_shape import asof_decayed_mean, day_index
    from model.perf_figures import parse_beaten_lengths
    n = len(m)
    day = day_index(m)
    race = pd.factorize(m["raceid"].astype(str), sort=True)[0]
    h = T._name_codes(m["horse_name"])
    out = pd.DataFrame(index=m.index)

    # the jockey's travel and rides
    if "jockey_name" in m.columns:
        jk = T._name_codes(m["jockey_name"])
        course = T.norm_track(m["track"])
        table = pd.DataFrame.from_dict(T.COURSES, orient="index", columns=["lat", "lon", "ire"])
        lat, lon = course.map(table["lat"]).to_numpy(float), course.map(table["lon"]).to_numpy(float)
        blat, nb = asof_decayed_mean(jk, day, lat, jk, day, T.HL)
        blon, _ = asof_decayed_mean(jk, day, lon, jk, day, T.HL)
        km = np.where(nb >= 5, T.km(lat, lon, blat, blon), np.nan)
        usual, nu = asof_decayed_mean(jk, day, km, jk, day, T.HL)
        meet = pd.factorize(m["race_date"].astype(str) + "|" + course, sort=True)[0]
        ones = pd.Series(np.ones(n))
        rides_meet = ones.groupby([meet, jk]).transform("sum").to_numpy(float)
        rides_day = ones.groupby([day, jk]).transform("sum").to_numpy(float)
        out["jk_km"] = km
        out["jk_km_vs_usual"] = km - np.where(nu >= 5, usual, np.nan)
        out["jk_rides_meeting"] = np.where(jk >= 0, rides_meet, np.nan)
        out["jk_rides_day"] = np.where(jk >= 0, rides_day, np.nan)
        out["jk_single_far"] = np.where(np.isfinite(km), np.where(rides_meet == 1, km, 0.0), np.nan)

    o, p = prev_rows(h, day, race)
    has = p >= 0
    pp = np.maximum(p, 0)

    def at_prev(sorted_vals):
        return np.where(has, sorted_vals[pp], np.nan)

    def to_rows(sorted_vals):
        full = np.full(n, np.nan)
        full[o] = sorted_vals
        return full

    # weight
    wt = pd.to_numeric(m["pounds"], errors="coerce").where(lambda s: s > 0).to_numpy(float)
    ws = wt[o]
    out["wt_vs_last"] = to_rows(ws - at_prev(ws))
    # the mean of the last run and the two before it, each found as the previous run on an earlier day
    p2 = np.where(has, p[pp], -1)
    p3 = np.where(p2 >= 0, p[np.maximum(p2, 0)], -1)
    w2 = np.where(p2 >= 0, ws[np.maximum(p2, 0)], np.nan)
    w3 = np.where(p3 >= 0, ws[np.maximum(p3, 0)], np.nan)
    with np.errstate(all="ignore"):
        mean3 = np.nanmean(np.vstack([at_prev(ws), w2, w3]), axis=0)
    out["wt_vs_mean3"] = to_rows(ws - mean3)

    # class ladder: Class 1 the top; Irish races carry no class
    cls = pd.to_numeric(m["race_class"].astype(str).str.extract(r"(\d)")[0], errors="coerce").to_numpy(float)
    cs = cls[o]
    med, best = np.full(n, np.nan), np.full(n, np.nan)
    orr = pd.to_numeric(m["official_rating"], errors="coerce").where(lambda s: s > 0).to_numpy(float)
    ors = orr[o]
    peak = np.full(n, np.nan)
    # running statistics over earlier days, per horse (a loop in sorted order; runs of a day see only earlier days)
    hs_, ds_ = h[o], day[o]
    cur, hist_c, hist_o, day_buf = -2, [], [], []
    cur_day = None
    for i in range(n):
        hh = hs_[i]
        if hh != cur:
            cur, hist_c, hist_o, day_buf, cur_day = hh, [], [], [], ds_[i]
        elif ds_[i] != cur_day:
            for c_, r_ in day_buf:
                if np.isfinite(c_):
                    hist_c.append(c_)
                if np.isfinite(r_):
                    hist_o.append(r_)
            day_buf, cur_day = [], ds_[i]
        if hist_c:
            med[i], best[i] = np.median(hist_c), min(hist_c)
        if hist_o:
            peak[i] = max(hist_o)
        day_buf.append((cs[i], ors[i]))
    out["cls_vs_median"] = to_rows(cs - med)
    out["cls_vs_best"] = to_rows(cs - best)
    out["or_vs_peak"] = to_rows(ors - peak)
    o3 = np.where(p2 >= 0, ors[np.maximum(p2, 0)], np.nan)
    o4 = np.where(p3 >= 0, ors[np.maximum(p3, 0)], np.nan)
    out["or_slope3"] = to_rows((at_prev(ors) - np.where(np.isfinite(o4), o4, o3)) / np.where(np.isfinite(o4), 2.0, 1.0))
    out["or_pct_field"] = pd.Series(orr).groupby(race).rank(pct=True).to_numpy(float)

    # headgear
    if "headgear" in m.columns:
        hg = " " + m["headgear"].fillna("").astype(str).str.lower().str.strip() + " "
        items = np.zeros(n)
        for code in ITEMS:
            worn = hg.str.contains(f" {code} ", regex=False).to_numpy()
            out[f"hg_{code.replace('+', '_')}"] = worn.astype(float)
            items += worn
        out["hg_items"] = items
        wore = (items > 0).astype(float)[o]
        out["hg_removed"] = to_rows(np.where(has, ((at_prev(wore) == 1) & (wore == 0)).astype(float), np.nan))

    # how strung out the last race was
    pos = pd.to_numeric(m["placing_numerical"], errors="coerce").to_numpy(float)
    lb = m["total_dst_bt"].map(parse_beaten_lengths).to_numpy(float)
    lb = np.where(pos == 1, 0.0, lb)
    spread = pd.Series(np.where(np.isfinite(pos), lb, np.nan)).groupby(race).transform("max").to_numpy(float)
    places = pd.Series(np.where(np.isfinite(pos) & np.isfinite(lb), pos, np.nan)).groupby(race).transform("max")
    lpp = spread / np.maximum(places.to_numpy(float) - 1, 1)
    out["lpp_last"] = to_rows(at_prev(lpp[o]))
    out["lb_vs_lpp_last"] = to_rows(at_prev((lb / np.where(lpp > 0, lpp, np.nan))[o]))
    return out


def main():
    T.OUT = T.OUT.parent / "misc_residuals"
    T.OUT.mkdir(parents=True, exist_ok=True)
    oos = T.fetch_oos()
    if oos is None:
        return 0
    mat = load()
    if mat is None:
        return 0
    say(f"forecasts {len(oos):,}, matrix {len(mat):,}")
    feats = readings(mat)
    say(f"readings: {feats.shape[1]}")
    frame = pd.concat([mat[T.KEY], feats], axis=1)
    for c in ("rPMW3", "rPMW5"):
        if c in mat.columns:
            frame[c] = mat[c].to_numpy()
    del mat
    fcols = [c for c in frame.columns if c not in T.KEY]
    left = T._norm_keys(oos[T.KEY + ["predicted_bfsp", "bfsp", "race_type"]]).drop_duplicates(T.KEY, keep=False)
    right = T._norm_keys(frame).drop_duplicates(T.KEY, keep=False)
    m = left.merge(right, on=T.KEY, how="inner")
    m = m[(m["predicted_bfsp"] > 1) & (m["bfsp"] > 1)].reset_index(drop=True)
    say(f"joined {len(m):,}")
    e0 = np.log(m["predicted_bfsp"].to_numpy(float)) - np.log(m["bfsp"].to_numpy(float))
    race = pd.factorize(m["race_date"].dt.strftime("%Y-%m-%d") + "|" + m["track"] + "|" + m["race_time"])[0]
    month = m["race_date"].dt.to_period("M").astype(str).replace({"2025-09": "2025-10"}).to_numpy()
    e_dm = T.demean(e0[:, None], race)[:, 0]
    print(f"\nThe pair's mean |log error| on these runners: {np.abs(e0).mean():.4f}")
    rng = np.random.default_rng(0)
    m["placebo"] = np.random.default_rng(1).normal(size=len(m))
    rows = []
    for c in fcols + ["placebo"]:
        x = pd.to_numeric(m[c], errors="coerce").to_numpy(float)
        if np.isfinite(x).sum() < 100 or np.nanstd(x) == 0:
            continue
        adj = T.oof_adjustment(T.design(x), e_dm, race, month)
        d, lo, hi = T.delta_with_ci(e0, adj, race, rng)
        rows.append((c, float(np.isfinite(x).mean()), d, lo, hi, float(np.nanmean(x))))
    print("\nEach reading alone (x 10^4; - = it explains some of the miss; * = the interval below zero)")
    for c, rd, d, lo, hi, mu in sorted(rows, key=lambda r: r[2]):
        print(f"  {c:18s} read {rd:5.3f} mean {mu:+9.3f}  {1e4 * d:+7.2f} ({1e4 * lo:+7.2f} to {1e4 * hi:+7.2f})"
              f"{' *' if hi < 0 else ''}")
    groups = {"jockey travel and rides": [c for c in fcols if c.startswith("jk_")],
              "weight": [c for c in fcols if c.startswith("wt_")],
              "class and rating ladder": [c for c in fcols if c.startswith(("cls_", "or_"))],
              "headgear items": [c for c in fcols if c.startswith("hg_")],
              "last race's spread": [c for c in fcols if c.startswith(("lpp_", "lb_vs"))],
              "controls (in the model)": [c for c in ("rPMW3", "rPMW5") if c in fcols],
              "all readings": [c for c in fcols if c not in ("rPMW3", "rPMW5")]}
    print("\nGroups jointly (ridge 100)")
    for label, cols in groups.items():
        if not cols:
            continue
        X = np.column_stack([T.design(pd.to_numeric(m[c], errors="coerce").to_numpy(float)) for c in cols])
        adj = T.oof_adjustment(X, e_dm, race, month, ridge=100.0)
        d, lo, hi = T.delta_with_ci(e0, adj, race, rng)
        print(f"  {label:28s} {len(cols):3d} features  {1e4 * d:+7.2f} ({1e4 * lo:+7.2f} to {1e4 * hi:+7.2f})")
    say("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
