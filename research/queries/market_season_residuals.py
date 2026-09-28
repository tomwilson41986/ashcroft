"""Market history and seasonality against what the best model still gets wrong, screened beyond the
forecast's own correction.

The screen of travel_residuals and misc_residuals fitted each reading to the out-of-sample error on its
own. Its first run here (research query run 36359800214) showed what that measures: the Kalman rating,
the handicap angles and travel read the same against the errors of a model that already holds them
(iteration 96's 968s5xh with all three, 0.3984: Kalman -4.58, handicap -2.48, travel -2.62 in the tree)
as against one that does not (iteration 94's pair, 0.3981: -5.07, -2.44, -3.27). A correction on the
forecast alone takes -5.11 the same way, all of it in the long shots (ranks 8+ -38.35) at the top of the
market's cost (rank 1 +53.08): the within-race stretch (ledger within-race-stretch, not carried). Any
reading that tracks ability stands in for part of that stretch, so the old screen could not tell new
information from the compression, which is why it misjudged iteration 96 (Kalman and handicap angles
resolved and gave nothing at the fit; travel unresolved and carried).

So every group is now read BEYOND the forecast: the correction fitted on the forecast's own readings
(its log price, that against the race, its rank, the field size) and on the group together, against the
correction on the forecast alone, out of month, the difference in mean |log error| x 10^4 with a 90%
race-bootstrap interval, by the forecast's rank (the bets are at the top of the market) and for maidens,
novices and bumpers and Irish races (where travel carried). Twice: the quadratic as before (ridge 100) and
a small, slow gradient-boosted tree (Huber loss), with placebos of noise columns through both.

Calibration: against iteration 94's pair (no blocks) travel should read and the Kalman rating and handicap
angles should not; against iteration 96's 968s5xh with all three, none of the three should.

The new readings, each from earlier days (or earlier years) only:

    spb_*   the SP-to-BSP spread: log net odds at the Betfair SP less at the industry SP (the
            `odds` column), less the median for that price, field size and code: + = the
            exchange longer than the bookmakers. The horse's last run and decayed mean (half-life
            365 days); its yard's (180 and 30 days) and its jockey's (180), shrunk by 20 runners
    wp_*    the win/place ratio: log net win BSP less log net place BSP, less the median for that
            price, field size and places paid: + = the place price short against the win price
            (a placer, in the market's view). The horse's last run and decayed mean; its yard's
    sea_*   seasonality: the horse's finishing position (NFP) in the month either side of today's
            month in earlier years (runs at least 300 days before today), and against its NFP over
            all earlier runs; the yard's winners against the Betfair SP (A-E) in those months of
            earlier years, against all months of earlier years (shrunk by 50 runners), and its
            strike rate the same way
    wt_*    weight carried against the last run and the mean of the last three (misc_residuals'
            one reading, -2.14 in the quadratic)

Read-only; nothing is trained on, or leaves, this query.
"""
import io
import os
import sys
import time
import warnings
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "research/queries/done")
import travel_residuals as T  # noqa: E402  (the quadratic screen's helpers)

BASES = [("it94 pair, no blocks (0.3981)", 10939044871, "oos_xthub_xtdml.csv"),
         ("it96 968s5xh + Kalman, handicap angles, travel (0.3984)", 10943604587, "oos_cw_dm_xthub_kr_hca_tv.csv")]
COLS = ["race_date", "race_time", "track", "horse_name", "trainer", "jockey_name", "odds", "bfsp", "bfsp_place",
        "bf_plcs_paid", "plcs_paid", "placing_numerical", "number_of_runners", "official_rating", "median_or",
        "pounds", "total_dst_bt", "dist_furlongs", "race_type", "race_name", "race_code"]
START = "2021-01-01"
LAG_YEAR_DAYS = 300
CLIP = 2.0
T0 = time.time()


def say(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def fetch(artifact_id, name):
    import requests
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        say("no GITHUB_TOKEN: no forecasts to read")
        return None
    auth = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    z = requests.get(f"https://api.github.com/repos/{repo}/actions/artifacts/{artifact_id}/zip", headers=auth,
                     timeout=600)
    if not z.ok:
        say(f"artifact {artifact_id}: download failed ({z.status_code})")
        return None
    zf = zipfile.ZipFile(io.BytesIO(z.content))
    inner = next((n for n in zf.namelist() if n.endswith("/" + name) or n == name), None)
    if inner is None:
        say(f"artifact {artifact_id}: no {name} in {zf.namelist()[:10]}")
        return None
    return pd.read_csv(zf.open(inner))


def load():
    import sqlite3
    with sqlite3.connect("horse_racing.db") as conn:
        have = {r[1] for r in conn.execute("PRAGMA table_info(race_results)")}
        use = [c for c in COLS if c in have]
        df = pd.read_sql(f"SELECT {', '.join(use)} FROM race_results WHERE race_date >= ?", conn, params=(START,))
    say(f"race_results from {START}: {len(df):,} rows; columns missing: {[c for c in COLS if c not in have]}")
    df = df[pd.to_numeric(df["bfsp"], errors="coerce") > 1.0].reset_index(drop=True)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["raceid"] = (df["race_date"].dt.strftime("%Y-%m-%d") + "|" + df["track"].astype(str) + "|"
                    + df["race_time"].astype(str))
    say(f"priced runs: {len(df):,} in {df['raceid'].nunique():,} races")
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


def against_cell(v, cells):
    """v less the median of v in its cell, clipped (NaN where v is)."""
    med = pd.Series(v).groupby(cells).transform("median").to_numpy(float)
    return np.clip(v - med, -CLIP, CLIP)


def readings(m):
    from model.race_shape import asof_decayed_mean, day_index
    n = len(m)
    day = day_index(m)
    race = pd.factorize(m["raceid"], sort=True)[0]
    h = T._name_codes(m["horse_name"])
    tr = T._name_codes(m["trainer"])
    jk = T._name_codes(m["jockey_name"]) if "jockey_name" in m.columns else np.full(n, -1)
    bsp = pd.to_numeric(m["bfsp"], errors="coerce").to_numpy(float)
    nr = pd.to_numeric(m["number_of_runners"], errors="coerce").to_numpy(float)
    rows = pd.Series(np.ones(n)).groupby(race).transform("sum").to_numpy(float)
    nr = np.where(np.isfinite(nr) & (nr > 0), nr, rows)
    fs_band = np.digitize(nr, [6, 8, 12, 16])
    code = (pd.factorize(m["race_code"].fillna("").astype(str))[0] if "race_code" in m.columns
            else np.zeros(n, dtype=np.int64))
    pbin = pd.qcut(pd.Series(np.log(bsp)), 40, labels=False, duplicates="drop").to_numpy().astype(np.int64)
    o, p = prev_rows(h, day, race)
    has = p >= 0
    pp = np.maximum(p, 0)

    def last_of(v):
        vs = v[o]
        full = np.full(n, np.nan)
        full[o] = np.where(has, vs[pp], np.nan)
        return full

    def decayed(key, v, hl, k=0.0):
        mu, nn = asof_decayed_mean(key, day, v, key, day, hl)
        if k > 0:
            mu = np.where(nn > 0, mu * nn / (nn + k), np.nan)
        return mu, nn

    out = pd.DataFrame(index=m.index)

    # the SP-to-BSP spread
    sp = pd.to_numeric(m["odds"], errors="coerce").to_numpy(float) if "odds" in m.columns else np.full(n, np.nan)
    short = (bsp < 1.8) & np.isfinite(sp) & (sp > 0)
    share_under1 = float(np.mean(sp[short] < 1.0)) if short.any() else float("nan")
    net = bool(share_under1 > 0.05)
    say(f"odds read as {'net (fractional) odds' if net else 'decimal odds'}: {100 * share_under1:.1f}% of runners "
        f"at a BSP under 1.8 have odds under 1; odds present on {100 * np.isfinite(sp).mean():.1f}% of rows")
    sp_net = sp if net else sp - 1.0
    ok = np.isfinite(sp_net) & (sp_net > 0.01) & (bsp > 1.01)
    s = np.where(ok, np.log(np.where(ok, bsp - 1.0, 1.0)) - np.log(np.where(ok, sp_net, 1.0)), np.nan)
    s_res = against_cell(s, pbin * 1000 + fs_band * 10 + code)
    q = np.nanquantile(s_res, [0.1, 0.5, 0.9])
    say(f"spread against the cell: read on {100 * np.isfinite(s_res).mean():.1f}%, deciles {q.round(3).tolist()}")
    out["spb_last"] = last_of(s_res)
    out["spb_h"], out["spb_h_n"] = decayed(h, s_res, 365.0)
    out["spb_tr"], _ = decayed(tr, s_res, 180.0, k=20.0)
    out["spb_tr30"], _ = decayed(tr, s_res, 30.0, k=20.0)
    out["spb_jk"], _ = decayed(jk, s_res, 180.0, k=20.0)

    # the win/place ratio
    bpl = pd.to_numeric(m["bfsp_place"], errors="coerce").to_numpy(float) if "bfsp_place" in m.columns \
        else np.full(n, np.nan)
    places = np.full(n, np.nan)
    for c in ("bf_plcs_paid", "plcs_paid"):
        if c in m.columns:
            v = pd.to_numeric(m[c], errors="coerce").to_numpy(float)
            places = np.where(np.isfinite(places), places, v)
    okp = (bpl > 1.01) & (bsp > 1.01)
    w = np.where(okp, np.log(np.where(okp, bsp - 1.0, 1.0)) - np.log(np.where(okp, bpl - 1.0, 1.0)), np.nan)
    cells_w = pbin * 100000 + np.clip(nr, 0, 40).astype(np.int64) * 100 + np.nan_to_num(places, nan=0).astype(np.int64)
    w_res = against_cell(w, cells_w)
    q = np.nanquantile(w_res, [0.1, 0.5, 0.9]) if np.isfinite(w_res).any() else [np.nan] * 3
    say(f"win/place against the cell: read on {100 * np.isfinite(w_res).mean():.1f}%, deciles "
        f"{np.round(q, 3).tolist()}")
    out["wp_last"] = last_of(w_res)
    out["wp_h"], out["wp_h_n"] = decayed(h, w_res, 365.0)
    out["wp_tr"], _ = decayed(tr, w_res, 180.0, k=20.0)

    # seasonality: the month either side of today's in earlier years
    pos = pd.to_numeric(m["placing_numerical"], errors="coerce").to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        nfp = np.where((pos > 0) & (nr > 1), (nr - pos) / (nr - 1), np.nan)
    won = np.where(pos > 0, (pos == 1).astype(float), np.nan)
    month = m["race_date"].dt.month.to_numpy().astype(np.int64)
    lag_day = day + LAG_YEAR_DAYS

    def season(key, v):
        tot, cnt = np.zeros(n), np.zeros(n)
        hist = np.where(key >= 0, key * 13 + month, -1)
        for off in (-1, 0, 1):
            qm = (month - 1 + off) % 12 + 1
            mu, nn = asof_decayed_mean(hist, lag_day, v, np.where(key >= 0, key * 13 + qm, -1), day)
            tot += np.where(nn > 0, mu * nn, 0.0)
            cnt += nn
        return np.where(cnt > 0, tot / np.maximum(cnt, 1e-12), np.nan), cnt

    all_nfp, _ = asof_decayed_mean(h, day, nfp, h, day)
    sea_nfp, sea_n = season(h, nfp)
    out["sea_nfp"] = sea_nfp
    out["sea_n"] = np.where(h >= 0, sea_n, np.nan)
    out["sea_nfp_vs_all"] = sea_nfp - all_nfp
    ae = np.where(np.isfinite(won), won - 1.0 / bsp, np.nan)
    t_ae, t_n = season(tr, ae)
    t_all, t_all_n = asof_decayed_mean(tr, lag_day, ae, tr, day)
    t_sea = np.where(t_n > 0, t_ae * t_n / (t_n + 50.0), np.nan)
    out["sea_tr_ae"] = t_sea
    out["sea_tr_ae_vs_all"] = t_sea - np.where(t_all_n > 0, t_all * t_all_n / (t_all_n + 50.0), np.nan)
    t_wr, t_wn = season(tr, won)
    t_wall, _ = asof_decayed_mean(tr, lag_day, won, tr, day)
    out["sea_tr_wr_vs_all"] = np.where(t_wn >= 20, t_wr - t_wall, np.nan)

    # weight against the last run and the last three
    wt = pd.to_numeric(m["pounds"], errors="coerce").where(lambda x: x > 0).to_numpy(float)
    ws = wt[o]
    p2 = np.where(has, p[pp], -1)
    p3 = np.where(p2 >= 0, p[np.maximum(p2, 0)], -1)
    w1 = np.where(has, ws[pp], np.nan)
    w2 = np.where(p2 >= 0, ws[np.maximum(p2, 0)], np.nan)
    w3 = np.where(p3 >= 0, ws[np.maximum(p3, 0)], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean3 = np.nanmean(np.vstack([w1, w2, w3]), axis=0)
    for name, arr in (("wt_vs_last", ws - w1), ("wt_vs_mean3", ws - mean3)):
        full = np.full(n, np.nan)
        full[o] = arr
        out[name] = full
    return out


def blocks(m):
    from model.blocks import handicap_angles, kalman, travel
    res = {}
    for label, mod in (("travel", travel), ("Kalman rating", kalman), ("handicap angles", handicap_angles)):
        missing = [c for c in mod.READS if c not in m.columns]
        if missing:
            say(f"{label}: columns missing {missing}; not built")
            continue
        t = time.time()
        b = mod.build(m[list(mod.READS)].copy())
        res[label] = b[mod.FEATURES].reset_index(drop=True)
        say(f"{label}: built in {time.time() - t:.0f}s")
    return res


# ----------------------------------------------------------------------------- the tree screen

def race_mean(X, race):
    return pd.DataFrame(X).groupby(race).transform("mean").to_numpy()


def tree_adjustment(X, e_dm, race, month, seed=0):
    """A small, slow gradient-boosted tree (Huber loss, 7 leaves of 1,500 runners or more, shrunk) on the
    features and their within-race deviations, fitted to the within-race error out of month; its prediction,
    centred within the race, is the correction. On planted signals at this scale (55k runners, heavy-tailed
    errors) it costs a nine-column group of noise about +1 x 10^-4, as the quadratic does, and finds a tail
    effect and a small linear one that the quadratic misses; an L1 tree at 15 leaves cost noise +7.7."""
    import lightgbm as lgb
    X = np.asarray(X, dtype=float)
    F = np.column_stack([X, X - race_mean(X, race)]).astype(np.float32)
    params = dict(objective="huber", alpha=0.3, learning_rate=0.02, num_leaves=7, min_data_in_leaf=1500,
                  lambda_l2=100.0, feature_fraction=1.0, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=seed,
                  deterministic=True, force_col_wise=True)
    adj = np.zeros(len(e_dm))
    for mo in np.unique(month):
        te = month == mo
        b = lgb.train(params, lgb.Dataset(F[~te], e_dm[~te]), num_boost_round=150)
        adj[te] = b.predict(F[te])
    return adj - pd.Series(adj).groupby(race).transform("mean").to_numpy()


def fmt(d, lo, hi):
    return f"{1e4 * d:+7.2f} ({1e4 * lo:+7.2f} to {1e4 * hi:+7.2f}){' *' if hi < 0 else '  '}"


def forecast_design(m, race):
    """The forecast's own readings: its log price, that against the race's mean, its rank as a share of the
    field, and the field size. A correction on these alone is the within-race stretch (ledger
    within-race-stretch): all long shots, at the top of the market's cost."""
    lp = np.log(m["predicted_bfsp"].to_numpy(float))
    lp_c = lp - pd.Series(lp).groupby(race).transform("mean").to_numpy()
    rank = pd.Series(lp).groupby(race).rank(method="first").to_numpy()
    nr = pd.Series(np.ones(len(m))).groupby(race).transform("sum").to_numpy()
    return np.column_stack([lp, lp_c, rank / nr, nr]), rank


def incr_ci(e0, adj_base, adj_full, race_codes, rng, rows=None):
    """Mean of |e0 - full| - |e0 - base| with a 90% race-bootstrap interval: what the readings add beyond the
    correction on the forecast alone (negative = they explain some of the miss the forecast does not)."""
    d = np.abs(e0 - adj_full) - np.abs(e0 - adj_base)
    rc = race_codes
    if rows is not None:
        d, rc = d[rows], pd.factorize(race_codes[rows])[0]
    sums = np.bincount(rc, weights=d)
    counts = np.bincount(rc).astype(float)
    boots = np.empty(T.N_BOOT)
    for i in range(T.N_BOOT):
        pick = rng.integers(0, len(sums), len(sums))
        boots[i] = sums[pick].sum() / counts[pick].sum()
    return d.mean(), np.quantile(boots, 0.05), np.quantile(boots, 0.95)


def screen(label, oos, frame, groups, singles):
    print(f"\n===== {label} =====")
    keep = [c for c in T.KEY + ["predicted_bfsp", "bfsp", "race_type", "race_class"] if c in oos.columns]
    left = T._norm_keys(oos[keep]).drop_duplicates(T.KEY, keep=False)
    right = T._norm_keys(frame).drop_duplicates(T.KEY, keep=False)
    m = left.merge(right, on=T.KEY, how="inner")
    m = m[(m["predicted_bfsp"] > 1) & (m["bfsp"] > 1)].reset_index(drop=True)
    e0 = np.log(m["predicted_bfsp"].to_numpy(float)) - np.log(m["bfsp"].to_numpy(float))
    race = pd.factorize(m["race_date"].dt.strftime("%Y-%m-%d") + "|" + m["track"] + "|" + m["race_time"])[0]
    month = m["race_date"].dt.to_period("M").astype(str).replace({"2025-09": "2025-10"}).to_numpy()
    e_dm = T.demean(e0[:, None], race)[:, 0]
    print(f"{len(m):,} of {len(left):,} forecasts joined; mean |log error| {np.abs(e0).mean():.4f}")
    rt = m["race_type"].fillna("").astype(str).str.lower() if "race_type" in m.columns else pd.Series([""] * len(m))
    irish = (m["race_class"].astype(str) == "Irish").to_numpy() if "race_class" in m.columns else np.zeros(len(m), bool)
    C, rank = forecast_design(m, race)
    Cq = np.column_stack([T.design(C[:, j]) for j in range(C.shape[1])])
    segs = {"rank 1": rank == 1, "ranks 2-3": (rank >= 2) & (rank <= 3), "ranks 4-7": (rank >= 4) & (rank <= 7),
            "ranks 8+": rank >= 8, "mdn/nov/bmp": rt.str.contains("maiden|novice|bumper|nh flat").to_numpy(),
            "Irish": irish}
    rng = np.random.default_rng(0)
    zero = np.zeros(len(m))
    base_q = T.oof_adjustment(Cq, e_dm, race, month, ridge=100.0)
    base_t = tree_adjustment(C, e_dm, race, month)
    print("the correction on the forecast alone (the stretch; x 10^4): quadratic "
          f"{fmt(*incr_ci(e0, zero, base_q, race, rng))}, tree {fmt(*incr_ci(e0, zero, base_t, race, rng))}")
    print("   tree by segment: " + ", ".join(f"{s} {1e4 * incr_ci(e0, zero, base_t, race, rng, rows=r)[0]:+.2f}"
                                          for s, r in segs.items()))
    noise = np.random.default_rng(1).normal(size=(len(m), 12))
    rows_out = []
    print("\nBeyond the forecast: each group added to the correction on the forecast (x 10^4; * = interval below zero)")
    print(f"{'group':28s} {'k':>3s}   {'uncorrected tree':16s}   {'quadratic beyond':27s}   {'tree beyond':27s}   "
          + "  ".join(f"{s:>11s}" for s in segs))
    all_groups = list(groups.items()) + [("placebo, 6 noise columns", None), ("placebo, 12 noise columns", None)]
    for gl, cols in all_groups:
        if cols is None:
            k = 6 if "6 " in gl else 12
            X = noise[:, :k]
        else:
            cols = [c for c in cols if c in m.columns]
            if not cols:
                continue
            X = np.column_stack([pd.to_numeric(m[c], errors="coerce").to_numpy(float) for c in cols])
            k = len(cols)
        t = time.time()
        raw_t = incr_ci(e0, zero, tree_adjustment(X, e_dm, race, month), race, rng)
        Xq = np.column_stack([T.design(X[:, j]) for j in range(X.shape[1])])
        q = incr_ci(e0, base_q, T.oof_adjustment(np.column_stack([Cq, Xq]), e_dm, race, month, ridge=100.0), race, rng)
        full_t = tree_adjustment(np.column_stack([C, X]), e_dm, race, month)
        tt = incr_ci(e0, base_t, full_t, race, rng)
        seg_txt = []
        for s, rows in segs.items():
            if rows.sum() > 500:
                d, lo, hi = incr_ci(e0, base_t, full_t, race, rng, rows=rows)
                seg_txt.append(f"{1e4 * d:+10.2f}{'*' if hi < 0 else ' '}")
            else:
                seg_txt.append(f"{'-':>11s}")
        print(f"{gl:28s} {k:3d}   {1e4 * raw_t[0]:+7.2f}{'*' if raw_t[2] < 0 else ' '}          {fmt(*q)}   {fmt(*tt)}   "
              + "  ".join(seg_txt) + f"   [{time.time() - t:.0f}s]", flush=True)
        rows_out.append(dict(base=label, group=gl, k=k, raw_tree=raw_t[0], quad=q[0], quad_lo=q[1], quad_hi=q[2],
                             tree=tt[0], tree_lo=tt[1], tree_hi=tt[2]))
    if singles:
        print("\nEach new reading alone, beyond the forecast (x 10^4; * = the interval below zero)")
        for c in singles:
            if c not in m.columns:
                continue
            x = pd.to_numeric(m[c], errors="coerce").to_numpy(float)
            if np.isfinite(x).sum() < 100 or np.nanstd(x) == 0:
                continue
            q = incr_ci(e0, base_q, T.oof_adjustment(np.column_stack([Cq, T.design(x)]), e_dm, race, month,
                                                     ridge=100.0), race, rng)
            full_t = tree_adjustment(np.column_stack([C, x]), e_dm, race, month)
            tt = incr_ci(e0, base_t, full_t, race, rng)
            top = incr_ci(e0, base_t, full_t, race, rng, rows=rank <= 3)
            print(f"  {c:18s} read {np.isfinite(x).mean():5.3f}   quad {fmt(*q)}   tree {fmt(*tt)}   "
                  f"ranks 1-3 {1e4 * top[0]:+6.2f}{'*' if top[2] < 0 else ' '}", flush=True)
    return rows_out


def main():
    out_dir = T.OUT.parent / "market_season_residuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    oos = {}
    for label, art, name in BASES:
        f = fetch(art, name)
        if f is not None:
            oos[label] = f
            say(f"{label}: {len(f):,} forecasts")
    if not oos:
        return 0
    mat = load()
    feats = readings(mat)
    say(f"readings: {feats.shape[1]}")
    blk = blocks(mat)
    frame = pd.concat([mat[T.KEY].reset_index(drop=True), feats.reset_index(drop=True)]
                      + [b for b in blk.values()], axis=1)
    del mat
    new = {"SP-to-BSP spread": [c for c in feats.columns if c.startswith("spb_")],
           "win/place ratio": [c for c in feats.columns if c.startswith("wp_")],
           "seasonality": [c for c in feats.columns if c.startswith("sea_")],
           "weight": [c for c in feats.columns if c.startswith("wt_")]}
    groups = {**{k: list(v.columns) for k, v in blk.items()}, **new,
              "all new readings": [c for c in feats.columns if not c.startswith("wt_")]}
    results = []
    for i, (label, _, _) in enumerate(BASES):
        if label in oos:
            results += screen(label, oos[label], frame, groups, singles=list(feats.columns) if i == 1 else [])
    pd.DataFrame(results).to_csv(out_dir / "screen.csv", index=False)
    say("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
