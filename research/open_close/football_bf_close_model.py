"""A closing-price model trained on Betfair's own prices: the move from a checkpoint to the close.

racing² football research, 11 Nov 2026 (the owner: "use it in modelling and to set up our own models"; the model
review's step 1: a Betfair-native, multi-horizon target). The warehouse's ``gold/move`` gives, for every match-odds
runner at every checkpoint, the move of its normalised chance to the close (``y_logit``, in log odds) with the
quality of both ends. This fits that move from what was known at the checkpoint and scores it out of sample.

**Rows.** Match odds, checkpoints 360, 120, 60 and 15 minutes before the scheduled kick-off, where the checkpoint's
book is fresh (``cp_fresh``: the gold/move screen on the checkpoint's side only: known at the time). The headline
is scored where the closing book is fresh too (``usable``: a stale close is not a close); the wider set is reported
beside it. One model a role (home, draw, away), the checkpoint as a feature.

**Features**, each known at the checkpoint's moment:
- *level*: the chance at the checkpoint (log odds), the book's sum and staleness, the runner's price updates so far
  (activity; BASIC has no volume), the minutes to the off, the hour and weekday of the kick-off;
- *path* (``gold/path``): what the price did before the checkpoint;
- *implied* (``gold/implied``): the total, supremacy, draw premium, the handicap against the 1X2;
- *memory*: each side's own early-to-close moves in its earlier matches (the last ten that kicked off before this
  checkpoint, shrunk to zero), and the same for the country's matches (the last 500);
- *rating* (``gold/rating_gap``): market-implied team ratings from earlier closes against the price now.
A feature table that is not built is left out and named in the output.

**Validation.** Walk-forward by calendar month of kick-off: each month from the fourth is forecast by models
fitted on the matches that kicked off at least a day before it began. Ridge and LightGBM, their mean. Against
"no move" (the martingale): skill = 1 - SSE(forecast) / SSE(zero), with a 95 per cent interval from resampled
match days; correlation, slope, and the share of the strongest tenth of forecasts that moved the right way.
Each feature family is also scored added to the level features (paired by day against the level arm).

**Trades** (an upper bound: last traded prices, not quotes): where the forecast's size clears a threshold, back at
the checkpoint and lay at the close (or the reverse), per unit staked in log price, less the round trip measured
at our recorded books (2.3 per cent, reports/football_model_rethink.md) and 2 per cent commission on a win.

One year of data (2026): development evidence. The untouched test is the matches from 12 Nov 2026 and the earlier
years once the owner's download of 2016-2025 is ingested.

    python -m research.open_close.football_bf_close_model --out-json out/bf_close_model.json [--root <local folder>]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

HORIZONS = (360, 120, 60, 15)
ROLES = ("home", "draw", "away")
MIN_AGE_S, AGE_SHARE = 900.0, 0.25           # gold/move's screen, the checkpoint's side
ROUND_TRIP, COMMISSION = 0.023, 0.02
THRESHOLDS = (0.03, 0.05, 0.08)
MEMORY_N, MEMORY_K = 10, 5.0
COUNTRY_N, COUNTRY_K = 500, 100.0
RIDGE_ALPHA, CLIP = 50.0, 4.0
KEY = ["market_id", "selection_id", "handicap", "cp_min"]
LEVEL = ["x_cp", "cp_book_sum", "cp_age_s", "cp_book_age_s", "log_ticks_before", "cp_min", "ko_hour", "ko_dow"]


# ---------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------

def load(store) -> tuple[pd.DataFrame, dict]:
    """The modelling frame (one row a match-odds runner a checkpoint) and {family: columns}."""
    from football.warehouse.store_io import read_table
    mv = read_table(store, "gold/move")
    if mv is None:
        return pd.DataFrame(), {}
    d = mv.to_pandas()
    d = d[(d["market_type"] == "MATCH_ODDS") & d["cp_min"].isin(HORIZONS) & d["role"].isin(ROLES)].copy()
    ev = read_table(store, "gold/event", ["event_id", "home_id", "away_id"]).to_pandas()
    d = d.merge(ev, on="event_id", how="left")
    d["cp_fresh"] = d["cp_book_age_s"] <= np.maximum(MIN_AGE_S, AGE_SHARE * d["cp_min"] * 60.0)
    d["x_cp"] = np.log(d["p_cp"] / (1 - d["p_cp"]))
    d["log_ticks_before"] = np.log1p(d["n_ticks_before"])
    d["ko_hour"] = d["kickoff"].dt.hour + d["kickoff"].dt.minute / 60.0
    d["ko_dow"] = d["kickoff"].dt.dayofweek
    d["day"] = d["kickoff"].dt.strftime("%Y-%m-%d")
    d["month"] = d["kickoff"].dt.strftime("%Y-%m")
    fam: dict[str, list[str]] = {"level": list(LEVEL)}

    path = read_table(store, "gold/path")
    if path is not None:
        p = path.to_pandas()
        cols = [c for c in p.columns if c not in KEY + ["event_id", "market_type", "role", "cp_time", "year", "month"]
                and pd.api.types.is_numeric_dtype(p[c])]
        d = d.merge(p[KEY + cols].rename(columns={c: f"path_{c}" for c in cols}), on=KEY, how="left")
        fam["path"] = [f"path_{c}" for c in cols]

    imp = read_table(store, "gold/implied", ["event_id", "cp_min", "total", "supremacy", "rho", "sup_gap", "ah_line",
                                              "p_over", "ou_age_s", "ah_age_s", "ah_lines"])
    if imp is not None:
        i = imp.to_pandas()
        cols = [c for c in i.columns if c not in ("event_id", "cp_min")]
        d = d.merge(i.rename(columns={c: f"imp_{c}" for c in cols}), on=["event_id", "cp_min"], how="left")
        fam["implied"] = [f"imp_{c}" for c in cols]

    gap = read_table(store, "gold/rating_gap")
    if gap is not None:
        g = gap.to_pandas()
        cols = [c for c in g.columns if c not in ("event_id", "cp_min", "cp_time", "kickoff", "year", "month")
                and pd.api.types.is_numeric_dtype(g[c]) and not c.endswith("_id")]
        d = d.merge(g[["event_id", "cp_min"] + cols].rename(columns={c: f"rate_{c}" for c in cols}),
                    on=["event_id", "cp_min"], how="left")
        fam["rating"] = [f"rate_{c}" for c in cols]

    d = memory(d)
    fam["memory"] = ["mm_own", "mm_own_n", "mm_opp", "mm_opp_n", "mm_country", "mm_country_n"]
    return d.reset_index(drop=True), fam


def prior_mean(group: np.ndarray, when: np.ndarray, value: np.ndarray, q_group: np.ndarray, q_when: np.ndarray,
               window: int, k: float) -> tuple[np.ndarray, np.ndarray]:
    """For each query (group, moment): the sum of ``value`` over the group's last ``window`` rows that happened
    strictly before the moment, over (their count + k), and that count. Vectorised: rows sorted by (group, when),
    one prefix sum, two binary searches a query."""
    ok = np.isfinite(value)
    group, when, value = group[ok], when[ok], value[ok]
    order = np.lexsort((when, group))
    g, t, v = group[order], when[order], value[order]
    span = np.int64(1) << np.int64(42)
    comp = g.astype(np.int64) * span + t.astype(np.int64)
    cs = np.concatenate([[0.0], np.cumsum(v)])
    qg = q_group.astype(np.int64)
    hi = np.searchsorted(comp, qg * span + q_when.astype(np.int64), side="left")       # strictly before
    lo = np.maximum(np.searchsorted(comp, qg * span, side="left"), hi - window)
    n = (hi - lo).astype(float)
    return (cs[hi] - cs[lo]) / (n + k), n


def memory(d: pd.DataFrame) -> pd.DataFrame:
    """Each side's and each country's earlier moves at the same checkpoint (matches that kicked off before this
    checkpoint's moment; the reliable ones only)."""
    base = d["kickoff"].min()
    ko = ((d["kickoff"] - base).dt.total_seconds() // 60).to_numpy(np.int64)            # minutes
    cp = ko - d["cp_min"].to_numpy(np.int64)                                           # the decision moment
    cpi = d["cp_min"].map({c: i for i, c in enumerate(HORIZONS)}).to_numpy(np.int64)
    team = np.where(d["role"] == "home", d["home_id"], np.where(d["role"] == "away", d["away_id"], np.nan))
    opp = np.where(d["role"] == "home", d["away_id"], np.where(d["role"] == "away", d["home_id"], np.nan))
    ids = pd.factorize(np.concatenate([d["home_id"].to_numpy(float), d["away_id"].to_numpy(float)]))[0]
    hid, aid = ids[:len(d)], ids[len(d):]
    own_code = np.where(d["role"] == "home", hid, np.where(d["role"] == "away", aid, -1))
    opp_code = np.where(d["role"] == "home", aid, np.where(d["role"] == "away", hid, -1))
    y = np.where(d["usable"].to_numpy(bool), d["y_logit"].to_numpy(float), np.nan)
    side = (d["role"] != "draw").to_numpy() & np.isfinite(team.astype(float)) & (own_code >= 0)
    # history: a side's own move in a match (home or away row), keyed by (team, checkpoint)
    hist_g = (own_code * len(HORIZONS) + cpi)[side]
    hist_t, hist_v = ko[side], y[side]
    out = d.copy()
    for name, code in (("mm_own", own_code), ("mm_opp", opp_code)):
        q_ok = code >= 0
        m, n = np.full(len(d), np.nan), np.full(len(d), np.nan)
        m[q_ok], n[q_ok] = prior_mean(hist_g, hist_t, hist_v, (code * len(HORIZONS) + cpi)[q_ok], cp[q_ok],
                                      MEMORY_N, MEMORY_K)
        out[name], out[f"{name}_n"] = m, n
    del opp
    # the draw row: the two sides' memories as home and away
    is_draw = (d["role"] == "draw").to_numpy()
    if is_draw.any():
        for name, code in (("mm_own", hid), ("mm_opp", aid)):
            m, n = prior_mean(hist_g, hist_t, hist_v, (code * len(HORIZONS) + cpi)[is_draw], cp[is_draw],
                              MEMORY_N, MEMORY_K)
            out.loc[is_draw, name], out.loc[is_draw, f"{name}_n"] = m, n
    # the country's matches, same role and checkpoint
    ccode = pd.factorize(d["country_code"].fillna("(none)"))[0].astype(np.int64)
    rcode = d["role"].map({r: i for i, r in enumerate(ROLES)}).to_numpy(np.int64)
    cg = (ccode * len(ROLES) + rcode) * len(HORIZONS) + cpi
    out["mm_country"], out["mm_country_n"] = prior_mean(cg, ko, y, cg, cp, COUNTRY_N, COUNTRY_K)
    return out


# ---------------------------------------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------------------------------------

def ridge_fit_predict(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str], y: np.ndarray) -> np.ndarray:
    X, Xt = tr[cols].to_numpy(float), te[cols].to_numpy(float)
    keep = np.mean(np.isfinite(X), axis=0) >= 0.2
    X, Xt = X[:, keep], Xt[:, keep]
    miss, miss_t = ~np.isfinite(X), ~np.isfinite(Xt)
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    z = lambda a: np.clip(np.nan_to_num((a - mu) / sd), -CLIP, CLIP)       # noqa: E731
    flag = miss.mean(0) >= 0.02                                            # a missing-value flag where it matters
    Z, Zt = np.c_[z(X), miss[:, flag]], np.c_[z(Xt), miss_t[:, flag]]
    beta = np.linalg.solve(Z.T @ Z + RIDGE_ALPHA * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return y.mean() + Zt @ beta


def gbm_fit(tr: pd.DataFrame, cols: list[str], y: np.ndarray):
    import lightgbm as lgb
    g = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0, importance_type="gain")
    g.fit(tr[cols], y)
    return g


def walk_forward(d: pd.DataFrame, cols: list[str], min_train_months: int = 3, with_gain: bool = False):
    """Out-of-sample forecasts of y_logit for every row from the fourth month on (blank before); one model a
    role, each month fitted on kick-offs at least a day before the month began. Trained on reliable rows only."""
    pred = np.full(len(d), np.nan)
    months = sorted(d["month"].unique())
    gain: dict[str, float] = {}
    for m in months[min_train_months:]:
        start = pd.Timestamp(m + "-01", tz="UTC")
        for role in ROLES:
            r = d["role"] == role
            tr = d[r & d["usable"] & (d["kickoff"] < start - pd.Timedelta(days=1))]
            te_idx = np.flatnonzero((r & (d["month"] == m)).to_numpy())
            if len(tr) < 2000 or not len(te_idx):
                continue
            te = d.iloc[te_idx]
            y = tr["y_logit"].to_numpy(float)
            g = gbm_fit(tr, cols, y)
            pred[te_idx] = 0.5 * (ridge_fit_predict(tr, te, cols, y) + g.predict(te[cols]))
            if with_gain and m == months[-1]:
                for c, v in zip(cols, g.feature_importances_):
                    gain[c] = gain.get(c, 0.0) + float(v)
    return pred, gain


# ---------------------------------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------------------------------

def skill_ci(day: np.ndarray, y: np.ndarray, p: np.ndarray, base: np.ndarray | None = None, n_boot: int = 1000,
             seed: int = 0) -> dict:
    """Skill against no move, 1 - SSE(p) / SSE(0), with a 95% interval from resampled days; with ``base`` (another
    forecast) also the paired difference in skill."""
    f = pd.DataFrame({"day": day, "e": (y - p) ** 2, "z": y ** 2,
                      "b": (y - base) ** 2 if base is not None else 0.0}).groupby("day").sum()
    e, z, b = f["e"].to_numpy(), f["z"].to_numpy(), f["b"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(f), size=(n_boot, len(f)))
    boot = 1 - e[idx].sum(1) / z[idx].sum(1)
    out = {"skill": float(1 - e.sum() / z.sum()), "lo95": float(np.quantile(boot, 0.025)),
           "hi95": float(np.quantile(boot, 0.975)), "days": int(len(f)), "rows": int(len(y))}
    if base is not None:
        diff = (b[idx].sum(1) - e[idx].sum(1)) / z[idx].sum(1)
        out.update({"vs_base": float((b.sum() - e.sum()) / z.sum()), "vs_base_lo95": float(np.quantile(diff, 0.025)),
                    "vs_base_hi95": float(np.quantile(diff, 0.975))})
    return out


def describe(y: np.ndarray, p: np.ndarray) -> dict:
    top = np.abs(p) >= np.quantile(np.abs(p), 0.9)
    slope = float(np.cov(p, y)[0, 1] / np.var(p, ddof=1)) if np.var(p) > 0 else None
    return {"corr": float(np.corrcoef(p, y)[0, 1]), "slope": slope,
            "top_tenth_right_way": float((np.sign(p[top]) * y[top] > 0).mean()),
            "top_tenth_mean_move": float((np.sign(p[top]) * y[top]).mean()),
            "sd_move": float(np.std(y)), "sd_forecast": float(np.std(p))}


def trades(t: pd.DataFrame, threshold: float, n_boot: int = 1000, seed: int = 0) -> dict:
    """Back at the checkpoint and lay at the close where the forecast says the chance rises (the reverse where it
    falls), in log price a unit staked, at last traded prices; net of the measured round trip and commission."""
    s = t[np.abs(t["pred"]) >= threshold]
    if len(s) < 30:
        return {"bets": int(len(s))}
    gross = np.sign(s["pred"].to_numpy()) * np.log(s["price_ratio"].to_numpy(float))
    net = gross - ROUND_TRIP
    net = np.where(net > 0, net * (1 - COMMISSION), net)
    f = pd.DataFrame({"day": s["day"].to_numpy(), "net": net, "n": 1.0}).groupby("day").sum()
    idx = np.random.default_rng(seed).integers(0, len(f), size=(n_boot, len(f)))
    boot = f["net"].to_numpy()[idx].sum(1) / f["n"].to_numpy()[idx].sum(1)
    return {"bets": int(len(s)), "days": int(len(f)), "gross": float(gross.mean()), "net": float(net.mean()),
            "net_lo90": float(np.quantile(boot, 0.05)), "net_hi90": float(np.quantile(boot, 0.95)),
            "right_way": float((gross > 0).mean())}


def run(store) -> dict:
    d, fam = load(store)
    if d.empty:
        return {"status": "no data found: gold/move is not built"}
    d = d[d["cp_fresh"]].reset_index(drop=True)
    all_cols = [c for cols in fam.values() for c in cols]
    res = {"rows": int(len(d)), "rows_usable": int(d["usable"].sum()), "months": sorted(d["month"].unique()),
           "families": {k: len(v) for k, v in fam.items()},
           "families_missing": [k for k in ("path", "implied", "rating") if k not in fam],
           "note": "development evidence: one year of data; last traded prices; nothing is traded"}
    if d["month"].nunique() < 4 or d["usable"].sum() < 10_000:
        res["status"] = "no data found: fewer than four months or 10,000 reliable rows"
        return res

    pred, gain = walk_forward(d, all_cols, with_gain=True)
    d["pred"] = pred
    level_pred, _ = walk_forward(d, fam["level"])
    scored = d[np.isfinite(d["pred"])]
    rel = scored[scored["usable"]]
    y, p = rel["y_logit"].to_numpy(float), rel["pred"].to_numpy(float)
    lp = level_pred[rel.index.to_numpy()]
    res["test_months"] = sorted(scored["month"].unique())
    res["headline"] = {**skill_ci(rel["day"].to_numpy(), y, p, base=lp), **describe(y, p)}
    res["level_only"] = skill_ci(rel["day"].to_numpy(), y, lp)
    wide = scored
    res["wider_set_stale_closes_included"] = skill_ci(wide["day"].to_numpy(), wide["y_logit"].to_numpy(float),
                                                      wide["pred"].to_numpy(float))
    res["by_horizon"] = {}
    for cp in HORIZONS:
        s = rel[rel["cp_min"] == cp]
        if len(s) >= 500:
            res["by_horizon"][str(cp)] = {**skill_ci(s["day"].to_numpy(), s["y_logit"].to_numpy(float),
                                                     s["pred"].to_numpy(float)),
                                          **describe(s["y_logit"].to_numpy(float), s["pred"].to_numpy(float))}
    res["by_role"] = {r: skill_ci(s["day"].to_numpy(), s["y_logit"].to_numpy(float), s["pred"].to_numpy(float))
                      for r, s in rel.groupby("role") if len(s) >= 500}
    # activity known at the checkpoint (never the match's final tier: that would look ahead)
    bands = pd.cut(rel["n_ticks_before"], [0, 20, 60, 150, 10**9], labels=["under 20", "20-59", "60-149", "150+"])
    res["by_activity_at_checkpoint"] = {
        str(b): skill_ci(s["day"].to_numpy(), s["y_logit"].to_numpy(float), s["pred"].to_numpy(float))
        for b, s in rel.groupby(bands, observed=True) if len(s) >= 500}
    res["by_month"] = {m: round(float(1 - ((s["y_logit"] - s["pred"]) ** 2).sum() / (s["y_logit"] ** 2).sum()), 5)
                       for m, s in rel.groupby("month")}

    # each family added to the level features, paired by day against the level arm
    res["families_added_to_level"] = {}
    for name, cols in fam.items():
        if name == "level":
            continue
        fp, _ = walk_forward(d, fam["level"] + cols)
        res["families_added_to_level"][name] = skill_ci(rel["day"].to_numpy(), y, fp[rel.index.to_numpy()], base=lp)
        log.info("level + %s: %s", name, res["families_added_to_level"][name])
    tot = sum(gain.values()) or 1.0
    by_fam = {k: round(sum(gain.get(c, 0.0) for c in cols) / tot, 4) for k, cols in fam.items()}
    res["gain_share_last_fold"] = {"by_family": by_fam,
                                   "top_features": {k: round(v / tot, 4) for k, v in
                                                    sorted(gain.items(), key=lambda kv: -kv[1])[:20]}}
    res["trades_at_last_traded_prices"] = {
        "cost": {"round_trip": ROUND_TRIP, "commission": COMMISSION},
        "reliable_rows": {str(th): trades(rel, th) for th in THRESHOLDS},
        "by_horizon_at_0.05": {str(cp): trades(rel[rel["cp_min"] == cp], 0.05) for cp in HORIZONS}}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None, help="a local warehouse folder in place of the capture bucket")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from sources.common import Store
    res = run(Store(root=a.root) if a.root else Store())
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
