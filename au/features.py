"""Betfair-only features for the Australian model: each horse's history from Betfair's own records, nothing bought.

One row a runner a market, from the BSP targets (au/data.py). The horse is its normalised name: Betfair's Australian
selection ids change with the saddlecloth number, so they do not follow a horse. Every history feature reads only the horse's EARLIER DAYS (a horse runs at most once a
day; model/lagsafe.py's rule), never today's market: today's BSP, result and money are targets or post-race, and
``FORBIDDEN`` keeps them out of the feature list. ``leak_check`` proves it by rebuilding the features with the future
cut away.

What a past run tells us here: the market's view of the horse that day (its normalised BSP chance and its rank in the
field), whether it won, and whether it won more than the market expected (A-E). No finishing positions beyond the
winner, no margins, no prize money, no ratings: those need a form source (au/README.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: today's market and result: targets or known only after the off, never features
FORBIDDEN = {"bsp", "won", "pp_vol", "ip_vol", "p_norm", "lp", "y", "mrank", "ae", "book"}
#: a selection id unseen for this long starts a new history (Betfair reuses a name's id; a horse rarely runs again
#: after four years off)
NEW_HORSE_DAYS = 1500
#: the features ranked and centred within today's race (the UK model's within-race ranks carry 41% of its gain)
RELATIVE = ["h_mean3_lp", "h_best5_lp", "h_last_lp", "h_wmean3_lp", "h_wr", "h_ae5", "h_ae_all", "h_days",
            "h_mean3_mrank", "h_n"]


def _race_values(d: pd.DataFrame) -> pd.DataFrame:
    """Each runner's own race values: known after the race, used only for the horse's LATER races."""
    d = d.copy()
    ok = d["bsp"] > 1
    inv = np.where(ok, 1.0 / d["bsp"].where(ok, np.nan), np.nan)
    d["book"] = pd.Series(inv, index=d.index).groupby(d["market_id"]).transform("sum")
    d["p_norm"] = inv / d["book"]
    d["lp"] = np.log(d["p_norm"])
    d["field"] = d.groupby("market_id")["selection_id"].transform("size")
    d["mrank"] = d.groupby("market_id")["p_norm"].rank(ascending=False, method="average") / d["field"]
    d["ae"] = d["won"] - d["p_norm"]
    return d


def _prior_window(g, x: pd.Series, k: int) -> tuple[pd.Series, pd.Series]:
    """Sum and count of x over each row's previous k rows in its group (NaN skipped), never the row itself."""
    v = x.fillna(0.0)
    c = x.notna().astype(float)
    cs, cc = v.groupby(g).cumsum(), c.groupby(g).cumsum()
    s = cs.groupby(g).shift(1).fillna(0.0) - cs.groupby(g).shift(k + 1).fillna(0.0)
    n = cc.groupby(g).shift(1).fillna(0.0) - cc.groupby(g).shift(k + 1).fillna(0.0)
    return s, n


def _prior_all(g, x: pd.Series) -> tuple[pd.Series, pd.Series]:
    v, c = x.fillna(0.0), x.notna().astype(float)
    return v.groupby(g).cumsum().groupby(g).shift(1).fillna(0.0), c.groupby(g).cumsum().groupby(g).shift(1).fillna(0.0)


def build_features(t: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The targets table (au/data.py) -> one row a runner with its features, and the feature names."""
    d = t.copy()
    d["date"] = pd.to_datetime(d["meeting_date"])
    d = _race_values(d)

    # the horse: its name. Betfair's Australian runner names carry the saddlecloth ('3. Aardvark'), so a horse's
    # selection id changes with its number (6.7 ids a name in 2020-2026); the name, normalised, is the horse. A name
    # unseen for NEW_HORSE_DAYS starts a new history (names are reused years apart).
    if "horse_norm" not in d:
        import betfair_prices as bp
        d["horse_norm"] = d["horse"].map(bp.normalise_horse)
    d = d.sort_values(["horse_norm", "date", "off_uk", "market_id"]).reset_index(drop=True)
    gap = d.groupby("horse_norm")["date"].diff().dt.days
    new = gap.isna() | (gap > NEW_HORSE_DAYS)
    d["horse_key"] = new.cumsum()
    g = d["horse_key"]

    d.attrs["same_day_name_clashes"] = int(d.duplicated(["horse_key", "date"]).sum())
    d["h_n"] = d.groupby(g).cumcount().astype(float)
    d["h_days"] = d.groupby(g)["date"].diff().dt.days
    prev = d.groupby(g).shift(1)
    d["h_last_lp"] = prev["lp"]
    d["h_last_mrank"] = prev["mrank"]
    d["h_last_won"] = prev["won"]
    d["h_last_field"] = prev["field"]
    d["h_last_dist"] = prev["distance_m"]
    d["h_last_family"] = prev["race_family"]
    d["h_prev_days"] = d.groupby(g)["h_days"].shift(1)                  # the gap before the last run
    for k in (3, 5, 10):
        s, n = _prior_window(g, d["lp"], k)
        d[f"h_mean{k}_lp"] = s / n.replace(0, np.nan)
    s, n = _prior_window(g, d["mrank"], 3)
    d["h_mean3_mrank"] = s / n.replace(0, np.nan)
    # the best market rating in the last five runs, and a recency-weighted mean of the last three (3, 2, 1)
    lags = [d.groupby(g)["lp"].shift(i) for i in range(1, 6)]
    d["h_best5_lp"] = pd.concat(lags, axis=1).max(axis=1)
    w = pd.concat([lags[0] * 3, lags[1] * 2, lags[2] * 1], axis=1)
    ww = pd.concat([lags[0].notna() * 3, lags[1].notna() * 2, lags[2].notna() * 1], axis=1)
    d["h_wmean3_lp"] = w.sum(axis=1, min_count=1) / ww.sum(axis=1).replace(0, np.nan)
    d["h_trend_lp"] = d["h_last_lp"] - d["h_mean5_lp"]
    s, n = _prior_all(g, d["won"])
    d["h_wins"], d["h_wr"] = s, (s + 0.1) / (n + 1.0)                   # shrunk towards about 10%
    s, n = _prior_window(g, d["won"], 5)
    d["h_wins5"] = s
    s, _ = _prior_all(g, d["ae"])
    d["h_ae_all"] = s
    s, _ = _prior_window(g, d["ae"], 5)
    d["h_ae5"] = s
    s, n = _prior_all(g, d["distance_m"])
    d["h_mean_dist"] = s / n.replace(0, np.nan)
    d["dist_change"] = d["distance_m"] - d["h_last_dist"]
    d["dist_vs_mean"] = d["distance_m"] - d["h_mean_dist"]
    gt = [g, d["track"]]
    d["h_track_runs"] = d.groupby(gt).cumcount().astype(float)
    d["h_track_wins"] = d.groupby(gt)["won"].cumsum() - d["won"]
    near = (d["distance_m"] // 200).astype("Int64")
    gd = [g, near]
    d["h_dist_runs"] = d.groupby(gd).cumcount().astype(float)
    d["h_dist_wins"] = d.groupby(gd)["won"].cumsum() - d["won"]
    d["family_change"] = (d["race_family"] != d["h_last_family"]).astype(float).where(d["h_last_family"].notna())

    # the race today: what is known before it
    d["first_starter"] = (d["h_n"] == 0).astype(float)
    d["month"] = d["date"].dt.month.astype(float)
    d["weekday"] = d["date"].dt.weekday.astype(float)
    d["track_code"] = d["track"].astype("category").cat.codes.astype(float)
    d["family_code"] = d["race_family"].astype("category").cat.codes.astype(float)
    d["last_family_code"] = d["h_last_family"].astype("category").cat.codes.astype(float)
    m = d.groupby("market_id")
    d["race_first_starters"] = m["first_starter"].transform("mean")
    d["race_mean3_lp"] = m["h_mean3_lp"].transform("mean")
    for f in RELATIVE:
        d[f"r_{f}"] = m[f].rank(pct=True, na_option="keep")
        d[f"c_{f}"] = d[f] - m[f].transform("mean")

    feats = ["field", "distance_m", "race_no", "month", "weekday", "track_code", "family_code", "first_starter",
             "h_n", "h_days", "h_prev_days", "h_last_lp", "h_last_mrank", "h_last_won", "h_last_field",
             "h_mean3_lp", "h_mean5_lp", "h_mean10_lp", "h_mean3_mrank", "h_best5_lp", "h_wmean3_lp", "h_trend_lp",
             "h_wins", "h_wr", "h_wins5", "h_ae_all", "h_ae5", "h_mean_dist", "dist_change", "dist_vs_mean",
             "h_track_runs", "h_track_wins", "h_dist_runs", "h_dist_wins", "family_change", "last_family_code",
             "race_first_starters", "race_mean3_lp"] + [f"r_{f}" for f in RELATIVE] + [f"c_{f}" for f in RELATIVE]
    bad = FORBIDDEN & set(feats)
    if bad:
        raise AssertionError(f"today's market or result in the features: {bad}")
    d = d.sort_values(["date", "off_uk", "market_id", "selection_id"]).reset_index(drop=True)
    return d, feats


def leak_check(t: pd.DataFrame, days: list[str], feats: list[str] | None = None) -> dict:
    """Rebuild the features with everything after each day removed: that day's features must not change. A feature
    that moves read the future. The race-level codes (track, family) are re-encoded on the cut data and are left
    out of the comparison; everything else must match to the last digit."""
    full, names = build_features(t)
    names = [f for f in (feats or names) if f not in ("track_code", "family_code", "last_family_code")]
    out = {}
    for day in days:
        cut, _ = build_features(t[t["meeting_date"] <= day])
        a = full[full["meeting_date"] == day].set_index(["market_id", "selection_id"])[names].sort_index()
        b = cut[cut["meeting_date"] == day].set_index(["market_id", "selection_id"])[names].sort_index()
        if len(a) == 0 or len(a) != len(b):
            out[day] = {"rows": int(len(a)), "rows_cut": int(len(b)), "ok": False}
            continue
        diff = {}
        for f in names:
            x, y = a[f].to_numpy(float), b[f].to_numpy(float)
            same = np.isclose(x, y, rtol=0, atol=1e-12) | (np.isnan(x) & np.isnan(y))
            if not same.all():
                diff[f] = int((~same).sum())
        out[day] = {"rows": int(len(a)), "features_that_moved": diff, "ok": not diff}
    return out
