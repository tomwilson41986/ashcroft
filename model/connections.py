"""
Lag-safe, credibility-shrunk connection features (racing² framework Part 5 /
Part 8.4) — market-free, so legal in Stage F.

For each entity column (trainer, jockey, sire, ...) and each run, using only
PRIOR runs of that entity:

    {e}_runs                 number of prior runs
    {e}_sr_shrunk            strike rate shrunk toward the population rate:
                             (wins + k·prior) / (runs + k)
    {e}_nmfp_shrunk          mean NMFP shrunk toward 0 (its population mean)
    {e}_form_{w}d            recency-weighted mean NMFP over the last w days
                             (stable form / momentum), shrunk by runner count
    {e}_form_vs_career       form_{w}d − career shrunk mean

Trainer × jockey pairing strike rate shrunk toward the trainer's rate, and
``jockey_booking_upgrade`` = today's jockey shrunk SR − the horse's usual
jockey SR (an intent signal).

k (the shrinkage constant) is fitted per family by out-of-sample
likelihood in the framework; here it is a parameter with sensible defaults
(k = 30 for strike rates, 15 for NMFP), and ``runs`` is always carried so a
model can discount thin evidence further.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _lag_cum(df: pd.DataFrame, key: str, col: str) -> pd.Series:
    g = df.groupby(key, group_keys=False)[col]
    return g.cumsum() - df[col]


def _time_window_prior_mean(df: pd.DataFrame, key: str, val_col: str, date_col: str, window_days: int) -> tuple[pd.Series, pd.Series]:
    """Mean of val over prior rows of the same entity within window_days (exclusive of today's rows)."""
    out_mean = np.full(len(df), np.nan); out_n = np.zeros(len(df))
    d = df[[key, date_col, val_col]].reset_index(drop=True)
    d["_pos"] = np.arange(len(d))
    d = d.sort_values([key, date_col, "_pos"])
    keys = d[key].values; dates = d[date_col].values.astype("datetime64[D]").astype(np.int64); vals = d[val_col].values.astype(float); pos = d["_pos"].values
    start = 0
    for i in range(len(d)):
        if i == 0 or keys[i] != keys[i - 1]:
            start = i
        lo = start
        while lo < i and dates[lo] < dates[i] - window_days:
            lo += 1
        # exclude rows on the same day (not known before the off)
        j = i
        while j > lo and dates[j - 1] == dates[i]:
            j -= 1
        w = vals[lo:j]; w = w[~np.isnan(w)]
        out_n[pos[i]] = len(w)
        if len(w):
            out_mean[pos[i]] = w.mean()
    return pd.Series(out_mean, index=df.index), pd.Series(out_n, index=df.index)


def add_connection_features(df: pd.DataFrame, entities=("trainer", "jockey_name"), nmfp_col: str = "nmfp", won_col: str = "won",
                            date_col: str = "race_date", time_col: str = "race_time", k_sr: float = 30.0, k_nmfp: float = 15.0,
                            form_windows=(14, 30)) -> tuple[pd.DataFrame, list[str]]:
    d = df.sort_values([date_col, time_col]).copy()
    d["_won"] = pd.to_numeric(d[won_col], errors="coerce").fillna(0.0); d["_nmfp"] = pd.to_numeric(d[nmfp_col], errors="coerce")
    d["_one"] = 1.0; d["_nmfp_ok"] = d["_nmfp"].notna().astype(float); d["_nmfp0"] = d["_nmfp"].fillna(0.0)
    prior_sr = float(d["_won"].mean())
    feats = []
    for e in entities:
        if e not in d.columns:
            continue
        d[e] = d[e].fillna("__missing__").astype(str)
        d = d.sort_values([e, date_col, time_col])
        runs = _lag_cum(d, e, "_one"); wins = _lag_cum(d, e, "_won"); nm = _lag_cum(d, e, "_nmfp0"); nmn = _lag_cum(d, e, "_nmfp_ok")
        p = e.split("_")[0]
        d[f"{p}_runs"] = runs
        d[f"{p}_sr_shrunk"] = (wins + k_sr * prior_sr) / (runs + k_sr)
        d[f"{p}_nmfp_shrunk"] = nm / (nmn + k_nmfp)
        feats += [f"{p}_runs", f"{p}_sr_shrunk", f"{p}_nmfp_shrunk"]
        for w in form_windows:
            m, n = _time_window_prior_mean(d, e, "_nmfp", date_col, w)
            d[f"{p}_form_{w}d"] = (m.fillna(0.0) * n) / (n + 6.0)   # shrink toward 0 by runner count
            feats.append(f"{p}_form_{w}d")
        d[f"{p}_form_vs_career"] = d[f"{p}_form_{form_windows[-1]}d"] - d[f"{p}_nmfp_shrunk"]
        feats.append(f"{p}_form_vs_career")
    if "trainer" in d.columns and "jockey_name" in d.columns:
        d["_tj"] = d["trainer"].astype(str) + "|" + d["jockey_name"].astype(str)
        d = d.sort_values(["_tj", date_col, time_col])
        runs = _lag_cum(d, "_tj", "_one"); wins = _lag_cum(d, "_tj", "_won")
        d["tj_runs"] = runs
        d["tj_sr_shrunk"] = (wins + 20.0 * d["trainer_sr_shrunk"]) / (runs + 20.0)  # toward the trainer's own rate
        feats += ["tj_runs", "tj_sr_shrunk"]
        # booking upgrade: today's jockey vs the horse's usual jockey quality
        d = d.sort_values(["horse_name", date_col, time_col])
        usual = d.groupby("horse_name")["jockey_sr_shrunk"].transform(lambda s: s.shift(1).expanding().mean())
        d["jockey_booking_upgrade"] = d["jockey_sr_shrunk"] - usual
        feats.append("jockey_booking_upgrade")
        d = d.drop(columns=["_tj"])
    d = d.drop(columns=["_won", "_nmfp", "_one", "_nmfp_ok", "_nmfp0"])
    return d.sort_index(), feats
