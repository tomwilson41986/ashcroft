"""Seasonality: how the horse and its yard have gone at this time of year in earlier years.

Some horses come to hand in the spring and go over the top by the autumn; some yards peak in
particular months. The model reads recent form and a race's date (the context block's season),
not a horse's or a yard's own record by time of year. For today's month and the month either side,
in EARLIER YEARS only (runs at least 300 days before today, so this year's form is not counted
twice):

    sn_nfp          the horse's normalised finishing position (1 = won, 0 = last) in those months
    sn_n            how many such runs it has
    sn_nfp_vs_all   sn_nfp less its position over all its earlier runs (+ = better at this time of year)
    sn_tr_ae        the yard's winners less the Betfair SP's chances (A-E per runner) in those months,
                    shrunk over 50 runners to 0
    sn_tr_ae_vs_all sn_tr_ae less the yard's A-E in all months of earlier years, shrunk the same way

A research query (market_season_residuals, 28 Sep) could not tell whether these add to the model:
its residual screen reads features the model already holds as strongly as new ones. Only a fit can.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.race_shape import asof_decayed_mean, day_index

FEATURES = ["sn_nfp", "sn_n", "sn_nfp_vs_all", "sn_tr_ae", "sn_tr_ae_vs_all"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "trainer", "number_of_runners",
         "placing_numerical", "bfsp"]
LAG_DAYS = 300           # a run counts from this many days on: earlier years, not this season
K_TRAINER = 50.0         # shrinkage of a yard's A-E to 0, in runners
#: tests/test_feature_blocks.py's history spans 60 days: its generic tests read the same code at this reach
TEST_REACH = {"LAG_DAYS": 3}


def _key(s: pd.Series) -> np.ndarray:
    t = s.fillna("").astype(str).str.strip().str.lower()
    bad = t.isin(["", "nan", "none"]).to_numpy()
    return np.where(bad, -1, pd.factorize(t, sort=True)[0]).astype(np.int64)


def build(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    day = day_index(df)
    lag_day = np.where(day >= 0, day + LAG_DAYS, -1)
    race = pd.factorize(df["raceid"].astype(str), sort=True)[0]
    h, tr = _key(df["horse_name"]), _key(df["trainer"])
    month = pd.to_datetime(df["race_date"], errors="coerce").dt.month.fillna(0).to_numpy().astype(np.int64)
    pos = pd.to_numeric(df["placing_numerical"], errors="coerce").to_numpy(dtype=float)
    bsp = pd.to_numeric(df["bfsp"], errors="coerce").to_numpy(dtype=float)
    nr = pd.to_numeric(df["number_of_runners"], errors="coerce").to_numpy(dtype=float)
    rows = pd.Series(np.ones(n)).groupby(race).transform("sum").to_numpy(dtype=float)
    nr = np.where(np.isfinite(nr) & (nr > 0), nr, rows)
    with np.errstate(invalid="ignore", divide="ignore"):
        nfp = np.where((pos > 0) & (nr > 1), (nr - pos) / (nr - 1), np.nan)
        won = np.where(pos > 0, (pos == 1).astype(float), np.nan)
        ae = np.where(np.isfinite(won) & (bsp > 1.0), won - 1.0 / bsp, np.nan)

    def season(key, v):
        """Mean of v over the key's runs in today's month and the month either side, LAG_DAYS or more
        before today, and the (undecayed) number of them."""
        tot, cnt = np.zeros(n), np.zeros(n)
        ok = (key >= 0) & (month > 0)
        hist = np.where(ok, key * 13 + month, -1)
        for off in (-1, 0, 1):
            qm = (month - 1 + off) % 12 + 1
            mu, nn = asof_decayed_mean(hist, lag_day, v, np.where(ok, key * 13 + qm, -1), day)
            tot += np.where(nn > 0, mu * nn, 0.0)
            cnt += nn
        return np.where(cnt > 0, tot / np.maximum(cnt, 1e-12), np.nan), cnt

    sea_nfp, sea_n = season(h, nfp)
    all_nfp, _ = asof_decayed_mean(h, day, nfp, h, day)
    t_ae, t_n = season(tr, ae)
    t_all, t_all_n = asof_decayed_mean(tr, lag_day, ae, tr, day)
    t_sea = np.where(t_n > 0, t_ae * t_n / (t_n + K_TRAINER), np.nan)
    t_base = np.where(t_all_n > 0, t_all * t_all_n / (t_all_n + K_TRAINER), np.nan)
    cols = {"sn_nfp": sea_nfp, "sn_n": np.where(h >= 0, sea_n, np.nan), "sn_nfp_vs_all": sea_nfp - all_nfp,
            "sn_tr_ae": t_sea, "sn_tr_ae_vs_all": t_sea - t_base}
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
