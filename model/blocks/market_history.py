"""Market history: how the bookmakers and the place market priced the horse, its yard and its rider before.

Every price the model reads is the Betfair win SP. Two other prices of each past run are in the
table and read nowhere: the industry starting price (`odds`, net odds: 4/1 is 4.0), and the Betfair
place SP (`bfsp_place`, `bf_plcs_paid` places). They say how two other markets saw a runner:

    spread   log net Betfair SP less log net industry SP: + = the exchange longer than the
             bookmakers (a runner the bookmakers' customers backed and the exchange did not)
    ratio    log net win SP less log net place SP: + = the place price short against the win
             price (a horse the market expects to be placed more than to win)

Each is read against what is usual for such a run (its price band, field size and code for the
spread; its price band, field size and places paid for the ratio), the cell's mean over EARLIER
days, so a reading means the same on the day it is made as later. From earlier days only:

    mh_spb_h        the horse's spread, decayed (half-life 365 days)
    mh_spb_h_n      the effective number of its runs behind that
    mh_spb_tr       its yard's runners' spread (half-life 180 days), shrunk over 20 runners to 0
    mh_spb_tr30     the same over the last weeks (half-life 30 days)
    mh_spb_jk       its rider's rides' spread (half-life 180 days), shrunk over 20
    mh_wp_h         the horse's win/place ratio, decayed (half-life 365 days)
    mh_wp_h_n       the effective number of its runs behind that
    mh_wp_tr        its yard's runners' ratio (half-life 180 days), shrunk over 20

A research query (market_season_residuals, 28 Sep) could not tell whether these add to the model:
its residual screen reads features the model already holds as strongly as new ones. Only a fit can.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.race_shape import asof_decayed_mean, day_index

FEATURES = ["mh_spb_h", "mh_spb_h_n", "mh_spb_tr", "mh_spb_tr30", "mh_spb_jk", "mh_wp_h", "mh_wp_h_n", "mh_wp_tr"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "trainer", "jockey_name", "odds", "bfsp",
         "bfsp_place", "bf_plcs_paid", "plcs_paid", "number_of_runners", "race_code"]
HALFLIFE_HORSE = 365.0
HALFLIFE_CONN = 180.0
HALFLIFE_RECENT = 30.0
K = 20.0                 # shrinkage of a yard's or rider's mean to 0 (the cell's usual), in runners
MIN_CELL = 30.0          # runs a cell needs on earlier days before its mean is the baseline
CLIP = 3.0               # a raw reading is clipped here, a reading against its cell at 2


def _key(s: pd.Series) -> np.ndarray:
    t = s.fillna("").astype(str).str.strip().str.lower()
    bad = t.isin(["", "nan", "none"]).to_numpy()
    return np.where(bad, -1, pd.factorize(t, sort=True)[0]).astype(np.int64)


def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _against_cell(v, day, fine, coarse):
    """v less its cell's mean over earlier days (the fine cell once it has MIN_CELL runs, else the coarse
    one), clipped at 2; NaN where v is or neither cell has enough history."""
    out = np.full(len(v), np.nan)
    for cell in (coarse, fine):                       # the fine cell overwrites where it has the runs
        mu, n = asof_decayed_mean(cell, day, v, cell, day)
        ok = np.isfinite(mu) & (n >= MIN_CELL) & np.isfinite(v)
        out[ok] = v[ok] - mu[ok]
    return np.clip(out, -2.0, 2.0)


def build(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    day = day_index(df)
    race = pd.factorize(df["raceid"].astype(str), sort=True)[0]
    h, tr, jk = _key(df["horse_name"]), _key(df["trainer"]), _key(df["jockey_name"])
    bsp, sp, bpl = _num(df, "bfsp"), _num(df, "odds"), _num(df, "bfsp_place")
    nr = _num(df, "number_of_runners")
    rows = pd.Series(np.ones(n)).groupby(race).transform("sum").to_numpy(dtype=float)
    nr = np.where(np.isfinite(nr) & (nr > 0), nr, rows)
    places = _num(df, "bf_plcs_paid")
    places = np.where(np.isfinite(places), places, _num(df, "plcs_paid"))
    code = _key(df["race_code"]) if "race_code" in df.columns else np.zeros(n, dtype=np.int64)
    code = np.where(code >= 0, np.minimum(code, 126), 127)
    with np.errstate(invalid="ignore", divide="ignore"):
        lb = np.log(bsp)
        pbin = np.where(np.isfinite(lb), np.clip(np.floor(lb / 0.2), 0, 40), -1).astype(np.int64)
        fs = np.digitize(nr, [6, 8, 12, 16]).astype(np.int64)
        ok_s = (bsp > 1.01) & (sp > 0.01)
        s = np.where(ok_s, np.clip(np.log(np.where(ok_s, bsp - 1.0, 1.0)) - np.log(np.where(ok_s, sp, 1.0)),
                                   -CLIP, CLIP), np.nan)
        ok_w = (bsp > 1.01) & (bpl > 1.01)
        w = np.where(ok_w, np.clip(np.log(np.where(ok_w, bsp - 1.0, 1.0)) - np.log(np.where(ok_w, bpl - 1.0, 1.0)),
                                   -CLIP, CLIP), np.nan)
    has_p = pbin >= 0
    s_cell = np.where(has_p, (pbin * 10 + fs) * 128 + code, -1)
    s_coarse = np.where(has_p, pbin, -1)
    pl = np.clip(np.nan_to_num(places, nan=0.0), 0, 9).astype(np.int64)
    w_cell = np.where(has_p, (pbin * 32 + np.clip(nr, 0, 30).astype(np.int64)) * 10 + pl, -1)
    w_coarse = np.where(has_p, pbin * 10 + pl, -1)
    s_res = _against_cell(s, day, s_cell, s_coarse)
    w_res = _against_cell(w, day, w_cell, w_coarse)

    def decayed(key, v, hl, k=0.0):
        mu, nn = asof_decayed_mean(key, day, v, key, day, hl)
        if k > 0:
            mu = np.where(nn > 0, mu * nn / (nn + k), np.nan)
        return mu, nn

    cols = {}
    cols["mh_spb_h"], cols["mh_spb_h_n"] = decayed(h, s_res, HALFLIFE_HORSE)
    cols["mh_spb_tr"], _ = decayed(tr, s_res, HALFLIFE_CONN, K)
    cols["mh_spb_tr30"], _ = decayed(tr, s_res, HALFLIFE_RECENT, K)
    cols["mh_spb_jk"], _ = decayed(jk, s_res, HALFLIFE_CONN, K)
    cols["mh_wp_h"], cols["mh_wp_h_n"] = decayed(h, w_res, HALFLIFE_HORSE)
    cols["mh_wp_tr"], _ = decayed(tr, w_res, HALFLIFE_CONN, K)
    cols["mh_spb_h_n"] = np.where(h >= 0, cols["mh_spb_h_n"], np.nan)
    cols["mh_wp_h_n"] = np.where(h >= 0, cols["mh_wp_h_n"], np.nan)
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
