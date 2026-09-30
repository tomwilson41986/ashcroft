"""Quant metrics: the yard's, rider's, sire's and horse's record against the Betfair SP, read the way a
fund manager reads a track record, and the horse's form read as a return series.

The owner's ask (30 Sep): measures from the financial markets, the Sharpe ratio and chi. Some are
served already and are not repeated here: model/financial_features.py reads the horse's normalised
finishing positions as a price series (RSI, MACD, z-score, Sharpe, drawdown from the peak, skew), and
the connection windows hold the yard's and rider's last 100 runners' wins less their BSP chances
(cw_tr_ae_l100, cw_jk_ae_l100).

New here (prefix qm_). Every past run with a price and a result is a GBP1 level-stake back bet at the
Betfair SP:

    r = won x (BSP - 1) x 0.95 - (1 - won)     its return, commission paid on a winner
    p = 1 / BSP, normalised within the race     the market's chance

For each entity, over its runs on days before today, a run of age a days weighted 2 ** (-a / half-life):

    qm_<e>_sh_<w>   Sharpe ratio: the weighted mean return (shrunk to MU0 by K_SH runs) over the
                    weighted standard deviation of the returns
    qm_<e>_z_<w>    signed chi: (winners - the winners the prices expected) / sqrt(sum w^2 p (1 - p)).
                    A z-score: its square is the chi-squared of the winners against the prices (the
                    'Archie' score), and |z| above 2 is seldom luck
    qm_<e>_ae_<w>   A/E: winners over the winners the prices expected, both plus K_AE
    qm_<e>_n_<w>    the weighted number of runs behind them

    e: tr the trainer, jk the jockey, tj the two together, tc the trainer at this course, sr the sire,
       hs the horse. w: 90 and 365 (the half-life in days; tr and jk both, the rest 365) and car (no
       decay; the horse).

The horse's form as a return series: pounds beaten per run (capped at 20, a non-finisher 20) and the
log Betfair SP it started at, over its last runs on earlier days (a window counts runs):

    qm_hs_lbs_sd5       volatility: the standard deviation of the last five
    qm_hs_lbs_dn5       downside deviation (Sortino's): root mean square of their excess over their mean
    qm_hs_lbs_best5     the best of the last five
    qm_hs_lbs_dd10      drawdown: the last run less the best of the last ten
    qm_hs_since_best10  runs since that best (0: the last run was it)
    qm_hs_mkt_sd5       the market's volatility on the horse: standard deviation of its last five log SPs
    qm_hs_mkt_tr5       the market's trend on it: least-squares slope of those log SPs, oldest to newest
                        (negative: shortening run by run)

Earlier days only: every sum is cumulative within its key and read at the key's last day before the
row's day, and every window steps back from the horse's first run on the row's day, so nothing from
the day being priced enters, not even as rounding.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from model.freshness_features import _col
from model.perf_figures import lbs_per_length, parse_beaten_lengths
from model.race_shape import asof_decayed_mean, day_index, race_key

COMMISSION = 0.05
MU0 = -0.05          # the level-stake return a Sharpe ratio's mean is shrunk toward (about the BSP's take)
K_SH = 30.0          # that shrinkage, in weighted runs
MIN_SH = 5.0         # weighted runs a Sharpe ratio needs
K_AE = 2.0           # expected winners added to A/E's winners and expected winners alike
LBS_CAP = 20.0
ENTITIES = {
    "tr": (("trainer",), (90.0, 365.0)),
    "jk": (("jockey_name",), (90.0, 365.0)),
    "tj": (("trainer", "jockey_name"), (365.0,)),
    "tc": (("trainer", "track"), (365.0,)),
    "sr": (("stallion",), (365.0,)),
    "hs": (("horse_name",), (np.inf,)),
}
STATS = ("sh", "z", "ae", "n")
SERIES = ["qm_hs_lbs_sd5", "qm_hs_lbs_dn5", "qm_hs_lbs_best5", "qm_hs_lbs_dd10", "qm_hs_since_best10",
          "qm_hs_mkt_sd5", "qm_hs_mkt_tr5"]


def _label(hl: float) -> str:
    return "car" if not np.isfinite(hl) else str(int(hl))


FEATURES = [f"qm_{e}_{s}_{_label(hl)}" for e, (_, hls) in ENTITIES.items() for hl in hls for s in STATS] + SERIES
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "trainer", "jockey_name", "stallion",
         "placing_numerical", "bfsp", "LB", "total_dst_bt", "dist_furlongs"]


def _codes(parts: list[pd.Series]) -> np.ndarray:
    """Order-independent codes for a name or a combination of names; -1 when any part is missing."""
    txt = [p.fillna("").astype(str).str.strip().str.lower() for p in parts]
    bad = np.zeros(len(txt[0]), dtype=bool)
    for t in txt:
        bad |= t.isin(["", "nan", "none"]).to_numpy()
    joined = txt[0] if len(txt) == 1 else txt[0].str.cat(txt[1:], sep="|")
    return np.where(bad, -1, pd.factorize(joined, sort=True)[0]).astype(np.int64)


def run_values(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each run's return, market chance, win and pounds beaten, from the run's own race. Post-race: they feed
    later days only and are never features."""
    race = pd.factorize(race_key(df), sort=True)[0]
    pos = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(dtype=float)
    fin = pos > 0
    has_result = pd.Series(fin).groupby(race).transform("any").to_numpy()
    won = np.where(fin & (pos == 1), 1.0, np.where(has_result, 0.0, np.nan))
    bsp = pd.to_numeric(_col(df, "bfsp"), errors="coerce").to_numpy(dtype=float)
    inv = pd.Series(np.where(bsp > 1, 1.0 / np.where(bsp > 1, bsp, 1.0), np.nan))
    p = (inv / inv.groupby(race).transform("sum")).to_numpy(dtype=float)
    ok = np.isfinite(won) & np.isfinite(p)
    r = np.where(ok, np.where(won == 1, (bsp - 1.0) * (1.0 - COMMISSION), -1.0), np.nan)
    if "LB" in df.columns:
        lb = pd.to_numeric(df["LB"], errors="coerce").to_numpy(dtype=float)
    else:
        lb = _col(df, "total_dst_bt").map(parse_beaten_lengths).to_numpy(dtype=float)
    dist = pd.to_numeric(_col(df, "dist_furlongs"), errors="coerce").to_numpy(dtype=float)
    lbs = np.minimum(lb * lbs_per_length(np.nan_to_num(dist, nan=8.0)), LBS_CAP)
    lbs = np.where(pos == 1, 0.0, np.where(fin, lbs, np.where(has_result, LBS_CAP, np.nan)))
    with np.errstate(invalid="ignore", divide="ignore"):
        lsp = np.where(bsp > 1, np.log(np.where(bsp > 1, bsp, 1.0)), np.nan)
    return {"r": r, "won": np.where(ok, won, np.nan), "p": np.where(ok, p, np.nan), "lbs": lbs, "lsp": lsp}


def entity_metrics(key: np.ndarray, day: np.ndarray, v: dict[str, np.ndarray], hl: float) -> dict[str, np.ndarray]:
    """Sharpe ratio, signed chi, A/E and weighted runs of each row's entity over its earlier days."""
    def asof(x, h):
        return asof_decayed_mean(key, day, x, key, day, h)

    m_r, n = asof(v["r"], hl)
    m_r2, _ = asof(v["r"] ** 2, hl)
    m_w, _ = asof(v["won"], hl)
    m_p, _ = asof(v["p"], hl)
    m_v2, n2 = asof(v["p"] * (1.0 - v["p"]), hl / 2.0)          # weights squared: half the half-life
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.sqrt(np.maximum(m_r2 - m_r ** 2, 0.0))
        mean = (m_r * n + MU0 * K_SH) / (n + K_SH)
        sh = np.where((n >= MIN_SH) & (sd > 0), mean / sd, np.nan)
        var = m_v2 * n2
        z = np.where(var > 0, (m_w - m_p) * n / np.sqrt(var), np.nan)
        ae = np.where(n > 0, (m_w * n + K_AE) / (m_p * n + K_AE), np.nan)
    return {"sh": sh, "z": z, "ae": ae, "n": np.where(key >= 0, n, np.nan)}


def series_metrics(horse: np.ndarray, day: np.ndarray, tie: np.ndarray, lbs: np.ndarray,
                   lsp: np.ndarray) -> dict[str, np.ndarray]:
    """The horse's last runs on earlier days, read as a return series."""
    n = len(horse)
    order = np.lexsort((tie, day, horse))
    h_s, d_s = horse[order], day[order]
    idx = np.arange(n)
    new_h = np.r_[True, h_s[1:] != h_s[:-1]]
    new_d = new_h | np.r_[True, d_s[1:] != d_s[:-1]]
    h_start = np.maximum.accumulate(np.where(new_h, idx, 0))
    d_start = np.maximum.accumulate(np.where(new_d, idx, 0))

    def back(v, k):
        """(n, k): column j-1 is the value j runs back (NaN past the horse's first run)."""
        v_s = v[order]
        out = np.full((n, k), np.nan)
        for j in range(1, k + 1):
            i = d_start - j
            ok = (i >= h_start) & (h_s >= 0)
            out[ok, j - 1] = v_s[i[ok]]
        return out

    L, M = back(lbs, 10), back(lsp, 5)
    L5 = L[:, :5]
    cnt5 = np.isfinite(L5).sum(axis=1)
    cnt10 = np.isfinite(L).sum(axis=1)
    cm = np.isfinite(M).sum(axis=1)
    res = {}
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN windows give NaN, as meant
        mu5 = np.nanmean(L5, axis=1)
        res["qm_hs_lbs_sd5"] = np.where(cnt5 >= 3, np.nanstd(L5, axis=1), np.nan)
        up = np.where(np.isfinite(L5), np.maximum(L5 - mu5[:, None], 0.0), np.nan)
        res["qm_hs_lbs_dn5"] = np.where(cnt5 >= 3, np.sqrt(np.nanmean(up ** 2, axis=1)), np.nan)
        res["qm_hs_lbs_best5"] = np.where(cnt5 >= 1, np.nanmin(L5, axis=1), np.nan)
        best10 = np.nanmin(L, axis=1)
        res["qm_hs_lbs_dd10"] = np.where((cnt10 >= 2) & np.isfinite(L[:, 0]), L[:, 0] - best10, np.nan)
        since = np.argmin(np.where(np.isfinite(L), L, np.inf), axis=1).astype(float)
        res["qm_hs_since_best10"] = np.where(cnt10 >= 2, since, np.nan)
        res["qm_hs_mkt_sd5"] = np.where(cm >= 3, np.nanstd(M, axis=1), np.nan)
        t = -np.arange(1, 6, dtype=float)[None, :] * np.ones((n, 1))      # older runs further back
        f = np.isfinite(M)
        tm = np.where(f, t, np.nan)
        dt = tm - np.nanmean(tm, axis=1)[:, None]
        dy = M - np.nanmean(M, axis=1)[:, None]
        num = np.nansum(np.where(f, dt * dy, 0.0), axis=1)
        den = np.nansum(np.where(f, dt ** 2, 0.0), axis=1)
        res["qm_hs_mkt_tr5"] = np.where((cm >= 3) & (den > 0), num / den, np.nan)
    out = {}
    for c, x in res.items():
        o = np.empty(n)
        o[order] = x
        out[c] = np.where((horse >= 0) & (day >= 0), o, np.nan)
    return out


def build(df: pd.DataFrame) -> pd.DataFrame:
    n_rows = len(df)
    day = day_index(df)
    v = run_values(df)
    cols = {}
    for e, (columns, hls) in ENTITIES.items():
        key = _codes([_col(df, c) for c in columns])
        for hl in hls:
            m = entity_metrics(key, day, v, hl)
            for s in STATS:
                cols[f"qm_{e}_{s}_{_label(hl)}"] = m[s]
    horse = _codes([_col(df, "horse_name")])
    race = pd.factorize(race_key(df), sort=True)[0].astype(np.int64)
    tie = race * (int(horse.max()) + 2) + (horse + 1)
    cols.update(series_metrics(horse, day, tie, v["lbs"], v["lsp"]))
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    assert len(new) == n_rows
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
