"""Jumps form: what a horse's earlier runs over obstacles say that its Flat-shaped features do not.

The served features were written for every code at once: a horse's form is its finishing
positions and margins, a non-finisher is unknown, and a chase is one more race. Over jumps
three things matter that those features cannot see: whether the horse gets round (a faller
is not "unknown", it is a risk the market prices), whether its form is over today's kind of
obstacle (a hurdler's first chase), and whether it stays three miles. One model with these
may do what a separate jumps model would, without halving the rows either fits on (the
owner's question of 11 Nov 2026: a separate model for National Hunt?).

All from earlier days. A jumps run is a hurdle or a chase (model.race_shape.race_code); a
non-finish is read from the run's comment (fell, unseated, brought down; pulled up, refused,
ran out). Rates are shrunk toward the code's rate with K_SHRINK pseudo-runs. On a row that is
not a hurdle or a chase today every feature is NaN.

    jf_hurdle_runs        earlier hurdle runs
    jf_chase_runs         earlier chase runs
    jf_first_today        1 when the horse has run over jumps but never over today's obstacle
    jf_fall_rate          share of earlier jumps runs ending fell / unseated / brought down (shrunk)
    jf_fall_rate_code     the same over today's obstacle only
    jf_fall_recent        the same, decayed (half-life 180 days)
    jf_pu_rate            share of earlier jumps runs pulled up / refused / ran out (shrunk)
    jf_mkt_code           the market's view at earlier runs over today's obstacle: the mean of
                          log(runners x share of the BSP book)
    jf_mkt_jumps          the same over every earlier jumps run
    jf_mkt_code_gap       jf_mkt_code less jf_mkt_jumps: better or worse fancied over today's obstacle
    jf_stay_gap           the market's view weighted by each run's trip, less the plain mean: positive
                          when the horse was better fancied over its longer trips
    jf_runs_3m            earlier jumps runs at 24 furlongs or more
    jf_complete_3m        share of them completed (shrunk toward the horse's overall completion)
    jf_trip_vs_usual      today's trip less the mean trip of its earlier jumps runs (furlongs)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.race_shape import asof_decayed_mean, day_index, race_code, race_key

FEATURES = ["jf_hurdle_runs", "jf_chase_runs", "jf_first_today", "jf_fall_rate", "jf_fall_rate_code",
            "jf_fall_recent", "jf_pu_rate", "jf_mkt_code", "jf_mkt_jumps", "jf_mkt_code_gap", "jf_stay_gap",
            "jf_runs_3m", "jf_complete_3m", "jf_trip_vs_usual"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "race_type", "surface_type", "dist_furlongs",
         "comment", "bfsp"]

K_SHRINK = 5.0
P_FALL, P_PU = 0.06, 0.10          # priors: about the share of jumps runs ending each way
RECENT_HALFLIFE = 180.0
STAYING_FURLONGS = 24.0
FALL_RX = r"\bfell\b|\bunseated\b|\bbrought down\b"
PU_RX = r"\bpulled up\b|\brefused\b|\bran out\b|\bcarried out\b"


def _horse_codes(s: pd.Series) -> np.ndarray:
    t = s.fillna("").astype(str).str.strip().str.lower()
    bad = t.isin(["", "nan", "none"]).to_numpy()
    return np.where(bad, -1, pd.factorize(t, sort=True)[0]).astype(np.int64)


def _market_view(df: pd.DataFrame, race: np.ndarray) -> np.ndarray:
    """log(priced runners x the runner's share of the BSP book); NaN without a usable price."""
    bsp = pd.to_numeric(df["bfsp"], errors="coerce").to_numpy(float) if "bfsp" in df.columns \
        else np.full(len(df), np.nan)
    ok = np.isfinite(bsp) & (bsp > 1.0)
    inv = np.where(ok, 1.0 / np.where(ok, bsp, 1.0), 0.0)
    nr = int(race.max()) + 1 if len(race) else 0
    tot = np.bincount(race, weights=inv, minlength=nr)
    n = np.bincount(race, weights=ok.astype(float), minlength=nr)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ok & (n[race] >= 2), np.log(n[race] * inv / np.where(tot[race] > 0, tot[race], 1.0)), np.nan)


def _shrunk(mean, n, prior):
    m = np.where(n > 0, mean, 0.0)
    return (m * n + K_SHRINK * prior) / (n + K_SHRINK)


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    h = _horse_codes(df["horse_name"])
    code = race_code(df).to_numpy()
    hurdle, chase = code == "hurdle", code == "chase"
    jumps = hurdle | chase
    race = pd.factorize(race_key(df), sort=True)[0].astype(np.int64)
    comment = df["comment"].fillna("").astype(str).str.lower() if "comment" in df.columns \
        else pd.Series("", index=df.index)
    fell = comment.str.contains(FALL_RX, regex=True).to_numpy()
    pu = comment.str.contains(PU_RX, regex=True).to_numpy()
    trip = pd.to_numeric(df.get("dist_furlongs"), errors="coerce").to_numpy(float) if "dist_furlongs" in df.columns \
        else np.full(len(df), np.nan)
    mkt = _market_view(df, race)

    def hist(value, rows, halflife=np.inf):
        """Mean and count of `value` over the horse's earlier-day rows in `rows`."""
        v = np.where(rows, value, np.nan).astype(float)
        return asof_decayed_mean(h, day, v, h, day, halflife)

    one = np.ones(len(df))
    _, n_h = hist(one, hurdle)
    _, n_c = hist(one, chase)
    fall_m, n_j = hist(fell.astype(float), jumps)
    fall_h, _ = hist(fell.astype(float), hurdle)
    fall_c, _ = hist(fell.astype(float), chase)
    fall_r, n_r = hist(fell.astype(float), jumps, RECENT_HALFLIFE)
    pu_m, _ = hist(pu.astype(float), jumps)
    mkt_h, nm_h = hist(mkt, hurdle)
    mkt_c, nm_c = hist(mkt, chase)
    stay = jumps & (trip >= STAYING_FURLONGS)
    _, n_3 = hist(one, stay)
    mkt_j, nm_j = hist(mkt, jumps)
    both = np.isfinite(mkt) & np.isfinite(trip)
    wm, n_w = hist(np.where(both, mkt * trip, np.nan), jumps)
    wt, _ = hist(np.where(both, trip, np.nan), jumps)
    mj_w, _ = hist(np.where(both, mkt, np.nan), jumps)
    done = (~fell & ~pu).astype(float)
    done_3, _ = hist(done, stay)
    done_all, _ = hist(done, jumps)
    trip_m, n_t = hist(trip, jumps)

    n_code = np.where(chase, n_c, n_h)
    fall_code = np.where(chase, fall_c, fall_h)
    mkt_code = np.where(chase, np.where(nm_c > 0, mkt_c, np.nan), np.where(nm_h > 0, mkt_h, np.nan))
    mkt_jumps = np.where(nm_j > 0, mkt_j, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        stay_gap = np.where((n_w > 0) & (wt > 0), wm / wt - mj_w, np.nan)
    prior_done = np.where(n_j > 0, done_all, 1.0 - P_FALL - P_PU)
    out = {
        "jf_hurdle_runs": n_h,
        "jf_chase_runs": n_c,
        "jf_first_today": ((n_j > 0) & (n_code == 0)).astype(float),
        "jf_fall_rate": _shrunk(fall_m, n_j, P_FALL),
        "jf_fall_rate_code": _shrunk(fall_code, n_code, P_FALL),
        "jf_fall_recent": _shrunk(fall_r, n_r, P_FALL),
        "jf_pu_rate": _shrunk(pu_m, n_j, P_PU),
        "jf_mkt_code": mkt_code,
        "jf_mkt_jumps": mkt_jumps,
        "jf_mkt_code_gap": mkt_code - mkt_jumps,
        "jf_stay_gap": stay_gap,
        "jf_runs_3m": n_3,
        "jf_complete_3m": (np.where(n_3 > 0, done_3, 0.0) * n_3 + K_SHRINK * prior_done) / (n_3 + K_SHRINK),
        "jf_trip_vs_usual": np.where(n_t > 0, trip - trip_m, np.nan),
    }
    new = pd.DataFrame(out, index=df.index)[FEATURES].astype(float)
    new.loc[~jumps, :] = np.nan
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
