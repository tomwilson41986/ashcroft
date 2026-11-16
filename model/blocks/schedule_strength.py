"""Strength of schedule, capped margins, formful races and negative evidence: five readings from one way of working.

From the owner's reading of 12 Nov 2026 (a ratings compiler's notes): measure a run against the strength of what it
met, cap what carries no signal, weight a race by how truly it was run, and use the evidence of when conditions were
wrong as well as when they were right. None of these is in the blocks so far.

1. Capped margins. Lengths beaten are squashed by LIMIT x tanh(lengths / LIMIT): a length is a length close to the
   winner and nothing counts beyond about LIMIT lengths (a horse beaten thirty has stopped trying; the margin is
   noise). The run's capped figure is c = 1 - capped lengths / LIMIT (1 the winner, 0 at the limit; a non-finisher 0).
       ss_cap_l1  ss_cap_m3  ss_cap_dec     the last run, the last three, decayed over a year

2. Strength of schedule. Each run's c is multiplied by the strength of the field it met: the field's mean market
   rating before the race, as a share of all fields' (model/blocks/market_rating.py mr_field, ranked over earlier
   days). q = sqrt(c x field strength).
       ss_sos_dec     the decayed mean of q: the record, credited for what it was run against
       ss_sos_gap     that less the best rival's; ss_sos_z it standardised within today's race
       ss_field_l1    the strength of the last race's field less today's (above zero: dropping in class, on the
                      market's own measure)
       ss_met_unexp   the mean share of unexposed runners (at most EXPOSED runs) in the fields of its last five
                      races: an unexposed horse that has been meeting unexposed horses

3. Formful races. How closely a race finished in its market order: the rank correlation of BSP and finishing position.
       ss_formful_l1  ss_formful_m3     of the last race and the last three
       ss_form_w      the horse's price-neutral record (model/blocks/_resid.py) weighted by how formful each race was

4. Negative evidence (the Pythagorean form). p = the decayed mean normalised finishing position inside today's trip
   band (going group), o = outside it; p^2 / (p^2 + o^2): above a half, better when conditions are as today's.
       ss_pyth_trip  ss_pyth_going     each side shrunk to a half by two runs, so a horse that has only run one
                                       way reads near a half
       ss_pyth_rating                  today's market rating share squared over that plus the rivals' mean squared

5. Rank as a chance. A rank r in a field of n is worth (1 / n) x (1/r + ... + 1/n) (the harmonic form).
       ss_hp_mr  ss_hp_sos     the market rating's and the schedule record's rank in today's race, as that chance

Earlier days only for every record; today's field, trip and going are the card's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks import _resid
from model.blocks.cond_form import GOING, classify_going
from model.blocks.form_variants import _ladder
from model.freshness_features import _Days, _col
from model.race_shape import DIST_BANDS, asof_decayed_mean, bands, day_index, race_key

FEATURES = ["ss_cap_l1", "ss_cap_m3", "ss_cap_dec", "ss_sos_dec", "ss_sos_gap", "ss_sos_z", "ss_field_l1", "ss_met_unexp",
            "ss_formful_l1", "ss_formful_m3", "ss_form_w", "ss_pyth_trip", "ss_pyth_going", "ss_pyth_rating", "ss_hp_mr",
            "ss_hp_sos"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "placing_numerical", "number_of_runners", "bfsp",
         "total_dst_bt", "dist_furlongs", "going_description", "career_runs", "mr_field", "mr_share", "mr_rating"]
AFTER = ("market_rating",)
LIMIT = 15.0
K_PYTH = 2.0
EXPOSED = 5


def _codes(s: pd.Series) -> np.ndarray:
    t = s.fillna("").astype(str).str.strip().str.lower()
    return np.where(t.isin(["", "nan", "none"]).to_numpy(), -1, pd.factorize(t, sort=True)[0]).astype(np.int64)


def _harmonic(rank: np.ndarray, n: np.ndarray) -> np.ndarray:
    """(1/n) x sum_{k=r..n} 1/k, for integer ranks (ties take the better rank)."""
    top = int(np.nanmax(n)) if np.isfinite(n).any() else 1
    h = np.r_[0.0, np.cumsum(1.0 / np.arange(1, top + 2))]
    ok = np.isfinite(rank) & np.isfinite(n) & (n >= 1) & (rank >= 1) & (rank <= n)
    r, m = np.where(ok, rank, 1).astype(int), np.where(ok, n, 1).astype(int)
    return np.where(ok, (h[m] - h[r - 1]) / m, np.nan)


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    race = pd.Series(pd.factorize(race_key(df), sort=True)[0], index=df.index)
    horse = _codes(df["horse_name"])
    H = _Days(horse, day)
    prev = np.clip(np.arange(len(H.u)) - 1, 0, None)
    has_prev = H.valid & (H.pos >= 1)
    pos = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(float)
    n = pd.to_numeric(_col(df, "number_of_runners"), errors="coerce").to_numpy(float)
    lb = pd.to_numeric(_col(df, "total_dst_bt"), errors="coerce").to_numpy(float)
    fin = np.isfinite(pos) & (pos >= 1)
    lb = np.where(pos == 1, 0.0, np.where(fin & np.isfinite(lb), np.maximum(lb, 0.0), np.nan))
    c = np.where(fin, 1.0 - np.tanh(np.where(np.isfinite(lb), lb, LIMIT * 3) / LIMIT), np.where(n > 1, 0.0, np.nan))

    cols: dict[str, np.ndarray] = {}
    lad = _ladder(H, H.per_block(c))
    cols["ss_cap_l1"], cols["ss_cap_m3"] = H.to_rows(lad["l1"]), H.to_rows(lad["m3"])
    cols["ss_cap_dec"], _ = asof_decayed_mean(horse, day, c, horse, day, halflife_days=365.0)

    # the field's strength as a share of all fields' on earlier days: one key, an as-of mean of "weaker than this"
    field = pd.to_numeric(_col(df, "mr_field"), errors="coerce").to_numpy(float)
    one = np.zeros(len(df), dtype=np.int64)
    mu, _ = asof_decayed_mean(one, day, field, one, day, halflife_days=730.0)
    sq, _ = asof_decayed_mean(one, day, field ** 2, one, day, halflife_days=730.0)
    sd = np.sqrt(np.maximum(sq - mu ** 2, 1e-9))
    strength = 1.0 / (1.0 + np.exp(-(field - mu) / sd))                     # 0..1, a half for an average field
    q = np.sqrt(np.clip(c, 0.0, 1.0) * strength)
    sos, _ = asof_decayed_mean(horse, day, q, horse, day, halflife_days=365.0)
    s = pd.Series(sos, index=df.index)
    g = s.groupby(race)
    top = g.transform("max")
    second = s.where(s < top).groupby(race).transform("max")
    ties = (s == top).groupby(race).transform("sum")
    cols["ss_sos_dec"] = sos
    cols["ss_sos_gap"] = np.where(g.transform("count") >= 2, s - np.where((s == top) & (ties < 2), second, top), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = ((s - g.transform("mean")) / g.transform("std")).where(g.transform("count") >= 3)
    cols["ss_sos_z"] = z.replace([np.inf, -np.inf], np.nan).to_numpy(float)
    sb = H.per_block(strength)
    cols["ss_field_l1"] = H.to_rows(np.where(has_prev, sb[prev], np.nan)) - strength
    runs = pd.to_numeric(_col(df, "career_runs"), errors="coerce").to_numpy(float)
    unexp = pd.Series(np.where(np.isfinite(runs), (runs <= EXPOSED).astype(float), np.nan), index=df.index)
    share = unexp.groupby(race).transform("mean").to_numpy(float)
    cols["ss_met_unexp"] = H.to_rows(_ladder(H, H.per_block(share))["m5"])

    # how formful each race was: Spearman of the BSP order and the finishing order, finishers with a price
    bsp = pd.to_numeric(_col(df, "bfsp"), errors="coerce").to_numpy(float)
    okr = fin & np.isfinite(bsp) & (bsp > 1.0)
    rb = pd.Series(np.where(okr, bsp, np.nan), index=df.index).groupby(race).rank()
    rp = pd.Series(np.where(okr, pos, np.nan), index=df.index).groupby(race).rank()
    k = pd.Series(okr.astype(float), index=df.index).groupby(race).transform("sum")
    d2 = ((rb - rp) ** 2).groupby(race).transform("sum")
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(k >= 4, 1.0 - 6.0 * d2 / (k * (k ** 2 - 1.0)), np.nan)
    lad = _ladder(H, H.per_block(rho))
    cols["ss_formful_l1"], cols["ss_formful_m3"] = H.to_rows(lad["l1"]), H.to_rows(lad["m3"])
    res, nfp = _resid.price_residual(df, day, race)
    w = np.clip((np.nan_to_num(rho, nan=0.0) + 1.0) / 2.0, 0.05, 1.0)
    num, _ = asof_decayed_mean(horse, day, np.where(np.isfinite(res), res * w, np.nan), horse, day, halflife_days=365.0)
    den, _ = asof_decayed_mean(horse, day, np.where(np.isfinite(res), w, np.nan), horse, day, halflife_days=365.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cols["ss_form_w"] = np.where(den > 0, num / den, np.nan)

    dist = pd.to_numeric(_col(df, "dist_furlongs"), errors="coerce").to_numpy(float)
    tb = pd.Series(bands(dist, DIST_BANDS)).fillna(-1).to_numpy().astype(np.int64)
    gg = pd.to_numeric(_col(df, "going_description").map(classify_going).map(GOING), errors="coerce").fillna(-1) \
        .to_numpy().astype(np.int64)
    allm, alln = asof_decayed_mean(horse, day, nfp, horse, day, halflife_days=730.0)
    for tag, cell in (("trip", tb), ("going", gg)):
        key = np.where((horse < 0) | (cell < 0), -1, pd.factorize(horse * 16 + np.where(cell < 0, 0, cell), sort=True)[0])
        inm, inn = asof_decayed_mean(key, day, nfp, key, day, halflife_days=730.0)
        inn, inm = np.nan_to_num(inn), np.nan_to_num(inm)
        outn = np.maximum(np.nan_to_num(alln) - inn, 0.0)
        outs = np.nan_to_num(allm) * np.nan_to_num(alln) - inm * inn
        pin = (inm * inn + K_PYTH * 0.5) / (inn + K_PYTH)             # each side pulled to a half by K_PYTH runs
        pout = (outs + K_PYTH * 0.5) / (outn + K_PYTH)
        cols[f"ss_pyth_{tag}"] = np.where((horse >= 0) & (cell >= 0) & (np.nan_to_num(alln) > 0),
                                          pin ** 2 / (pin ** 2 + pout ** 2), np.nan)
    share_r = pd.Series(pd.to_numeric(_col(df, "mr_share"), errors="coerce").to_numpy(float), index=df.index)
    gs = share_r.groupby(race)
    cnt = gs.transform("count")
    rival = (gs.transform("sum") - share_r) / (cnt - 1).where(cnt > 1)
    cols["ss_pyth_rating"] = (share_r ** 2 / (share_r ** 2 + rival ** 2)).to_numpy(float)

    rows = pd.Series(1.0, index=df.index).groupby(race).transform("sum").to_numpy()
    cols["ss_hp_mr"] = _harmonic(gs.rank(ascending=False, method="min").to_numpy(float), rows)
    cols["ss_hp_sos"] = _harmonic(g.rank(ascending=False, method="min").to_numpy(float), rows)

    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c2 for c2 in FEATURES if c2 in df.columns]), new], axis=1)
