"""HorseRaceBase's Ratings Machine and System Builder components, rebuilt from our own results (the owner, 18 Nov 2026:
"here are the features in ratings machine ... test them, work out predictiveness, see what weights should be"; "evaluate
these features in the HorseRaceBase system builder. Test them on our own data. Do not hit HRB").

The Ratings Machine scores each earlier run (class, prize, the race's top and winner's OR, the finishing place and
the share of the field beaten, distance beaten, non-finishers), weights runs by how long ago they were, and adds speed
(best, average and last-N figures, by track, distance and going), jockey, trainer and sire strike rates over windows,
weight against the top weight, the jockey's claim, the OR's change, and the horse's win and place rates by course,
going, trip, class, handicap type and code. The System Builder filters on the same quantities and the last runs'
market. Each is built here as HRB defines it, raw (no shrinkage), from EARLIER DAYS only, and grouped in sets so the
weights of each set can be fitted (research/queries/hrb_machine_weights_1118.py):

  form    hm_pr_l1..l3 (share of the field beaten, a non-finisher 0), hm_pr_m3, hm_pr_w10 (the last ten, run k back
          weighted 0.85^k), hm_won_l1, hm_wins_l6, hm_places_l6, hm_lbs_l1, hm_lbs_m3 (pounds beaten, capped at 30),
          hm_nonfin_l5, hm_runs, hm_dsr (days since the last run), hm_dslw (since the last win)
  class   hm_cls_l1, hm_cls_m3 (class level: Group/Grade 1 = 10 ... Class 7 = 1), hm_cls_move (today's level less the
          last run's), hm_prize_l1, hm_prize_max3 (log prize), hm_prize_move, hm_topor_l1, hm_winor_l1 (the last race's
          top and winner's official rating)
  speed   hm_fig_l1..l3, hm_fig_m3, hm_fig_best6 (model/blocks/time_figure's horse figure), hm_fig_avg_dist,
          hm_fig_avg_going, hm_fig_avg_track, hm_fig_avg_td (the horse's average figure at today's trip band, going,
          course, course and trip)
  conn    hm_jk_w2y, hm_jk_w30, hm_jk_w14, hm_jk_trk, hm_jk_horse, hm_tr_w2y, hm_tr_w30, hm_tr_w14, hm_tr_trk, hm_tj,
          hm_sire_w, hm_sire_dist, hm_sire_going (win rates, decayed: 2 years = half-life 365 days, 30 days = 15,
          14 days = 7; course and pairs over two years), hm_tr_n14 (the yard's runners, decayed 7 days)
  horse   hm_h_{win,plc}_{car,trk,going,dist,cls,hcap,code,td} (H-Win% and H-Plc% by condition, raw) and
          hm_h_n_{trk,going,dist,td} (the runs behind them)
  weight  hm_wt_vs_top (pounds less the race's top weight), hm_claim, hm_or_chg_lr, hm_or_chg_lw (today's OR less the
          last run's and the last win's), hm_or_vs_top (less the race's top OR)
  market  hm_lr_lbsp (log BSP of the last run), hm_lr_fav_rank (its rank in that race's betting), hm_lr_field

Not rebuilt: the System Builder's early-odds moves (no early price in the results), owner (not held), H2H and
draw (served blocks of their own), subsequent form (the later wins of a race's runners: kept out, as a run's
reading must not change after its day).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks import time_figure as _tf
from model.blocks.draw_v2 import going_group
from model.freshness_features import _Days, _col
from model.race_shape import DIST_BANDS, _codes, asof_decayed_mean, bands, day_index, race_code, race_key

SETS = {
    "form": ["hm_pr_l1", "hm_pr_l2", "hm_pr_l3", "hm_pr_m3", "hm_pr_w10", "hm_won_l1", "hm_wins_l6", "hm_places_l6",
             "hm_lbs_l1", "hm_lbs_m3", "hm_nonfin_l5", "hm_runs", "hm_dsr", "hm_dslw"],
    "class": ["hm_cls_l1", "hm_cls_m3", "hm_cls_move", "hm_prize_l1", "hm_prize_max3", "hm_prize_move", "hm_topor_l1",
              "hm_winor_l1"],
    "speed": ["hm_fig_l1", "hm_fig_l2", "hm_fig_l3", "hm_fig_m3", "hm_fig_best6", "hm_fig_avg_dist",
              "hm_fig_avg_going", "hm_fig_avg_track", "hm_fig_avg_td"],
    "conn": ["hm_jk_w2y", "hm_jk_w30", "hm_jk_w14", "hm_jk_trk", "hm_jk_horse", "hm_tr_w2y", "hm_tr_w30", "hm_tr_w14",
             "hm_tr_trk", "hm_tj", "hm_sire_w", "hm_sire_dist", "hm_sire_going", "hm_tr_n14"],
    "horse": [f"hm_h_{m}_{c}" for c in ("car", "trk", "going", "dist", "cls", "hcap", "code", "td") for m in ("win", "plc")]
             + ["hm_h_n_trk", "hm_h_n_going", "hm_h_n_dist", "hm_h_n_td"],
    "weight": ["hm_wt_vs_top", "hm_claim", "hm_or_chg_lr", "hm_or_chg_lw", "hm_or_vs_top"],
    "market": ["hm_lr_lbsp", "hm_lr_fav_rank", "hm_lr_field"],
}
FEATURES = [f for v in SETS.values() for f in v]
POST_RACE: set[str] = set()
READS = sorted(set(_tf.READS) | {"raceid", "race_date", "race_time", "track", "horse_name", "number_of_runners",
                                 "placing_numerical", "LB", "total_dst_bt", "dist_furlongs", "race_class", "prize_money",
                                 "official_rating", "pounds", "jockeys_claim", "jockey_name", "trainer", "stallion",
                                 "going_description", "race_type", "surface_type", "bfsp"})
LBS_CAP = 30.0
DECAY = 0.85


def _key(s: pd.Series) -> np.ndarray:
    s = s.fillna("").astype(str).str.strip().str.lower()
    return np.where(s.isin(["", "nan", "none"]).to_numpy(), -1, _codes(s))


def _pair(*keys: np.ndarray) -> np.ndarray:
    ok = np.logical_and.reduce([k >= 0 for k in keys])
    joined = pd.Series(keys[0].astype(str))
    for k in keys[1:]:
        joined = joined + "|" + pd.Series(k.astype(str))
    return np.where(ok, pd.factorize(joined)[0], -1).astype(np.int64)


def class_level(text: pd.Series) -> np.ndarray:
    """HRB's class ladder as a level, higher = better: Group/Grade 1 = 10, 2 = 9, 3 = 8, Listed = 7, Class 1 = 7 ...
    Class 7 = 1 ('Class 4' or a bare '4'); anything else NaN."""
    t = text.fillna("").astype(str).str.lower()
    g = pd.to_numeric(t.str.extract(r"(?:group|grade|g)\s*([123])\b", expand=False), errors="coerce")
    c = pd.to_numeric(t.str.extract(r"class\s*([1-7])\b", expand=False), errors="coerce")
    c = c.fillna(pd.to_numeric(t.str.extract(r"^\s*([1-7])(?:\.0)?\s*$", expand=False), errors="coerce"))  # bare "4"
    lvl = np.where(g.notna(), 11 - g, np.where(t.str.contains("listed"), 7, 8 - c))
    return pd.to_numeric(pd.Series(lvl), errors="coerce").to_numpy(dtype=float)


def _num(df, name) -> np.ndarray:
    return pd.to_numeric(_col(df, name), errors="coerce").to_numpy(dtype=float)


def build(df: pd.DataFrame) -> pd.DataFrame:
    from model.perf_figures import lbs_per_length, parse_beaten_lengths
    day = day_index(df)
    race = pd.factorize(race_key(df))[0]
    n = pd.to_numeric(_col(df, "number_of_runners"), errors="coerce")
    n = n.fillna(pd.Series(race).map(pd.Series(race).value_counts())).to_numpy(dtype=float)
    pos = _num(df, "placing_numerical")
    pos = np.where(pos > 0, pos, np.nan)
    has_result = pd.Series(np.isfinite(pos)).groupby(race).transform("any").to_numpy()
    fin = np.isfinite(pos)
    with np.errstate(invalid="ignore", divide="ignore"):
        pr = np.where(fin, np.clip((n - pos) / np.where(n > 1, n - 1, np.nan), 0, 1), np.where(has_result, 0.0, np.nan))
    won = np.where(has_result, (pos == 1).astype(float), np.nan)
    terms = np.where(n >= 8, 3, np.where(n >= 5, 2, 1))
    placed = np.where(has_result, (fin & (pos <= terms)).astype(float), np.nan)
    nonfin = np.where(has_result, (~fin).astype(float), np.nan)
    lb = _num(df, "LB") if "LB" in df.columns else _col(df, "total_dst_bt").map(parse_beaten_lengths).to_numpy(dtype=float)
    dist = _num(df, "dist_furlongs")
    lbs = np.where(pos == 1, 0.0, np.where(fin, np.minimum(lb * lbs_per_length(np.nan_to_num(dist, nan=8.0)), LBS_CAP),
                                           np.nan))
    cls = class_level(_col(df, "race_class"))
    prize = np.log1p(np.clip(_num(df, "prize_money"), 0, None))
    orr = _num(df, "official_rating")
    orr = np.where(orr > 0, orr, np.nan)
    top_or = pd.Series(orr).groupby(race).transform("max").to_numpy(dtype=float)
    win_or = pd.Series(np.where(pos == 1, orr, np.nan)).groupby(race).transform("max").to_numpy(dtype=float)
    win_or = np.where(has_result, win_or, np.nan)
    pounds = _num(df, "pounds")
    top_wt = pd.Series(pounds).groupby(race).transform("max").to_numpy(dtype=float)
    bsp = _num(df, "bfsp")
    lbsp = np.where(bsp > 1, np.log(np.where(bsp > 1, bsp, np.nan)), np.nan)
    fav_rank = pd.Series(np.where(bsp > 1, bsp, np.nan)).groupby(race).rank(method="min").to_numpy(dtype=float)
    fig = _tf.figures(df)["fig"]

    horse = _key(df["horse_name"])
    H = _Days(horse, day)
    nb = len(H.u)

    def lag(values, k):
        b = H.per_block(values)
        idx = np.arange(nb) - k
        ok = H.valid & (H.pos >= k)
        return H.to_rows(np.where(ok, b[np.clip(idx, 0, None)], np.nan))

    def lags(values, k_max):
        return np.column_stack([lag(values, k) for k in range(1, k_max + 1)])

    cols: dict[str, np.ndarray] = {}
    P = lags(pr, 10)
    cols["hm_pr_l1"], cols["hm_pr_l2"], cols["hm_pr_l3"] = P[:, 0], P[:, 1], P[:, 2]
    with np.errstate(invalid="ignore"):
        cols["hm_pr_m3"] = np.nanmean(np.where(np.isfinite(P[:, :3]), P[:, :3], np.nan), axis=1) \
            if P.shape[0] else np.array([])
        w = DECAY ** np.arange(1, 11)
        okP = np.isfinite(P)
        cols["hm_pr_w10"] = np.where(okP.any(axis=1), np.nansum(np.where(okP, P, 0) * w, axis=1)
                                     / np.where(okP.any(axis=1), (okP * w).sum(axis=1), np.nan), np.nan)
        W = lags(won, 6)
        cols["hm_won_l1"] = W[:, 0]
        cols["hm_wins_l6"] = np.where(np.isfinite(W).any(axis=1), np.nansum(W, axis=1), np.nan)
        PL = lags(placed, 6)
        cols["hm_places_l6"] = np.where(np.isfinite(PL).any(axis=1), np.nansum(PL, axis=1), np.nan)
        L = lags(lbs, 3)
        cols["hm_lbs_l1"] = L[:, 0]
        cols["hm_lbs_m3"] = np.nanmean(L, axis=1)
        NF = lags(nonfin, 5)
        cols["hm_nonfin_l5"] = np.where(np.isfinite(NF).any(axis=1), np.nansum(NF, axis=1), np.nan)
    cols["hm_runs"] = H.to_rows(np.where(H.valid, H.pos, np.nan))
    prev_day = lag(day.astype(float), 1)
    cols["hm_dsr"] = np.where(np.isfinite(prev_day), day - prev_day, np.nan)
    wb = H.per_block(np.nan_to_num(won), how="max") > 0
    lw = H.last_before(wb)
    lw_day = np.where(lw >= 0, H.day[np.clip(lw, 0, None)], np.nan)
    cols["hm_dslw"] = H.to_rows(np.where(lw >= 0, H.day - lw_day, np.nan))
    or_b = H.per_block(orr)
    cols["hm_or_chg_lw"] = orr - H.to_rows(np.where(lw >= 0, or_b[np.clip(lw, 0, None)], np.nan))

    C = lags(cls, 3)
    cols["hm_cls_l1"] = C[:, 0]
    with np.errstate(invalid="ignore"):
        cols["hm_cls_m3"] = np.nanmean(C, axis=1)
        PZ = lags(prize, 3)
        cols["hm_prize_max3"] = np.nanmax(np.where(np.isfinite(PZ), PZ, -np.inf), axis=1)
        cols["hm_prize_max3"] = np.where(np.isfinite(cols["hm_prize_max3"]), cols["hm_prize_max3"], np.nan)
        F = lags(fig, 6)
        cols["hm_fig_m3"] = np.nanmean(F[:, :3], axis=1)
        best = np.nanmax(np.where(np.isfinite(F), F, -np.inf), axis=1)
        cols["hm_fig_best6"] = np.where(np.isfinite(best), best, np.nan)
    cols["hm_cls_move"] = cls - C[:, 0]
    cols["hm_prize_l1"] = PZ[:, 0]
    cols["hm_prize_move"] = prize - PZ[:, 0]
    cols["hm_topor_l1"] = lag(top_or, 1)
    cols["hm_winor_l1"] = lag(win_or, 1)
    cols["hm_fig_l1"], cols["hm_fig_l2"], cols["hm_fig_l3"] = F[:, 0], F[:, 1], F[:, 2]

    track = _key(_col(df, "track"))
    dband = bands(dist, DIST_BANDS)
    dband = np.where(np.isfinite(dband), dband, -1).astype(np.int64)
    going = pd.factorize(going_group(df))[0].astype(np.int64)
    code = pd.factorize(race_code(df))[0].astype(np.int64)
    hcap = _col(df, "race_type").fillna("").astype(str).str.lower().str.contains("handicap|nursery").to_numpy()
    hcap = hcap.astype(np.int64)
    clsk = np.where(np.isfinite(cls), cls, -1).astype(np.int64)

    def mean(key, value, half=np.inf):
        return asof_decayed_mean(key, day, value, key, day, halflife_days=half)

    for name, key in (("dist", _pair(horse, dband)), ("going", _pair(horse, going)), ("track", _pair(horse, track)),
                      ("td", _pair(horse, track, dband))):
        cols[f"hm_fig_avg_{name}"], _ = mean(key, fig)

    jockey, trainer, sire = _key(_col(df, "jockey_name")), _key(_col(df, "trainer")), _key(_col(df, "stallion"))
    for nm, half in (("w2y", 365.0), ("w30", 15.0), ("w14", 7.0)):
        cols[f"hm_jk_{nm}"], _ = mean(jockey, won, half)
        cols[f"hm_tr_{nm}"], tn = mean(trainer, won, half)
        if nm == "w14":
            cols["hm_tr_n14"] = np.where(trainer >= 0, np.nan_to_num(tn), np.nan)
    cols["hm_jk_trk"], _ = mean(_pair(jockey, track), won, 730.0)
    cols["hm_tr_trk"], _ = mean(_pair(trainer, track), won, 730.0)
    cols["hm_jk_horse"], _ = mean(_pair(jockey, horse), won)
    cols["hm_tj"], _ = mean(_pair(trainer, jockey), won, 730.0)
    cols["hm_sire_w"], _ = mean(sire, won, 730.0)
    cols["hm_sire_dist"], _ = mean(_pair(sire, dband), won, 730.0)
    cols["hm_sire_going"], _ = mean(_pair(sire, going), won, 730.0)

    for name, key in (("car", horse), ("trk", _pair(horse, track)), ("going", _pair(horse, going)),
                      ("dist", _pair(horse, dband)), ("cls", _pair(horse, clsk)), ("hcap", _pair(horse, hcap)),
                      ("code", _pair(horse, code)), ("td", _pair(horse, track, dband))):
        cols[f"hm_h_win_{name}"], nn = mean(key, won)
        cols[f"hm_h_plc_{name}"], _ = mean(key, placed)
        if name in ("trk", "going", "dist", "td"):
            cols[f"hm_h_n_{name}"] = np.where(key >= 0, np.nan_to_num(nn), np.nan)

    cols["hm_wt_vs_top"] = pounds - top_wt
    cols["hm_claim"] = _num(df, "jockeys_claim")
    cols["hm_or_chg_lr"] = orr - lag(orr, 1)
    cols["hm_or_vs_top"] = orr - top_or
    cols["hm_lr_lbsp"] = lag(lbsp, 1)
    cols["hm_lr_fav_rank"] = lag(fav_rank, 1)
    cols["hm_lr_field"] = lag(n, 1)

    new = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in cols.items()}, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
