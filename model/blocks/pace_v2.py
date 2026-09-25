"""Early position and race shape, rebuilt from sharper projections: horse, trainer and jockey, by course and trip.

The owner's ask (25 Sep): improve the modelling of race shape (leader, pace
pressure, closers) with a field-normalised early position, recency weighting of
early positions, and the horse's, the trainer's and the jockey's early positions
at this course and at this course and trip.

Built on the engine's reading of each past run's comment, rs_epf: the early
position, normalised by field size (0 = led, 1 = last). The engine already
projects today's running style from the horse's recent runs (p_lead .. p_rear,
pred_epf) and the trainer's and jockey's overall front-running rates. New here
(p2_):

  the horse's early position over windows (last run, mean of 3 and 5, last 5
      weighted 5..1, exponential in runs with a half-life of 3, career) and its
      trend (last run less the mean of 5);
  the horse at this course, and at this course and trip band, each pulled
      toward the level above (empirical Bayes, the prior worth 2 runs);
  the trainer's and the jockey's runners at this course and at this course and
      trip, pulled toward their own overall record, which is pulled toward the
      average runner (0.5);
  a projection for today from those (the narrowest level with evidence), and
      from it the race: the most forward projection and the gap to the next (a
      clear leader or a contested lead), how many are projected to go forward,
      the heat of the early pace (the three most forward), and for each runner
      how many rivals are projected ahead of it, whether it meets a contested
      lead (a front-runner among others), and the pace it gets set up for (a
      closer behind a hot pace);
  and the market's miss at this course and trip for front-runners and for
      closers (won less the BSP's chance, runs that led or raced in the rear).

Earlier days only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks.form_variants import _ladder
from model.freshness_features import _Days, _col
from model.race_shape import DIST_BANDS, _codes, asof_decayed_mean, bands, day_index, horse_decayed_prior, race_key
from model.shrinkage import shrunk_mean

HALFLIFE_DAYS = 730.0
HORSE_K = 2.0          # a horse's early position at a course, in runs of prior weight
CONN_K = 20.0          # a trainer's or jockey's at a course, in runners
CONN_TOP_K = 50.0      # their overall record toward the average runner
AE_K = 400.0           # the market's miss by course and trip, in runs
FORWARD = 0.25         # a projected early position this forward "goes forward"
REAR = 0.6             # and this far back is a closer

FEATURES = (
    [f"p2_epf_{w}" for w in ("l1", "m3", "m5", "w5", "e3", "car")] + ["p2_epf_trend"]
    + ["p2_horse_track", "p2_horse_td"]
    + ["p2_trainer_track", "p2_trainer_td", "p2_jockey_track", "p2_jockey_td"]
    + ["p2_pred", "p2_pred_rank", "p2_rivals_ahead", "p2_race_front", "p2_race_gap", "p2_race_n_forward",
       "p2_race_heat", "p2_contested_lead", "p2_closer_setup"]
    + ["p2_ae_front_td", "p2_ae_rear_td"]
)
POST_RACE: set[str] = set()


def _name_codes(s: pd.Series) -> np.ndarray:
    s = s.fillna("").astype(str).str.strip().str.lower()
    return np.where(s.isin(["", "nan", "none"]).to_numpy(), -1, _codes(s))


def _asof(key, day, v, halflife=HALFLIFE_DAYS):
    m, n = asof_decayed_mean(key, day, v, key, day, halflife_days=halflife)
    return np.nan_to_num(m) * n, n


def build(df: pd.DataFrame) -> pd.DataFrame:
    if "rs_epf" not in df.columns:
        raise ValueError("pace_v2 reads the engine's rs_epf (the race-shape block): build the engine first")
    day = day_index(df)
    epf = pd.to_numeric(df["rs_epf"], errors="coerce").to_numpy(dtype=float)
    track = _col(df, "track").fillna("").astype(str).str.lower().str.strip()
    dband = bands(pd.to_numeric(_col(df, "dist_furlongs"), errors="coerce"), DIST_BANDS)
    horse = _name_codes(_col(df, "horse_name"))
    cols: dict[str, np.ndarray] = {}

    # the horse's early position over windows
    H = _Days(horse, day)
    lad = _ladder(H, H.per_block(epf))
    for w in ("l1", "m3", "m5", "w5", "car"):
        cols[f"p2_epf_{w}"] = H.to_rows(lad[w])
    names = np.where(horse >= 0, _col(df, "horse_name").fillna("").astype(str).str.lower(), "")
    s, c, _ = horse_decayed_prior(names, day, {"e": epf}, halflife_runs=3)
    with np.errstate(invalid="ignore", divide="ignore"):
        cols["p2_epf_e3"] = np.where((horse >= 0) & (c["e"] > 0), s["e"] / c["e"], np.nan)
    cols["p2_epf_trend"] = cols["p2_epf_l1"] - cols["p2_epf_m5"]

    # the horse at this course and at this course and trip, each toward the level above
    hs, hn = lad["_sum"], lad["_cnt"]
    h_all = shrunk_mean(H.to_rows(hs), H.to_rows(hn), 0.5, HORSE_K)
    k_track = np.where(horse >= 0, _codes(horse, track), -1)
    k_td = np.where(horse >= 0, _codes(horse, track, dband), -1)
    t_s, t_n = _asof(k_track, day, epf, np.inf)
    h_track = shrunk_mean(t_s, t_n, h_all, HORSE_K)
    d_s, d_n = _asof(k_td, day, epf, np.inf)
    h_td = shrunk_mean(d_s, d_n, h_track, HORSE_K)
    cols["p2_horse_track"] = np.where(horse >= 0, h_track, np.nan)
    cols["p2_horse_td"] = np.where(horse >= 0, h_td, np.nan)

    # the trainer's and the jockey's runners, by course and by course and trip
    for who, col in (("trainer", "trainer"), ("jockey", "jockey_name")):
        e = _name_codes(_col(df, col))
        a_s, a_n = _asof(e, day, epf)
        top = shrunk_mean(a_s, a_n, 0.5, CONN_TOP_K)
        tr = np.where(e >= 0, _codes(e, track), -1)
        s1, n1 = _asof(tr, day, epf)
        at_track = shrunk_mean(s1, n1, top, CONN_K)
        td = np.where(e >= 0, _codes(e, track, dband), -1)
        s2, n2 = _asof(td, day, epf)
        at_td = shrunk_mean(s2, n2, at_track, CONN_K)
        cols[f"p2_{who}_track"] = np.where(e >= 0, at_track, np.nan)
        cols[f"p2_{who}_td"] = np.where(e >= 0, at_td, np.nan)

    # today's projection: the horse's own course-and-trip reading where it has
    # history, otherwise its connections' (the trainer's runners at the course and trip)
    has_h = np.isfinite(cols["p2_epf_car"])
    pred = np.where(has_h, h_td, cols["p2_trainer_td"])
    pred = np.where(np.isfinite(pred), pred, 0.5)
    cols["p2_pred"] = pred
    rk = race_key(df).to_numpy()
    g = pd.DataFrame({"r": rk, "p": pred})
    cols["p2_pred_rank"] = g.groupby("r")["p"].rank(method="average").to_numpy()
    cols["p2_rivals_ahead"] = g.groupby("r")["p"].rank(method="min").to_numpy() - 1
    srt = g.sort_values(["r", "p"])
    srt["i"] = srt.groupby("r").cumcount()
    first = srt[srt["i"] == 0].set_index("r")["p"]
    second = srt[srt["i"] == 1].set_index("r")["p"]
    heat = srt[srt["i"] < 3].groupby("r")["p"].mean()
    nfwd = g.assign(f=(g["p"] < FORWARD).astype(float)).groupby("r")["f"].sum()
    cols["p2_race_front"] = pd.Series(rk).map(first).to_numpy(dtype=float)
    cols["p2_race_gap"] = pd.Series(rk).map(second - first).to_numpy(dtype=float)
    cols["p2_race_heat"] = pd.Series(rk).map(heat).to_numpy(dtype=float)
    n_forward = pd.Series(rk).map(nfwd).to_numpy(dtype=float)
    cols["p2_race_n_forward"] = n_forward
    forward = pred < FORWARD
    cols["p2_contested_lead"] = np.where(forward, n_forward - 1, 0.0)
    cols["p2_closer_setup"] = np.where(pred > REAR, n_forward, 0.0)

    # the market's miss for front-runners and closers here, earlier days
    bsp = pd.to_numeric(_col(df, "bfsp"), errors="coerce").to_numpy(dtype=float)
    race = pd.factorize(rk)[0]
    p = pd.Series(np.where(bsp > 1, 1.0 / bsp, np.nan))
    pn = (p / p.groupby(race).transform("sum")).to_numpy(dtype=float)
    pos = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(dtype=float)
    ran = np.isfinite(pos) & (pos > 0)
    ae = np.where(ran, (pos == 1).astype(float) - pn, np.nan)
    ktd = _codes(track, dband)
    for name, sel in (("front", epf < FORWARD), ("rear", epf > REAR)):
        s_, n_ = _asof(ktd, day, np.where(sel, ae, np.nan))
        cols[f"p2_ae_{name}_td"] = shrunk_mean(s_, n_, 0.0, AE_K)

    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
