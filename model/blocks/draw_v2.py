"""The draw by course, trip, going and stall placement, on NFP, pounds beaten and the result against the price.

The owner's ask (25 Sep): there is more in the draw by track, by distance, by
going and by where the stalls are placed, measured on finishing position, lengths
beaten and the result against the price.

What was tried before (reports/draw_pace_rebuild.md s3.4): splitting the finest
draw-curve cell by going or by stall placement added nothing to how well the
cells predict pounds beaten (within-race R2 x1000: going -0.01, stall placement
+0.05), and the draw block added nothing to the win model beyond the BSP. What
was not tried, and is here:

- each split as its own level of the hierarchy, so a going or stall-placement
  effect is measured against the cell it refines rather than hidden in a thin
  finest cell;
- the result against the price (won less the BSP's normalised chance) as an
  outcome beside NFP and pounds beaten: whether the market misses a draw bias,
  not whether one exists;
- the increments themselves as features: what the going adds to the course-and-
  trip cell, and what the stall placement adds to that;
- and the test that decides it, the price forecast, where the old splits were
  never measured.

Cells (draw position in fifths of the stalls in use, coarsest first):
    code x field band > course x code > course x code x trip band
    > ... x going (fast / soft / all-weather) > ... x stall placement
each pooled toward the one above in proportion to its effective sample
(model/shrinkage.py), every past race weighted by a two-year half-life. Flat and
all-weather races from stalls only; jumps get none.

Features (d2_, per outcome nfp / lbs / ae):
    d2_<o>          the finest cell's estimate for the horse's draw
    d2_<o>_rel      that less the race's mean: the edge over this field
    d2_<o>_going    what the going adds to the course-and-trip cell
    d2_<o>_stalls   what the stall placement adds to that
    d2_n_going, d2_n_stalls   effective runners behind the two finest cells

Earlier days only: every cell reads races on days before the row's own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.draw_curve import DRAW_BINS, DRAW_HALFLIFE_DAYS, DRAW_K, draw_position
from model.draw_metrics import classify_going
from model.freshness_features import _col
from model.race_shape import (DIST_BANDS, FIELD_BANDS, _codes, asof_decayed_mean, bands, day_index,
                              field_size, race_code, race_key)
from model.shrinkage import shrink_chain

OUTCOMES = ("nfp", "lbs", "ae")
#: Prior weights (effective runners), coarsest level first: the draw curve's
#: tuned four, the stall-placement level as strong as the finest of them. The
#: result against the price is a win-or-lose outcome, far noisier per runner, so
#: it pools four times as hard.
KS = tuple(DRAW_K) + (DRAW_K[-1],)
AE_KS = tuple(4.0 * k for k in KS)
LBS_CAP = 20.0

FEATURES = ([f"d2_{o}{s}" for o in OUTCOMES for s in ("", "_rel", "_going", "_stalls")]
            + ["d2_n_going", "d2_n_stalls"])
POST_RACE: set[str] = set()


def going_group(df: pd.DataFrame) -> pd.Series:
    g = _col(df, "going_description").map(classify_going)
    return g.map({"Firm": "fast", "GtF": "fast", "Good": "fast", "GtS": "soft", "Soft": "soft",
                  "Heavy": "soft", "AW": "aw"}).fillna("u")


def run_outcomes(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """A run's result three ways, centred on its race (post-race)."""
    from model.perf_figures import lbs_per_length, parse_beaten_lengths
    rk = race_key(df)
    race = pd.factorize(rk)[0]
    n = field_size(df).to_numpy(dtype=float)
    pos = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(dtype=float)
    pos = np.where(pos > 0, pos, np.nan)
    fin = np.isfinite(pos)
    with np.errstate(invalid="ignore", divide="ignore"):
        nfp = np.clip((n - pos) / np.where(n > 1, n - 1, np.nan), 0, 1)
    if "LB" in df.columns:
        lb = pd.to_numeric(df["LB"], errors="coerce").to_numpy(dtype=float)
    else:
        lb = _col(df, "total_dst_bt").map(parse_beaten_lengths).to_numpy(dtype=float)
    lb = np.where(pos == 1, 0.0, np.where(fin, lb, np.nan))
    dist = pd.to_numeric(_col(df, "dist_furlongs"), errors="coerce").to_numpy(dtype=float)
    lbs = np.minimum(lb * lbs_per_length(np.nan_to_num(dist, nan=8.0)), LBS_CAP)
    g = lambda v: pd.Series(v).groupby(race).transform("mean").to_numpy(dtype=float)   # noqa: E731
    bsp = pd.to_numeric(_col(df, "bfsp"), errors="coerce").to_numpy(dtype=float)
    p = pd.Series(np.where(bsp > 1, 1.0 / bsp, np.nan))
    pn = (p / p.groupby(race).transform("sum")).to_numpy(dtype=float)
    won = np.where(fin, (pos == 1).astype(float), np.nan)
    return {"nfp": nfp - g(nfp), "lbs": g(lbs) - lbs, "ae": won - pn}


def build(df: pd.DataFrame) -> pd.DataFrame:
    pct, dbin = draw_position(df)
    ok = dbin.notna().to_numpy()
    b = np.where(ok, dbin.fillna(0).to_numpy(), 0).astype(np.int64)
    code = race_code(df)
    track = _col(df, "track").fillna("").astype(str).str.lower().str.strip()
    dband = bands(pd.to_numeric(_col(df, "dist_furlongs"), errors="coerce"), DIST_BANDS)
    fband = bands(field_size(df), FIELD_BANDS)
    going = going_group(df)
    stalls = _col(df, "stall_positioning").fillna("").astype(str).str.lower().str.strip().replace("", "u")
    keys = [_codes(code, fband), _codes(track, code), _codes(track, code, dband),
            _codes(track, code, dband, going), _codes(track, code, dband, going, stalls)]
    levels = [np.where(ok, k * DRAW_BINS + b, -1) for k in keys]
    day = day_index(df)
    rk = race_key(df)
    y = run_outcomes(df)
    cols: dict[str, np.ndarray] = {}
    n_eff = []
    for o in OUTCOMES:
        v = np.where(ok, y[o], np.nan)
        sums = []
        for lev in levels:
            m, n = asof_decayed_mean(lev, day, v, lev, day, halflife_days=DRAW_HALFLIFE_DAYS)
            sums.append((np.nan_to_num(m) * n, n))
            if o == "nfp":
                n_eff.append(n)
        est = shrink_chain(sums, AE_KS if o == "ae" else KS, top_prior=0.0)
        finest = np.where(ok, est[-1], np.nan)
        cols[f"d2_{o}"] = finest
        cols[f"d2_{o}_rel"] = finest - pd.Series(finest).groupby(rk.to_numpy()).transform("mean").to_numpy()
        cols[f"d2_{o}_going"] = np.where(ok, est[3] - est[2], np.nan)
        cols[f"d2_{o}_stalls"] = np.where(ok, est[4] - est[3], np.nan)
    cols["d2_n_going"] = np.where(ok, n_eff[3], np.nan)
    cols["d2_n_stalls"] = np.where(ok, n_eff[4], np.nan)
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
