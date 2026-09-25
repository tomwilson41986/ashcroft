"""Form read against the pace and the draw each past run met.

The shape and draw features describe TODAY's race, and today's race is what the
market prices: iterations 17, 19 and 25 found nothing in them beyond the price
and no gain to the price forecast. What a price can miss is a past run read
against how that race was run. A hold-up horse beaten three lengths in a race
dominated from the front ran better than its margin; a horse that got a soft
lead from the best stall ran worse than its margin says. This block turns the
comments, stalls and margins of each past run into that reading:

    sf_pos_lbs    what the run's ACTUAL early position was worth, in pounds, in
                  races run the way this one was run (its actual shape: no
                  natural leader / one / a contested lead, from the runners'
                  comments) at this course, trip and field size, on EARLIER days;
                  centred within the race
    sf_draw_lbs   what its stall was worth here in pounds (the draw curve's
                  race-centred edge, model/draw_curve.py)
    sf_bias       the two together: + = the run was helped by the pace and draw
    sf_adj        pounds better than the race's average runner (rs_lbs_c) less
                  sf_bias: the run net of what the race handed it

All four are post-race: they describe the run and feed only the windows of later
days. The features, over the windows of model/form_windows.py (career, last run,
last 3, last 5, last 3/5/10 weighted by recency):

    sf_adj_<w>    form net of pace and draw
    sf_bias_<w>   how much the pace and the draw have helped (+) or hindered the
                  horse in its recent runs

three that read today's card, the projected styles and the stalls:

    sf_speed_inside    expected leaders (sum of p_lead) drawn in lower stalls
    sf_speed_outside   expected leaders drawn in higher stalls
    sf_speed_near      expected leaders within two stalls either side

and two that measure the market's own misses rather than the race's: wins less
the BSP's normalised chance (A - E), averaged on earlier days in the cells the
draw curve and the position value use, pooled the same way:

    sf_draw_ae         what runners drawn here have won above what the price gave them
    sf_pos_ae          the same for runners projected to race where this one is
                       projected to, in races of today's projected shape here

Needs the shape and draw blocks built first (model/race_shape.py,
model/draw_curve.py). Every cell and window reads earlier days only; a 06:00 card
and the same day with its results in get identical values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.draw_curve import DRAW_BINS, DRAW_HALFLIFE_DAYS, DRAW_K, draw_position, rating_residual
from model.form_windows import WINDOWS, window_ladder
from model.freshness_features import _Days
from model.race_shape import (
    DIST_BANDS, FIELD_BANDS, PV_HALFLIFE_DAYS, PV_K, _codes, _position_levels, bands, day_index, field_size,
    pooled_cell_value, race_code, race_key,
)

#: A race's actual shape is read only when the comments place at least this share
#: of its runners.
MIN_KNOWN_SHARE = 0.5

#: Stalls either side counted as "near".
NEAR_STALLS = 2

SHAPE_FORM_MEASURES = ["adj", "bias"]
SHAPE_FORM_TODAY = ["sf_speed_inside", "sf_speed_outside", "sf_speed_near", "sf_draw_ae", "sf_pos_ae"]
SHAPE_FORM_FEATURES = [f"sf_{m}_{w}" for m in SHAPE_FORM_MEASURES for w in WINDOWS] + SHAPE_FORM_TODAY

#: Per-run descriptions of the race itself: inputs to the windows, never features.
SHAPE_FORM_POST_RACE = frozenset({"sf_pos_lbs", "sf_draw_lbs", "sf_bias", "sf_adj", "sf_shape_act"})


def actual_shape(df: pd.DataFrame) -> np.ndarray:
    """How each race was actually run, from its runners' own comments: 0 nothing
    led from the start, 1 one runner led, 2 two or more disputed it. NaN when the
    comments place fewer than MIN_KNOWN_SHARE of the field."""
    rk = race_key(df)
    cls = df["rs_class"].to_numpy(dtype=float)
    known = pd.Series(np.isfinite(cls).astype(float)).groupby(rk.to_numpy()).transform("sum").to_numpy()
    leaders = pd.Series((cls == 0).astype(float)).groupby(rk.to_numpy()).transform("sum").to_numpy()
    n = field_size(df).to_numpy(dtype=float)
    shape = np.minimum(leaders, 2.0)
    return np.where(known >= MIN_KNOWN_SHARE * n, shape, np.nan)


def add_run_bias(df: pd.DataFrame, halflife_days: float = PV_HALFLIFE_DAYS, ks=PV_K) -> pd.DataFrame:
    """sf_shape_act, sf_pos_lbs, sf_draw_lbs, sf_bias and sf_adj on every run."""
    rk = race_key(df).to_numpy()
    day = day_index(df)
    shape = actual_shape(df)
    cls = df["rs_class"].to_numpy(dtype=float)
    ok = np.isfinite(shape) & np.isfinite(cls)
    code = race_code(df)
    dist = pd.to_numeric(df.get("dist_furlongs"), errors="coerce")
    fb = bands(field_size(df), FIELD_BANDS)
    db = bands(dist, DIST_BANDS)
    track = df["track"].astype(str).str.lower().str.strip()
    s = np.where(np.isfinite(shape), shape, -1).astype(np.int64)
    c = np.where(np.isfinite(cls), cls, 0).astype(np.int64)
    base = [_codes(code, s), _codes(code, db, fb, s), _codes(track, code, dist.round(0), s),
            _codes(track, code, dist.round(0), fb, s)]
    levels = [np.where(ok, b * 4 + c, -1) for b in base]
    y = df["rs_lbs_c"].to_numpy(dtype=float)
    # Who races where is not random: the cells are measured on the outcome less the
    # part the official ratings predict (fitted on earlier days), so a position's
    # value is what it gave its runners, not what its runners were worth.
    val, _ = pooled_cell_value(levels, day, rating_residual(df, y), halflife_days, ks)
    val = np.where(ok, val, np.nan)
    v = pd.Series(val)
    df["sf_shape_act"] = shape
    df["sf_pos_lbs"] = (v - v.groupby(rk).transform("mean")).to_numpy()
    draw = df["dc_edge_rel_lbs"].to_numpy(dtype=float) if "dc_edge_rel_lbs" in df.columns else np.full(len(df),
                                                                                                        np.nan)
    df["sf_draw_lbs"] = draw
    bias = np.nan_to_num(df["sf_pos_lbs"].to_numpy(dtype=float)) + np.nan_to_num(draw)
    df["sf_bias"] = bias
    df["sf_adj"] = y - bias
    return df


def add_speed_near(df: pd.DataFrame) -> pd.DataFrame:
    """Expected leaders drawn inside, outside and near each runner, from the card.

    Stall numbers, not ranks: an empty stall between two runners still separates
    them. A stall shared by two rows (a data error) counts as neither side."""
    stall = pd.to_numeric(df.get("stall"), errors="coerce")
    stall = stall.where(stall.gt(0) & race_code(df).isin(["flat", "aw"])).to_numpy(dtype=float)
    r = pd.factorize(race_key(df))[0].astype(np.int64)
    p = df["p_lead"].to_numpy(dtype=float)
    ok = np.isfinite(stall)
    cells = (pd.DataFrame({"r": r[ok], "s": stall[ok], "p": np.nan_to_num(p[ok])})
             .groupby(["r", "s"], sort=True)["p"].sum())
    t = cells.reset_index()
    below = (t.groupby("r")["p"].cumsum() - t["p"]).to_numpy()   # the lower stalls' total
    total = t.groupby("r")["p"].transform("sum").to_numpy()
    at = pd.MultiIndex.from_arrays([r[ok], stall[ok]])

    def lookup(values, index=at):
        return pd.Series(values, index=cells.index).reindex(index).to_numpy(dtype=float)

    inside = lookup(below)
    outside = lookup(total) - inside - lookup(cells.to_numpy())
    near = np.zeros(ok.sum())
    for d in [d for d in range(-NEAR_STALLS, NEAR_STALLS + 1) if d]:
        near += np.nan_to_num(lookup(cells.to_numpy(), pd.MultiIndex.from_arrays([r[ok], stall[ok] + d])))
    for name, vals in (("sf_speed_inside", inside), ("sf_speed_outside", outside), ("sf_speed_near", near)):
        col = np.full(len(df), np.nan)
        col[ok] = vals
        df[name] = col
    return df


def market_miss(df: pd.DataFrame) -> np.ndarray:
    """Won (1/0) less the BSP's win probability normalised within the race: what
    each run did beyond what the price gave it. Post-race; unknown without a
    finishing position or a price."""
    rk = pd.factorize(race_key(df))[0]
    bsp = pd.to_numeric(df.get("bfsp"), errors="coerce").to_numpy(dtype=float)
    p = pd.Series(np.where(bsp > 1, 1.0 / bsp, np.nan))
    pn = (p / p.groupby(rk).transform("sum")).to_numpy(dtype=float)
    pos = pd.to_numeric(df.get("placing_numerical"), errors="coerce").to_numpy(dtype=float)
    won = np.where(pos > 0, (pos == 1).astype(float), np.nan)
    return won - pn


def add_market_miss_cells(df: pd.DataFrame, halflife_days: float = DRAW_HALFLIFE_DAYS, ks=DRAW_K) -> pd.DataFrame:
    """sf_draw_ae and sf_pos_ae: the market's miss, on earlier days, by draw cell
    and by projected position in the projected shape. The misses are centred by
    construction (A - E sums to zero over a book), so the pooling shrinks toward 0."""
    y = market_miss(df)
    day = day_index(df)
    _, dbin = draw_position(df)
    ok = dbin.notna().to_numpy()
    b = np.where(ok, dbin.fillna(0).to_numpy(), 0).astype(np.int64)
    code = race_code(df)
    track = df["track"].astype(str).str.lower().str.strip()
    dist = pd.to_numeric(df.get("dist_furlongs"), errors="coerce").round(0)
    fb = bands(field_size(df), FIELD_BANDS)
    base = [_codes(code, fb), _codes(track, code), _codes(track, code, dist), _codes(track, code, dist, fb)]
    val, _ = pooled_cell_value([np.where(ok, k * DRAW_BINS + b, -1) for k in base], day,
                               np.where(ok, y, np.nan), halflife_days, ks)
    df["sf_draw_ae"] = np.where(ok, val, np.nan)
    P = df[["p_lead", "p_prom", "p_mid", "p_rear"]].to_numpy(dtype=float)
    known = np.isfinite(P).all(axis=1)
    g = np.where(known, np.nan_to_num(P).argmax(axis=1), 0).astype(np.int64)
    lv = [np.where((lvl >= 0) & known, lvl * 4 + g, -1) for lvl in _position_levels(df)]
    val, _ = pooled_cell_value(lv, day, y, PV_HALFLIFE_DAYS, PV_K)
    df["sf_pos_ae"] = np.where(known, val, np.nan)
    return df


def add_shape_form(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The block on a frame of history plus (optionally) today's card, after the
    shape and draw blocks. Card rows carry no comment and no result: they read the
    history before their day and add nothing to it."""
    missing = [c for c in ("rs_class", "rs_lbs_c", "p_lead", "shape_bin") if c not in df.columns]
    if missing:
        raise ValueError(f"shape_form needs the shape block built first (missing {missing})")
    df = add_run_bias(df)
    day = day_index(df)
    name = df["horse_name"].fillna("").astype(str).str.strip().str.lower()
    horse = np.where(name.isin(["", "nan", "none"]).to_numpy(), -1, _codes(name))
    H = _Days(horse, day)
    cols = {}
    for m, src in (("adj", "sf_adj"), ("bias", "sf_bias")):
        vals = df[src].to_numpy(dtype=float)
        if m == "bias":
            # a run with no margin says nothing of its bias's cost: windows of the
            # bias follow the runs the adjusted figure has
            vals = np.where(np.isfinite(df["sf_adj"].to_numpy(dtype=float)), vals, np.nan)
        ladder = window_ladder(H, H.per_block(vals))
        for w in WINDOWS:
            cols[f"sf_{m}_{w}"] = H.to_rows(ladder[w])
    df = pd.concat([df.drop(columns=[c for c in cols if c in df.columns]), pd.DataFrame(cols, index=df.index)],
                   axis=1)
    df = add_speed_near(df)
    df = add_market_miss_cells(df)
    return df, list(SHAPE_FORM_FEATURES)
