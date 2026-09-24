"""What the draw is worth, from finishing position and lengths beaten, by course, trip and field size.

For each past run from stalls this takes two outcomes, both centred within the
race (`model.race_shape.add_run_outcomes`):

    rs_nfp_c   normalised finishing position less the race average
    rs_lbs_c   pounds beaten (lengths at the trip) less than the race average

and averages them by where the horse was drawn -- its stall's position among
the runners, in fifths of the field, 0 = lowest stall -- in cells of course x
code x trip x field size. Most such cells are thin, so each is pooled down a
hierarchy, coarsest first:

    code x field band                   (is there a draw effect at this field size at all)
    course x code                       (this course, any trip)
    course x code x trip                (this course and trip)
    course x code x trip x field band   (this course, trip and field size)

every level shrunk toward the one above in proportion to its effective sample,
and every past race weighted by ``2 ** (-age / half-life)`` because draw biases
move with rail positions, watering and resurfacing.

Stalls are drawn at random in Britain and Ireland, so the better horses are no
more likely to be drawn low than high and a raw average by draw position is an
unbiased estimate of the draw's effect; it is the thinness of the cells, not a
confound, that needs the pooling. Lengths beaten carry more information per race
than finishing order (a head and ten lengths are both "one place"), which is why
the pound scale is measured alongside the finishing position.

Every statistic of a race uses races on EARLIER DAYS only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.race_shape import (
    FIELD_BANDS, _codes, add_run_outcomes, bands, day_index, field_size, pooled_cell_value, race_code, race_key,
)

#: Draw positions are fifths of the field.
DRAW_BINS = 5

#: Half-life in days of a past race in a draw cell.
DRAW_HALFLIFE_DAYS = 730.0

#: Shrinkage strengths, effective runners, coarsest level first.
DRAW_K = (300.0, 150.0, 80.0, 40.0)

DRAW_OUTCOMES = {"nfp": "rs_nfp_c", "lbs": "rs_lbs_c"}

#: Shrinkage (effective runners) of the draw-by-running-style cell toward the
#: draw cell it refines.
DRAW_STYLE_K = 40.0

DRAW_CURVE_FEATURES = [
    "dc_draw_pct", "dc_edge_nfp", "dc_edge_lbs", "dc_edge_rel_lbs", "dc_race_spread_lbs", "dc_n_eff",
]
#: Added when the frame carries the projected running style (model/race_shape.py first).
DRAW_STYLE_FEATURES = ["dc_edge_style_lbs", "dc_edge_style_rel_lbs"]


def draw_position(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(draw_pct, draw bin): the stall's rank among the race's runners from
    stalls, scaled 0 (lowest) to 1 (highest), and its fifth of the field."""
    rk = race_key(df)
    code = race_code(df)
    stall = pd.to_numeric(df.get("stall"), errors="coerce")
    stall = stall.where(stall.gt(0) & code.isin(["flat", "aw"]))
    rank = stall.groupby(rk).rank(method="min")
    n = stall.groupby(rk).transform("count")
    pct = ((rank - 1) / (n - 1).where(n > 1)).clip(0, 1)
    dbin = np.minimum(np.floor(pct * DRAW_BINS), DRAW_BINS - 1)
    return pct, dbin


def add_draw_curve(df: pd.DataFrame, halflife_days: float = DRAW_HALFLIFE_DAYS, ks=DRAW_K,
                   extra_key: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """The draw-curve block. `extra_key` (e.g. "stall_positioning" or a going
    class) splits the finest level further, for testing whether it matters.

    dc_draw_pct         where the horse is drawn, 0 lowest stall .. 1 highest
    dc_edge_nfp         the draw's worth in centred finishing position here
    dc_edge_lbs         the draw's worth in pounds
    dc_edge_rel_lbs     dc_edge_lbs less the race's mean: the edge over this field
    dc_race_spread_lbs  best draw less worst draw in this race: how much the draw matters today
    dc_n_eff            effective runners behind the finest cell
    """
    if "rs_lbs_c" not in df.columns:
        df = add_run_outcomes(df)
    pct, dbin = draw_position(df)
    df["dc_draw_pct"] = pct
    ok = dbin.notna().to_numpy()
    b = np.where(ok, dbin.fillna(0).to_numpy(), 0).astype(np.int64)
    code = race_code(df)
    track = df["track"].astype(str).str.lower().str.strip()
    dist = pd.to_numeric(df.get("dist_furlongs"), errors="coerce").round(0)
    fb = bands(field_size(df), FIELD_BANDS)
    finest = [track, code, dist, fb] + ([df[extra_key].fillna("").astype(str).str.lower()] if extra_key else [])
    base = [_codes(code, fb), _codes(track, code), _codes(track, code, dist), _codes(*finest)]
    levels = [np.where(ok, k * DRAW_BINS + b, -1) for k in base]
    day = day_index(df)
    rk = race_key(df)
    for name, col in DRAW_OUTCOMES.items():
        y = np.where(ok, df[col].to_numpy(dtype=float), np.nan)
        val, n = pooled_cell_value(levels, day, y, halflife_days, ks)
        val = np.where(ok, val, np.nan)
        df[f"dc_edge_{name}"] = val
        if name == "nfp":
            df["dc_n_eff"] = np.where(ok, n, np.nan)
    e = df["dc_edge_lbs"]
    g = e.groupby(rk)
    df["dc_edge_rel_lbs"] = e - g.transform("mean")
    df["dc_race_spread_lbs"] = g.transform("max") - g.transform("min")
    cols = list(DRAW_CURVE_FEATURES)
    if {"p_lead", "p_prom"} <= set(df.columns):
        # The same draw is not worth the same to every runner: an inside stall at
        # a turning sprint course is worth most to a horse that races handily and
        # can hold the rail. Split the finest cell by the runner's PROJECTED style
        # (forward: p_lead + p_prom >= 0.5), shrunk toward the draw cell itself.
        fwd = ((df["p_lead"] + df["p_prom"]).to_numpy() >= 0.5).astype(np.int64)
        style_lvl = np.where(ok, (base[-1] * DRAW_BINS + b) * 2 + fwd, -1)
        y = np.where(ok, df["rs_lbs_c"].to_numpy(dtype=float), np.nan)
        from model.race_shape import asof_decayed_mean, shrink
        m, n_s = asof_decayed_mean(style_lvl, day, y, style_lvl, day, halflife_days)
        df["dc_edge_style_lbs"] = np.where(ok, shrink(m, n_s, df["dc_edge_lbs"].fillna(0.0).to_numpy(),
                                                      DRAW_STYLE_K), np.nan)
        es = df["dc_edge_style_lbs"]
        df["dc_edge_style_rel_lbs"] = es - es.groupby(rk).transform("mean")
        cols += DRAW_STYLE_FEATURES
    return df, cols
