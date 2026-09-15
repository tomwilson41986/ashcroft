"""
Horse-jockey agent parameters, built from what the database actually holds.

The ABM report assumes McLloyd 10 Hz GPS and Equimetre biometrics. Ashcroft
has none of that, so every agent parameter here is derived from columns the
pipeline already computes (lag-safe form features) plus race-card facts:

    ability (lbs)   official rating, weight carried, jockey claim, lagged
                    speed figures  -> sustainable speed via the lbs->lengths->
                    speed conversion in model/perf_figures.py
    style           predicted early position (pace_metrics) -> early effort
                    and the gap the jockey wants to sit off the leader
    finishing kick  career late-move / hold-up finishing ability -> W'
    keenness        horse_keen_rate -> wasted energy when restrained
    trouble         horse_trouble_rate -> hazard multiplier
    jockey          jockey_front_rate -> aggression (how early the kick comes)
    draw            stall -> starting lane

An alternative ability source is the market itself (``abilities="market"``):
lbs = ln(p_i / mean p) / beta. That is the self-consistency check — an
interaction-free ABM fed market abilities should reproduce the market.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from model.abm.track import going_speed_factor
from model.perf_figures import FURLONG_METRES, LENGTH_METRES, lbs_per_length, lbs_to_speed_factor

# par sustainable speed (m/s) on good ground for an average-class field, by trip
PAR_SPEED_TABLE = {5: 17.3, 6: 17.0, 7: 16.7, 8: 16.4, 9: 16.2, 10: 16.0, 11: 15.85,
                   12: 15.7, 14: 15.4, 16: 15.2, 18: 15.0, 20: 14.8}

STYLE_COLS = ["pred_early_pos", "horse_early_pos_3r", "horse_career_early_pos",
              "Horse_Career_EPF", "LR_EPF", "LR_early_pos", "EPF"]


def par_speed(dist_furlongs: float, going_description=None) -> float:
    xs = np.array(sorted(PAR_SPEED_TABLE)); ys = np.array([PAR_SPEED_TABLE[k] for k in xs])
    return float(np.interp(dist_furlongs, xs, ys)) * going_speed_factor(going_description)


@dataclass
class AbilityConfig:
    rating_col: str | None = None      # explicit lbs rating column (overrides composite)
    use_weight: bool = True            # OR - (pounds - race mean pounds)
    speed_fig_col: str = "LR3_RSR"     # lagged speed figure (%), optional
    speed_fig_weight: float = 0.3
    speed_fig_cap_lbs: float = 5.0
    perf_col: str = "horse_perf_lbs_ewm"  # lag-safe expected performance figure (lbs)
    perf_weight: float = 1.0           # weight on (perf figure - OR) in the composite
    market_beta: float = 0.22          # log-odds per lb; with 7 lb form noise the solo ABM reproduces the market at ~0.22
    day_noise_lbs: float = 7.0         # per-run performance sd in lbs (Timeform-scale figures vary ~7 lbs)
    unexposed_noise_mult: float = 1.3  # extra uncertainty for horses with < 3 runs
    cp_fraction_of_par: float = 0.97   # sustainable speed as a fraction of average race speed
    w_prime_base_m: float = 80.0
    dv_surge_table: dict = field(default_factory=lambda: {5: 2.3, 8: 2.0, 12: 1.8, 16: 1.6})
    kick_table: dict = field(default_factory=lambda: {5: 300.0, 8: 400.0, 12: 500.0, 16: 600.0})


@dataclass
class FieldParams:
    names: list
    ability_lbs: np.ndarray
    v_cp: np.ndarray
    dv_surge: np.ndarray
    w_prime: np.ndarray
    style: np.ndarray
    early_effort: np.ndarray
    desired_gap_m: np.ndarray
    kick_dist_m: np.ndarray
    aggression: np.ndarray
    rail_seek: np.ndarray
    keenness: np.ndarray
    trouble_prone: np.ndarray
    form_sd_lbs: np.ndarray
    stall: np.ndarray
    dist_furlongs: float
    length_m: float
    going: str | None = None
    track_name: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.names)


def _col(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _first_available(df: pd.DataFrame, cols, default: float) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for c in cols:
        if c in df.columns:
            out = out.fillna(pd.to_numeric(df[c], errors="coerce"))
    return out.fillna(default)


def ability_lbs_composite(df: pd.DataFrame, cfg: AbilityConfig) -> np.ndarray:
    """Race-relative ability in lbs.

    ability = expected performance figure - weight carried (relative to the field)

    The expected figure is the lag-safe recency-weighted mean of the horse's
    past performance figures (``perf_col``, from model.perf_figures) when
    available, else the official rating. In a handicap the weights offset
    the ratings by construction, so what remains is exactly "how far the
    horse's form exceeds its mark" — the handicapping question.
    """
    orr = _col(df, "official_rating")
    orr = orr.where(orr > 0, _col(df, "median_or"))
    orr = orr.where(orr > 0, np.nan)
    if cfg.rating_col and cfg.rating_col in df.columns:
        base = _col(df, cfg.rating_col)
    elif cfg.perf_col in df.columns and _col(df, cfg.perf_col).notna().any():
        perf = _col(df, cfg.perf_col)
        base = orr + cfg.perf_weight * (perf - orr).fillna(0.0)
        base = base.fillna(perf)
    else:
        base = orr
    base = base.fillna(base.median() if base.notna().any() else 0.0)
    if cfg.use_weight and "pounds" in df.columns:
        lbs = _col(df, "pounds")
        base = base - (lbs - lbs.mean()).fillna(0.0)
    if cfg.speed_fig_col in df.columns:
        rsr = _col(df, cfg.speed_fig_col)
        if rsr.notna().sum() >= 2:
            dist = float(_col(df, "dist_furlongs").median())
            lbs_per_pct = (dist * FURLONG_METRES / 100.0 / LENGTH_METRES) * float(lbs_per_length(dist))
            adj = cfg.speed_fig_weight * (rsr - rsr.mean()) * lbs_per_pct
            base = base + adj.clip(-cfg.speed_fig_cap_lbs, cfg.speed_fig_cap_lbs).fillna(0.0)
    return (base - base.mean()).values.astype(float)


def ability_lbs_market(df: pd.DataFrame, cfg: AbilityConfig, price_col: str = "bfsp") -> np.ndarray:
    p = 1.0 / _col(df, price_col)
    p = (p / p.sum()).fillna(1.0 / max(len(df), 1))
    lbs = np.log(p / p.mean()) / cfg.market_beta
    return (lbs - lbs.mean()).fillna(0.0).values.astype(float)


def build_field(df_race: pd.DataFrame, abilities: str = "composite", cfg: AbilityConfig | None = None,
                rng: np.random.Generator | None = None) -> FieldParams:
    """Build agent parameters for one race (one row per runner)."""
    cfg = cfg or AbilityConfig()
    rng = rng or np.random.default_rng(0)
    d = df_race.reset_index(drop=True)
    n = len(d)
    dist = float(_col(d, "dist_furlongs").median()) if _col(d, "dist_furlongs").notna().any() else 8.0
    going = str(d["going_description"].iloc[0]) if "going_description" in d.columns else None
    track_name = str(d["track"].iloc[0]) if "track" in d.columns else None

    if abilities == "market":
        lbs = ability_lbs_market(d, cfg)
    elif abilities == "composite":
        lbs = ability_lbs_composite(d, cfg)
    else:  # a column name holding lbs
        lbs = ability_lbs_composite(d, AbilityConfig(rating_col=abilities, use_weight=False, speed_fig_col="__none__"))

    v_par = par_speed(dist, going)
    v_cp = cfg.cp_fraction_of_par * v_par * lbs_to_speed_factor(lbs, dist)

    style = _first_available(d, STYLE_COLS, 3.0).clip(1.0, 6.0).values
    early_effort = np.interp(style, [1, 2, 3, 4, 5, 6], [0.20, 0.25, 0.30, 0.40, 0.50, 0.55])
    field_factor = np.sqrt(max(n, 2) / 10.0)
    desired_gap = np.where(style >= 5.5, 0.0, (6.0 - style) * 2.5 * field_factor)

    xs = np.array(sorted(cfg.dv_surge_table)); dv_base = float(np.interp(dist, xs, [cfg.dv_surge_table[k] for k in xs]))
    dv = dv_base * (1.0 + 0.05 * (style - 3.0) / 3.0)

    late_move = _first_available(d, ["horse_career_late_move", "horse_late_move_3r"], 0.0).clip(-3, 3).values
    keen_rate = _first_available(d, ["horse_keen_rate"], 0.08).clip(0, 1).values
    age = _col(d, "horse_age").fillna(4).values
    w_prime = cfg.w_prime_base_m * (1.0 + 0.10 * late_move / 3.0) * (1.0 - 0.10 * keen_rate)
    w_prime = np.where(age <= 2, w_prime * 0.9, w_prime)

    xk = np.array(sorted(cfg.kick_table)); kick_base = float(np.interp(dist, xk, [cfg.kick_table[k] for k in xk]))
    aggression = _first_available(d, ["jockey_front_rate"], 0.5).clip(0, 1).values
    kick = kick_base * (1.0 + 0.15 * (aggression - 0.5)) + (3.0 - style) * 15.0

    trouble_rate = _first_available(d, ["horse_trouble_rate"], 0.10).clip(0, 1).values
    trouble_prone = np.clip(1.0 + (trouble_rate - 0.10) * 2.0, 0.5, 2.0)

    runs = _col(d, "career_runs").fillna(5).values
    form_sd = np.full(n, cfg.day_noise_lbs) * np.where(runs < 3, cfg.unexposed_noise_mult, 1.0)

    stall = _col(d, "stall").values
    missing = np.isnan(stall)
    if missing.any():
        free = [s for s in range(1, n + 1) if s not in set(stall[~missing].astype(int))]
        stall[missing] = rng.permutation(free)[: missing.sum()] if len(free) >= missing.sum() else rng.integers(1, n + 1, missing.sum())
    rail_seek = np.clip(0.5 + 0.02 * (n / 2 - stall), 0.2, 0.8)

    names = d["horse_name"].astype(str).tolist() if "horse_name" in d.columns else [f"runner_{i+1}" for i in range(n)]
    return FieldParams(names=names, ability_lbs=lbs, v_cp=v_cp, dv_surge=dv, w_prime=w_prime, style=style,
                       early_effort=early_effort, desired_gap_m=desired_gap, kick_dist_m=kick,
                       aggression=aggression, rail_seek=rail_seek, keenness=keen_rate, trouble_prone=trouble_prone,
                       form_sd_lbs=form_sd, stall=stall.astype(float), dist_furlongs=dist,
                       length_m=dist * FURLONG_METRES, going=going, track_name=track_name,
                       meta={"v_par": v_par, "abilities": abilities})
