"""
Reduced physiological core for a horse-jockey agent.

A deliberately low-parameter cousin of Mercier & Aftalion's (2020)
optimal-control model, chosen because Ashcroft has no GPS sectionals to
identify the full model. It keeps the two things that matter for race
shape:

    * a sustainable ("critical") speed v_cp the horse can hold, and
    * a finite anaerobic reserve W' (in "surge-metres") that drains when
      the horse runs above v_cp and refills slowly below it
      (Skiba-style W'-balance).

When W' runs out the horse cannot surge and its sustainable speed sags
(the "weakened" in a race comment). Speed follows the target with a
first-order lag (Aftalion's tau), faster when accelerating than when
easing.

Units: speed m/s, energy surge-metres (m/s * s), time s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EnergyConfig:
    cost_exponent: float = 1.0      # cost rate ∝ (v - v_cp)^exp
    recovery_rate: float = 0.25     # refill rate below v_cp (per unit deficit, scaled by headroom)
    fatigue_threshold: float = 0.25 # W' fraction below which sustainable speed starts to sag
    fatigue_drop: float = 0.08      # max fractional sag when W' is empty
    surge_fade_frac: float = 0.10   # W' fraction below which the surge capacity fades to zero
    tau_accel: float = 1.5          # s, first-order lag when speeding up
    tau_decel: float = 2.5          # s, when slowing
    v_start: float = 4.0            # m/s leaving the stalls
    keen_cost_multiplier: float = 1.6  # a keen horse burns W' faster while restrained


def speed_ceiling(v_cp: np.ndarray, w_frac: np.ndarray, cfg: EnergyConfig) -> np.ndarray:
    """Sustainable speed after fatigue sag."""
    depletion = np.clip((cfg.fatigue_threshold - w_frac) / cfg.fatigue_threshold, 0.0, 1.0)
    return v_cp * (1.0 - cfg.fatigue_drop * depletion**2)


def surge_available(dv: np.ndarray, w_frac: np.ndarray, cfg: EnergyConfig) -> np.ndarray:
    """Surge headroom above the ceiling; fades out as W' empties."""
    return dv * np.clip(w_frac / cfg.surge_fade_frac, 0.0, 1.0)


def step_speed(v: np.ndarray, v_target: np.ndarray, dt: float, cfg: EnergyConfig) -> np.ndarray:
    tau = np.where(v_target > v, cfg.tau_accel, cfg.tau_decel)
    return v + (v_target - v) * (1.0 - np.exp(-dt / tau))


def step_energy(w: np.ndarray, v: np.ndarray, v_cp: np.ndarray, w0: np.ndarray, dt: float,
                cfg: EnergyConfig, cost_mult=1.0) -> np.ndarray:
    """W'-balance update. cost_mult scales the drain (drafting < 1, keen / soft ground > 1)."""
    above = np.clip(v - v_cp, 0.0, None)
    below = np.clip(v_cp - v, 0.0, None)
    cost = cost_mult * above**cfg.cost_exponent * dt
    headroom = np.clip(1.0 - w / np.where(w0 > 0, w0, 1.0), 0.0, 1.0)
    recov = cfg.recovery_rate * below * headroom * dt
    return np.clip(w - cost + recov, 0.0, w0)
