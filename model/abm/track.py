"""
Track environment for the race ABM.

The simulator is "1.5-D": a single along-the-rail coordinate plus a lane
index. The environment supplies, per position along the track:

    * curvature (1/R) — zero on straights, 1/R on bends. Bends cost speed
      (a small multiplicative loss) and cost *ground* to horses racing wide
      (an outer lane covers more metres for the same rail progress).
    * going — a speed factor on sustainable speed and an energy factor on
      anaerobic cost (softer ground drains the tank faster).

Geometry is a coarse per-course table (straight-course maximum trip, home
straight length, representative bend radius). It is deliberately simple
and easy to override; a proper course-geometry file is the obvious
Stage-3 upgrade if the ABM earns its keep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.perf_figures import FURLONG_METRES

# going description -> multiplicative factor on sustainable speed (good = 1)
GOING_SPEED_FACTOR = {
    "hard": 1.012, "firm": 1.008, "good to firm": 1.004, "good": 1.000,
    "good to soft": 0.988, "soft": 0.975, "heavy": 0.955,
    "good to yielding": 0.994, "yielding": 0.988, "yielding to soft": 0.980,
    "standard": 1.000, "standard to fast": 1.003, "fast": 1.006,
    "standard to slow": 0.994, "slow": 0.988,
}

# course (lower-case) -> (max straight-course trip in furlongs, home straight metres, bend radius metres)
TRACK_GEOMETRY = {
    "ascot": (8, 500, 200), "newmarket": (10, 800, 250), "newmarket (july)": (8, 800, 250),
    "york": (7, 900, 250), "doncaster": (8, 900, 220), "newbury": (8, 900, 220),
    "haydock": (6, 800, 200), "goodwood": (6, 700, 150), "chester": (0, 230, 100),
    "epsom": (5, 700, 180), "sandown": (5, 800, 200), "kempton": (0, 500, 180),
    "lingfield": (7, 400, 170), "wolverhampton": (0, 380, 140), "southwell": (0, 500, 150),
    "newcastle": (8, 800, 200), "chelmsford": (0, 400, 160), "chelmsford city": (0, 400, 160),
    "windsor": (6, 800, 170), "leicester": (7, 800, 200), "thirsk": (6, 700, 190),
    "pontefract": (0, 400, 160), "beverley": (5, 500, 170), "ripon": (6, 600, 180),
    "redcar": (8, 900, 220), "catterick": (5, 500, 160), "musselburgh": (5, 500, 160),
    "hamilton": (6, 800, 170), "ayr": (6, 800, 220), "brighton": (0, 600, 170),
    "bath": (0, 600, 170), "yarmouth": (8, 900, 200), "salisbury": (0, 800, 200),
    "nottingham": (6, 600, 180), "carlisle": (6, 500, 160), "ffos las": (0, 500, 200),
    "chepstow": (8, 900, 200), "warwick": (5, 500, 160), "curragh": (8, 800, 250),
    "leopardstown": (0, 500, 200), "dundalk": (0, 400, 160), "naas": (6, 500, 190),
    "gowran park": (0, 500, 180), "cork": (7, 600, 200), "galway": (0, 300, 140),
    "tipperary": (5, 500, 170), "fairyhouse": (0, 500, 200), "navan": (0, 500, 190),
    "killarney": (0, 300, 140), "bellewstown": (0, 300, 130), "limerick": (0, 500, 180),
    "sligo": (0, 300, 120), "ballinrobe": (0, 300, 130), "roscommon": (0, 400, 150),
    "listowel": (0, 300, 140), "downpatrick": (0, 300, 120), "down royal": (0, 500, 200),
    "wexford": (0, 300, 140), "clonmel": (0, 400, 150), "thurles": (0, 400, 150),
    "punchestown": (0, 500, 200), "laytown": (7, 1400, 999),
}
DEFAULT_GEOMETRY = (6, 600, 180)


def going_speed_factor(desc) -> float:
    if desc is None or (isinstance(desc, float) and np.isnan(desc)):
        return 1.0
    s = str(desc).lower().strip()
    best, best_len = 1.0, 0
    for k, v in GOING_SPEED_FACTOR.items():
        if k in s and len(k) > best_len:
            best, best_len = v, len(k)
    return best


def going_energy_factor(desc) -> float:
    """Softer ground costs more anaerobic energy per metre above cruise."""
    return 1.0 + 2.5 * (1.0 - going_speed_factor(desc))


def lookup_geometry(track_name) -> tuple:
    if not track_name:
        return DEFAULT_GEOMETRY
    s = str(track_name).lower().strip()
    if s in TRACK_GEOMETRY:
        return TRACK_GEOMETRY[s]
    for k, v in TRACK_GEOMETRY.items():
        if s.startswith(k) or k.startswith(s):
            return v
    return DEFAULT_GEOMETRY


@dataclass
class Track:
    length_m: float
    home_straight_m: float
    bend_radius_m: float
    is_straight: bool
    going_speed: float = 1.0
    going_energy: float = 1.0
    lane_width_m: float = 1.0
    run_to_bend_m: float = 250.0
    name: str = ""

    @classmethod
    def from_race(cls, dist_furlongs: float, going_description=None, track_name=None,
                  lane_width_m: float = 1.0) -> "Track":
        straight_max, home, radius = lookup_geometry(track_name)
        length = float(dist_furlongs) * FURLONG_METRES
        is_straight = dist_furlongs <= straight_max
        return cls(length_m=length, home_straight_m=float(min(home, length)), bend_radius_m=float(radius),
                   is_straight=bool(is_straight), going_speed=going_speed_factor(going_description),
                   going_energy=going_energy_factor(going_description), lane_width_m=lane_width_m,
                   name=str(track_name or ""))

    @property
    def bend_start_m(self) -> float:
        if self.is_straight:
            return self.length_m
        return float(min(self.run_to_bend_m, max(0.0, self.length_m - self.home_straight_m)))

    @property
    def bend_end_m(self) -> float:
        if self.is_straight:
            return self.length_m
        return float(max(self.length_m - self.home_straight_m, self.bend_start_m))

    @property
    def bend_fraction(self) -> float:
        return (self.bend_end_m - self.bend_start_m) / self.length_m

    def curvature_at(self, x: np.ndarray) -> np.ndarray:
        """Curvature 1/R at rail positions x (any shape)."""
        if self.is_straight:
            return np.zeros_like(np.asarray(x, dtype=float))
        x = np.asarray(x, dtype=float)
        return np.where((x >= self.bend_start_m) & (x < self.bend_end_m), 1.0 / self.bend_radius_m, 0.0)
