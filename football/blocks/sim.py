"""The match played out from each side's venue record, against the opening price. ``sm_``.

racing² football research, 24 Nov 2026 (the owner: game simulation). A scoreline simulation in closed form: the
opening price's implied goal rates are the prior, each side's own scoring and conceding at this venue the evidence.

- The home rate: (goals the home side scored in its last 12 home matches + goals the away side conceded in its last
  12 away matches + 10 x the opening price's home rate) / (those matches + 10); the away rate the same way round.
  A gamma-Poisson update with the market as a prior worth 10 matches (a set constant, not fitted).
- The score matrix of two independent Poisson counts (0 to 10 goals each) gives the simulated chances.

``sm_h`` / ``sm_a``: the simulated home / away chance less the opening one, in log odds. ``sm_d``: the simulated draw
chance less the opening one. ``sm_total``: the simulated goal total less the opening price's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.blocks._util import per_match, sides, split_record

FEATURES = ["sm_h", "sm_a", "sm_d", "sm_total"]
WINDOW, PRIOR, MAX_GOALS = 12, 10.0, 10


def chances(lh: np.ndarray, la: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(home, draw, away) chances of two independent Poisson scores, rows of rates."""
    from scipy.stats import poisson
    g = np.arange(MAX_GOALS + 1)
    ph = poisson.pmf(g[None, :], np.asarray(lh, float)[:, None])
    pa = poisson.pmf(g[None, :], np.asarray(la, float)[:, None])
    cum_a = np.cumsum(pa, axis=1)
    home = (ph[:, 1:] * cum_a[:, :-1]).sum(1)
    draw = (ph * pa).sum(1)
    tot = ph.sum(1) * pa.sum(1)
    return home / tot, draw / tot, (tot - home - draw) / tot


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def build(m: pd.DataFrame) -> pd.DataFrame:
    s = sides(m)
    venue = s["venue"].to_numpy()
    gf, ga = split_record(s, "gf", venue, ("H", "A"), WINDOW), split_record(s, "ga", venue, ("H", "A"), WINDOW)
    p = per_match(s, {"gf_home": gf["H"][0], "n_home": gf["H"][1], "ga_home": ga["H"][0],
                      "gf_away": gf["A"][0], "n_away": gf["A"][1], "ga_away": ga["A"][0]})
    d = m[["match_id", "molam_h", "molam_a", "qo_h", "qo_a"]].merge(p, on="match_id", how="left")
    qo_d = 1 - d["qo_h"] - d["qo_a"]
    n = d["n_home_h"].fillna(0) + d["n_away_a"].fillna(0)
    lh = (d["gf_home_h"].fillna(0) + d["ga_away_a"].fillna(0) + PRIOR * d["molam_h"]) / (n + PRIOR)
    la = (d["gf_away_a"].fillna(0) + d["ga_home_h"].fillna(0) + PRIOR * d["molam_a"]) / (n + PRIOR)
    ok = (lh.notna() & la.notna() & d["qo_h"].notna() & d["qo_a"].notna() & (n > 0)).to_numpy()
    ph, pd_, pa = (np.full(len(d), np.nan) for _ in range(3))
    if ok.any():
        ph[ok], pd_[ok], pa[ok] = chances(lh[ok].to_numpy(), la[ok].to_numpy())
    out = d[["match_id"]].copy()
    out["sm_h"] = _logit(ph) - _logit(d["qo_h"])
    out["sm_a"] = _logit(pa) - _logit(d["qo_a"])
    out["sm_d"] = pd_ - qo_d.to_numpy()
    out["sm_total"] = np.where(ok, lh + la - (d["molam_h"] + d["molam_a"]), np.nan)
    out.loc[~ok, ["sm_h", "sm_a", "sm_d"]] = np.nan
    return out
