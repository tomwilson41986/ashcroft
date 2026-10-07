"""Team ratings: a Dixon-Coles goals model with time decay, pooled across a country's divisions.

    log lambda_home = mu + home + att[h] - def[a]          log lambda_away = mu + att[a] - def[h]
    att[i] = att_div[div(i)] + u_att[i]                    def[i] = def_div[div(i)] + u_def[i]

with the Dixon-Coles correction tau(rho) on the 0-0, 1-0, 0-1 and 1-1 scores. One fit a country (England's five
divisions together, Scotland's four ...): a promoted side carries its rating up, and the division offsets are
learned from the sides that move. The team terms are ridge-penalised towards their division's mean (a side with few
matches is priced as a typical side of its division); each match is weighted by exp(-xi * days ago).

The fitted target can mix goals with shots on target (``sot_mix``): ``(1 - a) * goals + a * c * sot`` where ``c``
is the pool's goals per shot on target. Shots on target are a steadier measure of a side's chances than goals; the
walk-forward decides whether they help (reports/football_model.md).

The score matrix then prices every market the trader reads: 1X2, over/under 2.5 and the Asian handicap lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

MAX_GOALS = 10


@dataclass
class Ratings:
    pool: str
    fitted_to: str                      # the last date of the data (exclusive bound of the fit)
    mu: float
    home: float
    rho: float
    divisions: list[str]
    att_div: list[float]
    def_div: list[float]
    teams: dict[str, dict] = field(default_factory=dict)   # team_id -> {"att", "def", "div", "n", "name"}

    def strength(self, team: str, division: str) -> tuple[float, float]:
        """(attack, defence) of a side; a side the fit has not seen is its match division's typical side."""
        t = self.teams.get(team)
        if t is not None:
            return t["att"], t["def"]
        k = self.divisions.index(division) if division in self.divisions else len(self.divisions) - 1
        return self.att_div[k], self.def_div[k]

    def rates(self, home: str, away: str, division: str) -> tuple[float, float]:
        ah, dh = self.strength(home, division)
        aa, da = self.strength(away, division)
        return float(np.exp(self.mu + self.home + ah - da)), float(np.exp(self.mu + aa - dh))

    def to_dict(self) -> dict:
        return {"pool": self.pool, "fitted_to": self.fitted_to, "mu": self.mu, "home": self.home, "rho": self.rho,
                "divisions": self.divisions, "att_div": self.att_div, "def_div": self.def_div, "teams": self.teams}

    @classmethod
    def from_dict(cls, d: dict) -> "Ratings":
        return cls(**d)


def tau(x, y, lh, la, rho):
    """The Dixon-Coles low-score correction factor (vectorised)."""
    t = np.ones(np.broadcast(x, y, lh, la).shape)
    t = np.where((x == 0) & (y == 0), 1 - lh * la * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lh * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + la * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


def fit(home_idx, away_idx, div_idx, y_home, y_away, weights, n_teams: int, n_div: int, l2: float = 3.0,
        x0: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Weighted Poisson maximum likelihood with a ridge on the team terms. Parameters, in order: mu, home,
    att_div[1:], def_div[1:] (the first division is the reference), u_att[n_teams], u_def[n_teams]. Returns the
    parameter vector and the objective."""
    nd = n_div - 1
    w = np.asarray(weights, float)
    yh, ya = np.asarray(y_home, float), np.asarray(y_away, float)
    h, a = np.asarray(home_idx), np.asarray(away_idx)

    def unpack(p):
        mu, hm = p[0], p[1]
        ad = np.concatenate([[0.0], p[2:2 + nd]])
        dd = np.concatenate([[0.0], p[2 + nd:2 + 2 * nd]])
        ua = p[2 + 2 * nd:2 + 2 * nd + n_teams]
        ud = p[2 + 2 * nd + n_teams:]
        return mu, hm, ad, dd, ua, ud

    def f(p):
        mu, hm, ad, dd, ua, ud = unpack(p)
        att = ad[div_idx] + ua
        dfn = dd[div_idx] + ud
        eh = mu + hm + att[h] - dfn[a]
        ea = mu + att[a] - dfn[h]
        lh, la = np.exp(eh), np.exp(ea)
        nll = np.sum(w * (lh - yh * eh)) + np.sum(w * (la - ya * ea))
        nll += 0.5 * l2 * (ua @ ua + ud @ ud)
        rh, ra = w * (lh - yh), w * (la - ya)           # d nll / d eta
        g = np.zeros_like(p)
        g[0] = rh.sum() + ra.sum()
        g[1] = rh.sum()
        g_att = np.bincount(h, rh, n_teams) + np.bincount(a, ra, n_teams)
        g_def = -np.bincount(a, rh, n_teams) - np.bincount(h, ra, n_teams)
        if nd:
            g[2:2 + nd] = np.bincount(div_idx, g_att, n_div)[1:]
            g[2 + nd:2 + 2 * nd] = np.bincount(div_idx, g_def, n_div)[1:]
        g[2 + 2 * nd:2 + 2 * nd + n_teams] = g_att + l2 * ua
        g[2 + 2 * nd + n_teams:] = g_def + l2 * ud
        return nll, g

    n_par = 2 + 2 * nd + 2 * n_teams
    if x0 is None or len(x0) != n_par:
        x0 = np.zeros(n_par)
        x0[0] = np.log(max(1e-3, np.average(ya, weights=w)))
        x0[1] = np.log(max(1e-3, np.average(yh, weights=w))) - x0[0]
    r = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return r.x, float(r.fun)


def fit_rho(lh, la, x, y, w) -> float:
    """Dixon-Coles rho by one-dimensional maximum likelihood, the rates held (the usual two-step fit)."""
    low = (x <= 1) & (y <= 1)
    if not low.any():
        return 0.0
    lh, la, x, y, w = lh[low], la[low], x[low], y[low], w[low]

    def nll(rho):
        t = tau(x, y, lh, la, rho)
        return -np.sum(w * np.log(np.clip(t, 1e-9, None)))
    lo = max(-0.2, -1 / max(1e-9, (lh * la).max()) + 1e-6)
    return float(minimize_scalar(nll, bounds=(lo, 0.2), method="bounded").x)


def fit_pool(df, pool: str, as_of: str, xi: float = 0.0023, l2: float = 3.0, window_days: int = 1100,
             sot_mix: float = 0.0, x0=None) -> tuple[Ratings, np.ndarray] | tuple[None, None]:
    """Ratings of one pool from its finished matches before ``as_of`` (a 'YYYY-MM-DD' bound, exclusive). ``df``
    holds the pool's canonical matches (football.data gold columns)."""
    import pandas as pd
    d = df[(df.match_date < as_of) & (df.status == "FINISHED") & df.ft_home.notna() & df.ft_away.notna()]
    start = (pd.Timestamp(as_of) - pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
    d = d[d.match_date >= start]
    if len(d) < 50:
        return None, None
    age = (pd.Timestamp(as_of) - pd.to_datetime(d.match_date)).dt.days.to_numpy()
    w = np.exp(-xi * age)
    # divisions, best tier first; each team's division is the one of its latest match
    divs = sorted(d.competition_id.unique(), key=lambda c: (int(c[-1]) if c[-1].isdigit() else 9, c))
    long = pd.concat([d[["match_date", "home_team_id", "competition_id"]].rename(columns={"home_team_id": "t"}),
                      d[["match_date", "away_team_id", "competition_id"]].rename(columns={"away_team_id": "t"})])
    last = long.sort_values("match_date").drop_duplicates("t", keep="last").set_index("t")["competition_id"]
    teams = list(last.index)
    tix = {t: i for i, t in enumerate(teams)}
    dix = {c: i for i, c in enumerate(divs)}
    div_idx = np.array([dix[last[t]] for t in teams])
    h = d.home_team_id.map(tix).to_numpy()
    a = d.away_team_id.map(tix).to_numpy()
    yh, ya = d.ft_home.to_numpy(float), d.ft_away.to_numpy(float)
    if sot_mix > 0 and {"sot_home", "sot_away"} <= set(d.columns):
        sh, sa = d.sot_home.to_numpy(float), d.sot_away.to_numpy(float)
        ok = np.isfinite(sh) & np.isfinite(sa)
        if ok.mean() > 0.5:
            conv = (yh[ok].sum() + ya[ok].sum()) / max(1.0, sh[ok].sum() + sa[ok].sum())
            yh = np.where(ok, (1 - sot_mix) * yh + sot_mix * conv * np.nan_to_num(sh), yh)
            ya = np.where(ok, (1 - sot_mix) * ya + sot_mix * conv * np.nan_to_num(sa), ya)
    p, _ = fit(h, a, div_idx, yh, ya, w, len(teams), len(divs), l2=l2, x0=x0)
    nd = len(divs) - 1
    mu, hm = p[0], p[1]
    ad = np.concatenate([[0.0], p[2:2 + nd]])
    dd = np.concatenate([[0.0], p[2 + nd:2 + 2 * nd]])
    ua = p[2 + 2 * nd:2 + 2 * nd + len(teams)]
    ud = p[2 + 2 * nd + len(teams):]
    att = ad[div_idx] + ua
    dfn = dd[div_idx] + ud
    lh = np.exp(mu + hm + att[h] - dfn[a])
    la = np.exp(mu + att[a] - dfn[h])
    rho = fit_rho(lh, la, d.ft_home.to_numpy(int), d.ft_away.to_numpy(int), w)
    n = np.bincount(np.concatenate([h, a]), minlength=len(teams))
    names = pd.concat([d[["home_team_id", "home_team"]].set_axis(["t", "name"], axis=1),
                       d[["away_team_id", "away_team"]].set_axis(["t", "name"], axis=1)]
                      ).drop_duplicates("t", keep="last").set_index("t")["name"]
    r = Ratings(pool=pool, fitted_to=as_of, mu=float(mu), home=float(hm), rho=rho, divisions=divs,
                att_div=[float(x) for x in ad], def_div=[float(x) for x in dd],
                teams={t: {"att": float(att[i]), "def": float(dfn[i]), "div": last[t], "n": int(n[i]),
                           "name": str(names.get(t, t))}
                       for t, i in tix.items()})
    return r, p


# --------------------------------------------------------------------------------------------------------------------
# Prices from the score matrix
# --------------------------------------------------------------------------------------------------------------------

def score_matrix(lh: float, la: float, rho: float, n: int = MAX_GOALS) -> np.ndarray:
    g = np.arange(n + 1)
    m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    x, y = np.meshgrid(g, g, indexing="ij")
    m = m * tau(x, y, lh, la, rho)
    m = np.clip(m, 0, None)
    return m / m.sum()


def markets(m: np.ndarray) -> dict:
    """1X2 and over/under 2.5 chances from a score matrix (home goals on rows)."""
    return {"p_h": float(np.tril(m, -1).sum()), "p_d": float(np.trace(m)), "p_a": float(np.triu(m, 1).sum()),
            "p_o25": float(m[np.add.outer(np.arange(m.shape[0]), np.arange(m.shape[1])) >= 3].sum()),
            "p_btts": float(m[1:, 1:].sum())}


def diff_dist(m: np.ndarray) -> dict[int, float]:
    """P(home goals - away goals = d)."""
    n = m.shape[0]
    out: dict[int, float] = {}
    for i in range(n):
        for j in range(n):
            out[i - j] = out.get(i - j, 0.0) + m[i, j]
    return out


def ah_return(diff: dict[int, float], line: float, price: float, side: str = "H", com: float = 0.0) -> float:
    """Expected return per unit staked on an Asian handicap bet: ``line`` is the home side's handicap (-0.75: home
    gives three quarters of a goal); ``side`` H backs home at ``line``, A backs away at ``-line``. A quarter line is
    two half stakes on the lines either side; a whole line can push."""
    q = round(line * 4)
    halves = [line - 0.25, line + 0.25] if q % 2 else [line]
    ev = 0.0
    for l in halves:
        e = 0.0
        for d, p in diff.items():
            adj = (d + l) if side == "H" else (-d - l)
            if adj > 0:
                e += p * (price - 1) * (1 - com)
            elif adj < 0:
                e -= p
        ev += e / len(halves)
    return ev


def ah_fair(diff: dict[int, float], line: float, side: str = "H") -> float:
    """The price at which the bet's expected return is nil (no commission)."""
    lo, hi = 1.01, 50.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if ah_return(diff, line, mid, side) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2
