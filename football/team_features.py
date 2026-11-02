"""Team-level features for the football model: strength, strength of schedule and performance added, all lag-safe.

The owner's ask (2 Nov 2026): engineer team strength, strength of schedule and "wins added / performance above
average" measures (reports/football_quant_approach.md s.6.2 F1, F5, F6). Each is judged by whether it adds to the
market (``football.feature_eval``), not by how well it predicts results alone: our team model loses to the open.

Per match (finished, from the gold table and the Dixon-Coles walk-forward prices):

- the model's pre-match expectation: ``lam_h``/``lam_a`` and ``p_h``/``p_d``/``p_a`` from ``football.model.walk_forward``
  (each match priced from the ratings fitted on the matches before its week);
- the market's: the de-vigged opening and closing 1X2 and over/under 2.5 of the gold table, and the two goal rates
  each implies (``implied_rates``: the independent-Poisson pair whose home, away and over-2.5 chances are nearest);
- the result: points, goal difference, shots-on-target difference.

Strength (``market_ratings``): each team's market-implied attack and defence, a weighted least-squares fit of the
log implied closing goal rates (log rate = mu + home + attack - opponent's defence), decayed by exp(-xi days) over the
``window_days`` before the fit, ridge-shrunk to zero, refitted weekly a country pool; ``r = attack + defence`` (in log
goal-rate units: above zero is stronger than the pool's average side). Each match carries both sides' ratings from the
fit before its week.

Per side and match (``team_rows``): points above the model's expectation (``pae_dc`` = points - (3 p_win + p_draw)),
above the market's (``pae_mkt``, from the close), goal difference above each (``gae_dc``, ``gae_mkt``), the
opponent's pre-match rating (``opp_r``, the schedule's strength) and its own (``own_r``).

Windows (``windows``): over each side's earlier matches only (the previous match is always on an earlier day): the
mean of the last 5 and the last 10, an exponentially weighted mean (half-life 8 matches) and the season to date, of
points, goal difference, shots-on-target difference, ``pae_dc``, ``pae_mkt``, ``gae_dc``, ``gae_mkt`` and ``opp_r``
(strength of schedule); and the change in the side's market rating over its last 5 matches.

Match features (``match_features``): home minus away of every window, both sides' ratings, and the disagreement
between our model and the market at the open (``dis_open`` = the model's log supremacy - the open's).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

XI = 0.0023
WINDOW_DAYS = 1100
RIDGE = 2.0
MEASURES = ["pts", "gd", "sotd", "pae_dc", "pae_mkt", "gae_dc", "gae_mkt", "opp_r"]
WINDOWS = {"l5": 5, "l10": 10}
EWM_HALFLIFE = 8


# ---------------------------------------------------------------------------
# the market's goal rates
# ---------------------------------------------------------------------------

def _poisson_grid(lo: float = 0.15, hi: float = 4.5, step: float = 0.02, max_goals: int = 12):
    lam = np.arange(lo, hi + 1e-9, step)
    k = np.arange(max_goals + 1)
    from scipy.stats import poisson
    pmf = poisson.pmf(k[None, :], lam[:, None])                       # (L, K)
    lh, la = np.meshgrid(lam, lam, indexing="ij")
    joint = pmf[:, None, :, None] * pmf[None, :, None, :]             # (L, L, K, K)
    diff = k[:, None] - k[None, :]
    tot = k[:, None] + k[None, :]
    ph = (joint * (diff > 0)).sum(axis=(2, 3))
    pa = (joint * (diff < 0)).sum(axis=(2, 3))
    po = (joint * (tot >= 3)).sum(axis=(2, 3))
    return lh.ravel(), la.ravel(), ph.ravel(), pa.ravel(), po.ravel()


_GRID = None


def implied_rates(ph, pa, po=None) -> tuple[np.ndarray, np.ndarray]:
    """The independent-Poisson goal rates (home, away) whose home-win, away-win and (where given) over-2.5 chances
    are nearest the market's de-vigged ones (a nearest-neighbour search over a 0.02 grid of rates)."""
    from scipy.spatial import cKDTree
    global _GRID
    if _GRID is None:
        _GRID = _poisson_grid()
    lh, la, gh, ga, go = _GRID
    ph, pa = np.asarray(ph, float), np.asarray(pa, float)
    po = np.full_like(ph, np.nan) if po is None else np.asarray(po, float)
    out_h, out_a = np.full(len(ph), np.nan), np.full(len(ph), np.nan)
    ok3 = np.isfinite(ph) & np.isfinite(pa) & np.isfinite(po)
    ok2 = np.isfinite(ph) & np.isfinite(pa) & ~ok3
    if ok3.any():
        _, i = cKDTree(np.c_[gh, ga, go]).query(np.c_[ph[ok3], pa[ok3], po[ok3]])
        out_h[ok3], out_a[ok3] = lh[i], la[i]
    if ok2.any():
        _, i = cKDTree(np.c_[gh, ga]).query(np.c_[ph[ok2], pa[ok2]])
        out_h[ok2], out_a[ok2] = lh[i], la[i]
    return out_h, out_a


def devig(*prices) -> list[np.ndarray]:
    inv = [1.0 / np.asarray(p, float) for p in prices]
    s = np.sum(inv, axis=0)
    return [x / s for x in inv]


# ---------------------------------------------------------------------------
# the per-match table
# ---------------------------------------------------------------------------

def match_table(gold: pd.DataFrame, dc: pd.DataFrame) -> pd.DataFrame:
    """Finished matches with a score, the model's pre-match prices and the market's open and close, de-vigged, with
    the goal rates each implies."""
    g = gold[(gold["status"] == "FINISHED") & gold["ft_home"].notna() & gold["ft_away"].notna()].copy()
    cols = ["match_id", "lam_h", "lam_a", "p_h", "p_d", "p_a", "p_o25"]
    g = g.merge(dc[[c for c in cols if c in dc.columns]].drop_duplicates("match_id"), on="match_id", how="left")
    for ph in ("open", "close"):
        h, d, a = devig(g[f"{ph}_h"], g[f"{ph}_d"], g[f"{ph}_a"])
        o, _u = devig(g[f"{ph}_o25"], g[f"{ph}_u25"])
        g[f"q{ph[0]}_h"], g[f"q{ph[0]}_d"], g[f"q{ph[0]}_a"], g[f"q{ph[0]}_o"] = h, d, a, o
        g[f"m{ph[0]}lam_h"], g[f"m{ph[0]}lam_a"] = implied_rates(h, a, o)
    g["date"] = pd.to_datetime(g["match_date"])
    return g.sort_values(["date", "match_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# market-implied strength
# ---------------------------------------------------------------------------

def fit_market_ratings(d: pd.DataFrame, as_of: pd.Timestamp, xi: float = XI, window_days: int = WINDOW_DAYS,
                       ridge: float = RIDGE) -> dict | None:
    """One pool's market-implied attack and defence from the closing goal rates of its matches before ``as_of``."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr
    w = d[(d["date"] < as_of) & (d["date"] >= as_of - pd.Timedelta(days=window_days))
          & d["mclam_h"].notna() & d["mclam_a"].notna()]
    if len(w) < 50:
        return None
    teams = pd.Index(pd.unique(pd.concat([w["home_team_id"], w["away_team_id"]])))
    n, t = len(w), len(teams)
    hi, ai = teams.get_indexer(w["home_team_id"]), teams.get_indexer(w["away_team_id"])
    wt = np.sqrt(np.exp(-xi * (as_of - w["date"]).dt.days.to_numpy(float)))
    # columns: mu, home, attack[t], defence[t]; rows: the home side's goals (0..n-1), then the away side's (n..2n-1)
    # log rate = mu + home (home rows only) + attack of the scoring side - defence of the other
    ih, ia = np.arange(n), np.arange(n, 2 * n)
    rows = np.concatenate([ih, ia, ih, ih, ia, ih, ia])
    cols = np.concatenate([np.zeros(n), np.zeros(n), np.ones(n), 2 + hi, 2 + ai, 2 + t + ai, 2 + t + hi])
    vals = np.concatenate([np.ones(5 * n), -np.ones(2 * n)])
    wrow = np.r_[wt, wt]
    X = coo_matrix((vals * wrow[rows], (rows, cols)), shape=(2 * n, 2 + 2 * t)).tocsr()
    y = np.log(np.r_[w["mclam_h"].to_numpy(float), w["mclam_a"].to_numpy(float)]) * np.r_[wt, wt]
    beta = lsqr(X, y, damp=np.sqrt(ridge / max(n, 1)))[0]
    att, dfn = beta[2:2 + t], beta[2 + t:]
    return {"att": dict(zip(teams, att)), "def": dict(zip(teams, dfn)), "mu": beta[0], "home": beta[1]}


def market_ratings(mt: pd.DataFrame, refit_days: int = 7, start: str | None = None) -> pd.DataFrame:
    """Each match's two sides' market-implied ratings from the pool's fit before the match's week (weekly refits
    from the pool's first match date in scope)."""
    out = []
    for pool, d in mt.groupby("country"):
        todo = d if start is None else d[d["date"] >= pd.Timestamp(start)]
        if todo.empty:
            continue
        dates = np.array(sorted(todo["date"].unique()))
        i = 0
        while i < len(dates):
            as_of = pd.Timestamp(dates[i])
            end = as_of + pd.Timedelta(days=refit_days)
            block = todo[(todo["date"] >= as_of) & (todo["date"] < end)]
            fit = fit_market_ratings(d, as_of)
            if fit is not None:
                for row in block.itertuples(index=False):
                    ah, dh = fit["att"].get(row.home_team_id), fit["def"].get(row.home_team_id)
                    aa, da = fit["att"].get(row.away_team_id), fit["def"].get(row.away_team_id)
                    out.append({"match_id": row.match_id, "att_h": ah, "def_h": dh, "att_a": aa, "def_a": da})
            i = int(np.searchsorted(dates, np.datetime64(end)))
    r = pd.DataFrame(out, columns=["match_id", "att_h", "def_h", "att_a", "def_a"])
    r["r_h"], r["r_a"] = r["att_h"] + r["def_h"], r["att_a"] + r["def_a"]
    return r


# ---------------------------------------------------------------------------
# per side, and the windows
# ---------------------------------------------------------------------------

def team_rows(mt: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Two rows a match, one a side, with the result, the expectations and the measures."""
    m = mt.merge(ratings, on="match_id", how="left")
    gd = m["ft_home"] - m["ft_away"]
    pts_h = np.select([gd > 0, gd == 0], [3.0, 1.0], 0.0)
    pts_a = np.select([gd < 0, gd == 0], [3.0, 1.0], 0.0)
    base = ["match_id", "date", "season", "country", "competition_id"]
    h = m[base].assign(team=m["home_team_id"], opp=m["away_team_id"], venue="H", pts=pts_h, gd=gd,
                       sotd=m["sot_home"] - m["sot_away"], xpts_dc=3 * m["p_h"] + m["p_d"],
                       xpts_mkt=3 * m["qc_h"] + m["qc_d"], xgd_dc=m["lam_h"] - m["lam_a"],
                       xgd_mkt=m["mclam_h"] - m["mclam_a"], own_r=m["r_h"], opp_r=m["r_a"])
    a = m[base].assign(team=m["away_team_id"], opp=m["home_team_id"], venue="A", pts=pts_a, gd=-gd,
                       sotd=m["sot_away"] - m["sot_home"], xpts_dc=3 * m["p_a"] + m["p_d"],
                       xpts_mkt=3 * m["qc_a"] + m["qc_d"], xgd_dc=m["lam_a"] - m["lam_h"],
                       xgd_mkt=m["mclam_a"] - m["mclam_h"], own_r=m["r_a"], opp_r=m["r_h"])
    t = pd.concat([h, a], ignore_index=True)
    t["pae_dc"], t["pae_mkt"] = t["pts"] - t["xpts_dc"], t["pts"] - t["xpts_mkt"]
    t["gae_dc"], t["gae_mkt"] = t["gd"] - t["xgd_dc"], t["gd"] - t["xgd_mkt"]
    return t.sort_values(["team", "date", "match_id"]).reset_index(drop=True)


def windows(t: pd.DataFrame) -> pd.DataFrame:
    """Each side's windows over its earlier matches only (shifted by one: a match never sees itself)."""
    g = t.groupby("team", sort=False)
    out = t[["match_id", "team", "venue", "date", "own_r"]].copy()
    for c in MEASURES:
        prev = g[c].shift(1)
        grp = prev.groupby(t["team"])
        for name, n in WINDOWS.items():
            out[f"{c}_{name}"] = grp.transform(lambda s, n=n: s.rolling(n, min_periods=max(2, n // 2)).mean())
        out[f"{c}_ewm"] = grp.transform(lambda s: s.ewm(halflife=EWM_HALFLIFE, min_periods=3).mean())
        sea = prev.groupby([t["team"], t["season"]])
        out[f"{c}_season"] = sea.transform(lambda s: s.expanding(min_periods=3).mean())
    out["n_prev"] = g.cumcount()
    # day-lagged: a side's matches on one day all take the windows of its first match that day, so nothing of the day
    # enters (league sides never play twice a day; the rule holds whatever the data says)
    wcols = [c for c in out.columns if c not in ("match_id", "team", "venue", "date", "own_r")]
    out[wcols] = out.groupby(["team", "date"])[wcols].transform("first")
    # the market's view of the side, moving: its pre-match rating now against five matches ago (no result enters)
    out["own_r_d5"] = t["own_r"] - g["own_r"].shift(5)
    return out


def match_features(mt: pd.DataFrame, ratings: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    """One row a match: home minus away of every window, both ratings, the model-market disagreement at the open."""
    wcols = [c for c in w.columns if c not in ("match_id", "team", "venue", "date")]
    hw = w[w["venue"] == "H"].set_index("match_id")[wcols]
    aw = w[w["venue"] == "A"].set_index("match_id")[wcols]
    m = mt.merge(ratings, on="match_id", how="left").set_index("match_id")
    f = pd.DataFrame(index=m.index)
    for c in wcols:
        f[f"d_{c}"] = hw[c].reindex(m.index) - aw[c].reindex(m.index)
    f["r_h"], f["r_a"] = m["r_h"], m["r_a"]
    f["n_prev_min"] = np.minimum(hw["n_prev"].reindex(m.index), aw["n_prev"].reindex(m.index))
    f["dis_open"] = np.log(m["lam_h"] / m["lam_a"]) - np.log(m["molam_h"] / m["molam_a"])
    return f.reset_index()


def build(gold: pd.DataFrame, dc: pd.DataFrame, start: str = "2012-07-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(the per-match table with its features, the per-side rows)."""
    mt = match_table(gold, dc)
    ratings = market_ratings(mt, start=start)
    t = team_rows(mt, ratings)
    w = windows(t)
    f = match_features(mt, ratings, w)
    out = mt.merge(f, on="match_id", how="left")
    log.info("team features: %d matches, %d side rows", len(out), len(t))
    return out, t
