"""Expected-goals features for the football model, from API-Football's team statistics (2022/23 on in the top
divisions; reports/football_xg_features.md for the evidence behind each construction and the results).

Per side and match (``side_rows``), from ``api_football/history_stats`` (xG, goals prevented, shots) and the events:

- ``npxg`` / ``npxga``: non-penalty xG for and against (xG less ``PEN_XG`` a penalty taken: the provider's xG counts
  penalties); ``npxgd`` their difference;
- shot quality: ``xg_per_shot`` and ``xga_per_shot`` (non-penalty), ``inside_share`` (shots inside the box),
  ``shots_d`` and ``sot_d`` (shots and shots on target, for minus against);
- luck terms: ``finishing`` (non-penalty goals minus npxg) and ``keeper`` (the side's goalkeeper ``goals_prevented``);
- against the market: ``npxgd_vs_mkt`` = npxgd less the close's expected goal difference (the implied rates of the
  de-vigged closing 1X2 and O/U 2.5) - the xG version of performance against the market;
- game state: ``trail_share``, the share of 90 minutes the side spent behind; ``npxg_adj`` / ``npxga_adj`` /
  ``npxgd_adj``, non-penalty xG projected onto a level score with eleven men (``state_factor``: score-state
  multipliers from Skripnikov et al. and the red-card rate change, time-weighted from the events).

Territory (the owner, 4 Nov 2026: "xThreat - other X metrics"; API-Football has no event locations, so no
expected threat: these are its proxies from the team statistics): ``poss`` (possession less 50, in points),
``pass_pct_d`` (pass completion, for minus against), ``corner_share``, ``box_share`` (the side's share of the
shots inside the box) and ``shot_share``.

Strength of schedule (the owner, 4 Nov 2026: "xG adjusted by strength of schedule"): ``npxg_sos`` = npxg less the
opponent's average npxg conceded over its previous ``SOS_N`` matches (before this one), ``npxga_sos`` = npxga less
the opponent's average npxg made, ``npxgd_sos`` their difference: what the side did against what that opponent
usually allows and makes.

Windows over each side's earlier matches (``football.team_features.windows``: last 5 and 10, EWM, season to date),
lagged to the collection day like every team feature; plus ``finish_shrunk``, finishing over the last 20 matches
shrunk towards zero by shot volume (``SHRINK_SHOTS`` pseudo-shots), and ``n_xg_prev`` (earlier matches with xG).
Short windows (``short_windows``): the last 2 and 3 matches and the last 5 and 10 weighted linearly by recency
(the latest match weighs n, the oldest 1) of ``SHORT_MEASURES``, lagged the same way.

xG ratings (``xg_ratings``): each side's attack and defence from a decayed ridge fit of the log non-penalty xG of its
matches before the collection day (the market ratings' fit with xG as the target): ``r_xg`` = attack + defence, and its
gap to the market-implied rating.

Match features (``match_features``): home minus away of every window, ``r_xg_diff`` and ``xg_gap_vs_market``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from football.team_features import XI, WINDOW_DAYS, collection_day, windows

log = logging.getLogger(__name__)

PEN_XG = 0.79            # Opta's flat penalty xG (theanalyst.com); StatsBomb 0.78, Wyscout 0.76; the provider is unnamed
SHRINK_SHOTS = 2000      # pseudo-shots: a team's 20-match finishing excess keeps about a tenth (persistence ~0.1)
KEEPER_SHRINK_SOT = 400  # pseudo-shots on target faced: keeper goals prevented shrunk hard (year-to-year r ~0.2)
FINISH_WINDOW, KEEPER_WINDOW = 20, 38
TERRITORY = ["poss", "pass_pct_d", "corner_share", "box_share", "shot_share"]
SOS = ["npxg_sos", "npxga_sos", "npxgd_sos"]
SOS_N = 10
XG_MEASURES = ["npxg", "npxga", "npxgd", "npxg_adj", "npxga_adj", "npxgd_adj", "xg_per_shot", "xga_per_shot",
               "inside_share", "shots_d", "sot_d", "finishing", "keeper", "npxgd_vs_mkt", "trail_share"] + TERRITORY + SOS
SHORT_MEASURES = ["npxgd", "npxgd_adj", "npxgd_sos", "npxgd_vs_mkt", "sot_d", "box_share", "poss"]
SHORT_EQUAL, SHORT_LINEAR = (2, 3), (5, 10)


#: projecting a side's attacking output onto a level score (Skripnikov et al., arXiv 2508.04008: trailing x0.70-0.90,
#: leading x1.10-1.75); our points inside those ranges, by margin. Play at two goals or more carries little signal
#: (ElHabr), so it gets the strongest correction
STATE_MULT = {"level": 1.0, "trail1": 0.85, "trail2": 0.75, "lead1": 1.2, "lead2": 1.5}
#: xG per minute a man down -65%, a man up +54% (databetweenthelines, via the research note)
RED_DOWN, RED_UP = -0.65, 0.54


def match_states(hm: pd.DataFrame, he: pd.DataFrame, full: int = 90) -> pd.DataFrame:
    """Per match, the home side's share of the first ``full`` minutes in each score state (level, ahead or behind by
    one, by two or more) and a man down or up, from the goal and red-card events (API-Football lists every goal, own
    goals too, under the side it counts for; a red card under the side that lost the player)."""
    ev = he[((he["type"] == "Goal") & ~he["detail"].astype(str).str.lower().str.contains("missed"))
            | ((he["type"] == "Card") & he["detail"].astype(str).str.lower().str.contains("red|second yellow"))]
    home = hm.set_index("apif_fixture_id")["apif_home_id"]
    by_fix = {k: g.sort_values("minute") for k, g in ev.groupby("apif_fixture_id")}
    rows = []
    for fid in hm["apif_fixture_id"]:
        hid = home.get(fid)
        acc = dict.fromkeys(("level", "lead1", "lead2", "trail1", "trail2", "down", "up"), 0.0)
        diff, men, t0 = 0, 0, 0                      # goal difference and men difference, home's view
        g = by_fix.get(fid)
        for e in ([] if g is None else g.itertuples(index=False)):
            m = min(int(e.minute) if e.minute == e.minute else 0, full)
            dt = m - t0
            acc[_state(diff)] += dt
            acc["down" if men < 0 else "up" if men > 0 else "level"] += 0 if men == 0 else dt
            t0 = m
            if e.type == "Goal":
                diff += 1 if e.apif_team_id == hid else -1
            else:
                men += -1 if e.apif_team_id == hid else 1
        dt = full - t0
        acc[_state(diff)] += dt
        if men:
            acc["down" if men < 0 else "up"] += dt
        rows.append({"apif_fixture_id": fid, **{f"h_{k}": v / full for k, v in acc.items()}})
    return pd.DataFrame(rows)


def _state(diff: int) -> str:
    return "level" if diff == 0 else ("lead1" if diff == 1 else "lead2" if diff > 1 else "trail1" if diff == -1
                                      else "trail2")


def state_factor(lead1, lead2, trail1, trail2, level, down, up) -> pd.Series:
    """What a side's xG becomes on a level score with eleven men: its score-state multiplier (time-weighted) over its
    red-card rate change (time-weighted), the latter floored at 0.3."""
    gs = (level * STATE_MULT["level"] + lead1 * STATE_MULT["lead1"] + lead2 * STATE_MULT["lead2"]
          + trail1 * STATE_MULT["trail1"] + trail2 * STATE_MULT["trail2"])
    rc = (1 + RED_DOWN * down + RED_UP * up).clip(lower=0.3)
    return gs / rc


def match_level(feats: pd.DataFrame, linked: pd.DataFrame, stats: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    """One row a match with both sides' statistics (home ``_h``, away ``_a``), joined to the team-feature table."""
    keep = ["apif_fixture_id", "xg", "goals_prevented", "shots", "shots_on", "shots_inside", "pens_scored",
            "pens_missed", "red_cards", "possession", "pass_pct", "corners"]
    st = stats[[c for c in keep if c in stats.columns] + ["side"]]
    h = st[st["side"] == "home"].drop(columns="side").add_suffix("_h").rename(columns={"apif_fixture_id_h": "apif_fixture_id"})
    a = st[st["side"] == "away"].drop(columns="side").add_suffix("_a").rename(columns={"apif_fixture_id_a": "apif_fixture_id"})
    m = linked[["apif_fixture_id", "match_id"]].merge(h, on="apif_fixture_id").merge(a, on="apif_fixture_id") \
        .merge(states, on="apif_fixture_id", how="left")
    cols = ["match_id", "date", "season", "country", "competition_id", "home_team_id", "away_team_id", "ft_home",
            "ft_away", "mclam_h", "mclam_a"]
    return feats[cols].merge(m, on="match_id")


def side_rows(m: pd.DataFrame) -> pd.DataFrame:
    """Two rows a match, one a side, with the xG measures."""
    parts = []
    for me, op, venue in (("h", "a", "H"), ("a", "h", "A")):
        pens_me = m[f"pens_scored_{me}"].fillna(0) + m[f"pens_missed_{me}"].fillna(0)
        pens_op = m[f"pens_scored_{op}"].fillna(0) + m[f"pens_missed_{op}"].fillna(0)
        goals = m["ft_home"] if me == "h" else m["ft_away"]
        against = m["ft_away"] if me == "h" else m["ft_home"]
        npxg = (m[f"xg_{me}"] - PEN_XG * pens_me).clip(lower=0)
        npxga = (m[f"xg_{op}"] - PEN_XG * pens_op).clip(lower=0)
        np_shots = (m[f"shots_{me}"] - pens_me).clip(lower=1)
        np_shots_a = (m[f"shots_{op}"] - pens_op).clip(lower=1)
        hs = {k: m[f"h_{k}"].fillna(1.0 if k == "level" else 0.0) for k in
              ("level", "lead1", "lead2", "trail1", "trail2", "down", "up")}
        if me == "h":
            f_me = state_factor(hs["lead1"], hs["lead2"], hs["trail1"], hs["trail2"], hs["level"], hs["down"], hs["up"])
            f_op = state_factor(hs["trail1"], hs["trail2"], hs["lead1"], hs["lead2"], hs["level"], hs["up"], hs["down"])
        else:
            f_me = state_factor(hs["trail1"], hs["trail2"], hs["lead1"], hs["lead2"], hs["level"], hs["up"], hs["down"])
            f_op = state_factor(hs["lead1"], hs["lead2"], hs["trail1"], hs["trail2"], hs["level"], hs["down"], hs["up"])
        trail = (hs["trail1"] + hs["trail2"]) if me == "h" else (hs["lead1"] + hs["lead2"])
        mkt_for = m["mclam_h"] if me == "h" else m["mclam_a"]
        mkt_against = m["mclam_a"] if me == "h" else m["mclam_h"]
        p = pd.DataFrame({
            "match_id": m["match_id"], "date": pd.to_datetime(m["date"]), "season": m["season"],
            "country": m["country"], "team": m["home_team_id"] if me == "h" else m["away_team_id"],
            "opp": m["away_team_id"] if me == "h" else m["home_team_id"], "venue": venue,
            "npxg": npxg, "npxga": npxga, "npxgd": npxg - npxga,
            "npxg_adj": npxg * f_me, "npxga_adj": npxga * f_op, "npxgd_adj": npxg * f_me - npxga * f_op,
            "xg_per_shot": npxg / np_shots, "xga_per_shot": npxga / np_shots_a,
            "inside_share": m[f"shots_inside_{me}"] / m[f"shots_{me}"].where(m[f"shots_{me}"] > 0),
            "shots_d": m[f"shots_{me}"] - m[f"shots_{op}"], "sot_d": m[f"shots_on_{me}"] - m[f"shots_on_{op}"],
            "finishing": (goals - m[f"pens_scored_{me}"].fillna(0)) - npxg,
            "np_goals_minus_xg": (goals - m[f"pens_scored_{me}"].fillna(0)) - npxg, "np_shots": np_shots,
            "keeper": m[f"goals_prevented_{me}"],
            "npxgd_vs_mkt": (npxg - npxga) - (mkt_for - mkt_against),
            "trail_share": trail, "sot_against": m[f"shots_on_{op}"],
            "poss": _col(m, f"possession_{me}") - 50, "pass_pct_d": _col(m, f"pass_pct_{me}") - _col(m, f"pass_pct_{op}"),
            "corner_share": _share(_col(m, f"corners_{me}"), _col(m, f"corners_{op}")),
            "box_share": _share(m[f"shots_inside_{me}"], m[f"shots_inside_{op}"]),
            "shot_share": _share(m[f"shots_{me}"], m[f"shots_{op}"]),
            "goals": goals, "against": against, "own_r": 0.0})
        p.loc[m[f"xg_{me}"].isna(), ["npxg", "npxga", "npxgd", "npxg_adj", "npxga_adj", "npxgd_adj", "xg_per_shot",
                                     "xga_per_shot", "finishing", "np_goals_minus_xg", "npxgd_vs_mkt"]] = np.nan
        parts.append(p)
    t = pd.concat(parts, ignore_index=True).sort_values(["team", "date", "match_id"]).reset_index(drop=True)
    return add_sos(t)


def _col(m: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(m[c], errors="coerce") if c in m.columns else pd.Series(np.nan, index=m.index)


def _share(a: pd.Series, b: pd.Series) -> pd.Series:
    tot = a + b
    return a / tot.where(tot > 0)


def add_sos(t: pd.DataFrame, n: int = SOS_N) -> pd.DataFrame:
    """npxg and npxga against what the opponent usually concedes and makes: its mean over its previous ``n`` matches
    with xG, strictly before this one (a side plays once a day, so the shifted window never holds this match)."""
    g = t.groupby("team", sort=False)
    prior = pd.DataFrame({"match_id": t["match_id"], "opp": t["team"]})
    for c in ("npxg", "npxga"):
        prior[f"opp_{c}"] = g[c].transform(lambda x: x.shift(1).rolling(n, min_periods=3).mean())
    t = t.merge(prior, on=["match_id", "opp"], how="left")
    t["npxg_sos"] = t["npxg"] - t["opp_npxga"]
    t["npxga_sos"] = t["npxga"] - t["opp_npxg"]
    t["npxgd_sos"] = t["npxg_sos"] - t["npxga_sos"]
    return t.drop(columns=["opp_npxg", "opp_npxga"]).sort_values(["team", "date", "match_id"]).reset_index(drop=True)


def short_windows(t: pd.DataFrame, measures: list[str], equal=SHORT_EQUAL, linear=SHORT_LINEAR,
                  key: str = "team") -> pd.DataFrame:
    """Each side's last 2 and 3 matches (mean) and last 5 and 10 weighted linearly by recency, over its earlier
    matches only, lagged to the collection day like ``windows``. ``t`` sorted by side and date. Missing values are
    skipped (the weights renormalised); a window needs half its matches present."""
    g = t.groupby(key, sort=False)
    out = pd.DataFrame(index=t.index)
    kmax = max(max(equal), max(linear))
    for c in measures:
        lags = np.column_stack([g[c].shift(k).to_numpy(float) for k in range(1, kmax + 1)])
        ok = np.isfinite(lags)
        x = np.where(ok, lags, 0.0)
        for n in equal:
            cnt = ok[:, :n].sum(1)
            out[f"{c}_l{n}"] = np.where(cnt >= max(2, n // 2), x[:, :n].sum(1) / np.maximum(cnt, 1), np.nan)
        for n in linear:
            w = np.arange(n, 0, -1, dtype=float)                  # the latest match weighs n
            cnt = ok[:, :n].sum(1)
            den = (ok[:, :n] * w).sum(1)
            out[f"{c}_lw{n}"] = np.where(cnt >= max(2, n // 2), (x[:, :n] * w).sum(1) / np.where(den > 0, den, 1),
                                         np.nan)
    cut = collection_day(t["date"]).to_numpy()
    return out.groupby([t[key].to_numpy(), cut]).transform("first")


def finishing_shrunk(t: pd.DataFrame) -> pd.Series:
    """Finishing over the side's last ``FINISH_WINDOW`` earlier matches: the sum of non-penalty goals minus npxg over
    the sum of non-penalty shots plus ``SHRINK_SHOTS`` pseudo-shots (a skill read only once the shots are many)."""
    g = t.groupby("team", sort=False)
    num = g["np_goals_minus_xg"].transform(lambda s: s.shift(1).rolling(FINISH_WINDOW, min_periods=3).sum())
    den = g["np_shots"].transform(lambda s: s.where(t.loc[s.index, "np_goals_minus_xg"].notna())
                                  .shift(1).rolling(FINISH_WINDOW, min_periods=3).sum())
    return num / (den + SHRINK_SHOTS)


def keeper_shrunk(t: pd.DataFrame) -> pd.Series:
    """The side's goalkeeping over its last ``KEEPER_WINDOW`` earlier matches: goals prevented over shots on target
    faced plus ``KEEPER_SHRINK_SOT`` pseudo-shots (hard shrinkage: the measure barely persists season to season)."""
    g = t.groupby("team", sort=False)
    num = g["keeper"].transform(lambda s: s.shift(1).rolling(KEEPER_WINDOW, min_periods=5).sum())
    den = g["sot_against"].transform(lambda s: s.where(t.loc[s.index, "keeper"].notna())
                                     .shift(1).rolling(KEEPER_WINDOW, min_periods=5).sum())
    return num / (den + KEEPER_SHRINK_SOT)


def fit_xg_ratings(m: pd.DataFrame, as_of: pd.Timestamp, ridge: float = 2.0) -> dict | None:
    """One pool's xG attack and defence from its matches before ``as_of``: weighted least squares of the log
    non-penalty xG (+0.1) of each side on attack, the opponent's defence and home, decayed, ridge-shrunk."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr
    w = m[(m["date"] < as_of) & (m["date"] >= as_of - pd.Timedelta(days=WINDOW_DAYS)) & m["npxg_h"].notna()
          & m["npxg_a"].notna()]
    if len(w) < 100:
        return None
    teams = pd.Index(pd.unique(pd.concat([w["home_team_id"], w["away_team_id"]])))
    n, t = len(w), len(teams)
    hi, ai = teams.get_indexer(w["home_team_id"]), teams.get_indexer(w["away_team_id"])
    wt = np.sqrt(np.exp(-XI * (as_of - w["date"]).dt.days.to_numpy(float)))
    ih, ia = np.arange(n), np.arange(n, 2 * n)
    rows = np.concatenate([ih, ia, ih, ih, ia, ih, ia])
    cols = np.concatenate([np.zeros(n), np.zeros(n), np.ones(n), 2 + hi, 2 + ai, 2 + t + ai, 2 + t + hi])
    vals = np.concatenate([np.ones(5 * n), -np.ones(2 * n)])
    wrow = np.r_[wt, wt]
    X = coo_matrix((vals * wrow[rows], (rows, cols)), shape=(2 * n, 2 + 2 * t)).tocsr()
    y = np.log(np.r_[w["npxg_h"].to_numpy(float), w["npxg_a"].to_numpy(float)] + 0.1) * wrow
    beta = lsqr(X, y, damp=np.sqrt(ridge / max(n, 1)))[0]
    return {"att": dict(zip(teams, beta[2:2 + t])), "def": dict(zip(teams, beta[2 + t:]))}


def xg_ratings(m: pd.DataFrame) -> pd.DataFrame:
    """Each match's two sides' xG ratings from the pool's fit on matches before its collection day."""
    out = []
    for pool, d in m.groupby("country"):
        cut = collection_day(d["date"]).to_numpy()
        for c in np.unique(cut):
            fit = fit_xg_ratings(d, pd.Timestamp(c))
            if fit is None:
                continue
            for r in d[cut == c].itertuples(index=False):
                out.append({"match_id": r.match_id,
                            "rxg_h": _r(fit, r.home_team_id), "rxg_a": _r(fit, r.away_team_id)})
    return pd.DataFrame(out, columns=["match_id", "rxg_h", "rxg_a"])


def _r(fit: dict, team) -> float | None:
    a, d = fit["att"].get(team), fit["def"].get(team)
    return None if a is None or d is None else a + d


MASSEY_XG_WEIGHT = 0.75          # the owner, 5 Nov 2026 (markstats.club's Massey: 75% xG, 25% goals, opponent-adjusted)
MASSEY_FROM = "2012-07-01"


def fit_massey(d: pd.DataFrame, as_of: pd.Timestamp, ridge: float = 2.0) -> dict | None:
    """One pool's Massey ratings from its matches before ``as_of``: weighted least squares of each match's margin
    (0.75 x non-penalty xG difference + 0.25 x goal difference; the goal difference alone where the match has no xG)
    on home advantage + the home side's rating - the away side's, decayed by exp(-XI days) over WINDOW_DAYS,
    ridge-shrunk to zero (so ratings sum near zero and a side with few matches stays near average)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr
    w = d[(d["date"] < as_of) & (d["date"] >= as_of - pd.Timedelta(days=WINDOW_DAYS)) & d["margin"].notna()]
    if len(w) < 50:
        return None
    teams = pd.Index(pd.unique(pd.concat([w["home_team_id"], w["away_team_id"]])))
    n, t = len(w), len(teams)
    hi, ai = teams.get_indexer(w["home_team_id"]), teams.get_indexer(w["away_team_id"])
    wt = np.sqrt(np.exp(-XI * (as_of - w["date"]).dt.days.to_numpy(float)))
    rows = np.concatenate([np.arange(n)] * 3)
    cols = np.concatenate([np.zeros(n), 1 + hi, 1 + ai])
    vals = np.concatenate([np.ones(n), np.ones(n), -np.ones(n)])
    X = coo_matrix((vals * wt[rows], (rows, cols)), shape=(n, 1 + t)).tocsr()
    beta = lsqr(X, w["margin"].to_numpy(float) * wt, damp=np.sqrt(ridge / max(n, 1)))[0]
    return {"home": float(beta[0]), "r": dict(zip(teams, beta[1:]))}


def massey_ratings(m: pd.DataFrame) -> pd.DataFrame:
    """Each match's two sides' Massey ratings from the pool's fit on matches before its collection day (point in
    time: a match never sees its own result or any later one). ``m``: one row a match with date, country, the sides,
    the score and npxg_h / npxg_a (blank where API-Football has no xG)."""
    d = m[pd.to_datetime(m["date"]) >= MASSEY_FROM].copy()
    d["date"] = pd.to_datetime(d["date"])
    gd = d["ft_home"] - d["ft_away"]
    xgd = d["npxg_h"] - d["npxg_a"]
    d["margin"] = np.where(xgd.notna(), MASSEY_XG_WEIGHT * xgd + (1 - MASSEY_XG_WEIGHT) * gd, gd)
    out = []
    for _, g in d.groupby("country"):
        cut = collection_day(g["date"]).to_numpy()
        for c in np.unique(cut):
            fit = fit_massey(g, pd.Timestamp(c))
            if fit is None:
                continue
            sel = g[cut == c]
            out.append(pd.DataFrame({"match_id": sel["match_id"].to_numpy(),
                                     "massey_h": sel["home_team_id"].map(fit["r"]).to_numpy(float),
                                     "massey_a": sel["away_team_id"].map(fit["r"]).to_numpy(float),
                                     "massey_home": fit["home"]}))
    return (pd.concat(out, ignore_index=True) if out
            else pd.DataFrame(columns=["match_id", "massey_h", "massey_a", "massey_home"]))


def build(feats: pd.DataFrame, linked: pd.DataFrame, stats: pd.DataFrame, hm: pd.DataFrame,
          he: pd.DataFrame) -> pd.DataFrame:
    """The team-feature table with the xG features joined (one row a match)."""
    m = match_level(feats, linked, stats, match_states(hm, he))
    m["date"] = pd.to_datetime(m["date"])
    t = side_rows(m)
    t["finish_shrunk"] = finishing_shrunk(t)
    t["keeper_shrunk"] = keeper_shrunk(t)
    w = windows(t, XG_MEASURES)
    sw = short_windows(t, SHORT_MEASURES)
    for c in sw.columns:
        w[c] = sw[c].to_numpy()
    # the shrunk finishing rate is already a window of earlier matches; day-lag it the same way
    w["finish_shrunk"] = t["finish_shrunk"].to_numpy()
    w["keeper_shrunk"] = t["keeper_shrunk"].to_numpy()
    w["_cut"] = collection_day(w["date"]).to_numpy()
    w["finish_shrunk"] = w.groupby(["team", "_cut"])["finish_shrunk"].transform("first")
    w["keeper_shrunk"] = w.groupby(["team", "_cut"])["keeper_shrunk"].transform("first")
    w["n_xg_prev"] = t.assign(has=t["npxg"].notna().astype(int)).groupby("team")["has"].transform(
        lambda s: s.shift(1, fill_value=0).cumsum()).astype(float).to_numpy()
    w["n_xg_prev"] = w.groupby(["team", "_cut"])["n_xg_prev"].transform("first")
    w = w.drop(columns="_cut")
    wcols = [c for c in w.columns if c not in ("match_id", "team", "venue", "date", "own_r", "own_r_d5", "n_prev")]
    hw = w[w["venue"] == "H"].set_index("match_id")[wcols]
    aw = w[w["venue"] == "A"].set_index("match_id")[wcols]
    f = pd.DataFrame(index=hw.index.union(aw.index))
    for c in wcols:
        if c == "n_xg_prev":
            continue
        f[f"x_{c}"] = hw[c].reindex(f.index) - aw[c].reindex(f.index)
    f["n_xg_prev_min"] = np.minimum(hw["n_xg_prev"].reindex(f.index), aw["n_xg_prev"].reindex(f.index))
    mm = m.copy()
    for s, side in (("h", "home"), ("a", "away")):
        pens = mm[f"pens_scored_{s}"].fillna(0) + mm[f"pens_missed_{s}"].fillna(0)
        mm[f"npxg_{s}"] = (mm[f"xg_{s}"] - PEN_XG * pens).clip(lower=0)
    r = xg_ratings(mm).set_index("match_id")
    f["r_xg_diff"] = r["rxg_h"].reindex(f.index) - r["rxg_a"].reindex(f.index)
    out = feats.merge(f.reset_index().rename(columns={"index": "match_id"}), on="match_id", how="left")
    out["xg_gap_vs_market"] = out["r_xg_diff"] - (out["r_h"] - out["r_a"])
    # the Massey rating on every match (goals alone where there is no xG), point in time
    base = feats[["match_id", "date", "country", "home_team_id", "away_team_id", "ft_home", "ft_away"]].merge(
        mm[["match_id", "npxg_h", "npxg_a"]].drop_duplicates("match_id"), on="match_id", how="left")
    ms = massey_ratings(base)
    out = out.merge(ms, on="match_id", how="left")
    out["x_massey_diff"] = out["massey_h"] - out["massey_a"]
    out["x_massey_h"], out["x_massey_a"] = out["massey_h"], out["massey_a"]
    out["x_massey_vs_open"] = out["x_massey_diff"] + out["massey_home"] - (out["molam_h"] - out["molam_a"]) \
        if "molam_h" in out else np.nan
    out = out.drop(columns=["massey_h", "massey_a", "massey_home"])
    log.info("xg features: %d matches with xG windows", int(out["n_xg_prev_min"].ge(3).sum()))
    return out


# ---------------------------------------------------------------------------
# the evaluation
# ---------------------------------------------------------------------------

WINS = ("l10", "ewm")            # two windows a measure: the samples with xG are small (QA trial, 2 Nov 2026)
CORE_WINS = ("l10", "ewm", "season")   # xG difference: last 10 and season to date (Brechot and Flepp)
TRAIN_FROM, DEV_FROM, DEV_TO = 2022, 2024, 2026
CLIP, RIDGE_ALPHA, STAGE2_L2 = 4.0, 50.0, 1e-2   # fixed before any result was read


def families() -> dict:
    from football.feature_eval import FAMILIES
    fam = {
        "xg_core": [f"x_{c}_{w}" for c in ("npxg", "npxga", "npxgd") for w in CORE_WINS],
        "xg_core_adjusted": [f"x_{c}_{w}" for c in ("npxg_adj", "npxga_adj", "npxgd_adj") for w in CORE_WINS],
        "xg_quality": [f"x_{c}_{w}" for c in ("xg_per_shot", "xga_per_shot", "inside_share") for w in WINS],
        "shots": [f"x_{c}_{w}" for c in ("shots_d", "sot_d") for w in WINS],
        "xg_vs_market": [f"x_npxgd_vs_mkt_{w}" for w in WINS],
        "luck_raw": [f"x_{c}_{w}" for c in ("finishing", "keeper") for w in WINS],
        "luck_shrunk": ["x_finish_shrunk", "x_keeper_shrunk"],
        "xg_rating": ["r_xg_diff", "xg_gap_vs_market"],
        "game_state": [f"x_trail_share_{w}" for w in WINS],
    }
    fam["xg_all"] = sorted({c for v in fam.values() for c in v})     # as registered before 4 Nov 2026: kept fixed
    # 4 Nov 2026 (the owner's fatigue, form and "other X metrics" ask): outside xg_all, so earlier arms keep meaning
    fam["x_territory"] = [f"x_{c}_{w}" for c in TERRITORY for w in WINS]
    fam["x_sos"] = [f"x_{c}_{w}" for c in SOS for w in CORE_WINS]
    # the owner, 5 Nov 2026: a Massey power rating, 75% xG and 25% goals, opponent-adjusted (markstats.club's
    # construction, fitted point in time on our own data); x_massey_vs_open: its expected margin less the opening
    # price's expected goal difference
    fam["x_massey"] = ["x_massey_diff", "x_massey_h", "x_massey_a", "x_massey_vs_open"]
    fam["x_short"] = [f"x_{c}_{w}" for c in SHORT_MEASURES
                      for w in [f"l{n}" for n in SHORT_EQUAL] + [f"lw{n}" for n in SHORT_LINEAR]]
    fam["team_all"] = FAMILIES["all"]
    fam["team_all+xg_all"] = FAMILIES["all"] + fam["xg_all"]
    return fam


def evaluate(f: pd.DataFrame, min_prev: int = 3) -> dict:
    """The outcome and move tests on matches with xG windows, walk-forward by season from DEV_FROM (each season
    fitted on the seasons from TRAIN_FROM before it). 2025/26 on was read before (twice) for other rules: here it
    is development data, and any finding needs the forward test."""
    from football.feature_eval import outcome_test, prepare
    d = prepare(f)
    d = d[(d["season"] >= TRAIN_FROM) & (d["n_xg_prev_min"] >= min_prev)]
    kw = {"train_from": TRAIN_FROM, "dev_from": DEV_FROM, "dev_to": DEV_TO, "holdout_from": DEV_TO + 1}
    fam = families()
    return {"rows": int(len(d)), "by_season": d.groupby("season").size().to_dict(),
            "same_book_rows": int(d["same_book"].sum()),
            "outcome": outcome_test(d, {k: v for k, v in fam.items()}, clip=CLIP, l2=STAGE2_L2, **kw),
            "move": ridge_move_test(d, fam)}


def _z(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Standardised on the training rows; columns over half empty in training are dropped; clipped at CLIP."""
    X, Xt = tr[cols].to_numpy(float), te[cols].to_numpy(float)
    keep = np.mean(np.isfinite(X), axis=0) >= 0.5
    X, Xt = X[:, keep], Xt[:, keep]
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    mu[~np.isfinite(mu)] = 0.0
    f = lambda A: np.clip(np.nan_to_num((A - mu) / sd), -CLIP, CLIP)       # noqa: E731
    return f(X), f(Xt)


def _ridge(Z: np.ndarray, y: np.ndarray, Zt: np.ndarray, alpha: float = RIDGE_ALPHA) -> np.ndarray:
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    beta = np.linalg.solve(A, Z.T @ (y - y.mean()))
    return y.mean() + Zt @ beta


def ridge_move_test(d: pd.DataFrame, fam: dict, seed: int = 0) -> dict:
    """The move test with ridge in place of least squares (the samples with xG are small and the features many),
    walk-forward by season from DEV_FROM; the permuted floor from the same fit on shuffled training moves."""
    m = d[d["same_book"] & d["move"].notna()]
    rng = np.random.default_rng(seed)
    out = {}
    for k, cols in fam.items():
        ys, ps, pp, mb = [], [], [], []
        for s in range(DEV_FROM, DEV_TO + 1):
            tr, te = m[m["season"] < s], m[m["season"] == s]
            if te.empty or len(tr) < 500:
                continue
            Z, Zt = _z(tr, te, cols)
            y = tr["move"].to_numpy(float)
            ys.append(te["move"].to_numpy(float))
            ps.append(_ridge(Z, y, Zt))
            pp.append(_ridge(Z, rng.permutation(y), Zt))
            mb.append(np.full(len(te), y.mean()))
        if not ys:
            continue
        y, p, q, b = (np.concatenate(x) for x in (ys, ps, pp, mb))
        sse = lambda f: float(np.sum((y - f) ** 2))        # noqa: E731
        out[k] = {"rows": int(len(y)), "r2_vs_mean": 1 - sse(p) / sse(b), "r2_permuted_vs_mean": 1 - sse(q) / sse(b),
                  "sign_right": float(np.mean(np.sign(p - b) == np.sign(y - b)))}
    return out


def stack_comparison(f: pd.DataFrame, min_prev: int = 3) -> dict:
    """Walk-forward by season: the early-to-close home move on team features vs team + xG features, linear,
    LightGBM (with market state and league) and their mean; same-book rows with xG windows."""
    import lightgbm as lgb
    from football.feature_eval import FAMILIES, prepare
    d = prepare(f)
    m = d[d["same_book"] & (d["season"] >= TRAIN_FROM) & (d["n_xg_prev_min"] >= min_prev)].copy()
    m["yh"] = np.log(m["qc_h"] / m["qo_h"])
    m["lq_fav"], m["over_open"] = np.log(m["qo_h"] / m["qo_a"]), 1 / m["open_h"] + 1 / m["open_d"] + 1 / m["open_a"]
    m["league"] = pd.Categorical(m["fd_code"].astype(str))
    mkt = ["lq_fav", "qo_d", "over_open", "qo_o", "league"]
    sets = {"team": FAMILIES["all"], "team+xg": FAMILIES["all"] + families()["xg_all"]}
    preds = {}
    for s in range(DEV_FROM, DEV_TO + 1):
        tr, te = m[m["season"] < s], m[m["season"] == s]
        if te.empty or len(tr) < 500:
            continue
        for name, cols in sets.items():
            Z, Zt = _z(tr, te, cols)
            ols = _ridge(Z, tr["yh"].to_numpy(float), Zt)
            g = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
            g.fit(tr[cols + mkt], tr["yh"])
            gb = g.predict(te[cols + mkt])
            for k, v in ((f"{name}:ridge", ols), (f"{name}:gbm", gb), (f"{name}:stack", (ols + gb) / 2)):
                preds.setdefault(k, []).append(pd.DataFrame({"y": te["yh"].to_numpy(), "p": v,
                                                             "ybar": tr["yh"].mean(), "season": s}))
    out = {}
    for k, parts in preds.items():
        t = pd.concat(parts)
        sse = float(((t.y - t.p) ** 2).sum())
        top = t.p.abs() >= t.p.abs().quantile(0.9)
        caught = np.sign(t.p[top]) * t.y[top]
        out[k] = {"rows": int(len(t)), "r2_vs_mean": 1 - sse / float(((t.y - t.ybar) ** 2).sum()),
                  "top_decile_caught": float(caught.mean()), "top_decile_right_way": float((caught > 0).mean()),
                  "seasons": sorted(int(x) for x in t.season.unique())}
    return out


def main(argv=None) -> int:
    import argparse
    import json
    from pathlib import Path

    from football.data import load_gold
    from football.player_ratings import link_matches
    from football.reference import load_aliases
    from sources.common import Store
    ap = argparse.ArgumentParser(description="Expected-goals features and their test against the market")
    ap.add_argument("--root", required=True, help="local sources folder with the gold table")
    ap.add_argument("--history-root", required=True, help="local sources folder with api_football/history_*")
    ap.add_argument("--features", required=True, help="team features parquet (football.feature_eval --features-out)")
    ap.add_argument("--table-out", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    gs, hs = Store(root=a.root), Store(root=a.history_root)
    gold = load_gold(gs)
    hm, he, st = (hs.get_parquet(f"api_football/history_{x}.parquet") for x in ("matches", "events", "stats"))
    linked = link_matches(hm, gold, load_aliases(extra=gs.get_parquet("football/gold/team_aliases.parquet")))
    f = build(pd.read_parquet(a.features), linked, st, hm, he)
    if a.table_out:
        f[[c for c in f.columns if c.startswith("x_") or c in ("match_id", "r_xg_diff", "xg_gap_vs_market",
                                                              "n_xg_prev_min")]].to_parquet(a.table_out, index=False)
    res = {"pen_xg": PEN_XG, "shrink_shots": SHRINK_SHOTS, "linked": int(len(linked)),
           "stat_rows": int(len(st)), "matches_with_xg_windows": int(f["n_xg_prev_min"].ge(3).sum()),
           "null_share": {c: float(f.loc[f["n_xg_prev_min"].ge(3), c].isna().mean()) for c in families()["xg_all"]},
           "dev": evaluate(f), "stack": stack_comparison(f)}

    def default(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x) if np.isfinite(x) else None
        return str(x)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=default) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("dev", "null_share")}, indent=1, default=default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
