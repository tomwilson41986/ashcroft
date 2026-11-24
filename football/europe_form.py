"""Each side's European matches as form, for its domestic matches: the raw columns of ``football/blocks/eu_form.py``.

racing² football research, 24 Nov 2026 (the owner: "use the European matches in team form"). The domestic models see
only league matches; a side's Champions League, Europa League and Conference League games (Sportmonks' Euro Club
Tournaments, 2024/25 on) are invisible to them. Here:

1. A Sportmonks team id is tied to our canonical team id through the domestic matches the lab stage linked
   (``football.sportmonks_stage``): the id each canonical side plays under most often.
2. Each European match becomes two side rows: goals for and against, non-penalty xG for and against, and points
   against the bookmakers' consensus near the kick-off (3 x win chance + draw chance).
3. For every domestic match from 2024/25, each side's European record **before the match's collection day**
   (``football.team_features.collection_day``): nothing played on or after it enters.

Columns, ``_h`` / ``_a`` for the two sides: ``eux_n90`` (European matches in the 90 days before), ``eux_days`` (days
from the last European match before the collection day to this match), ``eux_xgd`` and ``eux_pae`` (120-day
half-life means of non-penalty xG difference and of points above the market's expectation, over the last 365 days),
``eux_w`` (the weight behind those means).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EUROPEAN = ("UCL", "UEL", "UECL", "USC")
HALF_DAYS, WINDOW_DAYS, RECENT_DAYS = 120.0, 365, 90
SIDE_COLS = ["eux_n90", "eux_days", "eux_xgd", "eux_pae", "eux_w"]
COLUMNS = [f"{c}_{s}" for c in SIDE_COLS for s in ("h", "a")]


def team_map(linked: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """{Sportmonks team id: canonical team id} from the linked domestic matches."""
    g = gold[["match_id", "home_team_id", "away_team_id"]].drop_duplicates("match_id")
    j = linked[["match_id", "sm_home_id", "sm_away_id"]].merge(g, on="match_id")
    pairs = pd.concat([j[["sm_home_id", "home_team_id"]].set_axis(["sm", "team"], axis=1),
                       j[["sm_away_id", "away_team_id"]].set_axis(["sm", "team"], axis=1)]).dropna()
    best = pairs.groupby(["sm", "team"]).size().reset_index(name="n").sort_values("n").drop_duplicates("sm",
                                                                                                    keep="last")
    return dict(zip(best["sm"].astype("int64"), best["team"]))


def european_sides(sm: pd.DataFrame, tmap: dict) -> pd.DataFrame:
    """Two rows a European match whose side is one of ours: team, day, xG difference, points above the market."""
    e = sm[sm["competition_id"].isin(EUROPEAN)].copy()
    if e.empty:
        return pd.DataFrame(columns=["team", "day", "xgd", "pae"])
    e["day"] = pd.to_datetime(e["kickoff_utc"], utc=True).dt.tz_convert("Europe/London").dt.tz_localize(None) \
        .dt.normalize()
    rows = []
    for me, op, q_w in (("h", "a", "od_q_h"), ("a", "h", "od_q_a")):
        side = "home" if me == "h" else "away"
        team = e[f"sm_{side}_id"].map(lambda x: tmap.get(int(x)) if pd.notna(x) else None)
        gf, ga = e[f"ft_{side}"], e[f"ft_{'away' if me == 'h' else 'home'}"]
        pts = np.where(gf > ga, 3.0, np.where(gf == ga, 1.0, 0.0))
        exp = 3 * e.get(q_w, np.nan) + e.get("od_q_d", np.nan)
        xgd = e.get(f"xg_npxg_{me}", np.nan) - e.get(f"xg_npxg_{op}", np.nan)
        rows.append(pd.DataFrame({"team": team, "day": e["day"], "xgd": xgd,
                                  "pae": np.where(gf.notna() & ga.notna(), pts, np.nan) - exp}))
    s = pd.concat(rows, ignore_index=True)
    return s[s["team"].notna()].sort_values(["team", "day"]).reset_index(drop=True)


def side_form(eu: pd.DataFrame, team: np.ndarray, day: np.ndarray, coll: np.ndarray) -> dict:
    """The European record of each (team, match day) before its collection day ``coll``."""
    n = len(team)
    out = {c: np.full(n, np.nan) for c in SIDE_COLS}
    groups = {t: g for t, g in eu.groupby("team")}
    lam = np.log(2.0) / HALF_DAYS
    for i in range(n):
        g = groups.get(team[i])
        out["eux_n90"][i] = 0.0
        out["eux_w"][i] = 0.0
        if g is None:
            continue
        d = g["day"].to_numpy("datetime64[D]")
        k = np.searchsorted(d, coll[i], side="left")               # strictly before the collection day
        if k == 0:
            continue
        past = g.iloc[:k]
        age = (coll[i] - d[:k]).astype(float)
        out["eux_n90"][i] = float((age <= RECENT_DAYS).sum())
        out["eux_days"][i] = float((day[i] - d[k - 1]).astype(float))
        keep = age <= WINDOW_DAYS
        w = np.exp(-lam * age) * keep
        for c, col in (("eux_xgd", "xgd"), ("eux_pae", "pae")):
            v = past[col].to_numpy(float)
            ok = np.isfinite(v) & (w > 0)
            if ok.any():
                out[c][i] = float((w[ok] * v[ok]).sum() / w[ok].sum())
        out["eux_w"][i] = float(w[np.isfinite(past["pae"].to_numpy(float))].sum())
    return out


def build(gold: pd.DataFrame, sm: pd.DataFrame, linked: pd.DataFrame, since: str = "2024-07-01") -> pd.DataFrame:
    """``match_id`` + ``COLUMNS`` for the domestic matches from ``since``."""
    from football.team_features import collection_day
    need = {"match_id", "match_date", "home_team_id", "away_team_id"}
    if sm is None or sm.empty or linked is None or linked.empty or not need <= set(gold.columns):
        return pd.DataFrame(columns=["match_id"] + COLUMNS)
    eu = european_sides(sm, team_map(linked, gold))
    g = gold[pd.to_datetime(gold["match_date"]) >= pd.Timestamp(since)].drop_duplicates("match_id")
    day = pd.to_datetime(g["match_date"]).to_numpy("datetime64[D]")
    coll = collection_day(g["match_date"]).to_numpy("datetime64[D]")
    out = g[["match_id"]].copy()
    for side, col in (("h", "home_team_id"), ("a", "away_team_id")):
        f = side_form(eu, g[col].to_numpy(), day, coll)
        for c in SIDE_COLS:
            out[f"{c}_{side}"] = f[c]
    return out.reset_index(drop=True)
