"""Location-based football metrics built by racing² from event data: expected goals (xG) and expected threat (xT).

racing² football research, 6 Nov 2026 (the owner: "the most advanced set of football metrics"; stage 1: our own
models from free, commercially usable event data). Data: the Wyscout public match events (``sources.wyscout_public``,
CC BY 4.0), positions on a 0-100 grid in the acting team's attacking direction, converted to metres on a 105 x 68
pitch.

**xG** (``shots``, ``fit_xg``): every shot from open play and every direct free kick (penalties apart, priced at
their own rate). Features known when the ball is struck:

- distance to the goal centre and the angle the goal mouth subtends (the two that carry most of it);
- header or other body part (tag 403), a direct free kick (sub-event 33), a counter-attack (tag 1901);
- the assist type, from the team's previous event within 5 seconds: a cross (80), a smart / through pass
  (86, tag 901), a set piece (corner 30, free-kick cross 32), a rebound (the previous event a shot or a save).

Not used: Wyscout's 'opportunity' tag (201), a human judgement of a big chance that leaks the outcome's quality.
Scored leave-one-competition-out (each of the seven competitions predicted by a model fitted on the other six):
log loss, Brier and AUC against the base rate and a distance-and-angle logistic.

**xT** (``fit_xt``, Singh 2019): the pitch in a 16 x 12 grid; for each cell the chance a possession there ends in a
shot or a move, where successful moves (accurate passes and carries) take it, and the chance a shot there scores;
xT solved by value iteration. A move's value is xT(end) - xT(start).

    python -m football.event_models --root <store root> --out reports/football_runs/event_models.json \\
        --models models/football/event
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PITCH_X, PITCH_Y, GOAL_W = 105.0, 68.0, 7.32
SHOT, FK_SHOT, PENALTY = 100, 33, 35
GOAL_TAG, HEAD_TAG, COUNTER_TAG, ACCURATE_TAG, THROUGH_TAG, OWN_GOAL_TAG = 101, 403, 1901, 1801, 901, 102
ASSIST_WINDOW_S = 5.0
FEATURES = ["dist", "angle", "header", "free_kick", "counter", "from_cross", "from_through", "from_set_piece",
            "rebound"]
GRID_X, GRID_Y = 16, 12


def _has(tags: pd.Series, t: int) -> np.ndarray:
    return tags.fillna("").str.contains(rf"(?:^| ){t}(?: |$)", regex=True).to_numpy()


def geometry(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Distance (m) to the goal centre and the goal mouth's angle (radians) from a 0-100 position."""
    dx = (100 - np.asarray(x, float)) * PITCH_X / 100
    dy = (50 - np.asarray(y, float)) * PITCH_Y / 100
    dist = np.hypot(dx, dy)
    ang = np.arctan2(GOAL_W * dx, dx ** 2 + dy ** 2 - (GOAL_W / 2) ** 2)
    ang = np.where(ang < 0, ang + math.pi, ang)
    return dist, ang


def shots(ev: pd.DataFrame) -> pd.DataFrame:
    """One row a shot (open play and direct free kicks; penalties flagged), with its features and whether it scored."""
    ev = ev.sort_values(["match_id", "period", "sec", "id"]).reset_index(drop=True)
    prev = ev.groupby(["match_id", "period"]).shift(1)
    is_shot = ev["sub_event_id"].isin([SHOT, FK_SHOT, PENALTY]).to_numpy()
    s = ev[is_shot].copy()
    p = prev[is_shot]
    s["dist"], s["angle"] = geometry(s["x"], s["y"])
    s["goal"] = _has(s["tags"], GOAL_TAG).astype(int)
    s["header"] = _has(s["tags"], HEAD_TAG).astype(int)
    s["free_kick"] = (s["sub_event_id"] == FK_SHOT).astype(int)
    s["penalty"] = (s["sub_event_id"] == PENALTY).astype(int)
    s["counter"] = _has(s["tags"], COUNTER_TAG).astype(int)
    recent = ((s["sec"] - p["sec"]) <= ASSIST_WINDOW_S).to_numpy() & (p["team_id"] == s["team_id"]).to_numpy()
    psub = p["sub_event_id"].to_numpy()
    s["from_cross"] = (recent & (psub == 80)).astype(int)
    s["from_through"] = (recent & ((psub == 86) | _has(p["tags"], THROUGH_TAG))).astype(int)
    s["from_set_piece"] = (recent & np.isin(psub, [30, 32])).astype(int)
    s["rebound"] = (((s["sec"] - p["sec"]) <= ASSIST_WINDOW_S).to_numpy()
                    & np.isin(p["event_id"].to_numpy(), [9, 10])).astype(int)
    return s


def _scores(y, p) -> dict:
    from sklearn.metrics import roc_auc_score
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"log_loss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4),
            "brier": round(float(np.mean((p - y) ** 2)), 4),
            "auc": round(float(roc_auc_score(y, p)), 4) if 0 < y.mean() < 1 else None}


def _design(s: pd.DataFrame, cols) -> np.ndarray:
    X = s[cols].to_numpy(float).copy()
    if "dist" in cols:
        X[:, cols.index("dist")] = np.log1p(X[:, cols.index("dist")])
    return X


def fit_xg(s: pd.DataFrame) -> dict:
    """Leave-one-competition-out scores for three models, and the final models fitted on every competition."""
    from sklearn.linear_model import LogisticRegression
    from lightgbm import LGBMClassifier
    np_ = s[s["penalty"] == 0].reset_index(drop=True)
    models = {
        "base_rate": None,
        "logistic_distance_angle": (lambda: LogisticRegression(max_iter=1000), ["dist", "angle"]),
        "logistic_all": (lambda: LogisticRegression(max_iter=2000), FEATURES),
        "gbm_all": (lambda: LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                                           subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1),
                    FEATURES),
    }
    oof = {k: np.zeros(len(np_)) for k in models}
    for comp in sorted(np_["competition"].unique()):
        te = (np_["competition"] == comp).to_numpy()
        tr = ~te
        y = np_["goal"].to_numpy()
        for name, spec in models.items():
            if spec is None:
                oof[name][te] = y[tr].mean()
                continue
            make, cols = spec
            m = make().fit(_design(np_[tr], cols), y[tr])
            oof[name][te] = m.predict_proba(_design(np_[te], cols))[:, 1]
    y = np_["goal"].to_numpy()
    out = {"shots_non_penalty": int(len(np_)), "goals_non_penalty": int(y.sum()),
           "penalties": int(s["penalty"].sum()), "penalty_rate": round(float(s.loc[s["penalty"] == 1, "goal"].mean()), 4)
           if s["penalty"].any() else None,
           "leave_one_competition_out": {k: _scores(y, v) for k, v in oof.items()},
           "by_competition_gbm": {c: _scores(y[(np_["competition"] == c).to_numpy()],
                                             oof["gbm_all"][(np_["competition"] == c).to_numpy()])
                                  for c in sorted(np_["competition"].unique())}}
    # calibration of the out-of-fold GBM: predicted against scored by decile
    q = pd.qcut(oof["gbm_all"], 10, labels=False, duplicates="drop")
    out["gbm_calibration_deciles"] = [{"pred": round(float(oof["gbm_all"][q == i].mean()), 4),
                                       "scored": round(float(y[q == i].mean()), 4), "shots": int((q == i).sum())}
                                      for i in sorted(set(q))]
    lr = LogisticRegression(max_iter=2000).fit(_design(np_, FEATURES), y)
    out["logistic_all_final"] = {"intercept": float(lr.intercept_[0]),
                                 "coef": dict(zip(FEATURES, map(float, lr.coef_[0]))),
                                 "note": "dist enters as log(1 + metres)"}
    np_ = np_.assign(xg=oof["gbm_all"])
    return out, np_


def _cell(x, y):
    cx = np.clip((np.asarray(x, float) / 100 * GRID_X).astype(int), 0, GRID_X - 1)
    cy = np.clip((np.asarray(y, float) / 100 * GRID_Y).astype(int), 0, GRID_Y - 1)
    return cx, cy


def fit_xt(ev: pd.DataFrame, iters: int = 50, tol: float = 1e-6) -> tuple[np.ndarray, dict]:
    """The expected-threat grid (GRID_Y x GRID_X; x along the attack) by value iteration."""
    shots_ = ev["event_id"].eq(10) | ev["sub_event_id"].eq(FK_SHOT)
    moves = ev["event_id"].eq(8) | ev["sub_event_id"].eq(70)        # passes and carries (accelerations)
    ok = _has(ev["tags"], ACCURATE_TAG)
    s = ev[shots_ & ev["x"].notna()]
    m = ev[moves & ev["x"].notna()]
    sx, sy = _cell(s["x"], s["y"])
    mx, my = _cell(m["x"], m["y"])
    n_shot = np.zeros((GRID_Y, GRID_X))
    n_goal = np.zeros((GRID_Y, GRID_X))
    n_move = np.zeros((GRID_Y, GRID_X))
    np.add.at(n_shot, (sy, sx), 1)
    np.add.at(n_goal, (sy, sx), _has(s["tags"], GOAL_TAG).astype(float))
    np.add.at(n_move, (my, mx), 1)
    good = m[ok[m.index.to_numpy()] & m["x_end"].notna()]
    gx, gy = _cell(good["x"], good["y"])
    ex, ey = _cell(good["x_end"], good["y_end"])
    trans = np.zeros((GRID_Y * GRID_X, GRID_Y * GRID_X))
    np.add.at(trans, (gy * GRID_X + gx, ey * GRID_X + ex), 1)
    tot = n_shot + n_move
    with np.errstate(invalid="ignore", divide="ignore"):
        p_shot = np.where(tot > 0, n_shot / tot, 0)
        p_move = np.where(tot > 0, n_move / tot, 0)
        p_goal = np.where(n_shot > 0, n_goal / n_shot, 0)
        trans = trans / np.maximum(n_move.reshape(-1, 1), 1)       # failed moves end the possession (value 0)
    xt = np.zeros((GRID_Y, GRID_X))
    for i in range(iters):
        new = p_shot * p_goal + p_move * (trans @ xt.reshape(-1)).reshape(GRID_Y, GRID_X)
        if np.abs(new - xt).max() < tol:
            xt = new
            break
        xt = new
    return xt, {"iterations": i + 1, "shots": int(len(s)), "moves": int(len(m)), "successful_moves": int(len(good)),
                "max_cell": round(float(xt.max()), 4), "own_half_mean": round(float(xt[:, :GRID_X // 2].mean()), 4),
                "final_third_mean": round(float(xt[:, -GRID_X // 3:].mean()), 4)}


def move_values(ev: pd.DataFrame, xt: np.ndarray) -> pd.Series:
    """xT added by each successful pass or carry: xT(end) - xT(start); NaN for any other event."""
    mv = (ev["event_id"].eq(8) | ev["sub_event_id"].eq(70)) & pd.Series(_has(ev["tags"], ACCURATE_TAG),
                                                                       index=ev.index) & ev["x_end"].notna()
    out = pd.Series(np.nan, index=ev.index)
    sx, sy = _cell(ev.loc[mv, "x"], ev.loc[mv, "y"])
    ex, ey = _cell(ev.loc[mv, "x_end"], ev.loc[mv, "y_end"])
    out[mv] = xt[ey, ex] - xt[sy, sx]
    return out


def match_check(ev: pd.DataFrame, s_np: pd.DataFrame, matches: pd.DataFrame, xt: np.ndarray, pen_rate: float) -> dict:
    """Do the match-level metrics carry signal? Correlation of each side's xG (out of fold, penalties at their
    rate), shots and xT added with its goals in the same match, and xG difference with the goal difference."""
    pens = ev[ev["sub_event_id"] == PENALTY].groupby(["match_id", "team_id"]).size() * pen_rate
    xg = s_np.groupby(["match_id", "team_id"])["xg"].sum().add(pens, fill_value=0)
    sh = s_np.groupby(["match_id", "team_id"]).size()
    mvv = move_values(ev, xt)
    xtv = mvv.groupby([ev["match_id"], ev["team_id"]]).sum()
    rows = []
    for m in matches.itertuples(index=False):
        for team, opp, gf, ga in ((m.home_id, m.away_id, m.home_score, m.away_score),
                                  (m.away_id, m.home_id, m.away_score, m.home_score)):
            rows.append({"gf": gf, "ga": ga, "xg": xg.get((m.match_id, team), 0.0), "xga": xg.get((m.match_id, opp), 0.0),
                         "shots": sh.get((m.match_id, team), 0), "xt": xtv.get((m.match_id, team), 0.0),
                         "xt_against": xtv.get((m.match_id, opp), 0.0)})
    d = pd.DataFrame(rows).astype(float)
    c = lambda a, b: round(float(np.corrcoef(d[a], d[b])[0, 1]), 4)           # noqa: E731
    return {"team_matches": int(len(d)), "goals_per_side": round(float(d["gf"].mean()), 3),
            "xg_per_side": round(float(d["xg"].mean()), 3),
            "corr_goals_xg": c("gf", "xg"), "corr_goals_shots": c("gf", "shots"), "corr_goals_xt": c("gf", "xt"),
            "corr_gd_xgd": round(float(np.corrcoef(d["gf"] - d["ga"], d["xg"] - d["xga"])[0, 1]), 4),
            "corr_gd_xtd": round(float(np.corrcoef(d["gf"] - d["ga"], d["xt"] - d["xt_against"])[0, 1]), 4)}


def run(store, models_dir: str | None = None) -> dict:
    ev = store.get_parquet("wyscout_public/events.parquet")
    matches = store.get_parquet("wyscout_public/matches.parquet")
    if ev is None or matches is None:
        return {"status": "no data found (wyscout_public tables)"}
    s = shots(ev)
    xg_out, s_np = fit_xg(s)
    xt, xt_out = fit_xt(ev)
    out = {"source": "Wyscout public match events (Pappalardo et al. 2019, CC BY 4.0)", "events": int(len(ev)),
           "matches": int(ev["match_id"].nunique()), "xg": xg_out, "xt": xt_out,
           "match_level": match_check(ev, s_np, matches, xt, xg_out["penalty_rate"] or 0.75)}
    if models_dir:
        p = Path(models_dir)
        p.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(xt).to_csv(p / "xt_grid_16x12.csv", index=False, header=False)
        (p / "xg_logistic.json").write_text(json.dumps(xg_out["logistic_all_final"], indent=1) + "\n")
        (p / "README.md").write_text(
            "racing² event models, fitted on the Wyscout public match events (Pappalardo et al. 2019, CC BY 4.0, "
            "doi:10.6084/m9.figshare.c.4415000).\n\n- `xt_grid_16x12.csv`: expected threat, 12 rows (y, 0-100 "
            "across the pitch) by 16 columns (x, 0-100 towards the opponent's goal).\n- `xg_logistic.json`: the "
            "logistic xG (dist as log(1 + metres), angle in radians, binary features), penalties apart.\n\n"
            "Built by `football/event_models.py`.\n")
    return out


def main(argv=None) -> int:
    from sources.common import Store
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=None)
    a = ap.parse_args(argv)
    res = run(Store(root=a.root), a.models)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
