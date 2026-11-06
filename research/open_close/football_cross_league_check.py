"""Does cross-league strength predict cup and European results? The 'winning the bet' target, on results.

racing² football research, 6 Nov 2026. Out of sample by season from 2018: each season's cup and continental ties are
predicted from a mapping fitted on the seasons before it. Predictors, each a rating difference known before kick-off
(``football.cross_league``):

- **cross-league Elo** (strength carried through cup and continental ties);
- **league-only Elo** (each club rated within each competition apart: the baseline with no cross information);
- **no information** (the training seasons' result shares alone).

Scores: the log loss of the home/draw/away result (multinomial logistic on the difference) and the share of results
the most likely outcome gets right, for cup ties across divisions of one country, continental ties across countries,
and all league matches (a sanity check: within a league both ratings should do about as well).

The closing-price target and the betting returns need Betfair's prices for these matches: reported once the live
record holds enough of them (``football_cross_league_data``).

    python -m research.open_close.football_cross_league_check --out out/cross_league_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FIRST_TEST = 2018


def outcome(gd: np.ndarray) -> np.ndarray:
    return np.where(gd > 0, 0, np.where(gd == 0, 1, 2))


def score(d: pd.DataFrame, col) -> dict:
    """Walk-forward by season: multinomial logistic of the result on ``col`` (a column, a list of columns, or None:
    the shares alone)."""
    from sklearn.linear_model import LogisticRegression
    ll, right, n = 0.0, 0, 0
    for s in sorted(d["apif_season"].unique()):
        if s < FIRST_TEST:
            continue
        tr, te = d[d["apif_season"] < s], d[d["apif_season"] == s]
        if len(tr) < 500 or te.empty:
            continue
        ytr, yte = outcome(tr["gd"].to_numpy()), outcome(te["gd"].to_numpy())
        if col is None:
            p = np.tile(np.bincount(ytr, minlength=3) / len(ytr), (len(te), 1))
        else:
            cols = [col] if isinstance(col, str) else list(col)
            m = LogisticRegression(max_iter=500).fit(tr[cols].to_numpy(float) / 100, ytr)
            p = np.zeros((len(te), 3))
            p[:, m.classes_] = m.predict_proba(te[cols].to_numpy(float) / 100)
        p = np.clip(p, 1e-9, 1)
        ll += float(-np.log(p[np.arange(len(te)), yte]).sum())
        right += int((p.argmax(1) == yte).sum())
        n += len(te)
    return {"matches": n, "log_loss": round(ll / n, 4) if n else None, "right": round(right / n, 4) if n else None}


GROUP_NAMES = ("cup_ties_across_divisions", "continental_ties_across_countries", "league_matches")


def rated_matches(fx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every senior club match with its pre-match rating differences (v1 cross-league, competition-only, v2) and its
    group: a cup tie across divisions of one country, a continental tie across countries, a league match, or other."""
    from football import cross_league as cl
    f = cl.senior_club_matches(fx)
    a = cl.elo(f, cross=True)
    b = cl.elo(f, cross=False)
    v2 = cl.elo_hier(f)
    d = a.copy()
    d["gd"] = d["ft_home"] - d["ft_away"]
    d["diff_cross"] = d["r_home"] - d["r_away"]
    d["diff_league_only"] = b["r_home"].to_numpy() - b["r_away"].to_numpy()
    d["diff_v2"] = v2["r_home"].to_numpy() - v2["r_away"].to_numpy()
    # which ties cross a division or a country: each club's main domestic league that season
    lg = d[d["kind"] == "league"]
    sides = pd.concat([lg[["apif_season", "apif_home_id", "competition"]].rename(columns={"apif_home_id": "team"}),
                       lg[["apif_season", "apif_away_id", "competition"]].rename(columns={"apif_away_id": "team"})])
    home = sides.groupby(["team", "apif_season"])["competition"].agg(lambda x: x.mode().iloc[0])
    key_h = pd.MultiIndex.from_arrays([d["apif_home_id"], d["apif_season"]])
    key_a = pd.MultiIndex.from_arrays([d["apif_away_id"], d["apif_season"]])
    d["lg_h"], d["lg_a"] = home.reindex(key_h).to_numpy(), home.reindex(key_a).to_numpy()
    placed = d["lg_h"].notna() & d["lg_a"].notna()
    cty = lambda s: s.astype(str).str.split(" | ", regex=False).str[0]        # noqa: E731
    d["group"] = "other"
    d.loc[d["kind"] == "league", "group"] = "league_matches"
    d.loc[(d["kind"] == "cup") & placed & (d["lg_h"] != d["lg_a"]) & (cty(d["lg_h"]) == cty(d["lg_a"])),
          "group"] = "cup_ties_across_divisions"
    d.loc[(d["kind"] == "uefa") & placed & (cty(d["lg_h"]) != cty(d["lg_a"])),
          "group"] = "continental_ties_across_countries"
    for c in ("home", "away", "kickoff_utc"):
        if c in f.columns:
            d[c] = f[c].to_numpy()
    return d, v2


def run(fx: pd.DataFrame) -> dict:
    from football import cross_league as cl
    d, v2 = rated_matches(fx)
    groups = {g: d[d["group"] == g] for g in GROUP_NAMES}
    out = {"matches_rated": int(len(d)), "first_test_season": FIRST_TEST, "groups": {}}
    for name, g in groups.items():
        out["groups"][name] = {"cross_league_elo": score(g, "diff_cross"),
                               "v2_league_level_plus_place": score(g, "diff_v2"),
                               "v2_plus_competition_record": score(g, ["diff_v2", "diff_league_only"]),
                               "league_only_elo": score(g, "diff_league_only"),
                               "no_information": score(g, None)}
    ls = cl.league_strength(v2)
    top = ls[ls["apif_season"] == ls["apif_season"].max() - 1].sort_values("mean", ascending=False)
    out["strongest_leagues_last_full_season"] = {r.competition: round(r.mean, 1) for r in top.head(25).itertuples()
                                                 if r.count >= 100}
    return out


def main(argv=None) -> int:
    from sources.common import Store
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    fx = Store(root=a.root).get_parquet("api_football/fixtures.parquet")
    res = run(fx) if fx is not None and len(fx) else {"status": "no data found"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
