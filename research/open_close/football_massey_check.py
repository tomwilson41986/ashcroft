"""Does the Massey power rating predict results better than raw xG difference? (markstats.club's claim, 5 Nov 2026)

racing² football research. Out of sample by season: each season's matches are predicted from a linear fit on the
seasons before it (2016 on). The target is the match's goal difference (home less away); the score is the R2 against
the training mean and the share of matches whose result (home win, draw, away win) the sign agrees with, on the
matches every predictor covers (so the comparison is like for like). Early season: each side's matches this season
before the collection day at most ``EARLY`` (``ha_played_*``). Every predictor is known at the collection day.

    python -m research.open_close.football_massey_check --cache <lab cache> --out reports/football_runs/massey_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EARLY = 8
PREDICTORS = {
    "massey (75% xG + 25% goals, opponent-adjusted)": ["x_massey_diff"],
    "raw xG difference, season to date": ["x_npxgd_season"],
    "raw xG difference, last 10": ["x_npxgd_l10"],
    "goal difference, season to date": ["d_gd_season"],
    "market opening expected goal difference": ["open_xgd"],
}


def frame(cache: str) -> pd.DataFrame:
    from football import lab
    m = lab.latest(cache)
    m, _ = lab.add_blocks(m, ["home_adv"])
    m["gd"] = m["ft_home"] - m["ft_away"]
    m["open_xgd"] = m["molam_h"] - m["molam_a"]
    m["early"] = np.minimum(m["ha_played_h"], m["ha_played_a"]) <= EARLY
    return m


def score(m: pd.DataFrame, first: int = 2023, last: int = 2026) -> dict:
    cols = sorted({c for v in PREDICTORS.values() for c in v})
    d = m.dropna(subset=cols + ["gd"])
    out = {"rows": int(len(d)), "early_rows": int(d["early"].sum()), "seasons": [first, last], "by_predictor": {}}
    for name, xs in PREDICTORS.items():
        ys, ps, bs, early = [], [], [], []
        for s in range(first, last + 1):
            tr, te = d[d["season"] < s], d[d["season"] == s]
            if len(tr) < 500 or te.empty:
                continue
            X = np.c_[np.ones(len(tr)), tr[xs].to_numpy(float)]
            beta = np.linalg.lstsq(X, tr["gd"].to_numpy(float), rcond=None)[0]
            ps.append(np.c_[np.ones(len(te)), te[xs].to_numpy(float)] @ beta)
            ys.append(te["gd"].to_numpy(float))
            bs.append(np.full(len(te), tr["gd"].mean()))
            early.append(te["early"].to_numpy(bool))
        y, p, b, e = (np.concatenate(v) for v in (ys, ps, bs, early))
        res = {}
        for part, mask in (("all", np.ones(len(y), bool)), ("early_season", e), ("rest_of_season", ~e)):
            yy, pp, bb = y[mask], p[mask], b[mask]
            sse = float(np.sum((yy - pp) ** 2))
            res[part] = {"rows": int(mask.sum()), "r2": round(1 - sse / float(np.sum((yy - bb) ** 2)), 4),
                         "result_right": round(float(np.mean(np.sign(np.round(pp)) == np.sign(yy))), 4),
                         "corr": round(float(np.corrcoef(pp, yy)[0, 1]), 4)}
        out["by_predictor"][name] = res
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = score(frame(a.cache))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
