"""The baseline to beat (build spec 3, 8): logistic regression on the required run rate, IPL chases, per ball.

    python -m cricket.model.baseline --root data/cricket/store --test-from 2022 --out reports/cricket

Walk-forward by season: each test season is scored by a model fitted on every earlier IPL season only. Rows are the
state before every delivery of the second innings of decided, non-D/L men's IPL matches. Two fits:

- ``rrr``: the chase's required run rate alone (the published baseline's form);
- ``rrr_wkts_balls``: plus wickets in hand and balls left.

The published figure (log loss 0.4379, ECE 0.0773; Mishra, arXiv 2608.14696, per the research brief) is not
re-verified here: its exact rows and split are not known, so this run is a like-for-like attempt, recorded either way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from cricket.model.data import IPL, ball_frame, load, scope_matches
from cricket.model.metrics import reliability, score
from sources.common import Store

FEATURES = {
    "rrr": ["rrr"],
    "rrr_wkts_balls": ["rrr", "wkts_in_hand", "balls_left"],
}
PUBLISHED = {"log_loss": 0.4379, "ece": 0.0773, "source": "Mishra, arXiv 2608.14696 (research brief; not re-verified)"}


def chase_frame(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    m = scope_matches(t["matches"], ("T20",), "male", IPL)
    f = ball_frame(t, m)
    f = f[(f.innings == 2) & (f.balls_left > 0)].copy()
    f["rrr"] = f.required_rate.clip(lower=0, upper=36)
    f["wkts_in_hand"] = 10 - f.wickets_before
    return f


def walk_forward(f: pd.DataFrame, test_from: int) -> tuple[pd.DataFrame, dict]:
    seasons = sorted(y for y in f.year.unique() if y >= test_from)
    preds = []
    for y in seasons:
        tr, te = f[f.year < y], f[f.year == y]
        row = te[["match_id", "year", "innings", "legal_before", "bat_won"]].copy()
        for name, cols in FEATURES.items():
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(tr[cols], tr.bat_won)
            row[f"p_{name}"] = lr.predict_proba(te[cols])[:, 1]
        preds.append(row)
    p = pd.concat(preds, ignore_index=True)
    res = {"rows": int(len(p)), "matches": int(p.match_id.nunique()), "seasons": [int(s) for s in seasons],
           "train_first_season": int(f.year.min())}
    for name in FEATURES:
        res[name] = score(p.bat_won, p[f"p_{name}"])
        res[name]["by_season"] = {int(y): score(g.bat_won, g[f"p_{name}"]) for y, g in p.groupby("year")}
    return p, res


def report_md(res: dict, p: pd.DataFrame, source: str) -> str:
    lines = [
        "# Baseline: logistic on required run rate, IPL chases",
        "",
        f"Source tables: `{source}` (Cricsheet). Built by `python -m cricket.model.baseline`.",
        f"Rows: the state before every delivery of the second innings, decided non-D/L men's IPL matches; "
        f"{res['rows']:,} rows from {res['matches']} matches in the test seasons {res['seasons'][0]}-"
        f"{res['seasons'][-1]}. Walk-forward: each season scored by a fit on IPL seasons {res['train_first_season']} "
        "to the season before only.",
        "",
        "| model | log loss | Brier | ECE |",
        "|---|---:|---:|---:|",
    ]
    for name in FEATURES:
        r = res[name]
        lines.append(f"| {name} | {r['log_loss']:.4f} | {r['brier']:.4f} | {r['ece']:.4f} |")
    lines.append(f"| published (Mishra) | {PUBLISHED['log_loss']:.4f} | | {PUBLISHED['ece']:.4f} |")
    lines += ["", f"Published figure: {PUBLISHED['source']}. Its rows and split are not known, so the comparison is "
              "indicative only.", "", "## By season", "",
              "| season | rows | rrr log loss | rrr ECE | rrr+wkts+balls log loss | rrr+wkts+balls ECE |",
              "|---|---:|---:|---:|---:|---:|"]
    for y in res["seasons"]:
        a, b = res["rrr"]["by_season"][y], res["rrr_wkts_balls"]["by_season"][y]
        lines.append(f"| {y} | {a['n']:,} | {a['log_loss']:.4f} | {a['ece']:.4f} | {b['log_loss']:.4f} | "
                     f"{b['ece']:.4f} |")
    lines += ["", "## Reliability (rrr+wkts+balls, all test seasons)", "",
              "| p from | rows | mean p | chase won |", "|---|---:|---:|---:|"]
    for r in reliability(p.bat_won, p.p_rrr_wkts_balls).itertuples():
        lines.append(f"| {r.lo:.1f} | {r.n:,} | {r.mean_p:.3f} | {r.win_rate:.3f} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="local store folder (default: S3)")
    ap.add_argument("--test-from", type=int, default=2022)
    ap.add_argument("--out", default="reports/cricket")
    a = ap.parse_args(argv)
    store = Store(root=a.root)
    t = load(store, ("matches", "innings", "deliveries"))
    f = chase_frame(t)
    p, res = walk_forward(f, a.test_from)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md = report_md(res, p, store.where("cricket/tables/"))
    (out / "baseline_rrr.md").write_text(md)
    (out / "baseline_rrr.json").write_text(json.dumps(res, indent=1, default=float))
    store.put("cricket/reports/baseline_rrr.md", md.encode())
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
