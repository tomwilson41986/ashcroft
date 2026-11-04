"""First-innings evaluation v1: P(the side batting first wins) and the first-innings runs line (build spec 6.4-6.5).

    python -m cricket.model.evaluate_first --root data/cricket/store --seasons 2023,2024,2025,2026 --paths 256

For each season Y the next-ball model fitted on deliveries before 1 January of Y (cricket/model/evaluate.py, kept
under ``--work``) plays out every IPL first-innings state at the start of an over (the first delivery of each over,
decided non-D/L matches) and then the chase of each path's own total. Scored against:

- win probability: a logistic baseline on runs, wickets and legal balls at the same over starts, fitted on earlier
  IPL seasons; calibrated by isotonic regression on earlier evaluated seasons only (the first season is
  calibration-only);
- the runs line: the innings' real total against the simulated totals: mean absolute error of the median, the
  share of totals inside the central 50% and 80% ranges (a calibrated model gives 0.50 and 0.80), and CRPS; the
  baseline is a linear fit of the final total on runs, wickets and legal balls with its own residual spread, fitted
  on earlier IPL seasons.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cricket.model import simulator as sim
from cricket.model.calibration import walk_forward_calibrate
from cricket.model.data import IPL, load
from cricket.model.evaluate import choose_rounds, load_or_fit
from cricket.model.features import build
from cricket.model.metrics import reliability, score
from cricket.model.outcomes import extras_runs_distribution
from sources.common import Store

BASE_COLS = ["runs_before", "wickets_before", "legal_before"]


def crps_samples(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS from an ensemble, row-wise: E|X - y| - 0.5 E|X - X'| (sorted-sample identity for the second term)."""
    x = np.sort(samples, axis=1).astype(float)
    m = x.shape[1]
    t1 = np.abs(x - y[:, None]).mean(1)
    i = np.arange(1, m + 1)
    t2 = (2 * (x * (2 * i - m - 1)).sum(1)) / (m * m)
    return t1 - 0.5 * t2


def over_starts(f: pd.DataFrame) -> pd.DataFrame:
    s = f[(f.event_name == IPL) & (f.innings == 1) & (f.ball == 1) & (f.repeat == 0) & (f.balls_left > 0)]
    return s[s.legal_before % 6 == 0]


def run(store: Store, seasons: list[int], paths: int, work: Path, log=print) -> dict:
    t = load(store)
    f, br, bw, prior = build(t)
    totals = t["innings"][t["innings"].innings == 1].set_index("match_id")["runs"]
    rounds = choose_rounds(f, work, log)
    d = pd.to_datetime(f.start_date)
    starts_all = over_starts(f)
    starts_all = starts_all.assign(total=starts_all.match_id.map(totals).to_numpy())
    parts = []
    for y in seasons:
        out = work / f"first_{y}.parquet"
        if out.exists():
            parts.append(pd.read_parquet(out))
            log(f"{y}: resumed from {out}")
            continue
        t0 = time.time()
        train = f[d < f"{y}-01-01"]
        b = load_or_fit(work, y, train, rounds)
        ex_v, ex_p = extras_runs_distribution(train)
        s_y = starts_all[starts_all.year == y]
        ids = set(s_y.match_id)
        dsub = t["deliveries"][t["deliveries"].match_id.isin(ids)]
        psub = t["players"][t["players"].match_id.isin(ids)]
        rows = []
        for i, mid in enumerate(s_y.match_id.unique()):
            s = s_y[s_y.match_id == mid]
            batch = sim.prepare(s, dsub[dsub.match_id == mid], psub[psub.match_id == mid], br, bw, prior)
            p, tot = sim.match_probability(b, batch, paths, ex_v, ex_p, seed=(y * 1000 + i) % 2**32)
            q = np.quantile(tot, [0.1, 0.25, 0.5, 0.75, 0.9], axis=1).T
            r = s[["match_id", "year", "legal_before", "runs_before", "wickets_before", "bat_won", "total"]].copy()
            r["p_sim"] = p
            r[["q10", "q25", "q50", "q75", "q90"]] = q
            r["crps"] = crps_samples(tot, s.total.to_numpy(dtype=float))
            rows.append(r)
        ry = pd.concat(rows, ignore_index=True)
        prior_seasons = starts_all[starts_all.year < y]
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(prior_seasons[BASE_COLS], prior_seasons.bat_won)
        ry["p_base"] = lr.predict_proba(ry[BASE_COLS])[:, 1]
        lin = LinearRegression().fit(prior_seasons[BASE_COLS], prior_seasons.total)
        resid = prior_seasons.total - lin.predict(prior_seasons[BASE_COLS])
        ry["base_q50"] = lin.predict(ry[BASE_COLS]) + resid.median()
        for lo, hi, name in ((0.25, 0.75, "50"), (0.1, 0.9, "80")):
            ry[f"base_lo{name}"] = ry.base_q50 - resid.median() + resid.quantile(lo)
            ry[f"base_hi{name}"] = ry.base_q50 - resid.median() + resid.quantile(hi)
        work.mkdir(parents=True, exist_ok=True)
        ry.to_parquet(out, index=False)
        parts.append(ry)
        log(f"{y}: {ry.match_id.nunique()} first innings, {len(ry)} over starts, {time.time() - t0:.0f}s")
    allp = walk_forward_calibrate(pd.concat(parts, ignore_index=True))
    return {"preds": allp, "rounds": rounds, "paths": paths}


def runs_line(g: pd.DataFrame) -> dict:
    y = g.total.to_numpy(dtype=float)
    return {"mae_sim": float(np.abs(g.q50 - y).mean()), "mae_base": float(np.abs(g.base_q50 - y).mean()),
            "in50_sim": float(((y >= g.q25) & (y <= g.q75)).mean()),
            "in80_sim": float(((y >= g.q10) & (y <= g.q90)).mean()),
            "in50_base": float(((y >= g.base_lo50) & (y <= g.base_hi50)).mean()),
            "in80_base": float(((y >= g.base_lo80) & (y <= g.base_hi80)).mean()),
            "crps_sim": float(g.crps.mean())}


def report_md(res: dict, source: str) -> str:
    p = res["preds"]
    test = p[p.p_cal.notna()]
    yrs = sorted(test.year.unique())
    pooled = {c: score(test.bat_won, test[c]) for c in ("p_base", "p_sim", "p_cal")}
    L = [
        "# Cricket model v1, first innings: match winner and the runs line",
        "",
        f"Source tables: `{source}` (Cricsheet). Built by `python -m cricket.model.evaluate_first`. Paths per state: "
        f"{res['paths']}. Next-ball rounds: {res['rounds']}.",
        f"Rows: IPL first innings at the start of each over, decided non-D/L matches. Test seasons {yrs[0]}-"
        f"{yrs[-1]}: {len(test):,} over starts from {test.match_id.nunique()} matches ({p.year.min()} is "
        "calibration-only).",
        "",
        "## P(side batting first wins)",
        "",
        "| model | log loss | Brier | ECE |", "|---|---:|---:|---:|",
    ]
    for c, name in (("p_base", "baseline: logistic on runs, wickets, legal balls"), ("p_sim", "simulator, raw"),
                    ("p_cal", "simulator, calibrated")):
        L.append(f"| {name} | {pooled[c]['log_loss']:.4f} | {pooled[c]['brier']:.4f} | {pooled[c]['ece']:.4f} |")
    L += ["", "| season | over starts | baseline LL | raw LL | calibrated LL |", "|---|---:|---:|---:|---:|"]
    for y, g in p.groupby("year"):
        cal = f"{score(g.bat_won, g.p_cal)['log_loss']:.4f}" if g.p_cal.notna().all() else "calibration only"
        L.append(f"| {y} | {len(g):,} | {score(g.bat_won, g.p_base)['log_loss']:.4f} | "
                 f"{score(g.bat_won, g.p_sim)['log_loss']:.4f} | {cal} |")
    L += ["", "Reliability, calibrated (test seasons):", "", "| p from | rows | mean p | batting first won |",
          "|---|---:|---:|---:|"]
    for r in reliability(test.bat_won, test.p_cal).itertuples():
        L.append(f"| {r.lo:.1f} | {r.n:,} | {r.mean_p:.3f} | {r.win_rate:.3f} |")
    L += ["", "## First-innings runs line (all evaluated seasons)", "",
          "| overs bowled | rows | MAE sim | MAE baseline | in 50% sim | in 50% base | in 80% sim | in 80% base | "
          "CRPS sim |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    bands = [(0, 0), (1, 5), (6, 9), (10, 14), (15, 19)]
    for lo, hi in bands:
        g = p[(p.legal_before // 6 >= lo) & (p.legal_before // 6 <= hi)]
        if len(g):
            r = runs_line(g)
            L.append(f"| {lo}-{hi} | {len(g):,} | {r['mae_sim']:.1f} | {r['mae_base']:.1f} | {r['in50_sim']:.2f} | "
                     f"{r['in50_base']:.2f} | {r['in80_sim']:.2f} | {r['in80_base']:.2f} | {r['crps_sim']:.1f} |")
    r = runs_line(p)
    L.append(f"| all | {len(p):,} | {r['mae_sim']:.1f} | {r['mae_base']:.1f} | {r['in50_sim']:.2f} | "
             f"{r['in50_base']:.2f} | {r['in80_sim']:.2f} | {r['in80_base']:.2f} | {r['crps_sim']:.1f} |")
    L += ["", "The runs-line ranges are not calibrated after the simulation: coverage away from 0.50 and 0.80 is the "
          "simulator's own spread.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root")
    ap.add_argument("--seasons", default="2023,2024,2025,2026")
    ap.add_argument("--paths", type=int, default=256)
    ap.add_argument("--work", default="data/cricket/eval")
    ap.add_argument("--out", default="reports/cricket")
    a = ap.parse_args(argv)
    store = Store(root=a.root)
    res = run(store, [int(x) for x in a.seasons.split(",")], a.paths, Path(a.work),
              log=lambda m: print(m, flush=True))
    md = report_md(res, store.where("cricket/tables/"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "model_v1_first_innings.md").write_text(md)
    store.put("cricket/reports/model_v1_first_innings.md", md.encode())
    store.put_parquet("cricket/eval/model_v1_first_innings_preds.parquet", res["preds"])
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
