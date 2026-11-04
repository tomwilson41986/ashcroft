"""Pre-match model probabilities, walk-forward, for every in-scope men's T20 since 2016: the model side of the Benter
test against Betfair's close (cricket/model/benter.py).

    python -m cricket.model.prematch --root data/cricket/store --years 2016-2026 --paths 256

The state is the one before the first ball, after the toss: the side batting first and both XIs are known then,
as they are when Betfair's pre-play close forms (the toss is about 30 minutes before the start). For each year Y
everything the probability rests on is from before 1 January of Y:

- the population prior and venue population rates (``build(pop_before=Y)``);
- the next-ball model, fitted on deliveries before Y, its rounds chosen by early stopping on Y-1 with a fit on the
  years before (from 2023 the rounds chosen on 2022, as in cricket/model/evaluate.py);
- the player, venue and competition ratings, as of each match (always only earlier matches).

Each path plays the first innings, then the chase of its own total. Kept per year under ``--work``
(``prematch_<Y>.parquet``) and to the store (``cricket/eval/prematch.parquet``). The raw simulator probability is
kept: the Benter regression is itself the calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from cricket.model import next_ball as nb
from cricket.model import simulator as sim
from cricket.model.data import load
from cricket.model.features import build
from cricket.model.outcomes import extras_runs_distribution
from sources.common import Store

FIXED_ROUNDS_FROM = 2023      # from here, the rounds chosen on 2022 (cricket/model/evaluate.py)


def rounds_for(f: pd.DataFrame, y: int, work: Path) -> int:
    path = work / f"rounds_{y}.json"
    if path.exists():
        return json.loads(path.read_text())["rounds"]
    if y >= FIXED_ROUNDS_FROM and (work / "rounds.json").exists():
        r = json.loads((work / "rounds.json").read_text())["rounds"]
        src = "rounds.json (early stopping on 2022)"
    else:
        d = pd.to_datetime(f.start_date)
        r, v = nb.choose_rounds(f[d < f"{y - 1}-01-01"], f[(d >= f"{y - 1}-01-01") & (d < f"{y}-01-01")])
        src = f"early stopping on {y - 1}, valid mlogloss {v:.4f}"
    work.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rounds": r, "source": src}))
    return r


def start_states(f: pd.DataFrame, y: int) -> pd.DataFrame:
    s = f[(f.year == y) & (f.innings == 1)]
    return s.sort_values(["match_id", "over", "repeat", "ball"]).groupby("match_id", sort=False).head(1)


def run_year(t: dict, y: int, paths: int, work: Path, chunk: int = 250, log=print) -> pd.DataFrame:
    out = work / f"prematch_{y}.parquet"
    if out.exists():
        log(f"{y}: resumed from {out}")
        return pd.read_parquet(out)
    t0 = time.time()
    f, br, bw, prior = build(t, since="2012-01-01", pop_before=f"{y}-01-01")
    d = pd.to_datetime(f.start_date)
    rounds = rounds_for(f, y, work)
    train = f[d < f"{y}-01-01"]
    mpath = work / f"prematch_model_{y}.ubj"
    if mpath.exists():
        b = xgb.Booster()
        b.load_model(mpath)
    else:
        b = nb.fit(train, rounds)
        b.save_model(mpath)
    ex_v, ex_p = extras_runs_distribution(train)
    s = start_states(f, y)
    ids = set(s.match_id)
    dsub = t["deliveries"][t["deliveries"].match_id.isin(ids)]
    psub = t["players"][t["players"].match_id.isin(ids)]
    parts = []
    for i in range(0, len(s), chunk):
        c = s.iloc[i:i + chunk]
        batch = sim.prepare(c, dsub[dsub.match_id.isin(set(c.match_id))], psub[psub.match_id.isin(set(c.match_id))],
                            br, bw, prior)
        p, tot = sim.match_probability(b, batch, paths, ex_v, ex_p, seed=y * 100 + i // chunk)
        parts.append(c[["match_id", "start_date", "year", "event_name", "team", "bowl_team", "bat_won"]].assign(
            p_bat_first=p, total_q50=np.median(tot, axis=1)))
    r = pd.concat(parts, ignore_index=True).rename(columns={"team": "bat_first", "bowl_team": "bowl_first"})
    r["rounds"] = rounds
    work.mkdir(parents=True, exist_ok=True)
    r.to_parquet(out, index=False)
    log(f"{y}: {len(r)} matches, rounds {rounds}, {time.time() - t0:.0f}s")
    return r


def bat_first_rate_before(t: dict, y: int) -> float:
    """Share of in-scope matches before 1 January of ``y`` won by the side batting first: the constant to beat."""
    from cricket.model.features import t20_matches
    m = t20_matches(t["matches"])
    m = m[m.start_date < f"{y}-01-01"]
    first = t["innings"][(t["innings"].innings == 1) & t["innings"].match_id.isin(m.match_id)]
    j = m.merge(first[["match_id", "team"]], on="match_id")
    return float((j.winner == j.team).mean())


def report_md(allp: pd.DataFrame, rates: dict[int, float], source: str) -> str:
    from cricket.model.metrics import paired_bootstrap, score
    L = ["# Cricket - pre-match probabilities, walk-forward (the model side of the Benter test)", "",
         f"Source: `{source}`. Built by `python -m cricket.model.prematch`. The state before the first ball, after "
         "the toss; each year from a model, rounds and prior from before it. The constant is the share of matches "
         "won by the side batting first in every in-scope match before that year.", "",
         "| year | matches | rounds | model log loss | constant log loss | constant | model ECE |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    allp = allp.assign(p_const=allp.year.map(rates))
    for y, g in allp.groupby("year"):
        sm, sc = score(g.bat_won, g.p_bat_first), score(g.bat_won, g.p_const)
        L.append(f"| {y} | {len(g):,} | {int(g.rounds.iloc[0])} | {sm['log_loss']:.4f} | {sc['log_loss']:.4f} | "
                 f"{rates[y]:.3f} | {sm['ece']:.4f} |")
    sm, sc = score(allp.bat_won, allp.p_bat_first), score(allp.bat_won, allp.p_const)
    b = paired_bootstrap(allp.bat_won, allp.p_bat_first, allp.p_const, allp.match_id)
    L += [f"| all | {len(allp):,} | | {sm['log_loss']:.4f} | {sc['log_loss']:.4f} | | {sm['ece']:.4f} |", "",
          f"Model minus constant: {b['diff']:+.4f} (95% interval {b['lo']:+.4f} to {b['hi']:+.4f}, one row a match).",
          "The test that matters is against Betfair's close (cricket/model/benter.py): the constant only shows the "
          "model knows something before the first ball.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root")
    ap.add_argument("--years", default="2016-2026")
    ap.add_argument("--paths", type=int, default=256)
    ap.add_argument("--work", default="data/cricket/eval")
    ap.add_argument("--out", default="reports/cricket")
    ap.add_argument("--report-only", action="store_true", help="the report from the years kept under --work")
    a = ap.parse_args(argv)
    lo, hi = (int(x) for x in a.years.split("-"))
    store = Store(root=a.root)
    t = load(store)
    years = range(lo, hi + 1)
    if a.report_only:
        allp = pd.concat([pd.read_parquet(Path(a.work) / f"prematch_{y}.parquet") for y in years], ignore_index=True)
    else:
        allp = pd.concat([run_year(t, y, a.paths, Path(a.work), log=lambda m: print(m, flush=True))
                          for y in years], ignore_index=True)
    store.put_parquet("cricket/eval/prematch.parquet", allp)
    md = report_md(allp, {y: bat_first_rate_before(t, y) for y in years}, store.where("cricket/tables/"))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "prematch.md").write_text(md)
    store.put("cricket/reports/prematch.md", md.encode())
    print(md)
    print(json.dumps({"matches": len(allp), "years": f"{lo}-{hi}",
                      "saved": store.where("cricket/eval/prematch.parquet")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
