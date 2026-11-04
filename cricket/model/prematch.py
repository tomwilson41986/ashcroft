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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root")
    ap.add_argument("--years", default="2016-2026")
    ap.add_argument("--paths", type=int, default=256)
    ap.add_argument("--work", default="data/cricket/eval")
    a = ap.parse_args(argv)
    lo, hi = (int(x) for x in a.years.split("-"))
    store = Store(root=a.root)
    t = load(store)
    allp = pd.concat([run_year(t, y, a.paths, Path(a.work), log=lambda m: print(m, flush=True))
                      for y in range(lo, hi + 1)], ignore_index=True)
    store.put_parquet("cricket/eval/prematch.parquet", allp)
    print(json.dumps({"matches": len(allp), "years": f"{lo}-{hi}",
                      "saved": store.where("cricket/eval/prematch.parquet")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
