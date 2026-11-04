"""Phase 0 model evaluation v1: next-ball model, simulator, calibration, against the RRR baseline (build spec 6, 8).

    python -m cricket.model.evaluate --root data/cricket/store --seasons 2023,2024,2025,2026 --paths 256

Walk-forward by season Y:
1. the next-ball model is fitted on men's 20-over T20 deliveries (all competitions, 2012 onwards) dated before
   1 January of Y; its rounds are chosen once by early stopping on 2022 with a fit on 2012-2021;
2. it is scored on every men's T20 delivery of Y (and the IPL's) against the base rate;
3. the simulator rolls out every IPL chase state of Y (the state before each delivery of the second innings of
   decided, non-D/L matches): raw win probability;
4. the raw output of Y is calibrated by isotonic regression fitted on the earlier evaluated seasons only, so the
   first season is calibration-only;
5. the baseline (logistic on RRR, wickets in hand, balls left; IPL seasons before Y) is scored on the same rows.

Per-season simulator output is kept under ``--work`` so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from cricket.model import next_ball as nb
from cricket.model import simulator as sim
from cricket.model.baseline import PUBLISHED
from cricket.model.calibration import METHODS, SELECT_ON, all_methods, by_band
from cricket.model.data import IPL, load
from cricket.model.features import build
from cricket.model.gate import FIXED_BAR, gate_lines, model_gate
from cricket.model.metrics import reliability, score
from cricket.model.outcomes import CLASSES, extras_runs_distribution
from sources.common import Store

BASE_COLS = ["rrr", "wkts_in_hand", "balls_left"]


def _date(f):
    return pd.to_datetime(f.start_date)


def load_or_fit(work: Path, y: int, train: pd.DataFrame, rounds: int):
    """The season's next-ball model (fitted on deliveries before 1 January of ``y``), kept under ``work``."""
    import xgboost as xgb
    path = work / f"next_ball_{y}.ubj"
    if path.exists():
        b = xgb.Booster()
        b.load_model(path)
        return b
    b = nb.fit(train, rounds)
    work.mkdir(parents=True, exist_ok=True)
    b.save_model(path)
    return b


def choose_rounds(f: pd.DataFrame, work: Path, log=print) -> int:
    """Rounds by early stopping on 2022 (a fit on 2012-2021), kept under ``work``."""
    path = work / "rounds.json"
    if path.exists():
        return json.loads(path.read_text())["rounds"]
    d = _date(f)
    rounds, vloss = nb.choose_rounds(f[d < "2022-01-01"], f[(d >= "2022-01-01") & (d < "2023-01-01")])
    log(f"rounds {rounds} (valid 2022 mlogloss {vloss:.4f})")
    work.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rounds": rounds, "valid_2022_mlogloss": vloss}))
    return rounds


def run(store: Store, seasons: list[int], paths: int, work: Path, log=print) -> dict:
    t = load(store)
    f, br, bw, prior = build(t)
    d = _date(f)
    rounds = choose_rounds(f, work, log)
    vloss = json.loads((work / "rounds.json").read_text())["valid_2022_mlogloss"]
    f["wkts_in_hand"] = 10 - f.wickets_before
    meta = {"rounds": rounds, "valid_2022_mlogloss": vloss, "params": nb.PARAMS, "paths": paths,
            "train_rows_by_season": {}, "next_ball": {}}
    sims = []
    for y in seasons:
        cut = f"{y}-01-01"
        train = f[d < cut]
        meta["train_rows_by_season"][y] = int(len(train))
        test_all = f[f.year == y]
        out = work / f"sim_{y}.parquet"
        nb_out = work / f"next_ball_{y}.json"
        if out.exists() and nb_out.exists():
            sims.append(_smoothed(pd.read_parquet(out), paths))
            meta["next_ball"][y] = json.loads(nb_out.read_text())
            log(f"{y}: resumed from {out}")
            continue
        t0 = time.time()
        b = load_or_fit(work, y, train, rounds)
        ev = {"all_t20": nb.evaluate(b, train, test_all),
              "ipl": nb.evaluate(b, train, test_all[test_all.event_name == IPL])}
        meta["next_ball"][y] = ev
        log(f"{y}: fit {time.time() - t0:.0f}s, next-ball {ev['all_t20']['log_loss']:.4f} vs base "
            f"{ev['all_t20']['base_rate_log_loss']:.4f}")
        ex_v, ex_p = extras_runs_distribution(train)
        chase = test_all[(test_all.event_name == IPL) & (test_all.innings == 2) & (test_all.balls_left > 0)]
        ids = set(chase.match_id)
        dsub = t["deliveries"][t["deliveries"].match_id.isin(ids)]
        psub = t["players"][t["players"].match_id.isin(ids)]
        parts = []
        for i, mid in enumerate(chase.match_id.unique()):
            s = chase[chase.match_id == mid]
            prep = sim.prepare(s, dsub[dsub.match_id == mid], psub[psub.match_id == mid], br, bw, prior)
            p = sim.win_probability(b, prep, paths, ex_v, ex_p, seed=hash((y, i)) % 2**32)
            parts.append(s[["match_id", "innings", "over", "ball", "repeat", "year", "phase", "rrr", "wkts_in_hand",
                            "balls_left", "legal_before", "bat_won"]].assign(p_sim=p))
        sy = pd.concat(parts, ignore_index=True).assign(p_smoothed=True)
        # the baseline on the same rows: IPL chases of earlier seasons
        base_tr = f[(f.event_name == IPL) & (f.innings == 2) & (f.balls_left > 0) & (f.year < y)]
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(base_tr[BASE_COLS], base_tr.bat_won)
        sy["p_base"] = lr.predict_proba(sy[BASE_COLS])[:, 1]
        work.mkdir(parents=True, exist_ok=True)
        sy.to_parquet(out, index=False)
        nb_out.write_text(json.dumps(ev))
        sims.append(sy)
        log(f"{y}: {sy.match_id.nunique()} chases, {len(sy)} states, {time.time() - t0:.0f}s")
    allp, chosen = all_methods(pd.concat(sims, ignore_index=True))
    meta["calibration_chosen"] = chosen
    return {"meta": meta, "preds": allp}


def from_work(seasons: list[int], paths: int, work: Path) -> dict:
    """The report's inputs from what a run kept under ``work`` (no tables loaded, nothing simulated)."""
    sims = [_smoothed(pd.read_parquet(work / f"sim_{y}.parquet"), paths) for y in seasons]
    allp, chosen = all_methods(pd.concat(sims, ignore_index=True))
    r = json.loads((work / "rounds.json").read_text())
    meta = {"rounds": r["rounds"], "valid_2022_mlogloss": r["valid_2022_mlogloss"], "paths": paths,
            "params": nb.PARAMS, "calibration_chosen": chosen,
            "next_ball": {y: json.loads((work / f"next_ball_{y}.json").read_text()) for y in seasons}}
    return {"meta": meta, "preds": allp}


def _smoothed(df: pd.DataFrame, paths: int) -> pd.DataFrame:
    """Output saved before the simulator smoothed its probabilities (cricket/model/simulator.py ``smooth``):
    the same transform applied to the saved share of wins."""
    if "p_smoothed" in df and df.p_smoothed.all():
        return df
    from cricket.model.simulator import smooth
    return df.assign(p_sim=smooth(df.p_sim.to_numpy() * paths, paths), p_smoothed=True)


def _layer_table(test: pd.DataFrame, yrs: list) -> list[str]:
    def ll(g, c):
        return score(g.bat_won, g[c])["log_loss"]
    out = ["| layer | " + " | ".join(f"{y} LL" for y in yrs) + " | pooled LL | pooled ECE |",
           "|---|" + "---:|" * (len(yrs) + 2)]
    for name, c in [(m, f"p_{m}") for m in METHODS] + [("baseline", "p_base")]:
        per = " | ".join(f"{ll(test[test.year == y], c):.4f}" for y in yrs)
        out.append(f"| {name} | {per} | {ll(test, c):.4f} | {score(test.bat_won, test[c])['ece']:.4f} |")
    return out


def report_md(res: dict, source: str) -> str:
    meta, p = res["meta"], res["preds"]
    test = p[p.p_cal.notna()]
    yrs = sorted(test.year.unique())
    pooled = {c: score(test.bat_won, test[c]) for c in ("p_base", "p_sim", "p_cal")}
    gate = model_gate(test)
    L = [
        "# Cricket model v1: next-ball XGBoost, Monte Carlo simulator, calibration",
        "",
        f"Source tables: `{source}` (Cricsheet). Built by `python -m cricket.model.evaluate`. Paths per state: "
        f"{meta['paths']}. Next-ball rounds: {meta['rounds']} (early stopping on 2022, fit on 2012-2021).",
        "",
        "## Phase 0 gate, first condition: per-ball win probability on held-out IPL chases",
        "",
        f"Test seasons {yrs[0]}-{yrs[-1]} (each calibrated on earlier seasons only; {p.year.min()} is "
        f"calibration-only). Rows: the state before every delivery of the chase, {len(test):,} rows from "
        f"{test.match_id.nunique()} matches.",
        "",
        "| model | log loss | Brier | ECE |",
        "|---|---:|---:|---:|",
        f"| baseline: logistic on RRR, wickets, balls left | {pooled['p_base']['log_loss']:.4f} | "
        f"{pooled['p_base']['brier']:.4f} | {pooled['p_base']['ece']:.4f} |",
        f"| simulator, raw | {pooled['p_sim']['log_loss']:.4f} | {pooled['p_sim']['brier']:.4f} | "
        f"{pooled['p_sim']['ece']:.4f} |",
        f"| simulator, calibrated | {pooled['p_cal']['log_loss']:.4f} | {pooled['p_cal']['brier']:.4f} | "
        f"{pooled['p_cal']['ece']:.4f} |",
        f"| published baseline (Mishra) | {PUBLISHED['log_loss']:.4f} | | {PUBLISHED['ece']:.4f} |",
        "",
        *gate_lines(gate, "chases"),
        f"- For reference, the spec's fixed bar (log loss under {FIXED_BAR}): calibrated "
        f"{'clears' if pooled['p_cal']['log_loss'] < FIXED_BAR else 'misses'} it, and so does the baseline "
        f"({pooled['p_base']['log_loss']:.4f}): the bar is not a test of the model.",
        "",
        f"Calibration layer: **{meta.get('calibration_chosen')}**, chosen by the rule fixed before the results: the "
        f"lowest log loss on {SELECT_ON} (each layer fitted on the seasons before), confirmed on the later seasons.",
        "",
        *_layer_table(test, yrs),
        "",
        "- The market conditions (the model's Benter weight against Betfair's price, the strategy after 5% "
        "commission): pre-match, cricket/model/benter.py once Betfair's free BASIC cricket file is held; in play, "
        "no data found until a recorded season aligns prices to balls.",
        "",
        "## By season",
        "",
        "| season | rows | baseline LL | raw LL | calibrated LL | baseline ECE | raw ECE | calibrated ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for y, g in p.groupby("year"):
        b, r = score(g.bat_won, g.p_base), score(g.bat_won, g.p_sim)
        c = score(g.bat_won, g.p_cal) if g.p_cal.notna().all() else None
        L.append(f"| {y} | {len(g):,} | {b['log_loss']:.4f} | {r['log_loss']:.4f} | "
                 f"{c['log_loss']:.4f} | {b['ece']:.4f} | {r['ece']:.4f} | {c['ece']:.4f} |" if c else
                 f"| {y} (calibration only) | {len(g):,} | {b['log_loss']:.4f} | {r['log_loss']:.4f} | | "
                 f"{b['ece']:.4f} | {r['ece']:.4f} | |")
    L += ["", "## By phase and required-run-rate band (test seasons)", "",
          "| split | group | rows | baseline LL | calibrated LL | baseline ECE | calibrated ECE |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for r in by_band(test, ["p_base", "p_cal"]).itertuples():
        L.append(f"| {r.split} | {r.group} | {r.n:,} | {r.p_base_log_loss:.4f} | {r.p_cal_log_loss:.4f} | "
                 f"{r.p_base_ece:.4f} | {r.p_cal_ece:.4f} |")
    L += ["", "## Reliability, calibrated simulator (test seasons)", "",
          "| p from | rows | mean p | chase won |", "|---|---:|---:|---:|"]
    for r in reliability(test.bat_won, test.p_cal).itertuples():
        L.append(f"| {r.lo:.1f} | {r.n:,} | {r.mean_p:.3f} | {r.win_rate:.3f} |")
    L += ["", "## Next-ball model, held out by season", "",
          "| season | scope | deliveries | log loss | base-rate log loss | accuracy |", "|---|---|---:|---:|---:|---:|"]
    for y, ev in sorted(meta["next_ball"].items()):
        for scope, e in ev.items():
            L.append(f"| {y} | {scope} | {e['n']:,} | {e['log_loss']:.4f} | {e['base_rate_log_loss']:.4f} | "
                     f"{e['accuracy']:.3f} |")
    L += ["", f"Outcome classes: {', '.join(CLASSES)}. Accuracy above 0.90 would stop the run as leakage.",
          "", "## What v1 leaves out (stated, not filled)", "",
          "- Who bats next is the next listed player of the XI not yet in; Cricsheet's list order is not guaranteed "
          "to be the batting order.",
          "- Future overs are bowled by the bowling XI's experience-weighted average rating.",
          "- No runs on wicket balls, no wickets on wides or no-balls, no free-hit flag.",
          "- Ratings v1 have no phase x innings prior, no bowler type (no data found in Cricsheet), no matchups.",
          "- First-innings win probability and the runs line: reports/cricket/model_v1_first_innings.md.",
          f"- Monte Carlo noise: {meta['paths']} paths a state adds roughly 1/(2 x paths) to log loss near p = 0.5; "
          "the share of winning paths is smoothed to (wins + 1/2) / (paths + 1), never exactly 0 or 1.",
          ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root")
    ap.add_argument("--seasons", default="2023,2024,2025,2026")
    ap.add_argument("--paths", type=int, default=256)
    ap.add_argument("--work", default="data/cricket/eval")
    ap.add_argument("--out", default="reports/cricket")
    ap.add_argument("--report-only", action="store_true", help="rebuild the report from --work, no simulation")
    a = ap.parse_args(argv)
    store = Store(root=a.root)
    seasons = [int(x) for x in a.seasons.split(",")]
    if a.report_only:
        res = from_work(seasons, a.paths, Path(a.work))
    else:
        res = run(store, seasons, a.paths, Path(a.work), log=lambda m: print(m, flush=True))
    md = report_md(res, store.where("cricket/tables/"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "model_v1.md").write_text(md)
    if not a.report_only:   # the full run's record (training rows a season, parameters) is kept as it was
        (out / "model_v1.json").write_text(json.dumps(res["meta"], indent=1, default=str))
    store.put("cricket/reports/model_v1.md", md.encode())
    store.put_parquet("cricket/eval/model_v1_preds.parquet", res["preds"])
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
