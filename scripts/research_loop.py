#!/usr/bin/env python3
"""The research loop's price-forecast comparison, one fold per job.

An iteration used to fit its six models one after another in a single job --
the base on three folds, then the variant on the same three -- and took two
and a half hours, most of it waiting on fits that did not depend on each
other. Here each (arm, fold) is its own job, so the wall clock is one fold's
fit; the base, which is the served feature set and changes only when the
served model does, is fitted once and reused from the cache while its inputs
are unchanged; and `"recipe": "quick"` screens on a recipe about seven times
faster a fit (model/bfsp_model.py QUICK_PARAMS) before a block is confirmed on
the served one.

    plan     read research/loop.json and the cached matrix's dates: the folds,
             the key the base run is cached under
    arms     which arms need fitting, given whether the base is cached
    fit      one arm on one fold (evaluate_oos.py --fold, the serial run's fold)
    gather   each arm's folds into one prediction file and one summary
    compare  the paired comparison, the replicate check and the early-price
             trade, into reports/bfsp_compare/compare.md

The arms: `base` (production features, plus `bfsp_base_blocks`, less
`bfsp_base_withhold`); one arm per variant; and, with `"replicate_base": true`,
`base_rep`, the base fitted again in another job, whose distance from `base` is
the noise floor any difference between arms has to clear.

A variant is the base plus its own blocks, less its own withheld features, with
its own extra arguments. Several can run at once, each against the same base:

    "variants": [{"name": "fv", "blocks": "form_variants"},
                 {"name": "fv_no_lb", "blocks": "form_variants", "drop": "fv_lb"},
                 {"name": "r6000", "args": "--num-boost-round 6000"}]

Without "variants" the single variant is the old top-level keys (arm `blocks`:
`blocks`, `bfsp_variant_drop`, `bfsp_variant_withhold`, `bfsp_variant_args`).
`bfsp_args` go to every arm (e.g. "--num-boost-round 6000" to confirm a
promotion at the round cap it will be served with), `bfsp_base_args` to the
base arms alone.

An ensemble is scored as an arm but never fitted: the geometric mean of some
arms' price forecasts, renormalised to a book of 1 in each race, built at the
compare step (how several seeds of one recipe would be served as one):

    "ensembles": [{"name": "cw_avg3", "arms": ["cw", "cw_s7", "cw_s11"]}]

Every step reads the same config, so the YAML only wires jobs together.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

#: The locked holdout starts here and the loop never scores it.
EVAL_UNTIL = "2026-04-01"
START_DATE = os.environ.get("START_DATE", "2021-01-01")
FEATURE_CACHE = ".feature_cache"
FIT_DIR = "reports/fit"
OUT_DIR = "reports/bfsp_compare"
BASE_CACHE_DIR = "reports/base_cache"

#: What decides the base run's predictions besides the matrix and the arguments.
FIT_CODE = ("evaluate_oos.py", "model/bfsp_model.py")


def load_config(path: str = "research/loop.json") -> dict:
    return json.loads((REPO / path).read_text())


def recipe(cfg: dict) -> str:
    r = cfg.get("recipe", "served")
    if r not in ("served", "quick"):
        raise SystemExit(f"loop.json recipe {r!r}: served or quick")
    return r


def start_date(cfg: dict) -> str:
    """Where the matrix's history starts: the config's `start_date`, else START_DATE.

    An earlier start gives every career, sire and yard record the years before
    it; `fold_anchor` then keeps the folds those of the default start, so the
    runners scored and each fold's cut-off are unchanged."""
    return str(cfg.get("start_date") or START_DATE)


def common_args(cfg: dict) -> list[str]:
    """Arguments every arm shares: the matrix, the folds, the recipe."""
    anchor = ["--fold-anchor", str(cfg["fold_anchor"])] if cfg.get("fold_anchor") else []
    return ["--start-date", start_date(cfg), "--feature-cache", FEATURE_CACHE,
            "--eval-from", cfg.get("bfsp_from", "2025-07-01"), "--eval-until", EVAL_UNTIL,
            "--step-days", "91", "--val-window", "91", "--recipe", recipe(cfg), *anchor]


BASE_ARMS = ("base", "base_rep")


def variants(cfg: dict) -> list[dict]:
    """The variant arms, each {name, blocks, drop, withhold, args}."""
    if "variants" not in cfg:
        return [{"name": "blocks", "blocks": cfg.get("blocks", ""), "drop": cfg.get("bfsp_variant_drop", ""),
                 "withhold": cfg.get("bfsp_variant_withhold", ""), "args": cfg.get("bfsp_variant_args", "")}]
    out, seen = [], set()
    for v in cfg["variants"]:
        name = v.get("name", "")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,30}", name) or name in BASE_ARMS or name in seen:
            raise SystemExit(f"variant name {name!r}: lower-case, unique, not {BASE_ARMS}")
        seen.add(name)
        out.append({"name": name, "blocks": v.get("blocks", ""), "drop": v.get("drop", ""),
                    "withhold": v.get("withhold", ""), "args": v.get("args", "")})
    if not out:
        raise SystemExit("variants is empty")
    return out


def ensembles(cfg: dict) -> list[dict]:
    """The averaged arms, each {name, arms}: built from fitted arms, never fitted themselves."""
    names = {v["name"] for v in variants(cfg)}
    out, seen = [], set()
    for e in cfg.get("ensembles", []):
        name, arms = e.get("name", ""), list(e.get("arms", []))
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,30}", name) or name in BASE_ARMS or name in names or name in seen:
            raise SystemExit(f"ensemble name {name!r}: lower-case, unique, not an arm's")
        fitted = set(all_arms(cfg))
        if len(set(arms)) < 2 or len(set(arms)) != len(arms) or not set(arms) <= fitted:
            raise SystemExit(f"ensemble {name!r}: two or more distinct fitted arms, not {arms}")
        seen.add(name)
        out.append({"name": name, "arms": arms})
    return out


def average_arms(paths: list, out) -> None:
    """Several arms' forecasts as one: the geometric mean of their prices, renormalised to a book of 1 per race.

    Every arm must price every runner of the first; the result keeps the first's rows, order and other columns."""
    import numpy as np
    import pandas as pd
    key = ["race_date", "race_time", "track", "horse_name"]
    frames = [pd.read_csv(p, dtype={"race_time": str}) for p in paths]
    first = frames[0]
    idx = first.set_index(key).index
    if not idx.is_unique:
        raise SystemExit(f"{paths[0]}: a runner appears twice")
    logs, raws = [], []
    for p, f in zip(paths, frames):
        r = f.set_index(key).reindex(idx)
        if r["predicted_bfsp"].isna().any():
            raise SystemExit(f"{p}: lacks runners of {paths[0]}")
        logs.append(np.log(r["predicted_bfsp"].to_numpy(float)))
        if "predicted_bfsp_raw" in r:
            raws.append(np.log(r["predicted_bfsp_raw"].to_numpy(float)))
    race = (first["race_date"].astype(str) + "|" + first["track"].astype(str) + "|"
            + first["race_time"].astype(str)).to_numpy()
    p = np.exp(-np.mean(logs, axis=0))
    pn = p / pd.Series(p).groupby(race).transform("sum").to_numpy()
    avg = first.copy()
    avg["predicted_win_prob_norm"] = pn
    avg["predicted_bfsp"] = 1.0 / pn
    if len(raws) == len(logs):
        avg["predicted_bfsp_raw"] = np.exp(np.mean(raws, axis=0))
    if "overlay_pct" in avg and "bfsp" in avg:
        avg["overlay_pct"] = (avg["bfsp"] / avg["predicted_bfsp"] - 1) * 100
    avg.to_csv(out, index=False)


def _join_blocks(*parts: str) -> str:
    seen = []
    for p in parts:
        for b in (p or "").split(","):
            if b.strip() and b.strip() not in seen:
                seen.append(b.strip())
    return ",".join(seen)


def arm_args(cfg: dict, arm: str) -> list[str]:
    """One arm's evaluate_oos.py arguments beyond common_args.

    `bfsp_args` go to every arm (a round cap the whole comparison is run at),
    `bfsp_base_args` to the base alone, a variant's `args` to that variant."""
    base_blocks = cfg.get("bfsp_base_blocks", "")
    shared = shlex.split(cfg.get("bfsp_args", ""))
    if arm in BASE_ARMS:
        a = ["--blocks", base_blocks] if base_blocks else []
        wh = cfg.get("bfsp_base_withhold", "")
        return a + (["--withhold", wh] if wh else []) + shared + shlex.split(cfg.get("bfsp_base_args", ""))
    for v in variants(cfg):
        if v["name"] == arm:
            a = ["--blocks", _join_blocks(base_blocks, v["blocks"])]
            if v["drop"]:
                a += ["--drop-features", v["drop"]]
            wh = _join_blocks(cfg.get("bfsp_base_withhold", ""), v["withhold"])
            if wh:
                a += ["--withhold", wh]
            return a + shared + shlex.split(v["args"])
    raise SystemExit(f"unknown arm {arm!r}")


def fit_code_hash() -> str:
    h = hashlib.sha256()
    for rel in FIT_CODE:
        h.update(rel.encode())
        h.update((REPO / rel).read_bytes())
    return h.hexdigest()[:16]


def block_code_hash(blocks: str) -> str:
    """The source of the drop-in blocks in `blocks`, and of everything they import.

    A drop-in block is computed on the cached matrix, so the matrix's key, which
    covers the engine's code, does not cover it. Without this a base carrying a
    block would be reused from the cache after the block was edited: the old
    block's predictions served as the new one's. Engine blocks (flags such as
    form_windows) are in the matrix and its key already; they add nothing here."""
    from model import blocks as blk, feature_cache
    drop_in = [b for b in _join_blocks(blocks).split(",") if b in blk.names()]
    if not drop_in:
        return ""
    files = feature_cache.feature_source_files(
        ["model/blocks/__init__.py"] + [f"model/blocks/{b}.py" for b in drop_in])
    h = hashlib.sha256()
    for path in files:
        rel = path.relative_to(REPO) if path.is_relative_to(REPO) else path
        h.update(str(rel).encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def base_key(cfg: dict, feature_key: str, folds: list) -> str:
    """Everything the base run's predictions are a function of.

    The matrix (its key covers the feature code and the data), the arguments,
    the fold plan, the fitting code, the library, and the source of any drop-in
    block the base carries. Fits are deterministic
    (DEFAULT_PARAMS), so an equal key means the same predictions: reusing them
    is the base run, not an approximation of it."""
    import lightgbm
    parts = [feature_key, json.dumps(common_args(cfg)), json.dumps(arm_args(cfg, "base")),
             json.dumps(folds, sort_keys=True), fit_code_hash(), lightgbm.__version__,
             sys.version.split()[0], block_code_hash(cfg.get("bfsp_base_blocks", ""))]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:20]


def _outputs(pairs: dict) -> None:
    """Write step outputs (GITHUB_OUTPUT when set, stdout otherwise)."""
    target = os.environ.get("GITHUB_OUTPUT")
    lines = [f"{k}={v}" for k, v in pairs.items()]
    if target:
        with open(target, "a") as f:
            f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_plan(args) -> None:
    cfg = load_config(args.config)
    from model import feature_cache
    fkey = feature_cache.cache_key(args.db, start_date(cfg))
    Path(FIT_DIR).mkdir(parents=True, exist_ok=True)
    plan_file = Path(FIT_DIR) / "folds.json"
    subprocess.run([sys.executable, "evaluate_oos.py", "--db", args.db, *common_args(cfg),
                    "--list-folds", str(plan_file)], check=True, cwd=REPO)
    folds = json.loads(plan_file.read_text())
    if not folds:
        raise SystemExit("the fold plan is empty")
    _outputs({
        "folds": json.dumps([f["fold"] for f in folds]),
        "base_key": base_key(cfg, fkey, folds),
        "recipe": recipe(cfg),
        "replicate": "true" if cfg.get("replicate_base") else "false",
        "start_date": start_date(cfg),
    })


def all_arms(cfg: dict) -> list[str]:
    return ["base"] + [v["name"] for v in variants(cfg)] + (["base_rep"] if cfg.get("replicate_base") else [])


def cmd_arms(args) -> None:
    cfg = load_config(args.config)
    arms = [a for a in all_arms(cfg) if not (a == "base" and args.base_hit == "true")]
    _outputs({"arms": json.dumps(arms)})


def cmd_fit(args) -> None:
    cfg = load_config(args.config)
    Path(FIT_DIR).mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "evaluate_oos.py", "--db", args.db, *common_args(cfg),
           "--fold", str(args.fold), "--tag", f"{args.arm}_f{args.fold}", *arm_args(cfg, args.arm),
           "--output-csv", f"{FIT_DIR}/oos.csv", "--output-json", f"{FIT_DIR}/summary.json"]
    print("+", shlex.join(cmd), flush=True)
    raise SystemExit(subprocess.run(cmd, cwd=REPO).returncode)


def _fold_files(src: str, arm: str, kind: str, ext: str) -> list[tuple[int, str]]:
    pat = re.compile(rf"{kind}_{re.escape(arm)}_f(\d+)\.{ext}$")
    found = []
    for p in glob.glob(os.path.join(src, "**", f"{kind}_{arm}_f*.{ext}"), recursive=True):
        m = pat.search(os.path.basename(p))
        if m:
            found.append((int(m.group(1)), p))
    return sorted(found)


def gather(src: str, out: str, arm: str, expect: list[int] | None = None) -> bool:
    """One arm's fold files into oos_<arm>.csv and summary_<arm>.json. False if
    the arm has no files here (a base served from the cache)."""
    import pandas as pd
    csvs = _fold_files(src, arm, "oos", "csv")
    if not csvs:
        return False
    got = [k for k, _ in csvs]
    if expect is not None and sorted(got) != sorted(expect):
        raise SystemExit(f"{arm}: folds {got} gathered, {sorted(expect)} planned")
    Path(out).mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(p, dtype={"race_time": str}) for _, p in csvs], ignore_index=True)
    df.to_csv(os.path.join(out, f"oos_{arm}.csv"), index=False)
    per_fold = [json.loads(Path(p).read_text()) for _, p in _fold_files(src, arm, "summary", "json")]
    merged = {
        "arm": arm, "folds": got, "n_predictions": int(len(df)),
        "fold_fits": [f for s in per_fold for f in s.get("fold_fits", [])],
        "train_config": per_fold[0].get("train_config") if per_fold else None,
        "recipe": per_fold[0].get("recipe") if per_fold else None,
        "n_features_used": per_fold[0].get("n_features_used") if per_fold else None,
        "missing_features": per_fold[0].get("missing_features") if per_fold else None,
        "elapsed_seconds": [s.get("elapsed_seconds") for s in per_fold],
        "cpu": sorted({s.get("cpu") for s in per_fold if s.get("cpu")}),
        "per_fold": per_fold,
    }
    Path(out, f"summary_{arm}.json").write_text(json.dumps(merged, indent=2, default=str))
    print(f"{arm}: {len(df):,} predictions from folds {got}")
    return True


def cmd_gather(args) -> None:
    expect = json.loads(args.folds) if args.folds else None
    for arm in all_arms(load_config(args.config)):
        gather(args.src, OUT_DIR, arm, expect)


def replicate_report(a_path: str, b_path: str) -> str:
    """How far two fits of the same arm are apart: identical, or by how much."""
    import numpy as np
    import pandas as pd
    key = ["race_date", "race_time", "track", "horse_name"]
    a = pd.read_csv(a_path, dtype={"race_time": str})
    b = pd.read_csv(b_path, dtype={"race_time": str})
    m = a.merge(b, on=key, suffixes=("_a", "_b"))
    d = np.abs(np.log(m["predicted_bfsp_a"]) - np.log(m["predicted_bfsp_b"]))
    same = float((d == 0).mean()) if len(m) else float("nan")
    return (f"{len(m):,} runners paired; identical predictions on {same:.1%}; "
            f"median |log ratio| {float(np.median(d)) if len(m) else float('nan'):.6f}, "
            f"max {float(d.max()) if len(m) else float('nan'):.6f}")


def _note(cfg: dict, v: dict, base_cached: bool) -> str:
    bb, bwh = cfg.get("bfsp_base_blocks", ""), cfg.get("bfsp_base_withhold", "")
    base = f"production features{' + ' + bb if bb else ''}{' without ' + bwh if bwh else ''}"
    return (f"{cfg.get('tag', '')} / {v['name']}: {base} vs the same"
            f"{' + ' + v['blocks'] if v['blocks'] else ''}{' without ' + v['drop'] if v['drop'] else ''}"
            f"{' without ' + v['withhold'] if v['withhold'] else ''}{' with ' + v['args'] if v['args'] else ''}; "
            f"recipe {recipe(cfg)}{'; base from the cache' if base_cached else ''}")


def summary_table(rows: list[tuple[str, dict]]) -> str:
    """One line per variant: what the decision rule turns on."""
    L = ["| variant | paired error (90% CI) | rank 1 | Brier skill vs base (90% CI) | concordance | decision |",
         "|---|---|---|---|---|---|"]
    for name, j in rows:
        r1 = j.get("rank1_delta")
        L.append(f"| {name} | {j['delta']:+.4f} ({j['delta_ci'][0]:+.4f} to {j['delta_ci'][1]:+.4f}) | "
                 f"{'' if r1 is None else f'{r1:+.4f}'} | {j['brier_skill']:+.5f} ({j['brier_skill_ci'][0]:+.5f} to "
                 f"{j['brier_skill_ci'][1]:+.5f}) | {j['concordance']:+.5f} | "
                 f"{'**replaces the base**' if j['replaces'] else 'base stands'} |")
    return "\n".join(L)


def cmd_compare(args) -> None:
    cfg = load_config(args.config)
    out = Path(OUT_DIR)
    base = out / "oos_base.csv"
    if not base.exists():
        raise SystemExit("no base predictions: neither fitted here nor restored from the cache")
    cached = args.base_cached == "true"
    from concurrent.futures import ThreadPoolExecutor

    def one(arm):
        name, note = arm
        md, js = out / f"compare_{name}.md", out / f"compare_{name}.json"
        subprocess.run([sys.executable, "scripts/compare_oos_runs.py", "--base", str(base),
                        "--variant", str(out / f"oos_{name}.csv"), "--out", str(md), "--json-out", str(js),
                        "--note", note], check=True, cwd=REPO, capture_output=True)
        return name, json.loads(js.read_text()), md.read_text()

    arms = [(v["name"], _note(cfg, v, cached)) for v in variants(cfg)]
    for e in ensembles(cfg):
        average_arms([out / f"oos_{a}.csv" for a in e["arms"]], out / f"oos_{e['name']}.csv")
        arms.append((e["name"], f"{cfg.get('tag', '')} / {e['name']}: the geometric mean of "
                                f"{', '.join(e['arms'])}, renormalised per race; recipe {recipe(cfg)}"))
    # each comparison is its own process; run them side by side (a runner has four cores)
    with ThreadPoolExecutor(max_workers=4) as pool:
        done = list(pool.map(one, arms))
    rows = [(name, j) for name, j, _ in done]
    sections = [md for _, _, md in done]
    text = (f"# {cfg.get('tag', '')}: {len(rows)} variant(s) against the base, recipe {recipe(cfg)}\n\n"
            + summary_table(rows) + "\n\n" + "\n\n".join(sections))
    rep = out / "oos_base_rep.csv"
    if rep.exists():
        subprocess.run([sys.executable, "scripts/compare_oos_runs.py", "--base", str(base),
                        "--variant", str(rep), "--out", str(out / "replicate.md"),
                        "--note", "replicate: the base fitted twice, in separate jobs"],
                       check=True, cwd=REPO)
        text += ("\n\n## Replicate: the base fitted twice\n\n" + replicate_report(str(base), str(rep))
                 + "\n\n" + (out / "replicate.md").read_text())
    def clv(arm):
        res = subprocess.run([sys.executable, "scripts/clv_betfair.py", "--predictions",
                              str(out / f"oos_{arm}.csv"), "--db", args.db, "--until", EVAL_UNTIL],
                             cwd=REPO, capture_output=True, text=True)
        body = res.stdout + (res.stderr[-2000:] if res.returncode else "")
        (out / f"clv_{arm}.txt").write_text(body)
        return arm, body

    with ThreadPoolExecutor(max_workers=4) as pool:
        for arm, body in pool.map(clv, ["base"] + [name for name, _ in arms]):
            text += f"\n\n### Early-price trade on Betfair's morning prices: {arm}\n```\n{body}\n```\n"
    (out / "compare.md").write_text(text)
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="research/loop.json")
    ap.add_argument("--db", default="horse_racing.db")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    a = sub.add_parser("arms")
    a.add_argument("--base-hit", default="false")
    f = sub.add_parser("fit")
    f.add_argument("--arm", required=True)
    f.add_argument("--fold", required=True, type=int)
    g = sub.add_parser("gather")
    g.add_argument("--src", default="fits")
    g.add_argument("--folds", default=None, help="JSON list of the planned fold indices")
    c = sub.add_parser("compare")
    c.add_argument("--base-cached", default="false")
    args = ap.parse_args()
    {"plan": cmd_plan, "arms": cmd_arms, "fit": cmd_fit, "gather": cmd_gather,
     "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
